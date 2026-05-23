from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from scipy.optimize import linear_sum_assignment

from .bib_coupling import compute_modularity_for_clusters
from .constants import (
    COVERAGE_DIR,
    HUMAN_ALIGNMENT_DIR,
    MODULARITY_DIR,
    QUALITY_DIR,
    REFERENCE_GROUNDED_COVERAGE_DIR,
    SHARED_EMBEDDINGS_DIR,
    SILHOUETTE_DIR,
    SIMILARITY_DIR,
    SUMMAC_DIR,
)
from .descriptions import load_description_csv
from .embeddings import EmbeddingConfig, ensure_embeddings
from .utils import (
    clean_text,
    ensure_directory,
    parse_references,
    slugify_filename,
    split_sentences,
    text_hash,
)


SUMMAC_CONV_MODEL_URL = (
    "https://github.com/tingofurro/summac/raw/master/summac_conv_vitc_sent_perc_e.bin"
)
SUMMAC_CONV_MODEL_FILENAME = "summac_conv_vitc_sent_perc_e.bin"
_SUMMAC_MODEL_CACHE: dict[str, object] = {}
_BERT_SCORER_CACHE: dict[tuple[str, str, bool], object] = {}


def _normalize_embedding_matrix(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return matrix / norms


def _embeddings_to_matrix(embeddings: list[list[float]]) -> np.ndarray:
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("Embeddings must form a 2D matrix.")
    return matrix


def _shared_abstract_embeddings_cache_path(scopus_name: str) -> Path:
    return ensure_directory(SHARED_EMBEDDINGS_DIR / "abstracts") / _embedding_cache_filename(
        dataset_name=scopus_name,
        suffix="abstracts",
    )


def _embedding_cache_filename(dataset_name: str, suffix: str) -> str:
    slug = slugify_filename(dataset_name)
    short_slug = slug[:16] if slug else "cache"
    return f"{short_slug}_{text_hash(dataset_name)[:12]}_{suffix}.parquet"


def _resolve_summac_device(device: str | None = None) -> str:
    if device is not None:
        return str(device).strip().lower()

    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_torch_device(device: str | None = None) -> str:
    if device is not None:
        return str(device).strip().lower()

    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_bert_scorer(
    model_type: str = "roberta-large",
    lang: str = "en",
    device: str | None = None,
    batch_size: int = 64,
    rescale_with_baseline: bool = False,
):
    resolved_device = _resolve_torch_device(device)
    cache_key = (model_type, resolved_device, bool(rescale_with_baseline))
    cached = _BERT_SCORER_CACHE.get(cache_key)
    if cached is not None:
        return cached

    try:
        from bert_score import BERTScorer
    except ImportError as error:
        raise ImportError(
            "bert-score is not installed. Add the 'bert-score' package to the environment "
            "before running human alignment evaluation."
        ) from error

    scorer = BERTScorer(
        model_type=model_type,
        lang=lang,
        device=resolved_device,
        batch_size=batch_size,
        rescale_with_baseline=rescale_with_baseline,
    )
    _BERT_SCORER_CACHE[cache_key] = scorer
    return scorer


def _download_summac_conv_weights(target_path: Path) -> Path:
    ensure_directory(target_path.parent)
    if target_path.exists():
        return target_path

    response = requests.get(SUMMAC_CONV_MODEL_URL, timeout=120)
    response.raise_for_status()
    target_path.write_bytes(response.content)
    return target_path


def load_summac_model(
    device: str | None = None,
):
    resolved_device = _resolve_summac_device(device)
    cached_model = _SUMMAC_MODEL_CACHE.get(resolved_device)
    if cached_model is not None:
        return cached_model

    try:
        from summac.model_summac import SummaCConv
    except ImportError as error:
        raise ImportError(
            "SummaC is not installed. Add the 'summac' package to the environment "
            "before running factual consistency evaluation."
        ) from error

    model_path = _download_summac_conv_weights(
        ensure_directory(SUMMAC_DIR / "models") / SUMMAC_CONV_MODEL_FILENAME
    )
    model = SummaCConv(
        models=["vitc"],
        bins="percentile",
        granularity="sentence",
        nli_labels="e",
        device=resolved_device,
        start_file=str(model_path),
        agg="mean",
    )
    _SUMMAC_MODEL_CACHE[resolved_device] = model
    return model


def _cluster_abstracts_from_references(
    description_df: pd.DataFrame,
    scopus_df: pd.DataFrame,
) -> dict[int, list[str]]:
    paper_lookup = (
        scopus_df[["paper_id", "paper_abstract"]]
        .dropna(subset=["paper_abstract"])
        .assign(paper_id=lambda df: df["paper_id"].astype(int))
    )
    abstract_by_paper_id = {
        int(row["paper_id"]): clean_text(row["paper_abstract"])
        for _, row in paper_lookup.iterrows()
        if clean_text(row["paper_abstract"])
    }

    cluster_abstracts: dict[int, list[str]] = {}
    for _, row in description_df.iterrows():
        cluster_id = int(row["cluster_id"])
        reference_ids: list[int] = []
        for reference in parse_references(row.get("references")):
            stripped = str(reference).strip().strip("[]")
            if stripped.isdigit():
                reference_ids.append(int(stripped))
        cluster_abstracts[cluster_id] = [
            abstract_by_paper_id[paper_id]
            for paper_id in reference_ids
            if paper_id in abstract_by_paper_id
        ]
    return cluster_abstracts


def _cluster_reference_ids_from_references(
    description_df: pd.DataFrame,
    scopus_df: pd.DataFrame,
) -> dict[int, list[int]]:
    available_paper_ids = {
        int(row["paper_id"])
        for _, row in scopus_df[["paper_id", "paper_abstract"]].iterrows()
        if clean_text(row["paper_abstract"])
    }
    cluster_reference_ids: dict[int, list[int]] = {}
    for _, row in description_df.iterrows():
        cluster_id = int(row["cluster_id"])
        reference_ids: list[int] = []
        for reference in parse_references(row.get("references")):
            stripped = str(reference).strip().strip("[]")
            if stripped.isdigit():
                paper_id = int(stripped)
                if paper_id in available_paper_ids:
                    reference_ids.append(paper_id)
        cluster_reference_ids[cluster_id] = reference_ids
    return cluster_reference_ids


def summac_factual_consistency(
    cluster_descriptions: Mapping[int, str],
    cluster_abstracts: Mapping[int, list[str]],
    summac_model=None,
    device: str | None = None,
) -> dict[str, object]:
    model = summac_model or load_summac_model(device=device)

    cluster_scores: dict[str, float] = {}
    pipeline_scores: list[float] = []

    for cluster_id, description in cluster_descriptions.items():
        sentences = split_sentences(description)
        abstracts = [
            clean_text(text)
            for text in cluster_abstracts.get(int(cluster_id), [])
            if clean_text(text)
        ]
        if not sentences or not abstracts:
            continue

        best_sentence_scores: list[float] = []
        for sentence in sentences:
            score_result = model.score(abstracts, [sentence] * len(abstracts))
            sentence_scores = [float(score) for score in score_result["scores"]]
            if sentence_scores:
                best_sentence_scores.append(max(sentence_scores))

        if not best_sentence_scores:
            continue

        cluster_score = float(np.mean(best_sentence_scores))
        cluster_scores[str(int(cluster_id))] = cluster_score
        pipeline_scores.append(cluster_score)

    return {
        "summac_factual_consistency": float(np.mean(pipeline_scores)) if pipeline_scores else np.nan,
        "summac_by_cluster": cluster_scores,
    }


def compute_coverage_of_corpus(
    description_df: pd.DataFrame | str | Path,
    scopus_df: pd.DataFrame,
    description_name: str,
    scopus_name: str,
    embedding_model: str,
    batch_size: int = 128,
) -> pd.DataFrame:
    if not isinstance(description_df, pd.DataFrame):
        description_df = load_description_csv(description_df)

    base_dir = ensure_directory(COVERAGE_DIR)
    embeddings_dir = ensure_directory(base_dir / "embeddings")
    results_dir = ensure_directory(base_dir / "results")

    abstract_sentence_rows: list[dict[str, object]] = []
    for _, row in scopus_df.iterrows():
        for sentence_index, sentence in enumerate(split_sentences(row["paper_abstract"]), start=1):
            abstract_sentence_rows.append(
                {
                    "paper_id": int(row["paper_id"]),
                    "sentence_index": sentence_index,
                    "text": sentence,
                    "text_hash": text_hash(sentence),
                }
            )

    description_sentence_rows: list[dict[str, object]] = []
    for _, row in description_df.iterrows():
        for sentence_index, sentence in enumerate(split_sentences(row["description"]), start=1):
            description_sentence_rows.append(
                {
                    "cluster_id": int(row["cluster_id"]),
                    "sentence_index": sentence_index,
                    "text": sentence,
                    "text_hash": text_hash(sentence),
                }
            )

    abstract_sentences_df = pd.DataFrame(abstract_sentence_rows)
    description_sentences_df = pd.DataFrame(description_sentence_rows)
    if abstract_sentences_df.empty or description_sentences_df.empty:
        raise ValueError("Coverage evaluation requires non-empty abstract and description sentences.")

    abstract_embeddings = ensure_embeddings(
        records_df=abstract_sentences_df,
        cache_path=embeddings_dir / _embedding_cache_filename(
            dataset_name=scopus_name,
            suffix="abstract_sentences",
        ),
        config=EmbeddingConfig(model=embedding_model, batch_size=batch_size),
    )
    description_embeddings = ensure_embeddings(
        records_df=description_sentences_df,
        cache_path=embeddings_dir / _embedding_cache_filename(
            dataset_name=description_name,
            suffix="description_sentences",
        ),
        config=EmbeddingConfig(model=embedding_model, batch_size=batch_size),
    )

    abstract_matrix = _normalize_embedding_matrix(
        _embeddings_to_matrix(abstract_embeddings["embedding"].tolist())
    )
    description_matrix = _normalize_embedding_matrix(
        _embeddings_to_matrix(description_embeddings["embedding"].tolist())
    )
    cosine_matrix = abstract_matrix @ description_matrix.T
    average_cosine = float(cosine_matrix.max(axis=1).mean())

    result = pd.DataFrame(
        [
            {
                "scopus_file": scopus_name,
                "description_file": description_name,
                "average_cosine": average_cosine,
                "n_abstract_sentences": len(abstract_sentences_df),
                "n_description_sentences": len(description_sentences_df),
                "embedding_model": embedding_model,
                "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        ]
    )
    result.to_csv(results_dir / f"{description_name}_coverage.csv", index=False)
    return result


def compute_reference_grounded_coverage(
    description_df: pd.DataFrame | str | Path,
    scopus_df: pd.DataFrame,
    description_name: str,
    embedding_model: str,
    batch_size: int = 128,
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    if not isinstance(description_df, pd.DataFrame):
        description_df = load_description_csv(description_df)

    base_dir = ensure_directory(
        Path(output_dir)
        if output_dir is not None
        else REFERENCE_GROUNDED_COVERAGE_DIR
    )
    embeddings_dir = ensure_directory(base_dir / "embeddings")
    results_dir = ensure_directory(base_dir / "results")

    cluster_abstracts = _cluster_abstracts_from_references(
        description_df=description_df,
        scopus_df=scopus_df,
    )
    cluster_reference_ids = _cluster_reference_ids_from_references(
        description_df=description_df,
        scopus_df=scopus_df,
    )

    cluster_rows: list[dict[str, object]] = []
    cluster_scores: list[float] = []

    for _, row in description_df.iterrows():
        cluster_id = int(row["cluster_id"])
        description_text = clean_text(row["description"])
        description_sentences = split_sentences(description_text)
        abstract_texts = cluster_abstracts.get(cluster_id, [])

        abstract_sentence_rows: list[dict[str, object]] = []
        for paper_id, abstract in zip(cluster_reference_ids.get(cluster_id, []), abstract_texts):
            for sentence_index, sentence in enumerate(split_sentences(abstract), start=1):
                abstract_sentence_rows.append(
                    {
                        "paper_id": int(paper_id),
                        "sentence_index": sentence_index,
                        "text": sentence,
                        "text_hash": text_hash(sentence),
                    }
                )

        description_sentence_rows = [
            {
                "cluster_id": cluster_id,
                "sentence_index": sentence_index,
                "text": sentence,
                "text_hash": text_hash(sentence),
            }
            for sentence_index, sentence in enumerate(description_sentences, start=1)
        ]

        if not abstract_sentence_rows or not description_sentence_rows:
            continue

        abstract_sentences_df = pd.DataFrame(abstract_sentence_rows)
        description_sentences_df = pd.DataFrame(description_sentence_rows)

        cluster_name = f"{description_name}__cluster_{cluster_id}"
        abstract_embeddings = ensure_embeddings(
            records_df=abstract_sentences_df,
            cache_path=embeddings_dir / _embedding_cache_filename(
                dataset_name=cluster_name,
                suffix="reference_abstract_sentences",
            ),
            config=EmbeddingConfig(model=embedding_model, batch_size=batch_size),
        )
        description_embeddings = ensure_embeddings(
            records_df=description_sentences_df,
            cache_path=embeddings_dir / _embedding_cache_filename(
                dataset_name=cluster_name,
                suffix="cluster_description_sentences",
            ),
            config=EmbeddingConfig(model=embedding_model, batch_size=batch_size),
        )

        abstract_matrix = _normalize_embedding_matrix(
            _embeddings_to_matrix(abstract_embeddings["embedding"].tolist())
        )
        description_matrix = _normalize_embedding_matrix(
            _embeddings_to_matrix(description_embeddings["embedding"].tolist())
        )
        cosine_matrix = abstract_matrix @ description_matrix.T
        cluster_score = float(cosine_matrix.max(axis=1).mean())
        cluster_scores.append(cluster_score)

        cluster_rows.append(
            {
                "cluster_id": cluster_id,
                "reference_grounded_coverage": cluster_score,
                "n_reference_papers": len(cluster_reference_ids.get(cluster_id, [])),
                "n_reference_abstract_sentences": len(abstract_sentences_df),
                "n_description_sentences": len(description_sentences_df),
                "n_sentence_comparisons": int(cosine_matrix.shape[0] * cosine_matrix.shape[1]),
            }
        )

    detail_df = pd.DataFrame(cluster_rows)
    if detail_df.empty:
        detail_df = pd.DataFrame(
            columns=[
                "cluster_id",
                "reference_grounded_coverage",
                "n_reference_papers",
                "n_reference_abstract_sentences",
                "n_description_sentences",
                "n_sentence_comparisons",
            ]
        )
    else:
        detail_df = detail_df.sort_values("cluster_id").reset_index(drop=True)

    detail_path = results_dir / f"{description_name}_reference_grounded_coverage_by_cluster.csv"
    detail_df.to_csv(detail_path, index=False, encoding="utf-8")

    result = pd.DataFrame(
        [
            {
                "description_file": description_name,
                "reference_grounded_coverage": float(np.mean(cluster_scores)) if cluster_scores else np.nan,
                "n_clusters_scored": int(len(cluster_scores)),
                "embedding_model": embedding_model,
                "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
                "detail_file": detail_path.name,
            }
        ]
    )
    result.to_csv(
        results_dir / f"{description_name}_reference_grounded_coverage.csv",
        index=False,
        encoding="utf-8",
    )
    return result


def compute_quality_of_induced_clustering(
    description_df: pd.DataFrame | str | Path,
    scopus_df: pd.DataFrame,
    description_name: str,
    scopus_name: str,
    embedding_model: str,
    batch_size: int = 128,
    cluster_output_column: str = "cluster",
) -> pd.DataFrame:
    if not isinstance(description_df, pd.DataFrame):
        description_df = load_description_csv(description_df)

    base_dir = ensure_directory(QUALITY_DIR)
    embeddings_dir = ensure_directory(base_dir / "embeddings")
    results_dir = ensure_directory(base_dir / "results")

    abstract_df = scopus_df[["paper_id", "paper_title", "paper_abstract"]].copy()
    abstract_df["text"] = abstract_df["paper_abstract"].map(clean_text)
    abstract_df["text_hash"] = abstract_df["text"].map(text_hash)
    abstract_df = abstract_df[abstract_df["text"] != ""].copy()

    description_records = description_df[["cluster_id", "description"]].copy()
    description_records["text"] = description_records["description"].map(clean_text)
    description_records["text_hash"] = description_records["text"].map(text_hash)
    description_records = description_records[description_records["text"] != ""].copy()

    if abstract_df.empty or description_records.empty:
        raise ValueError("Induced clustering evaluation requires non-empty abstracts and descriptions.")

    abstract_embeddings = ensure_embeddings(
        records_df=abstract_df[["paper_id", "text", "text_hash"]],
        cache_path=_shared_abstract_embeddings_cache_path(scopus_name),
        config=EmbeddingConfig(model=embedding_model, batch_size=batch_size),
    )
    description_embeddings = ensure_embeddings(
        records_df=description_records[["cluster_id", "text", "text_hash"]],
        cache_path=embeddings_dir / _embedding_cache_filename(
            dataset_name=description_name,
            suffix="descriptions",
        ),
        config=EmbeddingConfig(model=embedding_model, batch_size=batch_size),
    )

    abstract_matrix = _normalize_embedding_matrix(
        _embeddings_to_matrix(abstract_embeddings["embedding"].tolist())
    )
    description_matrix = _normalize_embedding_matrix(
        _embeddings_to_matrix(description_embeddings["embedding"].tolist())
    )

    cosine_matrix = abstract_matrix @ description_matrix.T
    best_indices = cosine_matrix.argmax(axis=1)
    cluster_ids = description_embeddings["cluster_id"].astype(int).tolist()

    assignments = abstract_embeddings[["paper_id"]].copy()
    assignments[cluster_output_column] = [cluster_ids[index] for index in best_indices]

    output_df = scopus_df.copy()
    output_df = output_df.merge(assignments, on="paper_id", how="left", suffixes=("", "_induced"))
    output_path = results_dir / f"{description_name}_induced_clusters.csv"
    output_df.to_csv(output_path, index=False, encoding="utf-8")
    return output_df


def compute_silhouette_for_clusters(
    scopus_df: pd.DataFrame,
    dataset_name: str,
    embedding_model: str,
    cluster_column: str = "cluster",
    batch_size: int = 128,
    abstract_cache_name: str | None = None,
) -> pd.DataFrame:
    base_dir = ensure_directory(SILHOUETTE_DIR)

    work_df = scopus_df.dropna(subset=["paper_abstract", cluster_column]).copy()
    work_df["paper_abstract"] = work_df["paper_abstract"].map(clean_text)
    work_df = work_df[work_df["paper_abstract"] != ""].copy()
    if work_df.empty:
        raise ValueError("Silhouette score requires non-empty abstracts with cluster labels.")

    work_df["text"] = work_df["paper_abstract"]
    work_df["text_hash"] = work_df["text"].map(text_hash)

    embedded = ensure_embeddings(
        records_df=work_df[["paper_id", cluster_column, "text", "text_hash"]],
        cache_path=_shared_abstract_embeddings_cache_path(abstract_cache_name or dataset_name),
        config=EmbeddingConfig(model=embedding_model, batch_size=batch_size),
    )
    labels = embedded[cluster_column].astype(int).to_numpy()
    if len(np.unique(labels)) < 2:
        raise ValueError("Silhouette score requires at least two clusters.")

    matrix = _normalize_embedding_matrix(_embeddings_to_matrix(embedded["embedding"].tolist()))
    score = float(silhouette_score(matrix, labels, metric="cosine"))

    result = pd.DataFrame(
        [
            {
                "dataset_name": dataset_name,
                "cluster_column": cluster_column,
                "silhouette_score": score,
                "embedding_model": embedding_model,
                "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        ]
    )
    result.to_csv(base_dir / f"{dataset_name}_silhouette.csv", index=False)
    return result


def compare_clusterings(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    left_cluster_column: str = "cluster",
    right_cluster_column: str = "cluster",
    output_name: str | None = None,
) -> pd.DataFrame:
    left_alias = "left_cluster"
    right_alias = "right_cluster"
    merged = left_df[["paper_id", left_cluster_column]].rename(
        columns={left_cluster_column: left_alias}
    ).merge(
        right_df[["paper_id", right_cluster_column]].rename(
            columns={right_cluster_column: right_alias}
        ),
        on="paper_id",
        how="inner",
    ).dropna(subset=[left_alias, right_alias])

    if merged.empty:
        raise ValueError("No aligned paper_id values were found for clustering comparison.")

    ari = float(
        adjusted_rand_score(
            merged[left_alias].astype(int),
            merged[right_alias].astype(int),
        )
    )
    nmi = float(
        normalized_mutual_info_score(
            merged[left_alias].astype(int),
            merged[right_alias].astype(int),
        )
    )

    result = pd.DataFrame(
        [
            {
                "n_aligned_papers": len(merged),
                "ari": ari,
                "nmi": nmi,
            }
        ]
    )
    if output_name:
        ensure_directory(SIMILARITY_DIR)
        result.to_csv(SIMILARITY_DIR / f"{output_name}.csv", index=False)
    return result


def compute_modularity_result(
    scopus_df: pd.DataFrame,
    dataset_name: str,
    cluster_column: str = "cluster",
    min_weight: float = 0.001,
    top_n_edges: int | None = None,
    min_degree: int = 1,
) -> pd.DataFrame:
    score = compute_modularity_for_clusters(
        scopus_df=scopus_df,
        cluster_column=cluster_column,
        min_weight=min_weight,
        top_n_edges=top_n_edges,
        min_degree=min_degree,
    )
    ensure_directory(MODULARITY_DIR)
    result = pd.DataFrame(
        [
            {
                "dataset_name": dataset_name,
                "cluster_column": cluster_column,
                "modularity": score,
            }
        ]
    )
    result.to_csv(MODULARITY_DIR / f"{dataset_name}_modularity.csv", index=False)
    return result


def compute_summac_factual_consistency(
    description_df: pd.DataFrame | str | Path,
    scopus_df: pd.DataFrame,
    description_name: str,
    device: str | None = None,
    summac_model=None,
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    if not isinstance(description_df, pd.DataFrame):
        description_df = load_description_csv(description_df)

    base_dir = ensure_directory(Path(output_dir) if output_dir is not None else (SUMMAC_DIR / "results"))
    results_dir = base_dir

    cluster_descriptions = {
        int(row["cluster_id"]): clean_text(row["description"])
        for _, row in description_df.iterrows()
        if clean_text(row["description"])
    }
    cluster_abstracts = _cluster_abstracts_from_references(
        description_df=description_df,
        scopus_df=scopus_df,
    )
    score_payload = summac_factual_consistency(
        cluster_descriptions=cluster_descriptions,
        cluster_abstracts=cluster_abstracts,
        summac_model=summac_model,
        device=device,
    )

    detail_rows = [
        {
            "cluster_id": int(cluster_id),
            "summac_factual_consistency": float(score),
            "n_reference_abstracts": len(cluster_abstracts.get(int(cluster_id), [])),
        }
        for cluster_id, score in score_payload["summac_by_cluster"].items()
    ]
    detail_df = pd.DataFrame(detail_rows)
    if detail_df.empty:
        detail_df = pd.DataFrame(
            columns=["cluster_id", "summac_factual_consistency", "n_reference_abstracts"]
        )
    else:
        detail_df = detail_df.sort_values("cluster_id").reset_index(drop=True)

    detail_path = results_dir / f"{description_name}_summac_by_cluster.csv"
    detail_df.to_csv(detail_path, index=False, encoding="utf-8")

    result = pd.DataFrame(
        [
            {
                "description_file": description_name,
                "summac_factual_consistency": score_payload["summac_factual_consistency"],
                "n_clusters_scored": int(len(score_payload["summac_by_cluster"])),
                "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
                "device": _resolve_summac_device(device),
                "detail_file": detail_path.name,
            }
        ]
    )
    result.to_csv(results_dir / f"{description_name}_summac.csv", index=False, encoding="utf-8")
    return result


def compute_human_alignment_bertscore(
    generated_df: pd.DataFrame | str | Path,
    human_df: pd.DataFrame | str | Path,
    generated_name: str,
    provider: str,
    scopus_name: str,
    model_type: str = "roberta-large",
    lang: str = "en",
    device: str | None = None,
    batch_size: int = 64,
    rescale_with_baseline: bool = False,
    output_dir: str | Path | None = None,
    scorer=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not isinstance(generated_df, pd.DataFrame):
        generated_df = load_description_csv(generated_df)
    if not isinstance(human_df, pd.DataFrame):
        human_df = load_description_csv(human_df)

    if generated_df.empty or human_df.empty:
        raise ValueError("Human alignment requires non-empty generated and human description tables.")

    base_dir = ensure_directory(Path(output_dir) if output_dir is not None else HUMAN_ALIGNMENT_DIR)
    matches_dir = ensure_directory(base_dir / "matches" / provider)
    summaries_dir = ensure_directory(base_dir / "summaries" / provider)

    scorer = scorer or load_bert_scorer(
        model_type=model_type,
        lang=lang,
        device=device,
        batch_size=batch_size,
        rescale_with_baseline=rescale_with_baseline,
    )

    generated_records = [
        {
            "cluster_id": int(row["cluster_id"]),
            "description": clean_text(row["description"]),
        }
        for _, row in generated_df.iterrows()
    ]
    human_records = [
        {
            "cluster_id": int(row["cluster_id"]),
            "description": clean_text(row["description"]),
        }
        for _, row in human_df.iterrows()
    ]

    scoring_pairs: list[tuple[int, int]] = []
    cands: list[str] = []
    refs: list[str] = []
    pair_metrics: dict[tuple[int, int], dict[str, float]] = {}

    for generated_index, generated_record in enumerate(generated_records):
        for human_index, human_record in enumerate(human_records):
            generated_text = generated_record["description"]
            human_text = human_record["description"]
            if not generated_text or not human_text:
                pair_metrics[(generated_index, human_index)] = {
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                }
                continue
            scoring_pairs.append((generated_index, human_index))
            cands.append(generated_text)
            refs.append(human_text)

    if scoring_pairs:
        precision_scores, recall_scores, f1_scores = scorer.score(cands, refs)
        for pair_key, precision, recall, f1 in zip(
            scoring_pairs,
            precision_scores.tolist(),
            recall_scores.tolist(),
            f1_scores.tolist(),
        ):
            pair_metrics[pair_key] = {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }

    similarity_matrix = np.zeros((len(generated_records), len(human_records)), dtype=np.float32)
    for generated_index, generated_record in enumerate(generated_records):
        for human_index, human_record in enumerate(human_records):
            metrics = pair_metrics[(generated_index, human_index)]
            similarity_matrix[generated_index, human_index] = float(metrics["f1"])

    row_ind, col_ind = linear_sum_assignment(similarity_matrix, maximize=True)
    matched_rows: list[dict[str, object]] = []
    matched_scores: list[float] = []
    match_rank = 0
    matched_generated_indices = set()
    matched_human_indices = set()
    for generated_index, human_index in zip(row_ind.tolist(), col_ind.tolist()):
        match_rank += 1
        matched_generated_indices.add(generated_index)
        matched_human_indices.add(human_index)
        metrics = pair_metrics[(generated_index, human_index)]
        generated_record = generated_records[generated_index]
        human_record = human_records[human_index]
        matched_scores.append(float(metrics["f1"]))
        matched_rows.append(
            {
                "paper": scopus_name,
                "generated_name": generated_name,
                "match_rank": match_rank,
                "generated_cluster_id": generated_record["cluster_id"],
                "human_cluster_id": human_record["cluster_id"],
                "bertscore_precision": float(metrics["precision"]),
                "bertscore_recall": float(metrics["recall"]),
                "bertscore_f1": float(metrics["f1"]),
                "generated_description": generated_record["description"],
                "human_description": human_record["description"],
            }
        )

    matches_df = pd.DataFrame(matched_rows)
    matches_path = matches_dir / f"{scopus_name}__{generated_name}_human_alignment_matches.csv"
    matches_df.to_csv(matches_path, index=False, encoding="utf-8")

    summary_df = pd.DataFrame(
        [
            {
                "paper": scopus_name,
                "generated_name": generated_name,
                "provider": provider,
                "human_alignment_bertscore_mean": (
                    float(np.mean(matched_scores)) if matched_scores else np.nan
                ),
                "human_alignment_bertscore_sum": (
                    float(np.sum(matched_scores)) if matched_scores else np.nan
                ),
                "human_alignment_bertscore_min": (
                    float(np.min(matched_scores)) if matched_scores else np.nan
                ),
                "human_alignment_bertscore_std": (
                    float(np.std(matched_scores)) if matched_scores else np.nan
                ),
                "n_matches": int(len(matched_scores)),
                "generated_cluster_count": int(len(generated_records)),
                "human_cluster_count": int(len(human_records)),
                "unmatched_generated_clusters": int(len(generated_records) - len(matched_generated_indices)),
                "unmatched_human_clusters": int(len(human_records) - len(matched_human_indices)),
                "bert_score_model_type": model_type,
                "bert_score_lang": lang,
                "bert_score_rescale_with_baseline": bool(rescale_with_baseline),
                "device": _resolve_torch_device(device),
                "matches_file": matches_path.name,
                "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        ]
    )
    summary_path = summaries_dir / f"{scopus_name}__{generated_name}_human_alignment_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")
    return summary_df, matches_df
