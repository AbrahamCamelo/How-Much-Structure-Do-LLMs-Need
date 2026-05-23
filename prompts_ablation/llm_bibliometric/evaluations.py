from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)

from .bib_coupling import compute_modularity_for_clusters
from .constants import (
    COVERAGE_DIR,
    MODULARITY_DIR,
    QUALITY_DIR,
    SHARED_EMBEDDINGS_DIR,
    SILHOUETTE_DIR,
    SIMILARITY_DIR,
)
from .descriptions import load_description_csv
from .embeddings import EmbeddingConfig, ensure_embeddings
from .output_layout import evaluation_artifact_dir
from .utils import clean_text, ensure_directory, slugify_filename, split_sentences, text_hash


def _normalize_embedding_matrix(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return matrix / norms


def _embeddings_to_matrix(embeddings: list[list[float]]) -> np.ndarray:
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("Embeddings must form a 2D matrix.")
    return matrix


def _shared_abstract_embeddings_cache_path(scopus_name: str, embedding_model: str) -> Path:
    return ensure_directory(
        evaluation_artifact_dir(SHARED_EMBEDDINGS_DIR, embedding_model) / "abstracts"
    ) / _embedding_cache_filename(
        dataset_name=scopus_name,
        suffix="abstracts",
    )


def _embedding_cache_filename(dataset_name: str, suffix: str) -> str:
    slug = slugify_filename(dataset_name)
    short_slug = slug[:16] if slug else "cache"
    return f"{short_slug}_{text_hash(dataset_name)[:12]}_{suffix}.parquet"


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

    base_dir = ensure_directory(evaluation_artifact_dir(COVERAGE_DIR, embedding_model))
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

    base_dir = ensure_directory(evaluation_artifact_dir(QUALITY_DIR, embedding_model))
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
        cache_path=_shared_abstract_embeddings_cache_path(scopus_name, embedding_model),
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
    base_dir = ensure_directory(evaluation_artifact_dir(SILHOUETTE_DIR, embedding_model))

    work_df = scopus_df.dropna(subset=["paper_abstract", cluster_column]).copy()
    work_df["paper_abstract"] = work_df["paper_abstract"].map(clean_text)
    work_df = work_df[work_df["paper_abstract"] != ""].copy()
    if work_df.empty:
        raise ValueError("Silhouette score requires non-empty abstracts with cluster labels.")

    work_df["text"] = work_df["paper_abstract"]
    work_df["text_hash"] = work_df["text"].map(text_hash)

    embedded = ensure_embeddings(
        records_df=work_df[["paper_id", cluster_column, "text", "text_hash"]],
        cache_path=_shared_abstract_embeddings_cache_path(
            abstract_cache_name or dataset_name,
            embedding_model,
        ),
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
    embedding_model: str | None = None,
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
        target_dir = ensure_directory(evaluation_artifact_dir(SIMILARITY_DIR, embedding_model))
        result.to_csv(target_dir / f"{output_name}.csv", index=False)
    return result


def compute_modularity_result(
    scopus_df: pd.DataFrame,
    dataset_name: str,
    cluster_column: str = "cluster",
    embedding_model: str | None = None,
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
    target_dir = ensure_directory(evaluation_artifact_dir(MODULARITY_DIR, embedding_model))
    result = pd.DataFrame(
        [
            {
                "dataset_name": dataset_name,
                "cluster_column": cluster_column,
                "modularity": score,
            }
        ]
    )
    result.to_csv(target_dir / f"{dataset_name}_modularity.csv", index=False)
    return result
