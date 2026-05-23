from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .bib_coupling import (
    BiblioCouplingResult,
    run_bibliographic_coupling,
    select_top_clusters_by_link_strength,
)
from .cli_paths import resolve_labeled_scopus_file, resolve_raw_scopus_file
from .constants import (
    CLUSTERS_DIR,
    COVERAGE_DIR,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_SIGNIFICANT_PAPERS_PER_CLUSTER,
    DESCRIPTION_DIR,
    FULL_RUN_DIR,
    HUMAN_ALIGNMENT_DIR,
    MODULARITY_DIR,
    QUALITY_DIR,
    REFERENCE_GROUNDED_COVERAGE_DIR,
    SCOPUS_DIR,
    SILHOUETTE_DIR,
    SIMILARITY_DIR,
    SUMMAC_DIR,
)
from .descriptions import (
    load_description_csv,
    require_human_cluster_target,
    require_human_description_file,
)
from .evaluations import (
    compare_clusterings,
    compute_coverage_of_corpus,
    compute_human_alignment_bertscore,
    compute_modularity_result,
    compute_quality_of_induced_clustering,
    compute_reference_grounded_coverage,
    compute_silhouette_for_clusters,
    compute_summac_factual_consistency,
    load_bert_scorer,
    load_summac_model,
)
from .pipelines import (
    run_pipeline_1,
    run_pipeline_2,
    run_pipeline_3,
    run_pipeline_4,
    run_pipeline_5,
    run_pipeline_6,
)
from .prompt_query import resolve_query_text
from .scopus import load_scopus_csv
from .utils import ensure_directory, slugify_filename


PROVIDER_ALIASES = {
    "openai": "chatgpt",
    "chatgpt": "chatgpt",
    "google": "gemini",
    "gemini": "gemini",
    "anthropic": "claude",
    "claude": "claude",
}
PIPELINE_NAMES = (
    "pipeline_1",
    "pipeline_2",
    "pipeline_3",
    "pipeline_4",
    "pipeline_5",
    "pipeline_6",
)
INDUCED_CLUSTER_COLUMN = "induced_cluster"


@dataclass(frozen=True)
class DescriptionRunResult:
    raw_scopus_file: Path
    clustered_scopus_file: Path
    human_description_file: Path | None
    query: str
    summary_path: Path
    summary_df: pd.DataFrame
    cluster_result: BiblioCouplingResult | None


@dataclass(frozen=True)
class EvaluationRunResult:
    raw_scopus_file: Path
    clustered_scopus_file: Path
    human_description_file: Path | None
    summary_path: Path
    summary_df: pd.DataFrame


@dataclass(frozen=True)
class SummacRunResult:
    raw_scopus_file: Path
    summary_path: Path
    summary_df: pd.DataFrame
    output_dir: Path


@dataclass(frozen=True)
class ReferenceGroundedCoverageRunResult:
    raw_scopus_file: Path
    summary_path: Path
    summary_df: pd.DataFrame
    output_dir: Path


@dataclass(frozen=True)
class HumanAlignmentRunResult:
    raw_scopus_file: Path
    human_description_file: Path
    summary_path: Path
    summary_df: pd.DataFrame
    output_dir: Path


@dataclass(frozen=True)
class DescriptionArtifact:
    source_name: str
    description_path: Path
    description_df: pd.DataFrame
    dataset_name: str


@dataclass(frozen=True)
class DescriptionBatchRunResult:
    summary_path: Path
    summary_df: pd.DataFrame
    results: list[DescriptionRunResult]


def _canonical_provider(provider: str) -> str:
    normalized = str(provider).strip().lower()
    if normalized not in PROVIDER_ALIASES:
        raise ValueError(f"Unsupported provider: {provider}")
    return PROVIDER_ALIASES[normalized]


def _pipeline_output_path(
    pipeline_name: str,
    provider: str,
    scopus_file: str | Path,
) -> Path:
    return DESCRIPTION_DIR / pipeline_name / provider / Path(scopus_file).name


def _description_dataset_name(description_path: str | Path) -> str:
    path = Path(description_path)
    try:
        relative = path.relative_to(DESCRIPTION_DIR)
    except ValueError:
        relative = path
    parts = [slugify_filename(part) for part in relative.with_suffix("").parts]
    return "__".join(part for part in parts if part)


def _workflow_summary_path(kind: str, provider: str, scopus_file: str | Path) -> Path:
    paper_stem = Path(scopus_file).stem
    target_dir = ensure_directory(FULL_RUN_DIR / kind / provider)
    return target_dir / f"{paper_stem}_{kind}.csv"


def _display_name(path: str | Path | None) -> str:
    if path is None:
        return ""
    return Path(path).name


def list_available_raw_scopus_files() -> list[Path]:
    files = sorted(path for path in SCOPUS_DIR.glob("*.csv") if path.is_file())
    if not files:
        raise FileNotFoundError(f"No Scopus CSV files were found in {SCOPUS_DIR}.")
    return files


def _expected_description_artifacts(
    scopus_file: str | Path,
    provider: str,
) -> list[DescriptionArtifact]:
    artifacts: list[DescriptionArtifact] = []
    file_name = Path(scopus_file).name

    for pipeline_name in PIPELINE_NAMES:
        description_path = _pipeline_output_path(pipeline_name, provider, file_name)
        if not description_path.exists():
            raise FileNotFoundError(
                f"Missing description output for {pipeline_name}: {description_path}"
            )
        artifacts.append(
            DescriptionArtifact(
                source_name=pipeline_name,
                description_path=description_path,
                description_df=load_description_csv(description_path),
                dataset_name=_description_dataset_name(description_path),
            )
        )

    human_description_file = require_human_description_file(scopus_file)
    artifacts.append(
        DescriptionArtifact(
            source_name="human_descriptions",
            description_path=human_description_file,
            description_df=load_description_csv(human_description_file),
            dataset_name=_description_dataset_name(human_description_file),
        )
    )

    return artifacts


def _load_existing_description_summary(summary_path: Path) -> pd.DataFrame | None:
    if not summary_path.exists():
        return None
    try:
        summary_df = pd.read_csv(summary_path)
    except Exception:
        return None
    if summary_df.empty:
        return None
    return summary_df


def _validate_existing_description_output(
    description_path: Path,
    expected_cluster_ids: list[int],
) -> pd.DataFrame | None:
    if not description_path.exists():
        return None

    try:
        description_df = load_description_csv(description_path)
    except Exception:
        return None

    actual_cluster_ids = sorted(description_df["cluster_id"].astype(int).tolist())
    if actual_cluster_ids != sorted(expected_cluster_ids):
        return None
    return description_df


def _resolve_clustered_scopus_for_descriptions(
    resolved_scopus_file: Path,
    raw_scopus_df: pd.DataFrame,
    target_clusters: int,
    only_missing: bool,
    min_weight: float,
    top_n_edges: int | None,
    min_degree: int,
    resolution: float,
    min_cluster_size: int,
    resolution_step: float,
    min_resolution: float,
    max_resolution: float,
    resolution_max_iterations: int,
    resolution_min_interval: float,
) -> tuple[Path, pd.DataFrame, BiblioCouplingResult | None, bool]:
    clustered_scopus_file = CLUSTERS_DIR / f"{resolved_scopus_file.stem}_louvain.csv"

    if only_missing and clustered_scopus_file.exists():
        try:
            labeled_scopus_df = load_scopus_csv(clustered_scopus_file)
            observed_cluster_count = int(labeled_scopus_df["cluster"].dropna().nunique())
            if observed_cluster_count == int(target_clusters):
                return clustered_scopus_file, labeled_scopus_df, None, True
        except Exception:
            pass

    cluster_result = run_bibliographic_coupling(
        scopus_df=raw_scopus_df,
        min_weight=min_weight,
        top_n_edges=top_n_edges,
        min_degree=min_degree,
        resolution=resolution,
        min_cluster_size=min_cluster_size,
        target_cluster_count=target_clusters,
        resolution_step=resolution_step,
        min_resolution=min_resolution,
        max_resolution=max_resolution,
        resolution_max_iterations=resolution_max_iterations,
        resolution_min_interval=resolution_min_interval,
    )
    if int(cluster_result.cluster_count) != int(target_clusters):
        raise ValueError(
            "The Louvain clustering did not match the required number of human clusters. "
            f"Expected {int(target_clusters)} clusters, but obtained "
            f"{int(cluster_result.cluster_count)}."
        )

    ensure_directory(clustered_scopus_file.parent)
    cluster_result.labeled_scopus.to_csv(clustered_scopus_file, index=False, encoding="utf-8")
    return clustered_scopus_file, cluster_result.labeled_scopus.copy(), cluster_result, False


def _build_description_summary(
    outputs: list[tuple[str, Path, pd.DataFrame, bool]],
    provider: str,
    query: str,
    raw_scopus_file: Path,
    clustered_scopus_file: Path,
    human_description_file: Path | None,
    target_clusters: int,
    labeled_scopus_df: pd.DataFrame,
    cluster_result: BiblioCouplingResult | None,
    used_existing_clustered_scopus: bool,
    summary_path: Path,
) -> pd.DataFrame:
    existing_summary_df = _load_existing_description_summary(summary_path)
    observed_cluster_count = int(labeled_scopus_df["cluster"].dropna().nunique())

    if cluster_result is not None:
        louvain_resolution: object = float(cluster_result.resolution)
        matched_target_cluster_count: object = bool(cluster_result.matched_target_cluster_count)
    elif existing_summary_df is not None:
        first_row = existing_summary_df.iloc[0]
        louvain_resolution = first_row.get("louvain_resolution", pd.NA)
        matched_target_cluster_count = first_row.get(
            "matched_target_cluster_count",
            observed_cluster_count == int(target_clusters),
        )
    else:
        louvain_resolution = pd.NA
        matched_target_cluster_count = observed_cluster_count == int(target_clusters)

    summary_rows: list[dict[str, object]] = []
    for pipeline_name, output_path, output_df, used_existing_output in outputs:
        summary_rows.append(
            {
                "pipeline": pipeline_name,
                "provider": provider,
                "query": query,
                "scopus_file": _display_name(raw_scopus_file),
                "clustered_scopus_file": _display_name(clustered_scopus_file),
                "description_file": _display_name(output_path),
                "n_description_rows": int(len(output_df)),
                "n_description_clusters": int(output_df["cluster_id"].nunique()),
                "human_description_file": _display_name(human_description_file),
                "target_cluster_count": target_clusters,
                "observed_louvain_cluster_count": observed_cluster_count,
                "louvain_resolution": louvain_resolution,
                "matched_target_cluster_count": matched_target_cluster_count,
                "used_existing_clustered_scopus": used_existing_clustered_scopus,
                "used_existing_description_output": used_existing_output,
            }
        )
    return pd.DataFrame(summary_rows)


def run_all_descriptions(
    scopus_file: str | Path,
    provider: str,
    model: str | None = None,
    prompts_file: str | Path | None = None,
    queries_file: str | Path | None = None,
    only_missing: bool = False,
    temperature: float = 0.2,
    significant_papers_per_cluster: int = DEFAULT_SIGNIFICANT_PAPERS_PER_CLUSTER,
    min_weight: float = 0.0001,
    top_n_edges: int | None = None,
    min_degree: int = 1,
    resolution: float = 1.0,
    min_cluster_size: int = 3,
    resolution_step: float = 0.1,
    min_resolution: float = 0.05,
    max_resolution: float = 5.0,
    resolution_max_iterations: int = 50,
    resolution_min_interval: float = 0.001,
) -> DescriptionRunResult:
    canonical_provider = _canonical_provider(provider)
    resolved_scopus_file = resolve_raw_scopus_file(scopus_file)
    raw_scopus_df = load_scopus_csv(resolved_scopus_file)
    query = resolve_query_text(
        reference_path=resolved_scopus_file,
        queries_file=queries_file or "queries.xlsx",
    )

    human_description_file, target_clusters = require_human_cluster_target(resolved_scopus_file)
    clustered_scopus_file, labeled_scopus_df, cluster_result, used_existing_clustered_scopus = (
        _resolve_clustered_scopus_for_descriptions(
            resolved_scopus_file=resolved_scopus_file,
            raw_scopus_df=raw_scopus_df,
            target_clusters=target_clusters,
            only_missing=only_missing,
            min_weight=min_weight,
            top_n_edges=top_n_edges,
            min_degree=min_degree,
            resolution=resolution,
            min_cluster_size=min_cluster_size,
            resolution_step=resolution_step,
            min_resolution=min_resolution,
            max_resolution=max_resolution,
            resolution_max_iterations=resolution_max_iterations,
            resolution_min_interval=resolution_min_interval,
        )
    )

    labeled_cluster_ids = sorted(
        pd.to_numeric(labeled_scopus_df["cluster"].dropna(), errors="raise").astype(int).unique().tolist()
    )
    top_cluster_ids = select_top_clusters_by_link_strength(
        labeled_scopus_df=labeled_scopus_df,
        top_k_clusters=target_clusters,
    )
    rerun_all_pipelines = not only_missing or not used_existing_clustered_scopus

    pipeline_runners: dict[str, tuple[Path, list[int], object]] = {
        "pipeline_1": (
            _pipeline_output_path("pipeline_1", canonical_provider, resolved_scopus_file),
            list(range(1, target_clusters + 1)),
            lambda: run_pipeline_1(
                query=query,
                prompt=None,
                provider=canonical_provider,
                model=model,
                output_file=_pipeline_output_path(
                    "pipeline_1",
                    canonical_provider,
                    resolved_scopus_file,
                ),
                target_cluster_count=target_clusters,
                temperature=temperature,
                prompts_file=prompts_file,
            ),
        ),
        "pipeline_2": (
            _pipeline_output_path("pipeline_2", canonical_provider, resolved_scopus_file),
            list(range(1, target_clusters + 1)),
            lambda: run_pipeline_2(
                query=query,
                prompt=None,
                scopus_df=raw_scopus_df,
                provider=canonical_provider,
                model=model,
                output_file=_pipeline_output_path(
                    "pipeline_2",
                    canonical_provider,
                    resolved_scopus_file,
                ),
                target_cluster_count=target_clusters,
                temperature=temperature,
                prompts_file=prompts_file,
            ),
        ),
        "pipeline_3": (
            _pipeline_output_path("pipeline_3", canonical_provider, resolved_scopus_file),
            list(range(1, target_clusters + 1)),
            lambda: run_pipeline_3(
                query=query,
                prompt=None,
                scopus_df=raw_scopus_df,
                provider=canonical_provider,
                model=model,
                output_file=_pipeline_output_path(
                    "pipeline_3",
                    canonical_provider,
                    resolved_scopus_file,
                ),
                target_cluster_count=target_clusters,
                temperature=temperature,
                prompts_file=prompts_file,
            ),
        ),
        "pipeline_4": (
            _pipeline_output_path("pipeline_4", canonical_provider, resolved_scopus_file),
            labeled_cluster_ids,
            lambda: run_pipeline_4(
                query=query,
                prompt=None,
                labeled_scopus_df=labeled_scopus_df.copy(),
                provider=canonical_provider,
                model=model,
                output_file=_pipeline_output_path(
                    "pipeline_4",
                    canonical_provider,
                    resolved_scopus_file,
                ),
                target_cluster_count=target_clusters,
                temperature=temperature,
                prompts_file=prompts_file,
            ),
        ),
        "pipeline_5": (
            _pipeline_output_path("pipeline_5", canonical_provider, resolved_scopus_file),
            labeled_cluster_ids,
            lambda: run_pipeline_5(
                query=query,
                prompt=None,
                labeled_scopus_df=labeled_scopus_df.copy(),
                provider=canonical_provider,
                model=model,
                output_file=_pipeline_output_path(
                    "pipeline_5",
                    canonical_provider,
                    resolved_scopus_file,
                ),
                target_cluster_count=target_clusters,
                temperature=temperature,
                prompts_file=prompts_file,
            ),
        ),
        "pipeline_6": (
            _pipeline_output_path("pipeline_6", canonical_provider, resolved_scopus_file),
            top_cluster_ids,
            lambda: run_pipeline_6(
                query=query,
                prompt=None,
                labeled_scopus_df=labeled_scopus_df.copy(),
                provider=canonical_provider,
                model=model,
                output_file=_pipeline_output_path(
                    "pipeline_6",
                    canonical_provider,
                    resolved_scopus_file,
                ),
                top_k_clusters=target_clusters,
                target_cluster_count=target_clusters,
                significant_papers_per_cluster=significant_papers_per_cluster,
                temperature=temperature,
                prompts_file=prompts_file,
            ),
        ),
    }

    outputs: list[tuple[str, Path, pd.DataFrame, bool]] = []
    for pipeline_name in PIPELINE_NAMES:
        output_path, expected_cluster_ids, runner = pipeline_runners[pipeline_name]
        existing_output = None
        if not rerun_all_pipelines:
            existing_output = _validate_existing_description_output(
                description_path=output_path,
                expected_cluster_ids=expected_cluster_ids,
            )
        if existing_output is not None:
            outputs.append((pipeline_name, output_path, existing_output, True))
            continue
        outputs.append((pipeline_name, output_path, runner(), False))
    summary_path = _workflow_summary_path(
        kind="descriptions",
        provider=canonical_provider,
        scopus_file=resolved_scopus_file,
    )
    summary_df = _build_description_summary(
        outputs=outputs,
        provider=canonical_provider,
        query=query,
        raw_scopus_file=resolved_scopus_file,
        clustered_scopus_file=clustered_scopus_file,
        human_description_file=human_description_file,
        target_clusters=target_clusters,
        labeled_scopus_df=labeled_scopus_df,
        cluster_result=cluster_result,
        used_existing_clustered_scopus=used_existing_clustered_scopus,
        summary_path=summary_path,
    )
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")

    return DescriptionRunResult(
        raw_scopus_file=resolved_scopus_file,
        clustered_scopus_file=clustered_scopus_file,
        human_description_file=human_description_file,
        query=query,
        summary_path=summary_path,
        summary_df=summary_df,
        cluster_result=cluster_result,
    )


def run_all_descriptions_for_all_scopus(
    provider: str,
    model: str | None = None,
    prompts_file: str | Path | None = None,
    queries_file: str | Path | None = None,
    only_missing: bool = False,
    temperature: float = 0.2,
    significant_papers_per_cluster: int = DEFAULT_SIGNIFICANT_PAPERS_PER_CLUSTER,
    min_weight: float = 0.0001,
    top_n_edges: int | None = None,
    min_degree: int = 1,
    resolution: float = 1.0,
    min_cluster_size: int = 3,
    resolution_step: float = 0.1,
    min_resolution: float = 0.05,
    max_resolution: float = 5.0,
    resolution_max_iterations: int = 50,
    resolution_min_interval: float = 0.001,
) -> DescriptionBatchRunResult:
    canonical_provider = _canonical_provider(provider)
    results: list[DescriptionRunResult] = []
    summary_frames: list[pd.DataFrame] = []

    for scopus_file in list_available_raw_scopus_files():
        result = run_all_descriptions(
            scopus_file=scopus_file,
            provider=canonical_provider,
            model=model,
            prompts_file=prompts_file,
            queries_file=queries_file,
            only_missing=only_missing,
            temperature=temperature,
            significant_papers_per_cluster=significant_papers_per_cluster,
            min_weight=min_weight,
            top_n_edges=top_n_edges,
            min_degree=min_degree,
            resolution=resolution,
            min_cluster_size=min_cluster_size,
            resolution_step=resolution_step,
            min_resolution=min_resolution,
            max_resolution=max_resolution,
            resolution_max_iterations=resolution_max_iterations,
            resolution_min_interval=resolution_min_interval,
        )
        results.append(result)
        summary_frames.append(
            result.summary_df.assign(
                paper=result.raw_scopus_file.stem,
                per_paper_summary_file=_display_name(result.summary_path),
            )
        )

    combined_summary_df = pd.concat(summary_frames, ignore_index=True)
    summary_path = ensure_directory(FULL_RUN_DIR / "descriptions" / canonical_provider) / (
        "all_papers_descriptions.csv"
    )
    combined_summary_df.to_csv(summary_path, index=False, encoding="utf-8")

    return DescriptionBatchRunResult(
        summary_path=summary_path,
        summary_df=combined_summary_df,
        results=results,
    )


def run_summac_for_paper(
    scopus_file: str | Path,
    provider: str,
    output_dir: str | Path | None = None,
    device: str | None = None,
) -> SummacRunResult:
    canonical_provider = _canonical_provider(provider)
    resolved_scopus_file = resolve_raw_scopus_file(scopus_file)
    raw_scopus_df = load_scopus_csv(resolved_scopus_file)

    summac_artifacts = [
        artifact
        for artifact in _expected_description_artifacts(
            scopus_file=resolved_scopus_file,
            provider=canonical_provider,
        )
        if artifact.source_name in {"pipeline_2", "pipeline_3", "pipeline_4", "pipeline_5", "pipeline_6"}
    ]

    resolved_output_dir = ensure_directory(
        Path(output_dir)
        if output_dir is not None
        else (SUMMAC_DIR / canonical_provider / resolved_scopus_file.stem)
    )
    summac_model = load_summac_model(device=device)

    summary_rows: list[dict[str, object]] = []
    for artifact in summac_artifacts:
        summac_score = pd.NA
        summac_error = ""
        detail_file = f"{artifact.dataset_name}_summac_by_cluster.csv"

        try:
            result_df = compute_summac_factual_consistency(
                description_df=artifact.description_df,
                scopus_df=raw_scopus_df,
                description_name=artifact.dataset_name,
                summac_model=summac_model,
                device=device,
                output_dir=resolved_output_dir,
            )
            summac_score = float(result_df.iloc[0]["summac_factual_consistency"])
            detail_file = str(result_df.iloc[0]["detail_file"])
        except Exception as error:
            summac_error = str(error)

        summary_rows.append(
            {
                "source_name": artifact.source_name,
                "provider": canonical_provider,
                "scopus_file": resolved_scopus_file.name,
                "description_file": artifact.description_path.name,
                "description_dataset_name": artifact.dataset_name,
                "summac_factual_consistency": summac_score,
                "summac_error": summac_error,
                "summac_detail_file": detail_file,
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("source_name").reset_index(drop=True)
    summary_path = resolved_output_dir / f"{resolved_scopus_file.stem}_summac_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")

    return SummacRunResult(
        raw_scopus_file=resolved_scopus_file,
        summary_path=summary_path,
        summary_df=summary_df,
        output_dir=resolved_output_dir,
    )


def run_reference_grounded_coverage_for_paper(
    scopus_file: str | Path,
    provider: str,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 128,
    output_dir: str | Path | None = None,
) -> ReferenceGroundedCoverageRunResult:
    canonical_provider = _canonical_provider(provider)
    resolved_scopus_file = resolve_raw_scopus_file(scopus_file)
    raw_scopus_df = load_scopus_csv(resolved_scopus_file)

    eligible_artifacts = [
        artifact
        for artifact in _expected_description_artifacts(
            scopus_file=resolved_scopus_file,
            provider=canonical_provider,
        )
        if artifact.source_name in {"pipeline_2", "pipeline_3", "pipeline_4", "pipeline_5", "pipeline_6"}
    ]

    resolved_output_dir = ensure_directory(
        Path(output_dir)
        if output_dir is not None
        else (REFERENCE_GROUNDED_COVERAGE_DIR / canonical_provider / resolved_scopus_file.stem)
    )

    summary_rows: list[dict[str, object]] = []
    for artifact in eligible_artifacts:
        metric_score = pd.NA
        metric_error = ""
        detail_file = f"{artifact.dataset_name}_reference_grounded_coverage_by_cluster.csv"

        try:
            result_df = compute_reference_grounded_coverage(
                description_df=artifact.description_df,
                scopus_df=raw_scopus_df,
                description_name=artifact.dataset_name,
                embedding_model=embedding_model,
                batch_size=batch_size,
                output_dir=resolved_output_dir,
            )
            metric_score = float(result_df.iloc[0]["reference_grounded_coverage"])
            detail_file = str(result_df.iloc[0]["detail_file"])
        except Exception as error:
            metric_error = str(error)

        summary_rows.append(
            {
                "source_name": artifact.source_name,
                "provider": canonical_provider,
                "scopus_file": resolved_scopus_file.name,
                "description_file": artifact.description_path.name,
                "description_dataset_name": artifact.dataset_name,
                "reference_grounded_coverage": metric_score,
                "reference_grounded_coverage_error": metric_error,
                "reference_grounded_coverage_detail_file": detail_file,
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("source_name").reset_index(drop=True)
    summary_path = resolved_output_dir / f"{resolved_scopus_file.stem}_reference_grounded_coverage_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")

    return ReferenceGroundedCoverageRunResult(
        raw_scopus_file=resolved_scopus_file,
        summary_path=summary_path,
        summary_df=summary_df,
        output_dir=resolved_output_dir,
    )


def run_human_alignment_for_paper(
    scopus_file: str | Path,
    provider: str,
    model_type: str = "roberta-large",
    lang: str = "en",
    device: str | None = None,
    batch_size: int = 64,
    rescale_with_baseline: bool = False,
    output_dir: str | Path | None = None,
) -> HumanAlignmentRunResult:
    canonical_provider = _canonical_provider(provider)
    resolved_scopus_file = resolve_raw_scopus_file(scopus_file)
    human_description_file = require_human_description_file(resolved_scopus_file)
    human_description_df = load_description_csv(human_description_file)

    pipeline_artifacts = [
        artifact
        for artifact in _expected_description_artifacts(
            scopus_file=resolved_scopus_file,
            provider=canonical_provider,
        )
        if artifact.source_name.startswith("pipeline_")
    ]

    resolved_output_dir = ensure_directory(
        Path(output_dir) if output_dir is not None else HUMAN_ALIGNMENT_DIR
    )
    scorer = load_bert_scorer(
        model_type=model_type,
        lang=lang,
        device=device,
        batch_size=batch_size,
        rescale_with_baseline=rescale_with_baseline,
    )

    summary_frames: list[pd.DataFrame] = []
    for artifact in pipeline_artifacts:
        summary_df, _ = compute_human_alignment_bertscore(
            generated_df=artifact.description_df,
            human_df=human_description_df,
            generated_name=artifact.dataset_name,
            provider=canonical_provider,
            scopus_name=resolved_scopus_file.stem,
            model_type=model_type,
            lang=lang,
            device=device,
            batch_size=batch_size,
            rescale_with_baseline=rescale_with_baseline,
            output_dir=resolved_output_dir,
            scorer=scorer,
        )
        summary_df = summary_df.assign(
            source_name=artifact.source_name,
            description_file=artifact.description_path.name,
            human_description_file=human_description_file.name,
        )
        summary_frames.append(summary_df)

    combined_summary_df = pd.concat(summary_frames, ignore_index=True)
    summary_dir = ensure_directory(
        resolved_output_dir / "paper_summaries" / canonical_provider
    )
    summary_path = summary_dir / f"{resolved_scopus_file.stem}_human_alignment_summary.csv"
    combined_summary_df.to_csv(summary_path, index=False, encoding="utf-8")

    return HumanAlignmentRunResult(
        raw_scopus_file=resolved_scopus_file,
        human_description_file=human_description_file,
        summary_path=summary_path,
        summary_df=combined_summary_df,
        output_dir=resolved_output_dir,
    )


def run_all_evaluations(
    scopus_file: str | Path,
    provider: str,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 128,
    modularity_min_weight: float = 0.001,
    modularity_top_n_edges: int | None = None,
    modularity_min_degree: int = 1,
) -> EvaluationRunResult:
    canonical_provider = _canonical_provider(provider)
    resolved_scopus_file = resolve_raw_scopus_file(scopus_file)
    raw_scopus_df = load_scopus_csv(resolved_scopus_file)
    clustered_scopus_file = resolve_labeled_scopus_file(resolved_scopus_file.name)
    labeled_scopus_df = load_scopus_csv(clustered_scopus_file)
    human_description_file = require_human_description_file(resolved_scopus_file)

    louvain_silhouette_score = pd.NA
    louvain_silhouette_error = ""
    try:
        louvain_silhouette = compute_silhouette_for_clusters(
            scopus_df=labeled_scopus_df,
            dataset_name=clustered_scopus_file.stem,
            embedding_model=embedding_model,
            cluster_column="cluster",
            batch_size=batch_size,
            abstract_cache_name=resolved_scopus_file.stem,
        )
        louvain_silhouette_score = float(louvain_silhouette.iloc[0]["silhouette_score"])
    except Exception as error:
        louvain_silhouette_error = str(error)

    louvain_modularity_score = pd.NA
    louvain_modularity_error = ""
    try:
        louvain_modularity = compute_modularity_result(
            scopus_df=labeled_scopus_df,
            dataset_name=clustered_scopus_file.stem,
            cluster_column="cluster",
            min_weight=modularity_min_weight,
            top_n_edges=modularity_top_n_edges,
            min_degree=modularity_min_degree,
        )
        louvain_modularity_score = float(louvain_modularity.iloc[0]["modularity"])
    except Exception as error:
        louvain_modularity_error = str(error)

    description_artifacts = _expected_description_artifacts(
        scopus_file=resolved_scopus_file,
        provider=canonical_provider,
    )
    summac_model = None
    evidence_grounded_eligible_sources = {"pipeline_2", "pipeline_3", "pipeline_4", "pipeline_5", "pipeline_6"}

    summary_rows: list[dict[str, object]] = []
    for artifact in description_artifacts:
        coverage_score = pd.NA
        coverage_error = ""
        induced_output_path = QUALITY_DIR / "results" / f"{artifact.dataset_name}_induced_clusters.csv"
        coverage_output_path = COVERAGE_DIR / "results" / f"{artifact.dataset_name}_coverage.csv"
        induced_silhouette_path = (
            SILHOUETTE_DIR / f"{artifact.dataset_name}_induced_clusters_silhouette.csv"
        )
        induced_modularity_path = (
            MODULARITY_DIR / f"{artifact.dataset_name}_induced_clusters_modularity.csv"
        )
        reference_grounded_coverage_path = (
            REFERENCE_GROUNDED_COVERAGE_DIR / "results" / f"{artifact.dataset_name}_reference_grounded_coverage.csv"
        )
        reference_grounded_coverage_detail_path = (
            REFERENCE_GROUNDED_COVERAGE_DIR
            / "results"
            / f"{artifact.dataset_name}_reference_grounded_coverage_by_cluster.csv"
        )
        summac_detail_path = SUMMAC_DIR / "results" / f"{artifact.dataset_name}_summac_by_cluster.csv"
        similarity_output_name = f"{artifact.dataset_name}__vs__{clustered_scopus_file.stem}"
        similarity_output_path = SIMILARITY_DIR / f"{similarity_output_name}.csv"

        try:
            coverage_result = compute_coverage_of_corpus(
                description_df=artifact.description_df,
                scopus_df=raw_scopus_df,
                description_name=artifact.dataset_name,
                scopus_name=resolved_scopus_file.stem,
                embedding_model=embedding_model,
                batch_size=batch_size,
            )
            coverage_score = float(coverage_result.iloc[0]["average_cosine"])
        except Exception as error:
            coverage_error = str(error)

        induced_cluster_count = pd.NA
        induced_clustering_error = ""
        induced_silhouette_score = pd.NA
        induced_silhouette_error = ""
        induced_modularity_score = pd.NA
        induced_modularity_error = ""
        ari_score = pd.NA
        nmi_score = pd.NA
        similarity_error = ""
        reference_grounded_coverage_score = pd.NA
        reference_grounded_coverage_error = ""
        summac_score = pd.NA
        summac_error = ""

        try:
            induced_df = compute_quality_of_induced_clustering(
                description_df=artifact.description_df,
                scopus_df=raw_scopus_df,
                description_name=artifact.dataset_name,
                scopus_name=resolved_scopus_file.stem,
                embedding_model=embedding_model,
                batch_size=batch_size,
                cluster_output_column=INDUCED_CLUSTER_COLUMN,
            )
            induced_cluster_count = int(induced_df[INDUCED_CLUSTER_COLUMN].dropna().nunique())

            try:
                silhouette_result = compute_silhouette_for_clusters(
                    scopus_df=induced_df,
                    dataset_name=f"{artifact.dataset_name}_induced_clusters",
                    embedding_model=embedding_model,
                    cluster_column=INDUCED_CLUSTER_COLUMN,
                    batch_size=batch_size,
                    abstract_cache_name=resolved_scopus_file.stem,
                )
                induced_silhouette_score = float(
                    silhouette_result.iloc[0]["silhouette_score"]
                )
            except Exception as error:
                induced_silhouette_error = str(error)

            try:
                modularity_result = compute_modularity_result(
                    scopus_df=induced_df,
                    dataset_name=f"{artifact.dataset_name}_induced_clusters",
                    cluster_column=INDUCED_CLUSTER_COLUMN,
                    min_weight=modularity_min_weight,
                    top_n_edges=modularity_top_n_edges,
                    min_degree=modularity_min_degree,
                )
                induced_modularity_score = float(modularity_result.iloc[0]["modularity"])
            except Exception as error:
                induced_modularity_error = str(error)

            try:
                similarity_result = compare_clusterings(
                    left_df=labeled_scopus_df,
                    right_df=induced_df,
                    left_cluster_column="cluster",
                    right_cluster_column=INDUCED_CLUSTER_COLUMN,
                    output_name=similarity_output_name,
                )
                ari_score = float(similarity_result.iloc[0]["ari"])
                nmi_score = float(similarity_result.iloc[0]["nmi"])
            except Exception as error:
                similarity_error = str(error)
        except Exception as error:
            induced_clustering_error = str(error)

        if artifact.source_name in evidence_grounded_eligible_sources:
            try:
                reference_grounded_result = compute_reference_grounded_coverage(
                    description_df=artifact.description_df,
                    scopus_df=raw_scopus_df,
                    description_name=artifact.dataset_name,
                    embedding_model=embedding_model,
                    batch_size=batch_size,
                )
                reference_grounded_coverage_score = float(
                    reference_grounded_result.iloc[0]["reference_grounded_coverage"]
                )
            except Exception as error:
                reference_grounded_coverage_error = str(error)

        if artifact.source_name in evidence_grounded_eligible_sources:
            try:
                if summac_model is None:
                    summac_model = load_summac_model()
                summac_result = compute_summac_factual_consistency(
                    description_df=artifact.description_df,
                    scopus_df=raw_scopus_df,
                    description_name=artifact.dataset_name,
                    summac_model=summac_model,
                )
                summac_score = float(summac_result.iloc[0]["summac_factual_consistency"])
            except Exception as error:
                summac_error = str(error)

        summary_rows.append(
            {
                "source_name": artifact.source_name,
                "provider": canonical_provider,
                "scopus_file": _display_name(resolved_scopus_file),
                "clustered_scopus_file": _display_name(clustered_scopus_file),
                "description_file": _display_name(artifact.description_path),
                "description_dataset_name": artifact.dataset_name,
                "n_description_clusters": int(artifact.description_df["cluster_id"].nunique()),
                "coverage_average_cosine": coverage_score,
                "coverage_error": coverage_error,
                "coverage_result_file": _display_name(coverage_output_path),
                "induced_clustering_file": _display_name(induced_output_path),
                "induced_cluster_column": INDUCED_CLUSTER_COLUMN,
                "induced_clustering_error": induced_clustering_error,
                "n_induced_clusters": induced_cluster_count,
                "induced_silhouette_score": induced_silhouette_score,
                "induced_silhouette_error": induced_silhouette_error,
                "induced_silhouette_file": _display_name(induced_silhouette_path),
                "induced_modularity": induced_modularity_score,
                "induced_modularity_error": induced_modularity_error,
                "induced_modularity_file": _display_name(induced_modularity_path),
                "ari_vs_louvain": ari_score,
                "nmi_vs_louvain": nmi_score,
                "similarity_error": similarity_error,
                "similarity_file": _display_name(similarity_output_path),
                "reference_grounded_coverage": reference_grounded_coverage_score,
                "reference_grounded_coverage_error": reference_grounded_coverage_error,
                "reference_grounded_coverage_file": (
                    _display_name(reference_grounded_coverage_path)
                    if artifact.source_name in evidence_grounded_eligible_sources
                    else ""
                ),
                "reference_grounded_coverage_detail_file": (
                    _display_name(reference_grounded_coverage_detail_path)
                    if artifact.source_name in evidence_grounded_eligible_sources
                    else ""
                ),
                "summac_factual_consistency": summac_score,
                "summac_error": summac_error,
                "summac_detail_file": (
                    _display_name(summac_detail_path)
                    if artifact.source_name in evidence_grounded_eligible_sources
                    else ""
                ),
                "louvain_silhouette_score": louvain_silhouette_score,
                "louvain_silhouette_error": louvain_silhouette_error,
                "louvain_silhouette_file": _display_name(
                    SILHOUETTE_DIR / f"{clustered_scopus_file.stem}_silhouette.csv"
                ),
                "louvain_modularity": louvain_modularity_score,
                "louvain_modularity_error": louvain_modularity_error,
                "louvain_modularity_file": _display_name(
                    MODULARITY_DIR / f"{clustered_scopus_file.stem}_modularity.csv"
                ),
                "human_description_file": _display_name(human_description_file),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = _workflow_summary_path(
        kind="evaluations",
        provider=canonical_provider,
        scopus_file=resolved_scopus_file,
    )
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")

    return EvaluationRunResult(
        raw_scopus_file=resolved_scopus_file,
        clustered_scopus_file=clustered_scopus_file,
        human_description_file=human_description_file,
        summary_path=summary_path,
        summary_df=summary_df,
    )
