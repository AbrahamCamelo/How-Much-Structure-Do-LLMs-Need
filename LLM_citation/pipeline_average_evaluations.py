from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from llm_bibliometric.constants import (
    DEFAULT_SIGNIFICANT_PAPERS_PER_CLUSTER,
    full_run_provider_dir,
)
from llm_bibliometric.utils import ensure_directory


PROVIDER_ALIASES = {
    "openai": "chatgpt",
    "chatgpt": "chatgpt",
    "google": "gemini",
    "gemini": "gemini",
    "anthropic": "claude",
    "claude": "claude",
}

NUMERIC_METRICS = (
    "n_description_clusters",
    "coverage_average_cosine",
    "n_induced_clusters",
    "induced_silhouette_score",
    "induced_modularity",
    "ari_vs_louvain",
    "nmi_vs_louvain",
    "louvain_silhouette_score",
    "louvain_modularity",
)


@dataclass(frozen=True)
class PipelineAverageEvaluationsResult:
    input_dir: Path
    output_path: Path
    summary_df: pd.DataFrame


def _canonical_provider(provider: str) -> str:
    normalized = str(provider).strip().lower()
    if normalized not in PROVIDER_ALIASES:
        raise ValueError(f"Unsupported provider: {provider}")
    return PROVIDER_ALIASES[normalized]


def _default_input_dir(
    provider: str,
    significant_papers_per_cluster: int = DEFAULT_SIGNIFICANT_PAPERS_PER_CLUSTER,
) -> Path:
    return full_run_provider_dir(
        kind="evaluations",
        provider=provider,
        significant_papers_per_cluster=significant_papers_per_cluster,
    )


def _resolve_output_path(output_file: str | Path | None, default_directory: Path) -> Path:
    if output_file is None:
        return default_directory / "pipeline_average_evaluations.csv"
    path = Path(output_file)
    if path.is_absolute() or path.parent != Path("."):
        return path
    return default_directory / path.name


def _load_evaluation_frames(input_dir: Path) -> pd.DataFrame:
    files = sorted(path for path in input_dir.glob("*paper_evaluations.csv") if path.is_file())
    if not files:
        raise FileNotFoundError(f"No per-paper evaluation CSV files were found in {input_dir}.")

    frames: list[pd.DataFrame] = []
    for path in files:
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frame = frame.copy()
        frame["per_paper_summary_file"] = path.name
        frames.append(frame)

    if not frames:
        raise ValueError(f"All evaluation CSV files in {input_dir} were empty.")
    return pd.concat(frames, ignore_index=True)


def build_pipeline_average_evaluations(
    provider: str,
    input_dir: str | Path | None = None,
    output_file: str | Path | None = None,
    significant_papers_per_cluster: int = DEFAULT_SIGNIFICANT_PAPERS_PER_CLUSTER,
) -> PipelineAverageEvaluationsResult:
    canonical_provider = _canonical_provider(provider)
    resolved_input_dir = (
        Path(input_dir)
        if input_dir is not None
        else _default_input_dir(
            canonical_provider,
            significant_papers_per_cluster=significant_papers_per_cluster,
        )
    )
    combined_df = _load_evaluation_frames(resolved_input_dir)

    if "source_name" not in combined_df.columns:
        raise ValueError("Evaluation summaries must contain a 'source_name' column.")

    work_df = combined_df[combined_df["source_name"].astype(str).str.startswith("pipeline_")].copy()
    if work_df.empty:
        raise ValueError("No pipeline rows were found in the evaluation summaries.")

    available_metrics = [column for column in NUMERIC_METRICS if column in work_df.columns]
    if not available_metrics:
        raise ValueError("No expected numeric evaluation metrics were found to aggregate.")

    for column in available_metrics:
        work_df[column] = pd.to_numeric(work_df[column], errors="coerce")

    summary_rows: list[dict[str, object]] = []
    for source_name, group_df in work_df.groupby("source_name", sort=True):
        row: dict[str, object] = {
            "source_name": source_name,
            "provider": canonical_provider,
            "n_papers": int(group_df["scopus_file"].nunique()) if "scopus_file" in group_df.columns else int(len(group_df)),
            "n_rows": int(len(group_df)),
        }
        for column in available_metrics:
            row[f"{column}_mean"] = float(group_df[column].mean()) if group_df[column].notna().any() else pd.NA
            row[f"{column}_count"] = int(group_df[column].notna().sum())
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values("source_name").reset_index(drop=True)
    output_path = _resolve_output_path(output_file, resolved_input_dir)
    ensure_directory(output_path.parent)
    summary_df.to_csv(output_path, index=False, encoding="utf-8")

    return PipelineAverageEvaluationsResult(
        input_dir=resolved_input_dir,
        output_path=output_path,
        summary_df=summary_df,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate per-paper evaluation summaries into average metrics per pipeline."
    )
    parser.add_argument("--provider", required=True, choices=["chatgpt", "gemini", "claude"])
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-file", type=Path)
    parser.add_argument("--significant-papers-per-cluster", type=int, default=DEFAULT_SIGNIFICANT_PAPERS_PER_CLUSTER)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = build_pipeline_average_evaluations(
        provider=args.provider,
        input_dir=args.input_dir,
        output_file=args.output_file,
        significant_papers_per_cluster=args.significant_papers_per_cluster,
    )
    print(result.output_path)
    print(result.summary_df.to_csv(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
