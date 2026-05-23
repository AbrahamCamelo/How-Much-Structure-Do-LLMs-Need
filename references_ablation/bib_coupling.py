from __future__ import annotations

import argparse
from pathlib import Path

from llm_bibliometric.bib_coupling import run_bibliographic_coupling
from llm_bibliometric.cli_paths import (
    resolve_description_file,
    resolve_output_file,
    resolve_raw_scopus_file,
)
from llm_bibliometric.constants import CLUSTERS_DIR
from llm_bibliometric.descriptions import load_description_csv, require_human_cluster_target
from llm_bibliometric.scopus import load_scopus_csv
from llm_bibliometric.utils import ensure_directory


def _resolve_target_cluster_count(
    scopus_file: str | Path,
    target_clusters: int | None = None,
    human_description_file: str | Path | None = None,
    human_descriptions_dir: str | Path = Path("description/human_descriptions"),
) -> tuple[int | None, Path | None]:
    resolved_human_file: Path | None = None
    if human_description_file is not None:
        resolved_human_file = resolve_description_file(human_description_file)
    else:
        resolved_human_file, inferred_target_clusters = require_human_cluster_target(
            reference_path=scopus_file,
            human_descriptions_dir=human_descriptions_dir,
        )
        if target_clusters is None:
            return inferred_target_clusters, resolved_human_file

    if resolved_human_file is None or not resolved_human_file.exists():
        raise FileNotFoundError(
            f"A matching human description file is required for {scopus_file}."
        )

    human_descriptions = load_description_csv(resolved_human_file)
    target_cluster_count = int(human_descriptions["cluster_id"].nunique())
    if target_clusters is not None and int(target_clusters) != target_cluster_count:
        raise ValueError(
            f"--target-clusters={int(target_clusters)} does not match the human description "
            f"cluster count {target_cluster_count} in {resolved_human_file}."
        )
    return target_cluster_count, resolved_human_file


def run_pipeline(
    scopus_file: str | Path,
    output_file: str | Path | None = None,
    min_weight: float = 0.0001,
    top_n_edges: int | None = None,
    min_degree: int = 1,
    resolution: float = 1.0,
    min_cluster_size: int = 3,
    target_clusters: int | None = None,
    human_description_file: str | Path | None = None,
    human_descriptions_dir: str | Path = Path("description/human_descriptions"),
    resolution_step: float = 0.1,
    min_resolution: float = 0.05,
    max_resolution: float = 5.0,
    resolution_max_iterations: int = 50,
    resolution_min_interval: float = 0.001,
):
    resolved_scopus_file = resolve_raw_scopus_file(scopus_file)
    scopus_df = load_scopus_csv(resolved_scopus_file)
    resolved_target_clusters, _ = _resolve_target_cluster_count(
        scopus_file=resolved_scopus_file,
        target_clusters=target_clusters,
        human_description_file=human_description_file,
        human_descriptions_dir=human_descriptions_dir,
    )
    result = run_bibliographic_coupling(
        scopus_df=scopus_df,
        min_weight=min_weight,
        top_n_edges=top_n_edges,
        min_degree=min_degree,
        resolution=resolution,
        min_cluster_size=min_cluster_size,
        target_cluster_count=resolved_target_clusters,
        resolution_step=resolution_step,
        min_resolution=min_resolution,
        max_resolution=max_resolution,
        resolution_max_iterations=resolution_max_iterations,
        resolution_min_interval=resolution_min_interval,
    )
    if resolved_target_clusters is None:
        raise ValueError("A human description target cluster count is required.")
    if int(result.cluster_count) != int(resolved_target_clusters):
        raise ValueError(
            "The Louvain clustering did not match the required number of human clusters. "
            f"Expected {int(resolved_target_clusters)} but obtained {int(result.cluster_count)}."
        )
    if output_file is not None:
        output_path = resolve_output_file(output_file, CLUSTERS_DIR)
        ensure_directory(output_path.parent)
        result.labeled_scopus.to_csv(output_path, index=False, encoding="utf-8")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Louvain bibliographic coupling on a Scopus file.")
    parser.add_argument("--scopus-file", required=True, type=Path)
    parser.add_argument("--output-file", type=Path)
    parser.add_argument("--min-weight", type=float, default=0.0001)
    parser.add_argument("--top-n-edges", type=int)
    parser.add_argument("--min-degree", type=int, default=1)
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument("--min-cluster-size", type=int, default=3)
    parser.add_argument("--target-clusters", type=int)
    parser.add_argument("--human-description-file", type=Path)
    parser.add_argument(
        "--human-descriptions-dir",
        type=Path,
        default=Path("description/human_descriptions"),
    )
    parser.add_argument("--resolution-step", type=float, default=0.1)
    parser.add_argument("--min-resolution", type=float, default=0.05)
    parser.add_argument("--max-resolution", type=float, default=5.0)
    parser.add_argument("--resolution-max-iterations", type=int, default=50)
    parser.add_argument("--resolution-min-interval", type=float, default=0.001)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    resolved_scopus_file = resolve_raw_scopus_file(args.scopus_file)
    output_file = (
        resolve_output_file(args.output_file, CLUSTERS_DIR)
        if args.output_file
        else (CLUSTERS_DIR / f"{resolved_scopus_file.stem}_louvain.csv")
    )
    result = run_pipeline(
        scopus_file=resolved_scopus_file,
        output_file=output_file,
        min_weight=args.min_weight,
        top_n_edges=args.top_n_edges,
        min_degree=args.min_degree,
        resolution=args.resolution,
        min_cluster_size=args.min_cluster_size,
        target_clusters=args.target_clusters,
        human_description_file=args.human_description_file,
        human_descriptions_dir=args.human_descriptions_dir,
        resolution_step=args.resolution_step,
        min_resolution=args.min_resolution,
        max_resolution=args.max_resolution,
        resolution_max_iterations=args.resolution_max_iterations,
        resolution_min_interval=args.resolution_min_interval,
    )
    print(output_file)
    if result.target_cluster_count is not None:
        print(
            (
                f"target_clusters={result.target_cluster_count}, "
                f"observed_clusters={result.cluster_count}, "
                f"resolution_used={result.resolution:.6f}, "
                f"exact_match={result.matched_target_cluster_count}"
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
