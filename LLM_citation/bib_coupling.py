from __future__ import annotations

import argparse
from pathlib import Path

from llm_bibliometric.bib_coupling import run_bibliographic_coupling
from llm_bibliometric.cli_paths import (
    resolve_output_file,
    resolve_raw_scopus_file,
)
from llm_bibliometric.constants import (
    CLUSTERS_DIR,
    DEFAULT_MAX_LOUVAIN_CLUSTERS,
    DEFAULT_LOUVAIN_RESOLUTION,
)
from llm_bibliometric.scopus import load_scopus_csv
from llm_bibliometric.utils import ensure_directory


def run_pipeline(
    scopus_file: str | Path,
    output_file: str | Path | None = None,
    min_weight: float = 0.0001,
    top_n_edges: int | None = None,
    min_degree: int = 1,
    resolution: float = DEFAULT_LOUVAIN_RESOLUTION,
    min_cluster_size: int = 3,
    min_cluster_proportion: float | None = None,
    max_cluster_count: int | None = DEFAULT_MAX_LOUVAIN_CLUSTERS,
    resolution_step: float = 0.1,
    min_resolution: float = 0.05,
    max_resolution: float = 5.0,
    resolution_max_iterations: int = 50,
    resolution_min_interval: float = 0.001,
):
    resolved_scopus_file = resolve_raw_scopus_file(scopus_file)
    scopus_df = load_scopus_csv(resolved_scopus_file)
    result = run_bibliographic_coupling(
        scopus_df=scopus_df,
        min_weight=min_weight,
        top_n_edges=top_n_edges,
        min_degree=min_degree,
        resolution=resolution,
        min_cluster_size=min_cluster_size,
        min_cluster_proportion=min_cluster_proportion,
        target_cluster_count=None,
        max_cluster_count=max_cluster_count,
        resolution_step=resolution_step,
        min_resolution=min_resolution,
        max_resolution=max_resolution,
        resolution_max_iterations=resolution_max_iterations,
        resolution_min_interval=resolution_min_interval,
    )
    if output_file is not None:
        output_path = resolve_output_file(output_file, CLUSTERS_DIR)
        ensure_directory(output_path.parent)
        result.labeled_scopus.to_csv(output_path, index=False, encoding="utf-8")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Louvain direct citation clustering on a Scopus file.")
    parser.add_argument("--scopus-file", required=True, type=Path)
    parser.add_argument("--output-file", type=Path)
    parser.add_argument("--min-weight", type=float, default=0.0001)
    parser.add_argument("--top-n-edges", type=int)
    parser.add_argument("--min-degree", type=int, default=1)
    parser.add_argument("--resolution", type=float, default=DEFAULT_LOUVAIN_RESOLUTION)
    parser.add_argument("--min-cluster-size", type=int, default=3)
    parser.add_argument("--min-cluster-proportion", type=float)
    parser.add_argument("--max-clusters", type=int, default=DEFAULT_MAX_LOUVAIN_CLUSTERS)
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
        min_cluster_proportion=args.min_cluster_proportion,
        max_cluster_count=args.max_clusters,
        resolution_step=args.resolution_step,
        min_resolution=args.min_resolution,
        max_resolution=args.max_resolution,
        resolution_max_iterations=args.resolution_max_iterations,
        resolution_min_interval=args.resolution_min_interval,
    )
    print(output_file)
    print(
        f"observed_clusters={result.cluster_count}, resolution_used={result.resolution:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
