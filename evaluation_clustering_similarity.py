from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from llm_bibliometric.cli_paths import resolve_clustering_file
from llm_bibliometric.constants import DEFAULT_SIGNIFICANT_PAPERS_PER_CLUSTER
from llm_bibliometric.evaluations import compare_clusterings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluation 4: ARI and NMI between two clusterings.")
    parser.add_argument("--left-file", required=True, type=Path)
    parser.add_argument("--right-file", required=True, type=Path)
    parser.add_argument("--left-cluster-column", default="cluster")
    parser.add_argument("--right-cluster-column", default="cluster")
    parser.add_argument("--output-name")
    parser.add_argument("--significant-papers-per-cluster", type=int, default=DEFAULT_SIGNIFICANT_PAPERS_PER_CLUSTER)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    resolved_left_file = resolve_clustering_file(args.left_file)
    resolved_right_file = resolve_clustering_file(args.right_file)
    result = compare_clusterings(
        left_df=pd.read_csv(resolved_left_file),
        right_df=pd.read_csv(resolved_right_file),
        left_cluster_column=args.left_cluster_column,
        right_cluster_column=args.right_cluster_column,
        output_name=args.output_name,
        significant_papers_per_cluster=args.significant_papers_per_cluster,
    )
    print(result.to_csv(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
