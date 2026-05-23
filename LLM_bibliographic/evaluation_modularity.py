from __future__ import annotations

import argparse
from pathlib import Path

from llm_bibliometric.cli_paths import resolve_clustering_file
from llm_bibliometric.evaluations import compute_modularity_result
from llm_bibliometric.scopus import load_scopus_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluation 5: Modularity.")
    parser.add_argument("--scopus-file", required=True, type=Path)
    parser.add_argument("--cluster-column", default="cluster")
    parser.add_argument("--min-weight", type=float, default=0.001)
    parser.add_argument("--top-n-edges", type=int)
    parser.add_argument("--min-degree", type=int, default=1)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    resolved_scopus_file = resolve_clustering_file(args.scopus_file)
    result = compute_modularity_result(
        scopus_df=load_scopus_csv(resolved_scopus_file),
        dataset_name=resolved_scopus_file.stem,
        cluster_column=args.cluster_column,
        min_weight=args.min_weight,
        top_n_edges=args.top_n_edges,
        min_degree=args.min_degree,
    )
    print(result.to_csv(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
