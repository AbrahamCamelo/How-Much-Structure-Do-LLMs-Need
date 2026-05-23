from __future__ import annotations

import argparse
from pathlib import Path

from llm_bibliometric.cli_paths import resolve_clustering_file
from llm_bibliometric.evaluations import compute_silhouette_for_clusters
from llm_bibliometric.scopus import load_scopus_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluation 3: Silhouette score.")
    parser.add_argument("--scopus-file", required=True, type=Path)
    parser.add_argument("--embedding-model", default="text-embedding-3-large")
    parser.add_argument("--cluster-column", default="cluster")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--abstract-cache-name")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    resolved_scopus_file = resolve_clustering_file(args.scopus_file)
    result = compute_silhouette_for_clusters(
        scopus_df=load_scopus_csv(resolved_scopus_file),
        dataset_name=resolved_scopus_file.stem,
        embedding_model=args.embedding_model,
        cluster_column=args.cluster_column,
        batch_size=args.batch_size,
        abstract_cache_name=args.abstract_cache_name,
    )
    print(result.to_csv(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
