from __future__ import annotations

import argparse
from pathlib import Path

from llm_bibliometric.constants import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_SIGNIFICANT_PAPERS_PER_CLUSTER,
)
from llm_bibliometric.orchestration import run_all_evaluations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the evaluation suite over all available descriptions for one paper and provider."
    )
    parser.add_argument("--scopus-file", required=True, type=Path)
    parser.add_argument("--provider", required=True, choices=["chatgpt", "gemini", "claude"])
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--significant-papers-per-cluster", type=int, default=DEFAULT_SIGNIFICANT_PAPERS_PER_CLUSTER)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run_all_evaluations(
        scopus_file=args.scopus_file,
        provider=args.provider,
        embedding_model=args.embedding_model,
        batch_size=args.batch_size,
        significant_papers_per_cluster=args.significant_papers_per_cluster,
    )
    print(result.summary_path)
    print(result.summary_df.to_csv(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
