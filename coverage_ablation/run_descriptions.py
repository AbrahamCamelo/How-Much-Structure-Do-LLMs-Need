from __future__ import annotations

import argparse
from pathlib import Path

from llm_bibliometric.orchestration import (
    run_all_descriptions,
    run_all_descriptions_for_all_scopus,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run bibliographic coupling and generate description outputs for pipelines 1 to 6."
    )
    parser.add_argument("--scopus-file", type=Path)
    parser.add_argument("--provider", required=True, choices=["chatgpt", "gemini", "claude"])
    parser.add_argument("--model")
    parser.add_argument("--prompts-file", type=Path, default=Path("prompts.csv"))
    parser.add_argument("--queries-file", type=Path, default=Path("queries.xlsx"))
    parser.add_argument(
        "--only-missing",
        "--skip-existing",
        action="store_true",
        dest="only_missing",
        help=(
            "Reuse existing clustered and pipeline outputs when valid, and generate only "
            "missing or stale artifacts."
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--significant-papers-per-cluster", type=int, default=12)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.scopus_file is None:
        result = run_all_descriptions_for_all_scopus(
            provider=args.provider,
            model=args.model,
            prompts_file=args.prompts_file,
            queries_file=args.queries_file,
            only_missing=args.only_missing,
            temperature=args.temperature,
            significant_papers_per_cluster=args.significant_papers_per_cluster,
        )
    else:
        result = run_all_descriptions(
            scopus_file=args.scopus_file,
            provider=args.provider,
            model=args.model,
            prompts_file=args.prompts_file,
            queries_file=args.queries_file,
            only_missing=args.only_missing,
            temperature=args.temperature,
            significant_papers_per_cluster=args.significant_papers_per_cluster,
        )
    print(result.summary_path)
    print(result.summary_df.to_csv(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
