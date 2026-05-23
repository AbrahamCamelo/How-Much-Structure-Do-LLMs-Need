from __future__ import annotations

import argparse
from pathlib import Path

from llm_bibliometric.constants import DEFAULT_EMBEDDING_MODEL
from llm_bibliometric.orchestration import run_reference_grounded_coverage_for_paper


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run reference-grounded coverage only for pipelines 2 to 6 for one paper and provider."
    )
    parser.add_argument("--scopus-file", required=True, type=Path)
    parser.add_argument("--provider", required=True, choices=["chatgpt", "gemini", "claude"])
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Optional output directory. Defaults to "
            "evaluations/reference_grounded_coverage/<provider>/<paper>/"
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run_reference_grounded_coverage_for_paper(
        scopus_file=args.scopus_file,
        provider=args.provider,
        embedding_model=args.embedding_model,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )
    print(result.summary_path)
    print(result.summary_df.to_csv(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
