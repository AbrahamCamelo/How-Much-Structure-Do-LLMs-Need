from __future__ import annotations

import argparse
from pathlib import Path

from llm_bibliometric.orchestration import run_summac_for_paper


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run SummaC only for pipelines 2 to 6 for one paper and provider."
    )
    parser.add_argument("--scopus-file", required=True, type=Path)
    parser.add_argument("--provider", required=True, choices=["chatgpt", "gemini", "claude"])
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Optional output directory. Defaults to "
            "evaluations/summac/<provider>/<paper>/"
        ),
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        help="Override device selection. By default, CUDA is used when available, otherwise CPU.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run_summac_for_paper(
        scopus_file=args.scopus_file,
        provider=args.provider,
        output_dir=args.output_dir,
        device=args.device,
    )
    print(result.summary_path)
    print(result.summary_df.to_csv(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
