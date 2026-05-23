from __future__ import annotations

import argparse
from pathlib import Path

from llm_bibliometric.orchestration import run_human_alignment_for_paper


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run BERTScore-based human alignment for pipelines 1 to 6 for one paper and provider."
    )
    parser.add_argument("--scopus-file", required=True, type=Path)
    parser.add_argument("--provider", required=True, choices=["chatgpt", "gemini", "claude"])
    parser.add_argument("--model-type", default="roberta-large")
    parser.add_argument("--lang", default="en")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--rescale-with-baseline", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluations/human_alignment"),
        help="Base directory where matches/ and summaries/ provider folders will be written.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run_human_alignment_for_paper(
        scopus_file=args.scopus_file,
        provider=args.provider,
        model_type=args.model_type,
        lang=args.lang,
        device=args.device,
        batch_size=args.batch_size,
        rescale_with_baseline=args.rescale_with_baseline,
        output_dir=args.output_dir,
    )
    print(result.summary_path)
    print(result.summary_df.to_csv(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
