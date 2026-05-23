from __future__ import annotations

import argparse
from pathlib import Path

from llm_bibliometric.cli_paths import resolve_description_file, resolve_raw_scopus_file
from llm_bibliometric.evaluations import compute_summac_factual_consistency
from llm_bibliometric.scopus import load_scopus_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluation: evidence-grounded factual consistency with SummaC."
    )
    parser.add_argument("--description-file", required=True, type=Path)
    parser.add_argument("--scopus-file", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluations/summac/results"),
        help="Directory where the SummaC summary and per-cluster detail files will be written.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        help="Override device selection. By default, CUDA is used when available, otherwise CPU.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    resolved_description_file = resolve_description_file(args.description_file)
    resolved_scopus_file = resolve_raw_scopus_file(args.scopus_file)
    result = compute_summac_factual_consistency(
        description_df=resolved_description_file,
        scopus_df=load_scopus_csv(resolved_scopus_file),
        description_name=resolved_description_file.stem,
        device=args.device,
        output_dir=args.output_dir,
    )
    print(result.to_csv(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
