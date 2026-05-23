from __future__ import annotations

import argparse
from pathlib import Path

from llm_bibliometric.cli_paths import resolve_description_file, resolve_raw_scopus_file
from llm_bibliometric.descriptions import load_description_csv, require_human_description_file
from llm_bibliometric.evaluations import compute_human_alignment_bertscore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare one generated description file against the matching human descriptions using BERTScore alignment."
    )
    parser.add_argument("--description-file", required=True, type=Path)
    parser.add_argument("--scopus-file", required=True, type=Path)
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
    parser.add_argument("--provider", default="manual")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    resolved_description_file = resolve_description_file(args.description_file)
    resolved_scopus_file = resolve_raw_scopus_file(args.scopus_file)
    human_description_file = require_human_description_file(resolved_scopus_file)

    summary_df, _ = compute_human_alignment_bertscore(
        generated_df=load_description_csv(resolved_description_file),
        human_df=load_description_csv(human_description_file),
        generated_name=resolved_description_file.stem,
        provider=args.provider,
        scopus_name=resolved_scopus_file.stem,
        model_type=args.model_type,
        lang=args.lang,
        device=args.device,
        batch_size=args.batch_size,
        rescale_with_baseline=args.rescale_with_baseline,
        output_dir=args.output_dir,
    )
    print(summary_df.to_csv(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
