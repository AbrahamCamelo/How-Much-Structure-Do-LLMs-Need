from __future__ import annotations

import argparse
from pathlib import Path

from llm_bibliometric.cli_paths import resolve_description_file, resolve_raw_scopus_file
from llm_bibliometric.evaluations import compute_reference_grounded_coverage
from llm_bibliometric.scopus import load_scopus_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluation: reference-grounded coverage with cluster-level cited Scopus evidence."
    )
    parser.add_argument("--description-file", required=True, type=Path)
    parser.add_argument("--scopus-file", required=True, type=Path)
    parser.add_argument("--embedding-model", default="text-embedding-3-large")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluations/reference_grounded_coverage"),
        help=(
            "Directory where the reference-grounded coverage summary, embeddings cache, "
            "and per-cluster detail files will be written."
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    resolved_description_file = resolve_description_file(args.description_file)
    resolved_scopus_file = resolve_raw_scopus_file(args.scopus_file)
    result = compute_reference_grounded_coverage(
        description_df=resolved_description_file,
        scopus_df=load_scopus_csv(resolved_scopus_file),
        description_name=resolved_description_file.stem,
        embedding_model=args.embedding_model,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )
    print(result.to_csv(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
