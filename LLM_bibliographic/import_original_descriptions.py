from __future__ import annotations

import argparse
from pathlib import Path

from llm_bibliometric.descriptions import import_original_descriptions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize human-authored descriptions into description/human_descriptions. "
            "Use --source-dir to import from a legacy folder such as 03_original_text."
        )
    )
    parser.add_argument("--source-dir", default=Path("description/human_descriptions"), type=Path)
    parser.add_argument("--destination-dir", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    written_paths = import_original_descriptions(
        source_dir=args.source_dir,
        destination_dir=args.destination_dir,
    )
    for path in written_paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
