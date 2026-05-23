from __future__ import annotations

import argparse
from pathlib import Path

from llm_bibliometric.cli_paths import resolve_labeled_scopus_file, resolve_output_file
from llm_bibliometric.descriptions import require_human_cluster_target
from llm_bibliometric.output_layout import description_output_dir
from llm_bibliometric.pipelines import run_pipeline_4
from llm_bibliometric.prompt_query import resolve_query_text
from llm_bibliometric.scopus import load_scopus_csv


def run_pipeline(
    query: str | None,
    prompt: str | None,
    scopus_file: str | Path,
    provider: str,
    model: str | None,
    output_file: str | Path,
    query_id: int | None = None,
    temperature: float = 0.2,
    prompts_file: str | Path | None = None,
    queries_file: str | Path | None = None,
):
    resolved_scopus_file = resolve_labeled_scopus_file(scopus_file)
    _, target_cluster_count = require_human_cluster_target(resolved_scopus_file)
    resolved_output_file = resolve_output_file(
        output_file,
        description_output_dir("pipeline_4", provider, prompts_file),
    )
    resolved_query = resolve_query_text(
        query=query,
        query_id=query_id,
        reference_path=resolved_scopus_file,
        queries_file=queries_file or "queries.xlsx",
    )
    labeled_scopus_df = load_scopus_csv(resolved_scopus_file)
    observed_cluster_count = int(labeled_scopus_df["cluster"].dropna().nunique())
    if observed_cluster_count != target_cluster_count:
        raise ValueError(
            f"The labeled Scopus file contains {observed_cluster_count} clusters, "
            f"but the human description requires {target_cluster_count}."
        )
    return run_pipeline_4(
        query=resolved_query,
        prompt=prompt,
        labeled_scopus_df=labeled_scopus_df,
        provider=provider,
        model=model,
        output_file=resolved_output_file,
        target_cluster_count=target_cluster_count,
        temperature=temperature,
        prompts_file=prompts_file,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pipeline 4: Bibliographic full-cluster description.")
    parser.add_argument("--query")
    parser.add_argument("--query-id", type=int)
    parser.add_argument("--prompt")
    parser.add_argument("--provider", required=True, choices=["chatgpt", "gemini", "claude"])
    parser.add_argument("--model")
    parser.add_argument("--scopus-file", required=True, type=Path)
    parser.add_argument("--output-file", required=True, type=Path)
    parser.add_argument("--prompts-file", type=Path, default=Path("prompts.csv"))
    parser.add_argument("--queries-file", type=Path, default=Path("queries.xlsx"))
    parser.add_argument("--temperature", type=float, default=0.2)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_pipeline(
        query=args.query,
        prompt=args.prompt,
        scopus_file=args.scopus_file,
        provider=args.provider,
        model=args.model,
        output_file=args.output_file,
        query_id=args.query_id,
        temperature=args.temperature,
        prompts_file=args.prompts_file,
        queries_file=args.queries_file,
    )
    print(
        resolve_output_file(
            args.output_file,
            description_output_dir("pipeline_4", args.provider, args.prompts_file),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
