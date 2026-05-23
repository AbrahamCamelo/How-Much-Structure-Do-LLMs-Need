from __future__ import annotations

import argparse
from pathlib import Path

from llm_bibliometric.cli_paths import resolve_labeled_scopus_file, resolve_output_file
from llm_bibliometric.constants import DESCRIPTION_DIR
from llm_bibliometric.descriptions import require_human_cluster_target
from llm_bibliometric.pipelines import run_pipeline_4, run_pipeline_5, run_pipeline_6
from llm_bibliometric.prompt_query import resolve_query_text
from llm_bibliometric.scopus import load_scopus_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate cluster descriptions from labeled Scopus data.")
    parser.add_argument("--strategy", choices=["full", "full_two_step", "topk"], required=True)
    parser.add_argument("--query")
    parser.add_argument("--query-id", type=int)
    parser.add_argument("--prompt")
    parser.add_argument("--provider", required=True, choices=["chatgpt", "gemini", "claude"])
    parser.add_argument("--model")
    parser.add_argument("--scopus-file", required=True, type=Path)
    parser.add_argument("--output-file", required=True, type=Path)
    parser.add_argument("--prompts-file", type=Path, default=Path("prompts.csv"))
    parser.add_argument("--queries-file", type=Path, default=Path("queries.xlsx"))
    parser.add_argument("--significant-papers-per-cluster", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=0.2)
    return parser


def build_cluster_descriptions(
    strategy: str,
    query: str | None,
    prompt: str | None,
    provider: str,
    model: str | None,
    scopus_file: str | Path,
    output_file: str | Path,
    query_id: int | None = None,
    prompts_file: str | Path | None = None,
    queries_file: str | Path | None = None,
    significant_papers_per_cluster: int = 12,
    temperature: float = 0.2,
):
    resolved_scopus_file = resolve_labeled_scopus_file(scopus_file)
    _, target_cluster_count = require_human_cluster_target(resolved_scopus_file)
    pipeline_folder = {
        "full": "pipeline_4",
        "full_two_step": "pipeline_5",
        "topk": "pipeline_6",
    }[strategy]
    resolved_output_file = resolve_output_file(
        output_file,
        DESCRIPTION_DIR / pipeline_folder / provider.lower(),
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
    if strategy == "full":
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
    if strategy == "full_two_step":
        return run_pipeline_5(
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
    return run_pipeline_6(
        query=resolved_query,
        prompt=prompt,
        labeled_scopus_df=labeled_scopus_df,
        provider=provider,
        model=model,
        output_file=resolved_output_file,
        top_k_clusters=target_cluster_count,
        target_cluster_count=target_cluster_count,
        significant_papers_per_cluster=significant_papers_per_cluster,
        temperature=temperature,
        prompts_file=prompts_file,
    )


def main() -> int:
    args = build_parser().parse_args()
    build_cluster_descriptions(
        strategy=args.strategy,
        query=args.query,
        prompt=args.prompt,
        provider=args.provider,
        model=args.model,
        scopus_file=args.scopus_file,
        output_file=args.output_file,
        query_id=args.query_id,
        prompts_file=args.prompts_file,
        queries_file=args.queries_file,
        significant_papers_per_cluster=args.significant_papers_per_cluster,
        temperature=args.temperature,
    )
    pipeline_folder = {
        "full": "pipeline_4",
        "full_two_step": "pipeline_5",
        "topk": "pipeline_6",
    }[args.strategy]
    print(resolve_output_file(args.output_file, DESCRIPTION_DIR / pipeline_folder / args.provider.lower()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
