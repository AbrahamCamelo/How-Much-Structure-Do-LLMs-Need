from __future__ import annotations

import argparse
from pathlib import Path

from llm_bibliometric.bib_coupling import run_bibliographic_coupling
from llm_bibliometric.cli_paths import resolve_output_file
from llm_bibliometric.cli_paths import resolve_raw_scopus_file
from llm_bibliometric.constants import (
    DEFAULT_MAX_LOUVAIN_CLUSTERS,
    DEFAULT_LOUVAIN_RESOLUTION,
    DEFAULT_SIGNIFICANT_PAPERS_PER_CLUSTER,
    description_pipeline_dir,
)
from llm_bibliometric.pipelines import run_pipeline_1
from llm_bibliometric.prompt_query import resolve_query_text
from llm_bibliometric.scopus import load_scopus_csv


def run_pipeline(
    query: str | None,
    prompt: str | None,
    provider: str,
    model: str | None,
    output_file: str | Path,
    query_id: int | None = None,
    significant_papers_per_cluster: int = DEFAULT_SIGNIFICANT_PAPERS_PER_CLUSTER,
    resolution: float = DEFAULT_LOUVAIN_RESOLUTION,
    min_cluster_proportion: float | None = None,
    max_cluster_count: int | None = DEFAULT_MAX_LOUVAIN_CLUSTERS,
    temperature: float = 0.2,
    prompts_file: str | Path | None = None,
    queries_file: str | Path | None = None,
):
    resolved_output_file = resolve_output_file(
        output_file,
        description_pipeline_dir(
            pipeline_name="pipeline_1",
            provider=provider,
            significant_papers_per_cluster=significant_papers_per_cluster,
        ),
    )
    resolved_scopus_file = resolve_raw_scopus_file(resolved_output_file.name)
    cluster_result = run_bibliographic_coupling(
        scopus_df=load_scopus_csv(resolved_scopus_file),
        resolution=resolution,
        min_cluster_proportion=min_cluster_proportion,
        target_cluster_count=None,
        max_cluster_count=max_cluster_count,
    )
    target_cluster_count = int(cluster_result.cluster_count)
    resolved_query = resolve_query_text(
        query=query,
        query_id=query_id,
        reference_path=resolved_scopus_file,
        queries_file=queries_file or "queries.xlsx",
    )
    return run_pipeline_1(
        query=resolved_query,
        prompt=prompt,
        provider=provider,
        model=model,
        output_file=resolved_output_file,
        target_cluster_count=target_cluster_count,
        significant_papers_per_cluster=significant_papers_per_cluster,
        temperature=temperature,
        prompts_file=prompts_file,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pipeline 1: Direct one-step generation.")
    parser.add_argument("--query")
    parser.add_argument("--query-id", type=int)
    parser.add_argument("--prompt")
    parser.add_argument("--provider", required=True, choices=["chatgpt", "gemini", "claude"])
    parser.add_argument("--model")
    parser.add_argument("--output-file", required=True, type=Path)
    parser.add_argument("--prompts-file", type=Path, default=Path("prompts.csv"))
    parser.add_argument("--queries-file", type=Path, default=Path("queries.xlsx"))
    parser.add_argument("--significant-papers-per-cluster", type=int, default=DEFAULT_SIGNIFICANT_PAPERS_PER_CLUSTER)
    parser.add_argument("--resolution", type=float, default=DEFAULT_LOUVAIN_RESOLUTION)
    parser.add_argument("--min-cluster-proportion", type=float)
    parser.add_argument("--max-clusters", type=int, default=DEFAULT_MAX_LOUVAIN_CLUSTERS)
    parser.add_argument("--temperature", type=float, default=0.2)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_pipeline(
        query=args.query,
        prompt=args.prompt,
        provider=args.provider,
        model=args.model,
        output_file=args.output_file,
        query_id=args.query_id,
        significant_papers_per_cluster=args.significant_papers_per_cluster,
        resolution=args.resolution,
        min_cluster_proportion=args.min_cluster_proportion,
        max_cluster_count=args.max_clusters,
        temperature=args.temperature,
        prompts_file=args.prompts_file,
        queries_file=args.queries_file,
    )
    resolved_output_file = resolve_output_file(
        args.output_file,
        description_pipeline_dir(
            pipeline_name="pipeline_1",
            provider=args.provider,
            significant_papers_per_cluster=args.significant_papers_per_cluster,
        ),
    )
    print(resolved_output_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
