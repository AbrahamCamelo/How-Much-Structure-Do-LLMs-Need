from __future__ import annotations

import pandas as pd

from .utils import clean_text, format_reference_token


def format_paper_context(
    paper_row: pd.Series,
    include_cluster: bool = False,
    include_link_strength: bool = False,
    max_abstract_chars: int = 1400,
) -> str:
    lines = [
        f"paper_id: {format_reference_token(paper_row['paper_id'])}",
        f"title: {clean_text(paper_row.get('paper_title', ''))}",
    ]

    authors = clean_text(paper_row.get("authors", ""))
    if authors:
        lines.append(f"authors: {authors}")

    year = paper_row.get("year")
    if pd.notna(year):
        lines.append(f"year: {int(year)}")

    cited_by = paper_row.get("cited_by")
    if pd.notna(cited_by):
        lines.append(f"cited_by: {int(cited_by)}")

    if include_cluster:
        cluster = paper_row.get("cluster")
        if pd.notna(cluster):
            lines.append(f"cluster: {int(cluster)}")

    if include_link_strength:
        link_strength = paper_row.get("link_strength")
        if pd.notna(link_strength):
            lines.append(f"link_strength: {float(link_strength):.6f}")

    abstract = clean_text(paper_row.get("paper_abstract", ""))
    if len(abstract) > max_abstract_chars:
        abstract = abstract[:max_abstract_chars].rstrip() + "..."
    lines.append(f"abstract: {abstract}")
    return "\n".join(lines)


def build_cluster_context(
    cluster_id: int,
    cluster_df: pd.DataFrame,
    include_link_strength: bool = False,
    max_papers: int | None = None,
) -> str:
    ordered = cluster_df.copy()
    if "link_strength" in ordered.columns:
        ordered = ordered.sort_values(
            ["link_strength", "cited_by", "paper_id"],
            ascending=[False, False, True],
        )
    else:
        ordered = ordered.sort_values(["paper_id"], ascending=[True])

    if max_papers is not None:
        ordered = ordered.head(max_papers).copy()

    paper_blocks = "\n\n".join(
        format_paper_context(
            row,
            include_cluster=True,
            include_link_strength=include_link_strength,
        )
        for _, row in ordered.iterrows()
    )
    return (
        f"cluster_id: {cluster_id}\n"
        f"cluster_size: {len(cluster_df)}\n"
        f"papers:\n{paper_blocks}"
    )


def build_labeled_scopus_context(
    labeled_scopus_df: pd.DataFrame,
    include_link_strength: bool = False,
) -> str:
    labeled = labeled_scopus_df.dropna(subset=["cluster"]).copy()
    labeled["cluster"] = pd.to_numeric(labeled["cluster"], errors="raise").astype(int)
    cluster_blocks = [
        build_cluster_context(
            cluster_id=int(cluster_id),
            cluster_df=cluster_df.copy(),
            include_link_strength=include_link_strength,
        )
        for cluster_id, cluster_df in labeled.groupby("cluster", sort=True)
    ]
    return (
        f"cluster_count: {labeled['cluster'].nunique()}\n"
        f"paper_count: {len(labeled)}\n\n"
        + "\n\n".join(cluster_blocks)
    )
