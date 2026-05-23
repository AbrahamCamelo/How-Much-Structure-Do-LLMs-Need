from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .bib_coupling import select_top_clusters_by_link_strength
from .constants import (
    DEFAULT_SIGNIFICANT_PAPERS_PER_CLUSTER,
    DEFAULT_TOP_K_CLUSTERS,
    description_pipeline_dir,
)
from .description_utils import build_cluster_context, build_labeled_scopus_context
from .llm_clients import LLMClient, LLMConfig
from .prompt_query import render_prompt_template, resolve_pipeline_prompt
from .utils import (
    clean_text,
    ensure_directory,
    normalize_scopus_reference_list,
    parse_references,
    serialize_references,
    )

def _system_prompt(scopus_mode: bool) -> str:
    if scopus_mode:
        return (
            "You are an expert bibliometric analyst. Use only the supplied Scopus records. "
            "Do not use the internet, hidden browsing, or outside knowledge. "
            "Return JSON only."
        )
    return (
        "You are an expert bibliometric analyst. Use your existing knowledge only. "
        "Do not browse or imply internet access. Return JSON only."
    )


def _build_scopus_context(scopus_df: pd.DataFrame) -> str:
    ordered = scopus_df.copy()
    sort_columns: list[str] = []
    ascending: list[bool] = []
    if "cited_by" in ordered.columns:
        sort_columns.append("cited_by")
        ascending.append(False)
    sort_columns.append("paper_id")
    ascending.append(True)
    ordered = ordered.sort_values(sort_columns, ascending=ascending)
    paper_blocks = "\n\n".join(
        build_cluster_context(
            cluster_id=1,
            cluster_df=ordered,
            include_link_strength=False,
        ).split("papers:\n", maxsplit=1)[1:]
    )
    return (
        f"corpus_size: {len(ordered)}\n"
        f"papers:\n{paper_blocks}"
    )


def _reference_tokens_to_ids(references_value: str) -> list[int]:
    references = parse_references(references_value)
    paper_ids: list[int] = []
    for reference in references:
        stripped = reference.strip("[]")
        if stripped.isdigit():
            paper_ids.append(int(stripped))
    return paper_ids


def _build_selected_cluster_contexts(
    references_df: pd.DataFrame,
    scopus_df: pd.DataFrame,
    include_link_strength: bool = False,
) -> str:
    cluster_context_blocks: list[str] = []
    for _, row in references_df.iterrows():
        cluster_id = int(row["cluster_id"])
        reference_ids = _reference_tokens_to_ids(row["references"])
        relevant_df = scopus_df[scopus_df["paper_id"].isin(reference_ids)].copy()
        cluster_context_blocks.append(
            build_cluster_context(
                cluster_id=cluster_id,
                cluster_df=relevant_df,
                include_link_strength=include_link_strength,
            )
        )
    return "\n\n".join(cluster_context_blocks)


def _extract_records(payload: object) -> list[dict[str, object]]:
    if isinstance(payload, list):
        return [dict(item) for item in payload]
    if isinstance(payload, dict):
        for key in ("clusters", "results", "data", "items"):
            if isinstance(payload.get(key), list):
                return [dict(item) for item in payload[key]]
        return [dict(payload)]
    raise ValueError("LLM response could not be normalized into records.")


def _normalize_expected_cluster_output(
    output_df: pd.DataFrame,
    pipeline_name: str,
    target_cluster_count: int | None = None,
) -> pd.DataFrame:
    normalized = output_df.copy()
    if normalized.empty:
        raise ValueError(f"{pipeline_name} returned no cluster records.")

    normalized["cluster_id"] = pd.to_numeric(normalized["cluster_id"], errors="raise").astype(int)
    if normalized["cluster_id"].duplicated().any():
        raise ValueError(f"{pipeline_name} returned duplicate cluster_id values.")

    if target_cluster_count is None:
        return normalized.sort_values("cluster_id").reset_index(drop=True)

    actual_cluster_count = int(normalized["cluster_id"].nunique())
    if actual_cluster_count != int(target_cluster_count):
        raise ValueError(
            f"{pipeline_name} returned {actual_cluster_count} clusters, "
            f"but the human description requires {int(target_cluster_count)} clusters."
        )

    sorted_ids = sorted(normalized["cluster_id"].tolist())
    expected_ids = list(range(1, int(target_cluster_count) + 1))
    if sorted_ids != expected_ids:
        remap = {cluster_id: index for index, cluster_id in enumerate(sorted_ids, start=1)}
        normalized["cluster_id"] = normalized["cluster_id"].map(remap).astype(int)

    return normalized.sort_values("cluster_id").reset_index(drop=True)


def _standardize_output_records(
    payload: object,
    scopus_mode: bool,
    allowed_paper_ids: set[int] | None = None,
) -> pd.DataFrame:
    rows = []
    for index, record in enumerate(_extract_records(payload), start=1):
        normalized_keys = {str(key).strip().lower(): value for key, value in record.items()}
        cluster_id = normalized_keys.get("cluster_id") or normalized_keys.get("cluster") or index
        description = clean_text(normalized_keys.get("description", ""))
        references = parse_references(normalized_keys.get("references"))

        if scopus_mode:
            references = normalize_scopus_reference_list(
                references=references,
                allowed_paper_ids=allowed_paper_ids,
            )

        rows.append(
            {
                "cluster_id": int(cluster_id),
                "description": description,
                "references": serialize_references(references),
            }
        )

    output_df = pd.DataFrame(rows)
    output_df = output_df.sort_values("cluster_id").reset_index(drop=True)
    return output_df


def _save_pipeline_output(
    output_df: pd.DataFrame,
    pipeline_name: str,
    provider: str,
    output_file: str | Path | None,
    significant_papers_per_cluster: int = DEFAULT_SIGNIFICANT_PAPERS_PER_CLUSTER,
) -> pd.DataFrame:
    if output_file is None:
        raise ValueError("output_file must be provided so the pipeline result is persisted.")
    output_path = Path(output_file)
    ensure_directory(output_path.parent)
    output_df.to_csv(output_path, index=False, encoding="utf-8")

    pipeline_dir = ensure_directory(
        description_pipeline_dir(
            pipeline_name=pipeline_name,
            provider=provider,
            significant_papers_per_cluster=significant_papers_per_cluster,
        )
    )
    pipeline_output_path = pipeline_dir / output_path.name
    if pipeline_output_path != output_path:
        output_df.to_csv(pipeline_output_path, index=False, encoding="utf-8")
    return output_df


def run_pipeline_1(
    query: str,
    prompt: str | None,
    provider: str,
    model: str | None,
    output_file: str | Path,
    target_cluster_count: int | None = None,
    significant_papers_per_cluster: int = DEFAULT_SIGNIFICANT_PAPERS_PER_CLUSTER,
    temperature: float = 0.2,
    prompts_file: str | Path | None = None,
) -> pd.DataFrame:
    client = LLMClient(LLMConfig(provider=provider, model=model, temperature=temperature))
    client.set_request_context(
        paper=Path(output_file).stem,
        pipeline="pipeline_1",
        step="step_1",
    )
    prompt_template = resolve_pipeline_prompt(
        pipeline_name="pipeline1",
        step=1,
        override=prompt,
        prompts_file=prompts_file,
    )
    user_prompt = render_prompt_template(
        prompt_template,
        query=query,
        target_cluster_count=target_cluster_count,
        significant_papers_per_cluster=significant_papers_per_cluster,
    )
    payload = client.generate_json(_system_prompt(scopus_mode=False), user_prompt)
    output_df = _standardize_output_records(payload, scopus_mode=False)
    output_df = _normalize_expected_cluster_output(
        output_df=output_df,
        pipeline_name="pipeline_1",
        target_cluster_count=target_cluster_count,
    )
    return _save_pipeline_output(
        output_df,
        "pipeline_1",
        provider,
        output_file,
        significant_papers_per_cluster=significant_papers_per_cluster,
    )


def run_pipeline_2(
    query: str,
    prompt: str | None,
    scopus_df: pd.DataFrame,
    provider: str,
    model: str | None,
    output_file: str | Path,
    target_cluster_count: int | None = None,
    significant_papers_per_cluster: int = DEFAULT_SIGNIFICANT_PAPERS_PER_CLUSTER,
    temperature: float = 0.2,
    prompts_file: str | Path | None = None,
) -> pd.DataFrame:
    client = LLMClient(LLMConfig(provider=provider, model=model, temperature=temperature))
    client.set_request_context(
        paper=Path(output_file).stem,
        pipeline="pipeline_2",
        step="step_1",
    )
    scopus_context = _build_scopus_context(scopus_df)
    prompt_template = resolve_pipeline_prompt(
        pipeline_name="pipeline2",
        step=1,
        override=prompt,
        prompts_file=prompts_file,
    )
    user_prompt = render_prompt_template(
        prompt_template,
        query=query,
        scopus_context=scopus_context,
        target_cluster_count=target_cluster_count,
        significant_papers_per_cluster=significant_papers_per_cluster,
    )
    payload = client.generate_json(_system_prompt(scopus_mode=True), user_prompt)
    output_df = _standardize_output_records(
        payload=payload,
        scopus_mode=True,
        allowed_paper_ids=set(scopus_df["paper_id"].astype(int).tolist()),
    )
    output_df = _normalize_expected_cluster_output(
        output_df=output_df,
        pipeline_name="pipeline_2",
        target_cluster_count=target_cluster_count,
    )
    return _save_pipeline_output(
        output_df,
        "pipeline_2",
        provider,
        output_file,
        significant_papers_per_cluster=significant_papers_per_cluster,
    )


def run_pipeline_3(
    query: str,
    prompt: str | None,
    scopus_df: pd.DataFrame,
    provider: str,
    model: str | None,
    output_file: str | Path,
    target_cluster_count: int | None = None,
    significant_papers_per_cluster: int = DEFAULT_SIGNIFICANT_PAPERS_PER_CLUSTER,
    temperature: float = 0.2,
    prompts_file: str | Path | None = None,
) -> pd.DataFrame:
    client = LLMClient(LLMConfig(provider=provider, model=model, temperature=temperature))
    scopus_context = _build_scopus_context(scopus_df)
    paper = Path(output_file).stem

    step_one_template = resolve_pipeline_prompt(
        pipeline_name="pipeline3",
        step=1,
        prompts_file=prompts_file,
    )
    step_one_prompt = render_prompt_template(
        step_one_template,
        query=query,
        scopus_context=scopus_context,
        target_cluster_count=target_cluster_count,
        significant_papers_per_cluster=significant_papers_per_cluster,
    )
    client.set_request_context(
        paper=paper,
        pipeline="pipeline_3",
        step="step_1",
    )
    references_payload = client.generate_json(_system_prompt(scopus_mode=True), step_one_prompt)
    references_df = _standardize_output_records(
        payload=references_payload,
        scopus_mode=True,
        allowed_paper_ids=set(scopus_df["paper_id"].astype(int).tolist()),
    )
    references_df = _normalize_expected_cluster_output(
        output_df=references_df,
        pipeline_name="pipeline_3 step 1",
        target_cluster_count=target_cluster_count,
    )

    clusters_text = _build_selected_cluster_contexts(
        references_df=references_df,
        scopus_df=scopus_df,
        include_link_strength=False,
    )
    step_two_template = resolve_pipeline_prompt(
        pipeline_name="pipeline3",
        step=2,
        override=prompt,
        prompts_file=prompts_file,
    )
    step_two_prompt = render_prompt_template(
        step_two_template,
        query=query,
        cluster_contexts=clusters_text,
        target_cluster_count=target_cluster_count,
    )
    client.set_request_context(
        paper=paper,
        pipeline="pipeline_3",
        step="step_2",
    )
    descriptions_payload = client.generate_json(_system_prompt(scopus_mode=True), step_two_prompt)
    descriptions_records = []
    for record in _extract_records(descriptions_payload):
        normalized_keys = {str(key).strip().lower(): value for key, value in record.items()}
        descriptions_records.append(
            {
                "cluster_id": int(normalized_keys.get("cluster_id") or normalized_keys.get("cluster")),
                "description": clean_text(normalized_keys.get("description", "")),
            }
        )
    descriptions_df = pd.DataFrame(descriptions_records).sort_values("cluster_id")
    descriptions_df = _normalize_expected_cluster_output(
        output_df=descriptions_df,
        pipeline_name="pipeline_3 step 2",
        target_cluster_count=target_cluster_count,
    )
    output_df = descriptions_df.merge(
        references_df[["cluster_id", "references"]],
        on="cluster_id",
        how="left",
    )
    output_df = output_df[["cluster_id", "description", "references"]]
    output_df = _normalize_expected_cluster_output(
        output_df=output_df,
        pipeline_name="pipeline_3",
        target_cluster_count=target_cluster_count,
    )
    return _save_pipeline_output(
        output_df,
        "pipeline_3",
        provider,
        output_file,
        significant_papers_per_cluster=significant_papers_per_cluster,
    )


def _run_labeled_cluster_one_step(
    query: str,
    prompt: str | None,
    cluster_id: int,
    cluster_df: pd.DataFrame,
    client: LLMClient,
    pipeline_name: str,
    prompts_file: str | Path | None = None,
    significant_papers_per_cluster: int = DEFAULT_SIGNIFICANT_PAPERS_PER_CLUSTER,
) -> dict[str, object]:
    allowed_paper_ids = set(cluster_df["paper_id"].astype(int).tolist())
    prompt_template = resolve_pipeline_prompt(
        pipeline_name=pipeline_name,
        step=1,
        override=prompt,
        prompts_file=prompts_file,
    )
    user_prompt = render_prompt_template(
        prompt_template,
        query=query,
        cluster_id=cluster_id,
        cluster_context=build_cluster_context(cluster_id, cluster_df, include_link_strength=True),
        significant_papers_per_cluster=significant_papers_per_cluster,
    )
    payload = client.generate_json(_system_prompt(scopus_mode=True), user_prompt)
    return _standardize_output_records(
        payload=payload,
        scopus_mode=True,
        allowed_paper_ids=allowed_paper_ids,
    ).iloc[0].to_dict()


def _run_labeled_cluster_two_step(
    query: str,
    prompt: str | None,
    cluster_id: int,
    cluster_df: pd.DataFrame,
    client: LLMClient,
    prompts_file: str | Path | None = None,
    significant_papers_per_cluster: int = DEFAULT_SIGNIFICANT_PAPERS_PER_CLUSTER,
) -> dict[str, object]:
    allowed_paper_ids = set(cluster_df["paper_id"].astype(int).tolist())
    cluster_context = build_cluster_context(cluster_id, cluster_df, include_link_strength=True)
    step_one_template = resolve_pipeline_prompt(
        pipeline_name="pipeline5",
        step=1,
        prompts_file=prompts_file,
    )
    step_one_prompt = render_prompt_template(
        step_one_template,
        query=query,
        cluster_id=cluster_id,
        cluster_context=cluster_context,
        significant_papers_per_cluster=significant_papers_per_cluster,
    )
    references_payload = client.generate_json(_system_prompt(scopus_mode=True), step_one_prompt)
    references_df = _standardize_output_records(
        payload=references_payload,
        scopus_mode=True,
        allowed_paper_ids=allowed_paper_ids,
    )
    references = json.loads(references_df.iloc[0]["references"])
    reference_ids = [int(reference.strip("[]")) for reference in references]
    relevant_df = cluster_df[cluster_df["paper_id"].isin(reference_ids)].copy()

    step_two_template = resolve_pipeline_prompt(
        pipeline_name="pipeline5",
        step=2,
        override=prompt,
        prompts_file=prompts_file,
    )
    step_two_prompt = render_prompt_template(
        step_two_template,
        query=query,
        cluster_id=cluster_id,
        relevant_cluster_context=build_cluster_context(
            cluster_id,
            relevant_df,
            include_link_strength=True,
        ),
    )
    description_payload = client.generate_json(_system_prompt(scopus_mode=True), step_two_prompt)
    description_record = _extract_records(description_payload)[0]
    normalized_keys = {str(key).strip().lower(): value for key, value in description_record.items()}
    return {
        "cluster_id": int(normalized_keys.get("cluster_id") or normalized_keys.get("cluster") or cluster_id),
        "description": clean_text(normalized_keys.get("description", "")),
        "references": references_df.iloc[0]["references"],
    }


def run_pipeline_4(
    query: str,
    prompt: str | None,
    labeled_scopus_df: pd.DataFrame,
    provider: str,
    model: str | None,
    output_file: str | Path,
    target_cluster_count: int | None = None,
    significant_papers_per_cluster: int = DEFAULT_SIGNIFICANT_PAPERS_PER_CLUSTER,
    temperature: float = 0.2,
    prompts_file: str | Path | None = None,
) -> pd.DataFrame:
    client = LLMClient(LLMConfig(provider=provider, model=model, temperature=temperature))
    client.set_request_context(
        paper=Path(output_file).stem,
        pipeline="pipeline_4",
        step="step_1",
    )
    labeled_only_df = labeled_scopus_df.dropna(subset=["cluster"]).copy()
    expected_cluster_ids = sorted(
        pd.to_numeric(labeled_only_df["cluster"], errors="raise").astype(int).unique().tolist()
    )
    prompt_template = resolve_pipeline_prompt(
        pipeline_name="pipeline4",
        step=1,
        override=prompt,
        prompts_file=prompts_file,
    )
    user_prompt = render_prompt_template(
        prompt_template,
        query=query,
        target_cluster_count=len(expected_cluster_ids),
        labeled_scopus_context=build_labeled_scopus_context(
            labeled_scopus_df=labeled_scopus_df,
            include_link_strength=True,
        ),
        significant_papers_per_cluster=significant_papers_per_cluster,
    )
    payload = client.generate_json(_system_prompt(scopus_mode=True), user_prompt)
    output_df = _standardize_output_records(
        payload=payload,
        scopus_mode=True,
        allowed_paper_ids=set(labeled_scopus_df["paper_id"].astype(int).tolist()),
    )
    output_df = _normalize_expected_cluster_output(
        output_df=output_df,
        pipeline_name="pipeline_4",
        target_cluster_count=None,
    )
    returned_cluster_ids = sorted(output_df["cluster_id"].astype(int).tolist())
    if returned_cluster_ids != expected_cluster_ids:
        raise ValueError(
            "pipeline_4 must return exactly the labeled cluster_id values from the Scopus file. "
            f"Expected {expected_cluster_ids}, got {returned_cluster_ids}."
        )
    return _save_pipeline_output(
        output_df,
        "pipeline_4",
        provider,
        output_file,
        significant_papers_per_cluster=significant_papers_per_cluster,
    )


def run_pipeline_5(
    query: str,
    prompt: str | None,
    labeled_scopus_df: pd.DataFrame,
    provider: str,
    model: str | None,
    output_file: str | Path,
    target_cluster_count: int | None = None,
    significant_papers_per_cluster: int = DEFAULT_SIGNIFICANT_PAPERS_PER_CLUSTER,
    temperature: float = 0.2,
    prompts_file: str | Path | None = None,
) -> pd.DataFrame:
    client = LLMClient(LLMConfig(provider=provider, model=model, temperature=temperature))
    labeled_only_df = labeled_scopus_df.dropna(subset=["cluster"]).copy()
    paper = Path(output_file).stem
    expected_cluster_ids = sorted(
        pd.to_numeric(labeled_only_df["cluster"], errors="raise").astype(int).unique().tolist()
    )
    step_one_template = resolve_pipeline_prompt(
        pipeline_name="pipeline5",
        step=1,
        prompts_file=prompts_file,
    )
    step_one_prompt = render_prompt_template(
        step_one_template,
        query=query,
        target_cluster_count=len(expected_cluster_ids),
        labeled_scopus_context=build_labeled_scopus_context(
            labeled_scopus_df=labeled_only_df,
            include_link_strength=True,
        ),
        significant_papers_per_cluster=significant_papers_per_cluster,
    )
    client.set_request_context(
        paper=paper,
        pipeline="pipeline_5",
        step="step_1",
    )
    references_payload = client.generate_json(_system_prompt(scopus_mode=True), step_one_prompt)
    references_df = _standardize_output_records(
        payload=references_payload,
        scopus_mode=True,
        allowed_paper_ids=set(labeled_only_df["paper_id"].astype(int).tolist()),
    )
    references_df = _normalize_expected_cluster_output(
        output_df=references_df,
        pipeline_name="pipeline_5 step 1",
        target_cluster_count=None,
    )
    returned_cluster_ids = sorted(references_df["cluster_id"].astype(int).tolist())
    if returned_cluster_ids != expected_cluster_ids:
        raise ValueError(
            "pipeline_5 step 1 must return exactly the labeled cluster_id values from the Scopus file. "
            f"Expected {expected_cluster_ids}, got {returned_cluster_ids}."
        )

    selected_cluster_contexts = _build_selected_cluster_contexts(
        references_df=references_df,
        scopus_df=labeled_only_df,
        include_link_strength=True,
    )
    step_two_template = resolve_pipeline_prompt(
        pipeline_name="pipeline5",
        step=2,
        override=prompt,
        prompts_file=prompts_file,
    )
    step_two_prompt = render_prompt_template(
        step_two_template,
        query=query,
        target_cluster_count=len(expected_cluster_ids),
        selected_cluster_contexts=selected_cluster_contexts,
    )
    client.set_request_context(
        paper=paper,
        pipeline="pipeline_5",
        step="step_2",
    )
    descriptions_payload = client.generate_json(_system_prompt(scopus_mode=True), step_two_prompt)
    descriptions_records = []
    for record in _extract_records(descriptions_payload):
        normalized_keys = {str(key).strip().lower(): value for key, value in record.items()}
        descriptions_records.append(
            {
                "cluster_id": int(normalized_keys.get("cluster_id") or normalized_keys.get("cluster")),
                "description": clean_text(normalized_keys.get("description", "")),
            }
        )
    descriptions_df = pd.DataFrame(descriptions_records).sort_values("cluster_id")
    descriptions_df = _normalize_expected_cluster_output(
        output_df=descriptions_df,
        pipeline_name="pipeline_5 step 2",
        target_cluster_count=None,
    )
    returned_description_ids = sorted(descriptions_df["cluster_id"].astype(int).tolist())
    if returned_description_ids != expected_cluster_ids:
        raise ValueError(
            "pipeline_5 step 2 must return exactly the labeled cluster_id values from the Scopus file. "
            f"Expected {expected_cluster_ids}, got {returned_description_ids}."
        )

    output_df = descriptions_df.merge(
        references_df[["cluster_id", "references"]],
        on="cluster_id",
        how="inner",
        validate="one_to_one",
    )
    output_df = output_df[["cluster_id", "description", "references"]]
    return _save_pipeline_output(
        output_df,
        "pipeline_5",
        provider,
        output_file,
        significant_papers_per_cluster=significant_papers_per_cluster,
    )


def run_pipeline_6(
    query: str,
    prompt: str | None,
    labeled_scopus_df: pd.DataFrame,
    provider: str,
    model: str | None,
    output_file: str | Path,
    top_k_clusters: int = DEFAULT_TOP_K_CLUSTERS,
    target_cluster_count: int | None = None,
    significant_papers_per_cluster: int = DEFAULT_SIGNIFICANT_PAPERS_PER_CLUSTER,
    temperature: float = 0.2,
    prompts_file: str | Path | None = None,
) -> pd.DataFrame:
    client = LLMClient(LLMConfig(provider=provider, model=model, temperature=temperature))
    client.set_request_context(
        paper=Path(output_file).stem,
        pipeline="pipeline_6",
        step="step_1",
    )
    top_clusters = select_top_clusters_by_link_strength(
        labeled_scopus_df=labeled_scopus_df,
        top_k_clusters=top_k_clusters,
    )
    selected_cluster_frames: list[pd.DataFrame] = []
    for cluster_id in top_clusters:
        cluster_df = labeled_scopus_df[labeled_scopus_df["cluster"] == cluster_id].copy()
        cluster_df = cluster_df.sort_values(
            ["link_strength", "cited_by", "paper_id"],
            ascending=[False, False, True],
        ).head(significant_papers_per_cluster)
        selected_cluster_frames.append(cluster_df)

    selected_clusters_df = pd.concat(selected_cluster_frames, ignore_index=True)
    expected_cluster_ids = sorted(int(cluster_id) for cluster_id in top_clusters)
    prompt_template = resolve_pipeline_prompt(
        pipeline_name="pipeline6",
        step=1,
        override=prompt,
        prompts_file=prompts_file,
    )
    user_prompt = render_prompt_template(
        prompt_template,
        query=query,
        target_cluster_count=len(expected_cluster_ids),
        selected_clusters_context=build_labeled_scopus_context(
            labeled_scopus_df=selected_clusters_df,
            include_link_strength=True,
        ),
    )
    payload = client.generate_json(_system_prompt(scopus_mode=True), user_prompt)
    output_df = _standardize_output_records(
        payload=payload,
        scopus_mode=True,
        allowed_paper_ids=set(selected_clusters_df["paper_id"].astype(int).tolist()),
    )
    output_df = _normalize_expected_cluster_output(
        output_df=output_df,
        pipeline_name="pipeline_6",
        target_cluster_count=None,
    )
    returned_cluster_ids = sorted(output_df["cluster_id"].astype(int).tolist())
    if returned_cluster_ids != expected_cluster_ids:
        raise ValueError(
            "pipeline_6 must return exactly the selected top cluster_id values. "
            f"Expected {expected_cluster_ids}, got {returned_cluster_ids}."
        )
    return _save_pipeline_output(
        output_df,
        "pipeline_6",
        provider,
        output_file,
        significant_papers_per_cluster=significant_papers_per_cluster,
    )
