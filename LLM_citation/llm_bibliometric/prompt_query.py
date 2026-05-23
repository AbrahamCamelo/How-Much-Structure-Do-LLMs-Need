from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .constants import PROJECT_ROOT
from .utils import clean_text, normalize_column_name


DEFAULT_PROMPTS_FILE = PROJECT_ROOT / "prompts.csv"
DEFAULT_QUERIES_FILE = PROJECT_ROOT / "queries.xlsx"


def _decode_escaped_newlines(text: object) -> str:
    if isinstance(text, float) and pd.isna(text):
        return ""
    if text is None:
        return ""
    value = str(text).replace("\r\n", "\n").replace("\r", "\n").strip()
    return value.replace("\\n", "\n")


def _read_prompts_csv(file_path: str | Path | None) -> pd.DataFrame:
    path = Path(file_path or DEFAULT_PROMPTS_FILE)
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "latin-1", "cp1252"):
        try:
            dataframe = pd.read_csv(path, encoding=encoding, sep=";")
            if len(dataframe.columns) >= 2:
                return dataframe
        except Exception as error:
            last_error = error
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Unable to read prompts file: {path}")


def load_prompts_catalog(file_path: str | Path | None = DEFAULT_PROMPTS_FILE) -> pd.DataFrame:
    dataframe = _read_prompts_csv(file_path).copy()
    dataframe.columns = [normalize_column_name(column) for column in dataframe.columns]
    required_columns = {"pipeline", "prompt1"}
    missing = required_columns - set(dataframe.columns)
    if missing:
        raise ValueError(f"Prompts file is missing required columns: {sorted(missing)}")

    if "prompt2" not in dataframe.columns:
        dataframe["prompt2"] = ""

    dataframe["pipeline"] = dataframe["pipeline"].map(clean_text)
    dataframe["prompt1"] = dataframe["prompt1"].map(_decode_escaped_newlines)
    dataframe["prompt2"] = dataframe["prompt2"].map(_decode_escaped_newlines)
    dataframe = dataframe[dataframe["pipeline"] != ""].copy()
    return dataframe


def resolve_pipeline_prompt(
    pipeline_name: str,
    step: int = 1,
    override: str | None = None,
    prompts_file: str | Path | None = DEFAULT_PROMPTS_FILE,
) -> str:
    if clean_text(override):
        return clean_text(override)

    catalog = load_prompts_catalog(prompts_file)
    normalized_pipeline_name = normalize_column_name(pipeline_name).replace("_", "")

    pipeline_row = catalog[
        catalog["pipeline"].map(lambda value: normalize_column_name(value).replace("_", ""))
        == normalized_pipeline_name
    ]
    if pipeline_row.empty:
        raise ValueError(f"No prompt entry found for pipeline '{pipeline_name}'.")

    column_name = f"prompt{step}"
    if column_name not in pipeline_row.columns:
        raise ValueError(f"Prompt step {step} is not available for pipeline '{pipeline_name}'.")

    prompt_value = _decode_escaped_newlines(pipeline_row.iloc[0][column_name])
    if not prompt_value:
        raise ValueError(f"Prompt step {step} is empty for pipeline '{pipeline_name}'.")
    return prompt_value


def load_queries_catalog(file_path: str | Path | None = DEFAULT_QUERIES_FILE) -> pd.DataFrame:
    dataframe = pd.read_excel(file_path or DEFAULT_QUERIES_FILE).copy()
    dataframe.columns = [normalize_column_name(column) for column in dataframe.columns]

    paper_column = None
    query_column = None
    for column in dataframe.columns:
        if column == "paper":
            paper_column = column
        if column in {"advance_query", "advanced_query", "query"}:
            query_column = column

    if paper_column is None or query_column is None:
        raise ValueError(
            "queries.xlsx must contain a 'paper' column and a query column such as 'advance query'."
        )

    output = pd.DataFrame()
    output["paper"] = pd.to_numeric(dataframe[paper_column], errors="raise").astype(int)
    output["query"] = dataframe[query_column].map(clean_text)
    output = output[output["query"] != ""].copy()
    return output


def extract_paper_number(reference: str | Path | None) -> int | None:
    if reference is None:
        return None
    stem = Path(reference).stem
    match = re.search(r"(\d+)", stem)
    if not match:
        return None
    return int(match.group(1))


def resolve_query_text(
    query: str | None = None,
    query_id: int | None = None,
    reference_path: str | Path | None = None,
    queries_file: str | Path | None = DEFAULT_QUERIES_FILE,
) -> str:
    if clean_text(query):
        return clean_text(query)

    resolved_query_id = query_id
    if resolved_query_id is None:
        resolved_query_id = extract_paper_number(reference_path)

    if resolved_query_id is None:
        raise ValueError(
            "No query was provided and the query id could not be inferred from the file name."
        )

    catalog = load_queries_catalog(queries_file)
    matched = catalog[catalog["paper"] == int(resolved_query_id)]
    if matched.empty:
        raise ValueError(f"No query found for paper id {resolved_query_id}.")
    return clean_text(matched.iloc[0]["query"])


def render_prompt_template(template: str, **values: object) -> str:
    rendered = str(template)
    for key, value in values.items():
        if isinstance(value, float) and pd.isna(value):
            replacement = ""
        elif value is None:
            replacement = ""
        else:
            replacement = str(value)
        rendered = rendered.replace(f"{{{{{key}}}}}", replacement)
    return rendered
