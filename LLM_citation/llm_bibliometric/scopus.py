from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .utils import clean_text, coerce_numeric, ensure_directory, normalize_column_name


SCOPUS_COLUMN_ALIASES = {
    "paper_id": ("paper_id",),
    "paper_title": ("paper_title", "title"),
    "paper_abstract": ("paper_abstract", "abstract"),
    "references": ("references",),
    "cluster": ("cluster",),
    "link_strength": ("link_strength", "link strength"),
    "clustering_method": ("clustering_method", "clustering method"),
    "clustering_min_cluster_size": (
        "clustering_min_cluster_size",
        "clustering min cluster size",
    ),
    "clustering_effective_min_cluster_size": (
        "clustering_effective_min_cluster_size",
        "clustering effective min cluster size",
    ),
    "clustering_min_cluster_proportion": (
        "clustering_min_cluster_proportion",
        "clustering min cluster proportion",
    ),
    "clustering_min_cluster_proportion_basis": (
        "clustering_min_cluster_proportion_basis",
        "clustering min cluster proportion basis",
    ),
    "clustering_cluster_selection_strategy": (
        "clustering_cluster_selection_strategy",
        "clustering cluster selection strategy",
    ),
    "clustering_requested_resolution": (
        "clustering_requested_resolution",
        "clustering requested resolution",
    ),
    "clustering_resolution": ("clustering_resolution", "clustering resolution"),
    "clustering_max_clusters": (
        "clustering_max_clusters",
        "clustering max clusters",
    ),
    "authors": ("authors",),
    "year": ("year",),
    "cited_by": ("cited_by", "cited by"),
    "source_title": ("source_title", "source title"),
    "doi": ("doi",),
    "eid": ("eid",),
}

STANDARD_SCOPUS_COLUMNS = [
    "paper_id",
    "paper_title",
    "paper_abstract",
    "references",
    "cluster",
    "link_strength",
    "clustering_method",
    "clustering_min_cluster_size",
    "clustering_effective_min_cluster_size",
    "clustering_min_cluster_proportion",
    "clustering_min_cluster_proportion_basis",
    "clustering_cluster_selection_strategy",
    "clustering_requested_resolution",
    "clustering_resolution",
    "clustering_max_clusters",
    "authors",
    "year",
    "cited_by",
    "source_title",
    "doi",
    "eid",
]


def _resolve_aliases(columns: Iterable[str]) -> dict[str, str]:
    normalized_to_actual = {
        normalize_column_name(column_name): column_name for column_name in columns
    }
    resolved: dict[str, str] = {}
    for target_column, aliases in SCOPUS_COLUMN_ALIASES.items():
        for alias in aliases:
            actual = normalized_to_actual.get(normalize_column_name(alias))
            if actual is not None:
                resolved[target_column] = actual
                break
    return resolved


def read_csv_with_fallbacks(file_path: Path) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError as error:
            last_error = error
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Unable to read {file_path}")


def standardize_scopus_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    dataframe = raw_df.copy()
    dataframe.columns = [str(column).strip() for column in dataframe.columns]
    resolved = _resolve_aliases(dataframe.columns)

    required = {"paper_title", "paper_abstract", "references"}
    missing = sorted(required - set(resolved))
    if missing:
        raise ValueError(
            "Missing required Scopus column(s): "
            + ", ".join(missing)
            + f". Available columns: {list(dataframe.columns)}"
        )

    standardized = pd.DataFrame(index=dataframe.index)
    if "paper_id" in resolved:
        paper_ids = coerce_numeric(dataframe[resolved["paper_id"]]).astype("Int64")
        if paper_ids.isna().any():
            raise ValueError("Existing paper_id column contains non-numeric values.")
        standardized["paper_id"] = paper_ids.astype(int)
    else:
        standardized = standardized.reset_index(drop=True)
        standardized["paper_id"] = standardized.index + 1
        dataframe = dataframe.reset_index(drop=True)

    standardized["paper_title"] = dataframe[resolved["paper_title"]].map(clean_text)
    standardized["paper_abstract"] = dataframe[resolved["paper_abstract"]].map(clean_text)
    standardized["references"] = dataframe[resolved["references"]].map(clean_text)

    for optional_column in (
        "cluster",
        "link_strength",
        "clustering_method",
        "clustering_min_cluster_size",
        "clustering_effective_min_cluster_size",
        "clustering_min_cluster_proportion",
        "clustering_min_cluster_proportion_basis",
        "clustering_cluster_selection_strategy",
        "clustering_requested_resolution",
        "clustering_resolution",
        "clustering_max_clusters",
        "authors",
        "year",
        "cited_by",
        "source_title",
        "doi",
        "eid",
    ):
        if optional_column in resolved:
            standardized[optional_column] = dataframe[resolved[optional_column]]
        else:
            standardized[optional_column] = pd.NA

    for numeric_column in (
        "cluster",
        "link_strength",
        "clustering_min_cluster_size",
        "clustering_effective_min_cluster_size",
        "clustering_min_cluster_proportion",
        "clustering_requested_resolution",
        "clustering_resolution",
        "clustering_max_clusters",
        "year",
        "cited_by",
    ):
        standardized[numeric_column] = coerce_numeric(standardized[numeric_column])

    standardized = standardized[STANDARD_SCOPUS_COLUMNS].copy()
    standardized["paper_id"] = standardized["paper_id"].astype(int)

    if standardized["paper_id"].duplicated().any():
        raise ValueError("paper_id values must be unique.")

    return standardized


def load_scopus_csv(file_path: str | Path) -> pd.DataFrame:
    file_path = Path(file_path)
    dataframe = read_csv_with_fallbacks(file_path)
    return standardize_scopus_dataframe(dataframe)


def remove_generated_paper_id(raw_df: pd.DataFrame) -> pd.DataFrame:
    dataframe = raw_df.copy()
    removable_columns = [
        column for column in dataframe.columns if normalize_column_name(column) == "paper_id"
    ]
    if removable_columns:
        dataframe = dataframe.drop(columns=removable_columns)
    return dataframe


def save_scopus_csv(dataframe: pd.DataFrame, file_path: str | Path) -> Path:
    file_path = Path(file_path)
    ensure_directory(file_path.parent)
    standardized = standardize_scopus_dataframe(dataframe)
    standardized.to_csv(file_path, index=False, encoding="utf-8")
    return file_path


def prepare_scopus_documents(source_dir: str | Path, target_dir: str | Path) -> list[Path]:
    source_dir = Path(source_dir)
    target_dir = ensure_directory(Path(target_dir))

    prepared_paths: list[Path] = []
    for source_path in sorted(source_dir.glob("*.csv")):
        target_path = target_dir / source_path.name
        raw_df = read_csv_with_fallbacks(source_path)
        raw_df = remove_generated_paper_id(raw_df)
        raw_df.to_csv(target_path, index=False, encoding="utf-8")
        prepared_paths.append(target_path)
    return prepared_paths
