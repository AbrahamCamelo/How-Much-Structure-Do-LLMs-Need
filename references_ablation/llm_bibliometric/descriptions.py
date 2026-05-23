from __future__ import annotations

from pathlib import Path

import pandas as pd

from .constants import DESCRIPTION_DIR
from .prompt_query import extract_paper_number
from .utils import clean_text, ensure_directory, parse_references, serialize_references, slugify_filename


def _read_csv_with_candidates(file_path: str | Path) -> pd.DataFrame:
    path = Path(file_path)
    attempts = [
        {"sep": ",", "encoding": "utf-8-sig"},
        {"sep": ";", "encoding": "utf-8-sig"},
        {"sep": ",", "encoding": "utf-8"},
        {"sep": ";", "encoding": "utf-8"},
        {"sep": ",", "encoding": "latin-1"},
        {"sep": ";", "encoding": "latin-1"},
    ]
    last_error: Exception | None = None
    for options in attempts:
        try:
            return pd.read_csv(path, **options)
        except Exception as error:
            last_error = error
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Unable to read description file: {path}")


def normalize_description_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    normalized_columns = {column: str(column).strip().lower() for column in dataframe.columns}

    cluster_column = None
    description_column = None
    references_column = None
    for original, normalized in normalized_columns.items():
        if normalized in {"cluster_id", "cluster"}:
            cluster_column = original
        elif normalized == "description":
            description_column = original
        elif normalized == "references":
            references_column = original

    if cluster_column is None or description_column is None:
        raise ValueError(
            "Description file must contain cluster/cluster_id and description columns."
        )

    output_df = pd.DataFrame()
    output_df["cluster_id"] = pd.to_numeric(
        dataframe[cluster_column],
        errors="raise",
    ).astype(int)
    output_df["description"] = dataframe[description_column].map(clean_text)

    if references_column is None:
        output_df["references"] = serialize_references([])
    else:
        output_df["references"] = dataframe[references_column].map(
            lambda value: serialize_references(parse_references(value))
        )

    output_df = output_df.sort_values("cluster_id").reset_index(drop=True)
    return output_df


def load_description_csv(file_path: str | Path) -> pd.DataFrame:
    dataframe = _read_csv_with_candidates(file_path)
    return normalize_description_dataframe(dataframe)


def build_description_dataset_name(description_path: str | Path) -> str:
    path = Path(description_path)
    try:
        relative = path.relative_to(DESCRIPTION_DIR)
    except ValueError:
        relative = path
    parts = [slugify_filename(part) for part in relative.with_suffix("").parts]
    return "__".join(part for part in parts if part)


def infer_human_description_file(
    reference_path: str | Path,
    human_descriptions_dir: str | Path = DESCRIPTION_DIR / "human_descriptions",
) -> Path | None:
    paper_number = extract_paper_number(reference_path)
    if paper_number is None:
        return None

    candidate = Path(human_descriptions_dir) / (
        f"{paper_number:03d}paper_clusters_original_description.csv"
    )
    if candidate.exists():
        return candidate
    return None


def require_human_description_file(
    reference_path: str | Path,
    human_descriptions_dir: str | Path = DESCRIPTION_DIR / "human_descriptions",
) -> Path:
    human_description_file = infer_human_description_file(
        reference_path=reference_path,
        human_descriptions_dir=human_descriptions_dir,
    )
    if human_description_file is None:
        raise FileNotFoundError(
            "A matching human description file is required but was not found for "
            f"{reference_path}. Expected something like "
            f"{Path(human_descriptions_dir) / '001paper_clusters_original_description.csv'}."
        )
    return human_description_file


def require_human_cluster_target(
    reference_path: str | Path,
    human_descriptions_dir: str | Path = DESCRIPTION_DIR / "human_descriptions",
) -> tuple[Path, int]:
    human_description_file = require_human_description_file(
        reference_path=reference_path,
        human_descriptions_dir=human_descriptions_dir,
    )
    human_descriptions = load_description_csv(human_description_file)
    return human_description_file, int(human_descriptions["cluster_id"].nunique())


def import_original_descriptions(
    source_dir: str | Path = DESCRIPTION_DIR / "human_descriptions",
    destination_dir: str | Path | None = None,
) -> list[Path]:
    source_dir = Path(source_dir)
    destination_dir = (
        Path(destination_dir)
        if destination_dir is not None
        else DESCRIPTION_DIR / "human_descriptions"
    )
    ensure_directory(destination_dir)

    written_paths: list[Path] = []
    for source_path in sorted(source_dir.glob("*.csv")):
        normalized = load_description_csv(source_path)
        target_path = destination_dir / source_path.name
        normalized.to_csv(target_path, index=False, encoding="utf-8")
        written_paths.append(target_path)
    return written_paths
