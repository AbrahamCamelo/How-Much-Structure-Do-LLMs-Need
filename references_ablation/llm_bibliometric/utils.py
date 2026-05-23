from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+|[\r\n]+")
WHITESPACE_PATTERN = re.compile(r"\s+")
JSON_BLOCK_PATTERN = re.compile(r"(\[.*\]|\{.*\})", re.DOTALL)
REFERENCE_TOKEN_PATTERN = re.compile(r"\[(\d+)\]")


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_whitespace(text: object) -> str:
    if text is None:
        return ""
    return WHITESPACE_PATTERN.sub(" ", str(text)).strip()


def clean_text(value: object) -> str:
    if isinstance(value, float) and pd.isna(value):
        return ""
    if value is None:
        return ""
    return normalize_whitespace(value)


def split_sentences(text: object) -> list[str]:
    normalized = clean_text(text)
    if not normalized:
        return []
    parts = [clean_text(part) for part in SENTENCE_SPLIT_PATTERN.split(normalized)]
    sentences = [part for part in parts if part]
    return sentences if sentences else [normalized]


def safe_json_loads(text: str) -> object:
    payload = text.strip()
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        match = JSON_BLOCK_PATTERN.search(payload)
        if not match:
            raise
        return json.loads(match.group(1))


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def format_reference_token(paper_id: int | str) -> str:
    return f"[{int(paper_id)}]"


def unique_preserving_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def serialize_references(references: Sequence[str]) -> str:
    return json.dumps(list(references), ensure_ascii=False)


def parse_references(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    if isinstance(value, (list, tuple, set)):
        return [clean_text(item) for item in value if clean_text(item)]

    text = clean_text(value)
    if not text:
        return []

    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        loaded = None

    if isinstance(loaded, list):
        return [clean_text(item) for item in loaded if clean_text(item)]

    if REFERENCE_TOKEN_PATTERN.search(text):
        return unique_preserving_order(
            format_reference_token(match.group(1))
            for match in REFERENCE_TOKEN_PATTERN.finditer(text)
        )

    return unique_preserving_order(
        clean_text(item)
        for item in re.split(r"\s*;\s*|\s*,\s*", text)
        if clean_text(item)
    )


def normalize_scopus_reference_list(
    references: Sequence[str | int],
    allowed_paper_ids: set[int] | None = None,
) -> list[str]:
    normalized: list[str] = []
    for reference in references:
        if isinstance(reference, str):
            token_match = REFERENCE_TOKEN_PATTERN.fullmatch(reference.strip())
            if token_match:
                paper_id = int(token_match.group(1))
            else:
                stripped = reference.strip()
                if not stripped.isdigit():
                    continue
                paper_id = int(stripped)
        else:
            paper_id = int(reference)

        if allowed_paper_ids is not None and paper_id not in allowed_paper_ids:
            continue
        normalized.append(format_reference_token(paper_id))
    return unique_preserving_order(normalized)


def normalize_column_name(column_name: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(column_name).strip().lower())
    return normalized.strip("_")


def coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def slugify_filename(text: str) -> str:
    normalized = normalize_column_name(text)
    return normalized or "output"
