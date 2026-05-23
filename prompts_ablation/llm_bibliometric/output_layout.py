from __future__ import annotations

from pathlib import Path

from .constants import DEFAULT_EMBEDDING_MODEL, DESCRIPTION_DIR, FULL_RUN_DIR
from .embeddings import normalize_embedding_model_name
from .prompt_query import DEFAULT_PROMPTS_FILE
from .utils import slugify_filename


PROMPT_VARIANTS_DIRNAME = "prompt_variants"
EMBEDDING_VARIANTS_DIRNAME = "embedding_variants"


def prompt_variant_name(prompts_file: str | Path | None) -> str | None:
    candidate = Path(prompts_file or DEFAULT_PROMPTS_FILE)
    try:
        if candidate.resolve() == DEFAULT_PROMPTS_FILE.resolve():
            return None
    except OSError:
        if candidate == DEFAULT_PROMPTS_FILE:
            return None

    stem = slugify_filename(candidate.stem)
    return stem or "custom_prompts"


def description_output_dir(
    pipeline_name: str,
    provider: str,
    prompts_file: str | Path | None = None,
) -> Path:
    variant = prompt_variant_name(prompts_file)
    if variant is None:
        return DESCRIPTION_DIR / pipeline_name / provider.lower()
    return DESCRIPTION_DIR / PROMPT_VARIANTS_DIRNAME / variant / pipeline_name / provider.lower()


def embedding_variant_name(embedding_model: str | None) -> str | None:
    normalized = normalize_embedding_model_name(embedding_model or DEFAULT_EMBEDDING_MODEL)
    default_normalized = normalize_embedding_model_name(DEFAULT_EMBEDDING_MODEL)
    if normalized == default_normalized:
        return None
    slug = slugify_filename(normalized)
    return slug or "custom_embedding_model"


def evaluation_artifact_dir(
    base_dir: str | Path,
    embedding_model: str | None = None,
) -> Path:
    base_path = Path(base_dir)
    variant = embedding_variant_name(embedding_model)
    if variant is None:
        return base_path
    return base_path / EMBEDDING_VARIANTS_DIRNAME / variant


def full_run_output_dir(
    kind: str,
    provider: str,
    prompts_file: str | Path | None = None,
    embedding_model: str | None = None,
) -> Path:
    path = FULL_RUN_DIR / kind

    prompt_variant = prompt_variant_name(prompts_file)
    if prompt_variant is not None:
        path = path / PROMPT_VARIANTS_DIRNAME / prompt_variant

    embedding_variant = embedding_variant_name(embedding_model)
    if embedding_variant is not None:
        path = path / EMBEDDING_VARIANTS_DIRNAME / embedding_variant

    return path / provider.lower()
