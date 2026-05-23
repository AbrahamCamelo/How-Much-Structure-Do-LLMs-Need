from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd
from google import genai
from openai import OpenAI

from .constants import DEFAULT_EMBEDDING_MODEL
from .utils import ensure_directory, text_hash


@dataclass(frozen=True)
class EmbeddingConfig:
    model: str = DEFAULT_EMBEDDING_MODEL
    batch_size: int = 128


def normalize_embedding_model_name(model: str | None) -> str:
    normalized = str(model or DEFAULT_EMBEDDING_MODEL).strip()
    aliases = {
        "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
        "bge-large-en-v1.5": "BAAI/bge-large-en-v1.5",
        "gemini-embedding-2": "gemini/gemini-embedding-2",
    }
    return aliases.get(normalized, normalized)


class OpenAIEmbeddingService:
    def __init__(self, model: str = DEFAULT_EMBEDDING_MODEL) -> None:
        import os

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set.")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def embed_texts(self, texts: Sequence[str], batch_size: int = 128) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            response = self.client.embeddings.create(model=self.model, input=batch)
            embeddings.extend(item.embedding for item in response.data)
        return embeddings


class SentenceTransformerEmbeddingService:
    def __init__(self, model: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise ImportError(
                "sentence-transformers is required for local embedding models such as "
                "'sentence-transformers/all-mpnet-base-v2'. Install it from requirements.txt."
            ) from error

        self.model = model
        self.client = SentenceTransformer(model)

    def embed_texts(self, texts: Sequence[str], batch_size: int = 128) -> list[list[float]]:
        embeddings = self.client.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        return embeddings.tolist()


class GeminiEmbeddingService:
    def __init__(self, model: str) -> None:
        import os

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY is not set.")
        self.client = genai.Client(api_key=api_key)
        self.model = model.removeprefix("gemini/")

    def embed_texts(self, texts: Sequence[str], batch_size: int = 128) -> list[list[float]]:
        embeddings: list[list[float]] = []
        # Gemini's embed_content response shape does not match the OpenAI-style
        # batched assumption used elsewhere in this file. Request one text at a
        # time so we always get exactly one embedding per input row.
        for text in texts:
            response = self.client.models.embed_content(
                model=self.model,
                contents=text,
            )
            response_embeddings = getattr(response, "embeddings", None)
            if not response_embeddings:
                raise ValueError(
                    f"Gemini embedding response did not include embeddings for model {self.model}."
                )
            values = getattr(response_embeddings[0], "values", None)
            if values is None:
                raise ValueError(
                    f"Gemini embedding response included an embedding without values for model {self.model}."
                )
            embeddings.append(values)
        return embeddings


def _embedding_service_for_model(model: str):
    normalized_model = normalize_embedding_model_name(model)
    if normalized_model.startswith("sentence-transformers/") or normalized_model.startswith("BAAI/"):
        return SentenceTransformerEmbeddingService(normalized_model)
    if normalized_model.startswith("gemini/"):
        return GeminiEmbeddingService(normalized_model)
    return OpenAIEmbeddingService(model=normalized_model)


def load_embeddings_cache(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        return pd.DataFrame(columns=["text_hash", "text", "embedding_model", "embedding"])
    return pd.read_parquet(file_path)


def ensure_embeddings(
    records_df: pd.DataFrame,
    cache_path: str | Path,
    config: EmbeddingConfig | None = None,
) -> pd.DataFrame:
    config = config or EmbeddingConfig()
    normalized_model = normalize_embedding_model_name(config.model)
    config = EmbeddingConfig(model=normalized_model, batch_size=config.batch_size)
    cache_path = Path(cache_path)
    ensure_directory(cache_path.parent)

    work_df = records_df.copy()
    if "text_hash" not in work_df.columns:
        work_df["text_hash"] = work_df["text"].map(text_hash)

    cache_df = load_embeddings_cache(cache_path)
    if not cache_df.empty:
        cache_df = cache_df[cache_df["embedding_model"] == config.model].copy()

    merged = work_df.merge(
        cache_df[["text_hash", "embedding"]],
        on="text_hash",
        how="left",
    )
    missing_mask = merged["embedding"].isna()
    if missing_mask.any():
        missing_records = merged.loc[missing_mask, ["text_hash", "text"]].drop_duplicates(
            subset=["text_hash"]
        )
        service = _embedding_service_for_model(config.model)
        new_embeddings = service.embed_texts(
            missing_records["text"].tolist(),
            batch_size=config.batch_size,
        )
        additions = missing_records.copy()
        additions["embedding_model"] = config.model
        additions["embedding"] = new_embeddings

        cache_df = pd.concat([cache_df, additions], ignore_index=True)
        cache_df = cache_df.drop_duplicates(
            subset=["text_hash", "embedding_model"],
            keep="last",
        )
        cache_df.to_parquet(cache_path, index=False)
        merged = work_df.merge(
            cache_df[["text_hash", "embedding"]],
            on="text_hash",
            how="left",
        )
    else:
        if not cache_path.exists():
            cache_df.to_parquet(cache_path, index=False)

    return merged
