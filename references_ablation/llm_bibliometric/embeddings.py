from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd
from openai import OpenAI

from .constants import DEFAULT_EMBEDDING_MODEL
from .utils import ensure_directory, text_hash


@dataclass(frozen=True)
class EmbeddingConfig:
    model: str = DEFAULT_EMBEDDING_MODEL
    batch_size: int = 128


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
        service = OpenAIEmbeddingService(model=config.model)
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
