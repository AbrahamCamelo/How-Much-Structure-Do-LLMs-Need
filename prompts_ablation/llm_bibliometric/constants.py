from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCOPUS_DIR = PROJECT_ROOT / "scopus_documents"
DESCRIPTION_DIR = PROJECT_ROOT / "description"
CLUSTERS_DIR = PROJECT_ROOT / "clusters"
EVALUATIONS_DIR = PROJECT_ROOT / "evaluations"
SHARED_EMBEDDINGS_DIR = EVALUATIONS_DIR / "shared_embeddings"
COVERAGE_DIR = EVALUATIONS_DIR / "coverage_corpus"
QUALITY_DIR = EVALUATIONS_DIR / "quality_of_the_induced_clustering"
SILHOUETTE_DIR = EVALUATIONS_DIR / "silhouette_score"
SIMILARITY_DIR = EVALUATIONS_DIR / "clustering_similarity"
MODULARITY_DIR = EVALUATIONS_DIR / "modularity"
FULL_RUN_DIR = EVALUATIONS_DIR / "full_run"

DEFAULT_CHAT_MODELS = {
    "chatgpt": os.getenv("CHATGPT_MODEL")
    or os.getenv("OPENAI_CHAT_MODEL")
    or "gpt-5.4",
    "gemini": os.getenv("GEMINI_MODEL")
    or os.getenv("GEMINI_CHAT_MODEL")
    or "gemini-2.5-flash",
    "claude": os.getenv("CLAUDE_MODEL")
    or os.getenv("ANTHROPIC_CHAT_MODEL")
    or "claude-3-5-sonnet-latest",
}

DEFAULT_EMBEDDING_MODEL = (
    os.getenv("OPENAI_EMBEDDING_MODEL") or "text-embedding-3-large"
)
DEFAULT_SIGNIFICANT_PAPERS_PER_CLUSTER = 10
DEFAULT_TOP_K_CLUSTERS = 20


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEBUG_DIR = PROJECT_ROOT / "debug"
TOKEN_USAGE_CSV = PROJECT_ROOT / "token_usage.csv"
SCOPUS_DIR = PROJECT_ROOT / "scopus_documents"
