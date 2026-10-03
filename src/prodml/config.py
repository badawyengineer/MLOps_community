"""Central configuration — everything env-driven, nothing hardcoded.

Every path or tunable that other modules need lives here, read once
from environment variables (with sane defaults) via pydantic-settings.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PRODML_", env_file=".env", extra="ignore")

    # --- corpus / data ---
    raw_pdf_path: Path = Path("data/raw/egyptian_civil_code.pdf")
    corpus_path: Path = Path("data/processed/articles.json")
    repealed_ranges: str = "54-80"  # Presidential-Decree-repealed articles, flagged not dropped

    # --- retrieval ---
    top_k: int = 3
    embedding_model: str = (
        "hashing-1024"  # "hashing" | "lsa" | "<org>/<sentence-transformers model>"
    )
    chunk_chars: int = 0  # 0 = one chunk per article (the default, and what citations want)
    chunk_overlap: int = 0
    index_dir: Path = Path("data/index")
    retriever: str = (
        "hybrid-dense"  # "tfidf" | "bm25" | "hybrid" | "vector" | "hybrid-dense" | "embedding"
    )
    min_confidence: float = 0.2  # abstain when the top score is below this (calibrate via eval)
    max_context_chars: int = 4000

    # --- service ---
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    model_version: str = "0.1.0"

    # --- generation ---
    generation_backend: str = (
        "extractive"  # "extractive" | "vllm" | "api"  (vllm/api = OpenAI-compatible)
    )
    llm_base_url: str = (
        "http://localhost:8001/v1"  # vLLM default; any OpenAI-compatible endpoint works
    )
    llm_model: str = ""  # e.g. the model name you pass to `vllm serve`
    llm_api_key: str = ""
    llm_timeout: float = 30.0
    llm_max_tokens: int = 400
    llm_price_in_per_1k: float = 0.0  # USD, for the cost/hour Grafana panel; 0 for self-hosted
    llm_price_out_per_1k: float = 0.0

    # --- monitoring ---
    langfuse_host: str = ""  # e.g. http://localhost:3000 (self-hosted)
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    drift_baseline_path: Path = Path("data/eval/questions.json")


settings = Settings()
