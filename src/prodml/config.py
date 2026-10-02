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
    retriever: str = "hybrid"  # "tfidf" | "bm25" | "hybrid" | "embedding"
    min_confidence: float = 0.2  # abstain when the top score is below this (calibrate via eval)
    max_context_chars: int = 4000

    # --- service ---
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    model_version: str = "0.1.0"

    # --- future serving hook (Module 3 swaps this in) ---
    generation_backend: str = "extractive"  # "extractive" | "vllm" | "api"


settings = Settings()
