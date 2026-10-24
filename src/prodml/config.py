"""Central configuration — everything env-driven, nothing hardcoded.

Every path or tunable that other modules need lives here, read once
from environment variables (with sane defaults) via pydantic-settings.
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field, ValidationError, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# pydantic-settings only reads PRODML_* from .env. Variables that third-party libraries read straight
# from os.environ (MLFLOW_TRACKING_URI, OPENAI_API_KEY for the optional RAGAS judge, HF_TOKEN, ...) would be
# silently ignored, so a .env file is loaded into the environment too. Real environment variables win.
load_dotenv(override=False)

RETRIEVERS = {"tfidf", "bm25", "hybrid", "vector", "hybrid-dense", "embedding"}
GENERATION_BACKENDS = {"extractive", "vllm", "api"}
LOG_LEVELS = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PRODML_", env_file=".env", extra="ignore")

    # --- corpus / data ---
    raw_pdf_path: Path = Path("data/raw/egyptian_civil_code.pdf")
    corpus_path: Path = Path("data/processed/articles.json")
    repealed_ranges: str = (
        "54-80,389-417"  # Presidential-Decree-repealed articles, flagged not dropped
    )

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
    min_lexical_overlap: int = (
        2  # refuse when the top article shares fewer content words with the question (0 = off)
    )
    min_confidence: float = 0.25  # abstain when the top score is below this (calibrate via eval)
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
    llm_api_key: str = Field("", repr=False)  # secret: never shown in repr()/logs
    llm_timeout: float = 30.0
    llm_max_tokens: int = 400
    llm_price_in_per_1k: float = 0.0  # USD, for the cost/hour Grafana panel; 0 for self-hosted
    llm_price_out_per_1k: float = 0.0

    # --- monitoring ---
    langfuse_host: str = ""  # e.g. http://localhost:3000 (self-hosted)
    langfuse_public_key: str = Field("", repr=False)  # not secret by design, but never printed
    langfuse_secret_key: str = Field("", repr=False)  # secret: never shown in repr()/logs
    drift_baseline_path: Path = Path("data/eval/questions.json")

    @model_validator(mode="after")
    def _check_consistency(self) -> Settings:
        """Fail fast, in plain language, naming the variable the operator has to fix."""
        problems: list[str] = []
        if self.retriever not in RETRIEVERS:
            problems.append(
                f"PRODML_RETRIEVER={self.retriever!r} is not one of {sorted(RETRIEVERS)}"
            )
        if self.generation_backend not in GENERATION_BACKENDS:
            problems.append(
                f"PRODML_GENERATION_BACKEND={self.generation_backend!r} is not one of {sorted(GENERATION_BACKENDS)}"
            )
        emb = self.embedding_model
        if not (emb.split("-")[0] in {"hashing", "lsa"} or "/" in emb):
            problems.append(
                f"PRODML_EMBEDDING_MODEL={emb!r} must be 'hashing[-DIM]', 'lsa[-DIM]' or a '<org>/<model>' id"
            )
        if self.log_level.upper() not in LOG_LEVELS:
            problems.append(
                f"PRODML_LOG_LEVEL={self.log_level!r} is not one of {sorted(LOG_LEVELS)}"
            )
        if not 0.0 <= self.min_confidence <= 1.0:
            problems.append("PRODML_MIN_CONFIDENCE must be between 0 and 1")
        if not 1 <= self.port <= 65535:
            problems.append("PRODML_PORT must be between 1 and 65535")
        if self.min_lexical_overlap < 0:
            problems.append(
                "PRODML_MIN_LEXICAL_OVERLAP must be >= 0 (0 disables the lexical-support check)"
            )
        if self.top_k < 1:
            problems.append("PRODML_TOP_K must be >= 1")
        if self.generation_backend in {"vllm", "api"}:
            if not self.llm_model:
                problems.append(
                    f"PRODML_LLM_MODEL is required when PRODML_GENERATION_BACKEND={self.generation_backend} "
                    "(the model id your endpoint serves; use the extractive backend for no LLM at all)"
                )
            if not self.llm_base_url.startswith(("http://", "https://")):
                problems.append("PRODML_LLM_BASE_URL must start with http:// or https://")
        langfuse = [self.langfuse_host, self.langfuse_public_key, self.langfuse_secret_key]
        if any(langfuse) and not all(langfuse):
            problems.append(
                "Langfuse is partly configured: set all of PRODML_LANGFUSE_HOST, PRODML_LANGFUSE_PUBLIC_KEY and "
                "PRODML_LANGFUSE_SECRET_KEY, or none of them (tracing is optional)"
            )
        if problems:
            raise ValueError("; ".join(problems))
        return self

    def safe_summary(self) -> dict[str, object]:
        """Configuration for logs and diagnostics: secrets reduced to set/unset."""
        out: dict[str, object] = self.model_dump()
        for secret in ("llm_api_key", "langfuse_secret_key", "langfuse_public_key"):
            out[secret] = "<set>" if out[secret] else "<unset>"
        return out


def load_settings() -> Settings:
    """Build Settings, turning pydantic's traceback into an actionable message (values are never echoed)."""
    try:
        return Settings()
    except ValidationError as exc:
        lines = []
        for err in exc.errors():
            where = ".".join(str(p) for p in err["loc"]) or "configuration"
            lines.append(f"  - {where}: {err['msg'].removeprefix('Value error, ')}")
        raise SystemExit(
            "Invalid prodml configuration (fix your environment or .env; see .env.example):\n"
            + "\n".join(lines)
        ) from None


settings = load_settings()
