from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "Arabic Legal Document Q&A")
    app_version: str = os.getenv("APP_VERSION", "0.1.0")
    environment: str = os.getenv("ENVIRONMENT", "development")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    model_version: str = os.getenv("MODEL_VERSION", "session1-mock-v1")
    max_question_length: int = int(os.getenv("MAX_QUESTION_LENGTH", "2000"))


settings = Settings()
