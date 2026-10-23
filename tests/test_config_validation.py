"""Configuration fails fast with a message naming the variable, and never echoes secret values."""

from __future__ import annotations

import subprocess
import sys

import pytest
from pydantic import ValidationError

from prodml.config import Settings


def _make(**kw) -> Settings:
    return Settings(_env_file=None, **kw)


def test_defaults_are_valid_and_offline() -> None:
    s = _make()
    assert s.generation_backend == "extractive" and s.llm_api_key == ""


@pytest.mark.parametrize(
    ("kw", "needle"),
    [
        ({"retriever": "nope"}, "PRODML_RETRIEVER"),
        ({"generation_backend": "api"}, "PRODML_LLM_MODEL"),
        (
            {"generation_backend": "vllm", "llm_model": "m", "llm_base_url": "localhost:8001"},
            "http://",
        ),
        ({"min_confidence": 1.5}, "PRODML_MIN_CONFIDENCE"),
        ({"port": 70000}, "PRODML_PORT"),
        ({"embedding_model": "bogus"}, "PRODML_EMBEDDING_MODEL"),
        ({"langfuse_host": "http://x"}, "Langfuse is partly configured"),
        ({"log_level": "LOUD"}, "PRODML_LOG_LEVEL"),
    ],
)
def test_invalid_values_are_rejected_with_the_variable_name(kw: dict, needle: str) -> None:
    with pytest.raises(ValidationError, match=needle):
        _make(**kw)


def test_secrets_are_hidden_from_repr_and_summary() -> None:
    s = _make(
        llm_api_key="SUPERSECRET-llm",
        langfuse_host="http://l",
        langfuse_public_key="pk-x",
        langfuse_secret_key="SUPERSECRET-lf",
    )
    assert "SUPERSECRET" not in repr(s) and "SUPERSECRET" not in str(s)
    summary = s.safe_summary()
    assert "SUPERSECRET" not in str(summary) and summary["llm_api_key"] == "<set>"


def test_bad_environment_stops_startup_without_leaking_values() -> None:
    env = {
        "PRODML_GENERATION_BACKEND": "api",
        "PRODML_LLM_API_KEY": "SUPERSECRET-value",
        "PATH": "",
    }
    r = subprocess.run(
        [sys.executable, "-c", "import prodml.config"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode != 0
    assert "PRODML_LLM_MODEL" in r.stderr and "SUPERSECRET" not in r.stderr + r.stdout
    assert "Traceback" not in r.stderr  # a readable message, not a stack dump
