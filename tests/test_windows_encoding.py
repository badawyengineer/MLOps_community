"""Regression tests for the Windows failure mode: the default text encoding is NOT UTF-8 there
(cp1252), so any ``read_text()`` / ``print()`` of Arabic that omits an explicit encoding crashes.

We reproduce it on any OS by running the CLIs in a subprocess with UTF-8 mode disabled and an
ASCII locale (stricter than cp1252, so it catches everything cp1252 would)."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "bilingual_sample.pdf"
LEGACY_ENV = {
    **os.environ,
    "PYTHONUTF8": "0",
    "PYTHONCOERCECLOCALE": "0",
    "LC_ALL": "C",
    "LANG": "C",
}


def run_cli(*args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [sys.executable, "-m", *args],
        cwd=ROOT,
        env=LEGACY_ENV,
        capture_output=True,
        check=False,
        timeout=120,
    )


def test_legacy_environment_really_is_non_utf8() -> None:
    out = subprocess.run(
        [sys.executable, "-c", "import locale;print(locale.getpreferredencoding(False))"],
        env=LEGACY_ENV,
        capture_output=True,
        check=False,
    )
    if sys.platform != "win32":  # on Windows PYTHONUTF8=0 -> cp1252, also non-UTF-8
        assert b"UTF-8" not in out.stdout.upper()


def test_spot_check_prints_arabic_under_a_legacy_encoding() -> None:
    r = run_cli("prodml.validate_corpus", "--spot-check", "2")
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")[-400:]
    assert any(
        "\u0600" <= ch <= "\u06ff" for ch in r.stdout.decode("utf-8")
    )  # Arabic made it out, as UTF-8


def test_ingest_dump_prints_arabic_under_a_legacy_encoding() -> None:
    r = run_cli("prodml.ingest", str(FIXTURE), "--dump", "1")
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")[-400:]
    assert "ARABIC COLUMN" in r.stdout.decode("utf-8")


def test_smoke_passes_under_a_legacy_encoding() -> None:
    """Exercises the Arabic corpus, the thresholds JSON read and the eval-set read in one go."""
    r = run_cli("prodml.smoke")
    assert r.returncode == 0, (r.stdout + r.stderr).decode("utf-8", "replace")[-600:]
    assert b"12/12 checks passed" in r.stdout


def test_eval_gate_reads_thresholds_under_a_legacy_encoding() -> None:
    r = run_cli(
        "prodml.evaluation",
        "--retriever",
        "hybrid-dense",
        "--k",
        "1",
        "--threshold",
        "0.3",
        "--check",
        "data/eval/thresholds.json",
    )
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")[-400:]


@pytest.mark.parametrize("folder", ["src", "tests"])
def test_no_text_file_io_without_explicit_encoding(folder: str) -> None:
    """Static guard: every read_text()/write_text() must say encoding=... (checked by parsing)."""
    offenders = []
    for f in (ROOT / folder).rglob("*.py"):
        for n in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if (
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr in ("read_text", "write_text")
                and not any(k.arg == "encoding" for k in n.keywords)
            ):
                offenders.append(f"{f.relative_to(ROOT)}:{n.lineno}")
    assert not offenders, f"text I/O without encoding (breaks on Windows): {offenders}"
