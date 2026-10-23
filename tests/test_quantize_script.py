"""Only the calibration-data builder of scripts/quantize_awq.py is testable without a GPU."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "quantize_awq.py"


def _load():
    spec = importlib.util.spec_from_file_location("quantize_awq", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # importing must not pull in torch/awq
    return mod


def _write(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "articles.json"
    p.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return p


def test_calibration_skips_repealed_and_is_deterministic(tmp_path) -> None:
    mod = _load()
    rows = [
        {
            "article_number": i,
            "text_ar": f"نص {i}",
            "text_en": f"text {i}",
            "is_repealed": i % 10 == 0,
        }
        for i in range(1, 301)
    ]
    path = _write(tmp_path, rows)
    a, b = mod.calibration_texts(path, n=20), mod.calibration_texts(path, n=20)
    assert a == b and len(a) == 20
    assert all("نص" in t and "text" in t for t in a)


def test_calibration_truncates_long_articles(tmp_path) -> None:
    mod = _load()
    rows = [
        {"article_number": 1, "text_ar": "ا" * 5000, "text_en": "x" * 5000, "is_repealed": False}
    ]
    assert len(mod.calibration_texts(_write(tmp_path, rows), n=5, max_chars=1500)[0]) == 1500


def test_real_corpus_gives_requested_sample_count() -> None:
    mod = _load()
    assert len(mod.calibration_texts(Path("data/processed/articles.json"), n=128)) == 128


def test_retune_gates_sit_just_below_measured_values() -> None:
    spec = importlib.util.spec_from_file_location(
        "retune_gates", Path(__file__).resolve().parents[1] / "scripts" / "retune_gates.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    gates = mod.derive_gates(
        {
            "hit_at_k": 0.598,
            "mrr": 0.598,
            "correct_abstain_rate": 0.526,
            "false_abstain_rate": 0.108,
        },
        0.04,
    )
    assert gates == {
        "hit_at_k": 0.56,
        "mrr": 0.56,
        "correct_abstain_rate": 0.49,
        "max_false_abstain_rate": 0.15,
    }
