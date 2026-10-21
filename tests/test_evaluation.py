from __future__ import annotations

import json

import pytest

from prodml import evaluation as ev
from prodml.corpus import sample_corpus
from prodml.retriever import build_retriever

ITEMS = [
    {
        "id": "a",
        "question": "good faith in contracts",
        "relevant_articles": [148],
        "answerable": True,
    },
    {"id": "b", "question": "fault causing damage", "relevant_articles": [163], "answerable": True},
    {"id": "x", "question": "zzz qqq", "relevant_articles": [], "answerable": False},
]


@pytest.fixture(scope="module")
def retriever():
    return build_retriever("hybrid").fit(sample_corpus())


def test_metrics_perfect_on_easy_set(retriever) -> None:
    r = ev.evaluate(retriever, ITEMS, k=1, threshold=0.0)
    assert r["hit_at_k"] == 1.0 and r["mrr"] == 1.0
    assert r["n_answerable"] == 2 and r["n_unanswerable"] == 1


def test_high_threshold_causes_false_abstentions(retriever) -> None:
    r = ev.evaluate(retriever, ITEMS, k=1, threshold=0.99)
    assert r["false_abstain_rate"] == 1.0 and r["correct_abstain_rate"] == 1.0


def test_wrong_retrieval_is_recorded_as_miss(retriever) -> None:
    bad = [{"id": "m", "question": "good faith", "relevant_articles": [163], "answerable": True}]
    r = ev.evaluate(retriever, bad, k=1)
    assert r["retrieval_misses"] == ["m"] and r["mrr"] == 0.0


def test_sweep_and_best_threshold(retriever) -> None:
    sweep = ev.sweep_thresholds(retriever, ITEMS, k=1, grid=[0.0, 0.2, 0.99])
    assert [s["threshold"] for s in sweep] == [0.0, 0.2, 0.99]
    best = ev.best_threshold(sweep)
    assert best["false_abstain_rate"] == 0.0  # 0.99 would refuse everything


def test_gates_pass_and_fail() -> None:
    result = {"mrr": 0.8, "hit_at_k": 0.9, "false_abstain_rate": 0.2}
    assert ev.check_gates(result, {"mrr": 0.7, "max_false_abstain_rate": 0.3}) == []
    fails = ev.check_gates(result, {"mrr": 0.9, "max_false_abstain_rate": 0.1, "_note": "x"})
    assert len(fails) == 2


def test_eval_set_rejects_inconsistent_labels(tmp_path) -> None:
    p = tmp_path / "q.json"
    p.write_text(
        json.dumps(
            {"items": [{"id": "z", "question": "q", "relevant_articles": [1], "answerable": False}]}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="disagrees"):
        ev.load_eval_set(p)


def test_shipped_eval_set_is_valid_and_balanced() -> None:
    items = ev.load_eval_set()
    assert sum(i["answerable"] for i in items) >= 10
    assert sum(not i["answerable"] for i in items) >= 5
    assert {i["lang"] for i in items} == {"ar", "en"}


def test_cli_gate_exit_codes(tmp_path) -> None:
    ok = tmp_path / "ok.json"
    ok.write_text(json.dumps({"mrr": 0.5}), encoding="utf-8")
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"mrr": 1.01}), encoding="utf-8")
    out = tmp_path / "r.json"
    assert (
        ev.main(["--retriever", "hybrid", "--k", "1", "--check", str(ok), "--out", str(out)]) == 0
    )
    assert out.exists()
    assert ev.main(["--retriever", "hybrid", "--k", "1", "--check", str(bad)]) == 1


def test_cli_compare_with_sweep(capsys) -> None:
    assert ev.main(["--compare", "--sweep", "--k", "1"]) == 0
    out = capsys.readouterr().out
    assert "tfidf" in out and "bm25" in out and "hybrid" in out and "best threshold" in out


def test_gate_applies_only_to_shipped_retriever_in_compare_mode(tmp_path) -> None:
    gates = tmp_path / "g.json"
    gates.write_text(
        json.dumps({"mrr": 0.6}), encoding="utf-8"
    )  # on the real corpus hybrid clears this (0.67), tfidf does not (0.54)
    assert (
        ev.main(
            [
                "--compare",
                "--retriever",
                "hybrid",
                "--k",
                "1",
                "--threshold",
                "0.2",
                "--check",
                str(gates),
            ]
        )
        == 0
    )
    assert (
        ev.main(
            [
                "--compare",
                "--retriever",
                "tfidf",
                "--k",
                "1",
                "--threshold",
                "0.2",
                "--check",
                str(gates),
            ]
        )
        == 1
    )


def test_gate_targets_the_shipped_retriever_even_if_it_is_not_an_offline_baseline(tmp_path) -> None:
    """Regression: with --compare and a shipped retriever outside {tfidf, bm25, hybrid}, the gate
    used to fall back to the first baseline (tfidf) and fail CI for the wrong model."""
    gates = tmp_path / "g.json"
    gates.write_text(
        json.dumps({"mrr": 0.5}), encoding="utf-8"
    )  # tfidf is below, hybrid-dense above
    out = tmp_path / "r.json"
    code = ev.main(
        [
            "--compare",
            "--retriever",
            "hybrid-dense",
            "--k",
            "1",
            "--threshold",
            "0.3",
            "--check",
            str(gates),
            "--out",
            str(out),
        ]
    )
    assert code == 0
    assert "hybrid-dense" in json.loads(out.read_text(encoding="utf-8"))
