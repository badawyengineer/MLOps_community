from __future__ import annotations

import json

import httpx
import pytest
from prometheus_client import generate_latest

from prodml import compare_llms as cmp
from prodml import monitor
from prodml.evaluation import load_eval_set
from prodml.llm import OpenAICompatClient
from prodml.rag import LegalQAPredictor


@pytest.fixture(scope="module")
def predictor():
    return LegalQAPredictor().load()


def test_collect_scores_eval_set_with_proxy(predictor) -> None:
    report = monitor.collect(predictor, load_eval_set())
    assert report["source"] == "proxy" and report["n_answered"] > 10
    assert 0.9 <= report["faithfulness"] <= 1.0  # extractive answers quote their own context
    assert report["false_abstain_rate"] == 0.0


def test_push_builds_labelled_gauges_and_calls_gateway(monkeypatch) -> None:
    pushed = {}
    monkeypatch.setattr(
        monitor, "push_to_gateway", lambda url, job, registry: pushed.update(url=url, job=job)
    )
    reg = monitor.push(
        {"source": "proxy", "faithfulness": 0.93, "false_abstain_rate": 0.0}, "http://pg:9091"
    )
    text = generate_latest(reg).decode()
    assert 'prodml_eval_faithfulness{source="proxy"} 0.93' in text
    assert pushed == {"url": "http://pg:9091", "job": "prodml-eval"}


def test_monitor_cli_gate_and_report_file(tmp_path, monkeypatch) -> None:
    out = tmp_path / "r.json"
    assert monitor.main(["--out", str(out), "--min-faithfulness", "0.5"]) == 0
    assert json.loads(out.read_text())["source"] == "proxy"
    assert monitor.main(["--out", str(out), "--min-faithfulness", "1.01"]) == 1


def _client(answer_for_b: str, latency_text: str = "ok") -> OpenAICompatClient:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": answer_for_b}}],
                "usage": {"prompt_tokens": 40, "completion_tokens": 10},
            },
        )

    return OpenAICompatClient("http://x/v1", "m", transport=httpx.MockTransport(handler))


def test_compare_flags_faithfulness_drop_over_budget(predictor) -> None:
    items = load_eval_set()
    faithful = cmp.run_endpoint(
        _client("A contract must be performed in good faith. [Article 148]"), predictor, items
    )
    hallucinating = cmp.run_endpoint(
        _client("Imprisonment for ten years applies to every breach."), predictor, items
    )
    assert faithful["n"] == hallucinating["n"] > 10
    assert faithful["faithfulness_proxy"] > hallucinating["faithfulness_proxy"]
    table = cmp.render_table(faithful, hallucinating, "base", "quant")
    assert "EXCEEDS" in table and "latency p95" in table and "tokens / s" in table


def test_compare_within_budget_for_identical_models(predictor) -> None:
    items = load_eval_set()
    a = cmp.run_endpoint(_client("Contracts bind the parties [Article 147]."), predictor, items)
    b = cmp.run_endpoint(_client("Contracts bind the parties [Article 147]."), predictor, items)
    assert "within" in cmp.render_table(a, b, "a", "b")


def test_monitor_logs_each_pass_to_mlflow(tmp_path, monkeypatch) -> None:
    pytest.importorskip("mlflow")
    import mlflow

    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/m.db")
    for _ in range(2):  # two passes = a (tiny) trend
        assert monitor.main(["--mlflow", "--out", str(tmp_path / "r.json")]) == 0
    runs = mlflow.search_runs(experiment_names=["legal-qa-monitoring"])
    assert (
        len(runs) == 2
        and (runs["metrics.faithfulness"] > 0.9).all()
        and (runs["params.scorer"] == "proxy").all()
    )


def test_experiment_chart_is_written(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    from prodml import experiments as ex

    out = tmp_path / "chart.png"
    ex.plot_runs(ex.run_grid(["hashing-256"]), out)
    assert out.stat().st_size > 5000
