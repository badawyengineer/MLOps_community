from __future__ import annotations

from prodml import smoke


def test_smoke_in_process_passes_everything(capsys) -> None:
    assert smoke.main([]) == 0
    out = capsys.readouterr().out
    assert "12/12 checks passed" in out and "FAIL" not in out


def test_smoke_live_mode_against_test_client_skips_in_process_only_checks(capsys) -> None:
    from fastapi.testclient import TestClient

    from prodml.api.main import app

    with TestClient(app) as client:
        assert smoke.run(client, live=True) == 0
    assert "9/9 checks passed" in capsys.readouterr().out


def test_smoke_reports_failure_and_exits_nonzero(capsys) -> None:
    class Broken:
        def get(self, *a, **k):
            raise ConnectionError("server down")

        post = get

    assert smoke.run(Broken(), live=True) == 1
    out = capsys.readouterr().out
    assert "FAIL" in out and "ConnectionError" in out and "FAILURES ABOVE" in out


def test_ask_cli_in_process_answers_and_cites(capsys) -> None:
    from prodml import ask

    assert ask.main(["Must contracts be performed in good faith?"]) == 0
    out = capsys.readouterr().out
    assert "Article 148" in out and "Confidence:" in out


def test_ask_cli_refuses_out_of_scope(capsys) -> None:
    from prodml import ask

    assert ask.main(["كيف أطبخ الكشري؟"]) == 0
    assert "refused" in capsys.readouterr().out


def test_ask_cli_reports_unreachable_server(capsys) -> None:
    from prodml import ask

    assert ask.main(["anything at all", "--url", "http://127.0.0.1:9"]) == 1
    assert "Could not get an answer" in capsys.readouterr().err
