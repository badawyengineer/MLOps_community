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
