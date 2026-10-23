from __future__ import annotations

import json
import threading
import urllib.request
from http.server import HTTPServer

from prodml.alert_sink import format_alert, make_handler

ALERT = {
    "status": "firing",
    "labels": {"alertname": "FaithfulnessLow", "severity": "critical"},
    "annotations": {"summary": "below 0.80"},
}


def test_format_alert() -> None:
    assert format_alert(ALERT) == "[FIRING] FaithfulnessLow (critical) below 0.80"


def test_webhook_receiver_appends_jsonl(tmp_path) -> None:
    log = tmp_path / "alerts.jsonl"
    server = HTTPServer(("127.0.0.1", 0), make_handler(log))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{server.server_port}/alerts",
            data=json.dumps({"alerts": [ALERT]}).encode(),
            headers={"Content-Type": "application/json"},
        )
        assert urllib.request.urlopen(req, timeout=5).status == 200
    finally:
        server.shutdown()
    row = json.loads(log.read_text(encoding="utf-8").splitlines()[0])
    assert row["labels"]["alertname"] == "FaithfulnessLow" and "received_at" in row
