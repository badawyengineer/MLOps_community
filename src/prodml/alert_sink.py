"""A local Alertmanager webhook receiver, plus a test-alert sender. Stdlib only.

    prodml-alert-sink                                          # listen on :5001, print + append to reports/alerts.jsonl
    prodml-alert-sink --fire-test-alert http://localhost:9093  # POST a FaithfulnessLow alert to Alertmanager

docker/observability/alertmanager.yml points its webhook at http://host.docker.internal:5001/alerts.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


def format_alert(alert: dict) -> str:
    labels, ann = alert.get("labels", {}), alert.get("annotations", {})
    status = str(alert.get("status", "firing")).upper()
    return f"[{status}] {labels.get('alertname', '?')} ({labels.get('severity', '?')}) {ann.get('summary', '')}".rstrip()


def make_handler(log_path: Path) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self.send_response(400)
                self.end_headers()
                return
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as fh:
                for alert in payload.get("alerts", []):
                    print(format_alert(alert), flush=True)
                    fh.write(
                        json.dumps(
                            {"received_at": datetime.now(timezone.utc).isoformat(), **alert},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args: object) -> None:  # keep the console for alerts only
            return

    return Handler


def fire_test_alert(alertmanager_url: str) -> int:
    body = json.dumps(
        [
            {
                "labels": {
                    "alertname": "FaithfulnessLow",
                    "severity": "critical",
                    "source": "test-alert",
                },
                "annotations": {
                    "summary": "TEST alert fired by prodml-alert-sink --fire-test-alert (not a real incident)"
                },
            }
        ]
    ).encode()
    req = urllib.request.Request(
        alertmanager_url.rstrip("/") + "/api/v2/alerts",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        print(
            f"Alertmanager answered HTTP {resp.status}; it will deliver after its group_wait (~10 s)."
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5001)
    ap.add_argument("--log", type=Path, default=Path("reports/alerts.jsonl"))
    ap.add_argument("--fire-test-alert", metavar="ALERTMANAGER_URL")
    args = ap.parse_args(argv)
    if args.fire_test_alert:
        try:
            return fire_test_alert(args.fire_test_alert)
        except OSError as exc:
            print(f"could not reach Alertmanager at {args.fire_test_alert}: {exc}", file=sys.stderr)
            return 1
    print(
        f"listening on {args.host}:{args.port}, appending to {args.log} (Ctrl+C to stop)",
        flush=True,
    )
    HTTPServer((args.host, args.port), make_handler(args.log)).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
