"""Per-request tracing, exported to Langfuse over its public REST ingestion API.

Self-host Langfuse (see docs/monitoring.md), set PRODML_LANGFUSE_HOST / _PUBLIC_KEY /
_SECRET_KEY, and every /ask produces a trace with a retrieval span, a generation (with token
usage) and a faithfulness score attached. With no keys configured this is a no-op, so tests
and CI never need a Langfuse server. Export failures are swallowed: tracing must never break
a user request.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import Any

import httpx

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class Tracer:
    def __init__(
        self,
        host: str = "",
        public_key: str = "",
        secret_key: str = "",
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.enabled = bool(host and public_key and secret_key)
        self._client = (
            httpx.Client(
                base_url=host.rstrip("/"),
                auth=(public_key, secret_key),
                timeout=5.0,
                transport=transport,
            )
            if self.enabled
            else None
        )

    def start(self, name: str, user_input: str) -> Trace:
        return Trace(self, name, user_input)

    def _send(self, batch: list[dict[str, Any]]) -> None:
        if not self._client:
            return
        try:
            self._client.post("/api/public/ingestion", json={"batch": batch}).raise_for_status()
        except Exception as exc:  # noqa: BLE001 - tracing must never break a request
            logger.warning("langfuse export failed: %s", exc)


class Trace:
    def __init__(self, tracer: Tracer, name: str, user_input: str) -> None:
        self.id = uuid.uuid4().hex
        self._tracer = tracer
        self._events: list[dict[str, Any]] = [
            self._event(
                "trace-create",
                {"id": self.id, "name": name, "input": user_input, "timestamp": _now()},
            )
        ]

    @staticmethod
    def _event(kind: str, body: dict[str, Any]) -> dict[str, Any]:
        return {"id": uuid.uuid4().hex, "type": kind, "timestamp": _now(), "body": body}

    def span(self, name: str, start: str, end: str, **meta: Any) -> None:
        self._events.append(
            self._event(
                "span-create",
                {
                    "traceId": self.id,
                    "name": name,
                    "startTime": start,
                    "endTime": end,
                    "metadata": meta,
                },
            )
        )

    def generation(
        self, name: str, model: str, output: str, prompt_tokens: int, completion_tokens: int
    ) -> None:
        self._events.append(
            self._event(
                "generation-create",
                {
                    "traceId": self.id,
                    "name": name,
                    "model": model,
                    "output": output,
                    "usage": {
                        "input": prompt_tokens,
                        "output": completion_tokens,
                        "unit": "TOKENS",
                    },
                },
            )
        )

    def score(self, name: str, value: float) -> None:
        self._events.append(
            self._event("score-create", {"traceId": self.id, "name": name, "value": value})
        )

    def finish(self, output: str) -> None:
        self._events.append(self._event("trace-create", {"id": self.id, "output": output}))
        self._tracer._send(self._events)

    @property
    def events(self) -> list[dict[str, Any]]:
        return self._events


def now() -> str:
    return _now()
