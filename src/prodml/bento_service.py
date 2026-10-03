"""BentoML serving layer (handbook Module 3).

Same predictor as the FastAPI app, exposed through BentoML so it gets adaptive batching,
resource declarations, `bentoml build` / `containerize`, and async streaming:

    bentoml serve prodml.bento_service:LegalQAService --port 3000
    curl -X POST localhost:3000/ask -H 'Content-Type: application/json' -d '{"question": "..."}'
    curl -N -X POST localhost:3000/ask_stream -H 'Content-Type: application/json' -d '{"question": "..."}'
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Any

import bentoml

from prodml.rag import LegalQAPredictor


@bentoml.service(
    name="legal_qa",
    resources={"cpu": "1"},
    traffic={"timeout": 60, "concurrency": 32},
)
class LegalQAService:
    def __init__(self) -> None:
        # loaded once per worker at startup, never per request
        self.predictor = LegalQAPredictor().load()

    @bentoml.api
    def ask(self, question: str) -> dict[str, Any]:
        """Answer one question. Sync handler: BentoML runs it in a worker thread."""
        return {k: v for k, v in self.predictor.query_one(question).items() if k != "usage"}

    @bentoml.api
    async def ask_stream(self, question: str) -> AsyncGenerator[str, None]:
        """Server-sent tokens. Retrieval/generation is synchronous, so each event is produced
        in a thread and handed to the event loop."""
        import asyncio

        events = iter(self.predictor.stream_events(question))
        sentinel = object()
        while True:
            event = await asyncio.to_thread(next, events, sentinel)
            if event is sentinel:
                return
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    @bentoml.api
    def health(self) -> dict[str, Any]:
        return {
            "status": "healthy" if self.predictor.is_loaded else "loading",
            "documents_indexed": self.predictor.metadata()["articles_indexed"],
        }
