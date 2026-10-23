"""A MOCK OpenAI-compatible LLM server, for running the real LLM code path with no GPU and no API key.

    prodml-mock-llm --port 8001 --model Qwen/Qwen2.5-1.5B-Instruct --latency-ms 60

Then point the service at it:

    PRODML_GENERATION_BACKEND=vllm  PRODML_LLM_BASE_URL=http://localhost:8001/v1  PRODML_LLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct

This is NOT a language model. It answers deterministically by quoting the first retrieved article from the
prompt's context, and every answer starts with ``MOCK_PREFIX`` so it can never be mistaken for model output.
It exists to exercise the HTTP client, streaming, token accounting, guardrails, metrics and fallbacks.
Never report its latency, quality or token numbers as results of a real model.
"""

from __future__ import annotations

import argparse
import json
import re
import time

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse

MOCK_PREFIX = "[MOCK LLM - deterministic stub, not a language model]"
_ARTICLE = re.compile(r"\[Article (\d+)\]\s*(.*?)(?=\n\n\[Article \d+\]|\Z)", re.DOTALL)


def make_answer(user_content: str) -> str:
    """Quote the first context article (English line) with its citation; refuse if there is none."""
    m = _ARTICLE.search(user_content)
    if not m:
        return f"{MOCK_PREFIX} No context article was provided."
    number, body = m.group(1), m.group(2).strip().splitlines()[0]
    return f"{MOCK_PREFIX} According to [Article {number}]: {body[:400]}"


def _tokens(text: str) -> int:
    return len(text.split())


def create_app(model: str, latency_ms: float = 0.0) -> FastAPI:
    app = FastAPI(title="prodml mock LLM (NOT a real model)")

    @app.get("/v1/models")
    def models() -> dict:
        return {"object": "list", "data": [{"id": model, "object": "model", "owned_by": "mock"}]}

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "mock": True}

    @app.post("/v1/chat/completions")
    def chat(body: dict):
        time.sleep(latency_ms / 1000)
        messages = body.get("messages") or []
        user = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        answer = make_answer(user)
        usage = {
            "prompt_tokens": sum(_tokens(m.get("content", "")) for m in messages),
            "completion_tokens": _tokens(answer),
        }
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
        base = {
            "id": "mock-1",
            "object": "chat.completion",
            "model": model,
            "created": int(time.time()),
        }
        if not body.get("stream"):
            return JSONResponse(
                {
                    **base,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": answer},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": usage,
                }
            )

        def events():
            chunk = {**base, "object": "chat.completion.chunk"}
            for word in answer.split(" "):
                delta = {"index": 0, "delta": {"content": word + " "}, "finish_reason": None}
                yield f"data: {json.dumps({**chunk, 'choices': [delta]})}\n\n"
            if (body.get("stream_options") or {}).get("include_usage"):
                yield f"data: {json.dumps({**chunk, 'choices': [], 'usage': usage})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    return app


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--model", default="mock-model")
    ap.add_argument("--latency-ms", type=float, default=0.0)
    args = ap.parse_args(argv)
    import uvicorn

    print(f"{MOCK_PREFIX} serving model id {args.model!r} on http://{args.host}:{args.port}/v1")
    uvicorn.run(
        create_app(args.model, args.latency_ms), host=args.host, port=args.port, log_level="warning"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
