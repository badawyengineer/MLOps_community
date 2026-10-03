"""Client for any OpenAI-compatible chat-completions endpoint.

That one protocol covers the project's serving options: a self-hosted **vLLM** server
(``vllm serve <model>`` exposes /v1/chat/completions), or any hosted API that speaks it.
Plain ``httpx`` - no vendor SDK - so it is trivially mockable and has no extra dependency.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass

import httpx

SYSTEM_PROMPT = (
    "You are a careful legal research assistant for the Egyptian Civil Code. "
    "Answer ONLY from the numbered articles provided in the context. "
    "Cite every claim as [Article N]. Never cite an article that is not in the context. "
    "If the context does not contain the answer, say you cannot answer from the provided articles. "
    "Reply in the same language as the question. This is informational, not legal advice."
)


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLMError(RuntimeError):
    pass


def build_messages(question: str, context: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context articles:\n{context}\n\nQuestion: {question}"},
    ]


class OpenAICompatClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout: float = 30.0,
        max_tokens: int = 400,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.model, self.max_tokens = model, max_tokens
        self._http = httpx.Client(
            base_url=base_url.rstrip("/"), headers=headers, timeout=timeout, transport=transport
        )
        self.last_usage = LLMResult("")

    def _payload(self, messages: list[dict[str, str]], stream: bool) -> dict:
        body: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "stream": stream,
        }
        if stream:
            body["stream_options"] = {"include_usage": True}
        return body

    def complete(self, messages: list[dict[str, str]]) -> LLMResult:
        try:
            r = self._http.post("/chat/completions", json=self._payload(messages, False))
            r.raise_for_status()
            data = r.json()
            usage = data.get("usage") or {}
            return LLMResult(
                data["choices"][0]["message"]["content"] or "",
                usage.get("prompt_tokens", 0),
                usage.get("completion_tokens", 0),
            )
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
            raise LLMError(f"LLM call failed: {exc}") from exc

    def stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        """Yield text deltas as they arrive; token usage lands in ``self.last_usage`` at the end."""
        parts: list[str] = []
        usage: dict = {}
        try:
            with self._http.stream(
                "POST", "/chat/completions", json=self._payload(messages, True)
            ) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    chunk = line[5:].strip()
                    if chunk == "[DONE]":
                        break
                    event = json.loads(chunk)
                    usage = event.get("usage") or usage
                    for choice in event.get("choices", []):
                        delta = (choice.get("delta") or {}).get("content")
                        if delta:
                            parts.append(delta)
                            yield delta
        except (httpx.HTTPError, ValueError) as exc:
            raise LLMError(f"LLM stream failed: {exc}") from exc
        self.last_usage = LLMResult(
            "".join(parts), usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
        )
