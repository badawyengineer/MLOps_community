"""Faithfulness measurement.

``faithfulness_proxy`` is a cheap, dependency-free, OFFLINE check: what fraction of the
answer's sentences are lexically supported by the retrieved context? It is deliberately
named *proxy* - it is NOT RAGAS faithfulness (which uses an LLM judge) and the two must
never be reported under the same name. Because the default extractive generator copies the
article text verbatim, the proxy is trivially ~1.0 until a real generator is plugged in.

``ragas_scores`` runs real RAGAS when the optional extra and a judge LLM are configured.
"""

from __future__ import annotations

import re

from prodml.retriever import tokenize

_CITATION_TAG = re.compile(r"\[[^\]]*Article[^\]]*\]|\[[^\]]*مادة[^\]]*\]", re.IGNORECASE)
_SENT = re.compile(r"(?<=[.!?؟])\s+|\n+")


def faithfulness_proxy(answer: str, contexts: list[str], support_threshold: float = 0.6) -> float:
    """Fraction of answer sentences whose content tokens are >= threshold covered by the context."""
    context_tokens = set(tokenize(" ".join(contexts)))
    sentences = [s for s in _SENT.split(_CITATION_TAG.sub(" ", answer)) if tokenize(s)]
    if not sentences:
        return 0.0
    supported = 0
    for s in sentences:
        toks = tokenize(s)
        if sum(t in context_tokens for t in toks) / len(toks) >= support_threshold:
            supported += 1
    return supported / len(sentences)


def ragas_scores(
    rows: list[dict],
) -> dict[str, float]:  # pragma: no cover - needs ragas + a judge LLM
    """rows: [{"question","answer","contexts":[...],"ground_truth"}]. Returns the 4 RAGAS metrics.

    Written against the ragas 0.2-style ``evaluate`` API; ragas changes quickly, so pin the
    version you run and check the ragas docs if this raises. Needs a judge LLM (e.g.
    OPENAI_API_KEY) and costs real API calls.
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
    except ImportError as exc:
        raise RuntimeError('pip install -e ".[ragas]"  (and configure a judge LLM)') from exc
    ds = Dataset.from_list(
        [
            {
                "question": r["question"],
                "answer": r["answer"],
                "contexts": r["contexts"],
                "ground_truth": r["ground_truth"],
            }
            for r in rows
        ]
    )
    result = evaluate(
        ds, metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
    )
    return {k: float(v) for k, v in result.items()}
