"""Ask a question from the command line - no curl, no JSON quoting (works the same on Windows).

prodml-ask "Must contracts be performed in good faith?"          # in-process, no server needed
prodml-ask "هل يجوز تعديل العقد" --url http://localhost:8000      # against a running server / Docker
prodml-ask "..." --url http://localhost:8000 --stream             # watch tokens arrive
"""

from __future__ import annotations

import argparse
import json
import sys

from prodml._console import ensure_utf8_output


def _print_result(d: dict, show_answer: bool = True) -> None:
    if show_answer:
        print(d["answer"])
    if d.get("abstained"):
        print("\n[refused: no article matched confidently enough]")
    else:
        print("\nSources:", "; ".join(d["sources"]))
    print(f"Confidence: {d['confidence']}   Faithfulness proxy: {d.get('faithfulness_proxy')}")
    if d.get("guardrail_flags"):
        print("Guardrail flags:", ", ".join(d["guardrail_flags"]))


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_output()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("question")
    ap.add_argument("--url", help="ask a running server instead of loading the model in-process")
    ap.add_argument(
        "--stream",
        action="store_true",
        help="use /ask/stream and print tokens as they arrive (needs --url)",
    )
    args = ap.parse_args(argv)

    if args.stream and not args.url:
        ap.error("--stream needs --url (start the server first)")

    if args.url:
        import httpx

        base = args.url.rstrip("/")
        try:
            if args.stream:
                with httpx.stream(
                    "POST", f"{base}/ask/stream", json={"question": args.question}, timeout=60
                ) as r:
                    r.raise_for_status()
                    for line in r.iter_lines():
                        if line.startswith("data: "):
                            ev = json.loads(line[6:])
                            if ev["type"] == "token":
                                print(ev["text"], end="", flush=True)
                            elif ev["type"] == "replace":
                                print("\n[answer replaced by guardrail] " + ev["text"], end="")
                            elif ev["type"] == "done":
                                print()
                                _print_result(ev, show_answer=False)  # tokens were already printed
                return 0
            r = httpx.post(f"{base}/ask", json={"question": args.question}, timeout=60)
            r.raise_for_status()
            _print_result(r.json())
        except httpx.HTTPError as exc:
            print(f"Could not get an answer from {base}: {exc}", file=sys.stderr)
            return 1
        return 0

    import logging

    logging.disable(logging.CRITICAL)
    from prodml.rag import LegalQAPredictor

    _print_result(LegalQAPredictor().load().query_one(args.question))
    return 0


if __name__ == "__main__":
    sys.exit(main())
