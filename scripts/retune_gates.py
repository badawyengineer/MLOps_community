"""Re-derive the CI regression gates (data/eval/thresholds.json) from a fresh evaluation.

    prodml-eval --compare --sweep --retriever hybrid-dense --k 1 --threshold 0.25 --out reports/eval/latest.json
    python scripts/retune_gates.py --retriever hybrid-dense --threshold 0.25 [--margin 0.04] [--write]

Prints the measured values, the abstention threshold the sweep found best (balanced accuracy -
set PRODML_MIN_CONFIDENCE / config.py to it and re-run the eval at that value before --write),
and the gates "just below" the measurement. Nothing is written without --write.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def derive_gates(result: dict, margin: float) -> dict[str, float]:
    """Gates sit ``margin`` below measured quality and ``margin`` above the measured false-abstain
    rate, so noise does not fail CI but a real regression does."""
    return {
        "hit_at_k": round(max(0.0, result["hit_at_k"] - margin), 2),
        "mrr": round(max(0.0, result["mrr"] - margin), 2),
        "correct_abstain_rate": round(max(0.0, result["correct_abstain_rate"] - margin), 2),
        "max_false_abstain_rate": round(min(1.0, result["false_abstain_rate"] + margin), 2),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--report", type=Path, default=Path("reports/eval/latest.json"))
    ap.add_argument("--gates", type=Path, default=Path("data/eval/thresholds.json"))
    ap.add_argument("--retriever", default="hybrid-dense")
    ap.add_argument(
        "--threshold", type=float, required=True, help="abstention threshold the eval ran at"
    )
    ap.add_argument("--margin", type=float, default=0.04)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)

    result = json.loads(args.report.read_text(encoding="utf-8"))[args.retriever]
    gates = derive_gates(result, args.margin)
    print(f"measured ({args.retriever}, threshold {args.threshold}):")
    for k in ("hit_at_k", "mrr", "false_abstain_rate", "correct_abstain_rate"):
        print(f"  {k:22} {result[k]:.3f}")
    if "best_threshold" in result:
        print(f"sweep's best abstention threshold (balanced accuracy): {result['best_threshold']}")
    print("proposed gates:", gates)
    if args.write:
        note = (
            f"CI regression gates for `prodml-eval --retriever {args.retriever} --k 1 --threshold {args.threshold}`. "
            f"Derived by scripts/retune_gates.py from a measured run (hit@1 {result['hit_at_k']:.3f}, MRR {result['mrr']:.3f}, "
            f"correct-abstain {result['correct_abstain_rate']:.3f}, false-abstain {result['false_abstain_rate']:.3f}) with a "
            f"{args.margin} margin. Raise them only when retrieval genuinely improves."
        )
        args.gates.write_text(
            json.dumps({"_note": note, **gates}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print("wrote", args.gates)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
