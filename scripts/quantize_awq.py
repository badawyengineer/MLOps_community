"""Quantize a causal LM to AWQ 4-bit with AutoAWQ, calibrated on the Civil Code itself.

    python scripts/quantize_awq.py --model Qwen/Qwen2.5-1.5B-Instruct --out qwen-awq

Why calibrate on the corpus: AWQ picks per-channel scales from the activations it sees on the
calibration text, so using the bilingual legal text this service actually prompts with is a
better match than AutoAWQ's default English-only ``pile-val`` (and needs no extra download).

Run it in a SEPARATE virtualenv (AutoAWQ pins torch/transformers versions that can clash with
vLLM's); the Kaggle notebook (notebooks/kaggle_gpu_run.ipynb, QUANT_OPTION = "own") does that.
Needs a CUDA GPU. NOT exercised in the offline build environment - only ``calibration_texts``
is unit-tested (tests/test_quantize_script.py); report the quantization as yours only if
``<out>/config.json`` exists after a run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

QUANT_CONFIG = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}


def calibration_texts(articles_path: Path, n: int = 128, max_chars: int = 1500) -> list[str]:
    """``n`` evenly spaced, non-repealed articles as 'Arabic + English' calibration samples.

    Deterministic (no RNG) so a re-run quantizes from the same data."""
    articles = json.loads(Path(articles_path).read_text(encoding="utf-8"))
    live = [
        a for a in articles if not a.get("is_repealed") and a.get("text_ar") and a.get("text_en")
    ]
    if not live:
        raise SystemExit(f"no usable articles in {articles_path}")
    step = max(1, len(live) // n)
    picked = live[::step][:n]
    return [f"{a['text_ar']}\n{a['text_en']}"[:max_chars] for a in picked]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model", required=True, help="HF model id or local path (the FP16 original)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--articles", type=Path, default=Path("data/processed/articles.json"))
    ap.add_argument("--n-calib", type=int, default=128)
    ap.add_argument("--max-seq-len", type=int, default=512)
    args = ap.parse_args(argv)

    texts = calibration_texts(args.articles, args.n_calib)
    print(f"calibrating on {len(texts)} corpus samples; config {QUANT_CONFIG}")

    from awq import AutoAWQForCausalLM  # heavy, GPU-only: imported late on purpose
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoAWQForCausalLM.from_pretrained(args.model, low_cpu_mem_usage=True, use_cache=False)
    model.quantize(
        tokenizer,
        quant_config=QUANT_CONFIG,
        calib_data=texts,
        max_calib_samples=len(texts),
        max_calib_seq_len=args.max_seq_len,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    model.save_quantized(str(args.out))
    tokenizer.save_pretrained(str(args.out))
    (args.out / "quantization_note.json").write_text(
        json.dumps(
            {
                "base_model": args.model,
                "quant_config": QUANT_CONFIG,
                "calibration": "Civil Code articles (ar+en)",
                "n_samples": len(texts),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("saved", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
