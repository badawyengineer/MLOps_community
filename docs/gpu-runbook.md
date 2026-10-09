# GPU runbook: vLLM serving + AWQ quantization (Kaggle / Colab)

**Fastest route: import `notebooks/kaggle_gpu_run.ipynb` into Kaggle - it automates every step below. Run it once with `DRY_RUN = True` on a free CPU session first (a fake LLM; it proves the plumbing without spending GPU quota).**

**Status: executed once on Kaggle (2x Tesla T4, vLLM 0.31.0) with the publisher's official AWQ checkpoint.** Results are in
`reports/optimization.md`, `reports/loadtest/` and `reports/run_info.json`, and summarised in the README. What has NOT been
run: quantizing the model yourself (section 2 below / `QUANT_OPTION = "own"`) and real RAGAS (`RUN_RAGAS = True`).
Record what you actually measure; never paste numbers you did not measure.

**Environment notes from the real run:** Kaggle's image (Python 3.13, PyTorch 2.11 cu128, driver CUDA 13) needed
`/usr/local/lib/python3.13/dist-packages/nvidia/cu13/lib` on `LD_LIBRARY_PATH` before vLLM would start (the notebook has a cell
for it), and `VLLM_VERSION` is pinned to the version the reported run used. If pip replaces PyTorch, restart the session and run again.

## 1. Serve the baseline with vLLM
```bash
pip install vllm
vllm serve Qwen/Qwen2.5-1.5B-Instruct --port 8001 --max-model-len 4096 &
# point the API at it (the handbook asks you to name the model in the README)
export PRODML_GENERATION_BACKEND=vllm PRODML_LLM_BASE_URL=http://localhost:8001/v1 PRODML_LLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct
prodml-smoke        # /ask now returns generated, cited answers
```
If the free GPU session is too small for the model, say so and use a smaller model; do not claim a result you did not get.

## 2. Quantize to AWQ 4-bit
```python
from awq import AutoAWQForCausalLM            # pip install autoawq
from transformers import AutoTokenizer
src, dst = "Qwen/Qwen2.5-1.5B-Instruct", "qwen-awq"
model = AutoAWQForCausalLM.from_pretrained(src)
tok = AutoTokenizer.from_pretrained(src)
model.quantize(tok, quant_config={"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"})
model.save_quantized(dst); tok.save_pretrained(dst)
```
```bash
vllm serve ./qwen-awq --quantization awq --port 8002 &
```

## 3. Produce the before/after table
```bash
prodml-compare-llms --a-url http://localhost:8001/v1 --a-model Qwen/Qwen2.5-1.5B-Instruct \
                    --b-url http://localhost:8002/v1 --b-model ./qwen-awq \
                    --out reports/optimization.md --answers-json reports/answers.json
```
This reports latency p50/p95, tokens/s and the faithfulness-**proxy** drop (budget 0.03). For the
handbook's *RAGAS* faithfulness criterion, run RAGAS over `reports/answers.json` (see
`prodml.faithfulness.ragas_scores`) and report that number separately and under its real name.

## 4. Not implemented
Re-ranker distillation (the handbook's second optimization bullet) is not implemented; the
retriever is dense + BM25 with no cross-encoder re-ranking stage to distil.
