# GPU runbook: vLLM serving + AWQ quantization (Kaggle / Colab)

**Status: NOT executed in the environment this repository was built in (no GPU).** Everything the
repo needs for it is implemented and unit-tested against mocked endpoints (`prodml.llm`,
`prodml.compare_llms`). The commands below are what you run on a free T4. Record what you actually
measure in `reports/optimization.md`; never paste numbers you did not measure.

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
