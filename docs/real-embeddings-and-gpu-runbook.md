# Runbook: real embeddings, GPU re-run, own quantization, Article 1022

Everything here needs something the build environment did not have (Hugging Face access, a GPU,
or an authoritative Arabic edition). The code and scripts are ready; these are the exact commands.

## A. Real multilingual embeddings (CPU is enough; needs internet once to download the model)

```bash
pip install -e ".[dev,tracking,fixtures,pdf,embeddings]"
export PRODML_EMBEDDING_MODEL=intfloat/multilingual-e5-small     # Windows cmd: set PRODML_EMBEDDING_MODEL=...
prodml-index                                                     # re-embeds 1,149 articles
prodml-eval --compare --sweep --retriever hybrid-dense --k 1 --threshold 0.30 --out reports/eval/latest.json
```

1. e5 models are asymmetric: the embedder already adds `passage: ` to articles and `query: ` to
   questions (`src/prodml/embedders.py`, tested without a download).
2. **Re-tune the abstention threshold.** e5 cosine scores sit in a much narrower, higher band than the
   hashing embeddings, so 0.30 will be wrong. Read `best threshold` from the sweep output, set it as
   `PRODML_MIN_CONFIDENCE` (and `min_confidence` in `src/prodml/config.py` if you adopt the model), re-run
   `prodml-eval` at that threshold.
3. Re-derive the CI gates from the measurement: `python scripts/retune_gates.py --threshold <T> --write`.
4. Refresh the experiment grid with the real model: `prodml-experiments --models intfloat/multilingual-e5-small BAAI/bge-m3 --plot reports/mlflow_runs_comparison.png`.
5. `dvc repro`, then update README "Measured results" and `reports/final-checklist.md` with the NEW numbers.
6. If you make the model the default: CI, the Dockerfile and `prodml-smoke` currently assume the offline
   default (no model download). Add the `embeddings` extra and a model download to those, or keep
   `hashing-1024` as the default and set `PRODML_EMBEDDING_MODEL` only in deployment.

Nothing here is guaranteed to improve abstention; measure it. The old numbers (hit@1 0.598, false-abstain
0.108, correct-abstain 0.526) are the baseline to beat.

## B. Re-run the GPU notebook on the real corpus (Kaggle, 2x T4)

The notebook builds its index from the repo's committed corpus, so once the real corpus is pushed it
runs on it automatically. Compared with the old run, `prodml-compare-llms` now answers **all 102
answerable eval questions** (it was 16 on the 5-article sample), so the new numbers are not comparable
with the old ones - replace them, do not mix them.

1. Push the repo first (the notebook clones the public repo anonymously if Internet = On).
2. Kaggle -> New notebook -> File -> Import `notebooks/kaggle_gpu_run.ipynb`. Settings: Accelerator
   **GPU T4 x2** (not P100), Internet **On**.
3. Dry run first (`DRY_RUN = True`, free CPU session) to check the plumbing; its numbers describe a stub.
4. Real run: `DRY_RUN = False`, Run All. `QUANT_OPTION` is now `"own"` (see C).
5. Download the output zip and copy `optimization.md`, `loadtest/`, `answers.json`, `run_info.json`, `logs/`
   into `reports/`. Then update the README "GPU run" section and the checklist rows with the new numbers.
6. Optional: `RUN_RAGAS = True` plus an `OPENAI_API_KEY` Kaggle secret replaces the lexical proxy.

## C. Quantizing the model yourself

`QUANT_OPTION = "own"` runs `scripts/quantize_awq.py` in a separate virtualenv (AutoAWQ, w_bit=4,
group_size=128, GEMM), calibrated on 128 Civil Code articles (Arabic + English) instead of the default
English `pile-val`. **Check `reports/run_info.json` after the run:** if `quantization` says "quantized by
this notebook" it worked; if it says "official AWQ checkpoint", the own quantization failed and the
notebook fell back - report it as the publisher's checkpoint, not yours. The script has never been run on a
GPU by the author (only its calibration builder is unit-tested); AutoAWQ is no longer actively maintained,
so a failure from a torch/transformers version clash is plausible - the log is in `logs/`.

## D. Article 1022 (Arabic)

`data/manual_fixes.json` restores the Arabic of Articles 1021/1022 (the PDF lost 1022's marker and first
paragraph). Paragraphs 2-3 are the PDF's own words; **paragraph 1 comes from a secondary source** (a legal
information blog), corroborated by the PDF's English text. To make it authoritative: replace the `new`
text of the 1022 entry with the wording from the Official Gazette / Ministry of Justice edition, update
`source`, run `dvc repro`, commit.
