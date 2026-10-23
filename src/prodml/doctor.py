"""``prodml-doctor`` - check an installation before running anything else.

    prodml-doctor            # offline: Python, config, corpus, optional packages (no network)
    prodml-doctor --online   # ALSO probe the services you configured (LLM endpoint, Langfuse, MLflow)

Exit code 0 = nothing blocking; 1 = at least one ERROR. Secrets are reported only as "set"/"unset" -
their values are never printed. Services that are not configured are skipped, not failed: the
default setup (extractive answers, no LLM, no tracing) needs no credentials at all.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import platform
import sys
import urllib.error
import urllib.request

from prodml.config import settings
from prodml.corpus import load_corpus
from prodml.validate_corpus import load_known_issues, validate_corpus

OK, WARN, ERROR, SKIP = "OK  ", "WARN", "ERR ", "SKIP"

# extra name -> (module to look for, what it unlocks)
OPTIONAL_PACKAGES = {
    "tracking": ("mlflow", "prodml-experiments, MLflow logging (--mlflow)"),
    "dvc": ("dvc", "dvc repro / dvc pull"),
    "pdf": ("pdfplumber", "prodml-ingest (rebuilding the corpus from the PDF)"),
    "embeddings": (
        "sentence_transformers",
        "real multilingual embeddings (PRODML_EMBEDDING_MODEL=<org>/<model>)",
    ),
    "ragas": ("ragas", "real RAGAS faithfulness (needs OPENAI_API_KEY too)"),
}


def _probe(url: str, headers: dict[str, str] | None = None) -> str | None:
    """None if the URL answers (any status < 500), else a short reason. Never includes headers."""
    try:
        req = urllib.request.Request(url, headers=headers or {})
        with urllib.request.urlopen(req, timeout=5) as resp:
            return None if resp.status < 500 else f"HTTP {resp.status}"
    except urllib.error.HTTPError as exc:
        return None if exc.code < 500 and exc.code not in (401, 403) else f"HTTP {exc.code}"
    except (urllib.error.URLError, OSError) as exc:
        return f"unreachable ({type(exc).__name__})"


def run_checks(online: bool = False) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []

    def add(level: str, name: str, detail: str) -> None:
        out.append((level, name, detail))

    py = sys.version_info
    add(
        OK if py >= (3, 10) else ERROR,
        "python",
        f"{platform.python_version()} on {platform.system()} {platform.machine()} (needs >= 3.10)",
    )

    add(
        OK,
        "configuration",
        f"valid; backend={settings.generation_backend}, retriever={settings.retriever}, "
        f"embedding={settings.embedding_model}, min_confidence={settings.min_confidence}",
    )

    try:
        articles = load_corpus(settings.corpus_path)
        res = validate_corpus(articles, known_issues=load_known_issues())
        if res.errors:
            add(
                ERROR,
                "corpus",
                f"{settings.corpus_path}: {len(res.errors)} validation errors, first: {res.errors[0]}",
            )
        else:
            live = sum(not a["is_repealed"] for a in articles)
            add(OK, "corpus", f"{len(articles)} articles ({live} live) in {settings.corpus_path}")
    except Exception as exc:  # noqa: BLE001 - report, don't crash the doctor
        add(
            ERROR,
            "corpus",
            f"cannot load {settings.corpus_path}: {exc}. Run from the repository root, or set PRODML_CORPUS_PATH",
        )

    add(
        OK if settings.drift_baseline_path.exists() else WARN,
        "eval set",
        f"{settings.drift_baseline_path}",
    )
    add(
        OK if settings.index_dir.joinpath("meta.json").exists() else SKIP,
        "vector index",
        f"{settings.index_dir} (built automatically on first use / by prodml-index)",
    )
    pdf = settings.raw_pdf_path
    add(
        OK if pdf.exists() else SKIP,
        "source PDF",
        f"{pdf} ({'present' if pdf.exists() else 'absent: only needed to REBUILD the corpus; the committed JSON is enough to run everything else'})",
    )

    for extra, (module, unlocks) in OPTIONAL_PACKAGES.items():
        found = importlib.util.find_spec(module) is not None
        add(
            OK if found else SKIP,
            f"extra [{extra}]",
            f"{'installed' if found else 'not installed'} - {unlocks}",
        )

    # services: only the ones the operator configured
    if settings.generation_backend == "extractive":
        add(SKIP, "LLM endpoint", "generation_backend=extractive: no LLM, no credentials needed")
    else:
        url = settings.llm_base_url.rstrip("/") + "/models"
        key = f"api key {'set' if settings.llm_api_key else 'unset'}"
        if online:
            hdr = (
                {"Authorization": f"Bearer {settings.llm_api_key}"}
                if settings.llm_api_key
                else None
            )
            why = _probe(url, hdr)
            add(
                OK if why is None else ERROR,
                "LLM endpoint",
                f"{settings.llm_base_url} model={settings.llm_model} ({key}) -> {why or 'reachable'}",
            )
        else:
            add(
                SKIP,
                "LLM endpoint",
                f"{settings.llm_base_url} model={settings.llm_model} ({key}); use --online to probe",
            )
    if settings.langfuse_host:
        if online:
            why = _probe(settings.langfuse_host.rstrip("/") + "/api/public/health")
            add(
                OK if why is None else WARN,
                "Langfuse",
                f"{settings.langfuse_host} -> {why or 'reachable'} (keys set)",
            )
        else:
            add(SKIP, "Langfuse", f"{settings.langfuse_host} (keys set); use --online to probe")
    else:
        add(SKIP, "Langfuse", "not configured (tracing is optional)")
    mlflow = os.environ.get("MLFLOW_TRACKING_URI", "")
    if mlflow.startswith("http"):
        if online:
            why = _probe(mlflow)
            add(OK if why is None else WARN, "MLflow server", f"{mlflow} -> {why or 'reachable'}")
        else:
            add(SKIP, "MLflow server", f"{mlflow}; use --online to probe")
    else:
        add(
            SKIP,
            "MLflow server",
            f"MLFLOW_TRACKING_URI {'= ' + mlflow if mlflow else 'unset'}: runs log to a local sqlite file (mlflow.db)",
        )
    add(
        OK if os.environ.get("OPENAI_API_KEY") else SKIP,
        "RAGAS judge key",
        f"OPENAI_API_KEY {'set' if os.environ.get('OPENAI_API_KEY') else 'unset'} (only needed for real RAGAS; the lexical proxy needs none)",
    )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument(
        "--online", action="store_true", help="probe configured services over the network"
    )
    args = ap.parse_args(argv)
    results = run_checks(args.online)
    for level, name, detail in results:
        print(f"[{level}] {name:<16} {detail}")
    errors = sum(level == ERROR for level, _, _ in results)
    print(f"\n{errors} blocking problem(s)." if errors else "\nNothing blocking.")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
