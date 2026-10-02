# Architecture

```mermaid
flowchart LR
  subgraph S1["Session 1 - Package"]
    PDF[("Civil Code PDF<br/>(bilingual, DVC-tracked)")] -->|prodml-ingest<br/>script-based column split| JSON[("articles.json<br/>text_ar / text_en / is_repealed")]
    JSON -->|validate_corpus --strict| JSON
  end
  subgraph S2["Session 2 - Track & version"]
    JSON -->|chunk by article| CH[chunking]
    CH --> EMB[embedders<br/>hashing / LSA / sentence-transformers]
    EMB --> VS[("vector store<br/>vectors.npz + meta.json<br/>DVC out, baked in image")]
    EXP["prodml-experiments<br/>12 runs -> MLflow"] --> REG[("MLflow Registry<br/>legal-qa-retriever @champion")]
    EVAL["prodml-eval<br/>hit@k, MRR, abstention"] --> GATE{{"CI gate<br/>data/eval/thresholds.json"}}
  end
  subgraph S3["Session 3 - Serve"]
    Q([question]) --> API["FastAPI /ask /ask/stream<br/>or BentoML service"]
    API --> RET["hybrid-dense retriever<br/>dense + BM25"]
    VS --> RET
    RET -->|"score &lt; 0.20"| ABS["abstain<br/>(never calls the LLM)"]
    RET --> GEN["generator<br/>extractive | vLLM / OpenAI-compatible"]
    GEN --> GR["guardrails<br/>PII redaction + citation check"]
    GR --> A([answer + citations + confidence])
  end
  subgraph S4["Session 4 - Optimize"]
    CMP["prodml-compare-llms<br/>baseline vs AWQ-4bit<br/>latency, tokens/s, faithfulness"] -.-> GEN
    LT["Locust 50 users"] -.-> API
  end
  subgraph S5["Session 5 - Monitor"]
    API -->|/metrics| PROM[(Prometheus)]
    GR -->|spans + faithfulness score| LF[(Langfuse<br/>self-hosted)]
    MON["prodml-monitor<br/>eval faithfulness"] -->|Pushgateway| PROM
    PROM --> GRAF[Grafana dashboard]
    PROM --> AL["alerts: faithfulness &lt; 0.80,<br/>abstention rate, drift, LLM fallbacks"]
  end
```

## Why these choices
- **Chunk by article, not by token count.** A legal article is the natural unit of meaning and of citation; every chunk keeps its article number so retrieval can always cite.
- **Abstention is a feature.** Below a calibrated confidence the service refuses instead of guessing, and the LLM is never called.
- **The LLM is optional and replaceable.** Anything that speaks the OpenAI chat-completions protocol (vLLM included) plugs in via three environment variables; if it fails, answers fall back to extractive text and the failure is flagged and counted.
- **Guardrails run on the way out.** A hallucinated `[Article 999]` or a leaked phone number never reaches the user.
- **Everything measurable is exported**: latency, abstentions, retrieval-score distribution, query drift, token cost, guardrail trips, faithfulness.
