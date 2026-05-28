# adaptive-intelligence

Self-improving retrieval orchestration framework. Drop documents, ask questions, the system learns how to retrieve better over time.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/VK-Ant/adaptive-intelligence/blob/main/notebooks/adaptive_intelligence_v2_three_way.ipynb)

## Install

```
pip install adaptive-intelligence
pip install adaptive-intelligence[pdf]
pip install adaptive-intelligence[sql]
pip install adaptive-intelligence[all]
```

---

## Quick Start

```python
from adaptive_intelligence import AdaptiveAI

engine = AdaptiveAI()
engine.ingest("./documents")
response = engine.ask("What are the key risks?")
print(response.answer)
print(f"Confidence: {response.confidence:.0%}")
```

---

## Comparison

| S.No. | Capability | Traditional RAG | GraphRAG | Adaptive Intelligence |
|-------|-----------|----------------|----------|----------------------|
| 1 | Retrieval | Static vector | Always graph | RL-learned per query |
| 2 | Graph | None | Always on | Conditional (5-signal gate) |
| 3 | Learning | None | None | Improves every query |
| 4 | Evaluation | None | None | 6 metrics per response |
| 5 | Vector DB | Required | Required | Optional (vectorless mode) |
| 6 | Output | Text only | Text only | JSON, CSV, YAML, DataFrame |
| 7 | Feedback | None | None | Thumbs up/down updates RL |
| 8 | Reranking | None | None | Cross-encoder re-scoring |
| 9 | Complex queries | Single retrieval | Single retrieval | Multi-query decomposition |
| 10 | Domain warmup | Manual tuning | Manual tuning | Pre-trained policies (skip warmup) |
| 11 | LLM agnostic | Usually one | Usually one | 10+ providers |
| 12 | Crash recovery | None | Partial | Full auto-checkpoint |

## Results (20 queries, same LLM, same corpus)

| S.No. | Query type | Traditional RAG | Adaptive Intelligence | Delta |
|-------|-----------|----------------|----------------------|-------|
| 1 | Factual | 85% | 90% | +5% |
| 2 | Relational | 45% | 78% | +33% |
| 3 | Analytical | 55% | 75% | +20% |
| 4 | Comparative | 50% | 80% | +30% |
| 5 | Multi-hop | 35% | 72% | +37% |
| 6 | **Overall** | **54%** | **79%** | **+25%** |

---

## Version Evolution

| S.No. | Feature | v1 | v2 | v3 (current) |
|-------|---------|----|----|------|
| 1 | RL algorithm | Thompson Sampling | + configurable warmup | + PPO option |
| 2 | Graph | 5-signal gate + BFS | + persistence | + pre-trained policies |
| 3 | Evaluation | 6 metrics | + user feedback | + A/B testing |
| 4 | Ingestion | Basic | Hardened (every edge case) | Same |
| 5 | SQL connector | No | PostgreSQL, MySQL, SQLite | Same |
| 6 | Vectorless mode | No | Page BM25 + graph + RL | Same |
| 7 | Output formats | Text only | JSON, CSV, YAML, DataFrame | Same |
| 8 | Reranking | No | No | Cross-encoder re-scoring |
| 9 | Multi-query | No | No | Auto-decompose complex queries |
| 10 | Pre-trained policies | No | No | Financial, legal, healthcare |
| 11 | Transfer learning | No | No | Export/import policies |
| 12 | A/B testing | No | No | Compare two policies |
| 13 | Crash recovery | Partial | Full auto-checkpoint | Same |
| 14 | Providers | 3 | 10+ | Same |
| 15 | System prompt | No | Custom | Same |

---

## All Features

### RL + Retrieval

```python
# Default: Thompson Sampling
engine = AdaptiveAI()

# PPO algorithm
engine = AdaptiveAI(rl_algorithm="ppo")

# Cross-encoder reranking
engine = AdaptiveAI(reranking=True)

# Pre-trained policy (skip warmup)
engine = AdaptiveAI(domain="financial", pretrained_policy=True)

# Transfer learning
engine.export_policy("my_policy.json")
other_engine.import_policy("my_policy.json")

# A/B testing
engine.enable_ab_test(policy_a="thompson", policy_b="ppo")
print(engine.ab_results())
```

### Vectorless Mode

```python
engine = AdaptiveAI(vectorless=True)
# No ChromaDB. No embeddings. Pure BM25 + graph + RL.
# Page-number citations. Zero dependencies.
```

### Output Formats

```python
response = engine.ask("Extract vendors", output_format="json")
response = engine.ask("List items", output_format="csv")
response = engine.ask("Show data", output_format="dataframe")
response = engine.ask("Details", output_format="json",
    schema={"vendor": "str", "amount": "float"})
```

### User Feedback

```python
engine.feedback(response.query_id, "good")
engine.feedback(response.query_id, "bad", reason="Missing data")
```

### SQL Connector

```python
engine.ingest("sqlite:///data.db")
engine.ingest("postgresql://user:pass@host/db", tables=["orders"])
```

### Incremental Ingestion

```python
engine.ingest("./new_file.pdf")
engine.remove("old_file.pdf")
engine.update("./changed_file.pdf")
engine.ingest("./docs/", parallel=True, workers=4)
```

### System Prompt

```python
engine = AdaptiveAI(system_prompt="You are a financial analyst.")
response = engine.ask("Risks?", system_prompt="Rate each HIGH/MEDIUM/LOW.")
engine.set_system_prompt("You are a legal reviewer.")
```

---

## Providers — Copy, Paste, Run

### Free

```python
engine = AdaptiveAI()  # Ollama (default, local)
engine = AdaptiveAI(llm_model="llama3.2")  # Ollama specific model

engine = AdaptiveAI(api_key="nvapi-...",
    base_url="https://integrate.api.nvidia.com/v1",
    llm_model="meta/llama-3.1-70b-instruct")  # NVIDIA NIM

engine = AdaptiveAI(api_key="gsk_...",
    base_url="https://api.groq.com/openai/v1",
    llm_model="llama-3.3-70b-versatile")  # Groq

engine = AdaptiveAI(llm_backend="huggingface",
    llm_model="Qwen/Qwen2.5-1.5B-Instruct")  # HuggingFace local
```

### Paid

```python
engine = AdaptiveAI(api_key="sk-...", llm_model="gpt-4o")  # OpenAI
engine = AdaptiveAI(api_key="xai-...",
    base_url="https://api.x.ai/v1", llm_model="grok-3-mini")  # Grok
```

### No LLM

```python
engine = AdaptiveAI(llm_backend="none")  # Retrieval only
engine = AdaptiveAI(llm_backend="none", vectorless=True)  # Zero dependencies
```

---

## Supported Formats

| S.No. | Format | Extension |
|-------|--------|-----------|
| 1 | Text / Markdown | .txt, .md |
| 2 | CSV / TSV | .csv, .tsv |
| 3 | JSON | .json |
| 4 | HTML / XML | .html, .xml |
| 5 | PDF | .pdf |
| 6 | Word | .docx |
| 7 | Excel | .xlsx |
| 8 | PowerPoint | .pptx |
| 9 | Images (OCR) | .png, .jpg |
| 10 | SQL databases | PostgreSQL, MySQL, SQLite |

---

## FAQ

**Q: How is this different from ChatGPT / Claude?**
This is not an LLM. It's the retrieval layer that decides what context to feed TO the LLM. Works with any LLM as backend.

**Q: What is vectorless mode?**
No ChromaDB, no embeddings, no vector DB. Pure BM25 keyword search + graph + RL. Best for documents with standardized terminology or air-gapped environments.

**Q: Does the RL policy persist?**
Yes. Everything persists to disk with auto-checkpoint every 5 minutes.

**Q: Can I use it without an LLM?**
Yes. `AdaptiveAI(llm_backend="none")` returns ranked source excerpts directly.

---

## Also by the Author

- **[llmevalkit](https://pypi.org/project/llmevalkit/)** — 61 metrics for LLM evaluation
- **[Responsible AI Series](https://medium.com/@VK_Venkatkumar)** — HIPAA, GDPR, NIST AI RMF, CoSAI

## Citation

```bibtex
@article{venkatkumar2026adaptive,
  title={Adaptive Retrieval Orchestration for Self-Learning Knowledge Systems},
  author={Venkatkumar, Rajan},
  year={2026},
  url={https://www.researchgate.net/publication/405076088}
}
```

## License

Apache License 2.0

## Author

Venkatkumar Rajan

- LinkedIn: https://linkedin.com/in/venkatkumarvk
- GitHub: https://github.com/VK-Ant
- Portfolio: https://vk-ant.github.io/Venkatkumar/
- PyPI: https://pypi.org/project/adaptive-intelligence/
- ResearchGate: https://www.researchgate.net/publication/405076088
