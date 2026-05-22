# adaptive-intelligence

Self-improving retrieval orchestration framework for document intelligence. Drop documents, ask questions, the system learns how to retrieve better over time.

RL-based retrieval routing, conditional graph activation, evaluation-driven learning, vectorless mode, and zero-configuration architecture. Works with any LLM.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/VK-Ant/adaptive-intelligence/blob/main/notebooks/traditional_rag_vs_adaptive_intelligence.ipynb)


## Install

```
pip install adaptive-intelligence
pip install adaptive-intelligence[pdf]          # PDF support
pip install adaptive-intelligence[sql]          # SQL connector
pip install adaptive-intelligence[all]          # all document formats
pip install adaptive-intelligence[huggingface]  # local HuggingFace models
```

---

## Quick Start

```python
from adaptive_intelligence import AdaptiveAI

# Zero config — defaults to Ollama (free, local, private)
engine = AdaptiveAI()
engine.ingest("./documents")
response = engine.ask("What are the key operational risks?")

print(response.answer)
print(f"Confidence: {response.confidence:.0%}")
print(response.evaluation.display())
```

```python
# Vectorless mode — no embeddings, no ChromaDB, zero dependencies
engine = AdaptiveAI(vectorless=True)
engine.ingest("./documents")
response = engine.ask("Revenue details?")
print(response.citations[0].page)  # Page number citation
```

```python
# Structured output
response = engine.ask("Extract vendors", output_format="json")
print(response.structured)  # Parsed dict

# User feedback → RL reward
engine.feedback(response.query_id, "good")
engine.feedback(response.query_id, "bad", reason="Missing data")
```

---

## What Makes This Different

| | Traditional RAG | Adaptive Intelligence |
|---|---|---|
| Retrieval | Static vector similarity | RL-learned routing (6+ strategies) |
| Graph | None | Conditional activation (5-signal gate) |
| Prompts | Fixed template | Domain-adaptive, evolving |
| Learning | Same performance forever | Improves with each query |
| Evaluation | Manual | Automatic 6-metric + RL reward |
| Vector DB | Required | Optional (vectorless mode) |
| Output | Text only | JSON, CSV, YAML, DataFrame |
| Feedback | None | Thumbs up/down → RL update |
| Crash recovery | None | Auto-checkpoint + graceful shutdown |

---

## Three Core Innovations

### 1. RL Policy Engine

Contextual bandits with Thompson Sampling learn which retrieval strategy works best for each query type. First 15 queries use heuristic defaults, then RL takes over. No hardcoded rules.

### 2. Conditional Graph Activation

Knowledge graph built automatically during ingestion. Activated only when the query needs relational reasoning — not wasted on simple factual lookups. Five signals gate activation.

### 3. Self-Adaptive Retrieval

Every response evaluated on 6 metrics. Composite score becomes RL reward. System measurably improves over queries.

---

## v2 Features

### Vectorless Mode

No embeddings. No ChromaDB. No vector DB at all. Pure Python BM25 + knowledge graph + RL routing.

```python
engine = AdaptiveAI(vectorless=True)
# Page-level BM25 index, page citations, zero dependencies
# RL + graph + evaluation still fully active
```

### Output Formats

```python
response = engine.ask("Extract vendors", output_format="json")
response = engine.ask("List items", output_format="csv")
response = engine.ask("Show data", output_format="yaml")
response = engine.ask("Revenue breakdown", output_format="dataframe")

# Custom schema
response = engine.ask("Contract details", output_format="json",
    schema={"parties": ["str"], "value": "float", "date": "str"})
```

### User Feedback

```python
engine.feedback(response.query_id, "good")   # RL reward boost
engine.feedback(response.query_id, "bad", reason="Wrong data")  # RL penalty + prompt evolution
```

### Incremental Ingestion

```python
engine.ingest("./new_report.pdf")       # Add
engine.remove("old_report.pdf")          # Remove
engine.update("./updated_report.pdf")    # Re-index

# Parallel
engine.ingest("./docs/", parallel=True, workers=4)
```

### SQL Connector

```python
engine.ingest("sqlite:///data.db")
engine.ingest("postgresql://user:pass@host/db", tables=["invoices", "vendors"])
engine.ingest("mysql://user:pass@host/db", query="SELECT * FROM orders WHERE year=2025")
```

### Crash Recovery

Auto-checkpoint every 5 minutes. BM25, graph, RL policy, and memory all persist to disk. Graceful shutdown on SIGTERM/SIGINT. Auto-recovery on startup.

### Hardened Ingestion

Handles every edge case: corrupted PDFs, password-protected files, scanned images (auto-OCR), Excel merged cells, hidden sheets, formula evaluation, CSV wrong delimiters, mixed encodings, malformed rows.

---

## Supported Formats

| S.No. | Format | Extension | Required Package |
|-------|--------|-----------|-----------------|
| 1 | Text / Markdown | .txt, .md | — |
| 2 | CSV / TSV | .csv, .tsv | — |
| 3 | JSON | .json | — |
| 4 | HTML | .html | — |
| 5 | XML | .xml | — |
| 6 | PDF | .pdf | `PyMuPDF` or `pdfplumber` |
| 7 | Word | .docx | `python-docx` |
| 8 | Excel | .xlsx | `openpyxl` |
| 9 | PowerPoint | .pptx | `python-pptx` |
| 10 | Images (OCR) | .png, .jpg | `pytesseract`, `Pillow` |
| 11 | SQL databases | — | `sqlalchemy` |

---

## Providers — Copy, Paste, Run

### Free Providers

```python
# Ollama (default, local, free)
engine = AdaptiveAI()

# Ollama specific model
engine = AdaptiveAI(llm_model="llama3.2")

# NVIDIA NIM (free)
engine = AdaptiveAI(api_key="nvapi-...",
    base_url="https://integrate.api.nvidia.com/v1",
    llm_model="meta/llama-3.1-70b-instruct")

# Groq (free tier)
engine = AdaptiveAI(api_key="gsk_...",
    base_url="https://api.groq.com/openai/v1",
    llm_model="llama-3.3-70b-versatile")

# Google Gemini (free tier)
engine = AdaptiveAI(api_key="...",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai",
    llm_model="gemini-2.0-flash")

# HuggingFace (local, any model)
engine = AdaptiveAI(llm_backend="huggingface", llm_model="microsoft/phi-2")

# Together AI (free tier)
engine = AdaptiveAI(api_key="...",
    base_url="https://api.together.xyz/v1",
    llm_model="meta-llama/Llama-3-70b-chat-hf")
```

### Paid Providers

```python
# OpenAI
engine = AdaptiveAI(api_key="sk-...", llm_model="gpt-4o")

# Grok (xAI)
engine = AdaptiveAI(api_key="xai-...",
    base_url="https://api.x.ai/v1", llm_model="grok-3-mini")

# Azure OpenAI
engine = AdaptiveAI(llm_backend="azure_openai", api_key="...",
    azure_endpoint="https://your-resource.openai.azure.com",
    deployment_name="gpt-4o")

# AWS Bedrock (via gateway)
engine = AdaptiveAI(api_key="...",
    base_url="https://your-bedrock-gateway/v1",
    llm_model="anthropic.claude-v2")
```

### No LLM

```python
# Retrieval only — returns ranked excerpts
engine = AdaptiveAI(llm_backend="none")

# Full zero-dependency mode
engine = AdaptiveAI(llm_backend="none", vectorless=True)
```

---

## Code Examples

### Inspect the Full Pipeline

```python
response = engine.ask("Compare Q2 and Q3 revenue")

# What did the system understand?
print(response.query_analysis)

# What strategy did the RL policy choose?
pd = response.policy_decision
print(f"Route: {pd.retrieval_route}")
print(f"Graph: {pd.graph_activation}")
print(f"Explored: {pd.was_exploration}")

# Evaluation scores
print(response.evaluation.display())

# Citations with page numbers
for c in response.citations:
    print(f"  {c.source_document}, Page {c.page} ({c.confidence:.0%})")
```

### Dashboard and Monitoring

```python
print(engine.dashboard())

stats = engine.rl.get_stats()
print(f"Warmup: {stats['is_warmup']}")
print(f"Arms learned: {stats['total_arms']}")

curve = engine.learning_curve()
print(engine.memory.get_learning_summary())

engine.audit.export("audit.json")
```

### System Prompt

```python
# Set at init
engine = AdaptiveAI(system_prompt="You are a financial analyst. Cite page numbers.")

# Override per query
response = engine.ask("Risks?",
    system_prompt="You are a risk specialist. Rate each HIGH/MEDIUM/LOW.")

# Update anytime
engine.set_system_prompt("You are a legal reviewer. Flag violations.")
engine.set_system_prompt(None)  # Reset to default
```

### Advanced Configuration

```python
from adaptive_intelligence.core.config import (
    AdaptiveConfig, RLConfig, GraphConfig, EvaluationConfig,
    LLMBackend, Domain, SecurityLevel,
)

config = AdaptiveConfig(
    llm_backend=LLMBackend.OLLAMA,
    llm_model="llama3.2",
    domain=Domain.FINANCIAL,
    security_level=SecurityLevel.HIGH,
    rl=RLConfig(warmup_queries=20, exploration_rate=0.15),
    graph=GraphConfig(conditional_activation=True, max_hops=3),
    evaluation=EvaluationConfig(faithfulness_weight=0.35, enable_llm_judge=True),
)
engine = AdaptiveAI(config=config)
```

## Why This Exists

The world has 500+ LLMs. Every cloud provider (Azure, AWS, GCP), every platform (HuggingFace, Ollama), every company (NVIDIA, Meta, Google, xAI) is producing models. Many are free.

But every LLM has the same problem: garbage in, garbage out.

adaptive-intelligence is the layer that learns WHAT to feed the LLM. The LLM is replaceable. The retrieval intelligence is not.

---

## FAQ

**Q: How does ingestion handle mixed content (text + tables) from PDFs?**

Tables are extracted with structure preserved (`is_table=True`) and indexed into the same indexes as text. The RL policy learns to use the `table_first` retrieval route for structured queries. Separation happens at retrieval time via learned routing, not at ingestion time.

**Q: How is this different from just using ChatGPT / Claude?**

adaptive-intelligence is not an LLM — it's the retrieval layer that decides what context to feed TO the LLM. It uses ChatGPT, Claude, Grok, or Ollama as backends. The system learns which retrieval strategy works best for each query type on YOUR specific documents.

**Q: What is vectorless mode?**

No ChromaDB. No embeddings. No vector DB. Pure Python BM25 keyword search over full pages + knowledge graph + RL routing. Everything still learns and improves — just without vector similarity. Best for financial/legal/medical documents with standardized terminology, or air-gapped environments.

**Q: Does the RL policy persist across sessions?**

Yes. RL state, BM25 index, knowledge graph, and learning memory all persist to disk. Auto-checkpoint every 5 minutes. Auto-recovery on startup.

**Q: Can I use this without an LLM (offline)?**

Yes. `AdaptiveAI(llm_backend="none")` returns ranked source excerpts directly. RL, graph, and evaluation still work.

---

## Roadmap

### v2.0 (Current)
- [x] RL policy engine (Thompson Sampling)
- [x] Conditional graph activation (5-signal gate)
- [x] 6-metric evaluation → RL reward loop
- [x] Vectorless mode (page BM25, zero dependencies)
- [x] Output formats (JSON, CSV, YAML, DataFrame)
- [x] User feedback → RL reward
- [x] Crash recovery (auto-checkpoint, graceful shutdown)
- [x] Hardened ingestion (every PDF/Excel/CSV edge case)
- [x] SQL connector (PostgreSQL, MySQL, SQLite)
- [x] Incremental ingestion (add/remove/update)
- [x] Parallel ingestion
- [x] Chunk quality scoring + deduplication
- [x] Custom system prompt support
- [x] 10+ providers copy-paste ready
- [x] Page-number citations

### v3.0 — Intelligence Upgrades
- [ ] PPO/DQN alongside Thompson Sampling
- [ ] GraphSAGE embeddings replacing BFS
- [ ] Cross-encoder reranking
- [ ] Multi-query decomposition
- [ ] Pre-trained domain policies (financial, legal, healthcare)
- [ ] Transfer learning across deployments
- [ ] llmevalkit compliance integration
- [ ] Multi-modal ingestion (images, charts, audio)
- [ ] Enterprise connectors (S3, Notion, Confluence)
- [ ] Plugin API for community connectors

---

## Also by the Author

- **[llmevalkit](https://pypi.org/project/llmevalkit/)** — LLM evaluation, hallucination detection, compliance, and 61 metrics
- **[Responsible AI Series](https://medium.com/@VK_Venkatkumar/list/responsible-ai-engineer-series-4d9565c82bd2)** — HIPAA, GDPR, NIST AI RMF, CoSAI, EU AI Act

## Citation

```bibtex
@article{venkatkumar2026adaptive,
  title={Adaptive Retrieval Orchestration for Self-Learning Knowledge Systems},
  author={Venkatkumar, Rajan},
  year={2026},
  url={https://www.researchgate.net/publication/405076088},
  note={Available at ResearchGate and GitHub: github.com/VK-Ant/adaptive-intelligence}
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
