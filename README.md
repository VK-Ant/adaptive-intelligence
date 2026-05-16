# adaptive-intelligence

Self-improving retrieval orchestration framework for document intelligence. Drop documents, ask questions, the system learns how to retrieve better over time.

RL-based retrieval routing, conditional graph activation, evaluation-driven learning, and zero-configuration architecture. Works with any LLM.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/VK-Ant/adaptive-intelligence/blob/main/notebooks/traditional_rag_vs_adaptive_intelligence.ipynb)

- PyPI: https://pypi.org/project/adaptive-intelligence
- GitHub: https://github.com/VK-Ant/adaptive-intelligence
- Portfolio: https://vk-ant.github.io/Venkatkumar

## Install

```
pip install adaptive-intelligence
pip install adaptive-intelligence[pdf]          # adds PDF support
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
# With Grok API
engine = AdaptiveAI(
    llm_backend="openai",
    llm_model="grok-3-mini",
    api_key="xai-...",
    base_url="https://api.x.ai/v1",
    domain="financial",
)

# With OpenAI
engine = AdaptiveAI(
    llm_backend="openai",
    llm_model="gpt-4o",
    api_key="sk-...",
)
```

---

## What Makes This Different

| | Traditional RAG | Adaptive Intelligence |
|---|---|---|
| Retrieval | Static vector similarity | RL-learned routing (6 strategies) |
| Graph | None | Conditional activation (5-signal gate) |
| Prompts | Fixed template | Domain-adaptive, evolving |
| Learning | Same performance forever | Improves with each query |
| Evaluation | Manual | Automatic 6-metric + RL reward |

---

## Three Core Innovations

### 1. RL Policy Engine

Contextual bandits with Thompson Sampling learn which retrieval strategy works best for each query type. First 15 queries use heuristic defaults, then RL takes over. No hardcoded rules.

```python
# The RL policy decides per query:
# - Retrieval route: vector, keyword, hybrid, table-first, graph-hybrid
# - Retrieval depth: 3, 5, 8, 10, 15 chunks
# - Graph activation: on/off
# - Prompt template: extraction, analysis, summary, comparison
# - Verification level: none, citation, full
```

### 2. Conditional Graph Activation

Knowledge graph built automatically during ingestion. Activated only when the query needs relational reasoning — not wasted on simple factual lookups.

Five signals gate activation: relationship words, query analysis, entity density, complexity, historical success rate.

### 3. Self-Adaptive Retrieval

Every response evaluated on 6 metrics. Composite score becomes RL reward. System measurably improves over queries.

```python
# Evaluation metrics (all automatic, no ground truth needed):
# - Faithfulness: grounded in source documents?
# - Relevance: addresses the query?
# - Citation accuracy: sources cited?
# - Hallucination risk: fabricated content?
# - Retrieval precision: relevant chunks retrieved?
# - Retrieval recall: query terms covered?
```

---

## Supported Formats

| S.No. | Format | Extension | Required Package |
|-------|--------|-----------|-----------------|
| 1 | Text / Markdown | .txt, .md | — |
| 2 | CSV | .csv | — |
| 3 | JSON | .json | — |
| 4 | HTML | .html | — |
| 5 | XML | .xml | — |
| 6 | PDF | .pdf | `PyMuPDF` or `pdfplumber` |
| 7 | Word | .docx | `python-docx` |
| 8 | Excel | .xlsx | `openpyxl` |
| 9 | PowerPoint | .pptx | `python-pptx` |
| 10 | Images (OCR) | .png, .jpg | `pytesseract`, `Pillow` |

## Supported LLM Providers

| S.No. | Provider | Backend | Local? | Free? |
|-------|----------|---------|--------|-------|
| 1 | Ollama | `ollama` | Yes | Yes |
| 2 | OpenAI | `openai` | No | No |
| 3 | Grok (xAI) | `openai` | No | No |
| 4 | Azure OpenAI | `azure_openai` | No | No |
| 5 | Groq | `groq` | No | Free tier |
| 6 | Together AI | `together` | No | Free tier |
| 7 | HuggingFace | `huggingface` | Yes | Yes |
| 8 | Any OpenAI-compatible | `custom` | Varies | Varies |

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

# Citations
for c in response.citations:
    print(f"  {c.source_document} ({c.confidence:.0%})")
```

### Dashboard and Monitoring

```python
# System dashboard
print(engine.dashboard())

# RL policy stats
stats = engine.rl.get_stats()
print(f"Warmup: {stats['is_warmup']}")
print(f"Arms learned: {stats['total_arms']}")
print(f"Exploration: {stats['exploration_rate']:.1%}")

# Learning curve data
curve = engine.learning_curve()

# Learning memory
print(engine.memory.get_learning_summary())

# Audit trail
print(engine.audit.display_query_trail(response.query_id))
engine.audit.export("audit.json")
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

    rl=RLConfig(
        warmup_queries=20,
        exploration_rate=0.15,
        algorithm="thompson_sampling",
    ),

    graph=GraphConfig(
        enabled=True,
        conditional_activation=True,
        max_hops=3,
    ),

    evaluation=EvaluationConfig(
        faithfulness_weight=0.35,
        enable_llm_judge=True,
    ),
)

engine = AdaptiveAI(config=config)
```

---

## Architecture - Summary

![systemarchitecure](docs/images/Screenshot%202026-05-16%20075927.png)

- User Query enters the system as natural language.

- Trigger Interpreter classifies query type, complexity, domain, and extracts entities.

- RL Policy Engine selects the optimal retrieval strategy using Thompson Sampling.

- Retrieval Orchestrator queries the chosen index (Vector, Keyword, Graph, Table, or Hybrid).

- Graph Index activates only when a five-signal gate detects relationship reasoning is needed.

- Adaptive Prompt Engine builds a domain-aware prompt from evolving templates.

- LLM Generation produces the answer using any model (Ollama, OpenAI, Grok, Claude).

- Evaluation Engine scores the answer on faithfulness, relevance, hallucination, and citations.

- Reward Signal feeds the evaluation score back to update the RL policy.

- Response returns the answer with confidence score, sources, and audit trail.


---

## Also by the Author

- **[llmevalkit](https://pypi.org/project/llmevalkit/)** — LLM evaluation, hallucination detection, compliance, and 61 metrics
- **[Responsible AI Series](https://medium.com/@VK_Venkatkumar)** — HIPAA, GDPR, NIST AI RMF, CoSAI, EU AI Act

## Citation

```bibtex
@software{venkatkumar2026adaptive,
  title={Adaptive Intelligence: Self-Improving Retrieval Orchestration via Evaluation-Driven Policy Learning},
  author={Venkatkumar, Rajan},
  year={2026},
  url={https://github.com/VK-Ant/adaptive-intelligence}
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
