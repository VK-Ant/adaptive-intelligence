# Adaptive Intelligence

**Self-improving retrieval orchestration that learns from every query.**

[![PyPI](https://img.shields.io/pypi/v/adaptive-intelligence)](https://pypi.org/project/adaptive-intelligence/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

Most RAG systems use static retrieval pipelines — same strategy for every query, no learning, no adaptation. Adaptive Intelligence changes that. It uses **reinforcement learning** to discover the optimal retrieval strategy for each query type, **conditional graph activation** to enable relationship reasoning only when needed, and **evaluation-driven feedback loops** to measurably improve over time.

```python
from adaptive_intelligence import AdaptiveAI

engine = AdaptiveAI()
engine.ingest("./documents")
response = engine.ask("What are the key operational risks?")

print(response.answer)       # Synthesized answer
print(response.confidence)   # 0.87
print(response.evaluation.display())  # Full quality metrics
```

Three lines to start. The system handles everything else — and gets better with every query.

---

## What Makes This Different

### 1. RL-Based Retrieval Routing

Instead of hardcoded rules ("always use hybrid search"), Adaptive Intelligence uses **contextual bandits with Thompson Sampling** to learn which retrieval strategy works best for which type of query:

- First 15 queries: heuristic defaults (keyword for factual, vector for semantic, hybrid for complex)
- After warmup: the RL policy selects from 6 strategies (vector, keyword, hybrid, table-first, graph-first, graph-hybrid), choosing depth, graph activation, prompt template, and verification level
- Every response is evaluated → evaluation score becomes the RL reward → policy updates

The system discovers patterns like "financial extraction queries work best with keyword search at depth 8" without being told.

### 2. Conditional Graph Activation

Knowledge graphs are powerful but expensive. Adaptive Intelligence builds an entity-relationship graph from your documents automatically, but only activates graph traversal when the query actually needs relational reasoning. Five signals gate activation:

- Relationship words in query ("connected", "depends on", "affects")
- Entity density (multiple entities that might be related)
- Query complexity (multi-hop reasoning detected)
- Historical success rate (did graph help for this query type before?)
- RL policy recommendation

### 3. Self-Adaptive Retrieval

The system measurably improves through an evaluation-driven feedback loop:

- **Layer 1** (always on): Automatic metrics — faithfulness, relevance, citation accuracy, hallucination risk, retrieval precision/recall
- **Layer 2** (when L1 confidence is low): LLM-as-Judge evaluation
- **Layer 3** (periodic): Cross-reference consistency checks

The composite evaluation score feeds back as the RL reward signal, closing the loop.

---

## Installation

```bash
pip install adaptive-intelligence
```

For document format support:
```bash
# PDF support
pip install adaptive-intelligence[pdf]

# All document formats (PDF, DOCX, XLSX, PPTX)
pip install adaptive-intelligence[all]

# HuggingFace models (local, any model)
pip install adaptive-intelligence[huggingface]

# Development
pip install adaptive-intelligence[dev]
```

---

## Quick Start

### Local LLM (Free, Private)

```python
from adaptive_intelligence import AdaptiveAI

# Uses Ollama by default (install: https://ollama.ai, then: ollama pull llama3.2)
engine = AdaptiveAI()
engine.ingest("./documents")
response = engine.ask("What is the total revenue for Q3?")
```

### Cloud LLM

```python
engine = AdaptiveAI(
    llm_backend="openai",
    llm_model="gpt-4o",
    api_key="sk-...",
)
```

### Domain-Specific

```python
engine = AdaptiveAI(
    domain="financial",         # financial, legal, healthcare, technical, operational
    security_level="high",      # standard, high, maximum
)
```

---

## Three Levels of Access

**Level 0 (80% of users)** — Three lines, zero configuration:
```python
engine = AdaptiveAI()
engine.ingest("./docs")
response = engine.ask("query")
```

**Level 1 (15%)** — Domain, provider, security:
```python
engine = AdaptiveAI(
    domain="financial",
    llm_backend="openai",
    api_key="sk-...",
    security_level="high",
)
```

**Level 2 (5%)** — Full control over RL, graph, evaluation:
```python
from adaptive_intelligence.core.config import AdaptiveConfig, RLConfig, GraphConfig

config = AdaptiveConfig(
    rl=RLConfig(warmup_queries=20, exploration_rate=0.15),
    graph=GraphConfig(max_hops=3, conditional_activation=True),
)
engine = AdaptiveAI(config=config)
```

---

## Supported Formats

| Format | Extension | Required Package |
|--------|-----------|-----------------|
| Text / Markdown | .txt, .md | — |
| CSV | .csv | — |
| JSON | .json | — |
| HTML | .html | — |
| XML | .xml | — |
| PDF | .pdf | `PyMuPDF` or `pdfplumber` |
| Word | .docx | `python-docx` |
| Excel | .xlsx | `openpyxl` |
| PowerPoint | .pptx | `python-pptx` |
| Images (OCR) | .png, .jpg | `pytesseract`, `Pillow` |

---

## LLM Providers

| Provider | Backend | Local? | Free? |
|----------|---------|--------|-------|
| Ollama | `ollama` | ✓ | ✓ |
| OpenAI | `openai` | ✗ | ✗ |
| Azure OpenAI | `azure_openai` | ✗ | ✗ |
| Groq | `groq` | ✗ | Free tier |
| Together AI | `together` | ✗ | Free tier |
| HuggingFace | `huggingface` | ✓ | ✓ |
| Any OpenAI-compatible | `custom` | varies | varies |

---

## Monitoring

### Dashboard
```python
print(engine.dashboard())
```
```
┌─────────────────────────────────────────────────────────┐
│  ADAPTIVE INTELLIGENCE DASHBOARD                        │
│  Documents Indexed:       247                           │
│  Queries Processed:        38                           │
│  Average Accuracy:       82.3%                          │
│  Improvement Rate:       +14.7%                         │
│  RL Policy:              Active                         │
│  Exploration Rate:        8.2%                          │
│  Arms Learned:             12                           │
│  Graph Nodes:             156                           │
│  Graph Edges:             284                           │
└─────────────────────────────────────────────────────────┘
```

### Learning Curve
```python
curve = engine.learning_curve()
# [{"query_number": 1, "reward": 0.65, "rolling_avg": 0.65}, ...]
```

### Audit Trail
```python
trail = engine.audit.display_query_trail(response.query_id)
engine.audit.export("audit.json")
```

---

## Architecture

```
Query → Trigger Interpreter → RL Policy Decision → Retrieval
                                    ↓                   ↓
                              Graph Traversal    Vector + Keyword
                              (conditional)      (hybrid RRF)
                                    ↓                   ↓
                              Adaptive Prompt ← ────────┘
                                    ↓
                              LLM Generation
                                    ↓
                              Evaluation Engine
                                    ↓
                    ┌───────────────┼───────────────┐
                    ↓               ↓               ↓
              RL Update      Memory Update    Prompt Evolution
              (reward)       (patterns)       (template scores)
```

---

## Citation

If you use Adaptive Intelligence in research:

```bibtex
@software{venkatkumar2026adaptive,
  title={Adaptive Intelligence: Self-Improving Retrieval Orchestration via Evaluation-Driven Policy Learning},
  author={Venkatkumar, VK},
  year={2026},
  url={https://github.com/VK-Ant/adaptive-intelligence}
}
```

---

## License

Apache License 2.0. See [LICENSE](LICENSE).

---

Built by [Venkatkumar_VK](https://www.linkedin.com/in/venkatkumarvk/)
