# Adaptive Intelligence — API Reference

## Core API

### `AdaptiveAI`

The main engine class. Three levels of access:

```python
# Level 0: Zero config
engine = AdaptiveAI()

# Level 1: Provider + domain
engine = AdaptiveAI(llm_backend="openai", api_key="...", domain="financial")

# Level 2: Full config
engine = AdaptiveAI(config=AdaptiveConfig(...))
```

#### `AdaptiveAI.__init__(config=None, **kwargs)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config` | `AdaptiveConfig` | `None` | Full configuration object |
| `llm_backend` | `str` | `"ollama"` | Provider: `ollama`, `openai`, `groq`, `azure_openai`, `huggingface` |
| `llm_model` | `str` | `"llama3.2"` | Model name |
| `api_key` | `str` | `None` | API key for cloud providers |
| `base_url` | `str` | `None` | Custom API endpoint |
| `domain` | `str` | `"general"` | Domain: `financial`, `legal`, `healthcare`, `technical`, `operational` |
| `security_level` | `str` | `"standard"` | Security: `standard`, `high`, `maximum` |
| `storage_dir` | `str` | `"./.adaptive_intelligence"` | Directory for persisted state |
| `log_level` | `str` | `"INFO"` | Logging level |

#### `engine.ingest(source) → IngestionStats`

Ingest documents from a file, directory, or list of paths.

```python
stats = engine.ingest("./documents")
stats = engine.ingest("/path/to/report.pdf")
stats = engine.ingest(["./reports/", "./filings/10k.pdf"])
```

**Supported formats:** TXT, MD, CSV, JSON, HTML, XML, PDF, DOCX, XLSX, PPTX, PNG/JPG (OCR)

#### `engine.ask(query, priority=None, depth=None) → AdaptiveResponse`

Ask a question over ingested documents.

```python
response = engine.ask("What are the key risks?")
response = engine.ask("Compare Q1 and Q2", priority="accuracy", depth=15)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | required | The question |
| `priority` | `str` | `None` | Hint: `"accuracy"`, `"speed"`, `"detail"` |
| `depth` | `int` | `None` | Override retrieval depth |

#### `engine.dashboard() → str`

Returns a formatted dashboard string with system metrics.

#### `engine.learning_curve() → list[dict]`

Returns learning curve data: `[{"query_number": 1, "reward": 0.65, "rolling_avg": 0.65}, ...]`

#### `engine.status() → dict`

Returns complete system status.

#### `engine.reset()`

Clears all state (indexes, memory, RL policy, graph).

---

## Response Object

### `AdaptiveResponse`

Returned by `engine.ask()`.

| Attribute | Type | Description |
|-----------|------|-------------|
| `answer` | `str` | The synthesized answer |
| `confidence` | `float` | Confidence score (0-1) |
| `citations` | `list[Citation]` | Source citations |
| `evaluation` | `EvaluationResult` | Quality metrics |
| `retrieval_info` | `RetrievalInfo` | Retrieval strategy details |
| `policy_decision` | `PolicyDecision` | What the RL policy chose |
| `query_analysis` | `dict` | Query understanding output |
| `query_id` | `str` | Unique query identifier |
| `raw_chunks` | `list[dict]` | Retrieved source chunks |

```python
response.display()              # Full formatted output
response.retrieval_strategy     # e.g. "hybrid + graph(2-hop) + depth=8"
```

### `EvaluationResult`

| Attribute | Type | Description |
|-----------|------|-------------|
| `faithfulness` | `float` | Source grounding (0-1) |
| `relevance` | `float` | Query addressing (0-1) |
| `citation_accuracy` | `float` | Citation quality (0-1) |
| `hallucination_risk` | `float` | Fabrication risk (0-1, lower=better) |
| `retrieval_precision` | `float` | Relevant chunks / total chunks |
| `retrieval_recall` | `float` | Query terms covered |
| `composite_score` | `float` | Weighted composite (= RL reward) |
| `confidence_level` | `str` | `"high"`, `"medium"`, `"low"` |

```python
response.evaluation.display()   # Formatted bar chart
```

---

## Configuration

### `AdaptiveConfig`

Full configuration with nested config objects:

| Nested Config | Class | Key Parameters |
|---------------|-------|----------------|
| `config.rl` | `RLConfig` | `warmup_queries`, `exploration_rate`, `exploration_decay`, `algorithm` |
| `config.graph` | `GraphConfig` | `enabled`, `conditional_activation`, `max_hops` |
| `config.evaluation` | `EvaluationConfig` | `faithfulness_weight`, `relevance_weight`, `enable_llm_judge` |
| `config.chunking` | `ChunkingConfig` | `chunk_size`, `chunk_overlap`, `min_chunk_size` |

---

## Submodule Access

All internal components are accessible for advanced use:

```python
engine.rl              # RLPolicyEngine — get_stats(), get_learning_curve()
engine.graph           # KnowledgeGraph — get_stats(), traverse()
engine.evaluation      # EvaluationEngine — get_history(), get_average_score()
engine.memory          # LearningMemory — get_learning_summary(), get_stats()
engine.audit           # AuditTrail — display_query_trail(), export()
engine.prompts         # AdaptivePromptEngine — library access
engine.trigger         # TriggerInterpreter — analyze()
engine.vector_index    # VectorIndex — search(), count()
engine.keyword_index   # KeywordIndex — search(), count()
```
