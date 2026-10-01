# adaptive-intelligence v4.0.8 — Complete Technical Documentation

**Version:** 4.0.8 | **Modules:** 34 | **Lines:** 8,900+ | **Tests:** 99 | **License:** Apache 2.0 | **Author:** Venkatkumar Rajan

---

## 1. Architecture Overview

Five-phase pipeline with three feedback loops:

**Phase 1: Understand** — Trigger Interpreter classifies query (type, complexity, domain, entities) without LLM. Memory lookup searches past context.

**Phase 2: Decide** — RL policy selects retrieval route (6 options), depth, graph activation, and tool selection.

**Phase 3: Retrieve** — Vector (ChromaDB), BM25 keyword, or page index. Hybrid uses RRF fusion. Conditional graph via 5-signal gate. Cross-encoder reranks top-20.

**Phase 4: Generate** — Context engineer assembles full context window with token budgets. LLM generates answer.

**Phase 5: Learn** — 6-metric evaluation. Harness evaluates every decision. Loop engineer shapes reward. Policy updates. Memory updates.

**Three feedback loops:**
- Inner (~1ms): RL evaluates answer, updates policy
- Middle (~2ms): Harness evaluates every decision, shapes reward
- Outer (across sessions): Memory persists patterns, loop adapts exploration

---

## 2. Installation

```
pip install adaptive-intelligence                # Zero deps
pip install adaptive-intelligence[vector]         # + ChromaDB
pip install adaptive-intelligence[openai]         # + OpenAI-compatible APIs
pip install adaptive-intelligence[huggingface]    # + Local HuggingFace
pip install adaptive-intelligence[mcp]            # + Official MCP SDK
pip install adaptive-intelligence[pdf]            # + PyMuPDF
pip install adaptive-intelligence[docx]           # + python-docx
pip install adaptive-intelligence[xlsx]           # + openpyxl
pip install adaptive-intelligence[pptx]           # + python-pptx
pip install adaptive-intelligence[sql]            # + SQLAlchemy
pip install adaptive-intelligence[rerank]         # + sentence-transformers
pip install adaptive-intelligence[ocr]            # + Tesseract + Pillow
pip install adaptive-intelligence[all]            # Everything
```

---

## 3. Core Engine (AdaptiveAI)

### Constructor

```python
engine = AdaptiveAI(
    llm_backend="ollama",        # "ollama","openai","huggingface","none"
    llm_model="llama3.1",        # Model name
    api_key=None,                # API key for cloud providers
    base_url=None,               # Custom API URL
    azure_endpoint=None,         # Azure OpenAI endpoint
    domain="general",            # "general","financial","legal","healthcare","technical","scientific","operational"
    rl_algorithm="thompson",     # "thompson" or "ppo"
    pretrained_policy=True,      # Domain-specific pretrained policy
    warmup=15,                   # Queries before learned policy
    exploration_rate=0.20,       # Base exploration rate
    vectorless=False,            # No ChromaDB, page-level BM25
    reranking=False,             # Cross-encoder reranking
    memory=True,                 # Persistent memory
    context_engineering=True,    # Optimize context window
    token_budget=4096,           # Context window budget
    agentic_rounds=3,            # Max agentic rounds
    agentic_threshold=0.7,       # Confidence threshold
    storage_dir=".adaptive_intelligence",
    log_level="INFO",
)
```

### Methods (21 total)

**engine.ingest(source, parallel=False, workers=4, tables=None, query=None)**
Ingest documents. Supports incremental ingestion. Returns IngestionStats.
- source: file/dir path, glob pattern, SQL connection string, or list of paths
- Formats: .txt, .pdf, .docx, .xlsx, .csv, .json, .html, .md, .pptx, .xml, SQL

**engine.ask(query, priority=None, depth=None, system_prompt=None, output_format=None, schema=None, mode=None)**
Ask a question. Returns AdaptiveResponse.
- priority: "accuracy", "speed", "detail"
- output_format: "json", "csv", "yaml"
- mode: "agentic" for multi-round retrieval

**engine.add_tool(name, description="", server=None, function=None, api_endpoint=None, api_key=None)**
Register external tool. RL learns which to call per query type.

**engine.remove_tool(name)**
Remove a registered tool.

**engine.list_tools()**
List all tools with stats (calls, success_rate, avg_latency).

**engine.serve_mcp(port=8080)**
Start MCP server. Uses official SDK if installed, HTTP fallback otherwise.

**engine.remember(key, value, category="fact")**
Store fact in long-term memory. Persists to disk.

**engine.recall(key)**
Retrieve fact by key. Returns value or None.

**engine.search_memory(query, top_k=5)**
Search memory with recency and frequency boosting.

**engine.feedback(query_id, rating, reason=None)**
Explicit feedback. "good" = +0.2 reward. "bad" = -0.3 + prompt evolution.

**engine.export_policy(filepath) / engine.import_policy(filepath)**
Transfer learned policy between deployments.

**engine.set_system_prompt(prompt)**
Custom system prompt. None resets to domain default.

**engine.dashboard()**
Formatted system status string.

**engine.status()**
System status dict.

**engine.learning_curve()**
List of (query_number, score) tuples.

**engine.enable_ab_test(policy_a, policy_b, test_queries=50) / engine.ab_results()**
A/B test between RL policies.

**engine.update(doc_id) / engine.remove(doc_id)**
Update or remove specific document.

**engine.reset()**
Reset all state. Irreversible.

---

## 4. Configuration

### LLMBackend enum
ollama, openai, azure_openai, anthropic, huggingface, groq, together, custom, none

### Domain enum
general, financial, legal, technical, healthcare, scientific, operational

Affects system prompt persona, pretrained policy, evaluation weights.

---

## 5. Response Object (AdaptiveResponse)

| Field | Type | Description |
|-------|------|-------------|
| answer | str | Generated answer |
| confidence | float | Composite score (0-1) |
| citations | List[Citation] | Source docs and pages |
| evaluation | EvaluationResult | 6-metric breakdown |
| retrieval_info | RetrievalInfo | Strategy, chunks, graph |
| policy_decision | PolicyDecision | RL decision details |
| query_analysis | Dict | Trigger interpreter output |
| query_id | str | ID for feedback |
| retrieval_strategy | str | Human-readable strategy |
| harness_report | HarnessReport | Per-decision evaluation (v4.0.7) |
| harness_efficiency | float | Helpful decisions ratio (v4.0.7) |
| shaped_reward | float | Harness-shaped reward (v4.0.7) |
| exploration_rate | float | Current exploration rate (v4.0.7) |
| agent_steps | List[AgentStep] | Agentic steps (when mode="agentic") |
| agent_rounds | int | Agentic rounds used |
| tools_called | List[str] | Tools called |

### Citation
source_document (str), page (int), chunk_text (str), relevance_score (float)

### EvaluationResult
faithfulness (30%), relevance (20%), citation_accuracy (20%), retrieval_precision (10%), retrieval_recall (10%), hallucination_risk (-2%), composite_score

### PolicyDecision
retrieval_route, retrieval_depth, graph_activation, graph_depth, model_used, prompt_template, was_exploration, policy_confidence

---

## 6. Ingestion Pipeline

### Parser (10+ formats)
.txt, .md, .html, .xml (built-in), .pdf (PyMuPDF), .docx (python-docx), .xlsx/.csv (openpyxl/csv), .pptx (python-pptx), .json (built-in), SQL (SQLAlchemy)

### Chunker
Semantic chunking on paragraph boundaries, page mode, quality scoring, SimHash dedup, configurable overlap.

### Incremental ingestion
Call engine.ingest() multiple times. RL, graph, memory all preserved.

---

## 7. Retrieval System

### 6 Retrieval Routes
| Route | Best for |
|-------|----------|
| keyword_only | Factual queries, specific terms |
| vector_only | Semantic queries, paraphrased |
| hybrid | Most queries (RRF fusion) |
| table_first | Numbers, metrics, structured data |
| graph_first | Entity relationships |
| graph_hybrid | Complex relational queries |

### Cross-Encoder Reranking
When reranking=True, top-20 chunks re-scored. Falls back to keyword-based if model unavailable.

---

## 8. RL Policy Engine

### Thompson Sampling (default)
Beta(alpha, beta) per (query-type, strategy) pair. Sample and choose highest. Update based on evaluation score.

### PPO (alternative)
Tabular PPO with Q-table, softmax action selection, clipped updates.

### Pretrained Policies
Skip warmup with domain-specific starting policies: financial, legal, healthcare.

### Policy Transfer
engine.export_policy("learned.json") / engine.import_policy("learned.json")

---

## 9. Knowledge Graph

### Auto-built during ingestion
Entity co-occurrences create connected nodes. No manual setup.

### 5-Signal Activation Gate
| Signal | Score |
|--------|-------|
| Relationship words in query | +2 |
| Trigger interpreter flags graph | +2 |
| Entity density >= 2 | +1 |
| Complex/multi-hop query | +1 |
| Graph helped before | +1 |

Threshold >= 2: graph ON. Below: OFF. ~70% queries skip graph.

### Traversal
BFS from detected entities. Configurable depth (1-3 hops).

---

## 10. Evaluation Engine

### 6 Metrics (no ground truth required)
| Metric | Weight | Measurement |
|--------|--------|-------------|
| Faithfulness | 30% | Answer-to-source token overlap |
| Relevance | 20% | Query-to-answer token overlap |
| Citation accuracy | 20% | Sources referenced |
| Retrieval precision | 10% | Relevant chunks / total chunks |
| Retrieval recall | 10% | Query terms found in chunks |
| Hallucination risk | -2% | Answer sentences with no source |

---

## 11. Context Engineering (v4)

### Token Budget
| Component | Budget | Purpose |
|-----------|--------|---------|
| System prompt | 200 | Domain persona |
| Memory | 300 | Long-term memory entries |
| History | 400 | Recent conversation |
| Chunks | 2,500 | Retrieved documents |
| Tool results | 500 | External tool outputs |

### Personas
general, financial, legal, healthcare, technical — each with domain-specific instructions.

### Methods
**build_context(query, chunks, memory_entries, session_history, tool_results, domain, custom_system_prompt)**
Returns ContextWindow with .assemble() for final prompt.

---

## 12. MCP Integration (v4)

### ToolRegistry
**add_tool(name, description, server/function/api_endpoint)** — Register tool.
**call_tool(name, query, params)** — Call tool. Returns ToolResult.
**select_tools(query)** — Suggest tools based on keyword matching.
**list_tools() / remove_tool(name) / get_stats()**

### MCPServer
**serve(port=8080)** — Start MCP server. Official SDK (stdio) or HTTP fallback.

Tool exposed: "adaptive_intelligence_search" with query input schema.

---

## 13. Agentic Workflow (v4)

### Flow
Query → Retrieve (RL-guided) → Evaluate confidence → Call tools → Refine query → Retrieve again → Answer

### Methods
**run(query, engine, tool_registry)** — Full agentic workflow. Returns AgentResult.
**run_simple(query, engine)** — Single-round fallback.

### AgentResult
answer, steps (List[AgentStep]), total_rounds, total_latency, tools_called

### AgentStep
action ("retrieve"/"tool_call"/"refine"/"answer"), query, tool_name, result, latency

---

## 14. Persistent Memory (v4)

### Methods
**remember(key, value, category)** — Store fact. Categories: fact, pattern, preference, context.
**recall(key)** — Retrieve by key.
**search(query, category, top_k)** — Keyword search with recency/frequency boosting.
**learn_pattern(query_type, domain, strategy, score)** — Store routing pattern.
**get_best_strategy(query_type, domain)** — Get best known strategy.
**add_session_context(query, answer, strategy)** — Store conversation turn.
**save() / clear() / get_stats()**

Storage: JSON file in storage_dir. Auto-evicts at 10,000 entries.

---

## 15. Harness Agent (v4.0.7)

### What It Evaluates
| Decision | Check | Impact |
|----------|-------|--------|
| Route | Matches query type? | -0.15 to +0.30 |
| Depth | Chunk utilization ratio | -0.10 to +0.10 |
| Graph | Graph context in answer? | -0.10 to +0.15 |
| Tool (each) | Tool result in answer? | -0.10 to +0.12 |
| Agentic round | Added new info? | -0.05 to +0.10 |
| Memory | Entries contributed? | -0.03 to +0.10 |

### Methods
**evaluate_pipeline(query, answer, chunks_used, chunks_retrieved, route_chosen, depth, graph_activated, graph_context, tools_called, agentic_steps, memory_entries, answer_score, latency)**
Returns HarnessReport.

**get_granular_rewards(report)** — Per-decision rewards dict for RL.

**get_stats()** — Total reports, avg efficiency, per-decision breakdown.

### HarnessReport
query, answer_score, decisions (List[DecisionScore]), total_waste, efficiency, recommendations

### DecisionScore
decision, name, helped (bool), impact (float), detail (str)

---

## 16. Loop Engineering (v4.0.7)

### What It Adapts
| Without | With |
|---------|------|
| Fixed 15-query warmup | Per-domain warmup (0-15) |
| Fixed exploration decay | Per-domain rates (3%-25%) |
| Raw answer score reward | Shaped by harness signals |
| No convergence detection | Auto-detected per domain |

### Methods
**should_explore(domain, query_type)** — Explore or exploit?
**get_exploration_rate(domain, query_type)** — Current rate (0-1).
**is_warmup(domain, query_type)** — Still in warmup?
**update(domain, query_type, strategy, score, harness_rewards)** — Update state.
**shape_reward(base_reward, harness_rewards)** — Shape reward: answer (60%) + combined (20%) + efficiency (20%).
**get_stats()** — Per-domain stats with convergence status.

### Convergence
Detected when: query_count >= 30 AND score_variance < 0.02. Exploration drops to 3%.

---

## 17. LM Cache (v4.0.8)

**Module:** `adaptive_intelligence/cache/__init__.py`

Caches LLM responses to skip redundant calls. Two strategies:

**ExactCache** — Normalized query string lookup. Dict-based, zero dependencies. Identical queries (ignoring case/whitespace) return instantly.

**SemanticCache** — Embedding cosine similarity. Similar queries (above threshold) return cached response. Built-in character trigram embeddings (zero deps). Plug in sentence-transformers or OpenAI via `embed_fn`.

### Usage

```python
engine = AdaptiveAI(cache=True)                                    # Exact only
engine = AdaptiveAI(cache=True, cache_mode="semantic")             # Semantic only
engine = AdaptiveAI(cache=True, cache_mode="both")                 # Both
engine = AdaptiveAI(cache=True, cache_threshold=0.90)              # Lower threshold
engine = AdaptiveAI(cache=True, cache_ttl=7200)                    # 2 hour TTL
engine = AdaptiveAI(cache=True, cache_max_entries=5000)            # Larger cache
```

### External Adapters

```python
from adaptive_intelligence.cache import RedisAdapter, CacheAdapter

# Redis
engine = AdaptiveAI(cache=True, cache_adapter=RedisAdapter(host="localhost"))

# Custom
class MyAdapter(CacheAdapter):
    def get(self, key): ...
    def set(self, key, value, ttl): ...
    def delete(self, key): ...
```

### Classes

**CacheManager** — Unified manager. `lookup(query)` checks exact then semantic. `store(query, answer, confidence, strategy)` stores in both.

**CacheEntry** — Stores query, answer, confidence, strategy, TTL, hit count, embedding. Serializable to/from dict for persistence.

**CacheStats** — Tracks total queries, hits/misses, exact/semantic splits, hit rate, avg similarity.

### Pipeline Integration

Cache check happens after query understanding (Step 1), before RL decision (Step 2). Only responses with `composite_score > 0.5` are cached. Cache persists to disk on shutdown.

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| cache | False | Enable caching |
| cache_mode | "exact" | "exact", "semantic", or "both" |
| cache_threshold | 0.92 | Semantic similarity threshold |
| cache_max_entries | 1000 | Maximum cached entries |
| cache_ttl | 3600.0 | Time-to-live in seconds |
| cache_embed_fn | None | Custom embedding function |
| cache_adapter | None | External backend adapter |

---

## 18. LLM Providers

### Free (no credit card)
| Provider | backend | base_url |
|----------|---------|----------|
| Ollama | ollama | local |
| NVIDIA NIM | openai | https://integrate.api.nvidia.com/v1 |
| Groq | openai | https://api.groq.com/openai/v1 |
| Google Gemini | openai | https://generativelanguage.googleapis.com/v1beta/openai/ |
| Together AI | openai | https://api.together.xyz/v1 |
| Fireworks AI | openai | https://api.fireworks.ai/inference/v1 |
| HuggingFace | huggingface | local GPU |
| No LLM | none | retrieval only |

### Paid
| Provider | backend | base_url |
|----------|---------|----------|
| OpenAI | openai | default |
| Grok (xAI) | openai | https://api.x.ai/v1 |
| Azure OpenAI | azure_openai | azure_endpoint |

### Self-hosted
vLLM or any OpenAI-compatible server via base_url.

---

## 19. Security

**Audit trail:** Every query logged with strategy, scores, timing.
**PII scanning:** Basic detection during ingestion.
**Crash recovery:** Auto-checkpoint on shutdown (SIGINT/SIGTERM). Loads on restart.

---

## 20. Project Structure

```
adaptive_intelligence/          34 modules, 8,900+ lines
    core/                       Engine, config, response
    ingestion/                  Parser, chunker, ingestion engine
    indexes/                    Vector (ChromaDB), BM25, page index
    query/                      Trigger interpreter
    rl/                         Thompson Sampling, PPO, reranker, multi-query, pretrained
    graph/                      Knowledge graph, 5-signal gate
    prompts/                    Adaptive prompt engine
    evaluation/                 6-metric evaluation
    llm/                        LLM providers
    memory/                     Learning memory, persistent memory
    context/                    Context engineering (v4)
    mcp/                        MCP server, tool registry (v4)
    agentic/                    Agentic workflow (v4)
    harness/                    Pipeline decision evaluation (v4.0.7)
    loop/                       Loop engineering (v4.0.7)
    cache/                      LM cache — exact + semantic (v4.0.8)
    security/                   Audit, PII
    utils/                      Logging, timing, IDs
```

---

## 21. Version History

| Version | Focus | Key additions |
|---------|-------|--------------|
| v1.0.0 | Core | RL routing, conditional graph, 6-metric evaluation |
| v2.0.0 | Production | Vectorless, crash recovery, SQL, structured output, 10+ providers |
| v3.0.0 | Intelligence | PPO, reranking, multi-query, pretrained policies, transfer learning |
| v4.0.0 | Context+Agentic | Context engineering, MCP, agentic workflow, memory, incremental learning |
| v4.0.5 | Bug fix | Fixed config.domain reference |
| v4.0.6 | MCP SDK | Official MCP SDK, client retry, server warmup |
| v4.0.7 | Faster Learning | Harness agent, loop engineering, reward shaping |
| v4.0.8 | LM Cache | Exact + semantic cache, Redis adapter, external backends |

### Planned
- v5 (Aug 2026): Spatial RAG via MCP, harness improvements
- v6 (Late 2026): Responsible AI — bias detection, PII redaction, compliance guardrails

---

## Links

- PyPI: https://pypi.org/project/adaptive-intelligence/
- GitHub: https://github.com/VK-Ant/adaptive-intelligence
- Paper: https://www.researchgate.net/publication/405076088
- Medium: https://medium.com/@VK_Venkatkumar/134fa927f25a
- Also: llmevalkit (https://pypi.org/project/llmevalkit/) — 61 metrics
