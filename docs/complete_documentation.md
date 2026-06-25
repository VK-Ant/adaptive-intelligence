# adaptive-intelligence — Complete Documentation

Version 3.0.0 | Apache 2.0 | Author: Venkatkumar Rajan

---

## Module Overview

| S.No. | Module | Files | Purpose | Version |
|-------|--------|-------|---------|---------|
| 1 | core/engine | engine.py | Main AdaptiveAI orchestrator | v1+ |
| 2 | core/config | config.py | All configuration classes | v1+ |
| 3 | core/response | response.py | Response, Citation, Evaluation objects | v1+ |
| 4 | ingestion/parser | parser.py | Document parsing (10+ formats) | v1, hardened v2 |
| 5 | ingestion/chunker | chunker.py | Chunking, quality scoring, dedup | v1, enhanced v2 |
| 6 | ingestion/engine | engine.py | Ingestion orchestration, incremental, parallel | v1, enhanced v2 |
| 7 | indexes/vector_index | vector_index.py | ChromaDB semantic search | v1 |
| 8 | indexes/keyword_index | keyword_index.py | BM25 lexical search | v1, persistence v2 |
| 9 | indexes/page_index | page_index.py | Page-level BM25 (vectorless) | v2 |
| 10 | indexes/base | base.py | BaseIndex ABC | v1 |
| 11 | query | __init__.py | TriggerInterpreter, query classification | v1 |
| 12 | rl | __init__.py | Thompson Sampling RL policy | v1, export/import v3 |
| 13 | rl/ppo | ppo.py | PPO algorithm option | v3 |
| 14 | rl/reranker | reranker.py | Cross-encoder reranking | v3 |
| 15 | rl/multi_query | multi_query.py | Multi-query decomposition | v3 |
| 16 | rl/pretrained | pretrained.py | Pre-trained domain policies, transfer learning | v3 |
| 17 | graph | __init__.py | Knowledge graph, conditional activation | v1, persistence v2 |
| 18 | prompts | __init__.py | Adaptive prompt engine, evolution | v1 |
| 19 | evaluation | __init__.py | 6-metric evaluation engine | v1, feedback v2 |
| 20 | llm | __init__.py | LLM providers (Ollama, OpenAI, HuggingFace) | v1, SDK v2 |
| 21 | memory | __init__.py | Learning memory, routing patterns | v1, feedback v2 |
| 22 | security | __init__.py | Audit trail, PII scanning | v1 |
| 23 | utils | __init__.py | Logging, timing, ID generation | v1 |

---

## 1. AdaptiveAI (core/engine.py)

Main entry point. 3 lines to start.

### Constructor

```python
AdaptiveAI(config=None, **kwargs)
```

| Parameter | Type | Default | Version | Description |
|-----------|------|---------|---------|-------------|
| config | AdaptiveConfig | None | v1 | Full configuration object |
| llm_backend | str | "ollama" | v1 | Provider: ollama, openai, huggingface, none |
| llm_model | str | "llama3.2" | v1 | Model name |
| api_key | str | None | v1 | API key for cloud providers |
| base_url | str | None | v1 | Custom API endpoint |
| domain | str | "general" | v1 | Domain: financial, legal, healthcare, technical |
| security_level | str | "standard" | v1 | Security: standard, high, maximum |
| storage_dir | str | ".adaptive_intelligence" | v1 | Directory for persisted state |
| log_level | str | "INFO" | v1 | Logging level |
| system_prompt | str | None | v2 | Custom system prompt for all queries |
| vectorless | bool | False | v2 | No ChromaDB, no embeddings, pure BM25 |
| warmup_queries | int | 15 | v2 | Queries before RL takes over |
| min_chunk_quality | float | 0.3 | v2 | Minimum chunk quality score |
| deduplicate | bool | False | v2 | Enable SimHash deduplication |
| checkpoint_minutes | int | 5 | v2 | Auto-checkpoint interval |
| rl_algorithm | str | "thompson" | v3 | RL algorithm: "thompson" or "ppo" |
| reranking | bool | False | v3 | Enable cross-encoder reranking |
| multi_query | bool | True | v3 | Enable multi-query decomposition |
| pretrained_policy | bool | False | v3 | Load pre-trained domain policy |

### Methods

| Method | Returns | Version | Description |
|--------|---------|---------|-------------|
| ingest(source, parallel, workers, on_progress, tables, query) | IngestionStats | v1, enhanced v2 | Ingest files, directories, or SQL |
| ask(query, priority, depth, system_prompt, output_format, schema) | AdaptiveResponse | v1, enhanced v2/v3 | Ask a question |
| remove(filename) | int | v2 | Remove document from all indexes |
| update(filepath) | IngestionStats | v2 | Re-ingest single document |
| feedback(query_id, rating, reason) | None | v2 | Thumbs up/down updates RL reward |
| set_system_prompt(prompt) | None | v2 | Set custom system prompt |
| export_policy(filepath) | None | v3 | Export learned RL policy |
| import_policy(filepath) | None | v3 | Import RL policy from file |
| enable_ab_test(policy_a, policy_b) | None | v3 | Start A/B testing |
| ab_results() | dict | v3 | Get A/B test results |
| dashboard() | str | v1 | Formatted system dashboard |
| learning_curve() | list[dict] | v1 | Learning curve data |
| status() | dict | v1 | Complete system status |
| reset() | None | v1 | Clear all state |

---

## 2. Configuration (core/config.py)

### Enums

| Enum | Values | Description |
|------|--------|-------------|
| LLMBackend | ollama, openai, azure_openai, anthropic, huggingface, groq, together, custom, none | LLM provider selection |
| SecurityLevel | standard, high, maximum | Security mode |
| Domain | general, financial, legal, healthcare, technical, operational | Domain specialization |

### Config Classes

**ChunkingConfig**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| chunk_size | int | 512 | Characters per chunk |
| chunk_overlap | int | 50 | Overlap between chunks |
| min_chunk_size | int | 50 | Minimum chunk size |

**RLConfig**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| warmup_queries | int | 15 | Queries before RL takes over |
| exploration_rate | float | 0.20 | Initial exploration probability |
| min_exploration_rate | float | 0.05 | Exploration floor |
| exploration_decay | float | 0.995 | Decay per query |
| algorithm | str | "thompson_sampling" | RL algorithm |

**GraphConfig**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| enabled | bool | True | Enable graph |
| conditional_activation | bool | True | Use 5-signal gate |
| max_hops | int | 3 | Maximum traversal depth |
| min_entity_count | int | 2 | Entities needed for activation |
| activation_threshold | float | 0.7 | Historical success threshold |

**EvaluationConfig**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| faithfulness_weight | float | 0.30 | Weight in reward formula |
| relevance_weight | float | 0.20 | Weight in reward formula |
| citation_weight | float | 0.20 | Weight in reward formula |
| precision_weight | float | 0.10 | Weight in reward formula |
| recall_weight | float | 0.10 | Weight in reward formula |
| enable_llm_judge | bool | False | Use LLM for secondary eval |
| llm_judge_threshold | float | 0.8 | Score below which LLM judge triggers |

---

## 3. Response Objects (core/response.py)

**AdaptiveResponse**

| Field | Type | Description |
|-------|------|-------------|
| answer | str | Synthesized answer |
| confidence | float | Composite evaluation score (0-1) |
| citations | list[Citation] | Source citations with page numbers |
| evaluation | EvaluationResult | Quality metrics |
| retrieval_info | RetrievalInfo | Strategy details |
| policy_decision | PolicyDecision | RL decision details |
| query_analysis | dict | Query understanding output |
| query_id | str | Unique query identifier |
| raw_chunks | list[dict] | Retrieved source chunks |
| structured | Any | Parsed JSON/CSV/DataFrame (v2) |

**Citation**

| Field | Type | Description |
|-------|------|-------------|
| text | str | Cited text excerpt |
| source_document | str | Source filename |
| chunk_id | str | Chunk identifier |
| confidence | float | Citation confidence (0-1) |
| page | int | Page number (v2) |

**EvaluationResult**

| Field | Type | Description |
|-------|------|-------------|
| faithfulness | float | Source grounding (0-1) |
| relevance | float | Query addressing (0-1) |
| citation_accuracy | float | Citation quality (0-1) |
| hallucination_risk | float | Fabrication risk (0-1) |
| retrieval_precision | float | Relevant chunks / total |
| retrieval_recall | float | Query terms covered |
| composite_score | float | Weighted composite = RL reward |
| confidence_level | str | "high" / "medium" / "low" |

---

## 4. Document Parser (ingestion/parser.py)

### DocumentParser

| Method | Returns | Description |
|--------|---------|-------------|
| parse(filepath) | ParsedDocument | Parse single document |
| parse_directory(directory, recursive) | list[ParsedDocument] | Parse all supported files |

**Supported formats:** txt, md, csv, tsv, json, html, xml, pdf, docx, xlsx, pptx, png, jpg

**v2 edge case handling:**

| Format | Edge Cases Handled |
|--------|-------------------|
| PDF | Password-protected, corrupted, scanned (auto-OCR), multi-column, header/footer stripping, page-spanning tables |
| Excel | Merged cells, hidden sheets, formula evaluation, empty row detection, large file streaming, date normalization |
| CSV | Auto-delimiter detection, encoding fallback, BOM stripping, malformed rows, headerless files |
| DOCX | Track changes, comments, nested tables, password protection |
| PPTX | Speaker notes, grouped shapes, SmartArt, password protection |

### SQLConnector (v2)

| Method | Returns | Description |
|--------|---------|-------------|
| parse(connection_string, tables, query) | list[ParsedDocument] | Ingest from SQL database |

Supports: PostgreSQL, MySQL, SQLite via SQLAlchemy.

### ParsedDocument

| Field | Type | Description |
|-------|------|-------------|
| doc_id | str | Unique document ID |
| filename | str | Source filename |
| content | str | Full text content |
| file_type | str | Detected format |
| tables | list[dict] | Extracted tables |
| pages | list[PageContent] | Per-page content |
| page_count | int | Total pages |
| parse_errors | list[str] | Any parse errors |

---

## 5. Chunker (ingestion/chunker.py)

### Chunk

| Field | Type | Description |
|-------|------|-------------|
| chunk_id | str | Unique chunk ID |
| doc_id | str | Parent document ID |
| content | str | Chunk text |
| source_document | str | Source filename |
| is_table | bool | Table chunk flag |
| is_page | bool | Page-level chunk (v2) |
| page_number | int | Source page number (v2) |
| quality_score | float | Quality score (v2) |
| prev_chunk_id | str | Previous chunk link (v2) |
| next_chunk_id | str | Next chunk link (v2) |

### DocumentChunker

| Parameter | Type | Default | Version | Description |
|-----------|------|---------|---------|-------------|
| config | ChunkingConfig | None | v1 | Chunking settings |
| page_mode | bool | False | v2 | Full page as retrieval unit |
| min_quality | float | 0.3 | v2 | Reject below threshold |
| deduplicate | bool | False | v2 | SimHash dedup |

### ChunkQualityScorer (v2)

Scores chunks 0-1 based on: word count, whitespace ratio, repetitive content, alphabetic ratio. Chunks below threshold rejected before indexing.

### ChunkDeduplicator (v2)

SimHash-based near-duplicate detection. Configurable similarity threshold (default 0.9).

---

## 6. Ingestion Engine (ingestion/engine.py)

### IngestionEngine

| Method | Returns | Version | Description |
|--------|---------|---------|-------------|
| ingest(source, parallel, workers, on_progress, tables, query) | IngestionStats | v1, enhanced v2 | Ingest from any source |
| remove(filename) | int | v2 | Remove document |
| update(filepath) | IngestionStats | v2 | Re-ingest single file |
| get_chunks() | list[Chunk] | v1 | All current chunks |
| get_documents() | list[ParsedDocument] | v1 | All parsed documents |
| get_document_list() | list[dict] | v2 | Document summary list |
| clear() | None | v1 | Clear all state |

### IngestionStats

| Field | Type | Description |
|-------|------|-------------|
| total_files | int | Files attempted |
| successful | int | Files ingested |
| failed | int | Files failed |
| total_chunks | int | Chunks created |
| total_tables | int | Tables extracted |
| file_types | dict | Count per format |
| errors | list[str] | Error messages |
| duration_seconds | float | Total time |

---

## 7. Indexes

### BaseIndex (ABC)

| Method | Returns | Description |
|--------|---------|-------------|
| add(chunks) | int | Add chunks to index |
| search(query, top_k) | list[RetrievalResult] | Search index |
| clear() | None | Clear index |
| count() | int | Number of entries |

### VectorIndex (v1)

ChromaDB-based semantic search. Persists automatically via PersistentClient.

### KeywordIndex (v1, persistence v2)

BM25 lexical search. In-memory. Save/load via pickle (v2).

### PageIndex (v2)

Page-level BM25 for vectorless mode. Each page is a retrieval unit. Page-number citations.

---

## 8. Query Understanding (query/__init__.py)

### TriggerInterpreter

Classifies queries without LLM call. Output becomes RL state.

| Method | Returns | Description |
|--------|---------|-------------|
| analyze(query) | QueryAnalysis | Full query analysis |

### QueryAnalysis

| Field | Type | Description |
|-------|------|-------------|
| query_type | QueryType | factual, relational, analytical, comparative, etc. |
| complexity | QueryComplexity | simple, moderate, complex, multi_hop |
| domain | str | Detected domain |
| entities | list[str] | Extracted entities |
| relationships | list[str] | Detected relationships |
| temporal_references | list[str] | Time references |
| intent | QueryIntent | answer, compare, extract, analyze, etc. |
| suggested_indexes | list[str] | Recommended retrieval indexes |
| graph_needed | bool | Graph traversal recommended |
| table_needed | bool | Structured data needed |
| confidence | float | Analysis confidence |

---

## 9. RL Policy Engine (rl/__init__.py)

### RLPolicyEngine

| Method | Returns | Version | Description |
|--------|---------|---------|-------------|
| decide(query_analysis) | PolicyAction | v1 | Select retrieval strategy |
| update(query_id, analysis, action, reward) | None | v1 | Update policy from reward |
| get_stats() | dict | v1 | Policy statistics |
| get_learning_curve() | list[dict] | v1 | Learning curve data |
| export_policy(filepath) | None | v3 | Export to file |
| import_policy(filepath) | None | v3 | Import from file |
| load_pretrained(domain) | bool | v3 | Load domain policy |
| enable_ab_test(policy_b_arms) | None | v3 | Start A/B test |
| ab_results() | dict | v3 | Get A/B results |

### RetrievalRoute

| Route | Description | Version |
|-------|-------------|---------|
| VECTOR_ONLY | ChromaDB semantic search | v1 |
| KEYWORD_ONLY | BM25 exact matching | v1 |
| HYBRID | Vector + keyword with RRF | v1 |
| TABLE_FIRST | Prioritize table chunks | v1 |
| GRAPH_FIRST | Graph traversal first | v1 |
| GRAPH_HYBRID | Graph + hybrid | v1 |
| PAGE_BM25 | Page-level BM25 | v2 |
| PAGE_GRAPH | Page-level + graph | v2 |

### PolicyAction

| Field | Type | Description |
|-------|------|-------------|
| retrieval_route | RetrievalRoute | Selected route |
| retrieval_depth | int | Number of chunks |
| graph_activation | bool | Graph traversal on/off |
| graph_depth | int | Traversal hops |
| prompt_template | str | Template type |
| verification_level | str | Verification depth |
| was_exploration | bool | Exploration flag |
| action_confidence | float | Policy confidence |

### ArmStatistics

Thompson Sampling arm. Beta(alpha, beta) distribution per (context, action) pair.

| Method | Returns | Description |
|--------|---------|-------------|
| sample() | float | Sample from Beta distribution |
| update(reward) | None | Update alpha/beta |
| mean | float | Expected value |
| variance | float | Uncertainty |

---

## 10. PPO Policy (rl/ppo.py) — v3

### PPOPolicy

Tabular PPO with softmax policy. No PyTorch dependency.

| Method | Returns | Description |
|--------|---------|-------------|
| select_action(state_features, temperature) | (action_idx, log_prob) | Select action via softmax |
| get_value(state_features) | float | State value estimate |
| store_experience(state, action_idx, reward, log_prob, value) | None | Store for batch update |
| get_stats() | dict | Policy statistics |
| to_dict() / from_dict() | dict / PPOPolicy | Serialization |

---

## 11. Cross-Encoder Reranker (rl/reranker.py) — v3

### CrossEncoderReranker

| Method | Returns | Description |
|--------|---------|-------------|
| rerank(query, chunks, top_k) | list[(Chunk, score)] | Re-score chunks with cross-encoder |
| is_available | bool | Model loaded successfully |

Falls back to KeywordReranker if sentence-transformers not installed.

### KeywordReranker

Query-term overlap scoring. Zero dependencies. Always available.

---

## 12. Multi-Query Decomposition (rl/multi_query.py) — v3

### MultiQueryDecomposer

Rule-based, no LLM call. Handles compare, trace, chain, multi-part patterns.

| Method | Returns | Description |
|--------|---------|-------------|
| should_decompose(query) | bool | Check if decomposition beneficial |
| decompose(query) | list[str] | Split into sub-queries (max 5) |

---

## 13. Pre-trained Policies (rl/pretrained.py) — v3

### Available Domains

| Domain | Arms | Description |
|--------|------|-------------|
| financial | 11 | Optimized for reports, earnings, filings |
| legal | 8 | Optimized for contracts, regulations |
| healthcare | 6 | Optimized for clinical data, research |

### Functions

| Function | Returns | Description |
|----------|---------|-------------|
| get_pretrained_policy(domain) | dict | Get policy bundle |
| export_policy(arms, filepath, metadata) | None | Export to JSON |
| import_policy(filepath) | dict | Import from JSON |

---

## 14. Knowledge Graph (graph/__init__.py)

### KnowledgeGraph

| Method | Returns | Version | Description |
|--------|---------|---------|-------------|
| build_from_chunks(chunks) | None | v1 | Auto-build from document chunks |
| should_activate(query_analysis) | bool | v1 | 5-signal gate check |
| traverse(seed_entities, max_hops) | GraphTraversalResult | v1 | BFS traversal |
| record_activation_outcome(was_helpful) | None | v1 | Track success rate |
| add_node(label, entity_type, properties) | str | v1 | Add node |
| add_edge(source, target, relation, weight) | None | v1 | Add edge |
| get_stats() | dict | v1 | Graph statistics |
| save(filepath) | None | v2 | Persist to JSON |
| load(filepath) | None | v2 | Load from JSON |
| clear() | None | v1 | Clear graph |

### 5-Signal Activation Gate

| Signal | Score | Trigger |
|--------|-------|---------|
| Relationship words | +2 | "connected", "depends on", "affects" |
| Trigger analysis | +2 | TriggerInterpreter flags graph |
| Entity density | +1 | >= 2 named entities in query |
| Query complexity | +1 | Complex or multi_hop classification |
| Historical success | +1 | Activation success rate > 0.7 |
| **Threshold** | **>= 2** | **Both RL and gate must agree** |

---

## 15. Evaluation Engine (evaluation/__init__.py)

### EvaluationEngine

| Method | Returns | Description |
|--------|---------|-------------|
| evaluate(query, answer, chunks, latency, tokens) | EvaluationResult | Score response |
| compute_reward(evaluation) | float | Composite reward for RL |
| get_history() | list[dict] | Evaluation history |
| get_average_score(window) | float | Rolling average |
| get_improvement_rate() | float | Score trend |

### 6 Metrics (all automatic, no ground truth)

| S.No. | Metric | Weight | How it works |
|-------|--------|--------|-------------|
| 1 | Faithfulness | 0.30 | Token overlap between answer sentences and source chunks |
| 2 | Relevance | 0.20 | Token overlap between query and answer terms |
| 3 | Citation accuracy | 0.20 | Detection of citation patterns, penalty for uncited answers |
| 4 | Retrieval precision | 0.10 | Fraction of chunks containing query terms |
| 5 | Retrieval recall | 0.10 | Fraction of query terms in retrieved chunks |
| 6 | Hallucination risk | -0.02 | Fraction of answer sentences with no source overlap |

**Reward formula:**
R = 0.30F + 0.20R + 0.20C + 0.10P + 0.10L - 0.05Lat - 0.03Tok - 0.02H

---

## 16. LLM Providers (llm/__init__.py)

### LLMManager.create()

Factory method. Creates provider based on backend string.

### Providers

| Provider | Class | Local | Free |
|----------|-------|-------|------|
| Ollama | OllamaProvider | Yes | Yes |
| OpenAI / Groq / Grok / NVIDIA / Together | OpenAIProvider | No | Varies |
| Azure OpenAI | OpenAIProvider (custom endpoint) | No | No |
| HuggingFace | HuggingFaceProvider | Yes | Yes |

### LLMResponse

| Field | Type | Description |
|-------|------|-------------|
| text | str | Generated text |
| model | str | Model name |
| provider | str | Provider name |
| input_tokens | int | Prompt tokens |
| output_tokens | int | Generated tokens |
| latency_seconds | float | Generation time |

---

## 17. Learning Memory (memory/__init__.py)

### LearningMemory

| Method | Returns | Description |
|--------|---------|-------------|
| record_query(log) | None | Store query outcome |
| get_best_route(query_type, domain) | str | Best route for pattern |
| record_graph_activation(query_type, was_helpful, score) | None | Track graph outcomes |
| should_activate_graph(query_type) | bool | Graph recommendation |
| record_feedback(query_id, rating, reason) | None | Store feedback (v2) |
| get_learning_summary() | str | Formatted summary |
| get_stats() | dict | Memory statistics |

---

## 18. Prompt Engine (prompts/__init__.py)

### AdaptivePromptEngine

| Method | Returns | Description |
|--------|---------|-------------|
| build_prompt(query, analysis, chunks, graph_context) | str | Build complete prompt |
| update_from_evaluation(template_id, scores) | None | Update template scores |

### PromptLibrary

| Method | Returns | Description |
|--------|---------|-------------|
| get_template(template_type) | PromptTemplate | Get template |
| update_score(template_id, score) | None | Update EMA score |
| evolve_template(template_type, feedback) | PromptTemplate | Create improved variant |
| get_domain_template(domain, query_type) | PromptTemplate | Domain-specific template |

Templates auto-evolve when evaluation scores drop below 0.7.

---

## 19. Security (security/__init__.py)

### AuditTrail

| Method | Returns | Description |
|--------|---------|-------------|
| log(event_type, details, query_id) | None | Log event |
| get_query_trail(query_id) | list[AuditEntry] | Trail for query |
| export(filepath) | str | Export to JSON |
| display_query_trail(query_id) | str | Formatted trail |

### SecurityManager

| Method | Returns | Description |
|--------|---------|-------------|
| scan_for_pii(text) | list[dict] | Detect PII patterns |
| check_network_safety(url) | bool | Validate URL |
| hash_content(content) | str | SHA-256 hash |

---

## Feature Summary by Version

### v1 — Core Innovation (15 features)

| S.No. | Feature |
|-------|---------|
| 1 | Thompson Sampling RL policy engine |
| 2 | Conditional graph activation (5-signal gate) |
| 3 | 6-metric evaluation engine |
| 4 | Evaluation-driven RL reward loop |
| 5 | Multi-index retrieval (vector + keyword + RRF) |
| 6 | Knowledge graph with BFS traversal |
| 7 | Adaptive prompt engine with template evolution |
| 8 | Trigger interpreter (query classification) |
| 9 | Learning memory (routing patterns) |
| 10 | Multi-provider LLM support (Ollama, OpenAI, HF) |
| 11 | 10 document format parsing |
| 12 | Security manager + PII scanning |
| 13 | Full audit trail |
| 14 | Zero-configuration architecture |
| 15 | 99 tests |

### v2 — Production Ready (28 features, +13 new)

| S.No. | Feature |
|-------|---------|
| 16 | Vectorless mode (page BM25 + graph + RL) |
| 17 | Page-level index with page citations |
| 18 | Output formats (JSON, CSV, YAML, DataFrame) |
| 19 | Custom schema output |
| 20 | User feedback (thumbs up/down updates RL) |
| 21 | Feedback-driven prompt evolution |
| 22 | Crash recovery (persist BM25 + graph) |
| 23 | Auto-checkpoint every 5 minutes |
| 24 | Graceful shutdown handler |
| 25 | Hardened PDF/Excel/CSV/DOCX/PPTX parsing |
| 26 | SQL connector (PostgreSQL, MySQL, SQLite) |
| 27 | Incremental ingestion (add/remove/update) |
| 28 | Parallel ingestion with progress callback |
| 29 | Chunk quality scoring |
| 30 | Chunk deduplication (SimHash) |
| 31 | Prior chunk linking |
| 32 | Custom system prompt (3 levels) |
| 33 | Configurable warmup period |
| 34 | LLM backend "none" (retrieval only) |
| 35 | 10+ provider support (NVIDIA, Groq, Gemini, etc.) |
| 36 | OpenAI SDK integration |
| 37 | Dashboard v2 |
| 38 | 4 Colab notebooks |

### v3 — Intelligence Upgrades (44 features, +6 new)

| S.No. | Feature |
|-------|---------|
| 39 | PPO algorithm alongside Thompson Sampling |
| 40 | Cross-encoder reranking |
| 41 | Multi-query decomposition |
| 42 | Pre-trained domain policies (financial, legal, healthcare) |
| 43 | Transfer learning (export/import policies) |
| 44 | A/B testing (compare two policies) |

---

## File Inventory

| S.No. | Path | Lines | Description |
|-------|------|-------|-------------|
| 1 | core/engine.py | 840 | Main orchestrator |
| 2 | core/config.py | 140 | Configuration |
| 3 | core/response.py | 130 | Response objects |
| 4 | ingestion/parser.py | 866 | Document parser + SQL |
| 5 | ingestion/chunker.py | 356 | Chunking + quality + dedup |
| 6 | ingestion/engine.py | 307 | Ingestion orchestration |
| 7 | indexes/vector_index.py | 142 | ChromaDB |
| 8 | indexes/keyword_index.py | 178 | BM25 + persistence |
| 9 | indexes/page_index.py | 61 | Page-level BM25 |
| 10 | indexes/base.py | 48 | BaseIndex ABC |
| 11 | query/__init__.py | 290 | Trigger interpreter |
| 12 | rl/__init__.py | 560 | Thompson Sampling + export |
| 13 | rl/ppo.py | 178 | PPO policy |
| 14 | rl/reranker.py | 98 | Cross-encoder reranker |
| 15 | rl/multi_query.py | 114 | Multi-query decomposition |
| 16 | rl/pretrained.py | 84 | Pre-trained policies |
| 17 | graph/__init__.py | 410 | Knowledge graph |
| 18 | prompts/__init__.py | 330 | Prompt engine |
| 19 | evaluation/__init__.py | 390 | Evaluation engine |
| 20 | llm/__init__.py | 380 | LLM providers |
| 21 | memory/__init__.py | 280 | Learning memory |
| 22 | security/__init__.py | 220 | Audit + security |
| 23 | utils/__init__.py | 55 | Helpers |
| | **Total** | **~6,477** | **27 Python files** |
