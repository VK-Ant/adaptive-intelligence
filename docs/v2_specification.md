# adaptive-intelligence v2.0 — Complete Feature Specification

## Philosophy

```
v1: Proved the concept (RL routing, graph, evaluation)
v2: Make it usable (survive real documents, run anywhere, output anything)

RULE 1: User API stays 3 lines. Everything new is optional.
RULE 2: Zero dependencies mode must work.
RULE 3: Every provider, every format, copy-paste-run.
```

---

## Pillar 1: Data Ingestion — "Survive Every Document"

### 1.1 PDF Edge Cases

| S.No. | Scenario | Fix |
|-------|----------|-----|
| 1 | Scanned PDF (no text layer) | Auto-detect → OCR fallback |
| 2 | Password-protected | Try empty password → skip with warning |
| 3 | Corrupted/truncated | Parse per-page, extract what's readable |
| 4 | Multi-column layout | Reading-order extraction via bbox |
| 5 | Headers/footers repeated | Detect repeated blocks → strip |
| 6 | Table spanning pages | Track state → merge continuation |
| 7 | Embedded images with text | Extract images → OCR each |
| 8 | 500+ page PDF | Stream page-by-page, no full load |
| 9 | Forms/fillable fields | Extract as key-value pairs |
| 10 | Footnotes/endnotes | Separate into metadata |

### 1.2 Excel Edge Cases

| S.No. | Scenario | Fix |
|-------|----------|-----|
| 1 | Merged cells | Unmerge and fill correctly |
| 2 | Hidden sheets | Skip by default, option to include |
| 3 | Formula cells | Use computed values, fallback to formula |
| 4 | Empty rows/columns | Strip >80% empty rows, detect data boundaries |
| 5 | Multiple header rows | Detect via bold/border/type pattern |
| 6 | 100K+ rows | Read-only stream mode, process in batches |
| 7 | Cell comments | Extract as linked metadata |
| 8 | Date/currency formats | Normalize dates to ISO, preserve currency |

### 1.3 CSV/TSV Edge Cases

| S.No. | Scenario | Fix |
|-------|----------|-----|
| 1 | Wrong delimiter | Auto-detect via csv.Sniffer |
| 2 | Mixed encodings | Fallback chain: UTF-8 → Latin-1 → CP1252 |
| 3 | Malformed rows | Pad short rows, truncate long rows |
| 4 | No header row | Auto-generate: col_1, col_2 |
| 5 | BOM at start | Strip before parsing |
| 6 | Quoted fields with newlines | Proper csv.reader quoting |
| 7 | Large CSV (1GB+) | Stream in chunks |

### 1.4 DOCX Edge Cases

| S.No. | Scenario | Fix |
|-------|----------|-----|
| 1 | Track changes | Extract accepted text only |
| 2 | Comments | Extract as linked metadata |
| 3 | Nested tables | Recursive extraction |
| 4 | Password-protected | Skip with warning |

### 1.5 PPTX Edge Cases

| S.No. | Scenario | Fix |
|-------|----------|-----|
| 1 | Speaker notes | Include as chunk metadata |
| 2 | Grouped shapes | Flatten, extract all text |
| 3 | SmartArt | Extract from XML |
| 4 | Password-protected | Skip with warning |

### 1.6 SQL Connector

```python
engine.ingest("sqlite:///data.db")
engine.ingest("postgresql://user:pass@host/db", tables=["orders"])
engine.ingest("mysql://user:pass@host/db", query="SELECT * FROM invoices WHERE year=2025")
```

- SQLAlchemy-based (one dependency)
- Auto-discover tables if none specified
- Each 50-row group → one chunk with headers
- Column names/types → graph nodes
- Foreign keys → graph edges automatically
- NULL handling, date normalization

### 1.7 Ingestion Infrastructure

```python
# Incremental
engine.ingest("./new_report.pdf")       # Add
engine.remove("old_report.pdf")          # Remove
engine.update("./updated_report.pdf")    # Re-index

# Parallel
engine.ingest("./docs/", parallel=True, workers=4)

# Quality gate
engine.ingest("./docs/", min_chunk_quality=0.5, deduplicate=True)
```

- Chunk quality scoring (reject short, garbled, repetitive)
- Near-duplicate detection (SimHash)
- Metadata auto-extraction (date, author, title)
- Progress callback support

---

## Pillar 2: Vectorless Mode — "No Embeddings, No DB"

### What It Is

```python
engine = AdaptiveAI(vectorless=True)
```

No ChromaDB. No embedding model. No internet needed. Pure Python BM25 + knowledge graph + RL routing.

### Page Index

- Full page as retrieval unit (not 500-char chunks)
- Page-number citations in answers
- BM25 keyword search over pages

### RL Still Works

Available routes in vectorless mode:

```
page_bm25, keyword_only, table_first, graph_hybrid, page_graph
```

Same Thompson Sampling. Same reward function. Different action space.

### When to Use

| Use Case | Mode |
|----------|------|
| Financial reports, legal, medical | Vectorless (standard terminology) |
| Air-gapped / no internet | Vectorless |
| Quick demo | Vectorless |
| Mixed topics, research papers | Vector (need semantic matching) |
| 1000+ documents | Vector (better recall) |

---

## Pillar 3: Output + Feedback — "Your Format, Your Voice"

### Output Formats

```python
response = engine.ask("Extract vendors", output_format="json")
response = engine.ask("List items", output_format="csv")
response = engine.ask("Show data", output_format="yaml")
response = engine.ask("Quarterly revenue", output_format="dataframe")

# Custom schema
response = engine.ask("Contract details", output_format="json",
    schema={"parties": ["str"], "value": "float", "date": "str"})
```

### User Feedback

```python
engine.feedback(response.query_id, "good")
engine.feedback(response.query_id, "bad", reason="Missing page 5")
```

- Good → RL reward boost (+0.2)
- Bad → RL penalty (-0.3) + prompt evolution triggered
- Feedback stored in audit trail

### Crash Recovery

| Component | v1 | v2 |
|-----------|----|----|
| Vector index (ChromaDB) | Persisted | Same |
| BM25 keyword index | Lost on crash | Persisted to disk |
| Knowledge graph | Lost on crash | Persisted to disk |
| RL policy | Every N queries | Auto-checkpoint every 5 min |
| Learning memory | Every 50 queries | Auto-checkpoint every 5 min |
| Page index | N/A | Persisted to disk |

- Graceful shutdown handler (SIGTERM/SIGINT)
- Startup recovery: detect + rebuild missing state

---

## Pillar 4: Providers — Copy, Paste, Run

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
engine = AdaptiveAI(llm_backend="huggingface",
    llm_model="microsoft/phi-2")

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
    base_url="https://api.x.ai/v1",
    llm_model="grok-3-mini")

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

## Complete v2 User API

```python
from adaptive_intelligence import AdaptiveAI

# ─── SAME AS v1 (zero breaking changes) ────────────────
engine = AdaptiveAI()
engine.ingest("./documents")
response = engine.ask("What are the key risks?")
print(response.answer)
print(response.confidence)

# ─── NEW: Vectorless ───────────────────────────────────
engine = AdaptiveAI(vectorless=True)
response = engine.ask("Revenue details?")
print(response.citations[0].page)         # Page number

# ─── NEW: SQL ──────────────────────────────────────────
engine.ingest("sqlite:///data.db")
engine.ingest("postgresql://user:pass@host/db", tables=["orders"])

# ─── NEW: Output formats ──────────────────────────────
response = engine.ask("Extract vendors", output_format="json")
response = engine.ask("List items", output_format="csv")
response = engine.ask("Show data", output_format="dataframe")

# ─── NEW: Custom schema ───────────────────────────────
response = engine.ask("Contract details", output_format="json",
    schema={"parties": ["str"], "value": "float"})

# ─── NEW: Feedback ─────────────────────────────────────
engine.feedback(response.query_id, "good")
engine.feedback(response.query_id, "bad", reason="Wrong data")

# ─── NEW: Incremental ─────────────────────────────────
engine.ingest("./new_file.pdf")
engine.remove("old_file.pdf")
engine.update("./changed_file.pdf")

# ─── NEW: Parallel ────────────────────────────────────
engine.ingest("./docs/", parallel=True, workers=4)

# ─── NEW: Configurable warmup ─────────────────────────
engine = AdaptiveAI(warmup_queries=5)

# ─── NEW: System prompt + output ──────────────────────
engine = AdaptiveAI(
    system_prompt="You are a financial analyst.",
    vectorless=True,
    warmup_queries=5,
)
response = engine.ask("Revenue breakdown", output_format="json")
```

---

## What v2 Does NOT Include

| Feature | Reason | Goes to |
|---------|--------|---------|
| PPO/DQN | Thompson Sampling works | v3 |
| GraphSAGE | BFS sufficient for now | v3 |
| Multi-modal (images, audio) | Complex, low demand | v3 |
| Enterprise connectors (S3, SharePoint) | Need users first | v3 |
| Plugin system | Need community first | v3 |
| Compliance (HIPAA, GDPR) | llmevalkit handles this | v3 |
| Distributed ingestion | Nobody has 100K docs | v3 |
| Agentic retrieval | Research, not production | v3 |

---

## v3 Preview — Intelligence Upgrades

| S.No. | Feature | Category |
|-------|---------|----------|
| 1 | PPO/DQN alongside Thompson Sampling | RL |
| 2 | GraphSAGE embeddings replacing BFS | GNN |
| 3 | Cross-encoder reranking | Retrieval |
| 4 | Multi-query decomposition | Retrieval |
| 5 | Pre-trained domain policies (financial, legal, healthcare) | RL |
| 6 | Transfer learning across deployments | RL |
| 7 | A/B testing (two policies simultaneously) | RL |
| 8 | Multi-modal ingestion (images, charts, audio) | Ingestion |
| 9 | Enterprise connectors (S3, Notion, Confluence) | Platform |
| 10 | llmevalkit compliance integration | Governance |
| 11 | Plugin API for community connectors | Platform |
| 12 | Scale benchmarks (10K, 50K, 100K docs) | Performance |

---

## Build Plan

```
WEEK 1: Ingestion
  - PDF edge cases (10 scenarios)
  - Excel edge cases (8 scenarios)
  - CSV + SQL connector
  - Tests

WEEK 2: Ingestion + Vectorless
  - Incremental ingestion
  - Parallel ingestion
  - Chunk quality + dedup
  - Page index + BM25-only mode
  - Page citations
  - Tests

WEEK 3: Output + Feedback
  - JSON/CSV/YAML/DataFrame output
  - Custom schema
  - User feedback → RL reward
  - Crash recovery (persist BM25 + graph)
  - Auto-checkpoint
  - Tests

WEEK 4: Polish + Ship
  - Provider examples (all copy-paste)
  - Integration tests with real documents
  - README update
  - Colab notebook update
  - PyPI publish v2.0
  - LinkedIn post
```

---

## Summary

```
v2 adds 4 pillars to v1:

1. INGESTION:   Survive every document (PDF, Excel, CSV, SQL, DOCX, PPTX)
2. VECTORLESS:  No embeddings, no DB, page index + BM25 + graph + RL
3. OUTPUT:      JSON, CSV, YAML, DataFrame, custom schema
4. PROVIDERS:   Copy-paste setup for 10+ providers (free and paid)

Plus: feedback, crash recovery, incremental ingestion, parallel processing

User API: still 3 lines. Everything new is optional parameters.
```
