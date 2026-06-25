# Changelog

## v4.0.0 — Context Engineering + Agentic RAG

- Context engineering — optimizes entire context window (system prompt, memory, history, chunks, tool results)
- MCP integration — connect MCP servers, REST APIs, Python functions as tools. Serve as MCP server.
- Agentic workflow — multi-round retrieval with query refinement and tool calls
- Persistent memory — long-term memory across sessions with pattern learning
- Incremental learning — add new documents anytime, system continues from current state
- Tool registry — RL learns which tools to call per query type
- Zero required dependencies — vectorless mode works out of the box
- 31 modules, 7,605 lines, 99 tests

## v3.0.2

- Fixed LLM judge JSON parsing (regex extraction, multiple fallback approaches)
- Downgraded evaluation warning to debug level
- Fixed HuggingFace output cleanup (repetition penalty, truncation)
- Auto-fallback to retrieval-only when Ollama not running

## v3.0.0 — Intelligence Upgrades

- PPO algorithm alongside Thompson Sampling
- Cross-encoder reranking (top-20 chunks re-scored)
- Multi-query decomposition (rule-based, no LLM call)
- Pre-trained domain policies (financial, legal, healthcare)
- Transfer learning (export/import policies)
- A/B testing

## v2.0.0 — Production Ready

- Vectorless mode (page BM25 + graph + RL, zero dependencies)
- Structured output (JSON, CSV, YAML, DataFrame)
- User feedback updates RL reward
- Crash recovery with auto-checkpoint
- Hardened PDF/Excel/CSV/DOCX/PPTX parsing
- SQL connector (PostgreSQL, MySQL, SQLite)
- Incremental ingestion (add/remove/update)
- 10+ LLM providers via OpenAI SDK

## v1.0.0 — Core Innovation

- Thompson Sampling RL policy engine
- Conditional graph activation (5-signal gate)
- 6-metric evaluation engine (no ground truth needed)
- Evaluation-driven RL reward loop
- Multi-index retrieval (vector + BM25 + hybrid RRF)
- Knowledge graph with BFS traversal
- Adaptive prompt engine with template evolution
- Trigger interpreter (query classification)
- 10 document format support
- Audit trail and PII scanning
