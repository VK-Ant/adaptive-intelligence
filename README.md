<div align="center">

<img src="https://raw.githubusercontent.com/VK-Ant/adaptive-intelligence/main/docs/images/image_ai.png" alt="adaptive-intelligence" width="100%">

# adaptive-intelligence

**Self-improving retrieval framework that learns, remembers, and connects tools.**

[![PyPI](https://img.shields.io/pypi/v/adaptive-intelligence)](https://pypi.org/project/adaptive-intelligence/)
[![Python](https://img.shields.io/pypi/pyversions/adaptive-intelligence)](https://pypi.org/project/adaptive-intelligence/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-ResearchGate-00CCBB)](https://www.researchgate.net/publication/405076088)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/VK-Ant/adaptive-intelligence/blob/main/notebooks/adaptive_intelligence_v4_demo.ipynb)

[PyPI](https://pypi.org/project/adaptive-intelligence/) · [Paper](https://www.researchgate.net/publication/405076088) · [Portfolio](https://vk-ant.github.io/Venkatkumar/) · [llmevalkit](https://pypi.org/project/llmevalkit/)

</div>

---

## Why adaptive-intelligence?

Every RAG system uses the same retrieval strategy for every query. A revenue lookup gets vector search. A multi-document relationship chain gets the same vector search.

adaptive-intelligence fixes this. It uses reinforcement learning to select the best retrieval strategy per query type. The system evaluates every response, uses the score as a reward signal, and improves with every query answered.

```python
from adaptive_intelligence import AdaptiveAI

engine = AdaptiveAI()
engine.ingest("./documents")
response = engine.ask("What are the key risks?")
```

## Install

```bash
pip install adaptive-intelligence                # Zero deps (Ollama, no-LLM mode)
pip install adaptive-intelligence[vector]         # + ChromaDB vector search
pip install adaptive-intelligence[openai]         # + Any OpenAI-compatible API
pip install adaptive-intelligence[huggingface]    # + Local HuggingFace models
pip install adaptive-intelligence[all]            # Everything
```

## Features

### Context Engineering
Optimizes the entire context window — not just which chunks to retrieve, but what memory to include, how much history to keep, which tool results to add, and how to structure the prompt.

```python
engine = AdaptiveAI(context_engineering=True)
```

### MCP Integration
Connect external tools. Register MCP servers, REST APIs, or Python functions. The RL policy learns which tools to call per query type.

```python
engine.add_tool("financial", server="http://localhost:8081")
engine.add_tool("calculator", function=my_calc_function)
engine.add_tool("search", api_endpoint="https://api.example.com/search")

engine.list_tools()
engine.remove_tool("search")

# Serve your retrieval as an MCP server
engine.serve_mcp(port=8080)
```

### Agentic Workflow
Multi-round retrieval. The system retrieves, evaluates confidence, refines the query, calls tools, and retrieves again until the answer is sufficient.

```python
response = engine.ask("Analyze supply chain risks and mitigation", mode="agentic")
```

### Persistent Memory
Remembers across sessions. Routing patterns, user preferences, and facts persist to disk.

```python
engine.remember("focus_area", "supply chain risk")
engine.recall("focus_area")
engine.search_memory("supply chain")
```

### Incremental Learning
Add new documents anytime. The RL policy, knowledge graph, and memory continue from their current state — no restart needed.

```python
engine.ingest("./initial_docs")
engine.ingest("./quarterly_update.pdf")   # System continues, no restart
```

### RL-Based Retrieval Routing
Thompson Sampling or PPO learns which retrieval strategy works best per query type.

```python
engine = AdaptiveAI(rl_algorithm="ppo")
engine = AdaptiveAI(pretrained_policy=True, domain="financial")
engine.export_policy("learned.json")
engine.import_policy("learned.json")
```

### Conditional Graph Activation
Knowledge graph auto-built during ingestion. A 5-signal gate activates graph traversal only when the query needs relational reasoning. Saves compute on 70% of queries.

### Vectorless Mode
No ChromaDB, no embeddings, zero dependencies. Uses page-level BM25 with page citations.

```python
engine = AdaptiveAI(vectorless=True)
```

### Structured Output

```python
response = engine.ask("Extract metrics", output_format="json")
response = engine.ask("List items", output_format="csv")
response = engine.ask("Summarize", output_format="yaml")
```

### User Feedback

```python
response = engine.ask("What are the risks?")
engine.feedback(response.query_id, "good")   # +0.2 RL reward
engine.feedback(response.query_id, "bad")    # -0.3 RL reward + prompt evolution
```

## LLM Providers

The library works with any LLM. Zero required dependencies for basic usage.

### Free (no credit card needed)

```python
# Ollama — local, no extras needed
engine = AdaptiveAI()

# NVIDIA NIM
engine = AdaptiveAI(api_key="nvapi-...", base_url="https://integrate.api.nvidia.com/v1",
                    llm_model="meta/llama-3.1-70b-instruct")

# Groq
engine = AdaptiveAI(api_key="gsk_...", base_url="https://api.groq.com/openai/v1",
                    llm_model="llama-3.3-70b-versatile")

# Google Gemini
engine = AdaptiveAI(api_key="...", base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                    llm_model="gemini-2.0-flash")

# Together AI
engine = AdaptiveAI(api_key="...", base_url="https://api.together.xyz/v1",
                    llm_model="meta-llama/Llama-3-70b-chat-hf")

# Fireworks AI
engine = AdaptiveAI(api_key="...", base_url="https://api.fireworks.ai/inference/v1",
                    llm_model="accounts/fireworks/models/llama-v3p1-70b-instruct")

# HuggingFace local — needs [huggingface] extras
engine = AdaptiveAI(llm_backend="huggingface", llm_model="Qwen/Qwen2.5-1.5B-Instruct")

# No LLM — retrieval only
engine = AdaptiveAI(llm_backend="none")
```

### Paid

```python
# OpenAI
engine = AdaptiveAI(api_key="sk-...", llm_model="gpt-4o")

# Grok (xAI)
engine = AdaptiveAI(api_key="xai-...", base_url="https://api.x.ai/v1")

# Azure OpenAI
engine = AdaptiveAI(azure_endpoint="https://your.openai.azure.com/", api_key="...")
```

### Self-hosted

```python
# vLLM server
engine = AdaptiveAI(base_url="http://localhost:8000/v1")

# Any OpenAI-compatible server
engine = AdaptiveAI(base_url="http://your-server:8000/v1")
```

## How It Works

1. **Understand** — Trigger interpreter classifies query type, complexity, domain, entities (no LLM call)
2. **Decide** — RL policy selects retrieval route, depth, graph activation, tools to call
3. **Retrieve** — Executes via vector, BM25, or page index with RRF fusion. Graph activates conditionally.
4. **Generate** — Cross-encoder reranks chunks. Context engineer assembles full context window. LLM generates.
5. **Learn** — 6 metrics evaluate response. Composite score = RL reward. Policy updates. Next query is better.

## Demos

### Colab (no setup needed)

Run on free T4 GPU with no API key: [`notebooks/adaptive_intelligence_v4_demo.ipynb`](notebooks/adaptive_intelligence_v4_demo.ipynb)

### Local demos

```bash
cd demo_mcp_agenticai
pip install -r requirements.txt
python demo_basic.py      # Basic usage + incremental learning
python demo_tools.py      # Tool registry + cost optimization
python demo_agentic.py    # Agentic multi-round retrieval
python demo_mcp_server.py # Serve as MCP server (terminal 1)
python demo_mcp_client.py # Connect to MCP server (terminal 2)
```

## FAQ

**1. Does it work without an LLM?**

Yes. Set `llm_backend="none"` and the system returns relevant document excerpts. RL routing, graph activation, memory, and evaluation all work without an LLM.

**2. Does it work without a vector database?**

Yes. Set `vectorless=True` for page-level BM25 search with zero dependencies. Same RL routing, same graph, same learning.

**3. How many queries before it starts learning?**

Default warmup is 15 queries. During warmup, the system uses smart heuristic defaults while collecting RL statistics. After 15 queries, the learned policy takes over. Use `pretrained_policy=True` to skip warmup entirely.

**4. Which retrieval strategies does the RL choose from?**

Six routes: keyword only, vector only, hybrid (keyword + vector with RRF), table first, graph first, and graph hybrid. The RL also selects retrieval depth and whether to activate the knowledge graph.

**5. How is the knowledge graph built?**

Automatically during document ingestion from entity co-occurrences. No manual setup. A 5-signal gate decides per-query whether to activate graph traversal.

**6. What is context engineering?**

Instead of just stuffing retrieved chunks into a prompt, context engineering optimizes the entire context window — system prompt, memory entries, conversation history, tool results, and chunks — with token budget allocation per component.

**7. How does agentic mode work?**

The system retrieves, evaluates confidence, and if it's below threshold, refines the query and retrieves again. It can also call registered tools between rounds. Maximum 3 rounds by default.

**8. Can I add documents after initial ingestion?**

Yes. Call `engine.ingest()` again with new documents. The RL policy, knowledge graph, and memory all continue from their current state. No restart needed.

**9. What about latency?**

The system adds approximately 100-150ms overhead per query (classification, RL decision, evaluation, policy update). The LLM call typically takes 500-3000ms. Overhead is roughly 5-10% of total response time.

**10. How does it save cost?**

The RL learns optimal retrieval depth per query type. Factual queries get depth 2 (2 chunks), complex queries get depth 8. Fewer chunks = fewer tokens = lower LLM cost. The RL also learns which tools to skip, reducing unnecessary API calls.

**11. Is it production-ready?**

The library has 99 tests, crash recovery with auto-checkpoint, and graceful shutdown. BM25 is in-memory which works for hundreds of documents. For 50K+ documents, a disk-backed index is planned.

**12. How is this different from LangChain or LlamaIndex?**

Those are orchestration frameworks with static pipelines. You configure the retrieval strategy once. adaptive-intelligence learns the optimal strategy per query type through reinforcement learning. The system improves with every query answered.

## Version History

| Version | Focus | Features |
|---------|-------|----------|
| v1 | Core Innovation | RL routing, conditional graph, 6-metric evaluation, 15 features |
| v2 | Production Ready | Vectorless mode, crash recovery, SQL connector, 10+ providers, 28 features |
| v3 | Intelligence | PPO, reranking, multi-query, pre-trained policies, transfer learning, 44 features |
| v4 | Context + Agentic | Context engineering, MCP, agentic workflow, persistent memory, incremental learning |

## Project Structure

```
adaptive_intelligence/
    core/           # Engine, config, response objects
    ingestion/      # Parser (10+ formats), chunker, ingestion engine
    indexes/        # Vector (ChromaDB), keyword (BM25), page index
    query/          # Trigger interpreter, query classification
    rl/             # Thompson Sampling, PPO, reranker, multi-query, pretrained
    graph/          # Knowledge graph, conditional activation
    prompts/        # Adaptive prompt engine, template evolution
    evaluation/     # 6-metric evaluation engine
    llm/            # LLM providers (Ollama, OpenAI, HuggingFace)
    memory/         # Learning memory, persistent memory
    context/        # Context engineering, token budget
    mcp/            # MCP server, tool registry
    agentic/        # Agentic workflow, multi-round retrieval
    security/       # Audit trail, PII scanning
    utils/          # Logging, timing, ID generation
```

## Citation

If you use adaptive-intelligence in your research, please cite:

```bibtex
@article{rajan2026adaptive,
  title={Adaptive Retrieval Orchestration for Self-Learning Knowledge Systems},
  author={Rajan, Venkatkumar},
  year={2026},
  publisher={ResearchGate},
  url={https://www.researchgate.net/publication/405076088}
}
```

## Contributing

Contributions are welcome.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Add tests for your changes
4. Run tests (`pytest tests/ -v`)
5. Submit a pull request

## Also by me

[**llmevalkit**](https://pypi.org/project/llmevalkit/) — LLM evaluation, hallucination detection, AI content detection, compliance, document parsing, governance, security, observability, ground truth testing, conversation evaluation, red team testing, and anomaly detection. 78 metrics across 13 modules. Works with or without API.

adaptive-intelligence was born from llmevalkit. If you can measure LLM quality (llmevalkit), you can use those measurements as a reward signal to improve retrieval (adaptive-intelligence).

## Author

**Venkatkumar Rajan** · [@VK_Venkatkumar](https://linkedin.com/in/venkatkumarvk) · [Portfolio](https://vk-ant.github.io/Venkatkumar/)

## License

[Apache License 2.0](LICENSE)
