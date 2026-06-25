# Architecture Overview

## Pipeline

Every query goes through a 10-step pipeline with an RL feedback loop:

```
                        ┌─────────────────────┐
                        │       Query         │
                        └─────────┬───────────┘
                                  ▼
                   ┌──────────────────────────────┐
                   │     Trigger Interpreter       │
                   │  Type · Complexity · Domain   │
                   └──────────────┬────────────────┘
                                  ▼
              ┌───────────────────────────────────────┐
   ┌──────────│        RL Policy Engine               │
   │  Learning│   Thompson Sampling → Route, Depth    │
   │   loop   └────────┬────────────────┬─────────────┘
   │                    ▼                ▼
   │     ┌──────────────────┐  ┌──────────────────┐
   │     │ Graph Traversal  │  │ Vector + Keyword  │
   │     │  (conditional)   │  │  (hybrid RRF)     │
   │     └────────┬─────────┘  └────────┬──────────┘
   │              └──────────┬──────────┘
   │                         ▼
   │          ┌──────────────────────────────┐
   │          │      Adaptive Prompt         │
   │          │  Domain-aware · Evolving     │
   │          └──────────────┬───────────────┘
   │                         ▼
   │          ┌──────────────────────────────┐
   │          │      LLM Generation          │
   │          │ Ollama · OpenAI · Groq · HF  │
   │          └──────────────┬───────────────┘
   │                         ▼
   │          ┌──────────────────────────────┐
   │          │    Evaluation Engine          │
   │          │ Faithfulness · Relevance ·    │
   │          │ Hallucination · Citations     │
   │          └───┬──────────┬───────────┬───┘
   │              ▼          ▼           ▼
   │       ┌──────────┐ ┌────────┐ ┌──────────┐
   └◄──────│RL Update │ │Memory  │ │ Prompt   │
           │ (reward) │ │Update  │ │Evolution │
           └──────────┘ └────────┘ └──────────┘
```

## Module Map

```
adaptive_intelligence/
├── core/
│   ├── engine.py          # AdaptiveAI orchestrator (10-step pipeline)
│   ├── config.py          # AdaptiveConfig + nested configs
│   └── response.py        # AdaptiveResponse, Citation, EvaluationResult
├── ingestion/
│   ├── parser.py          # DocumentParser (10 formats)
│   ├── chunker.py         # Smart chunking with boundary detection
│   └── engine.py          # IngestionEngine (parse → chunk pipeline)
├── indexes/
│   ├── base.py            # BaseIndex ABC
│   ├── vector_index.py    # ChromaDB semantic search
│   └── keyword_index.py   # BM25 lexical search
├── query/
│   └── __init__.py        # TriggerInterpreter (query classification)
├── rl/
│   └── __init__.py        # RLPolicyEngine (Thompson Sampling)
├── graph/
│   └── __init__.py        # KnowledgeGraph (conditional activation)
├── prompts/
│   └── __init__.py        # AdaptivePromptEngine (evolving templates)
├── evaluation/
│   └── __init__.py        # EvaluationEngine (6 metrics → reward)
├── llm/
│   └── __init__.py        # LLM providers (Ollama, OpenAI, HuggingFace)
├── memory/
│   └── __init__.py        # LearningMemory (routing patterns)
├── security/
│   └── __init__.py        # AuditTrail, SecurityManager
└── utils/
    └── __init__.py        # Helpers (logging, timing, IDs)
```

## Three Differentiators

### 1. RL Policy Engine

Contextual bandits with Thompson Sampling. Each (query_context, action) pair maintains a Beta(α,β) distribution. The agent samples from all arms and picks the highest — naturally balancing exploration and exploitation.

- **State**: query type + complexity + domain + graph_needed + table_needed
- **Action**: retrieval route × depth × graph activation × prompt template × verification level
- **Reward**: composite evaluation score (0-1)
- **Warmup**: first N queries use heuristic defaults

### 2. Conditional Graph Activation

Entity co-occurrence graph built during ingestion. Five-signal gate:

1. Relationship words in query (+2)
2. Trigger Interpreter flags graph (+2)
3. Entity density ≥ 2 (+1)
4. Complex/multi-hop query (+1)
5. Historical success rate > threshold (+1)

Activates only when total signals ≥ 2. AND-gated with RL policy decision.

### 3. Self-Adaptive Retrieval

Closed evaluation → reward → policy update loop:

- Evaluation: faithfulness, relevance, citation, hallucination, precision, recall
- Composite score = weighted sum = RL reward
- Policy updates Beta distributions for the selected arm
- Prompt templates evolve when scores drop below 0.7
- Learning memory stores routing patterns for future queries
