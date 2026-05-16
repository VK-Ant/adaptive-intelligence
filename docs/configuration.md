# Configuration Guide

## Three Levels of Access

### Level 0: Zero config (80% of users)

```python
from adaptive_intelligence import AdaptiveAI

engine = AdaptiveAI()
engine.ingest("./documents")
response = engine.ask("What are the key risks?")
```

Defaults: Ollama backend, llama3.2 model, general domain, standard security.

### Level 1: Provider + domain (15% of users)

```python
# Groq (free tier, fast)
engine = AdaptiveAI(
    llm_backend="openai",
    llm_model="llama-3.3-70b-versatile",
    api_key="gsk_...",
    base_url="https://api.groq.com/openai/v1",
    domain="financial",
)

# Grok (xAI)
engine = AdaptiveAI(
    llm_backend="openai",
    llm_model="grok-3-mini",
    api_key="xai-...",
    base_url="https://api.x.ai/v1",
)

# OpenAI
engine = AdaptiveAI(
    llm_backend="openai",
    llm_model="gpt-4o",
    api_key="sk-...",
)

# Azure OpenAI
engine = AdaptiveAI(
    llm_backend="azure_openai",
    llm_model="gpt-4o",
    azure_endpoint="https://your-resource.openai.azure.com",
    deployment_name="your-deployment",
    api_key="your-azure-key",
)

# HuggingFace (local, any model)
engine = AdaptiveAI(
    llm_backend="huggingface",
    llm_model="microsoft/phi-2",
)
```

### Level 2: Full config (5% of users)

```python
from adaptive_intelligence.core.config import (
    AdaptiveConfig, RLConfig, GraphConfig, EvaluationConfig,
    LLMBackend, Domain, SecurityLevel,
)

config = AdaptiveConfig(
    llm_backend=LLMBackend.OLLAMA,
    llm_model="llama3.2",
    temperature=0.05,
    max_tokens=4096,

    domain=Domain.FINANCIAL,
    security_level=SecurityLevel.HIGH,
    enable_audit_trail=True,

    rl=RLConfig(
        warmup_queries=20,
        exploration_rate=0.15,
        min_exploration_rate=0.03,
        exploration_decay=0.99,
        algorithm="thompson_sampling",
    ),

    graph=GraphConfig(
        enabled=True,
        conditional_activation=True,
        max_hops=3,
        min_entity_count_for_activation=2,
        activation_success_threshold=0.65,
    ),

    evaluation=EvaluationConfig(
        faithfulness_weight=0.35,
        relevance_weight=0.25,
        citation_weight=0.15,
        enable_llm_judge=True,
        llm_judge_threshold=0.75,
    ),
)

engine = AdaptiveAI(config=config)
```

## RL Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `warmup_queries` | 15 | Queries before RL takes over from heuristics |
| `exploration_rate` | 0.20 | Initial random exploration probability |
| `min_exploration_rate` | 0.05 | Floor for exploration after decay |
| `exploration_decay` | 0.995 | Multiplicative decay per query |
| `algorithm` | `"thompson_sampling"` | Bandit algorithm |

## Graph Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enabled` | `True` | Whether graph is available |
| `conditional_activation` | `True` | Use 5-signal gate (set False to always use graph) |
| `max_hops` | 3 | Maximum traversal depth |
| `min_entity_count_for_activation` | 2 | Minimum entities in query to consider graph |
| `activation_success_threshold` | 0.7 | Historical success rate needed to boost activation |

## Evaluation Weights

Default reward function: `R = 0.30×Faithfulness + 0.20×Relevance + 0.20×Citation + 0.10×Precision + 0.10×Recall - 0.05×Latency - 0.03×TokenCost - 0.02×Hallucination`

All weights are configurable via `EvaluationConfig`.

## Domains

| Domain | Prompt specialization |
|--------|----------------------|
| `general` | No domain rules |
| `financial` | GAAP/non-GAAP, currency, fiscal years, forward-looking statements |
| `legal` | Legal terminology, clause citations, jurisdiction notes |
| `healthcare` | Medical terminology, HIPAA boundaries, evidence levels |
| `technical` | Version numbers, specs, deprecation notes |
| `operational` | Actionable metrics, timeframes, dependencies |

## Security Levels

| Level | Behavior |
|-------|----------|
| `standard` | Default. Network allowed. Basic PII scanning. |
| `high` | Network restricted to whitelisted domains. Full PII detection. |
| `maximum` | Zero network. Full PII detection. All content hashed. |
