"""
Adaptive Intelligence — Advanced Usage Examples

Shows Level 1 and Level 2 access patterns:
domain configuration, security settings, RL tuning,
graph inspection, evaluation analysis, and audit trails.
"""

from adaptive_intelligence import AdaptiveAI
from adaptive_intelligence.core.config import (
    AdaptiveConfig, LLMBackend, Domain, SecurityLevel,
    RLConfig, GraphConfig, EvaluationConfig,
)

# ═══════════════════════════════════════════════════════════════════
# LEVEL 1: Domain & Provider Configuration
# ═══════════════════════════════════════════════════════════════════

# Financial domain with OpenAI
financial_engine = AdaptiveAI(
    domain="financial",
    llm_backend="openai",
    llm_model="gpt-4o",
    api_key="sk-...",
    security_level="high",
)

# Healthcare domain with Ollama (local, HIPAA-friendly)
healthcare_engine = AdaptiveAI(
    domain="healthcare",
    llm_backend="ollama",
    llm_model="llama3.2",
    security_level="maximum",
)

# Azure OpenAI deployment
azure_engine = AdaptiveAI(
    llm_backend="azure_openai",
    llm_model="gpt-4o",
    azure_endpoint="https://your-resource.openai.azure.com",
    deployment_name="your-deployment",
    api_key="your-azure-key",
)


# ═══════════════════════════════════════════════════════════════════
# LEVEL 2: Full Configuration Control
# ═══════════════════════════════════════════════════════════════════

config = AdaptiveConfig(
    # LLM
    llm_backend=LLMBackend.OLLAMA,
    llm_model="llama3.2",
    temperature=0.05,
    max_tokens=4096,

    # Domain & Security
    domain=Domain.FINANCIAL,
    security_level=SecurityLevel.HIGH,
    enable_audit_trail=True,
    network_enabled=False,

    # RL Policy Tuning
    rl=RLConfig(
        warmup_queries=20,          # Learn from 20 queries before using RL
        exploration_rate=0.15,      # 15% random exploration
        min_exploration_rate=0.03,  # Floor at 3%
        exploration_decay=0.99,     # Slow decay
        algorithm="thompson_sampling",
    ),

    # Graph Configuration
    graph=GraphConfig(
        enabled=True,
        conditional_activation=True,   # Only activate when query needs it
        max_hops=3,
        min_entity_count_for_activation=2,
        activation_success_threshold=0.65,
    ),

    # Evaluation Weights
    evaluation=EvaluationConfig(
        faithfulness_weight=0.35,    # Extra emphasis on grounding
        relevance_weight=0.25,
        citation_weight=0.15,
        enable_llm_judge=True,       # Use LLM to evaluate when unsure
        llm_judge_threshold=0.75,
    ),

    # Retrieval
    default_retrieval_depth=8,
    max_retrieval_depth=25,
    enable_reranking=True,
)

engine = AdaptiveAI(config=config)


# ═══════════════════════════════════════════════════════════════════
# WORKING WITH THE ENGINE
# ═══════════════════════════════════════════════════════════════════

# Ingest multiple sources
engine.ingest([
    "./financial_reports/",
    "./board_minutes/",
    "./regulatory_filings/sec_10k.pdf",
])

# Ask with priority hints
response = engine.ask(
    "What are the key risk factors mentioned across all filings?",
    priority="accuracy",
    depth=15,  # Override retrieval depth
)

# ── Inspect the full pipeline ────────────────────────────────────

# What did the system understand about the query?
print("Query Analysis:", response.query_analysis)

# What strategy did the RL policy choose?
pd = response.policy_decision
print(f"Route: {pd.retrieval_route}")
print(f"Graph active: {pd.graph_activation}")
print(f"Was exploration: {pd.was_exploration}")
print(f"Policy confidence: {pd.policy_confidence:.2f}")

# How was the retrieval done?
ri = response.retrieval_info
print(f"Chunks retrieved: {ri.chunks_retrieved}")
print(f"Graph hops: {ri.graph_hops}")

# How well did the system do?
ev = response.evaluation
print(ev.display())


# ═══════════════════════════════════════════════════════════════════
# MONITORING & INSPECTION
# ═══════════════════════════════════════════════════════════════════

# System dashboard
print(engine.dashboard())

# Learning curve (for plotting)
curve = engine.learning_curve()
# curve = [{"query_number": 1, "reward": 0.72, "rolling_avg": 0.72}, ...]

# RL Policy stats
rl_stats = engine.rl.get_stats()
print(f"RL warmup: {rl_stats['is_warmup']}")
print(f"Arms learned: {rl_stats['total_arms']}")
print(f"Exploration rate: {rl_stats['exploration_rate']:.2%}")

# Graph stats
graph_stats = engine.graph.get_stats()
print(f"Graph nodes: {graph_stats['nodes']}")
print(f"Graph edges: {graph_stats['edges']}")
print(f"Graph activation success: {graph_stats['activation_success_rate']:.0%}")

# Learning memory
print(engine.memory.get_learning_summary())

# Audit trail for a specific query
trail = engine.audit.display_query_trail(response.query_id)
print(trail)

# Export full audit trail
engine.audit.export("audit_trail.json")

# Full system status
print(engine.status())


# ═══════════════════════════════════════════════════════════════════
# DEPENDENCY CHECK
# ═══════════════════════════════════════════════════════════════════

from adaptive_intelligence.security import DependencyVerifier
print(DependencyVerifier.full_report())
