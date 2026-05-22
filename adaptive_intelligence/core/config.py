"""Configuration for Adaptive Intelligence engine."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum
from pathlib import Path


class LLMBackend(str, Enum):
    OLLAMA = "ollama"
    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    GROQ = "groq"
    TOGETHER = "together"
    CUSTOM = "custom"
    NONE = "none"


class SecurityLevel(str, Enum):
    STANDARD = "standard"
    HIGH = "high"
    MAXIMUM = "maximum"


class Domain(str, Enum):
    GENERAL = "general"
    FINANCIAL = "financial"
    LEGAL = "legal"
    TECHNICAL = "technical"
    HEALTHCARE = "healthcare"
    SCIENTIFIC = "scientific"
    OPERATIONAL = "operational"


@dataclass
class ChunkingConfig:
    chunk_size: int = 500
    chunk_overlap: int = 50
    respect_boundaries: bool = True
    min_chunk_size: int = 100


@dataclass
class RLConfig:
    warmup_queries: int = 15
    exploration_rate: float = 0.2
    min_exploration_rate: float = 0.05
    exploration_decay: float = 0.995
    learning_rate: float = 0.1
    discount_factor: float = 0.95
    update_frequency: int = 5
    algorithm: str = "thompson_sampling"  # "epsilon_greedy", "ucb", "thompson_sampling"


@dataclass
class GraphConfig:
    enabled: bool = True
    conditional_activation: bool = True
    max_hops: int = 3
    min_entity_count_for_activation: int = 2
    relationship_confidence_threshold: float = 0.5
    activation_success_threshold: float = 0.7


@dataclass
class EvaluationConfig:
    faithfulness_weight: float = 0.30
    relevance_weight: float = 0.20
    citation_weight: float = 0.20
    precision_weight: float = 0.10
    recall_weight: float = 0.10
    latency_penalty_weight: float = 0.05
    token_cost_weight: float = 0.03
    hallucination_penalty_weight: float = 0.02
    llm_judge_threshold: float = 0.8
    enable_llm_judge: bool = True
    enable_consistency_checks: bool = False


@dataclass
class AdaptiveConfig:
    # LLM Configuration
    llm_backend: LLMBackend = LLMBackend.OLLAMA
    llm_model: str = "llama3.2"
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    azure_endpoint: Optional[str] = None
    deployment_name: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 2048

    # Domain
    domain: Domain = Domain.GENERAL

    # Security
    security_level: SecurityLevel = SecurityLevel.STANDARD
    network_enabled: bool = False
    enable_audit_trail: bool = True

    # Storage
    storage_dir: str = "./.adaptive_intelligence"
    vector_db: str = "chromadb"

    # Chunking
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)

    # RL Engine
    rl: RLConfig = field(default_factory=RLConfig)

    # Graph
    graph: GraphConfig = field(default_factory=GraphConfig)

    # Evaluation
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    # Retrieval
    default_retrieval_depth: int = 5
    max_retrieval_depth: int = 20
    enable_reranking: bool = True

    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize config to dictionary."""
        import dataclasses
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AdaptiveConfig":
        """Create config from dictionary."""
        # Handle nested dataclasses
        if "chunking" in data and isinstance(data["chunking"], dict):
            data["chunking"] = ChunkingConfig(**data["chunking"])
        if "rl" in data and isinstance(data["rl"], dict):
            data["rl"] = RLConfig(**data["rl"])
        if "graph" in data and isinstance(data["graph"], dict):
            data["graph"] = GraphConfig(**data["graph"])
        if "evaluation" in data and isinstance(data["evaluation"], dict):
            data["evaluation"] = EvaluationConfig(**data["evaluation"])
        if "llm_backend" in data and isinstance(data["llm_backend"], str):
            data["llm_backend"] = LLMBackend(data["llm_backend"])
        if "security_level" in data and isinstance(data["security_level"], str):
            data["security_level"] = SecurityLevel(data["security_level"])
        if "domain" in data and isinstance(data["domain"], str):
            data["domain"] = Domain(data["domain"])
        return cls(**data)
