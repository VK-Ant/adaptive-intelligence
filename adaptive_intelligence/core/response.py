"""Response object returned by AdaptiveAI.ask()."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class Citation:
    """A single citation linking a claim to a source."""
    text: str
    source_document: str
    chunk_id: str
    confidence: float
    page: Optional[int] = None

    def __repr__(self) -> str:
        return f"Citation(source='{self.source_document}', confidence={self.confidence:.2f})"


@dataclass
class RetrievalInfo:
    """Details about retrieval strategy used."""
    strategy: str  # "vector", "keyword", "hybrid", "graph", "table"
    indexes_used: List[str] = field(default_factory=list)
    chunks_retrieved: int = 0
    retrieval_depth: int = 0
    graph_activated: bool = False
    graph_hops: int = 0
    reranking_applied: bool = False


@dataclass
class EvaluationResult:
    """Self-evaluation metrics for the response."""
    faithfulness: float = 0.0
    relevance: float = 0.0
    citation_accuracy: float = 0.0
    hallucination_risk: float = 0.0
    retrieval_precision: float = 0.0
    retrieval_recall: float = 0.0
    latency_seconds: float = 0.0
    token_usage: int = 0
    composite_score: float = 0.0
    confidence_level: str = "medium"  # "high", "medium", "low"

    @property
    def confidence_emoji(self) -> str:
        if self.composite_score >= 0.85:
            return "🟢"
        elif self.composite_score >= 0.70:
            return "🟡"
        return "🔴"

    def display(self) -> str:
        """Human-readable evaluation display."""
        bar = lambda v: "█" * int(v * 20) + "░" * (20 - int(v * 20))
        lines = [
            f"Source Grounding:   {bar(self.faithfulness)} {self.faithfulness:.0%}",
            f"Query Relevance:    {bar(self.relevance)} {self.relevance:.0%}",
            f"Hallucination Risk: {bar(self.hallucination_risk)} {self.hallucination_risk:.0%}",
            f"Citation Accuracy:  {bar(self.citation_accuracy)} {self.citation_accuracy:.0%}",
            f"Response Time:      {self.latency_seconds:.2f}s",
            f"Tokens Used:        {self.token_usage}",
            f"Confidence:         {self.confidence_emoji} {self.confidence_level.title()} ({self.composite_score:.0%})",
        ]
        return "\n".join(lines)


@dataclass
class PolicyDecision:
    """What the RL policy decided for this query."""
    retrieval_route: str
    retrieval_depth: int
    graph_activation: bool
    graph_depth: int
    model_used: str
    prompt_template: str
    verification_level: str
    was_exploration: bool = False
    policy_confidence: float = 0.0


@dataclass
class AdaptiveResponse:
    """Complete response from the Adaptive Intelligence engine."""
    answer: str
    confidence: float
    citations: List[Citation] = field(default_factory=list)
    evaluation: Optional[EvaluationResult] = None
    retrieval_info: Optional[RetrievalInfo] = None
    policy_decision: Optional[PolicyDecision] = None
    query_analysis: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)
    query_id: str = ""
    raw_chunks: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def retrieval_strategy(self) -> str:
        """Human-readable retrieval strategy description."""
        if self.retrieval_info:
            parts = [self.retrieval_info.strategy]
            if self.retrieval_info.graph_activated:
                parts.append(f"graph({self.retrieval_info.graph_hops}-hop)")
            parts.append(f"depth={self.retrieval_info.chunks_retrieved}")
            return " + ".join(parts)
        return "unknown"

    def __repr__(self) -> str:
        conf = f"{self.confidence:.0%}"
        strat = self.retrieval_strategy
        return f"AdaptiveResponse(confidence={conf}, strategy='{strat}', citations={len(self.citations)})"

    def display(self) -> str:
        """Full formatted display of response."""
        lines = [
            "=" * 60,
            "ADAPTIVE INTELLIGENCE RESPONSE",
            "=" * 60,
            "",
            self.answer,
            "",
            "-" * 60,
            f"Strategy: {self.retrieval_strategy}",
            f"Confidence: {self.confidence:.0%}",
            f"Citations: {len(self.citations)}",
        ]
        if self.evaluation:
            lines.append("")
            lines.append(self.evaluation.display())
        if self.citations:
            lines.append("")
            lines.append("Sources:")
            for i, c in enumerate(self.citations, 1):
                lines.append(f"  [{i}] {c.source_document} (confidence: {c.confidence:.0%})")
        lines.append("=" * 60)
        return "\n".join(lines)
