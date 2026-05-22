"""RL Policy Engine — Contextual Bandits with Thompson Sampling.

This is the core differentiator: the system learns optimal retrieval
strategies from evaluation feedback rather than using static rules.
"""

import json
import logging
import random
import math
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum

from adaptive_intelligence.core.config import RLConfig
from adaptive_intelligence.query import QueryAnalysis, QueryType, QueryComplexity

logger = logging.getLogger(__name__)


class RetrievalRoute(str, Enum):
    """Available retrieval strategies."""
    VECTOR_ONLY = "vector_only"
    KEYWORD_ONLY = "keyword_only"
    HYBRID = "hybrid"
    TABLE_FIRST = "table_first"
    GRAPH_FIRST = "graph_first"
    GRAPH_HYBRID = "graph_hybrid"
    PAGE_BM25 = "page_bm25"
    PAGE_GRAPH = "page_graph"


@dataclass
class PolicyAction:
    """A complete action decided by the RL policy."""
    retrieval_route: RetrievalRoute
    retrieval_depth: int
    graph_activation: bool
    graph_depth: int
    prompt_template: str  # "extraction", "analysis", "summary", "comparison"
    verification_level: str  # "none", "citation", "full"
    was_exploration: bool = False
    action_confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "retrieval_route": self.retrieval_route.value,
            "retrieval_depth": self.retrieval_depth,
            "graph_activation": self.graph_activation,
            "graph_depth": self.graph_depth,
            "prompt_template": self.prompt_template,
            "verification_level": self.verification_level,
            "was_exploration": self.was_exploration,
            "action_confidence": self.action_confidence,
        }


@dataclass
class Experience:
    """A single experience tuple for learning."""
    query_id: str
    state: Dict[str, Any]
    action: Dict[str, Any]
    reward: float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "state": self.state,
            "action": self.action,
            "reward": self.reward,
            "timestamp": self.timestamp,
        }


class ArmStatistics:
    """Beta distribution statistics for Thompson Sampling."""

    def __init__(self):
        self.alpha: float = 1.0  # Successes + 1 (prior)
        self.beta: float = 1.0   # Failures + 1 (prior)
        self.total_pulls: int = 0
        self.total_reward: float = 0.0
        self.reward_history: List[float] = []

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        ab = self.alpha + self.beta
        return (self.alpha * self.beta) / (ab * ab * (ab + 1))

    def sample(self) -> float:
        """Sample from the Beta distribution (Thompson Sampling)."""
        return random.betavariate(self.alpha, self.beta)

    def update(self, reward: float):
        """Update statistics with observed reward."""
        self.total_pulls += 1
        self.total_reward += reward
        self.reward_history.append(reward)

        # Update Beta distribution
        # Reward is in [0, 1], treat as Bernoulli-like
        self.alpha += reward
        self.beta += (1.0 - reward)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "total_pulls": self.total_pulls,
            "total_reward": self.total_reward,
            "mean": self.mean,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ArmStatistics":
        arm = cls()
        arm.alpha = data.get("alpha", 1.0)
        arm.beta = data.get("beta", 1.0)
        arm.total_pulls = data.get("total_pulls", 0)
        arm.total_reward = data.get("total_reward", 0.0)
        return arm


class RLPolicyEngine:
    """Contextual Bandit policy engine with Thompson Sampling.

    The engine learns which retrieval strategy works best for which
    type of query by observing evaluation rewards over time.

    Key idea: Each (context_key, action) pair maintains a Beta distribution.
    The engine samples from these distributions to balance exploration
    and exploitation automatically.
    """

    # Available retrieval depths
    DEPTH_OPTIONS = [3, 5, 8, 10, 15]

    # Prompt template options
    PROMPT_OPTIONS = ["extraction", "analysis", "summary", "comparison", "verification"]

    def __init__(self, config: Optional[RLConfig] = None, persist_dir: Optional[str] = None):
        self.config = config or RLConfig()
        self._persist_dir = persist_dir

        # Per-context arm statistics
        # Key: f"{context_key}:{action}" -> ArmStatistics
        self._arms: Dict[str, ArmStatistics] = {}

        # Experience replay buffer
        self._experiences: List[Experience] = []

        # Query counter
        self._total_queries = 0

        # Current exploration rate
        self._exploration_rate = self.config.exploration_rate

        # Default heuristic mappings (used during warmup)
        self._default_routes = {
            QueryType.FACTUAL: RetrievalRoute.KEYWORD_ONLY,
            QueryType.RELATIONAL: RetrievalRoute.GRAPH_HYBRID,
            QueryType.ANALYTICAL: RetrievalRoute.HYBRID,
            QueryType.SUMMARIZATION: RetrievalRoute.VECTOR_ONLY,
            QueryType.EXTRACTION: RetrievalRoute.KEYWORD_ONLY,
            QueryType.COMPARATIVE: RetrievalRoute.HYBRID,
            QueryType.REASONING: RetrievalRoute.HYBRID,
            QueryType.STRUCTURED_LOOKUP: RetrievalRoute.TABLE_FIRST,
        }

        # Load persisted state if available
        if self._persist_dir:
            self._load_state()

    @property
    def is_warmup(self) -> bool:
        """Whether we're still in the warmup phase."""
        return self._total_queries < self.config.warmup_queries

    @property
    def total_queries(self) -> int:
        return self._total_queries

    def decide(self, query_analysis: QueryAnalysis) -> PolicyAction:
        """Decide the optimal retrieval strategy for a query.

        During warmup (first N queries): uses heuristic defaults.
        After warmup: uses Thompson Sampling to select actions.
        """
        self._total_queries += 1
        context_key = self._make_context_key(query_analysis)

        if self.is_warmup:
            action = self._heuristic_decision(query_analysis)
            action.was_exploration = False
            action.action_confidence = 0.5  # Low confidence during warmup
            logger.info(
                f"[Warmup {self._total_queries}/{self.config.warmup_queries}] "
                f"Heuristic: {action.retrieval_route.value}"
            )
            return action

        # Thompson Sampling
        should_explore = random.random() < self._exploration_rate

        if should_explore:
            action = self._explore(query_analysis)
            action.was_exploration = True
            logger.info(f"[Explore] Random strategy: {action.retrieval_route.value}")
        else:
            action = self._exploit(query_analysis, context_key)
            action.was_exploration = False
            logger.info(
                f"[Exploit] Best strategy: {action.retrieval_route.value} "
                f"(confidence: {action.action_confidence:.2f})"
            )

        # Decay exploration rate
        self._exploration_rate = max(
            self.config.min_exploration_rate,
            self._exploration_rate * self.config.exploration_decay,
        )

        return action

    def update(self, query_id: str, query_analysis: QueryAnalysis,
               action: PolicyAction, reward: float):
        """Update policy based on observed reward.

        This is the learning step: after each query is evaluated,
        the reward signal updates the arm statistics.
        """
        context_key = self._make_context_key(query_analysis)
        action_key = self._make_action_key(action)
        full_key = f"{context_key}:{action_key}"

        # Get or create arm
        if full_key not in self._arms:
            self._arms[full_key] = ArmStatistics()

        # Update arm statistics
        self._arms[full_key].update(reward)

        # Store experience
        experience = Experience(
            query_id=query_id,
            state=query_analysis.to_dict(),
            action=action.to_dict(),
            reward=reward,
        )
        self._experiences.append(experience)

        logger.info(
            f"Policy updated: {full_key} | reward={reward:.3f} | "
            f"arm_mean={self._arms[full_key].mean:.3f} | "
            f"pulls={self._arms[full_key].total_pulls}"
        )

        # Persist state periodically
        if self._persist_dir and self._total_queries % self.config.update_frequency == 0:
            self._save_state()

    def _make_context_key(self, analysis: QueryAnalysis) -> str:
        """Create a context key for the bandit from query analysis."""
        parts = [
            analysis.query_type.value,
            analysis.complexity.value,
            analysis.domain,
        ]
        if analysis.requires_graph:
            parts.append("graph")
        if analysis.requires_table:
            parts.append("table")
        return ":".join(parts)

    def _make_action_key(self, action: PolicyAction) -> str:
        """Create an action key for the bandit."""
        return f"{action.retrieval_route.value}:d{action.retrieval_depth}:g{int(action.graph_activation)}"

    def _heuristic_decision(self, analysis: QueryAnalysis) -> PolicyAction:
        """Fallback heuristic decision during warmup."""
        route = self._default_routes.get(analysis.query_type, RetrievalRoute.HYBRID)

        # Depth based on complexity
        depth_map = {
            QueryComplexity.SIMPLE: 3,
            QueryComplexity.MODERATE: 5,
            QueryComplexity.COMPLEX: 8,
            QueryComplexity.MULTI_HOP: 10,
        }
        depth = depth_map.get(analysis.complexity, 5)

        # Graph activation
        graph_active = analysis.requires_graph
        graph_depth = 2 if graph_active else 0

        # Prompt template
        prompt_map = {
            QueryType.EXTRACTION: "extraction",
            QueryType.SUMMARIZATION: "summary",
            QueryType.COMPARATIVE: "comparison",
            QueryType.ANALYTICAL: "analysis",
            QueryType.RELATIONAL: "analysis",
        }
        prompt = prompt_map.get(analysis.query_type, "extraction")

        # Verification level
        verification = "citation" if analysis.complexity in (
            QueryComplexity.COMPLEX, QueryComplexity.MULTI_HOP
        ) else "none"

        return PolicyAction(
            retrieval_route=route,
            retrieval_depth=depth,
            graph_activation=graph_active,
            graph_depth=graph_depth,
            prompt_template=prompt,
            verification_level=verification,
        )

    def _explore(self, analysis: QueryAnalysis) -> PolicyAction:
        """Random exploration."""
        route = random.choice(list(RetrievalRoute))
        depth = random.choice(self.DEPTH_OPTIONS)
        graph_active = random.random() < 0.3  # 30% chance
        graph_depth = random.choice([1, 2, 3]) if graph_active else 0
        prompt = random.choice(self.PROMPT_OPTIONS)
        verification = random.choice(["none", "citation", "full"])

        return PolicyAction(
            retrieval_route=route,
            retrieval_depth=depth,
            graph_activation=graph_active,
            graph_depth=graph_depth,
            prompt_template=prompt,
            verification_level=verification,
            action_confidence=0.0,
        )

    def _exploit(self, analysis: QueryAnalysis, context_key: str) -> PolicyAction:
        """Thompson Sampling exploitation."""
        best_score = -1.0
        best_action = None

        # Sample from all known arms for this context
        candidate_arms = {
            k: v for k, v in self._arms.items()
            if k.startswith(context_key + ":")
        }

        if not candidate_arms:
            # No data for this context yet → use heuristic
            return self._heuristic_decision(analysis)

        for arm_key, arm_stats in candidate_arms.items():
            sampled_value = arm_stats.sample()
            if sampled_value > best_score:
                best_score = sampled_value
                best_action = arm_key

        if best_action is None:
            return self._heuristic_decision(analysis)

        # Parse action from key
        action_part = best_action.split(":", maxsplit=len(context_key.split(":")))[-1]
        # action_part format: "route:dN:gN"
        parts = action_part.rsplit(":", 2)

        try:
            route_str = parts[0] if len(parts) >= 3 else "hybrid"
            depth_str = parts[1] if len(parts) >= 3 else "d5"
            graph_str = parts[2] if len(parts) >= 3 else "g0"

            route = RetrievalRoute(route_str)
            depth = int(depth_str[1:]) if depth_str.startswith("d") else 5
            graph_active = graph_str == "g1"
        except (ValueError, IndexError):
            return self._heuristic_decision(analysis)

        arm = candidate_arms[best_action]
        return PolicyAction(
            retrieval_route=route,
            retrieval_depth=depth,
            graph_activation=graph_active,
            graph_depth=2 if graph_active else 0,
            prompt_template=self._best_prompt_for_context(context_key),
            verification_level="citation" if arm.mean > 0.8 else "none",
            action_confidence=arm.mean,
        )

    def _best_prompt_for_context(self, context_key: str) -> str:
        """Get the best prompt template for a context."""
        if "analytical" in context_key or "reasoning" in context_key:
            return "analysis"
        elif "summarization" in context_key:
            return "summary"
        elif "comparative" in context_key:
            return "comparison"
        return "extraction"

    def get_stats(self) -> Dict[str, Any]:
        """Get policy statistics for monitoring."""
        arm_stats = {}
        for key, arm in self._arms.items():
            arm_stats[key] = arm.to_dict()

        return {
            "total_queries": self._total_queries,
            "is_warmup": self.is_warmup,
            "warmup_progress": f"{self._total_queries}/{self.config.warmup_queries}",
            "exploration_rate": self._exploration_rate,
            "total_arms": len(self._arms),
            "total_experiences": len(self._experiences),
            "arms": arm_stats,
        }

    def get_learning_curve(self) -> List[Dict[str, Any]]:
        """Get the learning curve data for visualization."""
        if not self._experiences:
            return []

        window_size = 5
        curve = []
        rewards = [e.reward for e in self._experiences]

        for i in range(len(rewards)):
            start = max(0, i - window_size + 1)
            window = rewards[start:i + 1]
            curve.append({
                "query_number": i + 1,
                "reward": rewards[i],
                "rolling_avg": sum(window) / len(window),
            })

        return curve

    def _save_state(self):
        """Persist policy state to disk."""
        if not self._persist_dir:
            return

        path = Path(self._persist_dir)
        path.mkdir(parents=True, exist_ok=True)

        state = {
            "total_queries": self._total_queries,
            "exploration_rate": self._exploration_rate,
            "arms": {k: v.to_dict() for k, v in self._arms.items()},
            "experiences": [e.to_dict() for e in self._experiences[-1000:]],  # Keep last 1000
        }

        state_file = path / "rl_policy_state.json"
        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)

        logger.debug(f"Policy state saved to {state_file}")

    def _load_state(self):
        """Load persisted policy state."""
        if not self._persist_dir:
            return

        state_file = Path(self._persist_dir) / "rl_policy_state.json"
        if not state_file.exists():
            return

        try:
            with open(state_file) as f:
                state = json.load(f)

            self._total_queries = state.get("total_queries", 0)
            self._exploration_rate = state.get("exploration_rate", self.config.exploration_rate)

            for key, arm_data in state.get("arms", {}).items():
                self._arms[key] = ArmStatistics.from_dict(arm_data)

            for exp_data in state.get("experiences", []):
                self._experiences.append(Experience(
                    query_id=exp_data["query_id"],
                    state=exp_data["state"],
                    action=exp_data["action"],
                    reward=exp_data["reward"],
                    timestamp=exp_data.get("timestamp", 0),
                ))

            logger.info(
                f"Policy state loaded: {self._total_queries} queries, "
                f"{len(self._arms)} arms"
            )
        except Exception as e:
            logger.warning(f"Failed to load policy state: {e}")
