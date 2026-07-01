"""Loop Engineering — v4.0.7 addition.

Optimizes the RL feedback loop itself:
- Adaptive warmup: faster warmup for domains with enough data
- Per-domain exploration rates: explore more where data is sparse
- Reward shaping: different weights for different decision types
- Meta-learning: learn how fast to learn per domain
- Convergence detection: stop exploring when policy is stable

Instead of one fixed learning rate for everything, the loop
adapts its own learning behavior based on what it has seen.
"""

import logging
import time
import math
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class DomainState:
    """Learning state for a specific domain/query-type combination."""
    domain: str
    query_type: str
    query_count: int = 0
    exploration_rate: float = 0.20
    warmup_remaining: int = 15
    converged: bool = False
    best_strategy: str = ""
    best_score: float = 0.0
    recent_scores: List[float] = field(default_factory=list)
    score_variance: float = 1.0
    last_updated: float = field(default_factory=time.time)


class LoopEngineer:
    """Optimizes the RL feedback loop.

    Standard RL uses fixed exploration rates and warmup periods.
    Loop engineering adapts these per domain and query type:

    - Domain with 50 queries: low exploration (3%), exploit learned policy
    - Domain with 5 queries: high exploration (25%), still learning
    - New query type detected: reset exploration for that type
    - Score variance dropping: convergence detected, reduce exploration
    - Score variance high: policy unstable, increase exploration
    """

    def __init__(self, base_warmup: int = 15,
                 base_exploration: float = 0.20,
                 min_exploration: float = 0.03,
                 convergence_threshold: float = 0.02):
        self.base_warmup = base_warmup
        self.base_exploration = base_exploration
        self.min_exploration = min_exploration
        self.convergence_threshold = convergence_threshold
        self._domains: Dict[str, DomainState] = {}
        self._global_query_count = 0
        self._max_recent_scores = 20

    def get_domain_key(self, domain: str, query_type: str) -> str:
        """Generate key for domain + query type combination."""
        return f"{domain}:{query_type}"

    def get_state(self, domain: str, query_type: str) -> DomainState:
        """Get or create learning state for a domain/query-type."""
        key = self.get_domain_key(domain, query_type)
        if key not in self._domains:
            self._domains[key] = DomainState(
                domain=domain, query_type=query_type,
                exploration_rate=self.base_exploration,
                warmup_remaining=self.base_warmup,
            )
        return self._domains[key]

    def should_explore(self, domain: str, query_type: str) -> bool:
        """Should the RL explore or exploit for this query?

        Returns True if the system should try a random strategy,
        False if it should use the best known strategy.
        """
        state = self.get_state(domain, query_type)

        # During warmup, always explore
        if state.warmup_remaining > 0:
            return True

        # If converged, rarely explore
        if state.converged:
            return hash(time.time()) % 100 < (self.min_exploration * 100)

        # Otherwise, use adaptive exploration rate
        return hash(time.time()) % 100 < (state.exploration_rate * 100)

    def get_exploration_rate(self, domain: str, query_type: str) -> float:
        """Get current exploration rate for this domain/query-type."""
        state = self.get_state(domain, query_type)
        if state.warmup_remaining > 0:
            return 1.0  # Full exploration during warmup
        return state.exploration_rate

    def is_warmup(self, domain: str, query_type: str) -> bool:
        """Is this domain/query-type still in warmup?"""
        state = self.get_state(domain, query_type)
        return state.warmup_remaining > 0

    def update(self, domain: str, query_type: str,
               strategy: str, score: float,
               harness_rewards: Dict[str, float] = None):
        """Update learning state after a query.

        Args:
            domain: Query domain (financial, healthcare, etc.)
            query_type: Query type (factual, relational, etc.)
            strategy: Strategy that was used
            score: Answer score (0-1)
            harness_rewards: Per-decision rewards from HarnessAgent
        """
        state = self.get_state(domain, query_type)
        state.query_count += 1
        state.last_updated = time.time()
        self._global_query_count += 1

        # Update warmup
        if state.warmup_remaining > 0:
            state.warmup_remaining -= 1
            if state.warmup_remaining == 0:
                logger.info(
                    f"Warmup complete for {domain}:{query_type} "
                    f"after {state.query_count} queries"
                )

        # Track best strategy
        if score > state.best_score:
            state.best_score = score
            state.best_strategy = strategy

        # Track recent scores for variance
        state.recent_scores.append(score)
        if len(state.recent_scores) > self._max_recent_scores:
            state.recent_scores = state.recent_scores[-self._max_recent_scores:]

        # Update variance
        if len(state.recent_scores) >= 5:
            mean = sum(state.recent_scores) / len(state.recent_scores)
            state.score_variance = sum(
                (s - mean) ** 2 for s in state.recent_scores
            ) / len(state.recent_scores)

        # Adaptive exploration rate
        state.exploration_rate = self._compute_exploration_rate(state)

        # Check convergence
        state.converged = self._check_convergence(state)

        # Apply harness rewards for faster learning
        if harness_rewards:
            self._apply_harness_feedback(state, harness_rewards)

    def _compute_exploration_rate(self, state: DomainState) -> float:
        """Compute adaptive exploration rate.

        More queries → less exploration.
        High variance → more exploration.
        Low variance → less exploration.
        """
        if state.warmup_remaining > 0:
            return 1.0

        # Base decay: exploration decreases with query count
        decay = 1.0 / (1.0 + state.query_count / 20.0)
        base_rate = self.base_exploration * decay

        # Variance adjustment: high variance → explore more
        if state.score_variance > 0.05:
            variance_factor = 1.5  # Scores are unstable, explore more
        elif state.score_variance < 0.01:
            variance_factor = 0.5  # Scores are stable, exploit more
        else:
            variance_factor = 1.0

        rate = base_rate * variance_factor

        # Clamp
        return max(self.min_exploration, min(rate, self.base_exploration))

    def _check_convergence(self, state: DomainState) -> bool:
        """Check if the policy has converged for this domain/query-type."""
        if state.query_count < 30:
            return False

        if len(state.recent_scores) < 10:
            return False

        # Converged if score variance is very low
        return state.score_variance < self.convergence_threshold

    def _apply_harness_feedback(self, state: DomainState,
                                harness_rewards: Dict[str, float]):
        """Use harness feedback to speed up learning.

        If harness shows high efficiency, reduce exploration faster.
        If harness shows low efficiency, increase exploration.
        """
        efficiency_bonus = harness_rewards.get("efficiency_bonus", 0)
        combined = harness_rewards.get("combined", 0)

        if combined > 0.7:
            # Good pipeline, reduce exploration faster
            state.exploration_rate *= 0.9
        elif combined < 0.3:
            # Poor pipeline, explore more
            state.exploration_rate = min(
                state.exploration_rate * 1.2,
                self.base_exploration
            )

        state.exploration_rate = max(
            self.min_exploration, state.exploration_rate
        )

    def shape_reward(self, base_reward: float,
                     harness_rewards: Dict[str, float] = None) -> float:
        """Shape the RL reward using harness signals.

        Instead of just the answer score, combine:
        - Answer quality (60%)
        - Decision efficiency (20%)
        - Per-decision correctness (20%)
        """
        if not harness_rewards:
            return base_reward

        combined = harness_rewards.get("combined", base_reward)
        efficiency = harness_rewards.get("efficiency_bonus", 0)

        shaped = (
            base_reward * 0.6 +
            combined * 0.2 +
            efficiency * 2.0 * 0.2  # efficiency_bonus is 0-0.1, scale up
        )

        return max(0.0, min(1.0, shaped))

    def get_stats(self) -> Dict[str, Any]:
        """Get loop engineering statistics."""
        domains = {}
        for key, state in self._domains.items():
            domains[key] = {
                "queries": state.query_count,
                "exploration": f"{state.exploration_rate:.1%}",
                "converged": state.converged,
                "warmup_left": state.warmup_remaining,
                "best_strategy": state.best_strategy,
                "best_score": f"{state.best_score:.0%}",
                "variance": f"{state.score_variance:.4f}",
            }

        converged_count = sum(1 for s in self._domains.values() if s.converged)

        return {
            "global_queries": self._global_query_count,
            "domain_count": len(self._domains),
            "converged": converged_count,
            "domains": domains,
        }

    def get_recommended_warmup(self, domain: str, query_type: str) -> int:
        """Get recommended warmup for a domain/query-type.

        If similar domains have converged quickly, suggest shorter warmup.
        """
        state = self.get_state(domain, query_type)

        # If this domain has been seen before, shorter warmup
        if state.query_count > 0:
            return max(0, state.warmup_remaining)

        # Check if any similar domain converged fast
        for key, other in self._domains.items():
            other_domain = other.domain
            if other_domain == domain and other.converged:
                # Same domain, different query type converged — shorter warmup
                return max(5, self.base_warmup // 2)

        return self.base_warmup
