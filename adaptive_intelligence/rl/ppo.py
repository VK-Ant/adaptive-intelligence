"""PPO-based RL Policy — v3 addition.

Proximal Policy Optimization for retrieval routing.
Uses a simple neural policy network with state features.
Falls back to Thompson Sampling if torch not available.
"""

import logging
import math
import random
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PPOExperience:
    """Single experience for PPO training."""
    state: List[float]
    action_idx: int
    reward: float
    log_prob: float
    value: float


class PPOPolicy:
    """Simple PPO policy using tabular state-action values.

    Uses a tabular approach (no PyTorch dependency):
    - State: query features encoded as tuple
    - Action: retrieval route index
    - Policy: softmax over learned Q-values
    - Update: clipped surrogate objective
    """

    def __init__(self, n_actions: int = 8, lr: float = 0.01,
                 gamma: float = 0.99, clip_epsilon: float = 0.2,
                 entropy_coeff: float = 0.01):
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.entropy_coeff = entropy_coeff

        # Q-table: state_key -> [action_values]
        self._q_table: Dict[str, List[float]] = {}
        # Value table: state_key -> value
        self._v_table: Dict[str, float] = {}
        # Experience buffer
        self._buffer: List[PPOExperience] = []
        self._update_frequency = 10
        self._step_count = 0

    def _get_state_key(self, state_features: List[float]) -> str:
        """Discretize state features into a key."""
        return str(tuple(round(f, 1) for f in state_features))

    def _get_q_values(self, state_key: str) -> List[float]:
        """Get Q-values for a state, initializing if needed."""
        if state_key not in self._q_table:
            self._q_table[state_key] = [0.0] * self.n_actions
        return self._q_table[state_key]

    def _softmax(self, values: List[float], temperature: float = 1.0) -> List[float]:
        """Compute softmax probabilities."""
        max_v = max(values)
        exp_v = [math.exp((v - max_v) / max(temperature, 0.01)) for v in values]
        total = sum(exp_v)
        return [e / total for e in exp_v]

    def select_action(self, state_features: List[float],
                      temperature: float = 1.0) -> Tuple[int, float]:
        """Select action using softmax policy.

        Returns: (action_index, log_probability)
        """
        state_key = self._get_state_key(state_features)
        q_values = self._get_q_values(state_key)
        probs = self._softmax(q_values, temperature)

        # Sample from distribution
        r = random.random()
        cumulative = 0.0
        action = 0
        for i, p in enumerate(probs):
            cumulative += p
            if r <= cumulative:
                action = i
                break

        log_prob = math.log(max(probs[action], 1e-10))
        return action, log_prob

    def get_value(self, state_features: List[float]) -> float:
        """Get estimated state value."""
        state_key = self._get_state_key(state_features)
        return self._v_table.get(state_key, 0.0)

    def store_experience(self, state: List[float], action_idx: int,
                         reward: float, log_prob: float, value: float):
        """Store experience for batch update."""
        self._buffer.append(PPOExperience(
            state=state, action_idx=action_idx,
            reward=reward, log_prob=log_prob, value=value,
        ))
        self._step_count += 1

        # Update periodically
        if self._step_count % self._update_frequency == 0 and len(self._buffer) >= self._update_frequency:
            self._update()

    def _update(self):
        """PPO-style update using clipped surrogate objective."""
        if not self._buffer:
            return

        # Compute advantages
        rewards = [e.reward for e in self._buffer]
        values = [e.value for e in self._buffer]

        advantages = []
        for i in range(len(rewards)):
            if i < len(rewards) - 1:
                advantage = rewards[i] + self.gamma * values[i + 1] - values[i]
            else:
                advantage = rewards[i] - values[i]
            advantages.append(advantage)

        # Normalize advantages
        if len(advantages) > 1:
            mean_adv = sum(advantages) / len(advantages)
            std_adv = max(math.sqrt(sum((a - mean_adv) ** 2 for a in advantages) / len(advantages)), 1e-8)
            advantages = [(a - mean_adv) / std_adv for a in advantages]

        # Update Q-values and V-values
        for exp, advantage in zip(self._buffer, advantages):
            state_key = self._get_state_key(exp.state)
            q_values = self._get_q_values(state_key)

            # Clipped update
            ratio = 1.0  # Simplified: on-policy so ratio ≈ 1
            clipped_advantage = max(min(advantage, self.clip_epsilon), -self.clip_epsilon)
            update = min(ratio * advantage, clipped_advantage)

            # Update Q-value for selected action
            q_values[exp.action_idx] += self.lr * update

            # Update value estimate
            v = self._v_table.get(state_key, 0.0)
            self._v_table[state_key] = v + self.lr * (exp.reward - v)

        # Clear buffer
        self._buffer.clear()
        logger.debug(f"PPO update complete: {len(self._q_table)} states")

    def get_stats(self) -> Dict[str, Any]:
        return {
            "algorithm": "ppo",
            "states_learned": len(self._q_table),
            "buffer_size": len(self._buffer),
            "step_count": self._step_count,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "q_table": self._q_table,
            "v_table": self._v_table,
            "step_count": self._step_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], n_actions: int = 8) -> "PPOPolicy":
        policy = cls(n_actions=n_actions)
        policy._q_table = data.get("q_table", {})
        policy._v_table = data.get("v_table", {})
        policy._step_count = data.get("step_count", 0)
        return policy
