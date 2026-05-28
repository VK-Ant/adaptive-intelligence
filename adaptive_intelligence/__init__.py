"""
Adaptive Intelligence v3 — Self-Improving Retrieval Orchestration Framework

v3: PPO/DQN algorithms, cross-encoder reranking, multi-query decomposition,
pre-trained domain policies, transfer learning, A/B testing.

Usage:
    from adaptive_intelligence import AdaptiveAI

    engine = AdaptiveAI()
    engine.ingest("./documents")
    response = engine.ask("What are the key risks?")

    # v3: PPO + reranking + pretrained
    engine = AdaptiveAI(rl_algorithm="ppo", reranking=True, pretrained_policy=True)
"""

__version__ = "3.0.1"
__author__ = "Venkatkumar Rajan"

from adaptive_intelligence.core.engine import AdaptiveAI
from adaptive_intelligence.core.response import AdaptiveResponse

__all__ = ["AdaptiveAI", "AdaptiveResponse"]
