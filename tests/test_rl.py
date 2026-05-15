"""Tests for the RL Policy Engine."""

import pytest
from adaptive_intelligence.rl import (
    RLPolicyEngine, PolicyAction, RetrievalRoute, ArmStatistics, Experience,
)
from adaptive_intelligence.query import (
    TriggerInterpreter, QueryAnalysis, QueryType, QueryComplexity,
)
from adaptive_intelligence.core.config import RLConfig


@pytest.fixture
def rl_engine():
    config = RLConfig(warmup_queries=5, exploration_rate=0.3)
    return RLPolicyEngine(config=config)


@pytest.fixture
def interpreter():
    return TriggerInterpreter()


class TestWarmupPhase:
    def test_starts_in_warmup(self, rl_engine):
        assert rl_engine.is_warmup
        assert rl_engine.total_queries == 0

    def test_warmup_uses_heuristics(self, rl_engine, interpreter):
        analysis = interpreter.analyze("What is the total revenue?")
        action = rl_engine.decide(analysis)
        assert not action.was_exploration
        assert action.action_confidence == 0.5  # Warmup confidence

    def test_exits_warmup(self, rl_engine, interpreter):
        for i in range(6):
            analysis = interpreter.analyze(f"Query number {i}")
            rl_engine.decide(analysis)
        assert not rl_engine.is_warmup

    def test_warmup_heuristic_routes(self, rl_engine, interpreter):
        # Factual → keyword
        analysis = interpreter.analyze("What is the company name?")
        action = rl_engine.decide(analysis)
        assert action.retrieval_route in (RetrievalRoute.KEYWORD_ONLY, RetrievalRoute.HYBRID)

        # Summarization → vector
        analysis = interpreter.analyze("Summarize the document overview")
        action = rl_engine.decide(analysis)
        assert action.retrieval_route == RetrievalRoute.VECTOR_ONLY


class TestPolicyDecision:
    def test_returns_policy_action(self, rl_engine, interpreter):
        analysis = interpreter.analyze("What is the revenue?")
        action = rl_engine.decide(analysis)
        assert isinstance(action, PolicyAction)
        assert isinstance(action.retrieval_route, RetrievalRoute)
        assert action.retrieval_depth > 0

    def test_graph_activation_for_relational(self, rl_engine, interpreter):
        analysis = interpreter.analyze("How is Alpha connected to Beta?")
        action = rl_engine.decide(analysis)
        assert action.graph_activation  # Should activate for relational queries


class TestPolicyUpdate:
    def test_update_with_reward(self, rl_engine, interpreter):
        analysis = interpreter.analyze("What is the revenue?")
        action = rl_engine.decide(analysis)
        # Update with high reward
        rl_engine.update("q1", analysis, action, reward=0.9)
        stats = rl_engine.get_stats()
        assert stats["total_experiences"] >= 1

    def test_learning_curve(self, rl_engine, interpreter):
        for i in range(10):
            analysis = interpreter.analyze(f"Query {i} about revenue")
            action = rl_engine.decide(analysis)
            rl_engine.update(f"q{i}", analysis, action, reward=0.5 + i * 0.05)

        curve = rl_engine.get_learning_curve()
        assert len(curve) == 10
        assert all("rolling_avg" in point for point in curve)


class TestArmStatistics:
    def test_initial_state(self):
        arm = ArmStatistics()
        assert arm.alpha == 1.0
        assert arm.beta == 1.0
        assert arm.total_pulls == 0

    def test_update(self):
        arm = ArmStatistics()
        arm.update(0.8)
        assert arm.total_pulls == 1
        assert arm.alpha > 1.0

    def test_sample_range(self):
        arm = ArmStatistics()
        for _ in range(100):
            arm.update(0.7)
        samples = [arm.sample() for _ in range(100)]
        assert all(0.0 <= s <= 1.0 for s in samples)
        # Should be biased toward 0.7 after many updates
        avg = sum(samples) / len(samples)
        assert 0.5 < avg < 0.9

    def test_serialization(self):
        arm = ArmStatistics()
        arm.update(0.8)
        arm.update(0.6)
        data = arm.to_dict()
        restored = ArmStatistics.from_dict(data)
        assert restored.alpha == arm.alpha
        assert restored.beta == arm.beta
        assert restored.total_pulls == arm.total_pulls


class TestExplorationDecay:
    def test_exploration_decays(self, interpreter):
        config = RLConfig(warmup_queries=2, exploration_rate=0.5, exploration_decay=0.9)
        engine = RLPolicyEngine(config=config)

        initial_rate = engine._exploration_rate
        # Run past warmup
        for i in range(10):
            analysis = interpreter.analyze(f"Query {i}")
            engine.decide(analysis)

        assert engine._exploration_rate < initial_rate


class TestStats:
    def test_get_stats(self, rl_engine):
        stats = rl_engine.get_stats()
        assert "total_queries" in stats
        assert "is_warmup" in stats
        assert "exploration_rate" in stats
        assert "total_arms" in stats
