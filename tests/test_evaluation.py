"""Tests for the Evaluation Engine."""

import pytest
from adaptive_intelligence.evaluation import EvaluationEngine
from adaptive_intelligence.core.config import EvaluationConfig
from adaptive_intelligence.ingestion.chunker import Chunk


@pytest.fixture
def evaluator():
    return EvaluationEngine()


@pytest.fixture
def sample_chunks():
    return [
        Chunk(chunk_id="c1", doc_id="d1",
              content="Acme Corporation reported revenue of $500 million in Q3 2024. "
                      "The company's operating margin improved to 15.2% from 12.8% in the prior year.",
              source_document="annual_report.pdf"),
        Chunk(chunk_id="c2", doc_id="d1",
              content="The board of directors approved a quarterly dividend of $0.50 per share. "
                      "Total shareholder returns exceeded 20% for the fiscal year.",
              source_document="annual_report.pdf"),
    ]


class TestFaithfulness:
    def test_grounded_answer(self, evaluator, sample_chunks):
        answer = "Acme Corporation reported revenue of $500 million in Q3 2024 with an operating margin of 15.2%."
        result = evaluator.evaluate("What is the revenue?", answer, sample_chunks)
        assert result.faithfulness > 0.5

    def test_ungrounded_answer(self, evaluator, sample_chunks):
        answer = "The weather in Paris is lovely this time of year with temperatures reaching 25 degrees."
        result = evaluator.evaluate("What is the revenue?", answer, sample_chunks)
        assert result.faithfulness < 0.5

    def test_empty_answer(self, evaluator, sample_chunks):
        result = evaluator.evaluate("What is the revenue?", "", sample_chunks)
        assert result.faithfulness == 0.0


class TestRelevance:
    def test_relevant_answer(self, evaluator, sample_chunks):
        result = evaluator.evaluate(
            "What is the revenue?",
            "The revenue was $500 million in Q3 2024.",
            sample_chunks,
        )
        assert result.relevance > 0.3

    def test_irrelevant_answer(self, evaluator, sample_chunks):
        result = evaluator.evaluate(
            "What is the revenue?",
            "The sky is blue and the grass is green.",
            sample_chunks,
        )
        assert result.relevance < 0.5


class TestHallucinationRisk:
    def test_low_risk_grounded(self, evaluator, sample_chunks):
        answer = "Revenue was $500 million. The operating margin was 15.2%."
        result = evaluator.evaluate("Revenue details?", answer, sample_chunks)
        assert result.hallucination_risk < 0.5

    def test_high_risk_fabricated(self, evaluator, sample_chunks):
        answer = ("The company is headquartered on Mars. "
                  "They employ 50,000 aliens. "
                  "Their stock price is 1 billion dollars.")
        result = evaluator.evaluate("Company details?", answer, sample_chunks)
        assert result.hallucination_risk > 0.3


class TestCompositeScore:
    def test_score_range(self, evaluator, sample_chunks):
        result = evaluator.evaluate(
            "What is the revenue?",
            "Revenue was $500 million according to the annual report.",
            sample_chunks,
        )
        assert 0.0 <= result.composite_score <= 1.0

    def test_confidence_levels(self, evaluator, sample_chunks):
        result = evaluator.evaluate(
            "Revenue?",
            "Revenue was $500 million in Q3 2024 per the annual report.",
            sample_chunks,
        )
        assert result.confidence_level in ("high", "medium", "low")


class TestReward:
    def test_compute_reward(self, evaluator, sample_chunks):
        result = evaluator.evaluate(
            "Revenue?",
            "Revenue was $500 million.",
            sample_chunks,
        )
        reward = evaluator.compute_reward(result)
        assert 0.0 <= reward <= 1.0
        assert reward == result.composite_score


class TestHistory:
    def test_history_tracking(self, evaluator, sample_chunks):
        evaluator.evaluate("Q1?", "A1", sample_chunks)
        evaluator.evaluate("Q2?", "A2", sample_chunks)
        history = evaluator.get_history()
        assert len(history) == 2

    def test_average_score(self, evaluator, sample_chunks):
        evaluator.evaluate("Q1?", "Revenue was $500 million.", sample_chunks)
        avg = evaluator.get_average_score()
        assert 0.0 <= avg <= 1.0


class TestDisplay:
    def test_evaluation_display(self, evaluator, sample_chunks):
        result = evaluator.evaluate("Revenue?", "Revenue was $500 million.", sample_chunks)
        display = result.display()
        assert "Source Grounding" in display
        assert "Query Relevance" in display
        assert "Confidence" in display
