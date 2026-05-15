"""Tests for the Query Understanding (Trigger Interpreter) module."""

import pytest
from adaptive_intelligence.query import (
    TriggerInterpreter, QueryAnalysis, QueryType, QueryComplexity, QueryIntent,
)


@pytest.fixture
def interpreter():
    return TriggerInterpreter()


class TestQueryTypeDetection:
    def test_factual_query(self, interpreter):
        analysis = interpreter.analyze("What is the company's founding date?")
        assert analysis.query_type in (QueryType.FACTUAL, QueryType.EXTRACTION)

    def test_relational_query(self, interpreter):
        analysis = interpreter.analyze("How is Alpha Corp connected to Beta Ltd?")
        assert analysis.query_type == QueryType.RELATIONAL
        assert analysis.has_relationship_indicators

    def test_analytical_query(self, interpreter):
        analysis = interpreter.analyze("Analyze the risk factors in the Q3 report")
        assert analysis.query_type == QueryType.ANALYTICAL

    def test_summarization_query(self, interpreter):
        analysis = interpreter.analyze("Provide a summary of the annual report")
        assert analysis.query_type == QueryType.SUMMARIZATION

    def test_comparative_query(self, interpreter):
        analysis = interpreter.analyze("Compare revenue between Q1 and Q2")
        assert analysis.query_type == QueryType.COMPARATIVE

    def test_structured_lookup(self, interpreter):
        analysis = interpreter.analyze("What is the total revenue for 2024?")
        assert analysis.query_type == QueryType.STRUCTURED_LOOKUP

    def test_reasoning_query(self, interpreter):
        analysis = interpreter.analyze("Why did the profit margin decline last quarter?")
        assert analysis.query_type == QueryType.REASONING


class TestEntityDetection:
    def test_proper_nouns(self, interpreter):
        analysis = interpreter.analyze("What did Acme Corporation report in their filing?")
        assert "Acme Corporation" in analysis.entities or "Acme" in analysis.entities

    def test_acronyms(self, interpreter):
        analysis = interpreter.analyze("What is the EBITDA for FY2024?")
        assert "EBITDA" in analysis.entities

    def test_filters_common_words(self, interpreter):
        analysis = interpreter.analyze("What is the revenue?")
        assert "What" not in analysis.entities


class TestTemporalDetection:
    def test_quarter_detection(self, interpreter):
        analysis = interpreter.analyze("What happened in Q3?")
        assert analysis.has_temporal_markers

    def test_year_detection(self, interpreter):
        analysis = interpreter.analyze("Revenue in 2024 was strong")
        assert analysis.has_temporal_markers

    def test_relative_time(self, interpreter):
        analysis = interpreter.analyze("What changed last quarter?")
        assert analysis.has_temporal_markers

    def test_no_temporal(self, interpreter):
        analysis = interpreter.analyze("What is the company structure?")
        assert not analysis.has_temporal_markers


class TestDomainDetection:
    def test_financial_domain(self, interpreter):
        analysis = interpreter.analyze("What is the EBITDA margin and ROI?")
        assert analysis.domain == "financial"

    def test_legal_domain(self, interpreter):
        analysis = interpreter.analyze("What are the compliance regulations?")
        assert analysis.domain == "legal"

    def test_healthcare_domain(self, interpreter):
        analysis = interpreter.analyze("What is the patient treatment protocol?")
        assert analysis.domain == "healthcare"

    def test_general_domain(self, interpreter):
        analysis = interpreter.analyze("Tell me about the weather")
        assert analysis.domain == "general"


class TestComplexity:
    def test_simple_query(self, interpreter):
        analysis = interpreter.analyze("What is the company name?")
        assert analysis.complexity in (QueryComplexity.SIMPLE, QueryComplexity.MODERATE)

    def test_complex_query(self, interpreter):
        analysis = interpreter.analyze(
            "How is Acme Corp connected to Beta Ltd and what are the financial "
            "implications of their partnership on Q3 2024 revenue?"
        )
        assert analysis.complexity in (QueryComplexity.COMPLEX, QueryComplexity.MULTI_HOP)


class TestGraphActivation:
    def test_requires_graph_for_relationships(self, interpreter):
        analysis = interpreter.analyze("What entities are connected to Acme Corp?")
        assert analysis.requires_graph

    def test_no_graph_for_simple(self, interpreter):
        analysis = interpreter.analyze("What is the company name?")
        assert not analysis.requires_graph


class TestConfidence:
    def test_confidence_range(self, interpreter):
        analysis = interpreter.analyze("What is the EBITDA for Q3 2024?")
        assert 0.0 <= analysis.confidence <= 1.0

    def test_higher_confidence_for_specific(self, interpreter):
        specific = interpreter.analyze("What is the EBITDA for Acme Corp in Q3 2024?")
        vague = interpreter.analyze("Tell me something")
        assert specific.confidence >= vague.confidence
