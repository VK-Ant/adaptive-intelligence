"""Tests for core engine, config, response objects, and integration."""

import os
import pytest
import tempfile
from pathlib import Path

from adaptive_intelligence.core.config import (
    AdaptiveConfig, LLMBackend, SecurityLevel, Domain,
    ChunkingConfig, RLConfig, GraphConfig, EvaluationConfig,
)
from adaptive_intelligence.core.response import (
    AdaptiveResponse, Citation, RetrievalInfo, EvaluationResult, PolicyDecision,
)
from adaptive_intelligence.core.engine import AdaptiveAI


class TestAdaptiveConfig:
    def test_defaults(self):
        config = AdaptiveConfig()
        assert config.llm_backend == LLMBackend.OLLAMA
        assert config.domain == Domain.GENERAL
        assert config.security_level == SecurityLevel.STANDARD

    def test_custom_config(self):
        config = AdaptiveConfig(
            llm_backend=LLMBackend.OPENAI,
            llm_model="gpt-4o",
            domain=Domain.FINANCIAL,
            security_level=SecurityLevel.HIGH,
        )
        assert config.llm_backend == LLMBackend.OPENAI
        assert config.domain == Domain.FINANCIAL

    def test_nested_configs(self):
        config = AdaptiveConfig()
        assert isinstance(config.chunking, ChunkingConfig)
        assert isinstance(config.rl, RLConfig)
        assert isinstance(config.graph, GraphConfig)
        assert isinstance(config.evaluation, EvaluationConfig)

    def test_serialization(self):
        config = AdaptiveConfig(llm_model="test-model", domain=Domain.LEGAL)
        data = config.to_dict()
        restored = AdaptiveConfig.from_dict(data)
        assert restored.llm_model == "test-model"
        assert restored.domain == Domain.LEGAL


class TestAdaptiveResponse:
    def test_basic_response(self):
        response = AdaptiveResponse(answer="Test answer", confidence=0.85)
        assert response.answer == "Test answer"
        assert response.confidence == 0.85

    def test_citations(self):
        response = AdaptiveResponse(
            answer="Test",
            confidence=0.9,
            citations=[
                Citation(text="source text", source_document="doc.pdf",
                         chunk_id="c1", confidence=0.95),
            ],
        )
        assert len(response.citations) == 1
        assert response.citations[0].source_document == "doc.pdf"

    def test_display(self):
        response = AdaptiveResponse(
            answer="Revenue was $500M",
            confidence=0.85,
            evaluation=EvaluationResult(
                faithfulness=0.9, relevance=0.8, composite_score=0.85,
                confidence_level="high",
            ),
        )
        display = response.display()
        assert "Revenue was $500M" in display
        assert "Confidence" in display

    def test_retrieval_strategy(self):
        response = AdaptiveResponse(
            answer="Test",
            confidence=0.5,
            retrieval_info=RetrievalInfo(
                strategy="hybrid",
                graph_activated=True,
                graph_hops=2,
                chunks_retrieved=5,
            ),
        )
        assert "hybrid" in response.retrieval_strategy
        assert "graph" in response.retrieval_strategy


class TestEvaluationResult:
    def test_confidence_emoji(self):
        high = EvaluationResult(composite_score=0.9)
        assert high.confidence_emoji == "🟢"
        medium = EvaluationResult(composite_score=0.75)
        assert medium.confidence_emoji == "🟡"
        low = EvaluationResult(composite_score=0.5)
        assert low.confidence_emoji == "🔴"

    def test_display(self):
        result = EvaluationResult(
            faithfulness=0.9, relevance=0.8, hallucination_risk=0.1,
            composite_score=0.85, confidence_level="high",
        )
        display = result.display()
        assert "Source Grounding" in display
        assert "█" in display


class TestAdaptiveAIInit:
    def test_default_init(self):
        with tempfile.TemporaryDirectory() as d:
            engine = AdaptiveAI(storage_dir=d, log_level="WARNING")
            assert engine.config.llm_backend == LLMBackend.OLLAMA
            status = engine.status()
            assert status["version"] == "0.1.0"

    def test_kwargs_init(self):
        with tempfile.TemporaryDirectory() as d:
            engine = AdaptiveAI(
                domain="financial",
                llm_backend="ollama",
                llm_model="llama3.2",
                storage_dir=d,
                log_level="WARNING",
            )
            assert engine.config.domain == Domain.FINANCIAL

    def test_dashboard(self):
        with tempfile.TemporaryDirectory() as d:
            engine = AdaptiveAI(storage_dir=d, log_level="WARNING")
            dashboard = engine.dashboard()
            assert "ADAPTIVE INTELLIGENCE DASHBOARD" in dashboard

    def test_reset(self):
        with tempfile.TemporaryDirectory() as d:
            engine = AdaptiveAI(storage_dir=d, log_level="WARNING")
            engine.reset()
            assert engine._total_queries == 0


def _chromadb_embedding_available():
    """Check if ChromaDB can download its embedding model."""
    try:
        import chromadb
        client = chromadb.Client()
        col = client.get_or_create_collection("_test_embed")
        col.add(ids=["t1"], documents=["test"])
        client.delete_collection("_test_embed")
        return True
    except Exception:
        return False

_chromadb_ok = _chromadb_embedding_available()
skip_no_chromadb = pytest.mark.skipif(not _chromadb_ok, reason="ChromaDB embedding model unavailable")


class TestAdaptiveAIIngestion:
    @skip_no_chromadb
    def test_ingest_text_files(self):
        with tempfile.TemporaryDirectory() as storage_dir:
            with tempfile.TemporaryDirectory() as doc_dir:
                # Create test documents
                for i in range(3):
                    with open(os.path.join(doc_dir, f"doc{i}.txt"), "w") as f:
                        f.write(f"Document {i} contains important information about topic {i}. " * 10)

                engine = AdaptiveAI(storage_dir=storage_dir, log_level="WARNING")
                stats = engine.ingest(doc_dir)

                assert stats.successful == 3
                assert stats.total_chunks >= 3
                assert engine.vector_index.count() > 0
                assert engine.keyword_index.count() > 0

    @skip_no_chromadb
    def test_ingest_single_file(self):
        with tempfile.TemporaryDirectory() as storage_dir:
            with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
                f.write("Single document with revenue data of $500 million. " * 10)
                f.flush()

                try:
                    engine = AdaptiveAI(storage_dir=storage_dir, log_level="WARNING")
                    stats = engine.ingest(f.name)
                    assert stats.successful == 1
                finally:
                    os.unlink(f.name)


class TestAdaptiveAIAsk:
    """Test ask() without a real LLM (fallback mode)."""

    @skip_no_chromadb
    def test_ask_fallback_mode(self):
        with tempfile.TemporaryDirectory() as storage_dir:
            with tempfile.TemporaryDirectory() as doc_dir:
                with open(os.path.join(doc_dir, "report.txt"), "w") as f:
                    f.write(
                        "Acme Corporation reported revenue of $500 million in Q3 2024. "
                        "The operating margin improved to 15.2% from 12.8%. "
                        "Total shareholder returns exceeded 20% for the fiscal year. "
                        "The board approved a quarterly dividend of $0.50 per share. " * 5
                    )

                engine = AdaptiveAI(storage_dir=storage_dir, log_level="WARNING")
                engine.ingest(doc_dir)

                response = engine.ask("What is the revenue?")

                assert isinstance(response, AdaptiveResponse)
                assert len(response.answer) > 0
                assert 0.0 <= response.confidence <= 1.0
                assert response.query_id
                assert response.evaluation is not None
                assert response.retrieval_info is not None
                assert response.policy_decision is not None

    @skip_no_chromadb
    def test_ask_builds_audit_trail(self):
        with tempfile.TemporaryDirectory() as storage_dir:
            with tempfile.TemporaryDirectory() as doc_dir:
                with open(os.path.join(doc_dir, "doc.txt"), "w") as f:
                    f.write("Test content about operations and risks. " * 10)

                engine = AdaptiveAI(storage_dir=storage_dir, log_level="WARNING")
                engine.ingest(doc_dir)
                response = engine.ask("What are the risks?")

                trail = engine.audit.get_query_trail(response.query_id)
                assert len(trail) >= 4  # At least: query, analyze, rl, retrieval

    @skip_no_chromadb
    def test_learning_over_queries(self):
        with tempfile.TemporaryDirectory() as storage_dir:
            with tempfile.TemporaryDirectory() as doc_dir:
                with open(os.path.join(doc_dir, "doc.txt"), "w") as f:
                    f.write("Revenue of $500M. Profit margin 15%. EBITDA $200M. " * 10)

                engine = AdaptiveAI(storage_dir=storage_dir, log_level="WARNING")
                engine.ingest(doc_dir)

                # Run multiple queries
                for q in ["revenue?", "profit?", "EBITDA?", "margin?", "total?"]:
                    engine.ask(q)

                assert engine._total_queries == 5
                assert engine.rl.total_queries == 5
