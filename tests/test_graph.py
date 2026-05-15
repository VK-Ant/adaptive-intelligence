"""Tests for the Knowledge Graph module."""

import pytest
from adaptive_intelligence.graph import KnowledgeGraph, GraphNode, GraphEdge
from adaptive_intelligence.core.config import GraphConfig
from adaptive_intelligence.query import TriggerInterpreter
from adaptive_intelligence.ingestion.chunker import Chunk


@pytest.fixture
def graph():
    return KnowledgeGraph()


@pytest.fixture
def populated_graph():
    g = KnowledgeGraph()
    g.add_edge("Acme Corp", "Beta Ltd", "owns", source_type="organization", target_type="organization")
    g.add_edge("Acme Corp", "Gamma Inc", "supplies", source_type="organization", target_type="organization")
    g.add_edge("Beta Ltd", "Delta Co", "depends_on", source_type="organization", target_type="organization")
    g.add_edge("John Smith", "Acme Corp", "reports_to", source_type="person", target_type="organization")
    return g


class TestGraphConstruction:
    def test_add_node(self, graph):
        node = graph.add_node("Acme Corp", "organization")
        assert isinstance(node, GraphNode)
        assert node.label == "Acme Corp"
        assert graph.node_count == 1

    def test_add_edge(self, graph):
        edge = graph.add_edge("Acme", "Beta", "owns")
        assert isinstance(edge, GraphEdge)
        assert graph.node_count == 2
        assert graph.edge_count == 1

    def test_duplicate_node_merges(self, graph):
        graph.add_node("Acme Corp", "organization", source_chunk="c1")
        graph.add_node("Acme Corp", "organization", source_chunk="c2")
        assert graph.node_count == 1
        # Should have both source chunks
        node = list(graph._nodes.values())[0]
        assert len(node.source_chunks) == 2


class TestConditionalActivation:
    def test_activates_for_relational_query(self, populated_graph):
        interpreter = TriggerInterpreter()
        analysis = interpreter.analyze("How is Acme Corp connected to Beta Ltd?")
        assert populated_graph.should_activate(analysis)

    def test_no_activation_for_simple_query(self, populated_graph):
        interpreter = TriggerInterpreter()
        analysis = interpreter.analyze("What is the date?")
        assert not populated_graph.should_activate(analysis)

    def test_no_activation_when_empty(self):
        graph = KnowledgeGraph()
        interpreter = TriggerInterpreter()
        analysis = interpreter.analyze("How is A connected to B?")
        assert not graph.should_activate(analysis)

    def test_no_activation_when_disabled(self, populated_graph):
        populated_graph.config.enabled = False
        interpreter = TriggerInterpreter()
        analysis = interpreter.analyze("How is Acme connected to Beta?")
        assert not populated_graph.should_activate(analysis)


class TestGraphTraversal:
    def test_basic_traversal(self, populated_graph):
        result = populated_graph.traverse(["Acme Corp"], max_hops=2)
        assert len(result.nodes_visited) > 0
        assert len(result.paths) > 0

    def test_traversal_depth(self, populated_graph):
        result_1 = populated_graph.traverse(["Acme Corp"], max_hops=1)
        result_2 = populated_graph.traverse(["Acme Corp"], max_hops=3)
        assert len(result_2.nodes_visited) >= len(result_1.nodes_visited)

    def test_traversal_unknown_entity(self, populated_graph):
        result = populated_graph.traverse(["NonExistent Corp"], max_hops=2)
        assert len(result.paths) == 0
        assert result.confidence == 0.0

    def test_context_generation(self, populated_graph):
        result = populated_graph.traverse(["Acme Corp"], max_hops=2)
        context = result.to_context()
        assert isinstance(context, str)
        if result.paths:
            assert "Graph Relationships Found" in context


class TestGraphBuilding:
    def test_build_from_chunks(self, graph):
        chunks = [
            Chunk(chunk_id="c1", doc_id="d1",
                  content="Acme Corporation owns Beta Ltd. John Smith reports to the CEO.",
                  source_document="test.pdf"),
            Chunk(chunk_id="c2", doc_id="d1",
                  content="Beta Ltd depends on Gamma Inc for supply chain operations.",
                  source_document="test.pdf"),
        ]
        graph.build_from_chunks(chunks)
        assert graph.node_count > 0
        assert graph.edge_count > 0


class TestActivationTracking:
    def test_record_outcome(self, populated_graph):
        populated_graph.record_activation_outcome(True)
        populated_graph.record_activation_outcome(True)
        populated_graph.record_activation_outcome(False)
        assert populated_graph.activation_success_rate == pytest.approx(2 / 3, abs=0.01)

    def test_initial_success_rate(self, graph):
        assert graph.activation_success_rate == 0.5  # Prior


class TestGraphStats:
    def test_stats(self, populated_graph):
        stats = populated_graph.get_stats()
        assert stats["nodes"] > 0
        assert stats["edges"] > 0
        assert "activation_success_rate" in stats


class TestGraphClear:
    def test_clear(self, populated_graph):
        populated_graph.clear()
        assert populated_graph.node_count == 0
        assert populated_graph.edge_count == 0
