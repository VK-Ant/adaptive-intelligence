"""Graph Intelligence Layer — Conditional Entity Graph with Traversal.

Key innovation: Graph is activated ONLY when the query requires
relational reasoning. The RL engine learns the activation threshold.
"""

import logging
import hashlib
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from adaptive_intelligence.core.config import GraphConfig
from adaptive_intelligence.ingestion.chunker import Chunk
from adaptive_intelligence.query import QueryAnalysis

logger = logging.getLogger(__name__)


@dataclass
class GraphNode:
    """A node in the knowledge graph."""
    node_id: str
    label: str
    entity_type: str  # "organization", "person", "location", "concept", "metric", "document"
    properties: Dict[str, Any] = field(default_factory=dict)
    source_chunks: List[str] = field(default_factory=list)  # chunk_ids

    def __hash__(self):
        return hash(self.node_id)

    def __eq__(self, other):
        return isinstance(other, GraphNode) and self.node_id == other.node_id


@dataclass
class GraphEdge:
    """An edge in the knowledge graph."""
    source_id: str
    target_id: str
    relation: str  # "owns", "affects", "depends_on", "references", etc.
    weight: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)
    source_chunk: str = ""

    @property
    def edge_id(self) -> str:
        return f"{self.source_id}->{self.relation}->{self.target_id}"


@dataclass
class GraphTraversalResult:
    """Result from graph traversal."""
    paths: List[List[Tuple[str, str, str]]]  # [(node, relation, node), ...]
    nodes_visited: List[GraphNode] = field(default_factory=list)
    edges_traversed: List[GraphEdge] = field(default_factory=list)
    related_chunks: List[str] = field(default_factory=list)
    hops: int = 0
    confidence: float = 0.0

    def to_context(self) -> str:
        """Convert traversal result to text context for LLM."""
        if not self.paths:
            return ""

        lines = ["[Graph Relationships Found]"]
        for path in self.paths:
            path_str = " → ".join(
                f"{src} --{rel}--> {tgt}" for src, rel, tgt in path
            )
            lines.append(f"  {path_str}")
        return "\n".join(lines)


class KnowledgeGraph:
    """Entity-relationship knowledge graph with conditional activation.

    The graph is built during ingestion by extracting entities and
    relationships from document chunks. During retrieval, the graph
    is activated only when the query requires relational reasoning.
    """

    def __init__(self, config: Optional[GraphConfig] = None):
        self.config = config or GraphConfig()
        self._nodes: Dict[str, GraphNode] = {}
        self._edges: List[GraphEdge] = []
        self._adjacency: Dict[str, List[GraphEdge]] = defaultdict(list)
        self._reverse_adjacency: Dict[str, List[GraphEdge]] = defaultdict(list)

        # Activation statistics (learned by RL)
        self._activation_count = 0
        self._activation_success_count = 0

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    @property
    def activation_success_rate(self) -> float:
        if self._activation_count == 0:
            return 0.5  # Prior
        return self._activation_success_count / self._activation_count

    def add_node(self, label: str, entity_type: str,
                 properties: Optional[Dict] = None,
                 source_chunk: str = "") -> GraphNode:
        """Add a node to the graph."""
        node_id = hashlib.md5(f"{label}:{entity_type}".encode()).hexdigest()[:10]

        if node_id in self._nodes:
            # Update existing node
            if source_chunk:
                self._nodes[node_id].source_chunks.append(source_chunk)
            return self._nodes[node_id]

        node = GraphNode(
            node_id=node_id,
            label=label,
            entity_type=entity_type,
            properties=properties or {},
            source_chunks=[source_chunk] if source_chunk else [],
        )
        self._nodes[node_id] = node
        return node

    def add_edge(self, source_label: str, target_label: str,
                 relation: str, source_type: str = "concept",
                 target_type: str = "concept",
                 weight: float = 1.0, source_chunk: str = "") -> GraphEdge:
        """Add an edge between two nodes (creates nodes if needed)."""
        source_node = self.add_node(source_label, source_type, source_chunk=source_chunk)
        target_node = self.add_node(target_label, target_type, source_chunk=source_chunk)

        edge = GraphEdge(
            source_id=source_node.node_id,
            target_id=target_node.node_id,
            relation=relation,
            weight=weight,
            source_chunk=source_chunk,
        )
        self._edges.append(edge)
        self._adjacency[source_node.node_id].append(edge)
        self._reverse_adjacency[target_node.node_id].append(edge)
        return edge

    def should_activate(self, query_analysis: QueryAnalysis) -> bool:
        """Determine if graph should be activated for this query.

        This is the CONDITIONAL ACTIVATION gate — a key differentiator.
        """
        if not self.config.enabled:
            return False

        if not self.config.conditional_activation:
            return True  # Always active if conditional is disabled

        if self.node_count == 0:
            return False

        # Rule-based activation signals
        signals = 0

        # Signal 1: Query has relationship indicators
        if query_analysis.has_relationship_indicators:
            signals += 2

        # Signal 2: Query requires graph (detected by trigger interpreter)
        if query_analysis.requires_graph:
            signals += 2

        # Signal 3: High entity count
        if len(query_analysis.entities) >= self.config.min_entity_count_for_activation:
            signals += 1

        # Signal 4: Complex or multi-hop query
        from adaptive_intelligence.query import QueryComplexity
        if query_analysis.complexity in (QueryComplexity.COMPLEX, QueryComplexity.MULTI_HOP):
            signals += 1

        # Signal 5: Historical success rate (learned from past)
        if self.activation_success_rate > self.config.activation_success_threshold:
            signals += 1

        # Activate if enough signals
        should_activate = signals >= 2

        logger.info(
            f"Graph activation decision: {should_activate} "
            f"(signals={signals}, success_rate={self.activation_success_rate:.2f})"
        )
        return should_activate

    def record_activation_outcome(self, was_helpful: bool):
        """Record whether graph activation improved the result."""
        self._activation_count += 1
        if was_helpful:
            self._activation_success_count += 1

    def traverse(self, seed_entities: List[str], max_hops: int = 2) -> GraphTraversalResult:
        """Traverse the graph from seed entities.

        Args:
            seed_entities: Entity labels to start from
            max_hops: Maximum traversal depth
        """
        max_hops = min(max_hops, self.config.max_hops)

        # Find seed nodes
        seed_nodes = []
        for entity in seed_entities:
            entity_lower = entity.lower()
            for node in self._nodes.values():
                if entity_lower in node.label.lower() or node.label.lower() in entity_lower:
                    seed_nodes.append(node)

        if not seed_nodes:
            return GraphTraversalResult(paths=[], confidence=0.0)

        # BFS traversal
        visited: Set[str] = set()
        paths: List[List[Tuple[str, str, str]]] = []
        all_nodes: List[GraphNode] = []
        all_edges: List[GraphEdge] = []
        related_chunks: Set[str] = set()

        queue: List[Tuple[GraphNode, int, List[Tuple[str, str, str]]]] = [
            (node, 0, []) for node in seed_nodes
        ]

        while queue:
            current_node, depth, current_path = queue.pop(0)

            if current_node.node_id in visited:
                continue
            visited.add(current_node.node_id)
            all_nodes.append(current_node)
            related_chunks.update(current_node.source_chunks)

            if depth >= max_hops:
                if current_path:
                    paths.append(current_path)
                continue

            # Follow outgoing edges
            for edge in self._adjacency.get(current_node.node_id, []):
                target = self._nodes.get(edge.target_id)
                if target and target.node_id not in visited:
                    new_path = current_path + [
                        (current_node.label, edge.relation, target.label)
                    ]
                    all_edges.append(edge)
                    related_chunks.add(edge.source_chunk)
                    queue.append((target, depth + 1, new_path))

            # Follow incoming edges (bidirectional traversal)
            for edge in self._reverse_adjacency.get(current_node.node_id, []):
                source = self._nodes.get(edge.source_id)
                if source and source.node_id not in visited:
                    new_path = current_path + [
                        (source.label, edge.relation, current_node.label)
                    ]
                    all_edges.append(edge)
                    queue.append((source, depth + 1, new_path))

        # Calculate confidence based on traversal quality
        confidence = min(1.0, len(paths) * 0.2 + len(all_nodes) * 0.05)

        return GraphTraversalResult(
            paths=paths,
            nodes_visited=all_nodes,
            edges_traversed=all_edges,
            related_chunks=list(related_chunks),
            hops=max_hops,
            confidence=confidence,
        )

    def build_from_chunks(self, chunks: List[Chunk]):
        """Build graph from document chunks using entity co-occurrence."""
        import re

        for chunk in chunks:
            # Extract capitalized entities from chunk
            entities = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", chunk.content)
            acronyms = re.findall(r"\b[A-Z]{2,}\b", chunk.content)

            # Filter common words
            common = {"The", "This", "That", "These", "Those", "What", "Where",
                       "When", "How", "Why", "Which", "Who", "And", "But", "For",
                       "Not", "All", "Can", "Had", "Her", "Was", "One", "Our",
                       "Out", "Are", "Has", "His", "May", "New", "Now", "Old", "See"}
            entities = [e for e in entities if e not in common and len(e) > 2]
            all_entities = entities + [a for a in acronyms if len(a) > 1]

            # Create nodes
            for entity in all_entities:
                self.add_node(entity, "concept", source_chunk=chunk.chunk_id)

            # Create co-occurrence edges
            for i, e1 in enumerate(all_entities):
                for e2 in all_entities[i + 1:]:
                    if e1 != e2:
                        self.add_edge(
                            e1, e2, "co_occurs_with",
                            source_chunk=chunk.chunk_id,
                        )

            # Detect relationship patterns
            relationship_patterns = [
                (r"(\w+)\s+owns\s+(\w+)", "owns"),
                (r"(\w+)\s+reports?\s+to\s+(\w+)", "reports_to"),
                (r"(\w+)\s+depends?\s+on\s+(\w+)", "depends_on"),
                (r"(\w+)\s+affects?\s+(\w+)", "affects"),
                (r"(\w+)\s+supplies?\s+(\w+)", "supplies"),
                (r"(\w+)\s+connected\s+to\s+(\w+)", "connected_to"),
            ]
            for pattern, relation in relationship_patterns:
                matches = re.findall(pattern, chunk.content, re.IGNORECASE)
                for src, tgt in matches:
                    if len(src) > 2 and len(tgt) > 2:
                        self.add_edge(src, tgt, relation, source_chunk=chunk.chunk_id)

        logger.info(f"Graph built: {self.node_count} nodes, {self.edge_count} edges")

    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        return {
            "nodes": self.node_count,
            "edges": self.edge_count,
            "activation_count": self._activation_count,
            "activation_success_rate": self.activation_success_rate,
            "entity_types": dict(self._count_entity_types()),
        }

    def _count_entity_types(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for node in self._nodes.values():
            counts[node.entity_type] = counts.get(node.entity_type, 0) + 1
        return counts

    def clear(self):
        """Clear the graph."""
        self._nodes.clear()
        self._edges.clear()
        self._adjacency.clear()
        self._reverse_adjacency.clear()
