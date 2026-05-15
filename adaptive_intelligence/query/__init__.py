"""Query understanding and classification (Trigger Interpreter)."""

import re
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class QueryType(str, Enum):
    FACTUAL = "factual"
    RELATIONAL = "relational"
    ANALYTICAL = "analytical"
    SUMMARIZATION = "summarization"
    EXTRACTION = "extraction"
    COMPARATIVE = "comparative"
    REASONING = "reasoning"
    STRUCTURED_LOOKUP = "structured_lookup"


class QueryComplexity(str, Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    MULTI_HOP = "multi_hop"


class QueryIntent(str, Enum):
    EXTRACT = "extract"
    EXPLAIN = "explain"
    SUMMARIZE = "summarize"
    COMPARE = "compare"
    ANALYZE_RISK = "analyze_risk"
    DISCOVER_RELATIONSHIPS = "discover_relationships"
    VERIFY = "verify"


@dataclass
class QueryAnalysis:
    """Complete analysis of a user query."""
    original_query: str
    query_type: QueryType = QueryType.FACTUAL
    complexity: QueryComplexity = QueryComplexity.SIMPLE
    domain: str = "general"
    entities: List[str] = field(default_factory=list)
    intent: QueryIntent = QueryIntent.EXTRACT
    has_relationship_indicators: bool = False
    has_temporal_markers: bool = False
    temporal_markers: List[str] = field(default_factory=list)
    confidence: float = 0.0
    suggested_indexes: List[str] = field(default_factory=list)
    requires_graph: bool = False
    requires_table: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_type": self.query_type.value,
            "complexity": self.complexity.value,
            "domain": self.domain,
            "entities": self.entities,
            "intent": self.intent.value,
            "has_relationship_indicators": self.has_relationship_indicators,
            "has_temporal_markers": self.has_temporal_markers,
            "temporal_markers": self.temporal_markers,
            "confidence": self.confidence,
            "suggested_indexes": self.suggested_indexes,
            "requires_graph": self.requires_graph,
            "requires_table": self.requires_table,
        }


class TriggerInterpreter:
    """Analyzes queries to determine optimal retrieval strategy."""

    # Relationship indicator words
    RELATIONSHIP_WORDS = {
        "connected", "related", "owns", "affects", "depends",
        "reports to", "linked", "associated", "caused by", "leads to",
        "influences", "impacts", "parent", "subsidiary", "partner",
        "supplies", "between", "relationship", "chain", "network",
    }

    # Temporal markers
    TEMPORAL_PATTERNS = [
        r"\bQ[1-4]\b",
        r"\b20\d{2}\b",
        r"\blast\s+(year|quarter|month|week)\b",
        r"\brecent\b",
        r"\bthis\s+(year|quarter|month)\b",
        r"\byear[\s-]over[\s-]year\b",
        r"\bYTD\b",
        r"\bMTD\b",
        r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\b",
    ]

    # Structured data indicators
    STRUCTURED_INDICATORS = {
        "revenue", "profit", "margin", "cost", "price", "total",
        "average", "sum", "count", "percentage", "ratio", "growth",
        "rate", "volume", "quantity", "amount", "budget", "forecast",
        "how much", "how many",
    }

    # Domain indicators
    DOMAIN_INDICATORS = {
        "financial": {"revenue", "profit", "earnings", "financial", "fiscal", "budget",
                      "GAAP", "EBITDA", "ROI", "margin", "dividend", "stock", "equity"},
        "legal": {"regulation", "compliance", "contract", "clause", "liability",
                  "legal", "law", "statute", "violation", "regulatory", "terms"},
        "healthcare": {"patient", "clinical", "diagnosis", "treatment", "medical",
                       "healthcare", "hospital", "drug", "therapy", "HIPAA"},
        "technical": {"system", "architecture", "API", "database", "code",
                      "deployment", "server", "infrastructure", "technical"},
        "operational": {"operations", "supply chain", "logistics", "process",
                        "workflow", "efficiency", "capacity", "inventory"},
    }

    # Comparative indicators
    COMPARATIVE_WORDS = {"compare", "vs", "versus", "difference", "between",
                         "better", "worse", "more", "less", "higher", "lower"}

    def __init__(self, llm_provider=None):
        self._llm = llm_provider
        self._compiled_temporal = [re.compile(p, re.IGNORECASE) for p in self.TEMPORAL_PATTERNS]

    def analyze(self, query: str) -> QueryAnalysis:
        """Analyze a query and return classification."""
        analysis = QueryAnalysis(original_query=query)

        # Run all analyzers
        query_lower = query.lower()

        self._detect_entities(query, analysis)
        self._detect_relationships(query_lower, analysis)
        self._detect_temporal(query, analysis)
        self._detect_domain(query_lower, analysis)
        self._detect_query_type(query_lower, analysis)
        self._detect_complexity(analysis)
        self._detect_intent(query_lower, analysis)
        self._suggest_indexes(analysis)
        self._calculate_confidence(analysis)

        logger.debug(f"Query analysis: {analysis.to_dict()}")
        return analysis

    def _detect_entities(self, query: str, analysis: QueryAnalysis):
        """Extract entities from query using pattern matching."""
        # Capitalized words (potential proper nouns)
        caps = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", query)
        # Acronyms
        acronyms = re.findall(r"\b[A-Z]{2,}\b", query)
        # Numbers and metrics
        numbers = re.findall(r"\b\d+(?:\.\d+)?[%KMBkm]?\b", query)

        entities = list(set(caps + acronyms + numbers))
        # Filter common words
        common = {"What", "How", "Why", "When", "Where", "Which", "Who",
                  "Are", "Is", "The", "Can", "Does", "Did", "Do", "Has", "Have"}
        analysis.entities = [e for e in entities if e not in common]

    def _detect_relationships(self, query_lower: str, analysis: QueryAnalysis):
        """Detect relationship indicators."""
        for word in self.RELATIONSHIP_WORDS:
            if word in query_lower:
                analysis.has_relationship_indicators = True
                analysis.requires_graph = True
                return

    def _detect_temporal(self, query: str, analysis: QueryAnalysis):
        """Detect temporal markers."""
        for pattern in self._compiled_temporal:
            matches = pattern.findall(query)
            if matches:
                analysis.has_temporal_markers = True
                analysis.temporal_markers.extend(matches)

    def _detect_domain(self, query_lower: str, analysis: QueryAnalysis):
        """Detect the domain of the query."""
        domain_scores = {}
        for domain, keywords in self.DOMAIN_INDICATORS.items():
            score = sum(1 for kw in keywords if kw.lower() in query_lower)
            if score > 0:
                domain_scores[domain] = score

        if domain_scores:
            analysis.domain = max(domain_scores, key=domain_scores.get)
        else:
            analysis.domain = "general"

    def _detect_query_type(self, query_lower: str, analysis: QueryAnalysis):
        """Classify the query type."""
        # Order matters: check more specific patterns first
        if any(w in query_lower for w in {"why", "how does", "explain", "reason", "cause"}):
            analysis.query_type = QueryType.REASONING
        elif any(w in query_lower for w in self.COMPARATIVE_WORDS - self.RELATIONSHIP_WORDS):
            analysis.query_type = QueryType.COMPARATIVE
        elif analysis.has_relationship_indicators:
            analysis.query_type = QueryType.RELATIONAL
        elif any(w in query_lower for w in {"summarize", "summary", "overview", "brief"}):
            analysis.query_type = QueryType.SUMMARIZATION
        elif any(w in query_lower for w in {"analyze", "analysis", "assess", "evaluate", "risk"}):
            analysis.query_type = QueryType.ANALYTICAL
        elif any(w in query_lower for w in self.STRUCTURED_INDICATORS):
            analysis.query_type = QueryType.STRUCTURED_LOOKUP
        elif any(w in query_lower for w in {"extract", "find", "list", "what are", "who"}):
            analysis.query_type = QueryType.EXTRACTION
        else:
            analysis.query_type = QueryType.FACTUAL

    def _detect_complexity(self, analysis: QueryAnalysis):
        """Estimate query complexity."""
        score = 0
        if len(analysis.entities) > 3:
            score += 2
        elif len(analysis.entities) > 1:
            score += 1

        if analysis.has_relationship_indicators:
            score += 2
        if analysis.has_temporal_markers:
            score += 1
        if analysis.query_type in (QueryType.COMPARATIVE, QueryType.ANALYTICAL, QueryType.REASONING):
            score += 1

        word_count = len(analysis.original_query.split())
        if word_count > 20:
            score += 1

        if score >= 4:
            analysis.complexity = QueryComplexity.MULTI_HOP
        elif score >= 3:
            analysis.complexity = QueryComplexity.COMPLEX
        elif score >= 1:
            analysis.complexity = QueryComplexity.MODERATE
        else:
            analysis.complexity = QueryComplexity.SIMPLE

    def _detect_intent(self, query_lower: str, analysis: QueryAnalysis):
        """Determine user intent."""
        if any(w in query_lower for w in {"compare", "vs", "versus", "difference"}):
            analysis.intent = QueryIntent.COMPARE
        elif any(w in query_lower for w in {"risk", "threat", "vulnerability", "danger"}):
            analysis.intent = QueryIntent.ANALYZE_RISK
        elif any(w in query_lower for w in {"summarize", "summary", "overview"}):
            analysis.intent = QueryIntent.SUMMARIZE
        elif any(w in query_lower for w in {"explain", "why", "how does"}):
            analysis.intent = QueryIntent.EXPLAIN
        elif any(w in query_lower for w in {"verify", "confirm", "check", "validate"}):
            analysis.intent = QueryIntent.VERIFY
        elif analysis.has_relationship_indicators:
            analysis.intent = QueryIntent.DISCOVER_RELATIONSHIPS
        else:
            analysis.intent = QueryIntent.EXTRACT

    def _suggest_indexes(self, analysis: QueryAnalysis):
        """Suggest which indexes to use based on analysis."""
        indexes = []

        if analysis.query_type == QueryType.STRUCTURED_LOOKUP:
            indexes.append("table")
            indexes.append("keyword")
            analysis.requires_table = True
        elif analysis.query_type == QueryType.RELATIONAL:
            indexes.append("graph")
            indexes.append("vector")
            analysis.requires_graph = True
        elif analysis.query_type in (QueryType.FACTUAL, QueryType.EXTRACTION):
            indexes.append("keyword")
            indexes.append("vector")
        elif analysis.query_type == QueryType.SUMMARIZATION:
            indexes.append("vector")
        else:
            indexes.append("vector")
            indexes.append("keyword")

        analysis.suggested_indexes = indexes

    def _calculate_confidence(self, analysis: QueryAnalysis):
        """Calculate confidence in the analysis."""
        score = 0.5  # Base confidence

        if analysis.domain != "general":
            score += 0.15
        if analysis.entities:
            score += 0.1
        if analysis.has_temporal_markers:
            score += 0.1
        if analysis.has_relationship_indicators:
            score += 0.1
        if len(analysis.original_query.split()) >= 5:
            score += 0.05

        analysis.confidence = min(1.0, score)
