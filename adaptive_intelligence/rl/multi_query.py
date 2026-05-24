"""Multi-query decomposition — v3 addition.

Breaks complex queries into sub-queries, retrieves for each,
merges context. No LLM needed — uses rule-based decomposition.
"""

import re
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class MultiQueryDecomposer:
    """Decompose complex queries into simpler sub-queries.

    Uses rule-based patterns (no LLM call needed):
    - "Compare X and Y" → ["What is X?", "What is Y?"]
    - "X vs Y" → ["What is X?", "What is Y?"]
    - "How does X affect Y?" → ["What is X?", "What is Y?", "X affect Y"]
    - "Trace X to Y to Z" → ["What is X?", "What is Y?", "What is Z?"]
    """

    # Patterns that trigger decomposition
    COMPARE_PATTERNS = [
        r"compare\s+(.+?)\s+(?:and|vs|versus|with|to)\s+(.+?)(?:\?|$)",
        r"(.+?)\s+vs\.?\s+(.+?)(?:\?|$)",
        r"difference\s+between\s+(.+?)\s+and\s+(.+?)(?:\?|$)",
        r"how\s+(?:does|did|do)\s+(.+?)\s+change\s+(?:from|between)\s+(.+?)\s+(?:to|and)\s+(.+?)(?:\?|$)",
    ]

    TRACE_PATTERNS = [
        r"trace\s*:?\s*(.+?)(?:\?|$)",
        r"chain\s*:?\s*(.+?)(?:\?|$)",
        r"(.+?)\s+to\s+(.+?)\s+to\s+(.+?)(?:\?|$)",
    ]

    MULTI_PART_PATTERNS = [
        r"(.+?),\s*(.+?),\s*(?:and\s+)?(.+?)(?:\?|$)",
        r"(.+?):\s*(.+?),\s*(.+?),\s*(.+?)(?:\?|$)",
    ]

    def should_decompose(self, query: str) -> bool:
        """Check if query would benefit from decomposition."""
        q = query.lower().strip()
        indicators = [
            "compare", " vs ", "versus", "difference between",
            "trace", "chain", " and ", "how does", "how did",
            " to ", "from ", "between",
        ]
        complexity_score = sum(1 for ind in indicators if ind in q)
        return complexity_score >= 2 or len(query.split()) > 15

    def decompose(self, query: str) -> List[str]:
        """Decompose a complex query into sub-queries.

        Returns the original query if decomposition not beneficial.
        """
        q = query.lower().strip()

        # Try compare patterns
        for pattern in self.COMPARE_PATTERNS:
            match = re.search(pattern, q, re.IGNORECASE)
            if match:
                parts = [g.strip() for g in match.groups() if g.strip()]
                sub_queries = [f"What is {p}?" for p in parts]
                sub_queries.append(query)  # Include original
                return self._deduplicate(sub_queries)

        # Try trace patterns (A → B → C)
        for pattern in self.TRACE_PATTERNS:
            match = re.search(pattern, q, re.IGNORECASE)
            if match:
                groups = [g.strip() for g in match.groups() if g.strip()]
                if len(groups) == 1:
                    # Split on " to " or " → "
                    chain = re.split(r'\s+to\s+|→|->|,\s*', groups[0])
                    chain = [c.strip() for c in chain if c.strip()]
                else:
                    chain = groups
                sub_queries = [f"What is {c}?" for c in chain]
                sub_queries.append(query)
                return self._deduplicate(sub_queries)

        # Try multi-part patterns
        for pattern in self.MULTI_PART_PATTERNS:
            match = re.search(pattern, q, re.IGNORECASE)
            if match:
                parts = [g.strip() for g in match.groups() if g.strip()]
                sub_queries = [p if '?' in p else f"What about {p}?" for p in parts]
                sub_queries.append(query)
                return self._deduplicate(sub_queries)

        # Default: split on "and" if query has multiple parts
        if " and " in q and len(q.split()) > 10:
            parts = q.split(" and ")
            sub_queries = [p.strip() + "?" if not p.strip().endswith("?") else p.strip()
                          for p in parts if len(p.strip().split()) > 2]
            if len(sub_queries) > 1:
                sub_queries.append(query)
                return self._deduplicate(sub_queries)

        return [query]

    def _deduplicate(self, queries: List[str]) -> List[str]:
        """Remove duplicate or near-duplicate queries."""
        seen = set()
        result = []
        for q in queries:
            key = q.lower().strip().rstrip("?")
            if key not in seen and len(key) > 3:
                seen.add(key)
                result.append(q)
        return result[:5]  # Cap at 5 sub-queries
