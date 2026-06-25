"""Long-term persistent memory — v4 addition.

Remembers across sessions:
- User query patterns and preferences
- Successful routing strategies per domain/topic
- Document relationships discovered over time
- Conversation context for follow-up queries
"""

import json
import time
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """Single memory entry."""
    key: str
    value: Any
    category: str  # "pattern", "preference", "context", "fact"
    created_at: float = field(default_factory=time.time)
    accessed_at: float = field(default_factory=time.time)
    access_count: int = 1
    relevance_score: float = 1.0

    def touch(self):
        self.accessed_at = time.time()
        self.access_count += 1


class PersistentMemory:
    """Long-term memory that persists across sessions.

    Stores:
    - Query patterns: what types of queries the user asks frequently
    - Routing preferences: which strategies worked for which topics
    - Document facts: key entities and relationships discovered
    - Session context: recent conversation for follow-up queries
    """

    def __init__(self, storage_dir: str = ".adaptive_intelligence",
                 max_entries: int = 10000):
        self._storage_dir = Path(storage_dir)
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._filepath = self._storage_dir / "long_term_memory.json"
        self._entries: Dict[str, MemoryEntry] = {}
        self._max_entries = max_entries
        self._session_context: List[Dict[str, str]] = []
        self._max_session_context = 20
        self._load()

    def remember(self, key: str, value: Any, category: str = "fact"):
        """Store a memory entry."""
        if key in self._entries:
            self._entries[key].value = value
            self._entries[key].touch()
        else:
            self._entries[key] = MemoryEntry(
                key=key, value=value, category=category,
            )

        # Evict old entries if over limit
        if len(self._entries) > self._max_entries:
            self._evict()

    def recall(self, key: str) -> Optional[Any]:
        """Recall a memory by exact key."""
        entry = self._entries.get(key)
        if entry:
            entry.touch()
            return entry.value
        return None

    def search(self, query: str, category: str = None,
               top_k: int = 5) -> List[MemoryEntry]:
        """Search memory by keyword matching."""
        query_terms = set(query.lower().split())
        scored = []

        for entry in self._entries.values():
            if category and entry.category != category:
                continue

            key_terms = set(entry.key.lower().split())
            value_str = str(entry.value).lower() if entry.value else ""
            value_terms = set(value_str.split())

            overlap = len(query_terms & (key_terms | value_terms))
            if overlap > 0:
                score = overlap / max(len(query_terms), 1)
                # Boost by recency and access frequency
                recency = 1.0 / (1.0 + (time.time() - entry.accessed_at) / 86400)
                frequency = min(entry.access_count / 10, 1.0)
                score = score * 0.6 + recency * 0.2 + frequency * 0.2
                scored.append((entry, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [entry for entry, score in scored[:top_k]]

    def add_session_context(self, query: str, answer: str, strategy: str):
        """Add to session context for follow-up queries."""
        self._session_context.append({
            "query": query,
            "answer": answer[:500],
            "strategy": strategy,
            "timestamp": time.time(),
        })
        if len(self._session_context) > self._max_session_context:
            self._session_context = self._session_context[-self._max_session_context:]

    def get_session_context(self, last_n: int = 5) -> List[Dict[str, str]]:
        """Get recent session context for follow-up queries."""
        return self._session_context[-last_n:]

    def learn_pattern(self, query_type: str, domain: str,
                      strategy: str, score: float):
        """Learn a routing pattern from experience."""
        key = f"pattern:{query_type}:{domain}"
        existing = self.recall(key)
        if existing and isinstance(existing, dict):
            # Running average
            n = existing.get("count", 1)
            existing["best_strategy"] = strategy if score > existing.get("best_score", 0) else existing.get("best_strategy", strategy)
            existing["best_score"] = max(score, existing.get("best_score", 0))
            existing["avg_score"] = (existing.get("avg_score", 0) * n + score) / (n + 1)
            existing["count"] = n + 1
            self.remember(key, existing, "pattern")
        else:
            self.remember(key, {
                "best_strategy": strategy,
                "best_score": score,
                "avg_score": score,
                "count": 1,
            }, "pattern")

    def get_best_strategy(self, query_type: str, domain: str) -> Optional[str]:
        """Get the best known strategy for a query type + domain."""
        key = f"pattern:{query_type}:{domain}"
        pattern = self.recall(key)
        if pattern and isinstance(pattern, dict):
            return pattern.get("best_strategy")
        return None

    def get_stats(self) -> Dict[str, Any]:
        categories = {}
        for entry in self._entries.values():
            categories[entry.category] = categories.get(entry.category, 0) + 1
        return {
            "total_entries": len(self._entries),
            "categories": categories,
            "session_context_length": len(self._session_context),
        }

    def _evict(self):
        """Evict least relevant entries."""
        entries = sorted(
            self._entries.items(),
            key=lambda x: x[1].relevance_score * x[1].access_count,
        )
        to_remove = len(self._entries) - int(self._max_entries * 0.8)
        for key, _ in entries[:to_remove]:
            del self._entries[key]

    def save(self):
        """Persist memory to disk."""
        state = {
            "entries": {
                k: {
                    "key": v.key, "value": v.value, "category": v.category,
                    "created_at": v.created_at, "accessed_at": v.accessed_at,
                    "access_count": v.access_count, "relevance_score": v.relevance_score,
                }
                for k, v in self._entries.items()
            },
            "session_context": self._session_context,
        }
        with open(self._filepath, "w") as f:
            json.dump(state, f)

    def _load(self):
        """Load memory from disk."""
        if not self._filepath.exists():
            return
        try:
            with open(self._filepath) as f:
                state = json.load(f)
            for k, data in state.get("entries", {}).items():
                self._entries[k] = MemoryEntry(**data)
            self._session_context = state.get("session_context", [])
            logger.info(f"Memory loaded: {len(self._entries)} entries")
        except Exception as e:
            logger.warning(f"Memory load failed: {e}")

    def clear(self):
        self._entries.clear()
        self._session_context.clear()
