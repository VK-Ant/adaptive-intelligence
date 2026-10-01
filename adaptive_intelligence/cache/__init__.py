"""LM Cache — Exact + Semantic response caching with external adapters.

Saves LLM calls by caching responses for identical or similar queries.
Two built-in strategies:
    - ExactCache: dict lookup on normalized query string
    - SemanticCache: embedding similarity (cosine) with configurable threshold

External adapters let you plug in Redis, GPTCache, or any custom backend.

Usage:
    # Built-in (zero dependencies)
    engine = AdaptiveAI(cache=True)

    # With semantic similarity
    engine = AdaptiveAI(cache=True, cache_mode="semantic", cache_threshold=0.92)

    # External Redis adapter
    from adaptive_intelligence.cache import RedisAdapter
    engine = AdaptiveAI(cache=True, cache_adapter=RedisAdapter(host="localhost"))

    # Custom adapter
    class MyCache(CacheAdapter):
        def get(self, key): ...
        def set(self, key, value, ttl): ...
    engine = AdaptiveAI(cache=True, cache_adapter=MyCache())
"""

import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ─── Cache Entry ──────────────────────────────────────────


@dataclass
class CacheEntry:
    """A single cached response."""
    query: str
    query_hash: str
    answer: str
    confidence: float
    strategy: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    ttl_seconds: float = 3600.0  # 1 hour default
    hit_count: int = 0
    embedding: Optional[List[float]] = None

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.created_at) > self.ttl_seconds

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "query": self.query,
            "query_hash": self.query_hash,
            "answer": self.answer,
            "confidence": self.confidence,
            "strategy": self.strategy,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "ttl_seconds": self.ttl_seconds,
            "hit_count": self.hit_count,
        }
        if self.embedding:
            d["embedding"] = self.embedding
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CacheEntry":
        return cls(
            query=d["query"],
            query_hash=d["query_hash"],
            answer=d["answer"],
            confidence=d.get("confidence", 0.0),
            strategy=d.get("strategy", "unknown"),
            metadata=d.get("metadata", {}),
            created_at=d.get("created_at", time.time()),
            ttl_seconds=d.get("ttl_seconds", 3600.0),
            hit_count=d.get("hit_count", 0),
            embedding=d.get("embedding"),
        )


@dataclass
class CacheStats:
    """Cache performance statistics."""
    total_queries: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    exact_hits: int = 0
    semantic_hits: int = 0
    avg_similarity: float = 0.0
    entries_count: int = 0
    evictions: int = 0

    @property
    def hit_rate(self) -> float:
        if self.total_queries == 0:
            return 0.0
        return self.cache_hits / self.total_queries

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_queries": self.total_queries,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "exact_hits": self.exact_hits,
            "semantic_hits": self.semantic_hits,
            "hit_rate": f"{self.hit_rate:.1%}",
            "avg_similarity": round(self.avg_similarity, 3),
            "entries_count": self.entries_count,
            "evictions": self.evictions,
        }


# ─── Cache Adapter (for external backends) ───────────────


class CacheAdapter(ABC):
    """Base class for external cache backends (Redis, GPTCache, etc.).

    Implement get/set/delete to plug in any backend.

    Example:
        class RedisAdapter(CacheAdapter):
            def __init__(self, host="localhost", port=6379, db=0, prefix="ai_cache:"):
                import redis
                self.client = redis.Redis(host=host, port=port, db=db)
                self.prefix = prefix

            def get(self, key):
                data = self.client.get(self.prefix + key)
                return json.loads(data) if data else None

            def set(self, key, value, ttl=3600):
                self.client.setex(self.prefix + key, ttl, json.dumps(value))

            def delete(self, key):
                self.client.delete(self.prefix + key)
    """

    @abstractmethod
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached entry by key. Return None if not found."""

    @abstractmethod
    def set(self, key: str, value: Dict[str, Any], ttl: int = 3600) -> None:
        """Store entry with time-to-live in seconds."""

    @abstractmethod
    def delete(self, key: str) -> None:
        """Remove entry by key."""

    def clear(self) -> None:
        """Clear all entries. Override if your backend supports bulk delete."""
        pass

    def keys(self) -> List[str]:
        """List all keys. Override if your backend supports key listing."""
        return []


class RedisAdapter(CacheAdapter):
    """Redis cache adapter. Requires: pip install redis"""

    def __init__(self, host: str = "localhost", port: int = 6379,
                 db: int = 0, prefix: str = "adaptive_cache:"):
        self.prefix = prefix
        try:
            import redis
            self.client = redis.Redis(host=host, port=port, db=db,
                                      decode_responses=True)
            self.client.ping()
            logger.info(f"Redis cache connected: {host}:{port}")
        except ImportError:
            raise ImportError(
                "Redis adapter requires 'redis' package. "
                "Install: pip install redis"
            )
        except Exception as e:
            raise ConnectionError(f"Redis connection failed: {e}")

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        data = self.client.get(self.prefix + key)
        if data:
            return json.loads(data)
        return None

    def set(self, key: str, value: Dict[str, Any], ttl: int = 3600) -> None:
        self.client.setex(self.prefix + key, ttl, json.dumps(value))

    def delete(self, key: str) -> None:
        self.client.delete(self.prefix + key)

    def clear(self) -> None:
        keys = self.client.keys(self.prefix + "*")
        if keys:
            self.client.delete(*keys)

    def keys(self) -> List[str]:
        return [k.replace(self.prefix, "")
                for k in self.client.keys(self.prefix + "*")]


# ─── Exact Cache ──────────────────────────────────────────


def _normalize_query(query: str) -> str:
    """Normalize query for exact matching."""
    return " ".join(query.lower().strip().split())


def _hash_query(query: str) -> str:
    """Create hash of normalized query."""
    normalized = _normalize_query(query)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


class ExactCache:
    """Dictionary-based exact-match cache.

    Identical (normalized) queries return cached response instantly.
    Zero dependencies. Works everywhere.
    """

    def __init__(self, max_entries: int = 1000, default_ttl: float = 3600.0,
                 adapter: Optional[CacheAdapter] = None):
        self._store: Dict[str, CacheEntry] = {}
        self._max_entries = max_entries
        self._default_ttl = default_ttl
        self._adapter = adapter

    def get(self, query: str) -> Optional[CacheEntry]:
        """Look up exact match for query."""
        key = _hash_query(query)

        # Try external adapter first
        if self._adapter:
            data = self._adapter.get(key)
            if data:
                entry = CacheEntry.from_dict(data)
                if not entry.is_expired:
                    entry.hit_count += 1
                    self._adapter.set(key, entry.to_dict(),
                                      int(entry.ttl_seconds))
                    return entry
                else:
                    self._adapter.delete(key)
                    return None

        # In-memory lookup
        entry = self._store.get(key)
        if entry:
            if entry.is_expired:
                del self._store[key]
                return None
            entry.hit_count += 1
            return entry
        return None

    def put(self, query: str, answer: str, confidence: float,
            strategy: str, metadata: Optional[Dict] = None,
            ttl: Optional[float] = None) -> CacheEntry:
        """Store a response in cache."""
        key = _hash_query(query)
        entry = CacheEntry(
            query=_normalize_query(query),
            query_hash=key,
            answer=answer,
            confidence=confidence,
            strategy=strategy,
            metadata=metadata or {},
            ttl_seconds=ttl or self._default_ttl,
        )

        if self._adapter:
            self._adapter.set(key, entry.to_dict(), int(entry.ttl_seconds))
        else:
            # Evict oldest if at capacity
            if len(self._store) >= self._max_entries:
                self._evict_oldest()
            self._store[key] = entry

        return entry

    def invalidate(self, query: str) -> bool:
        """Remove a specific cached entry."""
        key = _hash_query(query)
        if self._adapter:
            self._adapter.delete(key)
            return True
        if key in self._store:
            del self._store[key]
            return True
        return False

    def clear(self) -> None:
        """Remove all cached entries."""
        if self._adapter:
            self._adapter.clear()
        self._store.clear()

    def _evict_oldest(self) -> int:
        """Remove expired entries first, then oldest by creation time."""
        evicted = 0
        # Remove expired
        expired = [k for k, v in self._store.items() if v.is_expired]
        for k in expired:
            del self._store[k]
            evicted += 1

        # Still at capacity? Remove least recently created
        if len(self._store) >= self._max_entries:
            oldest_key = min(self._store, key=lambda k: self._store[k].created_at)
            del self._store[oldest_key]
            evicted += 1

        return evicted

    @property
    def size(self) -> int:
        if self._adapter:
            return len(self._adapter.keys())
        return len(self._store)


# ─── Semantic Cache ───────────────────────────────────────


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _simple_embedding(text: str, dim: int = 64) -> List[float]:
    """Simple character-level embedding (no dependencies).

    For production, use sentence-transformers or OpenAI embeddings.
    This is a lightweight fallback that captures basic text similarity.
    """
    normalized = _normalize_query(text)
    vec = [0.0] * dim

    # Character trigram hashing into fixed-dim vector
    for i in range(len(normalized) - 2):
        trigram = normalized[i:i+3]
        h = int(hashlib.md5(trigram.encode()).hexdigest(), 16)
        idx = h % dim
        vec[idx] += 1.0

    # Word-level features in second half
    words = normalized.split()
    for word in words:
        h = int(hashlib.md5(word.encode()).hexdigest(), 16)
        idx = (h % (dim // 2)) + (dim // 2)
        vec[idx] += 1.0

    # L2 normalize
    norm = sum(x * x for x in vec) ** 0.5
    if norm > 0:
        vec = [x / norm for x in vec]

    return vec


class SemanticCache:
    """Embedding-similarity cache for near-duplicate queries.

    Finds cached responses for queries that are semantically similar
    (not just identical). Uses cosine similarity with configurable threshold.

    Built-in: zero-dependency character trigram embeddings.
    Better: plug in sentence-transformers or OpenAI embeddings via embed_fn.
    """

    def __init__(self, threshold: float = 0.92, max_entries: int = 500,
                 default_ttl: float = 3600.0,
                 embed_fn=None,
                 adapter: Optional[CacheAdapter] = None):
        self._entries: List[CacheEntry] = []
        self._threshold = threshold
        self._max_entries = max_entries
        self._default_ttl = default_ttl
        self._embed_fn = embed_fn or _simple_embedding
        self._adapter = adapter
        self._similarity_scores: List[float] = []  # for stats

    def get(self, query: str) -> Optional[Tuple[CacheEntry, float]]:
        """Find semantically similar cached response.

        Returns (entry, similarity_score) or None.
        """
        query_embedding = self._embed_fn(query)

        best_entry = None
        best_score = 0.0

        for entry in self._entries:
            if entry.is_expired:
                continue
            if entry.embedding is None:
                continue

            score = _cosine_similarity(query_embedding, entry.embedding)
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_entry and best_score >= self._threshold:
            best_entry.hit_count += 1
            self._similarity_scores.append(best_score)
            return best_entry, best_score

        return None

    def put(self, query: str, answer: str, confidence: float,
            strategy: str, metadata: Optional[Dict] = None,
            ttl: Optional[float] = None) -> CacheEntry:
        """Store response with embedding for similarity search."""
        embedding = self._embed_fn(query)

        entry = CacheEntry(
            query=_normalize_query(query),
            query_hash=_hash_query(query),
            answer=answer,
            confidence=confidence,
            strategy=strategy,
            metadata=metadata or {},
            ttl_seconds=ttl or self._default_ttl,
            embedding=embedding,
        )

        # Evict expired + overflow
        self._entries = [e for e in self._entries if not e.is_expired]
        if len(self._entries) >= self._max_entries:
            # Remove oldest
            self._entries.sort(key=lambda e: e.created_at)
            self._entries = self._entries[1:]

        self._entries.append(entry)

        # Also store in adapter if available
        if self._adapter:
            self._adapter.set(entry.query_hash, entry.to_dict(),
                              int(entry.ttl_seconds))

        return entry

    def invalidate(self, query: str) -> bool:
        """Remove entries similar to this query."""
        query_embedding = self._embed_fn(query)
        removed = False
        kept = []
        for entry in self._entries:
            if entry.embedding:
                score = _cosine_similarity(query_embedding, entry.embedding)
                if score >= self._threshold:
                    removed = True
                    continue
            kept.append(entry)
        self._entries = kept
        return removed

    def clear(self) -> None:
        self._entries.clear()
        self._similarity_scores.clear()

    @property
    def size(self) -> int:
        return len([e for e in self._entries if not e.is_expired])

    @property
    def avg_similarity(self) -> float:
        if not self._similarity_scores:
            return 0.0
        return sum(self._similarity_scores) / len(self._similarity_scores)


# ─── Cache Manager ────────────────────────────────────────


class CacheManager:
    """Unified cache manager combining exact + semantic caching.

    Lookup order:
        1. Exact match (instant, zero cost)
        2. Semantic match (embedding comparison)
        3. Cache miss -> run full pipeline

    Usage:
        cache = CacheManager(mode="both")
        hit = cache.lookup(query)
        if hit:
            return hit.answer  # saved an LLM call
        else:
            answer = llm.generate(...)
            cache.store(query, answer, confidence, strategy)
    """

    def __init__(self, mode: str = "exact",
                 semantic_threshold: float = 0.92,
                 max_entries: int = 1000,
                 default_ttl: float = 3600.0,
                 embed_fn=None,
                 adapter: Optional[CacheAdapter] = None,
                 persist_dir: Optional[str] = None):
        """
        Args:
            mode: "exact", "semantic", or "both"
            semantic_threshold: Minimum cosine similarity for semantic match
            max_entries: Maximum cache entries
            default_ttl: Time-to-live in seconds (default 1 hour)
            embed_fn: Custom embedding function(text) -> List[float]
            adapter: External cache backend (Redis, etc.)
            persist_dir: Directory to save/load cache to disk
        """
        self.mode = mode
        self._stats = CacheStats()
        self._persist_dir = Path(persist_dir) if persist_dir else None

        # Always create exact cache (cheap, no deps)
        self._exact = ExactCache(
            max_entries=max_entries,
            default_ttl=default_ttl,
            adapter=adapter,
        )

        # Semantic cache when mode requires it
        self._semantic = None
        if mode in ("semantic", "both"):
            self._semantic = SemanticCache(
                threshold=semantic_threshold,
                max_entries=max_entries,
                default_ttl=default_ttl,
                embed_fn=embed_fn,
                adapter=adapter,
            )

        # Load persisted cache
        if self._persist_dir:
            self._load()

        logger.info(
            f"LM Cache initialized: mode={mode}, "
            f"threshold={semantic_threshold}, "
            f"max_entries={max_entries}, "
            f"ttl={default_ttl}s, "
            f"adapter={'external' if adapter else 'in-memory'}"
        )

    def lookup(self, query: str) -> Optional[CacheEntry]:
        """Check cache for query. Returns CacheEntry or None."""
        self._stats.total_queries += 1

        # 1. Try exact match
        exact_hit = self._exact.get(query)
        if exact_hit:
            self._stats.cache_hits += 1
            self._stats.exact_hits += 1
            logger.debug(f"Cache HIT (exact): {query[:50]}...")
            return exact_hit

        # 2. Try semantic match
        if self._semantic:
            result = self._semantic.get(query)
            if result:
                entry, similarity = result
                self._stats.cache_hits += 1
                self._stats.semantic_hits += 1
                self._stats.avg_similarity = self._semantic.avg_similarity
                logger.debug(
                    f"Cache HIT (semantic, sim={similarity:.3f}): "
                    f"{query[:50]}..."
                )
                return entry

        self._stats.cache_misses += 1
        return None

    def store(self, query: str, answer: str, confidence: float,
              strategy: str, metadata: Optional[Dict] = None,
              ttl: Optional[float] = None) -> None:
        """Store response in cache."""
        # Always store in exact cache
        self._exact.put(query, answer, confidence, strategy, metadata, ttl)

        # Also store in semantic cache
        if self._semantic:
            self._semantic.put(query, answer, confidence, strategy,
                               metadata, ttl)

        self._stats.entries_count = self._exact.size
        logger.debug(f"Cache STORE: {query[:50]}...")

    def invalidate(self, query: str) -> None:
        """Remove cached entry for a query."""
        self._exact.invalidate(query)
        if self._semantic:
            self._semantic.invalidate(query)

    def clear(self) -> None:
        """Clear all cached entries."""
        self._exact.clear()
        if self._semantic:
            self._semantic.clear()
        self._stats = CacheStats()
        logger.info("Cache cleared")

    @property
    def stats(self) -> CacheStats:
        self._stats.entries_count = self._exact.size
        return self._stats

    def save(self) -> None:
        """Persist cache to disk."""
        if not self._persist_dir:
            return
        self._persist_dir.mkdir(parents=True, exist_ok=True)

        # Save exact cache entries
        data = {
            "mode": self.mode,
            "entries": [],
            "stats": self._stats.to_dict(),
        }

        for entry in self._exact._store.values():
            if not entry.is_expired:
                data["entries"].append(entry.to_dict())

        path = self._persist_dir / "lm_cache.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.debug(f"Cache saved: {len(data['entries'])} entries")

    def _load(self) -> None:
        """Load cache from disk."""
        if not self._persist_dir:
            return
        path = self._persist_dir / "lm_cache.json"
        if not path.exists():
            return

        try:
            with open(path, "r") as f:
                data = json.load(f)

            loaded = 0
            for entry_data in data.get("entries", []):
                entry = CacheEntry.from_dict(entry_data)
                if not entry.is_expired:
                    self._exact._store[entry.query_hash] = entry
                    if self._semantic:
                        # Re-embed for semantic search
                        entry.embedding = self._semantic._embed_fn(entry.query)
                        self._semantic._entries.append(entry)
                    loaded += 1

            logger.info(f"Cache loaded: {loaded} entries from disk")
        except Exception as e:
            logger.warning(f"Cache load failed: {e}")

    def display(self) -> str:
        """Human-readable cache status."""
        s = self._stats
        lines = [
            "LM Cache Status",
            f"  Mode:           {self.mode}",
            f"  Entries:        {s.entries_count}",
            f"  Total queries:  {s.total_queries}",
            f"  Cache hits:     {s.cache_hits} ({s.hit_rate:.1%})",
            f"    Exact hits:   {s.exact_hits}",
            f"    Semantic hits: {s.semantic_hits}",
            f"  Cache misses:   {s.cache_misses}",
        ]
        if s.avg_similarity > 0:
            lines.append(f"  Avg similarity: {s.avg_similarity:.3f}")
        return "\n".join(lines)
