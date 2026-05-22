"""Continuous Learning Memory — Stores what the system learns.

Memory stores: routing patterns, prompt library scores, success/failure
logs, graph thresholds, model performance, and optimal chunk sizes.
"""

import json
import logging
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class RoutingPattern:
    """A learned routing pattern."""
    query_type: str
    domain: str
    best_route: str
    success_rate: float
    total_queries: int
    last_updated: float = field(default_factory=time.time)


@dataclass
class QueryLog:
    """Log entry for a single query."""
    query_id: str
    query: str
    query_type: str
    domain: str
    route_used: str
    retrieval_depth: int
    graph_activated: bool
    composite_score: float
    latency: float
    timestamp: float = field(default_factory=time.time)


class LearningMemory:
    """Persistent memory that stores learned patterns and logs.

    This module enables the "self-adaptive" behavior:
    the system remembers what worked and what didn't.
    """

    def __init__(self, persist_dir: Optional[str] = None):
        self._persist_dir = persist_dir

        # Routing pattern memory
        self._routing_patterns: Dict[str, RoutingPattern] = {}

        # Success/failure logs
        self._success_log: List[QueryLog] = []
        self._failure_log: List[QueryLog] = []

        # Performance tracking per route
        self._route_performance: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"total": 0, "success": 0, "avg_score": 0.0}
        )

        # Graph activation memory
        self._graph_thresholds: Dict[str, float] = {}

        # Model performance memory
        self._model_performance: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"total": 0, "avg_score": 0.0, "avg_latency": 0.0}
        )

        # Chunk size optimization memory
        self._chunk_performance: Dict[str, float] = {}

        # Load persisted state
        if self._persist_dir:
            self._load_state()

    def record_query(self, log: QueryLog):
        """Record a query outcome."""
        # Classify as success or failure
        if log.composite_score >= 0.7:
            self._success_log.append(log)
        else:
            self._failure_log.append(log)

        # Update routing patterns
        pattern_key = f"{log.query_type}:{log.domain}"
        perf = self._route_performance[f"{pattern_key}:{log.route_used}"]
        perf["total"] += 1
        perf["success"] += 1 if log.composite_score >= 0.7 else 0
        n = perf["total"]
        perf["avg_score"] = perf["avg_score"] * ((n - 1) / n) + log.composite_score / n

        # Update best routing pattern
        self._update_routing_pattern(log)

        # Trim logs to prevent unbounded growth
        max_logs = 5000
        if len(self._success_log) > max_logs:
            self._success_log = self._success_log[-max_logs:]
        if len(self._failure_log) > max_logs:
            self._failure_log = self._failure_log[-max_logs:]

        # Periodic persistence
        if (len(self._success_log) + len(self._failure_log)) % 50 == 0:
            self._save_state()

    def _update_routing_pattern(self, log: QueryLog):
        """Update the best routing pattern for a query type."""
        pattern_key = f"{log.query_type}:{log.domain}"

        # Find best route for this pattern
        best_route = None
        best_score = 0.0
        best_total = 0

        for key, perf in self._route_performance.items():
            if key.startswith(pattern_key + ":") and perf["total"] >= 3:
                if perf["avg_score"] > best_score:
                    best_score = perf["avg_score"]
                    best_route = key.split(":")[-1]
                    best_total = perf["total"]

        if best_route:
            self._routing_patterns[pattern_key] = RoutingPattern(
                query_type=log.query_type,
                domain=log.domain,
                best_route=best_route,
                success_rate=best_score,
                total_queries=best_total,
            )

    def get_best_route(self, query_type: str, domain: str) -> Optional[str]:
        """Get the best known route for a query type and domain."""
        pattern_key = f"{query_type}:{domain}"
        pattern = self._routing_patterns.get(pattern_key)
        if pattern and pattern.total_queries >= 5:
            return pattern.best_route
        return None

    def record_graph_activation(self, query_type: str, was_helpful: bool, score: float):
        """Record whether graph activation helped for a query type."""
        key = f"graph:{query_type}"
        current = self._graph_thresholds.get(key, 0.5)
        alpha = 0.2
        new_val = alpha * score + (1 - alpha) * current
        self._graph_thresholds[key] = new_val

    def should_activate_graph(self, query_type: str) -> Optional[bool]:
        """Check memory for whether graph typically helps for this query type."""
        key = f"graph:{query_type}"
        threshold = self._graph_thresholds.get(key)
        if threshold is not None:
            return threshold > 0.6
        return None  # No opinion yet

    def record_model_performance(self, model: str, score: float, latency: float):
        """Record model performance for a query."""
        perf = self._model_performance[model]
        perf["total"] += 1
        n = perf["total"]
        perf["avg_score"] = perf["avg_score"] * ((n - 1) / n) + score / n
        perf["avg_latency"] = perf["avg_latency"] * ((n - 1) / n) + latency / n

    def get_stats(self) -> Dict[str, Any]:
        """Get learning memory statistics."""
        return {
            "total_queries": len(self._success_log) + len(self._failure_log),
            "success_count": len(self._success_log),
            "failure_count": len(self._failure_log),
            "routing_patterns": len(self._routing_patterns),
            "graph_thresholds": dict(self._graph_thresholds),
            "top_patterns": {
                k: {"route": v.best_route, "score": v.success_rate, "queries": v.total_queries}
                for k, v in sorted(
                    self._routing_patterns.items(),
                    key=lambda x: x[1].success_rate,
                    reverse=True,
                )[:10]
            },
        }

    def get_learning_summary(self) -> str:
        """Human-readable summary of what the system has learned."""
        stats = self.get_stats()
        lines = [
            f"Queries processed: {stats['total_queries']}",
            f"Success rate: {stats['success_count']}/{stats['total_queries']}",
            f"Routing patterns learned: {stats['routing_patterns']}",
            "",
            "Top routing patterns:",
        ]
        for key, info in stats.get("top_patterns", {}).items():
            lines.append(
                f"  {key} → {info['route']} "
                f"(score: {info['score']:.2f}, queries: {info['queries']})"
            )
        return "\n".join(lines)

    def _save_state(self):
        """Persist memory to disk."""
        if not self._persist_dir:
            return

        path = Path(self._persist_dir)
        path.mkdir(parents=True, exist_ok=True)

        state = {
            "routing_patterns": {
                k: {
                    "query_type": v.query_type,
                    "domain": v.domain,
                    "best_route": v.best_route,
                    "success_rate": v.success_rate,
                    "total_queries": v.total_queries,
                }
                for k, v in self._routing_patterns.items()
            },
            "graph_thresholds": self._graph_thresholds,
            "route_performance": dict(self._route_performance),
            "model_performance": dict(self._model_performance),
        }

        with open(path / "learning_memory.json", "w") as f:
            json.dump(state, f, indent=2)

    def _load_state(self):
        """Load persisted memory."""
        if not self._persist_dir:
            return

        state_file = Path(self._persist_dir) / "learning_memory.json"
        if not state_file.exists():
            return

        try:
            with open(state_file) as f:
                state = json.load(f)

            for k, v in state.get("routing_patterns", {}).items():
                self._routing_patterns[k] = RoutingPattern(**v)

            self._graph_thresholds = state.get("graph_thresholds", {})

            for k, v in state.get("route_performance", {}).items():
                self._route_performance[k] = v

            for k, v in state.get("model_performance", {}).items():
                self._model_performance[k] = v

            logger.info(f"Learning memory loaded: {len(self._routing_patterns)} patterns")
        except Exception as e:
            logger.warning(f"Failed to load learning memory: {e}")

    def record_feedback(self, query_id: str, rating: str, reason: str = None):
        """Record user feedback for a query."""
        self._feedback_log = getattr(self, '_feedback_log', [])
        self._feedback_log.append({
            "query_id": query_id,
            "rating": rating,
            "reason": reason,
            "timestamp": time.time(),
        })

    def clear(self):
        """Clear all memory."""
        self._routing_patterns.clear()
        self._success_log.clear()
        self._failure_log.clear()
        self._route_performance.clear()
        self._graph_thresholds.clear()
        self._model_performance.clear()
        self._chunk_performance.clear()
