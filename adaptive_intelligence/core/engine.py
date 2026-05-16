"""AdaptiveAI — The Main Orchestrator.

This is the core engine that connects ingestion, indexing, query understanding,
RL policy, graph intelligence, prompt engineering, LLM generation, evaluation,
and continuous learning into a single self-improving pipeline.

Usage:
    engine = AdaptiveAI()
    engine.ingest("./documents")
    response = engine.ask("What are the operational risks?")
"""

import logging
import time
from typing import Optional, Dict, Any, List, Union
from pathlib import Path

from adaptive_intelligence.core.config import AdaptiveConfig, LLMBackend
from adaptive_intelligence.core.response import (
    AdaptiveResponse, Citation, RetrievalInfo, EvaluationResult, PolicyDecision,
)
from adaptive_intelligence.ingestion.engine import IngestionEngine, IngestionStats
from adaptive_intelligence.ingestion.chunker import Chunk
from adaptive_intelligence.indexes.vector_index import VectorIndex
from adaptive_intelligence.indexes.keyword_index import KeywordIndex
from adaptive_intelligence.indexes.base import RetrievalResult
from adaptive_intelligence.query import TriggerInterpreter, QueryAnalysis
from adaptive_intelligence.rl import RLPolicyEngine, PolicyAction, RetrievalRoute
from adaptive_intelligence.graph import KnowledgeGraph
from adaptive_intelligence.prompts import AdaptivePromptEngine
from adaptive_intelligence.evaluation import EvaluationEngine
from adaptive_intelligence.memory import LearningMemory, QueryLog
from adaptive_intelligence.llm import LLMManager, BaseLLMProvider
from adaptive_intelligence.security import AuditTrail, SecurityManager
from adaptive_intelligence.utils import generate_query_id, setup_logging

logger = logging.getLogger(__name__)


class AdaptiveAI:
    """Self-improving retrieval orchestration engine.

    Drop documents. Ask questions. The system learns optimal retrieval
    strategies through reinforcement learning from evaluation feedback.

    Three levels of access:
        Level 0 (80%): engine = AdaptiveAI(); engine.ingest("./docs"); engine.ask("query")
        Level 1 (15%): AdaptiveAI(domain="financial", security_level="maximum")
        Level 2 (5%):  engine.rl.exploration_rate = 0.1; engine.graph.config.max_hops = 3
    """

    def __init__(self, config: Optional[AdaptiveConfig] = None, **kwargs):
        """Initialize Adaptive Intelligence engine.

        Args:
            config: Full configuration object. If None, uses defaults.
            **kwargs: Shortcut configuration (domain, llm_backend, llm_model,
                      api_key, base_url, security_level, etc.)
        """
        # Build config from kwargs if no config provided
        if config is None:
            config = self._build_config_from_kwargs(kwargs)
        self.config = config

        # Setup logging
        setup_logging(config.log_level, config.log_file)
        logger.info(f"Initializing Adaptive Intelligence v1.0.1")

        # Storage directory
        self._storage_dir = Path(config.storage_dir)
        self._storage_dir.mkdir(parents=True, exist_ok=True)

        # Initialize all components
        self.ingestion = IngestionEngine(config)

        # Indexes
        vector_persist = str(self._storage_dir / "vector_index") if config.storage_dir else None
        self.vector_index = VectorIndex(persist_dir=vector_persist)
        self.keyword_index = KeywordIndex()

        # Query understanding
        self.trigger = TriggerInterpreter()

        # RL Policy Engine (core differentiator #1)
        rl_persist = str(self._storage_dir / "rl_state")
        self.rl = RLPolicyEngine(config=config.rl, persist_dir=rl_persist)

        # Graph Intelligence (core differentiator #2)
        self.graph = KnowledgeGraph(config=config.graph)

        # Prompt Engine
        self.prompts = AdaptivePromptEngine()

        # LLM Provider
        self._llm: Optional[BaseLLMProvider] = None
        self._init_llm()

        # Evaluation Engine
        self.evaluation = EvaluationEngine(config=config.evaluation, llm_provider=self._llm)

        # Continuous Learning Memory
        memory_persist = str(self._storage_dir / "memory")
        self.memory = LearningMemory(persist_dir=memory_persist)

        # Security & Audit
        self.audit = AuditTrail(enabled=config.enable_audit_trail)
        self.security = SecurityManager(security_level=config.security_level.value)

        # State tracking
        self._is_ingested = False
        self._total_queries = 0

        logger.info(
            f"Engine initialized: llm={config.llm_backend.value}/{config.llm_model}, "
            f"domain={config.domain.value}, security={config.security_level.value}"
        )

    def _build_config_from_kwargs(self, kwargs: Dict[str, Any]) -> AdaptiveConfig:
        """Build AdaptiveConfig from keyword arguments."""
        config = AdaptiveConfig()

        # Map simple kwargs to config
        if "domain" in kwargs:
            from adaptive_intelligence.core.config import Domain
            config.domain = Domain(kwargs["domain"])
        if "llm_backend" in kwargs:
            config.llm_backend = LLMBackend(kwargs["llm_backend"])
        if "llm_model" in kwargs:
            config.llm_model = kwargs["llm_model"]
        if "api_key" in kwargs:
            config.api_key = kwargs["api_key"]
            config.network_enabled = True
        if "base_url" in kwargs:
            config.base_url = kwargs["base_url"]
        if "azure_endpoint" in kwargs:
            config.azure_endpoint = kwargs["azure_endpoint"]
        if "deployment_name" in kwargs:
            config.deployment_name = kwargs["deployment_name"]
        if "security_level" in kwargs:
            from adaptive_intelligence.core.config import SecurityLevel
            config.security_level = SecurityLevel(kwargs["security_level"])
        if "network_enabled" in kwargs:
            config.network_enabled = kwargs["network_enabled"]
        if "temperature" in kwargs:
            config.temperature = kwargs["temperature"]
        if "storage_dir" in kwargs:
            config.storage_dir = kwargs["storage_dir"]
        if "log_level" in kwargs:
            config.log_level = kwargs["log_level"]

        return config

    def _init_llm(self):
        """Initialize the LLM provider."""
        try:
            self._llm = LLMManager.create(
                backend=self.config.llm_backend.value,
                model=self.config.llm_model,
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                azure_endpoint=self.config.azure_endpoint,
                deployment_name=self.config.deployment_name,
            )
            logger.info(f"LLM provider: {self._llm.provider_name}")
        except Exception as e:
            logger.warning(f"LLM initialization failed: {e}. Generation will be unavailable.")
            self._llm = None

    # ─── Ingestion ─────────────────────────────────────────────────────

    def ingest(self, source: Union[str, List[str]]) -> IngestionStats:
        """Ingest documents from files or directories.

        Args:
            source: File path, directory path, or list of paths.

        Returns:
            IngestionStats with details about what was parsed.
        """
        self.audit.log("ingestion_start", {"source": str(source)})
        total_stats = IngestionStats()

        sources = source if isinstance(source, list) else [source]

        for src in sources:
            stats = self.ingestion.ingest(src)
            total_stats.total_files += stats.total_files
            total_stats.successful += stats.successful
            total_stats.failed += stats.failed
            total_stats.total_chunks += stats.total_chunks
            total_stats.total_tables += stats.total_tables
            total_stats.duration_seconds += stats.duration_seconds
            for ft, count in stats.file_types.items():
                total_stats.file_types[ft] = total_stats.file_types.get(ft, 0) + count

        # Index all chunks
        chunks = self.ingestion.get_chunks()
        if chunks:
            self.vector_index.add(chunks)
            self.keyword_index.add(chunks)

            # Build knowledge graph from chunks
            if self.config.graph.enabled:
                self.graph.build_from_chunks(chunks)

        self._is_ingested = True

        self.audit.log("ingestion_complete", {
            "documents": total_stats.total_files,
            "chunks": total_stats.total_chunks,
            "tables": total_stats.total_tables,
            "graph_nodes": self.graph.node_count,
            "graph_edges": self.graph.edge_count,
            "duration": total_stats.duration_seconds,
        })

        logger.info(
            f"Ingestion complete: {total_stats.successful} docs, "
            f"{total_stats.total_chunks} chunks, "
            f"{self.graph.node_count} graph nodes"
        )
        return total_stats

    # ─── Query ─────────────────────────────────────────────────────────

    def ask(self, query: str, priority: Optional[str] = None,
            depth: Optional[int] = None) -> AdaptiveResponse:
        """Ask a question over your ingested documents.

        This is where the magic happens:
        1. Query understanding (Trigger Interpreter)
        2. RL policy decides retrieval strategy
        3. Execute retrieval (vector, keyword, graph as decided)
        4. Build adaptive prompt
        5. Generate answer via LLM
        6. Evaluate response quality
        7. Update RL policy with reward
        8. Store learning in memory

        Args:
            query: The question to ask.
            priority: Optional hint — "accuracy", "speed", "detail".
            depth: Override retrieval depth.

        Returns:
            AdaptiveResponse with answer, confidence, citations, evaluation.
        """
        start_time = time.time()
        query_id = generate_query_id()
        self._total_queries += 1

        self.audit.log("query_received", {"query": query, "priority": priority}, query_id)

        # Step 1: Query Understanding
        analysis = self.trigger.analyze(query)
        self.audit.log("query_analyzed", analysis.to_dict(), query_id)

        # Step 2: RL Policy Decision
        policy_action = self.rl.decide(analysis)
        if depth:
            policy_action.retrieval_depth = depth

        self.audit.log("rl_decision", policy_action.to_dict(), query_id)

        # Step 3: Execute Retrieval
        retrieved_chunks = self._execute_retrieval(query, analysis, policy_action)
        self.audit.log("retrieval_complete", {
            "chunks_retrieved": len(retrieved_chunks),
            "strategy": policy_action.retrieval_route.value,
        }, query_id)

        # Step 4: Graph Traversal (conditional)
        graph_context = ""
        graph_activated = False
        if policy_action.graph_activation and self.graph.should_activate(analysis):
            graph_result = self.graph.traverse(
                seed_entities=analysis.entities,
                max_hops=policy_action.graph_depth,
            )
            graph_context = graph_result.to_context()
            graph_activated = True

            # Add graph-related chunks to retrieval
            for chunk_id in graph_result.related_chunks:
                for chunk in self.ingestion.get_chunks():
                    if chunk.chunk_id == chunk_id and chunk not in retrieved_chunks:
                        retrieved_chunks.append(chunk)

            self.audit.log("graph_traversal", {
                "nodes_visited": len(graph_result.nodes_visited),
                "paths_found": len(graph_result.paths),
                "confidence": graph_result.confidence,
            }, query_id)

        # Step 5: Build Adaptive Prompt
        prompt = self.prompts.build_prompt(
            query=query,
            query_analysis=analysis,
            chunks=retrieved_chunks,
            graph_context=graph_context,
            template_override=policy_action.prompt_template,
        )

        # Step 6: Generate Answer via LLM
        answer_text = ""
        llm_latency = 0.0
        token_usage = 0

        if self._llm:
            try:
                system_prompt = (
                    "You are an intelligent document analysis assistant. "
                    "Answer questions based strictly on the provided context. "
                    "Cite sources for every claim. If information is not in the "
                    "context, say so clearly."
                )
                llm_response = self._llm.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )
                answer_text = llm_response.text
                llm_latency = llm_response.latency_seconds
                token_usage = llm_response.total_tokens
            except Exception as e:
                logger.error(f"LLM generation failed: {e}")
                answer_text = self._fallback_answer(query, retrieved_chunks)
        else:
            answer_text = self._fallback_answer(query, retrieved_chunks)

        total_latency = time.time() - start_time

        self.audit.log("llm_response", {
            "latency": llm_latency,
            "tokens": token_usage,
            "answer_length": len(answer_text),
        }, query_id)

        # Step 7: Evaluate Response
        eval_result = self.evaluation.evaluate(
            query=query,
            answer=answer_text,
            retrieved_chunks=retrieved_chunks,
            latency_seconds=total_latency,
            token_usage=token_usage,
        )

        self.audit.log("evaluation", {
            "faithfulness": eval_result.faithfulness,
            "relevance": eval_result.relevance,
            "composite": eval_result.composite_score,
            "confidence": eval_result.confidence_level,
        }, query_id)

        # Step 8: Update RL Policy with Reward
        reward = self.evaluation.compute_reward(eval_result)
        self.rl.update(query_id, analysis, policy_action, reward)

        self.audit.log("rl_updated", {
            "reward": reward,
            "exploration_rate": self.rl._exploration_rate,
        }, query_id)

        # Step 9: Update Learning Memory
        self.memory.record_query(QueryLog(
            query_id=query_id,
            query=query,
            query_type=analysis.query_type.value,
            domain=analysis.domain,
            route_used=policy_action.retrieval_route.value,
            retrieval_depth=policy_action.retrieval_depth,
            graph_activated=graph_activated,
            composite_score=eval_result.composite_score,
            latency=total_latency,
        ))

        if graph_activated:
            self.graph.record_activation_outcome(eval_result.composite_score > 0.7)
            self.memory.record_graph_activation(
                analysis.query_type.value,
                eval_result.composite_score > 0.7,
                eval_result.composite_score,
            )

        # Step 10: Update Prompt Engine
        template_type = policy_action.prompt_template
        self.prompts.update_from_evaluation(
            template_id=f"{template_type}_v1",
            evaluation_scores={
                "faithfulness": eval_result.faithfulness,
                "relevance": eval_result.relevance,
                "citation_accuracy": eval_result.citation_accuracy,
                "hallucination_risk": eval_result.hallucination_risk,
                "composite_score": eval_result.composite_score,
            },
        )

        # Build citations
        citations = self._extract_citations(answer_text, retrieved_chunks)

        # Build response
        response = AdaptiveResponse(
            answer=answer_text,
            confidence=eval_result.composite_score,
            citations=citations,
            evaluation=eval_result,
            retrieval_info=RetrievalInfo(
                strategy=policy_action.retrieval_route.value,
                indexes_used=[policy_action.retrieval_route.value],
                chunks_retrieved=len(retrieved_chunks),
                retrieval_depth=policy_action.retrieval_depth,
                graph_activated=graph_activated,
                graph_hops=policy_action.graph_depth if graph_activated else 0,
                reranking_applied=False,
            ),
            policy_decision=PolicyDecision(
                retrieval_route=policy_action.retrieval_route.value,
                retrieval_depth=policy_action.retrieval_depth,
                graph_activation=graph_activated,
                graph_depth=policy_action.graph_depth,
                model_used=self.config.llm_model,
                prompt_template=policy_action.prompt_template,
                verification_level=policy_action.verification_level,
                was_exploration=policy_action.was_exploration,
                policy_confidence=policy_action.action_confidence,
            ),
            query_analysis=analysis.to_dict(),
            query_id=query_id,
            raw_chunks=[{"content": c.content, "source": c.source_document} for c in retrieved_chunks],
        )

        logger.info(
            f"Query '{query[:50]}...' → confidence={eval_result.composite_score:.2f}, "
            f"strategy={policy_action.retrieval_route.value}, "
            f"latency={total_latency:.2f}s"
        )

        return response

    # ─── Retrieval Execution ──────────────────────────────────────────

    def _execute_retrieval(self, query: str, analysis: QueryAnalysis,
                           action: PolicyAction) -> List[Chunk]:
        """Execute retrieval based on RL policy decision."""
        depth = action.retrieval_depth
        route = action.retrieval_route

        if route == RetrievalRoute.VECTOR_ONLY:
            results = self.vector_index.search(query, top_k=depth)
        elif route == RetrievalRoute.KEYWORD_ONLY:
            results = self.keyword_index.search(query, top_k=depth)
        elif route == RetrievalRoute.HYBRID:
            results = self._hybrid_search(query, depth)
        elif route == RetrievalRoute.TABLE_FIRST:
            # Search keyword (better for structured data) then vector
            kw_results = self.keyword_index.search(query, top_k=depth)
            table_results = [r for r in kw_results if r.chunk.is_table]
            if len(table_results) < depth:
                vec_results = self.vector_index.search(query, top_k=depth - len(table_results))
                results = table_results + vec_results
            else:
                results = table_results[:depth]
        elif route in (RetrievalRoute.GRAPH_FIRST, RetrievalRoute.GRAPH_HYBRID):
            # Start with vector/hybrid, graph augmentation happens in ask()
            results = self._hybrid_search(query, depth)
        else:
            results = self._hybrid_search(query, depth)

        # Deduplicate by chunk_id
        seen = set()
        unique_chunks = []
        for r in results:
            if r.chunk.chunk_id not in seen:
                seen.add(r.chunk.chunk_id)
                unique_chunks.append(r.chunk)

        return unique_chunks[:depth]

    def _hybrid_search(self, query: str, top_k: int) -> List[RetrievalResult]:
        """Combine vector and keyword search with score fusion."""
        vec_results = self.vector_index.search(query, top_k=top_k)
        kw_results = self.keyword_index.search(query, top_k=top_k)

        # Reciprocal Rank Fusion
        scores: Dict[str, float] = {}
        chunk_map: Dict[str, RetrievalResult] = {}

        for rank, result in enumerate(vec_results):
            cid = result.chunk.chunk_id
            scores[cid] = scores.get(cid, 0) + 1.0 / (rank + 60)  # k=60
            chunk_map[cid] = result

        for rank, result in enumerate(kw_results):
            cid = result.chunk.chunk_id
            scores[cid] = scores.get(cid, 0) + 1.0 / (rank + 60)
            if cid not in chunk_map:
                chunk_map[cid] = result

        # Sort by fused score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for cid, score in ranked[:top_k]:
            r = chunk_map[cid]
            r.score = score
            results.append(r)

        return results

    # ─── Helper Methods ────────────────────────────────────────────────

    def _fallback_answer(self, query: str, chunks: List[Chunk]) -> str:
        """Generate a simple answer without LLM by presenting relevant chunks."""
        if not chunks:
            return "I could not find relevant information to answer this query."

        parts = ["Based on the available documents:\n"]
        for i, chunk in enumerate(chunks[:5]):
            source = chunk.source_document or "Unknown"
            preview = chunk.content[:300].strip()
            parts.append(f"[{i+1}] From {source}:\n{preview}\n")

        parts.append(
            "\nNote: This is a direct excerpt from the source documents. "
            "Connect an LLM provider for synthesized answers."
        )
        return "\n".join(parts)

    def _extract_citations(self, answer: str, chunks: List[Chunk]) -> List[Citation]:
        """Extract citations from the answer by matching to source chunks."""
        citations = []
        for chunk in chunks[:10]:  # Top chunks as citations
            # Simple overlap check
            answer_lower = answer.lower()
            chunk_words = set(chunk.content.lower().split())
            answer_words = set(answer_lower.split())
            overlap = len(chunk_words & answer_words)
            if overlap > 5:
                confidence = min(1.0, overlap / max(len(chunk_words), 1) * 2)
                citations.append(Citation(
                    text=chunk.content[:200],
                    source_document=chunk.source_document,
                    chunk_id=chunk.chunk_id,
                    confidence=confidence,
                ))

        # Sort by confidence
        citations.sort(key=lambda c: c.confidence, reverse=True)
        return citations[:5]

    # ─── Dashboard & Monitoring ────────────────────────────────────────

    def dashboard(self) -> str:
        """Display system performance dashboard."""
        eval_avg = self.evaluation.get_average_score()
        improvement = self.evaluation.get_improvement_rate()
        rl_stats = self.rl.get_stats()
        memory_stats = self.memory.get_stats()
        graph_stats = self.graph.get_stats()

        lines = [
            "┌─────────────────────────────────────────────────────────┐",
            "│  ADAPTIVE INTELLIGENCE DASHBOARD                       │",
            "│                                                        │",
            f"│  Documents Indexed:    {self.vector_index.count():>6}                          │",
            f"│  Queries Processed:    {self._total_queries:>6}                          │",
            f"│  Average Accuracy:     {eval_avg:>5.1%}                          │",
            f"│  Improvement Rate:     {improvement:>+5.1%}                          │",
            "│                                                        │",
            f"│  RL Policy:            {'Warmup' if rl_stats['is_warmup'] else 'Active':>10}                    │",
            f"│  Exploration Rate:     {rl_stats['exploration_rate']:>5.1%}                          │",
            f"│  Arms Learned:         {rl_stats['total_arms']:>6}                          │",
            "│                                                        │",
            f"│  Graph Nodes:          {graph_stats['nodes']:>6}                          │",
            f"│  Graph Edges:          {graph_stats['edges']:>6}                          │",
            f"│  Graph Success Rate:   {graph_stats['activation_success_rate']:>5.1%}                          │",
            "│                                                        │",
            f"│  Routing Patterns:     {memory_stats['routing_patterns']:>6}                          │",
            "└─────────────────────────────────────────────────────────┘",
        ]
        return "\n".join(lines)

    def learning_curve(self) -> List[Dict[str, Any]]:
        """Get the learning curve data showing improvement over queries."""
        return self.rl.get_learning_curve()

    def status(self) -> Dict[str, Any]:
        """Get complete system status."""
        return {
            "version": "1.0.1",
            "documents_indexed": self.vector_index.count(),
            "total_queries": self._total_queries,
            "llm_provider": self._llm.provider_name if self._llm else "none",
            "llm_model": self.config.llm_model,
            "domain": self.config.domain.value,
            "rl_status": "warmup" if self.rl.is_warmup else "active",
            "graph_nodes": self.graph.node_count,
            "graph_edges": self.graph.edge_count,
            "average_accuracy": self.evaluation.get_average_score(),
        }

    def reset(self):
        """Reset all state (indexes, memory, RL policy, graph)."""
        self.vector_index.clear()
        self.keyword_index.clear()
        self.graph.clear()
        self.memory.clear()
        self.audit.clear()
        self.ingestion.clear()
        self._is_ingested = False
        self._total_queries = 0
        logger.info("Engine reset complete")
