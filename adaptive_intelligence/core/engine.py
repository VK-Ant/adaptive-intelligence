"""AdaptiveAI — The Main Orchestrator (v4).

v4 additions: context engineering, MCP integration, agentic workflow,
persistent memory, incremental learning, tool registry.

Usage:
    engine = AdaptiveAI()
    engine.ingest("./documents")
    response = engine.ask("What are the key risks?")

    # v4: Add docs later, system continues learning
    engine.ingest("./new_report.pdf")

    # v4: Connect tools
    engine.add_tool("financial", server="http://localhost:8081")

    # v4: Agentic mode
    response = engine.ask("Deep research on risks", mode="agentic")

    # v4: MCP server
    engine.serve_mcp(port=8080)
"""

import json
import logging
import time
import signal
import atexit
from typing import Optional, Dict, Any, List, Union, Callable
from pathlib import Path

from adaptive_intelligence.core.config import AdaptiveConfig, LLMBackend
from adaptive_intelligence.core.response import (
    AdaptiveResponse, Citation, RetrievalInfo, EvaluationResult, PolicyDecision,
)
from adaptive_intelligence.ingestion.engine import IngestionEngine, IngestionStats
from adaptive_intelligence.ingestion.chunker import Chunk
from adaptive_intelligence.indexes.vector_index import VectorIndex
from adaptive_intelligence.indexes.keyword_index import KeywordIndex
from adaptive_intelligence.indexes.page_index import PageIndex
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
    """Self-improving retrieval orchestration engine — v3.

    v3 new features:
        - rl_algorithm: "thompson" (default) or "ppo"
        - reranking: cross-encoder re-scoring of retrieved chunks
        - multi-query: auto-decompose complex queries into sub-queries
        - pretrained_policy: skip warmup with domain-specific policy
        - export_policy / import_policy: transfer learning
        - enable_ab_test: compare two policies simultaneously
    """

    def __init__(self, config: Optional[AdaptiveConfig] = None, **kwargs):
        if config is None:
            config = self._build_config_from_kwargs(kwargs)
        self.config = config

        # v2: vectorless mode
        self._vectorless = kwargs.get("vectorless", False)

        # v2: custom system prompt
        self._system_prompt = kwargs.get("system_prompt", None)

        # Setup
        setup_logging(config.log_level, config.log_file)
        logger.info(f"Initializing Adaptive Intelligence v4.0.0")

        self._storage_dir = Path(config.storage_dir)
        self._storage_dir.mkdir(parents=True, exist_ok=True)

        # Ingestion
        self.ingestion = IngestionEngine(
            config=config,
            page_mode=self._vectorless,
            min_chunk_quality=kwargs.get("min_chunk_quality", 0.3),
            deduplicate=kwargs.get("deduplicate", False),
        )

        # Indexes
        if self._vectorless:
            self.page_index = PageIndex()
            self.vector_index = None
            logger.info("Vectorless mode: using page BM25 index")
        else:
            vector_persist = str(self._storage_dir / "vector_index")
            self.vector_index = VectorIndex(persist_dir=vector_persist)
            self.page_index = None

        self.keyword_index = KeywordIndex()

        # Query understanding
        self.trigger = TriggerInterpreter()

        # RL Policy Engine
        rl_persist = str(self._storage_dir / "rl_state")
        rl_config = config.rl
        if "warmup_queries" in kwargs:
            rl_config.warmup_queries = kwargs["warmup_queries"]
        self.rl = RLPolicyEngine(config=rl_config, persist_dir=rl_persist)

        # Graph Intelligence
        self.graph = KnowledgeGraph(config=config.graph)

        # Prompt Engine
        self.prompts = AdaptivePromptEngine()

        # LLM Provider
        self._llm: Optional[BaseLLMProvider] = None
        if config.llm_backend != LLMBackend.NONE:
            self._init_llm()

        # Evaluation Engine
        self.evaluation = EvaluationEngine(config=config.evaluation, llm_provider=self._llm)

        # Learning Memory
        memory_persist = str(self._storage_dir / "memory")
        self.memory = LearningMemory(persist_dir=memory_persist)

        # Security & Audit
        self.audit = AuditTrail(enabled=config.enable_audit_trail)
        self.security = SecurityManager(security_level=config.security_level.value)

        # State
        self._is_ingested = False
        self._total_queries = 0
        self._feedback_cache: Dict[str, Dict] = {}  # query_id -> {analysis, action, reward}

        # v2: Crash recovery — load persisted state
        self._recover_state()

        # v2: Auto-checkpoint on shutdown
        self._checkpoint_interval = kwargs.get("checkpoint_minutes", 5) * 60
        self._last_checkpoint = time.time()
        atexit.register(self._on_shutdown)

        # v3: Reranking
        self._reranking = kwargs.get("reranking", False)
        self._reranker = None
        if self._reranking:
            try:
                from adaptive_intelligence.rl.reranker import CrossEncoderReranker
                self._reranker = CrossEncoderReranker()
                if not self._reranker.is_available:
                    from adaptive_intelligence.rl.reranker import KeywordReranker
                    self._reranker = KeywordReranker()
                    logger.info("Using keyword reranker (install sentence-transformers for cross-encoder)")
            except Exception:
                from adaptive_intelligence.rl.reranker import KeywordReranker
                self._reranker = KeywordReranker()

        # v3: Multi-query decomposition
        self._multi_query = kwargs.get("multi_query", True)
        if self._multi_query:
            from adaptive_intelligence.rl.multi_query import MultiQueryDecomposer
            self._decomposer = MultiQueryDecomposer()
        else:
            self._decomposer = None

        # v3: Pre-trained policy
        if kwargs.get("pretrained_policy", False):
            domain_str = config.domain.value if hasattr(config.domain, 'value') else str(config.domain)
            self.rl.load_pretrained(domain_str)

        # v3: PPO algorithm
        self._rl_algorithm = kwargs.get("rl_algorithm", "thompson")
        self._ppo_policy = None
        if self._rl_algorithm == "ppo":
            from adaptive_intelligence.rl.ppo import PPOPolicy
            self._ppo_policy = PPOPolicy(n_actions=len(RetrievalRoute))

        # v4: Persistent memory
        self._memory_enabled = kwargs.get("memory", True)
        self._persistent_memory = None
        if self._memory_enabled:
            from adaptive_intelligence.memory.persistent import PersistentMemory
            self._persistent_memory = PersistentMemory(
                storage_dir=str(self._storage_dir)
            )

        # v4: Context engineering
        self._context_engineering = kwargs.get("context_engineering", True)
        self._context_engineer = None
        if self._context_engineering:
            from adaptive_intelligence.context import ContextEngineer
            self._context_engineer = ContextEngineer(
                token_budget=kwargs.get("token_budget", 4096)
            )

        # v4: Tool registry
        from adaptive_intelligence.mcp import ToolRegistry
        self._tool_registry = ToolRegistry()

        # v4: Agentic workflow
        from adaptive_intelligence.agentic import AgenticWorkflow
        self._agentic = AgenticWorkflow(
            max_rounds=kwargs.get("agentic_rounds", 3),
            confidence_threshold=kwargs.get("agentic_threshold", 0.7),
        )

        # v4.0.7: Harness agent
        from adaptive_intelligence.harness import HarnessAgent
        self._harness = HarnessAgent()

        # v4.0.7: Loop engineering
        from adaptive_intelligence.loop import LoopEngineer
        self._loop_engineer = LoopEngineer(
            base_warmup=kwargs.get("warmup", 15),
            base_exploration=kwargs.get("exploration_rate", 0.20),
        )

        logger.info(
            f"Engine initialized: llm={config.llm_backend.value}/{config.llm_model}, "
            f"vectorless={self._vectorless}, domain={config.domain.value}, "
            f"rl={self._rl_algorithm}, reranking={self._reranking}, "
            f"memory={self._memory_enabled}, context_engineering={self._context_engineering}, "
            f"harness=True, loop_engineering=True"
        )

    def _build_config_from_kwargs(self, kwargs: Dict[str, Any]) -> AdaptiveConfig:
        config = AdaptiveConfig()
        simple_map = {
            "llm_model": "llm_model", "api_key": "api_key",
            "base_url": "base_url", "azure_endpoint": "azure_endpoint",
            "deployment_name": "deployment_name", "temperature": "temperature",
            "storage_dir": "storage_dir", "log_level": "log_level",
        }
        for kw, attr in simple_map.items():
            if kw in kwargs:
                setattr(config, attr, kwargs[kw])

        if "domain" in kwargs:
            from adaptive_intelligence.core.config import Domain
            config.domain = Domain(kwargs["domain"])
        if "llm_backend" in kwargs:
            val = kwargs["llm_backend"]
            if val == "none":
                config.llm_backend = LLMBackend.NONE
            else:
                config.llm_backend = LLMBackend(val)
        if "security_level" in kwargs:
            from adaptive_intelligence.core.config import SecurityLevel
            config.security_level = SecurityLevel(kwargs["security_level"])
        if "api_key" in kwargs:
            config.network_enabled = True

        return config

    def _init_llm(self):
        try:
            self._llm = LLMManager.create(
                backend=self.config.llm_backend.value,
                model=self.config.llm_model,
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                azure_endpoint=self.config.azure_endpoint,
                deployment_name=self.config.deployment_name,
            )
            # Quick availability check for Ollama
            if self.config.llm_backend.value == "ollama" and not self._llm.is_available():
                logger.info("Ollama not running. Using retrieval-only mode.")
                self._llm = None
        except Exception as e:
            logger.info(f"LLM not available ({e}). Using retrieval-only mode.")
            self._llm = None

    # ─── INGESTION ─────────────────────────────────────────

    def ingest(self, source: Union[str, List[str]],
               parallel: bool = False, workers: int = 4,
               on_progress=None,
               tables: Optional[List[str]] = None,
               query: Optional[str] = None) -> IngestionStats:
        """Ingest documents, directories, or SQL databases."""
        self.audit.log("ingestion_start", {"source": str(source)})

        stats = self.ingestion.ingest(
            source, parallel=parallel, workers=workers,
            on_progress=on_progress, tables=tables, query=query,
        )

        # Index all chunks
        chunks = self.ingestion.get_chunks()
        if chunks:
            if self._vectorless:
                self.page_index.add(chunks)
            else:
                self.vector_index.add(chunks)
            self.keyword_index.add(chunks)

            if self.config.graph.enabled:
                self.graph.build_from_chunks(chunks)

        self._is_ingested = True
        self._checkpoint()

        self.audit.log("ingestion_complete", {
            "documents": stats.successful,
            "chunks": stats.total_chunks,
            "graph_nodes": self.graph.node_count,
        })

        return stats

    def remove(self, filename: str) -> int:
        """Remove a document from all indexes."""
        removed = self.ingestion.remove(filename)
        # Rebuild indexes from remaining chunks
        if removed > 0:
            self._rebuild_indexes()
        return removed

    def update(self, filepath: str) -> IngestionStats:
        """Re-ingest a single document."""
        stats = self.ingestion.update(filepath)
        if stats.successful > 0:
            self._rebuild_indexes()
        return stats

    def _rebuild_indexes(self):
        """Rebuild all indexes from current chunks."""
        chunks = self.ingestion.get_chunks()
        if self._vectorless:
            self.page_index.clear()
            self.page_index.add(chunks)
        else:
            self.vector_index.clear()
            self.vector_index.add(chunks)
        self.keyword_index.clear()
        self.keyword_index.add(chunks)
        self.graph.clear()
        if self.config.graph.enabled:
            self.graph.build_from_chunks(chunks)

    # ─── QUERY ─────────────────────────────────────────────

    def ask(self, query: str, priority: Optional[str] = None,
            depth: Optional[int] = None,
            system_prompt: Optional[str] = None,
            output_format: Optional[str] = None,
            schema: Optional[Dict] = None,
            mode: Optional[str] = None) -> AdaptiveResponse:
        """Ask a question over ingested documents.

        Args:
            query: The question.
            priority: Hint — "accuracy", "speed", "detail".
            depth: Override retrieval depth.
            system_prompt: Override system prompt for this query.
            output_format: "json", "csv", "yaml", "dataframe", or None.
            schema: Custom output schema (for json format).
            mode: "agentic" for multi-round retrieval, None for standard.
        """
        # v4: Agentic mode
        if mode == "agentic":
            agent_result = self._agentic.run(
                query, self, tool_registry=self._tool_registry
            )
            # Wrap in AdaptiveResponse
            response = AdaptiveResponse(
                answer=agent_result.answer,
                confidence=0.0,
                query_id=generate_query_id(),
            )
            response.agent_steps = agent_result.steps
            response.agent_rounds = agent_result.total_rounds
            response.tools_called = agent_result.tools_called
            return response
        start_time = time.time()
        query_id = generate_query_id()
        self._total_queries += 1

        self.audit.log("query_received", {"query": query, "output_format": output_format}, query_id)

        # Step 1: Query Understanding
        analysis = self.trigger.analyze(query)

        # Step 2: RL Policy Decision
        policy_action = self.rl.decide(analysis)
        if depth:
            policy_action.retrieval_depth = depth

        # Adjust routes for vectorless mode
        if self._vectorless:
            policy_action = self._adjust_for_vectorless(policy_action)

        # Step 3: Execute Retrieval
        retrieved_chunks = self._execute_retrieval(query, analysis, policy_action)

        # v3: Multi-query decomposition for complex queries
        if self._decomposer and self._decomposer.should_decompose(query):
            sub_queries = self._decomposer.decompose(query)
            if len(sub_queries) > 1:
                for sq in sub_queries:
                    if sq != query:
                        extra = self._execute_retrieval(sq, analysis, policy_action)
                        for chunk in extra:
                            if chunk not in retrieved_chunks:
                                retrieved_chunks.append(chunk)

        # v3: Reranking
        if self._reranker and retrieved_chunks:
            reranked = self._reranker.rerank(query, retrieved_chunks, top_k=policy_action.retrieval_depth)
            retrieved_chunks = [chunk for chunk, score in reranked]

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
            for chunk_id in graph_result.related_chunks:
                for chunk in self.ingestion.get_chunks():
                    if chunk.chunk_id == chunk_id and chunk not in retrieved_chunks:
                        retrieved_chunks.append(chunk)

        # Step 5: Build Prompt (v4: context engineering)
        if self._context_engineer:
            # Search memory for relevant context
            memory_entries = []
            if self._persistent_memory:
                memory_entries = self._persistent_memory.search(query, top_k=3)

            # Get session history
            session_history = []
            if self._persistent_memory:
                session_history = self._persistent_memory.get_session_context(last_n=3)

            # Call tools if any match
            tool_results = []
            if self._tool_registry._tools:
                suggested = self._tool_registry.select_tools(query)
                for tool_name in suggested[:2]:
                    tr = self._tool_registry.call_tool(tool_name, query)
                    if tr.success:
                        tool_results.append({"tool": tr.tool_name, "result": tr.result})

            # Build optimized context window
            domain_str = self.config.domain.value if hasattr(self.config.domain, 'value') else str(self.config.domain)
            ctx = self._context_engineer.build_context(
                query=query, chunks=retrieved_chunks,
                memory_entries=memory_entries,
                session_history=session_history,
                tool_results=tool_results,
                domain=domain_str,
                custom_system_prompt=system_prompt or self._system_prompt,
            )
            prompt = ctx.assemble()
        else:
            prompt = self.prompts.build_prompt(
                query=query, query_analysis=analysis,
                chunks=retrieved_chunks, graph_context=graph_context,
                template_override=policy_action.prompt_template,
            )

        # v2: Add output format instructions to prompt
        if output_format:
            prompt += "\n\n" + self._build_output_instructions(output_format, schema)

        # Step 6: LLM Generation
        answer_text = ""
        llm_latency = 0.0
        token_usage = 0

        if self._llm:
            try:
                active_prompt = (
                    system_prompt or self._system_prompt or
                    "You are an intelligent document analysis assistant. "
                    "Answer based strictly on the provided context. "
                    "Cite sources for every claim."
                )
                llm_response = self._llm.generate(
                    prompt=prompt, system_prompt=active_prompt,
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

        # Step 7: Evaluate
        eval_result = self.evaluation.evaluate(
            query=query, answer=answer_text,
            retrieved_chunks=retrieved_chunks,
            latency_seconds=total_latency, token_usage=token_usage,
        )

        # Step 8: Update RL Policy
        reward = self.evaluation.compute_reward(eval_result)
        self.rl.update(query_id, analysis, policy_action, reward)

        # Step 9: Update Memory
        self.memory.record_query(QueryLog(
            query_id=query_id, query=query,
            query_type=analysis.query_type.value, domain=analysis.domain,
            route_used=policy_action.retrieval_route.value,
            retrieval_depth=policy_action.retrieval_depth,
            graph_activated=graph_activated,
            composite_score=eval_result.composite_score, latency=total_latency,
        ))

        if graph_activated:
            self.graph.record_activation_outcome(eval_result.composite_score > 0.7)

        # Step 10: Prompt Evolution
        self.prompts.update_from_evaluation(
            template_id=f"{policy_action.prompt_template}_v1",
            evaluation_scores={
                "faithfulness": eval_result.faithfulness,
                "relevance": eval_result.relevance,
                "citation_accuracy": eval_result.citation_accuracy,
                "hallucination_risk": eval_result.hallucination_risk,
                "composite_score": eval_result.composite_score,
            },
        )

        # Cache for feedback
        self._feedback_cache[query_id] = {
            "analysis": analysis, "action": policy_action, "reward": reward,
        }

        # Build citations with page numbers
        citations = self._extract_citations(answer_text, retrieved_chunks)

        # Parse structured output
        structured = None
        if output_format:
            structured = self._parse_structured_output(answer_text, output_format)

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
            raw_chunks=[{"content": c.content, "source": c.source_document,
                         "page": c.page_number} for c in retrieved_chunks],
        )

        # v2: attach structured output
        if structured is not None:
            response.structured = structured

        # v4.0.7: Harness evaluation — evaluate every pipeline decision
        harness_report = None
        harness_rewards = {}
        try:
            graph_ctx = ""
            if graph_context and isinstance(graph_context, list):
                graph_ctx = " ".join(str(c) for c in graph_context)
            elif graph_context:
                graph_ctx = str(graph_context)

            tool_calls = []
            if hasattr(response, "tools_called") and response.tools_called:
                tool_calls = [{"tool": t, "result": ""} for t in response.tools_called]

            harness_report = self._harness.evaluate_pipeline(
                query=query,
                answer=answer_text,
                chunks_used=retrieved_chunks[:policy_action.retrieval_depth],
                chunks_retrieved=retrieved_chunks,
                route_chosen=policy_action.retrieval_route.value,
                depth=policy_action.retrieval_depth,
                graph_activated=policy_action.graph_activation and self.graph.node_count > 0,
                graph_context=graph_ctx,
                tools_called=tool_calls,
                answer_score=eval_result.composite_score,
                latency=total_latency,
            )

            harness_rewards = self._harness.get_granular_rewards(harness_report)
            response.harness_report = harness_report
            response.harness_efficiency = harness_report.efficiency
        except Exception as e:
            logger.debug(f"Harness evaluation skipped: {e}")

        # v4.0.7: Loop engineering — shape reward and update learning state
        try:
            domain_str = analysis.domain if hasattr(analysis, "domain") else "general"
            qtype_str = analysis.query_type.value if hasattr(analysis.query_type, "value") else str(analysis.query_type)

            # Shape the RL reward using harness signals
            shaped_reward = self._loop_engineer.shape_reward(
                eval_result.composite_score, harness_rewards
            )

            # Update loop state
            self._loop_engineer.update(
                domain=domain_str,
                query_type=qtype_str,
                strategy=policy_action.retrieval_route.value,
                score=shaped_reward,
                harness_rewards=harness_rewards,
            )

            response.shaped_reward = shaped_reward
            response.exploration_rate = self._loop_engineer.get_exploration_rate(
                domain_str, qtype_str
            )
        except Exception as e:
            logger.debug(f"Loop engineering skipped: {e}")

        # v4: Update persistent memory
        if self._persistent_memory:
            self._persistent_memory.add_session_context(
                query=query, answer=answer_text,
                strategy=policy_action.retrieval_route.value,
            )
            self._persistent_memory.learn_pattern(
                query_type=analysis.query_type.value,
                domain=analysis.domain,
                strategy=policy_action.retrieval_route.value,
                score=eval_result.composite_score,
            )

        # Auto-checkpoint
        self._maybe_checkpoint()

        return response

    # ─── v4: TOOLS ─────────────────────────────────────────

    def add_tool(self, name: str, description: str = "",
                 server: str = None, function: Callable = None,
                 api_endpoint: str = None, api_key: str = None):
        """Register an external tool (MCP server, function, or API).

        Args:
            name: Tool identifier (e.g., "financial", "medical")
            description: What the tool does
            server: MCP server URL
            function: Python callable
            api_endpoint: REST API URL
            api_key: API authentication
        """
        self._tool_registry.add_tool(
            name=name, description=description,
            server=server, function=function,
            api_endpoint=api_endpoint, api_key=api_key,
        )

    def remove_tool(self, name: str):
        """Remove a registered tool."""
        self._tool_registry.remove_tool(name)

    def list_tools(self) -> List[Dict[str, Any]]:
        """List all registered tools."""
        return self._tool_registry.list_tools()

    # ─── v4: MCP SERVER ────────────────────────────────────

    def serve_mcp(self, port: int = 8080):
        """Start MCP server — any MCP client can use your retrieval."""
        from adaptive_intelligence.mcp import MCPServer
        server = MCPServer(self)
        server.serve(port=port)

    # ─── v4: MEMORY ────────────────────────────────────────

    def remember(self, key: str, value: Any, category: str = "fact"):
        """Store a fact in long-term memory."""
        if self._persistent_memory:
            self._persistent_memory.remember(key, value, category)

    def recall(self, key: str) -> Optional[Any]:
        """Recall a fact from long-term memory."""
        if self._persistent_memory:
            return self._persistent_memory.recall(key)
        return None

    def search_memory(self, query: str, top_k: int = 5) -> List[Any]:
        """Search long-term memory."""
        if self._persistent_memory:
            return self._persistent_memory.search(query, top_k=top_k)
        return []

    # ─── v2: FEEDBACK ──────────────────────────────────────

    def feedback(self, query_id: str, rating: str, reason: str = None):
        """Record user feedback as additional RL reward.

        Args:
            query_id: From response.query_id
            rating: "good" or "bad"
            reason: Optional explanation
        """
        cached = self._feedback_cache.get(query_id)
        if not cached:
            logger.warning(f"No cache found for query {query_id}")
            return

        feedback_reward = 0.2 if rating == "good" else -0.3
        adjusted_reward = max(0.0, min(1.0, cached["reward"] + feedback_reward))

        self.rl.update(
            query_id=query_id,
            query_analysis=cached["analysis"],
            action=cached["action"],
            reward=adjusted_reward,
        )

        if rating == "bad":
            template_type = cached["action"].prompt_template
            self.prompts.library.evolve_template(template_type, {
                "faithfulness": 0.4, "relevance": 0.4,
            })

        self.audit.log("user_feedback", {
            "rating": rating, "reason": reason,
            "adjusted_reward": adjusted_reward,
        }, query_id)

        self.memory.record_feedback(query_id, rating, reason)
        logger.info(f"Feedback recorded: {rating} for query {query_id}")

    # ─── v2: SYSTEM PROMPT ─────────────────────────────────

    def set_system_prompt(self, prompt: str):
        """Set custom system prompt for all future queries."""
        self._system_prompt = prompt
        logger.info(f"System prompt updated" if prompt else "System prompt reset")

    # ─── RETRIEVAL ─────────────────────────────────────────

    def _execute_retrieval(self, query: str, analysis: QueryAnalysis,
                           action: PolicyAction) -> List[Chunk]:
        depth = action.retrieval_depth
        route = action.retrieval_route

        if route == RetrievalRoute.VECTOR_ONLY and self.vector_index:
            results = self.vector_index.search(query, top_k=depth)
        elif route == RetrievalRoute.KEYWORD_ONLY:
            results = self.keyword_index.search(query, top_k=depth)
        elif route == RetrievalRoute.HYBRID and self.vector_index:
            results = self._hybrid_search(query, depth)
        elif route == RetrievalRoute.TABLE_FIRST:
            kw = self.keyword_index.search(query, top_k=depth)
            table_results = [r for r in kw if r.chunk.is_table]
            if len(table_results) < depth:
                extra = self.keyword_index.search(query, top_k=depth - len(table_results))
                results = table_results + extra
            else:
                results = table_results[:depth]
        elif route == RetrievalRoute.PAGE_BM25 and self.page_index:
            results = self.page_index.search(query, top_k=depth)
        elif route == RetrievalRoute.PAGE_GRAPH and self.page_index:
            results = self.page_index.search(query, top_k=depth)
        elif route in (RetrievalRoute.GRAPH_FIRST, RetrievalRoute.GRAPH_HYBRID):
            if self._vectorless and self.page_index:
                results = self.page_index.search(query, top_k=depth)
            elif self.vector_index:
                results = self._hybrid_search(query, depth)
            else:
                results = self.keyword_index.search(query, top_k=depth)
        else:
            # Fallback
            if self._vectorless and self.page_index:
                results = self.page_index.search(query, top_k=depth)
            elif self.vector_index:
                results = self._hybrid_search(query, depth)
            else:
                results = self.keyword_index.search(query, top_k=depth)

        seen = set()
        unique = []
        for r in results:
            if r.chunk.chunk_id not in seen:
                seen.add(r.chunk.chunk_id)
                unique.append(r.chunk)
        return unique[:depth]

    def _hybrid_search(self, query: str, top_k: int) -> List[RetrievalResult]:
        vec_results = self.vector_index.search(query, top_k=top_k) if self.vector_index else []
        kw_results = self.keyword_index.search(query, top_k=top_k)

        scores: Dict[str, float] = {}
        chunk_map: Dict[str, RetrievalResult] = {}

        for rank, r in enumerate(vec_results):
            cid = r.chunk.chunk_id
            scores[cid] = scores.get(cid, 0) + 1.0 / (rank + 60)
            chunk_map[cid] = r

        for rank, r in enumerate(kw_results):
            cid = r.chunk.chunk_id
            scores[cid] = scores.get(cid, 0) + 1.0 / (rank + 60)
            if cid not in chunk_map:
                chunk_map[cid] = r

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for cid, score in ranked[:top_k]:
            r = chunk_map[cid]
            r.score = score
            results.append(r)
        return results

    def _adjust_for_vectorless(self, action: PolicyAction) -> PolicyAction:
        """Map vector routes to page routes in vectorless mode."""
        route_map = {
            RetrievalRoute.VECTOR_ONLY: RetrievalRoute.PAGE_BM25,
            RetrievalRoute.HYBRID: RetrievalRoute.PAGE_BM25,
            RetrievalRoute.GRAPH_FIRST: RetrievalRoute.PAGE_GRAPH,
            RetrievalRoute.GRAPH_HYBRID: RetrievalRoute.PAGE_GRAPH,
        }
        if action.retrieval_route in route_map:
            action.retrieval_route = route_map[action.retrieval_route]
        return action

    # ─── v2: OUTPUT FORMATS ────────────────────────────────

    def _build_output_instructions(self, output_format: str, schema: Optional[Dict] = None) -> str:
        if output_format == "json":
            inst = "IMPORTANT: Return ONLY valid JSON. No markdown, no explanation, no code blocks."
            if schema:
                inst += f" Follow this exact schema: {json.dumps(schema)}"
            return inst
        elif output_format == "csv":
            return "IMPORTANT: Return ONLY CSV format with column headers on the first line. No markdown, no explanation."
        elif output_format == "yaml":
            return "IMPORTANT: Return ONLY valid YAML. No markdown, no explanation, no code blocks."
        elif output_format == "dataframe":
            return "IMPORTANT: Return data as CSV format with headers. No markdown, no explanation."
        return ""

    def _parse_structured_output(self, answer: str, output_format: str) -> Any:
        """Try to parse answer into structured format."""
        clean = answer.strip()
        # Strip markdown code blocks
        if clean.startswith("```"):
            lines = clean.split("\n")
            clean = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        if output_format == "json":
            try:
                return json.loads(clean)
            except json.JSONDecodeError:
                return None
        elif output_format in ("csv", "dataframe"):
            try:
                import pandas as pd
                from io import StringIO
                return pd.read_csv(StringIO(clean))
            except Exception:
                return clean
        elif output_format == "yaml":
            try:
                import yaml
                return yaml.safe_load(clean)
            except Exception:
                return None
        return None

    # ─── HELPERS ───────────────────────────────────────────

    def _fallback_answer(self, query: str, chunks: List[Chunk]) -> str:
        if not chunks:
            return "No relevant information found in the ingested documents for this query."

        # Build a clean, readable answer from retrieved chunks
        parts = []
        for i, chunk in enumerate(chunks[:5]):
            source = chunk.source_document or "document"
            page = f" (page {chunk.page_number})" if chunk.page_number else ""
            content = chunk.content.strip()

            # Clean up content - take meaningful portion
            if len(content) > 400:
                # Cut at sentence boundary
                cut = content[:400]
                last_period = cut.rfind(".")
                if last_period > 100:
                    content = cut[:last_period + 1]
                else:
                    content = cut + "..."

            parts.append(f"From {source}{page}:\n{content}")

        answer = "\n\n".join(parts)
        return answer

    def _extract_citations(self, answer: str, chunks: List[Chunk]) -> List[Citation]:
        citations = []
        for chunk in chunks[:10]:
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
                    page=chunk.page_number if chunk.page_number else None,
                ))
        citations.sort(key=lambda c: c.confidence, reverse=True)
        return citations[:5]

    # ─── v2: CRASH RECOVERY ────────────────────────────────

    def _recover_state(self):
        """Recover persisted state on startup."""
        bm25_path = self._storage_dir / "bm25_index.pkl"
        graph_path = self._storage_dir / "graph.json"
        page_path = self._storage_dir / "page_index.pkl"

        if bm25_path.exists():
            self.keyword_index.load(str(bm25_path))
            logger.info(f"BM25 recovered: {self.keyword_index.count()} entries")

        if graph_path.exists():
            self.graph.load(str(graph_path))
            logger.info(f"Graph recovered: {self.graph.node_count} nodes")

        if page_path.exists() and self.page_index:
            self.page_index.load(str(page_path))
            logger.info(f"Page index recovered: {self.page_index.count()} pages")

    def _checkpoint(self):
        """Save all state to disk."""
        try:
            self.keyword_index.save(str(self._storage_dir / "bm25_index.pkl"))
            self.graph.save(str(self._storage_dir / "graph.json"))
            if self.page_index:
                self.page_index.save(str(self._storage_dir / "page_index.pkl"))
            self.rl._save_state()
            self.memory._save_state()
            if self._persistent_memory:
                self._persistent_memory.save()
            self._last_checkpoint = time.time()
            logger.debug("Checkpoint saved")
        except Exception as e:
            logger.warning(f"Checkpoint failed: {e}")

    def _maybe_checkpoint(self):
        """Checkpoint if interval elapsed."""
        if time.time() - self._last_checkpoint >= self._checkpoint_interval:
            self._checkpoint()

    def _on_shutdown(self):
        """Graceful shutdown — save everything."""
        try:
            self._checkpoint()
            logger.info("Graceful shutdown: state saved")
        except Exception:
            pass

    # ─── v3: EXPORT / IMPORT / AB TEST ───────────────────

    def export_policy(self, filepath: str):
        """Export learned RL policy for transfer learning."""
        self.rl.export_policy(filepath)

    def import_policy(self, filepath: str):
        """Import RL policy from another deployment."""
        self.rl.import_policy(filepath)

    def enable_ab_test(self, policy_a: str = "thompson", policy_b: str = "ppo"):
        """Enable A/B testing between two policies."""
        self.rl.enable_ab_test()
        logger.info(f"A/B test: {policy_a} vs {policy_b}")

    def ab_results(self) -> Dict[str, Any]:
        """Get A/B test results."""
        return self.rl.ab_results()

    # ─── DASHBOARD ─────────────────────────────────────────

    def dashboard(self) -> str:
        eval_avg = self.evaluation.get_average_score()
        improvement = self.evaluation.get_improvement_rate()
        rl_stats = self.rl.get_stats()
        graph_stats = self.graph.get_stats()

        index_count = 0
        if self._vectorless and self.page_index:
            index_count = self.page_index.count()
        elif self.vector_index:
            index_count = self.vector_index.count()

        mode = "Vectorless" if self._vectorless else "Vector+Keyword"

        lines = [
            "+" + "-" * 57 + "+",
            "|  ADAPTIVE INTELLIGENCE v2 DASHBOARD                    |",
            "|" + " " * 57 + "|",
            f"|  Mode:                 {mode:>10}                    |",
            f"|  Documents Indexed:    {index_count:>6}                          |",
            f"|  Queries Processed:    {self._total_queries:>6}                          |",
            f"|  Average Accuracy:     {eval_avg:>5.1%}                          |",
            f"|  Improvement Rate:     {improvement:>+5.1%}                          |",
            "|" + " " * 57 + "|",
            f"|  RL Policy:            {'Warmup' if rl_stats['is_warmup'] else 'Active':>10}                    |",
            f"|  Exploration Rate:     {rl_stats['exploration_rate']:>5.1%}                          |",
            f"|  Arms Learned:         {rl_stats['total_arms']:>6}                          |",
            "|" + " " * 57 + "|",
            f"|  Graph Nodes:          {graph_stats['nodes']:>6}                          |",
            f"|  Graph Edges:          {graph_stats['edges']:>6}                          |",
            f"|  Graph Success Rate:   {graph_stats['activation_success_rate']:>5.1%}                          |",
            "+" + "-" * 57 + "+",
        ]
        return "\n".join(lines)

    def learning_curve(self) -> List[Dict[str, Any]]:
        return self.rl.get_learning_curve()

    def status(self) -> Dict[str, Any]:
        return {
            "version": "4.0.0",
            "vectorless": self._vectorless,
            "documents_indexed": self.page_index.count() if self._vectorless else (self.vector_index.count() if self.vector_index else 0),
            "total_queries": self._total_queries,
            "llm_provider": self._llm.provider_name if self._llm else "none",
            "llm_model": self.config.llm_model,
            "domain": self.config.domain.value,
            "rl_status": "warmup" if self.rl.is_warmup else "active",
            "graph_nodes": self.graph.node_count,
        }

    def reset(self):
        if self.vector_index:
            self.vector_index.clear()
        if self.page_index:
            self.page_index.clear()
        self.keyword_index.clear()
        self.graph.clear()
        self.memory.clear()
        self.audit.clear()
        self.ingestion.clear()
        self._is_ingested = False
        self._total_queries = 0
        self._feedback_cache.clear()
        logger.info("Engine reset complete")
