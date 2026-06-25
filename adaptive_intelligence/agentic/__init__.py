"""Agentic Workflow — v4 addition.

Multi-round retrieval where the LLM can:
- Request additional context
- Refine the query
- Call external tools
- Decide when it has enough information to answer

The RL policy guides initial retrieval, the agent refines.
"""

import json
import logging
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class AgentStep:
    """Single step in an agentic workflow."""
    action: str  # "retrieve", "tool_call", "refine", "answer"
    query: str = ""
    tool_name: str = ""
    result: str = ""
    reasoning: str = ""
    latency: float = 0.0


@dataclass
class AgentResult:
    """Complete result from an agentic workflow."""
    answer: str = ""
    steps: List[AgentStep] = field(default_factory=list)
    total_rounds: int = 0
    total_latency: float = 0.0
    tools_called: List[str] = field(default_factory=list)
    context_assembled: str = ""


class AgenticWorkflow:
    """Multi-round retrieval with LLM-driven refinement.

    Flow:
    1. Initial RL-guided retrieval
    2. LLM evaluates if context is sufficient
    3. If not: LLM suggests refinement (new query, tool call, more depth)
    4. Execute refinement
    5. Repeat until sufficient or max rounds
    6. Final answer generation

    Works with any LLM. Falls back to single-round if no LLM available.
    """

    def __init__(self, max_rounds: int = 3, confidence_threshold: float = 0.7):
        self.max_rounds = max_rounds
        self.confidence_threshold = confidence_threshold

    def run(self, query: str, engine, tool_registry=None) -> AgentResult:
        """Execute agentic workflow.

        Args:
            query: User question
            engine: AdaptiveAI engine instance
            tool_registry: Optional ToolRegistry for external tools
        """
        result = AgentResult()
        start = time.time()
        accumulated_context = []
        current_query = query

        for round_num in range(self.max_rounds):
            step_start = time.time()

            # Round 1: Use RL-guided retrieval
            response = engine.ask(current_query)
            accumulated_context.append(response.answer)

            step = AgentStep(
                action="retrieve",
                query=current_query,
                result=response.answer[:500],
                latency=time.time() - step_start,
            )
            result.steps.append(step)

            # Check if we have enough confidence
            if response.confidence >= self.confidence_threshold:
                result.answer = response.answer
                break

            # Check if tools should be called
            if tool_registry and round_num < self.max_rounds - 1:
                suggested_tools = tool_registry.select_tools(current_query)
                for tool_name in suggested_tools[:2]:
                    tool_start = time.time()
                    tool_result = tool_registry.call_tool(tool_name, current_query)
                    if tool_result.success and tool_result.result:
                        accumulated_context.append(tool_result.result)
                        result.tools_called.append(tool_name)
                        result.steps.append(AgentStep(
                            action="tool_call",
                            tool_name=tool_name,
                            result=tool_result.result[:300],
                            latency=time.time() - tool_start,
                        ))

            # Try query refinement for next round
            if round_num < self.max_rounds - 1:
                refined = self._refine_query(query, current_query, accumulated_context)
                if refined and refined != current_query:
                    current_query = refined
                    result.steps.append(AgentStep(
                        action="refine",
                        query=refined,
                        reasoning="Refined based on accumulated context",
                    ))
                else:
                    # No refinement possible, use what we have
                    result.answer = response.answer
                    break
        else:
            # Max rounds reached, synthesize from accumulated context
            result.answer = response.answer if response else ""

        if not result.answer and accumulated_context:
            result.answer = "\n\n".join(accumulated_context[:3])

        result.total_rounds = len([s for s in result.steps if s.action == "retrieve"])
        result.total_latency = time.time() - start
        result.context_assembled = "\n---\n".join(accumulated_context[:5])

        return result

    def _refine_query(self, original: str, current: str,
                      context: List[str]) -> Optional[str]:
        """Generate a refined query based on what we've found so far.

        Uses rule-based refinement (no LLM call needed).
        """
        if not context:
            return None

        # Extract entities mentioned in context but not in query
        context_text = " ".join(context).lower()
        query_terms = set(original.lower().split())

        # Find new terms that appear frequently in context
        context_words = context_text.split()
        word_freq = {}
        for w in context_words:
            if len(w) > 4 and w not in query_terms:
                word_freq[w] = word_freq.get(w, 0) + 1

        # Get top new terms
        new_terms = sorted(word_freq.items(), key=lambda x: -x[1])[:3]
        new_terms = [t for t, c in new_terms if c >= 2]

        if new_terms:
            refined = f"{original} {' '.join(new_terms)}"
            return refined

        return None

    def run_simple(self, query: str, engine) -> AgentResult:
        """Single-round execution (no LLM, no tools).

        Falls back to standard engine.ask() with agentic metadata.
        """
        start = time.time()
        response = engine.ask(query)
        return AgentResult(
            answer=response.answer,
            steps=[AgentStep(
                action="retrieve", query=query,
                result=response.answer[:500],
                latency=time.time() - start,
            )],
            total_rounds=1,
            total_latency=time.time() - start,
        )
