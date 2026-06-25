"""Context Engineering — v4 addition.

Optimizes the ENTIRE context window, not just chunk retrieval:
- System prompt selection per domain
- Memory inclusion (relevant long-term memory)
- History management (conversation context)
- Tool result assembly
- Chunk ordering and token budget allocation
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ContextWindow:
    """Complete assembled context ready for LLM."""
    system_prompt: str = ""
    memory_context: str = ""
    history_context: str = ""
    retrieved_chunks: str = ""
    tool_results: str = ""
    user_query: str = ""
    total_tokens: int = 0

    def assemble(self) -> str:
        parts = []
        if self.system_prompt:
            parts.append(self.system_prompt)
        if self.memory_context:
            parts.append(f"Relevant memory:\n{self.memory_context}")
        if self.history_context:
            parts.append(f"Previous context:\n{self.history_context}")
        if self.tool_results:
            parts.append(f"Tool results:\n{self.tool_results}")
        if self.retrieved_chunks:
            parts.append(f"Retrieved documents:\n{self.retrieved_chunks}")
        parts.append(f"Question: {self.user_query}")
        return "\n\n".join(parts)


@dataclass
class ContextBudget:
    """Token budget allocation per context component."""
    total: int = 4096
    system_prompt: int = 200
    memory: int = 300
    history: int = 400
    chunks: int = 2500
    tool_results: int = 500
    query: int = 196


class ContextEngineer:
    """Optimizes context window assembly."""

    PERSONAS = {
        "general": "You are an intelligent document analysis assistant. Answer based on provided context.",
        "financial": "You are a financial analyst. Cite exact figures, compare metrics, note trends.",
        "legal": "You are a legal reviewer. Reference specific clauses, flag compliance issues.",
        "healthcare": "You are a clinical data analyst. Use precise medical terminology.",
        "technical": "You are a technical documentation expert. Provide exact specifications.",
    }

    def __init__(self, token_budget: int = 4096):
        self.budget = ContextBudget(total=token_budget)

    def build_context(self, query: str, chunks: List[Any],
                      memory_entries: List[Any] = None,
                      session_history: List[Dict] = None,
                      tool_results: List[Dict] = None,
                      domain: str = "general",
                      custom_system_prompt: str = None) -> ContextWindow:
        ctx = ContextWindow(user_query=query)

        # System prompt
        ctx.system_prompt = custom_system_prompt or self.PERSONAS.get(
            domain, self.PERSONAS["general"]
        )

        # Memory — most relevant entries
        if memory_entries:
            parts = []
            budget = self.budget.memory * 4
            used = 0
            for entry in memory_entries[:5]:
                text = f"- {entry.key}: {entry.value}"
                if used + len(text) < budget:
                    parts.append(text)
                    used += len(text)
            ctx.memory_context = "\n".join(parts)

        # History — recent turns
        if session_history:
            parts = []
            budget = self.budget.history * 4
            used = 0
            for turn in reversed(session_history[-5:]):
                text = f"Q: {turn.get('query', '')}\nA: {turn.get('answer', '')[:200]}"
                if used + len(text) < budget:
                    parts.insert(0, text)
                    used += len(text)
            ctx.history_context = "\n---\n".join(parts)

        # Chunks — ordered by relevance, trimmed to budget
        if chunks:
            parts = []
            budget = self.budget.chunks * 4
            used = 0
            for chunk in chunks:
                content = chunk.content if hasattr(chunk, 'content') else str(chunk)
                source = chunk.source_document if hasattr(chunk, 'source_document') else ""
                page = f" (p{chunk.page_number})" if hasattr(chunk, 'page_number') and chunk.page_number else ""
                text = f"[{source}{page}]\n{content}"
                if used + len(text) < budget:
                    parts.append(text)
                    used += len(text)
            ctx.retrieved_chunks = "\n---\n".join(parts)

        # Tool results
        if tool_results:
            parts = []
            budget = self.budget.tool_results * 4
            used = 0
            for result in tool_results:
                text = f"[{result.get('tool', 'tool')}]: {result.get('result', '')}"
                if used + len(text) < budget:
                    parts.append(text)
                    used += len(text)
            ctx.tool_results = "\n".join(parts)

        ctx.total_tokens = len(ctx.assemble()) // 4
        return ctx

    def get_stats(self) -> Dict[str, Any]:
        return {
            "token_budget": self.budget.total,
            "personas": list(self.PERSONAS.keys()),
        }
