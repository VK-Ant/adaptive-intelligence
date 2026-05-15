"""
Adaptive Intelligence — Quickstart Example

This example shows the simplest way to use Adaptive Intelligence:
3 lines to start, with the system learning from every query.

Prerequisites:
    pip install adaptive-intelligence
    # For local LLM (free):
    # Install Ollama (https://ollama.ai) and run: ollama pull llama3.2
"""

from adaptive_intelligence import AdaptiveAI

# ── Step 1: Initialize ──────────────────────────────────────────────
engine = AdaptiveAI()
# That's it. Defaults to Ollama (free, local, private).
# For OpenAI: AdaptiveAI(llm_backend="openai", api_key="sk-...")

# ── Step 2: Ingest documents ────────────────────────────────────────
stats = engine.ingest("./documents")  # File, directory, or list of paths
print(stats.display())
# Supports: PDF, DOCX, XLSX, PPTX, CSV, JSON, HTML, TXT, images (OCR)

# ── Step 3: Ask questions ───────────────────────────────────────────
response = engine.ask("What are the key operational risks?")

print(response.answer)           # The synthesized answer
print(f"Confidence: {response.confidence:.0%}")  # How confident the system is
print(f"Strategy: {response.retrieval_strategy}")  # What retrieval strategy was used

# ── Detailed evaluation ─────────────────────────────────────────────
print(response.evaluation.display())
# Shows: faithfulness, relevance, hallucination risk, citations, etc.

# ── Citations ────────────────────────────────────────────────────────
for citation in response.citations:
    print(f"  Source: {citation.source_document} (confidence: {citation.confidence:.0%})")

# ── Full formatted display ──────────────────────────────────────────
print(response.display())

# ── System learns over time ─────────────────────────────────────────
# Each query improves retrieval strategy through RL feedback.
# The more you ask, the better it gets.

# Ask more questions — the system learns from each one:
for query in [
    "What is the total revenue for Q3?",
    "Compare Q1 and Q2 performance",
    "How is Acme Corp connected to Beta Ltd?",
    "Summarize the financial highlights",
]:
    r = engine.ask(query)
    print(f"\nQ: {query}")
    print(f"A: {r.answer[:200]}...")
    print(f"Confidence: {r.confidence:.0%} | Strategy: {r.retrieval_strategy}")

# ── Dashboard ────────────────────────────────────────────────────────
print(engine.dashboard())
# Shows: accuracy, improvement rate, RL status, graph stats, routing patterns
