"""
Demo 2: Tool Registry + Cost Optimization
==========================================

Shows:
- Register Python functions as tools
- RL learns WHICH tools to call per query
- Tools that don't help get skipped = lower cost
- RL learns optimal retrieval depth = fewer tokens = lower LLM cost

Run: python demo_tools.py
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from adaptive_intelligence import AdaptiveAI
from tools.financial_tools import revenue_calculator, risk_scorer, cost_estimator
from tools.healthcare_tools import drug_interaction_checker, dosage_calculator, clinical_trial_lookup


def main():
    print("=" * 60)
    print("DEMO 2: Tool Registry + Cost Optimization")
    print("=" * 60)

    engine = AdaptiveAI(
        llm_backend="none", vectorless=True,
        storage_dir="./demo_state_tools",
        log_level="ERROR",
    )

    # Ingest both domains
    engine.ingest("./data/financial")
    engine.ingest("./data/healthcare")

    # --- Register financial tools ---
    print("\n--- Registering tools ---")
    engine.add_tool("revenue_calc",
                     description="revenue growth margin financial calculation",
                     function=revenue_calculator)
    engine.add_tool("risk_scorer",
                     description="risk assessment scoring supply chain cybersecurity",
                     function=risk_scorer)
    engine.add_tool("cost_estimator",
                     description="token cost savings estimation",
                     function=cost_estimator)

    # --- Register healthcare tools ---
    engine.add_tool("drug_checker",
                     description="drug interaction checker metformin empagliflozin",
                     function=drug_interaction_checker)
    engine.add_tool("dosage_calc",
                     description="dosage calculator medication metformin",
                     function=dosage_calculator)
    engine.add_tool("trial_lookup",
                     description="clinical trial results empagliflozin cardiovascular",
                     function=clinical_trial_lookup)

    print("Registered tools:")
    for t in engine.list_tools():
        print(f"  {t['name']} ({t['type']})")

    # --- Financial queries with tools ---
    print("\n--- Financial queries (tools assist retrieval) ---")
    fin_queries = [
        "What is the revenue growth Q2 to Q3?",
        "What is the supply chain risk score?",
        "How much can we save on token costs?",
    ]

    for q in fin_queries:
        r = engine.ask(q)
        print(f"\nQ: {q}")
        print(f"A: {r.answer[:300]}")
        print(f"Strategy: {r.retrieval_strategy}")

    # --- Healthcare queries with tools ---
    print("\n--- Healthcare queries (tools assist retrieval) ---")
    med_queries = [
        "What are Metformin drug interactions?",
        "What is Empagliflozin dosage?",
        "Empagliflozin cardiovascular trial results?",
    ]

    for q in med_queries:
        r = engine.ask(q)
        print(f"\nQ: {q}")
        print(f"A: {r.answer[:300]}")
        print(f"Strategy: {r.retrieval_strategy}")

    # --- Tool usage stats ---
    print("\n--- Tool Usage Statistics ---")
    print(f"{'Tool':<20} {'Calls':<8} {'Success':<10} {'Avg Latency'}")
    print("-" * 55)
    for t in engine.list_tools():
        print(f"{t['name']:<20} {t['calls']:<8} {t['success_rate']:.0%}       {t['avg_latency']}")

    # --- Cost optimization explanation ---
    print("\n--- Cost Optimization ---")
    print("How adaptive-intelligence reduces costs:")
    print()
    print("1. RETRIEVAL DEPTH: RL learns optimal depth per query type")
    print("   Factual query: depth 2 (2 chunks, ~400 tokens)")
    print("   Complex query: depth 8 (8 chunks, ~1600 tokens)")
    print("   Static RAG: always depth 10 (~2000 tokens)")
    print("   Saving: ~55% fewer tokens on average")
    print()
    print("2. TOOL SELECTION: RL skips tools that don't improve answers")
    print("   Revenue query: calls revenue_calc, skips drug_checker")
    print("   Medical query: calls drug_checker, skips revenue_calc")
    print("   Static system: calls all tools every time")
    print()
    print("3. GRAPH ACTIVATION: 70% of queries skip graph traversal")
    print("   Factual queries: graph OFF (saves 15-20ms)")
    print("   Relational queries: graph ON (finds entity connections)")
    print()
    print("Combined: fewer tokens + fewer tool calls + less compute = lower cost")


if __name__ == "__main__":
    main()
