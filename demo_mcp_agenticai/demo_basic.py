"""
Demo 1: Basic Usage + Incremental Learning
===========================================

Shows:
- Ingest documents, ask questions
- Add MORE documents later without restart
- RL policy, graph, memory all preserved

Run: python demo_basic.py
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from adaptive_intelligence import AdaptiveAI


def main():
    print("=" * 60)
    print("DEMO 1: Basic Usage + Incremental Learning")
    print("=" * 60)

    # --- Phase 1: Financial documents ---
    print("\n--- Phase 1: Ingest financial documents ---")
    engine = AdaptiveAI(
        llm_backend="huggingface",
        llm_model="Qwen/Qwen2.5-1.5B-Instruct",
        domain="financial",
        storage_dir="./demo_state",
        log_level="ERROR",
    )

    stats = engine.ingest("./data/financial")
    print(f"Ingested: {stats.successful} docs, {stats.total_chunks} chunks")
    print(f"Graph: {engine.graph.node_count} nodes, {engine.graph.edge_count} edges")

    # Ask financial questions
    queries = [
        "What is Q3 revenue?",
        "How is Meridian connected to supply chain risk?",
        "Compare Q2 vs Q3 revenue",
    ]

    print(f"\n{'Strategy':<30} {'Conf':<6} Query")
    print("-" * 70)
    for q in queries:
        r = engine.ask(q)
        print(f"{r.retrieval_strategy:<30} {r.confidence:.0%}    {q}")
        print(f"  Answer: {r.answer[:150]}...")
        print()

    # --- Phase 2: Add healthcare docs WITHOUT restart ---
    print("\n--- Phase 2: Add healthcare documents (no restart) ---")

    graph_before = engine.graph.node_count
    stats2 = engine.ingest("./data/healthcare")

    print(f"Added: {stats2.total_chunks} chunks")
    print(f"Graph grew: {graph_before} -> {engine.graph.node_count} nodes")
    print("RL policy PRESERVED — continues from current state")

    # Ask healthcare questions on same engine
    healthcare_queries = [
        "What is Metformin dosage?",
        "Empagliflozin trial results?",
    ]

    print(f"\n{'Strategy':<30} {'Conf':<6} Query")
    print("-" * 70)
    for q in healthcare_queries:
        r = engine.ask(q)
        print(f"{r.retrieval_strategy:<30} {r.confidence:.0%}    {q}")
        print(f"  Answer: {r.answer[:150]}...")
        print()

    print("\nKey point: Added healthcare docs to existing financial docs.")
    print("No restart. No re-training. RL continued learning seamlessly.")


if __name__ == "__main__":
    main()
