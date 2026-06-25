"""
Demo 3: Agentic Multi-Round Retrieval
======================================

Shows:
- System retrieves, evaluates confidence, retrieves again if needed
- Query refinement between rounds
- Tool calls integrated into agentic loop
- Final answer synthesized from accumulated context

Run: python demo_agentic.py
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from adaptive_intelligence import AdaptiveAI
from tools.financial_tools import revenue_calculator, risk_scorer
from tools.healthcare_tools import drug_interaction_checker, clinical_trial_lookup


def main():
    print("=" * 60)
    print("DEMO 3: Agentic Multi-Round Retrieval")
    print("=" * 60)

    engine = AdaptiveAI(
        llm_backend="none", vectorless=True,
        storage_dir="./demo_state_agentic",
        log_level="ERROR",
    )

    engine.ingest("./data/financial")
    engine.ingest("./data/healthcare")

    # Register tools
    engine.add_tool("risk_scorer", description="risk scoring supply chain",
                     function=risk_scorer)
    engine.add_tool("revenue_calc", description="revenue growth calculation",
                     function=revenue_calculator)
    engine.add_tool("drug_checker", description="drug interaction checker",
                     function=drug_interaction_checker)
    engine.add_tool("trial_lookup", description="clinical trial results",
                     function=clinical_trial_lookup)

    # --- Agentic: Financial scenario ---
    print("\n--- Scenario 1: Financial deep analysis ---")
    print("Query: Complete supply chain risk analysis with mitigation and Q4 impact")
    print()

    r = engine.ask(
        "Analyze the complete supply chain risk: Meridian dependency, "
        "Pacific Chip mitigation timeline, and impact on Q4 guidance",
        mode="agentic"
    )

    print(f"Answer:\n{r.answer[:500]}")
    print(f"\nAgent details:")
    if hasattr(r, "agent_rounds"):
        print(f"  Rounds: {r.agent_rounds}")
    if hasattr(r, "agent_steps"):
        for i, step in enumerate(r.agent_steps):
            q = f" - {step.query[:50]}..." if step.query else ""
            t = f" [{step.tool_name}]" if step.tool_name else ""
            print(f"  Step {i+1}: {step.action}{t}{q} ({step.latency:.2f}s)")
    if hasattr(r, "tools_called") and r.tools_called:
        print(f"  Tools used: {r.tools_called}")

    # --- Agentic: Healthcare scenario ---
    print("\n\n--- Scenario 2: Healthcare multi-hop ---")
    print("Query: Empagliflozin safety for a patient on Metformin with eGFR 40")
    print()

    r = engine.ask(
        "Is Empagliflozin safe for a Type 2 Diabetes patient already on "
        "Metformin with eGFR 40? Check drug interactions, dosing, and trial data.",
        mode="agentic"
    )

    print(f"Answer:\n{r.answer[:500]}")
    print(f"\nAgent details:")
    if hasattr(r, "agent_rounds"):
        print(f"  Rounds: {r.agent_rounds}")
    if hasattr(r, "agent_steps"):
        for i, step in enumerate(r.agent_steps):
            q = f" - {step.query[:50]}..." if step.query else ""
            t = f" [{step.tool_name}]" if step.tool_name else ""
            print(f"  Step {i+1}: {step.action}{t}{q} ({step.latency:.2f}s)")

    # --- Compare: Standard vs Agentic ---
    print("\n\n--- Comparison: Standard vs Agentic ---")

    query = "What is the Meridian supply chain risk and how does Pacific Chip mitigate it?"

    # Standard
    r1 = engine.ask(query)
    # Agentic
    r2 = engine.ask(query, mode="agentic")

    print(f"Query: {query}")
    print(f"\nStandard (1 round):")
    print(f"  Answer: {r1.answer[:200]}...")
    print(f"  Strategy: {r1.retrieval_strategy}")

    print(f"\nAgentic ({r2.agent_rounds if hasattr(r2, 'agent_rounds') else '?'} rounds):")
    print(f"  Answer: {r2.answer[:200]}...")
    if hasattr(r2, "tools_called") and r2.tools_called:
        print(f"  Tools: {r2.tools_called}")

    print("\nAgentic mode finds more context through multiple rounds and tool calls.")
    print("Use standard mode for simple queries. Agentic for complex analysis.")


if __name__ == "__main__":
    main()
