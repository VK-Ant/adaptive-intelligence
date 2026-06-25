"""Financial analysis tools for adaptive-intelligence demo."""


def revenue_calculator(query, **kwargs):
    """Calculate revenue metrics."""
    q = query.lower()
    results = []

    if "growth" in q or "compare" in q:
        q2, q3 = 798, 847
        growth = (q3 - q2) / q2 * 100
        results.append(f"Revenue growth Q2 to Q3: {growth:.1f}% (${q2}M to ${q3}M)")

    if "margin" in q:
        results.append("Operating margin: Q3 15.0% vs Q2 13.8% = +1.2pp improvement")
        results.append("EBITDA margin: Q3 19.8%")

    if "guidance" in q or "q4" in q:
        results.append("Q4 guidance: $870-890M revenue, 15.5-16.0% margin")
        midpoint = (870 + 890) / 2
        results.append(f"Q4 midpoint: ${midpoint:.0f}M")

    return "\n".join(results) if results else f"No financial data for: {query}"


def risk_scorer(query, **kwargs):
    """Score and analyze business risks."""
    risks = {
        "supply chain": {"level": "HIGH", "score": 0.85,
                         "detail": "65% Meridian dependency",
                         "mitigation": "Pacific Chip Alliance, target 45% by Q2 2026"},
        "cybersecurity": {"level": "MEDIUM", "score": 0.55,
                          "detail": "3 intrusion attempts Q2",
                          "mitigation": "CyberShield Partners, $12M zero-trust"},
        "regulatory": {"level": "MEDIUM", "score": 0.50,
                       "detail": "EU AI Act compliance needed",
                       "mitigation": "Estimated $8-12M cost"},
        "market": {"level": "MEDIUM", "score": 0.45,
                   "detail": "Share declined 28% to 26%",
                   "mitigation": "Product differentiation via NovaStar Edge"},
    }

    found = []
    q = query.lower()
    for name, data in risks.items():
        if name in q or "all" in q or "risk" in q:
            found.append(
                f"{name.upper()}: {data['level']} (score: {data['score']})\n"
                f"  Detail: {data['detail']}\n"
                f"  Mitigation: {data['mitigation']}"
            )

    if found:
        total_score = sum(risks[k]["score"] for k in risks) / len(risks)
        found.append(f"\nOverall risk score: {total_score:.2f}/1.00")
        return "\n".join(found)
    return "No matching risks found"


def cost_estimator(query, **kwargs):
    """Estimate costs and savings."""
    q = query.lower()
    if "token" in q or "cost" in q or "saving" in q:
        return (
            "Token cost analysis:\n"
            "  Static RAG: avg 10 chunks/query = ~2000 tokens context\n"
            "  Adaptive (factual): avg 2 chunks = ~400 tokens = 80% saving\n"
            "  Adaptive (complex): avg 8 chunks = ~1600 tokens = 20% saving\n"
            "  Weighted average: ~55% token reduction\n"
            "  At GPT-4o pricing ($2.50/1M input tokens):\n"
            "    1000 queries/day: $5.00 static vs $2.25 adaptive = $2.75/day saved\n"
            "    Monthly saving: ~$82"
        )
    return "Provide query about token costs or savings"
