"""Healthcare analysis tools for adaptive-intelligence demo."""


def drug_interaction_checker(query, **kwargs):
    """Check drug interactions."""
    interactions = {
        ("metformin", "contrast"): {
            "severity": "HIGH",
            "risk": "Lactic acidosis",
            "action": "Hold Metformin 48h before and after contrast procedure",
        },
        ("sglt2", "diuretic"): {
            "severity": "MEDIUM",
            "risk": "Dehydration and hypotension",
            "action": "Monitor fluid intake and blood pressure closely",
        },
        ("empagliflozin", "insulin"): {
            "severity": "MEDIUM",
            "risk": "Hypoglycemia",
            "action": "May need insulin dose reduction 10-20%",
        },
        ("warfarin", "nsaid"): {
            "severity": "HIGH",
            "risk": "Increased bleeding",
            "action": "Avoid combination. Use acetaminophen instead.",
        },
        ("ace", "potassium"): {
            "severity": "MEDIUM",
            "risk": "Hyperkalemia",
            "action": "Monitor serum potassium every 2 weeks",
        },
        ("metformin", "alcohol"): {
            "severity": "MEDIUM",
            "risk": "Lactic acidosis",
            "action": "Limit alcohol intake, monitor symptoms",
        },
    }

    q = query.lower()
    found = []
    for (drug1, drug2), info in interactions.items():
        if drug1 in q or drug2 in q:
            found.append(
                f"  {drug1.upper()} + {drug2.upper()}: {info['severity']}\n"
                f"    Risk: {info['risk']}\n"
                f"    Action: {info['action']}"
            )

    if found:
        return "Drug interactions found:\n" + "\n".join(found)
    return "No known interactions found for query"


def dosage_calculator(query, **kwargs):
    """Calculate medication dosages."""
    q = query.lower()

    if "metformin" in q:
        return (
            "Metformin dosing:\n"
            "  Start: 500mg twice daily with meals\n"
            "  Titrate: Increase by 500mg weekly if tolerated\n"
            "  Target: 1000mg twice daily\n"
            "  Max: 2550mg/day\n"
            "  Renal: Reduce if eGFR 30-45, contraindicated if eGFR < 30"
        )

    if "empagliflozin" in q or "sglt2" in q:
        return (
            "Empagliflozin dosing:\n"
            "  Standard: 10mg once daily in the morning\n"
            "  Max: 25mg once daily\n"
            "  Renal: No adjustment if eGFR >= 30\n"
            "  Contraindicated: eGFR < 30 or dialysis\n"
            "  CV benefit: Shown at 10mg dose in EMPA-REG trial"
        )

    return f"No dosage information for: {query}"


def clinical_trial_lookup(query, **kwargs):
    """Look up clinical trial results."""
    q = query.lower()

    if "empagliflozin" in q or "nct-2025" in q or "cardiovascular" in q:
        return (
            "Trial NCT-2025-0847 Summary:\n"
            "  Drug: Empagliflozin\n"
            "  N: 4200 patients\n"
            "  Duration: 36 months\n"
            "  MACE reduction: 23% (p=0.003)\n"
            "  Heart failure hospitalization: -35% (p<0.001)\n"
            "  All-cause mortality: -12% (p=0.08, NS)\n"
            "  Key adverse: genital infections 8% vs 2%\n"
            "  NNT (MACE): ~18 patients over 3 years"
        )

    return "No matching clinical trial data"
