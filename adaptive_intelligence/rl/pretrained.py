"""Pre-trained domain policies and transfer learning — v3.

Skip warmup entirely with pre-built policies for common domains.
Export/import policies between deployments.
"""

import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


# Pre-trained policies based on typical domain query patterns
PRETRAINED_POLICIES = {
    "financial": {
        "description": "Optimized for financial reports, earnings, filings",
        "arms": {
            "factual:simple:financial:keyword_only:5": {"alpha": 8.0, "beta": 2.0},
            "factual:simple:financial:table_first:5": {"alpha": 7.0, "beta": 2.5},
            "factual:moderate:financial:keyword_only:8": {"alpha": 6.0, "beta": 2.0},
            "analytical:moderate:financial:hybrid:8": {"alpha": 7.0, "beta": 3.0},
            "analytical:complex:financial:hybrid:10": {"alpha": 6.0, "beta": 2.5},
            "comparative:moderate:financial:hybrid:8": {"alpha": 8.0, "beta": 2.0},
            "comparative:complex:financial:hybrid:10": {"alpha": 7.0, "beta": 2.0},
            "relational:moderate:financial:graph_hybrid:10": {"alpha": 7.0, "beta": 2.5},
            "relational:complex:financial:graph_hybrid:15": {"alpha": 6.0, "beta": 2.0},
            "extraction:simple:financial:keyword_only:5": {"alpha": 8.0, "beta": 1.5},
            "structured_lookup:simple:financial:table_first:3": {"alpha": 9.0, "beta": 1.5},
        },
    },
    "legal": {
        "description": "Optimized for contracts, regulations, legal filings",
        "arms": {
            "factual:simple:legal:keyword_only:5": {"alpha": 7.0, "beta": 2.0},
            "factual:moderate:legal:keyword_only:8": {"alpha": 6.0, "beta": 2.0},
            "analytical:moderate:legal:hybrid:10": {"alpha": 6.0, "beta": 3.0},
            "analytical:complex:legal:hybrid:15": {"alpha": 5.0, "beta": 2.5},
            "relational:moderate:legal:graph_hybrid:10": {"alpha": 7.0, "beta": 2.0},
            "relational:complex:legal:graph_hybrid:15": {"alpha": 6.0, "beta": 2.0},
            "extraction:simple:legal:keyword_only:5": {"alpha": 8.0, "beta": 1.5},
            "comparative:moderate:legal:hybrid:10": {"alpha": 6.0, "beta": 2.5},
        },
    },
    "healthcare": {
        "description": "Optimized for medical records, clinical data, research",
        "arms": {
            "factual:simple:healthcare:keyword_only:5": {"alpha": 7.0, "beta": 2.0},
            "factual:moderate:healthcare:hybrid:8": {"alpha": 6.0, "beta": 2.5},
            "analytical:complex:healthcare:hybrid:10": {"alpha": 6.0, "beta": 3.0},
            "relational:moderate:healthcare:graph_hybrid:10": {"alpha": 7.0, "beta": 2.0},
            "relational:complex:healthcare:graph_hybrid:15": {"alpha": 6.0, "beta": 2.0},
            "extraction:simple:healthcare:keyword_only:5": {"alpha": 8.0, "beta": 1.5},
        },
    },
}


def get_pretrained_policy(domain: str) -> Optional[Dict[str, Any]]:
    """Get pre-trained policy for a domain."""
    return PRETRAINED_POLICIES.get(domain)


def export_policy(arms: Dict, filepath: str, metadata: Dict = None):
    """Export learned policy to file for transfer learning."""
    state = {
        "version": "3.0.0",
        "arms": {k: v.to_dict() if hasattr(v, 'to_dict') else v for k, v in arms.items()},
        "metadata": metadata or {},
    }
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(state, f, indent=2)
    logger.info(f"Policy exported: {len(arms)} arms to {filepath}")


def import_policy(filepath: str) -> Dict[str, Dict[str, Any]]:
    """Import policy from file."""
    with open(filepath) as f:
        state = json.load(f)
    arms = state.get("arms", {})
    logger.info(f"Policy imported: {len(arms)} arms from {filepath}")
    return arms
