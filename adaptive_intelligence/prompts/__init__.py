"""Adaptive Prompt Engine — Dynamic prompt construction that evolves.

Prompts are not static templates. They evolve based on evaluation
feedback: persona, context format, instructions, verification steps,
and output format all improve over time per query type.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from adaptive_intelligence.query import QueryAnalysis, QueryType
from adaptive_intelligence.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)


@dataclass
class PromptTemplate:
    """A single prompt template with performance tracking."""
    template_id: str
    persona: str
    instructions: str
    verification: str
    output_format: str
    domain_rules: str = ""
    score: float = 0.0
    usage_count: int = 0

    def render(self, query: str, context: str, graph_context: str = "") -> str:
        """Render the full prompt."""
        parts = []

        if self.persona:
            parts.append(f"You are {self.persona}.")

        if self.domain_rules:
            parts.append(f"\nDomain rules: {self.domain_rules}")

        if self.verification:
            parts.append(f"\nBefore answering: {self.verification}")

        parts.append(f"\n{self.instructions}")

        if self.output_format:
            parts.append(f"\nFormat: {self.output_format}")

        parts.append(f"\n\n--- Context from documents ---\n{context}")

        if graph_context:
            parts.append(f"\n\n--- Relationship graph ---\n{graph_context}")

        parts.append(f"\n\n--- Query ---\n{query}")

        return "\n".join(parts)


class PromptLibrary:
    """Stores and evolves prompt templates per query type."""

    def __init__(self):
        self._templates: Dict[str, List[PromptTemplate]] = {}
        self._best: Dict[str, PromptTemplate] = {}
        self._init_defaults()

    def _init_defaults(self):
        """Initialize default prompt templates."""
        defaults = {
            "extraction": PromptTemplate(
                template_id="extraction_v1",
                persona="a precise document analyst",
                instructions="Extract the specific information requested from the provided context. "
                             "Cite the source document for each fact. "
                             "If the information is not in the context, say so clearly.",
                verification="Verify that each claim is directly supported by the context.",
                output_format="Provide a clear, direct answer with source citations.",
            ),
            "analysis": PromptTemplate(
                template_id="analysis_v1",
                persona="a senior analyst with deep domain expertise",
                instructions="Analyze the provided information to answer the query comprehensively. "
                             "Consider multiple angles and synthesize findings across documents. "
                             "Identify patterns, risks, and implications.",
                verification="Cross-reference key findings across multiple sources. "
                             "Flag any contradictions or gaps in the data.",
                output_format="Structured analysis with key findings, supporting evidence, and implications.",
            ),
            "summary": PromptTemplate(
                template_id="summary_v1",
                persona="a concise and accurate summarizer",
                instructions="Summarize the key information from the provided context that relates "
                             "to the query. Prioritize the most important points. "
                             "Be comprehensive but concise.",
                verification="Ensure all major points are covered without fabrication.",
                output_format="A clear summary organized by importance.",
            ),
            "comparison": PromptTemplate(
                template_id="comparison_v1",
                persona="an objective analyst specializing in comparative analysis",
                instructions="Compare the items or concepts mentioned in the query using the "
                             "provided context. Identify similarities, differences, strengths, "
                             "and weaknesses for each.",
                verification="Verify each comparison point is grounded in the source documents.",
                output_format="Structured comparison with clear categories and cited evidence.",
            ),
            "verification": PromptTemplate(
                template_id="verification_v1",
                persona="a meticulous fact-checker",
                instructions="Verify the claim or information in the query against the provided "
                             "context. State whether it is supported, contradicted, or not "
                             "addressed by the sources.",
                verification="Double-check each source reference. Mark confidence level for each finding.",
                output_format="Verification result with [CONFIRMED], [CONTRADICTED], or [NOT FOUND] "
                              "for each claim, with supporting evidence.",
            ),
        }

        for key, template in defaults.items():
            self._templates[key] = [template]
            self._best[key] = template

    def get_template(self, template_type: str) -> PromptTemplate:
        """Get the best-performing template for a type."""
        if template_type in self._best:
            template = self._best[template_type]
            template.usage_count += 1
            return template
        # Fallback to extraction
        return self._best.get("extraction", list(self._templates.values())[0][0])

    def update_score(self, template_id: str, score: float):
        """Update a template's performance score."""
        for key, templates in self._templates.items():
            for t in templates:
                if t.template_id == template_id:
                    # Exponential moving average
                    alpha = 0.3
                    t.score = alpha * score + (1 - alpha) * t.score
                    # Update best if this is now the best
                    if t.score > self._best[key].score:
                        self._best[key] = t
                    logger.debug(f"Template {template_id} score updated: {t.score:.3f}")
                    return

    def evolve_template(self, template_type: str, feedback: Dict[str, float]) -> PromptTemplate:
        """Create an evolved version of a template based on feedback."""
        base = self.get_template(template_type)

        # Apply evolution rules based on feedback
        new_persona = base.persona
        new_instructions = base.instructions
        new_verification = base.verification
        new_output_format = base.output_format
        new_domain_rules = base.domain_rules

        if feedback.get("faithfulness", 1.0) < 0.7:
            new_verification = (
                "CRITICAL: Every claim must be directly traceable to a specific "
                "passage in the context. Never infer beyond what is stated. " +
                base.verification
            )

        if feedback.get("relevance", 1.0) < 0.7:
            new_instructions = (
                "Focus specifically on answering the exact question asked. "
                "Do not include tangential information. " + base.instructions
            )

        if feedback.get("hallucination_risk", 0.0) > 0.3:
            new_instructions += (
                " If you are uncertain about any fact, explicitly state your "
                "uncertainty rather than presenting it as fact."
            )

        if feedback.get("citation_accuracy", 1.0) < 0.6:
            new_output_format += (
                " Cite specific document names and page numbers for every factual claim."
            )

        version = len(self._templates.get(template_type, [])) + 1
        evolved = PromptTemplate(
            template_id=f"{template_type}_v{version}",
            persona=new_persona,
            instructions=new_instructions,
            verification=new_verification,
            output_format=new_output_format,
            domain_rules=new_domain_rules,
        )

        if template_type not in self._templates:
            self._templates[template_type] = []
        self._templates[template_type].append(evolved)

        logger.info(f"Evolved template: {evolved.template_id}")
        return evolved

    def get_domain_template(self, domain: str, query_type: str) -> PromptTemplate:
        """Get domain-specific template, creating one if needed."""
        key = f"{domain}_{query_type}"
        if key in self._best:
            return self.get_template(key)

        # Create domain-specialized template from base
        base = self.get_template(query_type)
        domain_rules = self._get_domain_rules(domain)

        specialized = PromptTemplate(
            template_id=f"{key}_v1",
            persona=f"{base.persona} specializing in {domain}",
            instructions=base.instructions,
            verification=base.verification,
            output_format=base.output_format,
            domain_rules=domain_rules,
        )

        self._templates[key] = [specialized]
        self._best[key] = specialized
        return specialized

    def _get_domain_rules(self, domain: str) -> str:
        """Get domain-specific rules."""
        rules = {
            "financial": (
                "Round numbers to appropriate precision. Include currency symbols. "
                "Distinguish between GAAP and non-GAAP metrics. Note fiscal year boundaries. "
                "Flag any forward-looking statements."
            ),
            "legal": (
                "Use precise legal terminology. Cite specific clauses or sections. "
                "Distinguish between binding and non-binding language. "
                "Note jurisdiction-specific variations."
            ),
            "healthcare": (
                "Use standard medical terminology with plain-language explanations. "
                "Note any HIPAA-relevant information boundaries. "
                "Distinguish between clinical evidence levels."
            ),
            "technical": (
                "Use precise technical terminology. Include version numbers and "
                "specifications where relevant. Note any deprecated features or "
                "compatibility requirements."
            ),
            "operational": (
                "Focus on actionable metrics. Include timeframes and responsible parties. "
                "Note dependencies between operational items."
            ),
        }
        return rules.get(domain, "")


class AdaptivePromptEngine:
    """Constructs optimal prompts that evolve based on performance."""

    def __init__(self):
        self.library = PromptLibrary()

    def build_prompt(self, query: str, query_analysis: QueryAnalysis,
                     chunks: List[Chunk], graph_context: str = "",
                     template_override: Optional[str] = None) -> str:
        """Build the optimal prompt for a query."""
        # Select template type
        template_type = template_override or self._select_template_type(query_analysis)

        # Get domain-specific template if domain is detected
        if query_analysis.domain != "general":
            template = self.library.get_domain_template(
                query_analysis.domain, template_type
            )
        else:
            template = self.library.get_template(template_type)

        # Build context from chunks
        context = self._format_context(chunks, query_analysis)

        # Render prompt
        prompt = template.render(query, context, graph_context)

        logger.debug(f"Built prompt using template: {template.template_id}")
        return prompt

    def update_from_evaluation(self, template_id: str, evaluation_scores: Dict[str, float]):
        """Update prompt template based on evaluation feedback."""
        composite = evaluation_scores.get("composite_score", 0.5)
        self.library.update_score(template_id, composite)

        # Evolve if performance is poor
        if composite < 0.7:
            template_type = template_id.rsplit("_v", 1)[0]
            self.library.evolve_template(template_type, evaluation_scores)

    def _select_template_type(self, analysis: QueryAnalysis) -> str:
        """Select the best template type based on query analysis."""
        type_map = {
            QueryType.FACTUAL: "extraction",
            QueryType.RELATIONAL: "analysis",
            QueryType.ANALYTICAL: "analysis",
            QueryType.SUMMARIZATION: "summary",
            QueryType.EXTRACTION: "extraction",
            QueryType.COMPARATIVE: "comparison",
            QueryType.REASONING: "analysis",
            QueryType.STRUCTURED_LOOKUP: "extraction",
        }
        return type_map.get(analysis.query_type, "extraction")

    def _format_context(self, chunks: List[Chunk], analysis: QueryAnalysis) -> str:
        """Format retrieved chunks into context string."""
        if not chunks:
            return "[No relevant context found]"

        parts = []
        for i, chunk in enumerate(chunks):
            source = chunk.source_document or "Unknown"
            prefix = f"[Source {i+1}: {source}]"
            if chunk.is_table:
                prefix += " [TABLE]"
            parts.append(f"{prefix}\n{chunk.content}")

        return "\n\n---\n\n".join(parts)
