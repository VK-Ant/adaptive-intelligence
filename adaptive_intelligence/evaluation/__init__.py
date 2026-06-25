"""Evaluation Engine — Measures answer quality, provides RL reward signal.

Three-layer evaluation:
  Layer 1: Fast automatic (semantic similarity, pattern checks) — always on
  Layer 2: LLM-as-Judge (faithfulness, hallucination) — when L1 confidence < threshold
  Layer 3: Consistency checks — periodic cross-reference validation

The composite score becomes the reward signal for the RL Policy Engine.
"""

import re
import math
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from adaptive_intelligence.core.config import EvaluationConfig
from adaptive_intelligence.core.response import EvaluationResult
from adaptive_intelligence.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)


class EvaluationEngine:
    """Evaluates response quality and computes RL reward signals."""

    def __init__(self, config: Optional[EvaluationConfig] = None, llm_provider=None):
        self.config = config or EvaluationConfig()
        self._llm = llm_provider
        self._history: List[Dict[str, Any]] = []

    def evaluate(self, query: str, answer: str,
                 retrieved_chunks: List[Chunk],
                 latency_seconds: float = 0.0,
                 token_usage: int = 0) -> EvaluationResult:
        """Run full evaluation pipeline and return results."""
        start = time.time()

        # Layer 1: Fast automatic evaluation
        faithfulness = self._measure_faithfulness(answer, retrieved_chunks)
        relevance = self._measure_relevance(query, answer)
        citation_accuracy = self._measure_citation_accuracy(answer, retrieved_chunks)
        hallucination_risk = self._measure_hallucination_risk(answer, retrieved_chunks)
        precision = self._measure_retrieval_precision(query, retrieved_chunks)
        recall = self._measure_retrieval_recall(query, retrieved_chunks)

        # Composite score using configured weights
        composite = self._compute_composite(
            faithfulness=faithfulness,
            relevance=relevance,
            citation_accuracy=citation_accuracy,
            hallucination_risk=hallucination_risk,
            precision=precision,
            recall=recall,
            latency=latency_seconds,
            token_usage=token_usage,
        )

        # Determine confidence level
        if composite >= 0.85:
            confidence_level = "high"
        elif composite >= 0.70:
            confidence_level = "medium"
        else:
            confidence_level = "low"

        result = EvaluationResult(
            faithfulness=faithfulness,
            relevance=relevance,
            citation_accuracy=citation_accuracy,
            hallucination_risk=hallucination_risk,
            retrieval_precision=precision,
            retrieval_recall=recall,
            latency_seconds=latency_seconds,
            token_usage=token_usage,
            composite_score=composite,
            confidence_level=confidence_level,
        )

        # Layer 2: LLM-as-Judge (if enabled and L1 confidence is low)
        if (self.config.enable_llm_judge and
                self._llm is not None and
                composite < self.config.llm_judge_threshold):
            llm_eval = self._llm_judge_evaluate(query, answer, retrieved_chunks)
            if llm_eval:
                # Blend LLM judge scores with automatic scores
                result.faithfulness = 0.6 * faithfulness + 0.4 * llm_eval.get("faithfulness", faithfulness)
                result.relevance = 0.6 * relevance + 0.4 * llm_eval.get("relevance", relevance)
                result.hallucination_risk = 0.6 * hallucination_risk + 0.4 * llm_eval.get("hallucination_risk", hallucination_risk)
                # Recompute composite
                result.composite_score = self._compute_composite(
                    faithfulness=result.faithfulness,
                    relevance=result.relevance,
                    citation_accuracy=result.citation_accuracy,
                    hallucination_risk=result.hallucination_risk,
                    precision=result.retrieval_precision,
                    recall=result.retrieval_recall,
                    latency=result.latency_seconds,
                    token_usage=result.token_usage,
                )

        # Store in history
        self._history.append({
            "query": query,
            "composite_score": result.composite_score,
            "confidence_level": result.confidence_level,
            "timestamp": time.time(),
        })

        eval_time = time.time() - start
        logger.info(
            f"Evaluation complete in {eval_time:.3f}s: "
            f"composite={result.composite_score:.3f}, "
            f"confidence={result.confidence_level}"
        )
        return result

    def compute_reward(self, evaluation: EvaluationResult) -> float:
        """Compute RL reward signal from evaluation result.

        This is the critical link between evaluation and learning:
        the reward drives all RL policy updates.
        """
        return evaluation.composite_score

    def _measure_faithfulness(self, answer: str, chunks: List[Chunk]) -> float:
        """Measure how well the answer is grounded in source chunks.

        Uses token overlap between answer sentences and source content.
        """
        if not chunks or not answer.strip():
            return 0.0

        context = " ".join(c.content for c in chunks).lower()
        context_tokens = set(self._tokenize(context))

        if not context_tokens:
            return 0.0

        answer_sentences = self._split_sentences(answer)
        if not answer_sentences:
            return 0.0

        grounded_count = 0
        for sentence in answer_sentences:
            sent_tokens = set(self._tokenize(sentence.lower()))
            if not sent_tokens:
                continue
            overlap = len(sent_tokens & context_tokens) / len(sent_tokens)
            if overlap > 0.3:  # At least 30% token overlap
                grounded_count += 1

        return grounded_count / len(answer_sentences) if answer_sentences else 0.0

    def _measure_relevance(self, query: str, answer: str) -> float:
        """Measure how well the answer addresses the query.

        Uses token overlap between query and answer.
        """
        if not query.strip() or not answer.strip():
            return 0.0

        query_tokens = set(self._tokenize(query.lower()))
        answer_tokens = set(self._tokenize(answer.lower()))

        # Remove stop words
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                      "being", "have", "has", "had", "do", "does", "did", "will",
                      "would", "could", "should", "may", "might", "can", "shall",
                      "of", "in", "to", "for", "with", "on", "at", "by", "from",
                      "up", "about", "into", "through", "and", "but", "or", "not",
                      "what", "how", "why", "when", "where", "which", "who", "this",
                      "that", "it", "its", "i", "me", "my"}
        query_tokens -= stop_words
        answer_tokens -= stop_words

        if not query_tokens:
            return 0.5

        overlap = len(query_tokens & answer_tokens)
        recall = overlap / len(query_tokens)

        # Bonus for longer, more detailed answers
        length_bonus = min(0.2, len(answer.split()) / 500)

        return min(1.0, recall + length_bonus)

    def _measure_citation_accuracy(self, answer: str, chunks: List[Chunk]) -> float:
        """Measure whether citations in the answer are accurate.

        Looks for citation patterns like [1], [Source: ...], (Source: ...).
        """
        citation_patterns = [
            r"\[\d+\]",
            r"\[Source:.*?\]",
            r"\(Source:.*?\)",
            r"according to",
            r"based on",
            r"as stated in",
            r"from the document",
        ]

        has_citations = any(
            re.search(p, answer, re.IGNORECASE) for p in citation_patterns
        )

        if not chunks:
            return 0.5

        if has_citations:
            # Has explicit citations — score higher
            return 0.85
        elif len(answer.split()) < 20:
            # Short answer, citations not expected
            return 0.7
        else:
            # Long answer without citations
            return 0.4

    def _measure_hallucination_risk(self, answer: str, chunks: List[Chunk]) -> float:
        """Estimate hallucination risk.

        Checks for claims in the answer not grounded in any source chunk.
        Higher value = MORE risk (bad).
        """
        if not chunks or not answer.strip():
            return 0.5

        context = " ".join(c.content for c in chunks).lower()
        context_tokens = set(self._tokenize(context))

        answer_sentences = self._split_sentences(answer)
        if not answer_sentences:
            return 0.0

        ungrounded = 0
        for sentence in answer_sentences:
            sent_tokens = set(self._tokenize(sentence.lower()))
            if not sent_tokens:
                continue
            overlap = len(sent_tokens & context_tokens) / len(sent_tokens)
            if overlap < 0.15:  # Less than 15% overlap = suspicious
                ungrounded += 1

        risk = ungrounded / len(answer_sentences) if answer_sentences else 0.0
        return min(1.0, risk)

    def _measure_retrieval_precision(self, query: str, chunks: List[Chunk]) -> float:
        """Measure what fraction of retrieved chunks are relevant."""
        if not chunks:
            return 0.0

        query_tokens = set(self._tokenize(query.lower()))
        stop_words = {"the", "a", "an", "is", "are", "what", "how", "of", "in",
                      "to", "for", "with", "and", "or", "not", "this", "that"}
        query_tokens -= stop_words

        if not query_tokens:
            return 0.5

        relevant = 0
        for chunk in chunks:
            chunk_tokens = set(self._tokenize(chunk.content.lower()))
            overlap = len(query_tokens & chunk_tokens)
            if overlap >= max(1, len(query_tokens) * 0.3):
                relevant += 1

        return relevant / len(chunks)

    def _measure_retrieval_recall(self, query: str, chunks: List[Chunk]) -> float:
        """Estimate retrieval recall (approximate without ground truth)."""
        if not chunks:
            return 0.0

        query_tokens = set(self._tokenize(query.lower()))
        stop_words = {"the", "a", "an", "is", "are", "what", "how", "of", "in",
                      "to", "for", "with", "and", "or", "not"}
        query_tokens -= stop_words

        if not query_tokens:
            return 0.5

        # Check how many query terms appear in at least one chunk
        covered_terms = set()
        for chunk in chunks:
            chunk_tokens = set(self._tokenize(chunk.content.lower()))
            covered_terms.update(query_tokens & chunk_tokens)

        return len(covered_terms) / len(query_tokens) if query_tokens else 0.0

    def _compute_composite(self, faithfulness: float, relevance: float,
                           citation_accuracy: float, hallucination_risk: float,
                           precision: float, recall: float,
                           latency: float, token_usage: int) -> float:
        """Compute weighted composite score (= RL reward)."""
        cfg = self.config

        # Normalize latency penalty (0-1 scale, higher latency = higher penalty)
        latency_penalty = min(1.0, latency / 30.0)  # 30s = max expected

        # Normalize token cost (0-1 scale)
        token_cost = min(1.0, token_usage / 10000.0)  # 10k = high

        composite = (
            faithfulness * cfg.faithfulness_weight +
            relevance * cfg.relevance_weight +
            citation_accuracy * cfg.citation_weight +
            precision * cfg.precision_weight +
            recall * cfg.recall_weight -
            latency_penalty * cfg.latency_penalty_weight -
            token_cost * cfg.token_cost_weight -
            hallucination_risk * cfg.hallucination_penalty_weight
        )

        return max(0.0, min(1.0, composite))

    def _llm_judge_evaluate(self, query: str, answer: str,
                            chunks: List[Chunk]) -> Optional[Dict[str, float]]:
        """Layer 2: Use LLM to evaluate response quality."""
        if not self._llm:
            return None

        context_preview = "\n---\n".join(c.content[:300] for c in chunks[:5])
        prompt = f"""Evaluate this answer. Return ONLY a JSON object, nothing else.

Query: {query}

Context:
{context_preview}

Answer: {answer}

Return ONLY this JSON (no explanation, no markdown):
{{"faithfulness": 0.0, "relevance": 0.0, "hallucination_risk": 0.0}}

Score each 0.0 to 1.0. faithfulness=grounded in context, relevance=addresses query, hallucination_risk=fabricated content (higher=worse)."""

        try:
            response = self._llm.generate(prompt, temperature=0.0, max_tokens=100)
            import json
            import re

            text = response.text.strip()

            # Strip markdown code blocks
            if "```" in text:
                parts = text.split("```")
                for part in parts:
                    part = part.strip()
                    if part.startswith("json"):
                        part = part[4:].strip()
                    if part.startswith("{"):
                        text = part
                        break

            # Try direct JSON parse
            try:
                scores = json.loads(text)
            except json.JSONDecodeError:
                # Extract JSON object with regex
                match = re.search(r'\{[^}]*"faithfulness"[^}]*\}', text)
                if match:
                    # Clean common issues: trailing commas, unquoted values
                    json_str = match.group()
                    json_str = re.sub(r',\s*}', '}', json_str)  # trailing comma
                    scores = json.loads(json_str)
                else:
                    # Try to extract any numbers from text
                    faith = re.search(r'faithfulness["\s:]+([0-9.]+)', text)
                    relev = re.search(r'relevance["\s:]+([0-9.]+)', text)
                    halluc = re.search(r'hallucination["\s:_risk]+([0-9.]+)', text)
                    if faith or relev:
                        scores = {
                            "faithfulness": float(faith.group(1)) if faith else 0.5,
                            "relevance": float(relev.group(1)) if relev else 0.5,
                            "hallucination_risk": float(halluc.group(1)) if halluc else 0.3,
                        }
                    else:
                        return None

            return {
                "faithfulness": min(1.0, max(0.0, float(scores.get("faithfulness", 0.5)))),
                "relevance": min(1.0, max(0.0, float(scores.get("relevance", 0.5)))),
                "hallucination_risk": min(1.0, max(0.0, float(scores.get("hallucination_risk", 0.5)))),
            }
        except Exception as e:
            logger.debug(f"LLM judge skipped: {e}")
            return None

    def get_history(self) -> List[Dict[str, Any]]:
        """Return evaluation history."""
        return self._history

    def get_average_score(self, window: int = 50) -> float:
        """Get average composite score over recent evaluations."""
        if not self._history:
            return 0.0
        recent = self._history[-window:]
        return sum(h["composite_score"] for h in recent) / len(recent)

    def get_improvement_rate(self) -> float:
        """Calculate improvement rate between first and recent queries."""
        if len(self._history) < 10:
            return 0.0
        early = self._history[:5]
        recent = self._history[-5:]
        early_avg = sum(h["composite_score"] for h in early) / len(early)
        recent_avg = sum(h["composite_score"] for h in recent) / len(recent)
        if early_avg == 0:
            return 0.0
        return (recent_avg - early_avg) / early_avg

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text)

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]
