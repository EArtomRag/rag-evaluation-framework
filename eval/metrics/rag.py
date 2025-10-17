"""RAG quality and contextual metrics using the DeepEval library."""

from typing import Any, Dict

from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase

from eval.metrics.base import BaseMetric
from eval.schemas.data_models import Query, RAGOutput


class Faithfulness(BaseMetric):
    """Computes the Faithfulness metric using DeepEval."""

    @property
    def name(self) -> str:
        return "faithfulness"

    def compute(self, query: Query, rag_output: RAGOutput) -> Dict[str, Any]:
        """Calculates if the answer is factually consistent with the contexts."""
        test_case = LLMTestCase(
            input=query.domanda,
            actual_output=rag_output.answer,
            retrieval_context=[ctx.text for ctx in rag_output.contexts],
        )
        metric = FaithfulnessMetric(threshold=0.5, model="gpt-4o")
        metric.measure(test_case)
        # Restituisce un dizionario con sia il punteggio che la motivazione
        return {
            self.name: {
                "score": metric.score,
                "reason": metric.reason
            }
        }


class AnswerRelevancy(BaseMetric):
    """Computes the Answer Relevancy metric using DeepEval."""

    @property
    def name(self) -> str:
        return "answer_relevancy"

    def compute(self, query: Query, rag_output: RAGOutput) -> Dict[str, Any]:
        """Calculates if the answer is relevant to the input query."""
        test_case = LLMTestCase(
            input=query.domanda,
            actual_output=rag_output.answer,
        )
        metric = AnswerRelevancyMetric(threshold=0.5, model="gpt-4o")
        metric.measure(test_case)
        # Restituisce un dizionario con sia il punteggio che la motivazione
        return {
            self.name: {
                "score": metric.score,
                "reason": metric.reason
            }
        }


class ContextualPrecision(BaseMetric):
    """Computes the Contextual Precision metric using DeepEval."""

    @property
    def name(self) -> str:
        return "contextual_precision"

    def compute(self, query: Query, rag_output: RAGOutput) -> Dict[str, Any]:
        """Calculates the precision of the retrieved contexts relative to the query."""
        if not query.gold_answer or not rag_output.contexts:
            # These metrics require a gold_answer AND at least one context.
            return {}
            
        test_case = LLMTestCase(
            input=query.domanda,
            actual_output=rag_output.answer,
            retrieval_context=[ctx.text for ctx in rag_output.contexts],
            expected_output=query.gold_answer,
        )
        metric = ContextualPrecisionMetric(threshold=0.5, model="gpt-4o")
        metric.measure(test_case)
        # Restituisce un dizionario con sia il punteggio che la motivazione
        return {
            self.name: {
                "score": metric.score,
                "reason": metric.reason
            }
        }


class ContextualRecall(BaseMetric):
    """Computes the Contextual Recall metric using DeepEval."""

    @property
    def name(self) -> str:
        return "contextual_recall"

    def compute(self, query: Query, rag_output: RAGOutput) -> Dict[str, Any]:
        """Calculates the recall of the retrieved contexts relative to the gold answer."""
        if not query.gold_answer or not rag_output.contexts:
            # These metrics require a gold_answer AND at least one context.
            return {}

        test_case = LLMTestCase(
            input=query.domanda,
            actual_output=rag_output.answer,
            expected_output=query.gold_answer,
            retrieval_context=[ctx.text for ctx in rag_output.contexts],
        )
        metric = ContextualRecallMetric(threshold=0.5, model="gpt-4o")
        metric.measure(test_case)
        # Restituisce un dizionario con sia il punteggio che la motivazione
        return {
            self.name: {
                "score": metric.score,
                "reason": metric.reason
            }
        }

