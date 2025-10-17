"""Manages the lifecycle of metrics for an evaluation run."""

from typing import Any, Dict, List
import pandas as pd

from eval.metrics.base import BaseMetric
from eval.metrics.rag import (
    AnswerRelevancy,
    ContextualPrecision,
    ContextualRecall,
    Faithfulness,
)
from eval.metrics.retrieval import RetrievalMetrics
from eval.metrics.operational import CostMetric
from eval.metrics.conversation import (
    DeepEvalAnswerCorrectness,
    DeepEvalAnswerRelevancy,
    DeepEvalFaithfulness,
    TurnByTurnMetric,  # Import the base class
)
from eval.schemas.data_models import Conversation, Query, RAGOutput, Turn

# A registry of all available metric classes
METRIC_REGISTRY = {
    "faithfulness": Faithfulness,
    "answer_relevancy": AnswerRelevancy,
    "contextual_precision": ContextualPrecision,
    "contextual_recall": ContextualRecall,
    "cost": CostMetric,
    # The retrieval metrics class handles multiple sub-metrics
    "retrieval_metrics": RetrievalMetrics,
    # Conversational metrics (turn-by-turn)
    "turn_faithfulness": DeepEvalFaithfulness,
    "turn_answer_relevancy": DeepEvalAnswerRelevancy,
    "turn_answer_correctness": DeepEvalAnswerCorrectness,
}

# Map suite metric names to the class that computes them
# This allows suite files to be more granular (e.g., "recall@10")
SUITE_METRIC_MAP = {
    "faithfulness": "faithfulness",
    "answer_relevancy": "answer_relevancy",
    "contextual_precision": "contextual_precision",
    "contextual_recall": "contextual_recall",
    "cost": "cost",
    "recall@3": "retrieval_metrics",
    "recall@5": "retrieval_metrics",
    "recall@10": "retrieval_metrics",
    "mrr": "retrieval_metrics",
    "ndcg@3": "retrieval_metrics",
    "ndcg@5": "retrieval_metrics",
    "ndcg@10": "retrieval_metrics",
    # Conversational metrics (turn-by-turn)
    "turn_faithfulness": "turn_faithfulness",
    "turn_answer_relevancy": "turn_answer_relevancy",
    "turn_answer_correctness": "turn_answer_correctness",
}


class MetricManager:
    """Initializes and runs all metrics specified in a suite."""

    def __init__(self, metric_names: List[str]):
        """
        Initializes metric objects based on the names provided in the suite.
        Args:
            metric_names: A list of metric names from the evaluation suite config.
        """
        self.metrics, self.conv_metrics = self._initialize_metrics(metric_names)
        self.results: List[Dict[str, Any]] = []

    def _initialize_metrics(
        self, metric_names: List[str]
    ) -> (List[BaseMetric], List[TurnByTurnMetric]):  # Correct the type hint
        """Creates instances of the required metric classes."""
        metric_classes_to_load = set()
        for name in metric_names:
            clean_name = name.split("@")[0]  # e.g., recall@10 -> recall
            class_name = SUITE_METRIC_MAP.get(clean_name)
            if class_name:
                metric_classes_to_load.add(class_name)

        # Instantiate the unique set of required metric classes
        metrics = []
        conv_metrics = []
        for name in metric_classes_to_load:
            if name in METRIC_REGISTRY:
                instance = METRIC_REGISTRY[name]()
                if isinstance(instance, TurnByTurnMetric):
                    conv_metrics.append(instance)
                else:
                    metrics.append(instance)
        return metrics, conv_metrics

    def compute_all_metrics(
        self, query: Query, rag_output: RAGOutput
    ) -> Dict[str, Any]:
        """
        Runs all configured metrics for a single query-response pair.
        Args:
            query: The input query object.
            rag_output: The output from the RAG system.
        Returns:
            A dictionary containing all computed metric scores for the item.
        """
        item_results = {"query_id": query.id}
        for metric in self.metrics:
            try:
                scores = metric.compute(query, rag_output)
                item_results.update(scores)
            except Exception as e:
                print(
                    f"Warning: Metric '{metric.name}' failed for query '{query.id}': {e}"
                )

        return item_results

    def compute_all_conversation_metrics(
        self, conversation: Conversation, rag_outputs: List[RAGOutput]
    ) -> Dict[str, Any]:
        """
        Runs all configured conversational metrics for a single conversation.

        Args:
            conversation: The conversation object.
            rag_outputs: The list of RAG outputs for each assistant turn.

        Returns:
            A dictionary containing all computed metric scores for the conversation.
        """
        item_results = {"conversation_id": conversation.id}
        for metric in self.conv_metrics:
            try:
                # The compute method from TurnByTurnMetric has a compatible signature
                scores = metric.compute(conversation, rag_outputs)
                item_results.update(scores)
            except Exception as e:
                print(
                    f"Warning: Metric '{metric.name}' failed for conversation '{conversation.id}': {e}"
                )
        return item_results

    def add_result(self, result: Dict[str, Any]):
        """Adds a single item's result dictionary to the list of results."""
        self.results.append(result)

    def get_aggregated_results(self) -> pd.DataFrame:
        """
        Aggregates results from all queries into a pandas DataFrame.
        Calculates mean for scores and p95 for latency.
        """
        if not self.results:
            return pd.DataFrame()

        # Estrae solo i punteggi numerici per l'aggregazione, gestendo la nuova struttura
        scores_for_agg = []
        for result in self.results:
            # Determine if the result is from a query or a conversation
            id_key = "query_id" if "query_id" in result else "conversation_id"
            flat_scores = {id_key: result[id_key]}
            for key, value in result.items():
                if isinstance(value, dict) and "score" in value:
                    flat_scores[key] = value["score"]
                elif isinstance(value, (int, float)):
                    flat_scores[key] = value
            scores_for_agg.append(flat_scores)

        df = pd.DataFrame(scores_for_agg)
        id_col = "query_id" if "query_id" in df.columns else "conversation_id"
        df = df.set_index(id_col)

        # Calculate mean for all numeric columns
        numeric_cols = df.select_dtypes(include="number").columns
        aggregated = df[numeric_cols].mean().to_frame(name="score")

        # Calculate p95 for latency if it exists and add it as a separate row
        if 'latency' in df.columns:
            p95_latency = df['latency'].quantile(0.95)
            p95_row = pd.DataFrame([p95_latency], columns=['score'], index=['latency_p95'])
            aggregated = pd.concat([aggregated, p95_row])
            # Optional: rename 'latency' to 'latency_mean' for clarity
            aggregated = aggregated.rename(index={'latency': 'latency_mean'})

        aggregated.index.name = "metric"
        return aggregated
