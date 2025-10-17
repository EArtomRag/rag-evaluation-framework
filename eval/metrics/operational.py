"""Operational metrics like cost and latency."""

from typing import Any, Dict

from eval.metrics.base import BaseMetric
from eval.schemas.data_models import Query, RAGOutput

# Simple cost model (per 1M tokens)
TOKEN_COSTS = {
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4o": {"input": 5.00, "output": 15.00},
    "default": {"input": 1.00, "output": 3.00},
}


class CostMetric(BaseMetric):
    """
    Computes the estimated cost of a RAG query based on token usage.
    """

    @property
    def name(self) -> str:
        return "cost"

    def compute(self, query: Query, rag_output: RAGOutput) -> Dict[str, Any]:
        """
        Calculates the cost if token usage data is available in the trace.

        Args:
            query: The input query object.
            rag_output: The output from the RAG system, containing trace info.

        Returns:
            A dictionary with the calculated cost, or an empty dict if
            trace information is missing.
        """
        if not rag_output.trace or not rag_output.trace.token_usage:
            return {}

        # This part assumes the model name is available. For now, we'll use a default.
        # In a real scenario, you'd get this from the suite or adapter.
        model_name = "default" 
        costs = TOKEN_COSTS.get(model_name, TOKEN_COSTS["default"])

        input_cost = (rag_output.trace.token_usage.input / 1_000_000) * costs["input"]
        output_cost = (rag_output.trace.token_usage.output / 1_000_000) * costs["output"]
        total_cost = input_cost + output_cost

        return {self.name: total_cost}

