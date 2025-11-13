"""Abstract base class for all metric computations."""

from abc import ABC, abstractmethod
from typing import Any, Dict

from eval.schemas.data_models import Query, RAGOutput


class BaseMetric(ABC):
    """
    Abstract interface for a metric.

    Each metric must implement the `compute` method, which takes the query
    and the RAG system's output and returns a dictionary of scores.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """The unique name of the metric."""
        raise NotImplementedError

    @abstractmethod
    def compute(self, query: Query, rag_output: RAGOutput) -> Dict[str, Any]:
        """
        Computes the metric for a single query-response pair.

        Args:
            query: The input query object.
            rag_output: The output from the RAG system.

        Returns:
            A dictionary containing the metric name(s) and their calculated score(s).
        """
        raise NotImplementedError








