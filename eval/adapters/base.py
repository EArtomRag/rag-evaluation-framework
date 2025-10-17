"""Abstract base class for RAG adapters."""

from abc import ABC, abstractmethod
from typing import List

from eval.schemas.data_models import Query, RAGOutput, Turn


class BaseRAGAdapter(ABC):
    """
    Abstract interface for a RAG system adapter.

    This class defines the contract that all RAG adapters must follow,
    ensuring they can be used interchangeably by the evaluation runner.
    """

    @abstractmethod
    def get_response(self, query: Query) -> RAGOutput:
        """
        Gets a response from the RAG system for a given query.

        Args:
            query: The input query object.

        Returns:
            The output from the RAG system, structured as a RAGOutput object.
        """
        raise NotImplementedError

    @abstractmethod
    def get_conversation_response(self, turns: List[Turn]) -> RAGOutput:
        """
        Gets a response from the RAG system for the latest turn in a conversation.

        Args:
            turns: The full list of turns in the conversation so far.

        Returns:
            The output from the RAG system for the latest user turn.
        """
        raise NotImplementedError

