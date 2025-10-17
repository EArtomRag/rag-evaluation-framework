"""A RAG adapter that communicates with a remote RAG system via HTTP."""

import requests
from requests.exceptions import RequestException
from typing import List

from eval.adapters.base import BaseRAGAdapter
from eval.schemas.data_models import Context, Query, RAGOutput, Turn


class RAGHttpAdapter(BaseRAGAdapter):
    """
    An adapter for a RAG system that is exposed via an HTTP API.
    """

    def __init__(self, base_url: str):
        """
        Initializes the adapter with the base URL of the RAG API.

        Args:
            base_url: The base URL, e.g., 'https://biblio-rag-app.fly.dev/'.
        """
        if not base_url.endswith('/'):
            base_url += '/'
        self.api_url = f"{base_url}chat"  # <-- MODIFICATO: da 'ask' a 'chat'
        self.headers = {"Content-Type": "application/json"}

    def get_response(self, query: Query) -> RAGOutput:
        """
        Sends the query to the remote RAG API and parses the response.

        For single-turn evaluation, this simulates a conversation with no history.

        Args:
            query: The input query object.

        Returns:
            The output from the RAG system, structured as a RAGOutput object.
        
        Raises:
            RuntimeError: If the API call fails or the response is invalid.
        """
        # For single-turn, we treat it as a conversation with a single turn
        return self.get_conversation_response(
            turns=[Turn(role="user", content=query.domanda)]
        )

    def get_conversation_response(self, turns: List[Turn]) -> RAGOutput:
        """
        Sends the full conversation history to the remote RAG API.

        Args:
            turns: The full list of turns in the conversation so far.

        Returns:
            The output from the RAG system, structured as a RAGOutput object.
        
        Raises:
            RuntimeError: If the API call fails or the response is invalid.
        """
        if not turns:
            raise ValueError("Conversation turns cannot be empty.")

        # The prompt is the content of the last turn
        prompt = turns[-1].content
        # The history is everything before the last turn
        history = [turn.model_dump() for turn in turns[:-1]]

        payload = {"prompt": prompt, "history": history}

        try:
            response = requests.post(
                self.api_url, json=payload, headers=self.headers, timeout=60
            )
            response.raise_for_status()

            api_response_data = response.json()

            # --- Mapping Layer ---
            contexts = []
            citation_ids = []

            citation_map = api_response_data.get("meta", {}).get("citation_map", {})
            used_citation_keys = [
                str(c)
                for c in api_response_data.get("meta", {}).get("used_citations", [])
            ]

            if citation_map and used_citation_keys:
                for key in used_citation_keys:
                    if key in citation_map:
                        citation_details = citation_map[key]
                        doc_id = str(citation_details.get("document_id"))

                        citation_ids.append(doc_id)
                        contexts.append(
                            Context(
                                doc_id=doc_id,
                                score=citation_details.get("distance", 0.0),
                                text=citation_details.get("snippet", ""),
                                metadata={
                                    "title": citation_details.get("document_title"),
                                    "sequence_number": citation_details.get(
                                        "sequence_number"
                                    ),
                                },
                            )
                        )

            mapped_data = {
                "answer": api_response_data.get("answer"),
                "contexts": contexts,
                "citations": citation_ids,
                "trace": None,
            }

            return RAGOutput.model_validate(mapped_data)

        except RequestException as e:
            raise RuntimeError(f"API request to {self.api_url} failed: {e}")
        except (ValueError, KeyError) as e:
            raise RuntimeError(f"Failed to parse or validate API response: {e}")
