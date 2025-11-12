"""Data models for the RAG evaluation runner."""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class QuestionType(str, Enum):
    """Enumeration for the type of question."""
    SYNTHESIS = "synthesis"
    FACTUAL = "factual"
    COMPARISON = "comparison"
    OUT_OF_SCOPE = "out_of_scope"
    SQL = "sql"
    MULTI_TURN = "multi-turn"


class Query(BaseModel):
    """Represents a single query from a dataset (e.g., a line in queries.jsonl)."""
    id: str = Field(..., description="Unique identifier for the query.")
    domanda: str = Field(..., description="The question text.")
    tipo_domanda: QuestionType = Field(..., description="The type of the question.")
    gold_answer: Optional[str] = Field(None, description="The ground truth answer, if available.")
    gold_citations: List[str] = Field([], description="List of relevant document IDs for the query.")


class Turn(BaseModel):
    """Represents a single turn in a conversation."""
    role: str = Field(..., description="The role of the speaker, e.g., 'user' or 'assistant'.")
    content: str = Field(..., description="The content of the turn.")
    # Ground truth for the expected response to this turn (if role is 'user')
    gold_answer: Optional[str] = Field(
        None, description="The ground truth answer expected after this turn."
    )
    gold_citations: List[str] = Field(
        default_factory=list,
        description="List of relevant document IDs for the expected answer.",
    )


class Conversation(BaseModel):
    """Represents a multi-turn conversation."""
    id: str = Field(..., description="Unique identifier for the conversation.")
    turns: List[Turn] = Field(..., description="A list of turns that make up the conversation.")
    metadata: Dict[str, Any] = Field({}, description="Additional metadata for the conversation.")


class Context(BaseModel):
    """Represents a retrieved context document."""
    doc_id: str = Field(..., description="The unique identifier of the document.")
    score: float = Field(..., description="The retrieval score of the document.")
    text: str = Field(..., description="The text content of the document.")
    metadata: Dict[str, Any] = Field({}, description="Additional metadata for the document.")


class TokenUsage(BaseModel):
    """Represents the token usage for a RAG model call."""
    input: int = Field(..., description="Number of input tokens.")
    output: int = Field(..., description="Number of output tokens.")


class RAGTrace(BaseModel):
    """Represents the trace information for a RAG system call."""
    prompt: Optional[str] = Field(None, description="The full prompt sent to the language model.")
    token_usage: Optional[TokenUsage] = Field(None, description="Token usage details.")


class RAGOutput(BaseModel):
    """Represents the output of a RAG system for a single query."""
    answer: str = Field(..., description="The generated answer.")
    contexts: List[Context] = Field(..., description="The list of retrieved contexts.")
    citations: List[str] = Field([], description="List of document IDs cited in the answer.")
    trace: Optional[RAGTrace] = Field(None, description="Optional trace information.")


class AdapterConfig(BaseModel):
    """Configuration for the RAG adapter."""
    type: str = Field(..., description="The type of adapter ('inproc' or 'http').")
    url: Optional[str] = Field(None, description="The base URL for the HTTP adapter.")


class RAGParameters(BaseModel):
    """Defines the parameters for the RAG system under evaluation."""
    k: int = Field(10, description="Number of documents to retrieve.")
    filters: Dict[str, Any] = Field({}, description="Any filters to apply during retrieval.")
    model: str = Field(..., description="The identifier for the generator model.")


class JudgeParameters(BaseModel):
    """Defines the parameters for the LLM-as-a-Judge."""
    model: str = Field("gpt-4", description="The identifier for the judge model.")
    use_llm_as_judge: bool = Field(True, description="Flag to enable or disable the LLM judge.")


class EvaluationSuite(BaseModel):
    """Defines a full evaluation suite configuration."""
    name: str = Field(..., description="The name of the evaluation suite.")
    dataset_path: str = Field(..., description="Path to the dataset file (e.g., queries.jsonl).")
    adapter: AdapterConfig = Field(..., description="Adapter configuration.")
    rag_params: RAGParameters = Field(..., description="Parameters for the RAG system.")
    metrics: List[str] = Field(..., description="List of metric names to compute.")
    judge: Optional[JudgeParameters] = Field(None, description="Parameters for the LLM judge.")
