"""Conversation-specific metrics, evaluated on a turn-by-turn basis."""
from typing import Any, Dict, List, Tuple
import numpy as np

from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    GEval,
)
from deepeval.test_case import LLMTestCase
from deepeval.test_case import LLMTestCaseParams

from eval.schemas.data_models import Conversation, RAGOutput, Turn
from eval.metrics.base import BaseMetric


def _get_evaluation_pairs(
    conversation: Conversation, rag_outputs: List[RAGOutput]
) -> List[Tuple[Turn, RAGOutput]]:
    """
    Pairs user turns with their corresponding assistant RAG output.
    """
    pairs = []
    assistant_response_index = 0
    # We only iterate through the original turns from the dataset
    for turn in conversation.turns:
        if turn.role == "user":
            if assistant_response_index < len(rag_outputs):
                pairs.append((turn, rag_outputs[assistant_response_index]))
                assistant_response_index += 1
    return pairs


class TurnByTurnMetric(BaseMetric):
    """
    Abstract base class for metrics computed turn-by-turn and then aggregated.
    """

    def compute_per_turn(self, user_turn: Turn, assistant_output: RAGOutput) -> float:
        raise NotImplementedError

    def aggregate_scores(self, scores: List[float]) -> float:
        return np.mean(scores) if scores else 0.0

    def compute(
        self, conversation: Conversation, rag_outputs: List[RAGOutput]
    ) -> Dict[str, Any]:
        evaluation_pairs = _get_evaluation_pairs(conversation, rag_outputs)
        turn_scores = []
        reasons = []

        for user_turn, assistant_output in evaluation_pairs:
            if self.is_computable(user_turn, assistant_output):
                score, reason = self.compute_per_turn_with_reason(user_turn, assistant_output)
                turn_scores.append(score)
                if reason:
                    reasons.append(f"Turn '{user_turn.content[:30]}...': {reason}")
        
        if not turn_scores:
            return {self.name: {"score": 0.0, "explanation": "No turns eligible for evaluation."}}

        avg_score = self.aggregate_scores(turn_scores)
        full_reason = " | ".join(reasons)
        
        return {
            self.name: {
                "score": float(avg_score),
                "explanation": f"Average score over {len(turn_scores)} turns. " + full_reason,
            }
        }

    def compute_per_turn_with_reason(self, user_turn: Turn, assistant_output: RAGOutput) -> Tuple[float, str]:
        # Default implementation for metrics that don't produce a reason
        score = self.compute_per_turn(user_turn, assistant_output)
        return score, ""

    def is_computable(self, user_turn: Turn, assistant_output: RAGOutput) -> bool:
        return True


class DeepEvalFaithfulness(TurnByTurnMetric):
    """Evaluates the Faithfulness for each turn of a conversation."""
    @property
    def name(self) -> str:
        return "turn_faithfulness"

    def is_computable(self, user_turn: Turn, assistant_output: RAGOutput) -> bool:
        return bool(assistant_output.contexts)

    def compute_per_turn_with_reason(self, user_turn: Turn, assistant_output: RAGOutput) -> Tuple[float, str]:
        print(f"        - Calculating Faithfulness for turn: '{user_turn.content[:30]}...'")
        metric = FaithfulnessMetric(threshold=0.5, model="gpt-3.5-turbo")
        test_case = LLMTestCase(
            input=user_turn.content,
            actual_output=assistant_output.answer,
            retrieval_context=[ctx.text for ctx in assistant_output.contexts],
        )
        metric.measure(test_case)
        print(f"          -> Faithfulness score: {metric.score:.2f}")
        return metric.score, metric.reason


class DeepEvalAnswerRelevancy(TurnByTurnMetric):
    """Evaluates the Answer Relevancy for each turn of a conversation."""
    @property
    def name(self) -> str:
        return "turn_answer_relevancy"

    def compute_per_turn_with_reason(self, user_turn: Turn, assistant_output: RAGOutput) -> Tuple[float, str]:
        print(f"        - Calculating Answer Relevancy for turn: '{user_turn.content[:30]}...'")
        metric = AnswerRelevancyMetric(threshold=0.5, model="gpt-3.5-turbo")
        test_case = LLMTestCase(
            input=user_turn.content, actual_output=assistant_output.answer
        )
        metric.measure(test_case)
        print(f"          -> Answer Relevancy score: {metric.score:.2f}")
        return metric.score, metric.reason


class DeepEvalAnswerCorrectness(TurnByTurnMetric):
    """Evaluates the Answer Correctness against a gold standard for each turn."""
    @property
    def name(self) -> str:
        return "turn_answer_correctness"

    def is_computable(self, user_turn: Turn, assistant_output: RAGOutput) -> bool:
        return bool(user_turn.gold_answer)

    def compute_per_turn_with_reason(self, user_turn: Turn, assistant_output: RAGOutput) -> Tuple[float, str]:
        print(f"        - Calculating Answer Correctness for turn: '{user_turn.content[:30]}...'")
        # GEval is the correct way to measure correctness against a gold standard
        correctness_metric = GEval(
            name="Answer Correctness",
            criteria="Correctness - determine if the actual output is semantically similar to the expected output.",
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
            model="gpt-3.5-turbo",
        )
        test_case = LLMTestCase(
            input=user_turn.content,
            actual_output=assistant_output.answer,
            expected_output=user_turn.gold_answer,
        )
        correctness_metric.measure(test_case)
        print(f"          -> Answer Correctness score: {correctness_metric.score:.2f}")
        return correctness_metric.score, correctness_metric.reason
