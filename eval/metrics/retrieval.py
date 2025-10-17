"""Offline retrieval metrics using pytrec-eval."""

import pytrec_eval
from typing import Any, Dict

from eval.metrics.base import BaseMetric
from eval.schemas.data_models import Query, RAGOutput


class RetrievalMetrics(BaseMetric):
    """
    Computes multiple offline retrieval metrics using pytrec-eval.
    This includes Recall@k, MRR, and NDCG@k.
    """

    def __init__(self, k_values: list[int] = [5, 10]):
        self._k_values = k_values
        # Usa i nomi delle metriche attesi da pytrec-eval
        self._metric_names = {f"recall.{k}" for k in k_values} | \
                             {f"ndcg_cut.{k}" for k in k_values} | \
                             {"recip_rank"}

    @property
    def name(self) -> str:
        return "retrieval_metrics"

    def compute(self, query: Query, rag_output: RAGOutput) -> Dict[str, Any]:
        """
        Computes retrieval metrics if ground truth citations are available.

        Args:
            query: The input query object, containing gold_citations.
            rag_output: The output from the RAG system, containing retrieved contexts.

        Returns:
            A dictionary with scores for recall, mrr, and ndcg for each k.
            Returns an empty dict if no gold_citations are provided.
        """
        if not query.gold_citations:
            return {}

        # pytrec_eval format for ground truth (qrels)
        qrels = {query.id: {doc_id: 1 for doc_id in query.gold_citations}}

        # pytrec_eval format for system's retrieved documents (run)
        run = {query.id: {ctx.doc_id: ctx.score for ctx in rag_output.contexts}}

        evaluator = pytrec_eval.RelevanceEvaluator(qrels, self._metric_names)
        results = evaluator.evaluate(run)

        # Extract the metrics for the current query
        query_results = results.get(query.id, {})
        
        # Rinomina le chiavi per renderle più leggibili nel report finale
        final_results = {}
        for key, value in query_results.items():
            if key.startswith('recall_'):
                final_results[key.replace('_', '@')] = value
            elif key.startswith('ndcg_cut_'):
                final_results[key.replace('ndcg_cut_', 'ndcg@')] = value
            elif key == 'recip_rank':
                final_results['mrr'] = value
            else:
                final_results[key] = value

        return final_results

