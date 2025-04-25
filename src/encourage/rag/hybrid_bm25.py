"""Hybrid RAG implementation combining vector embedding search with BM25 ranking."""

import logging
import uuid
from typing import Any, Dict, List, Tuple, override

import numpy as np
from rank_bm25 import BM25Okapi

from encourage.llm import BatchInferenceRunner
from encourage.prompts.context import Document
from encourage.rag.base_impl import BaseRAG

logger = logging.getLogger(__name__)


class HybridBM25RAG(BaseRAG):
    """Hybrid RAG combining dense embeddings with sparse lexical search.

    Combines semantic search (dense embeddings) with lexical search (sparse BM25).

    Attributes:
        alpha: Weight for dense retrieval scores (0-1)
        beta: Weight for sparse retrieval scores (0-1)
        bm25_index: BM25 index for lexical search
        document_texts: Preprocessed document texts
        document_map: Maps document text to Document objects

    """

    def __init__(
        self,
        context_collection: list[Document],
        collection_name: str,
        embedding_function: Any,
        top_k: int,
        retrieval_only: bool = False,
        runner: BatchInferenceRunner | None = None,
        additional_prompt: str = "",
        template_name: str = "",
        alpha: float = 0.5,  # Dense retrieval weight
        beta: float = 0.5,  # Sparse retrieval weight
        device: str = "cuda",
        where: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize HybridBM25RAG with configuration."""
        super().__init__(
            context_collection=context_collection,
            collection_name=collection_name,
            embedding_function=embedding_function,
            top_k=top_k,
            retrieval_only=retrieval_only,
            runner=runner,
            additional_prompt=additional_prompt,
            template_name=template_name,
            device=device,
            where=where,
            **kwargs,
        )

        # Validate weights
        assert 0 <= alpha <= 1, "Alpha must be between 0 and 1"
        assert 0 <= beta <= 1, "Beta must be between 0 and 1"
        assert abs(alpha + beta - 1.0) < 1e-6, "Alpha and beta must sum to 1"

        self.alpha = alpha
        self.beta = beta

        # Create BM25 index
        self._create_bm25_index(context_collection)

    def _create_bm25_index(self, context_collection: list[Document]) -> None:
        """Create a BM25 index from the documents."""
        # Store original documents by index for direct access
        self.documents = list(context_collection)

        # Create tokenized texts for BM25
        self.document_texts = [doc.content.lower().split() for doc in self.documents]

        # Create BM25 index
        self.bm25_index = BM25Okapi(self.document_texts)
        logger.info(f"BM25 index created with {len(self.document_texts)} documents.")

    def _retrieve_sparse_results(self, query: str) -> Tuple[List[Document], Dict[uuid.UUID, float]]:
        """Calculate BM25 scores and return top documents for a query.

        Returns:
            A tuple of (top sparse retrieval documents, normalized scores by document ID)

        """
        # Tokenize query and get scores
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25_index.get_scores(tokenized_query)

        # Get top k document indices
        top_bm25_indices = np.argsort(bm25_scores)[::-1]
        sparse_docs = [self.documents[i] for i in top_bm25_indices]

        # Normalize scores
        max_bm25_score = max(bm25_scores) if bm25_scores.any() else 1.0
        if max_bm25_score > 0:
            normalized_scores = {
                self.documents[i].id: score / max_bm25_score
                for i, score in enumerate(bm25_scores)
                if score > 0
            }
        else:
            normalized_scores = {}

        return sparse_docs, normalized_scores

    def _compute_hybrid_scores(
        self,
        dense_docs: List[Document],
        sparse_docs: List[Document],
        sparse_scores: Dict[uuid.UUID, float],
    ) -> List[Tuple[float, Document]]:
        """Compute hybrid scores combining dense and sparse retrieval results.

        Args:
            dense_docs: Documents from dense vector retrieval
            sparse_docs: Documents from sparse BM25 retrieval
            sparse_scores: Normalized BM25 scores by document ID

        Returns:
            List of (score, document) tuples ranked by hybrid score

        """
        # Get all unique document IDs
        all_doc_ids = {doc.id for doc in dense_docs} | {doc.id for doc in sparse_docs}

        # Create maps for lookup
        dense_map = {doc.id: (i, doc) for i, doc in enumerate(dense_docs)}
        sparse_map = {doc.id: (i, doc) for i, doc in enumerate(sparse_docs)}

        # Find minimum non-zero BM25 score for scaling fallback scores
        min_nonzero_bm25 = min(
            [score for score in sparse_scores.values() if score > 0], default=0.1
        )
        fallback_ceiling = min_nonzero_bm25 * 0.9  # Ensure fallbacks are below actual scores

        # Calculate scores for each document
        scored_docs = []

        for doc_id in all_doc_ids:
            # Get dense score (position-based)
            dense_pos = dense_map.get(doc_id, (len(dense_docs), None))[0]
            dense_score = (
                1.0 - (dense_pos / max(len(dense_docs), 1)) if dense_pos < len(dense_docs) else 0.0
            )

            # Get sparse score or calculate fallback
            sparse_score = sparse_scores.get(doc_id, 0.0)
            if sparse_score == 0.0 and doc_id in sparse_map:
                sparse_pos = sparse_map[doc_id][0]
                position_ratio = sparse_pos / max(len(sparse_docs), 1)
                # Scale fallback to be below minimum non-zero BM25 score
                sparse_score = fallback_ceiling * (1.0 - position_ratio)

            # Calculate hybrid score
            hybrid_score = (self.alpha * dense_score) + (self.beta * sparse_score)

            # Get document (prefer dense version)
            doc = dense_map.get(doc_id, (None, None))[1] or sparse_map.get(doc_id, (None, None))[1]
            if doc:
                scored_docs.append((hybrid_score, doc))

        return scored_docs

    def _rank_documents(self, dense_docs: List[Document], query: str) -> List[Document]:
        """Rank documents using the hybrid dense + sparse approach."""
        # Get sparse retrieval results
        sparse_docs, sparse_scores = self._retrieve_sparse_results(query)

        # Calculate hybrid scores
        scored_docs = self._compute_hybrid_scores(dense_docs, sparse_docs, sparse_scores)

        # Sort by score and return top_k
        sorted_docs = [doc for _, doc in sorted(scored_docs, key=lambda x: x[0], reverse=True)]
        return sorted_docs[: self.top_k]

    @override
    def retrieve_contexts(
        self,
        query_list: list[str],
        **kwargs: Any,
    ) -> list[list[Document]]:
        """Retrieve contexts using hybrid dense + sparse approach."""
        # Get dense retrieval results from parent class
        dense_results = super().retrieve_contexts(query_list=query_list, **kwargs)

        # Process each query using the hybrid ranking function
        final_results = []
        for idx, query in enumerate(query_list):
            # Get dense docs for this query
            dense_docs = dense_results[idx] if idx < len(dense_results) else []

            # Rank using hybrid approach
            ranked_docs = self._rank_documents(dense_docs, query)
            final_results.append(ranked_docs)

        return final_results
