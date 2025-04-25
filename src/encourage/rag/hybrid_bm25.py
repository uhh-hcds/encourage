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
    """Hybrid RAG combining vector embeddings with BM25 ranking.

    Combines semantic search (vectors) with lexical search (BM25).

    Attributes:
        alpha: Weight for vector search scores (0-1)
        beta: Weight for BM25 scores (0-1)
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
        alpha: float = 0.5,  # Vector search weight
        beta: float = 0.5,  # BM25 weight
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

    def _get_bm25_scores(self, query: str) -> Tuple[np.ndarray, Dict[uuid.UUID, float]]:
        """Calculate and normalize BM25 scores for a query.

        Returns:
            Raw scores array and dictionary of normalized scores by doc ID

        """
        # Tokenize query and get scores
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25_index.get_scores(tokenized_query)

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

        return bm25_scores, normalized_scores

    def _calculate_hybrid_scores(
        self,
        vector_docs: List[Document],
        bm25_docs: List[Document],
        normalized_bm25_scores: Dict[uuid.UUID, float],
    ) -> List[Tuple[float, Document]]:
        """Calculate hybrid scores for documents."""
        # Get all unique document IDs
        all_doc_ids = {doc.id for doc in vector_docs} | {doc.id for doc in bm25_docs}

        # Create maps for lookup
        vector_map = {doc.id: (i, doc) for i, doc in enumerate(vector_docs)}
        bm25_map = {doc.id: (i, doc) for i, doc in enumerate(bm25_docs)}

        # Find minimum non-zero BM25 score for scaling fallback scores
        min_nonzero_bm25 = min(
            [score for score in normalized_bm25_scores.values() if score > 0], default=0.1
        )
        fallback_ceiling = min_nonzero_bm25 * 0.9  # Ensure fallbacks are below actual scores

        # Calculate scores for each document
        scored_docs = []

        for doc_id in all_doc_ids:
            # Get vector score (position-based)
            vector_pos = vector_map.get(doc_id, (len(vector_docs), None))[0]
            vector_score = (
                1.0 - (vector_pos / max(len(vector_docs), 1))
                if vector_pos < len(vector_docs)
                else 0.0
            )

            # Get BM25 score
            bm25_score = normalized_bm25_scores.get(doc_id, 0.0)

            # If no BM25 score but in results, use position-based fallback scaled to be below min
            # non-zero score
            if bm25_score == 0.0 and doc_id in bm25_map:
                bm25_pos = bm25_map[doc_id][0]
                position_ratio = bm25_pos / max(len(bm25_docs), 1)
                # Scale to be below minimum non-zero BM25 score
                bm25_score = fallback_ceiling * (1.0 - position_ratio)

            # Calculate hybrid score
            hybrid_score = (self.alpha * vector_score) + (self.beta * bm25_score)

            # Get document (prefer vector version)
            doc = vector_map.get(doc_id, (None, None))[1] or bm25_map.get(doc_id, (None, None))[1]
            if doc:
                scored_docs.append((hybrid_score, doc))

        return scored_docs

    @override
    def retrieve_contexts(
        self,
        query_list: list[str],
        **kwargs: Any,
    ) -> list[list[Document]]:
        """Retrieve contexts using hybrid vector + BM25 approach."""
        # Get vector search results
        vector_results = super().retrieve_contexts(query_list=query_list, **kwargs)

        # Process each query
        final_results = []

        for idx, query in enumerate(query_list):
            # Get vector docs for this query
            vector_docs = vector_results[idx] if idx < len(vector_results) else []

            # Get BM25 scores and docs
            bm25_scores, normalized_bm25_scores = self._get_bm25_scores(query)
            top_bm25_indices = np.argsort(bm25_scores)[::-1][: self.top_k]
            bm25_docs = [self.documents[i] for i in top_bm25_indices]

            # Combine and rank documents
            scored_docs = self._calculate_hybrid_scores(
                vector_docs, bm25_docs, normalized_bm25_scores
            )
            sorted_docs = [doc for _, doc in sorted(scored_docs, key=lambda x: x[0], reverse=True)]
            final_results.append(sorted_docs[: self.top_k])

        return final_results

    def _hybrid_ranking(
        self,
        vector_docs: list[Document],
        bm25_docs: list[Document],
        query: str,
    ) -> list[Document]:
        """Legacy method for backward compatibility."""
        # Calculate BM25 scores
        _, normalized_bm25_scores = self._get_bm25_scores(query)

        # Calculate hybrid scores
        scored_docs = self._calculate_hybrid_scores(vector_docs, bm25_docs, normalized_bm25_scores)

        # Sort by score and return top results
        sorted_docs = [doc for _, doc in sorted(scored_docs, key=lambda x: x[0], reverse=True)]
        return sorted_docs[: self.top_k]
