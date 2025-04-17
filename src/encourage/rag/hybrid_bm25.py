"""Hybrid RAG implementation combining vector embedding search with BM25 ranking."""

import logging
from typing import Any, override

import numpy as np
from rank_bm25 import BM25Okapi

from encourage.llm import BatchInferenceRunner
from encourage.prompts.context import Document
from encourage.rag.base_impl import BaseRAG

logger = logging.getLogger(__name__)


class HybridBM25RAG(BaseRAG):
    """Hybrid RAG implementation that combines vector embeddings with BM25 ranking.

    This RAG method enhances traditional vector-based retrieval with BM25 scoring,
    which is particularly effective for keyword-based matching. The hybrid approach
    combines the strengths of semantic search (vectors) with lexical search (BM25).

    Attributes:
        alpha (float): Weight for vector search scores (between 0 and 1)
        beta (float): Weight for BM25 scores (between 0 and 1)
        bm25_index (BM25Okapi): The BM25 index for lexical search
        document_texts (List[str]): Preprocessed document texts for BM25
        document_map (Dict[str, Document]): Mapping from document text to Document objects

    In addition to the attributes inherited from BaseRAG.

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
        alpha: float = 0.5,  # Weight for vector search
        beta: float = 0.5,  # Weight for BM25
        device: str = "cuda",
        where: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize HybridBM25RAG with configuration.

        Args:
            context_collection: List of documents to index
            collection_name: Name for the vector collection
            embedding_function: Function to generate embeddings
            top_k: Number of documents to retrieve
            alpha: Weight for vector search scores (default: 0.5)
            beta: Weight for BM25 scores (default: 0.5)
            retrieval_only: If True, only retrieval is done without LLM inference
            runner: BatchInferenceRunner for LLM queries
            additional_prompt: Extra text to append to prompts
            template_name: Name of the prompt template
            device: Device to use for embeddings (default: "cuda")
            where: Optional filtering conditions
            **kwargs: Additional arguments

        """
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
        self.document_texts = [doc.content.lower().split() for doc in context_collection]
        self.document_map = {
            str(text): doc for text, doc in zip(self.document_texts, context_collection)
        }
        self.bm25_index = BM25Okapi(self.document_texts)
        logger.info(f"BM25 index created with {len(self.document_texts)} documents.")

    @override
    def retrieve_contexts(
        self,
        query_list: list[str],
        **kwargs: Any,
    ) -> list[list[Document]]:
        """Retrieve contexts using a hybrid vector + BM25 approach.

        This method performs both vector-based and BM25-based retrieval,
        then combines the results using specified weights.

        Args:
            query_list: List of queries to process
            **kwargs: Additional arguments

        Returns:
            List of lists of retrieved documents for each query

        """
        # Get vector search results (from parent class)
        vector_results = super().retrieve_contexts(query_list=query_list, **kwargs)

        # Process each query with BM25
        final_results = []

        for idx, query in enumerate(query_list):
            # Get vector search results for this query
            vector_docs = vector_results[idx] if idx < len(vector_results) else []

            # Tokenize the query for BM25
            tokenized_query = query.lower().split()

            # Get BM25 scores for all documents
            bm25_scores = self.bm25_index.get_scores(tokenized_query)

            # Sort documents by BM25 score and get top_k
            top_bm25_indices = np.argsort(bm25_scores)[::-1][: self.top_k]
            bm25_docs = [self.document_map[str(self.document_texts[i])] for i in top_bm25_indices]

            # Combine the results using a hybrid ranking
            combined_docs = self._hybrid_ranking(vector_docs, bm25_docs, query)

            # Add the combined results to the final list
            final_results.append(combined_docs)

        return final_results

    def _hybrid_ranking(
        self, vector_docs: list[Document], bm25_docs: list[Document], query: str
    ) -> list[Document]:
        """Combine vector and BM25 results using a weighted approach.

        Args:
            vector_docs: Documents retrieved by vector search
            bm25_docs: Documents retrieved by BM25
            query: The original query used to directly compute BM25 scores

        Returns:
            List of documents ranked by the hybrid score

        """
        # Create a set of all unique document IDs
        all_doc_ids = {doc.id for doc in vector_docs} | {doc.id for doc in bm25_docs}

        # Create a map for quick lookup
        vector_map = {doc.id: (i, doc) for i, doc in enumerate(vector_docs)}
        bm25_map = {doc.id: (i, doc) for i, doc in enumerate(bm25_docs)}

        # Tokenize the query for direct BM25 scoring
        tokenized_query = query.lower().split()

        # Get raw BM25 scores
        raw_bm25_scores = self.bm25_index.get_scores(tokenized_query)

        # Normalize BM25 scores (avoid division by zero)
        max_bm25_score = max(raw_bm25_scores) if raw_bm25_scores.any() else 1.0
        if max_bm25_score > 0:
            normalized_bm25_scores = {
                self.document_map[str(self.document_texts[i])].id: score / max_bm25_score
                for i, score in enumerate(raw_bm25_scores)
                if score > 0
            }
        else:
            normalized_bm25_scores = {}

        # Calculate hybrid scores
        hybrid_scores = []
        hybrid_docs = []

        for doc_id in all_doc_ids:
            # Get vector score (normalized by position)
            vector_pos = vector_map.get(doc_id, (len(vector_docs), None))[0]
            vector_score = (
                1.0 - (vector_pos / max(len(vector_docs), 1))
                if vector_pos < len(vector_docs)
                else 0.0
            )

            # Get BM25 score - use raw BM25 score if available, otherwise use position-based score
            bm25_score = normalized_bm25_scores.get(doc_id, 0.0)

            # If BM25 score is 0.0 but document is in BM25 results
            if bm25_score == 0.0 and doc_id in bm25_map:
                bm25_pos = bm25_map[doc_id][0]
                bm25_score = 1.0 - (bm25_pos / max(len(bm25_docs), 1))

            # Calculate hybrid score
            hybrid_score = (self.alpha * vector_score) + (self.beta * bm25_score)

            # Get the document (prefer vector version to maintain metadata)
            doc = vector_map.get(doc_id, (None, None))[1] or bm25_map.get(doc_id, (None, None))[1]
            if doc:
                hybrid_scores.append(hybrid_score)
                hybrid_docs.append(doc)

        # Sort by hybrid score (descending)
        sorted_indices = np.argsort(hybrid_scores)[::-1]
        sorted_docs = [hybrid_docs[i] for i in sorted_indices]

        # Return top_k documents
        return sorted_docs[: self.top_k]
