"""Module implementing a reranker that uses a cross-encoder model to improve retrieval quality."""

import logging
from typing import override

import torch
from sentence_transformers import CrossEncoder
from transformers import AutoModel

from encourage.prompts.context import Document
from encourage.rag.rerank.base import Reranker

logger = logging.getLogger(__name__)


class MSMarco(Reranker):
    """Reranker using the MS MARCO cross-encoder model."""

    def __init__(self, rerank_ratio: float, device: str = "cuda") -> None:
        super().__init__(rerank_ratio=rerank_ratio, device=device)
        self.reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)

    @override
    def rerank_documents(self, query: str, documents: list[Document], top_k: int) -> list[Document]:
        """Rerank documents based on relevance to the query.

        Args:
            query: The query to rank documents against
            documents: list of documents to rerank
            top_k: Number of documents to return after reranking

        Returns:
            list of documents reranked by relevance

        """
        if not documents:
            logger.warning(f"No documents to rerank for query: '{query}'")
            return []

        scores = self.reranker_model.predict([(query, doc.content) for doc in documents])
        scored_documents = sorted(zip(scores, documents), key=lambda pair: pair[0], reverse=True)
        top_documents = [doc for _, doc in scored_documents[:top_k]]
        return top_documents


class JinaV3(Reranker):
    """Reranker using the Jina V3 cross-encoder model."""

    def __init__(self, rerank_ratio: float, device: str = "cuda") -> None:
        super().__init__(rerank_ratio=rerank_ratio, device=device)
        self.reranker_model = AutoModel.from_pretrained(
            "jinaai/jina-reranker-v3",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

    @override
    def rerank_documents(self, query: str, documents: list[Document], top_k: int) -> list[Document]:
        """Rerank documents based on relevance to the query.

        Args:
            query: The query to rank documents against
            documents: list of documents to rerank
            top_k: Number of documents to return after reranking

        Returns:
            list of documents reranked by relevance

        """
        if not documents:
            logger.warning(f"No documents to rerank for query: '{query}'")
            return []

        # Get only top_k results
        document_contents = [doc.content for doc in documents]
        results = self.reranker_model.rerank(query, document_contents, top_n=top_k)

        top_documents = []
        for result in results:
            original_doc = documents[result["index"]]
            original_doc.distance = result["relevance_score"]
            top_documents.append(original_doc)

        return top_documents
