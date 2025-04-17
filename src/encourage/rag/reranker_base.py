"""Module implementing a reranker that uses a cross-encoder model to improve retrieval quality."""

import logging

from sentence_transformers import CrossEncoder

from encourage.prompts.context import Document

logger = logging.getLogger(__name__)


class Reranker:
    """A reranker that uses a cross-encoder model to improve retrieval quality.

    This class provides common functionality for reranking documents based on relevance
    scoring using a cross-encoder model.

    Attributes:
        reranker_model (str): Name or path of the cross-encoder reranker model
        rerank_ratio (float): Multiplier for how many documents to initially retrieve compared
        to top_k
        reranker (CrossEncoder): The cross-encoder model used for reranking
        device (str): Device to use for the reranker model

    """

    def __init__(
        self,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        rerank_ratio: float = 3.0,
        device: str = "cuda",
    ) -> None:
        """Initialize a Reranker with configuration.

        Args:
            reranker_model: Name or path of the cross-encoder reranker model
            rerank_ratio: How many times more documents to retrieve initially for reranking
                (e.g., 3.0 means retrieve 3*top_k documents initially)
            device: Device to use for reranker model

        """
        self.reranker_model = reranker_model
        self.rerank_ratio = rerank_ratio
        self.device = device

        # Load the reranker model
        logger.info(f"Loading reranker model: {reranker_model}")
        self.reranker = CrossEncoder(reranker_model, device=device)

    def calculate_initial_top_k(self, top_k: int) -> int:
        """Calculate how many documents to retrieve initially for reranking."""
        return max(int(top_k * self.rerank_ratio), top_k + 2)

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

        scores = self.reranker.predict([(query, doc.content) for doc in documents])
        return [doc for _, doc in sorted(zip(scores, documents), reverse=True)[:top_k]]
