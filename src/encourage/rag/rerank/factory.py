"""Factory to create the correct reranker subclass."""

from encourage.rag.rerank.base import Reranker, RerankerModel
from encourage.rag.rerank.models import JinaV3, MSMarco


class RerankerFactory:
    """Factory to create the correct reranker subclass."""

    @staticmethod
    def create(rerank_ratio: float, model: str, device: str = "cuda") -> Reranker:
        """Create a Reranker instance based on the specified model."""
        if isinstance(model, str):
            try:
                model = RerankerModel(model)
            except ValueError as err:
                raise ValueError(f"Unsupported RerankerModel: {model}") from err

        if model == RerankerModel.MS_MARCO:
            return MSMarco(rerank_ratio=rerank_ratio, device=device)
        elif model == RerankerModel.JINA_V3:
            return JinaV3(rerank_ratio=rerank_ratio, device=device)
        else:
            raise ValueError(f"Unsupported RerankerModel: {model}")
