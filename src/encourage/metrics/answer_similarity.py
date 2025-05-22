"""Estimate the similarity between answer and reference embeddings."""

from typing import Union

import numpy as np
from sentence_transformers import SentenceTransformer

from encourage.llm.response_wrapper import ResponseWrapper
from encourage.metrics.metric import Metric, MetricOutput
from encourage.metrics.registry import register_metric


@register_metric("AnswerSimilarity")
class AnswerSimilarity(Metric):
    """Estimate the similarity between answer and reference embeddings."""

    def __init__(self, model_name: str) -> None:
        super().__init__(
            name="answer_similarity",
            description="Estimate the similarity between answer and reference embeddings.",
            runner=None,  # type: ignore
            required_meta_data=["reference_answer"],
        )
        self.model = SentenceTransformer(model_name)

    def _get_embedding(self, text: Union[list[str], str]) -> np.ndarray:
        if isinstance(text, str):
            return self.model.encode([text])
        return self.model.encode(text)

    def __call__(self, responses: ResponseWrapper) -> MetricOutput:
        """Estimate the similarity between answer and reference embeddings."""
        generated_and_reference = [
            (response.response, response.meta_data["reference_answer"])
            for response in responses
            if response.response is not None
        ]

        if not generated_and_reference:
            return MetricOutput(score=0.0, raw=[])

        generated, reference = zip(*generated_and_reference)
        answer_emb = self._get_embedding(list(generated))
        reference_emb = self._get_embedding(list(reference))

        similarity_matrix = self.model.similarity(answer_emb, reference_emb)
        similarities = similarity_matrix.diagonal().tolist()
        return MetricOutput(score=float(np.mean(similarities)), raw=similarities)
