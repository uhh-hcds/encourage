"""Module containing RetrievalOnlyRAG implementation that only performs retrieval."""

import logging

from openai.types.chat.chat_completion import ChatCompletion
from openai.types.completion_usage import CompletionUsage

from encourage.llm import BatchInferenceRunner, ResponseWrapper
from encourage.prompts import PromptCollection
from encourage.rag.naive import NaiveRAG

logger = logging.getLogger(__name__)


class RetrievalOnlyRAG(NaiveRAG):
    """Implementation of RAG that only performs retrieval without calling the LLM."""

    def run(
        self,
        runner: BatchInferenceRunner,
        sys_prompt: str,
        user_prompts: list[str] = [],
        retrieval_instruction: list[str] = [],
    ) -> ResponseWrapper:
        """Execute only the retrieval part of the RAG pipeline and return mock responses.

        This implementation retrieves relevant documents but does not call the LLM.
        Instead, it returns mock responses with the retrieved context information.

        Args:
            runner: The BatchInferenceRunner that would normally be used
            sys_prompt: The system prompt to use
            user_prompts: List of user prompts to process (defaults to prompts from the dataset)
            retrieval_instruction: List of retrieval queries to use for retrieving contexts
            mock_content: Optional custom content for mock responses

        Returns:
            ResponseWrapper containing mock responses with context information

        """
        # Generate queries and retrieve contexts
        if retrieval_instruction:
            logger.info(f"Generating {len(retrieval_instruction)} retrieval queries.")
            self.contexts = self._get_contexts_from_db(retrieval_instruction)
        else:
            logger.info("No context retrieval queries provided. Using no context.")
            self.contexts = []

        user_prompts = user_prompts if user_prompts else self.user_prompts

        # Create prompt collection just like in normal RAG (but we won't run inference)
        prompt_collection = PromptCollection.create_prompts(
            sys_prompts=sys_prompt,
            user_prompts=user_prompts,
            contexts=self.contexts,
            meta_datas=self.metadata,
            template_name=self.template_name,
        )

        # Create mock ChatCompletion objects for each prompt using the provided example structure
        mock_llm_response = [
            ChatCompletion(
                id=f"mock-response-{i}",
                choices=[],  # Empty choices as per example
                created=0,
                model="mock",
                object="chat.completion",
                system_fingerprint="",
                usage=CompletionUsage(
                    completion_tokens=0,
                    prompt_tokens=0,
                    total_tokens=0,
                    completion_tokens_details=None,
                    prompt_tokens_details=None,
                ),
            )
            for i in range(len(prompt_collection.prompts))
        ]

        # Create and return ResponseWrapper
        return ResponseWrapper.from_prompt_collection(mock_llm_response, prompt_collection)
