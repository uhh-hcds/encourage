"""Self-RAG (Retrieval, Generation, and Critique through Self-Reflection) implementation.

Self-RAG is a framework that enhances LLM responses by combining retrieval with self-reflection.
The model determines when to retrieve, critique its own output for factuality, and adopts a
reflection-driven approach to improve response quality.

Reference: https://arxiv.org/abs/2310.11511
"""

import logging
from typing import Any, List, Optional, Tuple, override

from encourage.llm import BatchInferenceRunner, Response, ResponseWrapper
from encourage.prompts import PromptCollection
from encourage.prompts.context import Context, Document
from encourage.prompts.meta_data import MetaData
from encourage.rag.base_impl import BaseRAG
from encourage.utils.llm_mock import create_mock_response_wrapper

logger = logging.getLogger(__name__)


class SelfRAG(BaseRAG):
    """Self-RAG implementation based on the paper by Meta AI Research.

    This implementation enhances traditional RAG with self-reflection capabilities:
    1. Retrieval: Standard vector retrieval of relevant documents
    2. Generation: Initial response generation based on context
    3. Reflection: Self-critique of the response for factuality and relevance
    4. Refinement: Improving the response based on reflection
    """

    def __init__(
        self,
        context_collection: list[Document],
        collection_name: str,
        embedding_function: Any,
        top_k: int,
        retrieval_only: bool = False,
        device: str = "cuda",
        where: dict[str, str] | None = None,
        runner: BatchInferenceRunner | None = None,
        additional_prompt: str = "",
        template_name: str = "",
        reflection_rounds: int = 1,
        **kwargs: Any,
    ) -> None:
        """Initialize Self-RAG method.

        Args:
            context_collection: List of documents for context
            collection_name: Name for the vector collection
            embedding_function: Function to create embeddings
            top_k: Number of documents to retrieve
            retrieval_only: Whether to only do retrieval without generation
            device: Computing device (cuda/cpu)
            where: Optional filtering criteria
            runner: LLM runner for inference
            additional_prompt: Extra prompt instructions
            template_name: Template to use
            reflection_rounds: Number of reflection-refinement cycles
            **kwargs: Additional arguments

        """
        super().__init__(
            context_collection=context_collection,
            collection_name=collection_name,
            embedding_function=embedding_function,
            top_k=top_k,
            retrieval_only=retrieval_only,
            device=device,
            where=where,
            runner=runner,
            additional_prompt=additional_prompt,
            template_name=template_name,
        )

        self.reflection_rounds = reflection_rounds
        self.reflection_sys_prompt = (
            "You are a critical evaluator. Analyze the response for factuality, relevance, "
            "coherence, and information completeness. Identify any hallucinations or missing "
            "important information from the retrieved context."
        )

        self.refinement_sys_prompt = (
            "You are an expert assistant. Improve your previous response based on the critical "
            "feedback and retrieved information. Focus on factuality, completeness, and coherence."
        )

    def _generate_reflection(
        self,
        runner: BatchInferenceRunner,
        query: str,
        response: str,
        context: Optional[Context] = None,
    ) -> str:
        """Generate a self-reflection on the current response.

        Args:
            runner: LLM runner
            query: Original user query
            response: Current response to evaluate
            context: Retrieved context (if available)

        Returns:
            Reflection text critiquing the response

        """
        reflection_user_prompt = f"QUERY: {query}\n\nRESPONSE: {response}\n\n"

        if context:
            reflection_user_prompt += f"CONTEXT: {context.to_string()}\n\n"

        reflection_user_prompt += (
            "TASK: Evaluate the response for:\n"
            "1. Factual accuracy (especially compared to the context)\n"
            "2. Completeness of information\n"
            "3. Relevance to the query\n"
            "4. Coherence and clarity\n\n"
            "Provide specific issues that need improvement."
        )

        reflection_prompts = PromptCollection.create_prompts(
            sys_prompts=self.reflection_sys_prompt, user_prompts=[reflection_user_prompt]
        )

        reflection_response = runner.run(reflection_prompts)
        return reflection_response.get_responses()[0]

    def _generate_refined_response(
        self,
        runner: BatchInferenceRunner,
        query: str,
        initial_response: str,
        reflection: str,
        context: Optional[Context] = None,
    ) -> str:
        """Generate an improved response based on reflection.

        Args:
            runner: LLM runner
            query: Original user query
            initial_response: Initial response to refine
            reflection: Critical feedback from reflection
            context: Retrieved context (if available)

        Returns:
            Refined response text

        """
        refinement_user_prompt = (
            f"QUERY: {query}\n\n"
            f"INITIAL RESPONSE: {initial_response}\n\n"
            f"CRITICAL FEEDBACK: {reflection}\n\n"
        )

        if context:
            refinement_user_prompt += f"CONTEXT: {context.to_string()}\n\n"

        refinement_user_prompt += (
            "TASK: Provide an improved response that addresses the critical feedback. "
            "Focus on factuality, completeness, and directly answering the query."
        )

        refinement_prompts = PromptCollection.create_prompts(
            sys_prompts=self.refinement_sys_prompt, user_prompts=[refinement_user_prompt]
        )

        refined_response = runner.run(refinement_prompts)
        return refined_response.get_responses()[0]

    def _process_single_query(
        self,
        runner: BatchInferenceRunner,
        query: str,
        context: Optional[Context],
        meta_data: MetaData,
        sys_prompt: str,
        template_name: str,
    ) -> Tuple[ResponseWrapper, List[str]]:
        """Process a single query through the Self-RAG pipeline.

        Args:
            runner: LLM runner
            query: User query
            context: Retrieved context (if available)
            meta_data: Metadata for this query
            sys_prompt: System prompt
            template_name: Template name

        Returns:
            Tuple of (final response wrapper, list of reflection texts)

        """
        # Initial response generation
        initial_prompt_collection = PromptCollection.create_prompts(
            sys_prompts=sys_prompt,
            user_prompts=[query],
            contexts=[context] if context else [],
            meta_datas=[meta_data] if meta_data else [],
            template_name=template_name if template_name else self.template_name,
        )

        response_wrapper = runner.run(initial_prompt_collection)
        response_text = response_wrapper.get_responses()[0]

        # Store reflections for this query
        query_reflections = []

        # Reflection and refinement rounds
        for round_idx in range(self.reflection_rounds):
            logger.info(
                f"""
                Starting reflection round {round_idx + 1}/{self.reflection_rounds}
                for query: {query[:50]}...
                """
            )

            # Generate reflection
            reflection = self._generate_reflection(
                runner=runner, query=query, response=response_text, context=context
            )
            query_reflections.append(reflection)

            # Generate refined response
            response_text = self._generate_refined_response(
                runner=runner,
                query=query,
                initial_response=response_text,
                reflection=reflection,
                context=context,
            )

        # When we finish reflection rounds, create a fresh response with the final text
        # Rather than modifying the original response_wrapper which we don't have direct access to
        final_response = Response(
            request_id=f"self-rag-{query[:10]}",
            prompt_id="",  # We're not tracking the original prompt ID
            conversation_id=0,
            sys_prompt=sys_prompt,
            user_prompt=query,
            response=response_text,
            context=context,
            meta_data=meta_data,
            arrival_time=0.0,
            finished_time=0.0,
        )

        return ResponseWrapper([final_response]), query_reflections

    def _retrieve_and_setup_contexts(
        self,
        user_prompts: List[str],
        retrieval_queries: List[str],
    ) -> List[Context]:
        """Retrieve relevant contexts for the given queries.

        Args:
            user_prompts: User prompts/queries
            retrieval_queries: Optional separate retrieval queries

        Returns:
            List of retrieved contexts

        """
        query_list = retrieval_queries if retrieval_queries else user_prompts

        if not query_list:
            logger.warning("No queries provided for retrieval")
            return []

        # Retrieve contexts
        retrieved_documents = self.retrieve_contexts(query_list)
        return [Context.from_documents(docs) for docs in retrieved_documents]

    @override
    def run(
        self,
        runner: BatchInferenceRunner,
        sys_prompt: str,
        user_prompts: List[str] = [],
        meta_data: List[MetaData] = [],
        retrieval_queries: List[str] = [],
        template_name: str = "",
    ) -> ResponseWrapper:
        """Execute Self-RAG pipeline of retrieval, generation, reflection and refinement.

        Args:
            runner: LLM runner for inference
            sys_prompt: System prompt for generation
            user_prompts: User queries
            meta_data: Metadata for prompts
            retrieval_queries: Optional separate retrieval queries
            template_name: Template name to use

        Returns:
            Response wrapper with final responses and metadata

        """
        # Retrieve contexts
        contexts = self._retrieve_and_setup_contexts(user_prompts, retrieval_queries)

        # Handle retrieval-only mode
        if self.retrieval_only:
            logger.info("Retrieval-only mode: Skipping LLM inference.")
            prompt_collection = PromptCollection.create_prompts(
                sys_prompts=sys_prompt,
                user_prompts=user_prompts,
                contexts=contexts,
                meta_datas=meta_data,
                template_name=template_name if template_name else self.template_name,
            )
            return create_mock_response_wrapper(prompt_collection)

        # Check if we have any user prompts
        if not user_prompts:
            logger.warning("No user prompts provided")
            return create_mock_response_wrapper(PromptCollection())

        # If we only have one prompt, process it directly
        if len(user_prompts) == 1:
            context = contexts[0] if contexts else None
            current_meta_data = meta_data[0] if meta_data else {}

            response_wrapper, reflections = self._process_single_query(
                runner=runner,
                query=user_prompts[0],
                context=context,
                meta_data=current_meta_data,
                sys_prompt=sys_prompt,
                template_name=template_name,
            )

            # Add reflections to metadata
            if meta_data:
                meta_data[0]["self_rag_reflections"] = reflections

            return response_wrapper

        # Process multiple queries
        responses = []
        all_meta_data = meta_data.copy() if meta_data else [{} for _ in user_prompts]

        for idx, query in enumerate(user_prompts):
            context = contexts[idx] if idx < len(contexts) else None
            current_meta_data = all_meta_data[idx] if idx < len(all_meta_data) else {}

            # Process query through the self-reflection pipeline
            response_wrapper, reflections = self._process_single_query(
                runner=runner,
                query=query,
                context=context,
                meta_data=current_meta_data,
                sys_prompt=sys_prompt,
                template_name=template_name,
            )

            # Store the response
            responses.append(response_wrapper.get_responses()[0])

            # Add reflections to metadata
            if idx < len(all_meta_data):
                all_meta_data[idx]["self_rag_reflections"] = reflections

        # Use the correct way to create a ResponseWrapper with multiple responses
        combined_response_wrapper = ResponseWrapper(
            [
                Response(
                    request_id=f"self-rag-{i}",
                    prompt_id="",
                    conversation_id=i,
                    sys_prompt=sys_prompt,
                    user_prompt=user_prompts[i] if i < len(user_prompts) else "",
                    response=responses[i],
                    context=contexts[i] if i < len(contexts) else None,
                    meta_data=all_meta_data[i] if i < len(all_meta_data) else {},
                    arrival_time=0.0,
                    finished_time=0.0,
                )
                for i in range(len(responses))
            ]
        )

        return combined_response_wrapper
