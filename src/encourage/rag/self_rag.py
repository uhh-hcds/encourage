"""Self-RAG (Retrieval, Generation, and Critique through Self-Reflection) implementation.

Self-RAG is a framework that enhances LLM responses by combining retrieval with self-reflection.
The model determines when to retrieve, critique its own output for factuality, and adopts a
reflection-driven approach to improve response quality.

Reference: https://arxiv.org/abs/2310.11511
"""

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, override

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

    def _generate_batch_reflection(
        self,
        runner: BatchInferenceRunner,
        queries: List[str],
        responses: List[str],
        contexts: Sequence[Optional[Context]],
    ) -> List[str]:
        """Generate batch reflections on the current responses.

        Args:
            runner: LLM runner
            queries: Original user queries
            responses: Current responses to evaluate
            contexts: Retrieved contexts (if available)

        Returns:
            List of reflection texts critiquing each response

        """
        reflection_user_prompts = []

        for i, (query, response) in enumerate(zip(queries, responses)):
            prompt = f"QUERY: {query}\n\nRESPONSE: {response}\n\n"

            # Safely handle None contexts by checking before calling to_string()
            context = contexts[i] if i < len(contexts) else None
            if context is not None:
                prompt += f"CONTEXT: {context.to_string()}\n\n"

            prompt += (
                "TASK: Evaluate the response for:\n"
                "1. Factual accuracy (especially compared to the context)\n"
                "2. Completeness of information\n"
                "3. Relevance to the query\n"
                "4. Coherence and clarity\n\n"
                "Provide specific issues that need improvement."
            )
            reflection_user_prompts.append(prompt)

        # Create prompt collection for reflections
        reflection_prompts = PromptCollection.create_prompts(
            sys_prompts=self.reflection_sys_prompt, user_prompts=reflection_user_prompts
        )

        # Run batch inference for reflections
        reflection_response_wrapper = runner.run(reflection_prompts)
        return reflection_response_wrapper.get_responses()

    def _generate_batch_refined_responses(
        self,
        runner: BatchInferenceRunner,
        queries: List[str],
        initial_responses: List[str],
        reflections: List[str],
        contexts: Sequence[Optional[Context]],
    ) -> List[str]:
        """Generate improved responses based on reflection in batch.

        Args:
            runner: LLM runner
            queries: Original user queries
            initial_responses: Initial responses to refine
            reflections: Critical feedback from reflections
            contexts: Retrieved contexts (if available)

        Returns:
            List of refined response texts

        """
        refinement_user_prompts = []

        for i, (query, response, reflection) in enumerate(
            zip(queries, initial_responses, reflections)
        ):
            prompt = (
                f"QUERY: {query}\n\n"
                f"INITIAL RESPONSE: {response}\n\n"
                f"CRITICAL FEEDBACK: {reflection}\n\n"
            )

            # Safely handle None contexts by checking before calling to_string()
            context = contexts[i] if i < len(contexts) else None
            if context is not None:
                prompt += f"CONTEXT: {context.to_string()}\n\n"

            prompt += (
                "TASK: Provide an improved response that addresses the critical feedback. "
                "Focus on factuality, completeness, and directly answering the query."
            )
            refinement_user_prompts.append(prompt)

        # Create prompt collection for refinements
        refinement_prompts = PromptCollection.create_prompts(
            sys_prompts=self.refinement_sys_prompt, user_prompts=refinement_user_prompts
        )

        # Run batch inference for refined responses
        refined_response_wrapper = runner.run(refinement_prompts)
        return refined_response_wrapper.get_responses()

    def _batch_process_with_self_reflection(
        self,
        runner: BatchInferenceRunner,
        user_prompts: List[str],
        contexts: Sequence[Optional[Context]],
        meta_datas: List[MetaData],
        sys_prompt: str,
        template_name: str,
    ) -> Tuple[List[str], Dict[int, List[str]]]:
        """Process queries through the Self-RAG pipeline in batches.

        Args:
            runner: LLM runner
            user_prompts: List of user queries
            contexts: List of retrieved contexts
            meta_datas: List of metadata
            sys_prompt: System prompt
            template_name: Template name

        Returns:
            Tuple of (final responses, reflections by query index)

        """
        # Initial batch generation
        valid_contexts = [ctx for ctx in contexts if ctx is not None]
        initial_prompt_collection = PromptCollection.create_prompts(
            sys_prompts=sys_prompt,
            user_prompts=user_prompts,
            contexts=valid_contexts,
            meta_datas=meta_datas,
            template_name=template_name if template_name else self.template_name,
        )

        response_wrapper = runner.run(initial_prompt_collection)
        responses = response_wrapper.get_responses()

        # Store reflections for each query
        all_reflections: Dict[int, List[str]] = {i: [] for i in range(len(user_prompts))}

        # Reflection and refinement rounds
        current_responses = responses
        for _ in range(self.reflection_rounds):
            # Generate batch reflections
            reflections = self._generate_batch_reflection(
                runner=runner,
                queries=user_prompts,
                responses=current_responses,
                contexts=contexts,
            )

            # Store reflections
            for i, reflection in enumerate(reflections):
                if i in all_reflections:
                    all_reflections[i].append(reflection)

            # Generate batch refined responses
            current_responses = self._generate_batch_refined_responses(
                runner=runner,
                queries=user_prompts,
                initial_responses=current_responses,
                reflections=reflections,
                contexts=contexts,
            )

        return current_responses, all_reflections

    def _retrieve_and_setup_contexts(
        self,
        user_prompts: List[str],
        retrieval_queries: List[str],
    ) -> List[Optional[Context]]:
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
        return [Context.from_documents(docs) if docs else None for docs in retrieved_documents]

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

        # Get valid contexts for prompt collection
        valid_contexts = [ctx for ctx in contexts if ctx is not None]

        # Handle retrieval-only mode
        if self.retrieval_only:
            logger.info("Retrieval-only mode: Skipping LLM inference.")
            prompt_collection = PromptCollection.create_prompts(
                sys_prompts=sys_prompt,
                user_prompts=user_prompts,
                contexts=valid_contexts,
                meta_datas=meta_data,
                template_name=template_name if template_name else self.template_name,
            )
            return create_mock_response_wrapper(prompt_collection)

        # Check if we have any user prompts
        if not user_prompts:
            logger.warning("No user prompts provided")
            return create_mock_response_wrapper(PromptCollection(prompts=[]))

        # Make sure meta_data is populated
        all_meta_data = meta_data.copy() if meta_data else [MetaData() for _ in user_prompts]

        # Process all queries through batch self-reflection pipeline
        final_responses, all_reflections = self._batch_process_with_self_reflection(
            runner=runner,
            user_prompts=user_prompts,
            contexts=contexts,
            meta_datas=all_meta_data,
            sys_prompt=sys_prompt,
            template_name=template_name,
        )

        # Add reflections to metadata
        for idx, reflections in all_reflections.items():
            if idx < len(all_meta_data):
                # This ensures we're not trying to assign a list[str] to a str
                all_meta_data[idx].data["self_rag_reflections"] = reflections

        # Create Response objects
        response_objects = []
        for i, response_text in enumerate(final_responses):
            # Fix: Always use a valid Context object, never None
            context_obj = Context()  # Create default context
            if i < len(contexts) and contexts[i] is not None:
                context_obj = contexts[i]

            current_meta_data = all_meta_data[i] if i < len(all_meta_data) else MetaData()
            user_prompt = user_prompts[i] if i < len(user_prompts) else ""

            response_objects.append(
                Response(
                    request_id=f"self-rag-{i}",
                    prompt_id="",
                    conversation_id=i,
                    sys_prompt=sys_prompt,
                    user_prompt=user_prompt,
                    response=response_text,
                    context=context_obj,  # Now guaranteed to be a Context, not None
                    meta_data=current_meta_data,
                    arrival_time=0.0,
                    finished_time=0.0,
                )
            )

        return ResponseWrapper(response_objects)
