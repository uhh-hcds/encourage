"""Generate mock responses for testing purposes."""

import random
import uuid
from typing import Optional

from faker import Faker
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage

from encourage.llm import Response, ResponseWrapper
from encourage.llm.vllm_classes import CompletionOutput, RequestOutput
from encourage.prompts import Context, Document, MetaData, PromptCollection

fake: Faker = Faker()


def create_responses(
    n: int = 1,
    response_content_list: Optional[list[str]] = None,
    document_list: Optional[list[list[Document]]] = None,
    meta_data_list: Optional[list[MetaData]] = None,
) -> list[Response]:
    """Generate a list of random realistic Response objects using Faker.

    Args:
        n: Number of Response objects to generate.
        response_content_list: Optional list of response contents.
        document_list: Optional list of lists of Documents for each response.
        meta_data_list: Optional list of MetaData objects for each response.



    Returns:
        list[Response]: A list of populated mock Responses.

    """
    responses: list[Response] = []

    if response_content_list is not None and len(response_content_list) != n:
        raise ValueError("Length of response_content_list must be equal to n")

    ## Handle response content
    response_content = (
        response_content_list
        if response_content_list is not None
        else [fake.paragraph(nb_sentences=4) for _ in range(n)]
    )

    ## Generate random documents for each response
    documents_generated: list[list[Document]] = [
        [
            Document(
                id=uuid.uuid4(),
                content=fake.paragraph(nb_sentences=3),
                score=round(random.uniform(0.0, 1.0), 2),
            )
            for _ in range(random.randint(1, 5))
        ]
        for _ in range(n)
    ]
    if document_list is not None and len(document_list) != n:
        raise ValueError("Length of document_list must be equal to n")
    if document_list is not None:
        documents_generated = document_list

    ## Handle MetaData
    meta_data_generated: list[MetaData] = [
        MetaData(
            tags={
                "reference_answer": fake.sentence(nb_words=10),
                "reference_document": random.choice(documents_generated[i]),
            }
        )
        for i in range(n)
    ]
    if meta_data_list is not None and len(meta_data_list) != n:
        raise ValueError("Length of meta_data_list must be equal to n")
    if meta_data_list is not None:
        meta_data_generated = meta_data_list

    for i in range(n):
        response = Response(
            request_id=str(uuid.uuid4()),
            prompt_id=fake.word(),
            sys_prompt=fake.sentence(nb_words=6),
            user_prompt=fake.sentence(nb_words=8),
            response=response_content[i],
            conversation_id=random.randint(1, 100),
            meta_data=meta_data_generated[i],
            context=Context.from_documents(documents_generated[i]),
            arrival_time=random.uniform(0, 5),
            finished_time=random.uniform(5, 10),
        )
        responses.append(response)

    return responses


def create_prompt_collection(n_prompts: int) -> PromptCollection:
    """Create a PromptCollection with random system/user prompts, contexts, and meta_data.

    Args:
        n_prompts: Number of user prompts to generate.

    Returns:
        PromptCollection: A populated prompt collection.

    """
    sys_prompt: str = fake.sentence(nb_words=6)
    user_prompts: list[str] = [fake.sentence(nb_words=8) for _ in range(n_prompts)]
    template_name: str = "llama3_conv.j2"

    # Generate fake contexts and meta_data for each prompt
    contexts = []
    meta_datas = []
    for _ in range(n_prompts):
        documents = [
            Document(
                id=uuid.uuid4(),
                content=fake.paragraph(nb_sentences=3),
                score=round(random.uniform(0.0, 1.0), 2),
            )
            for _ in range(random.randint(1, 3))
        ]
        contexts.append(Context.from_documents(documents))
        meta_datas.append(
            MetaData(
                tags={
                    "reference_answer": fake.sentence(nb_words=10),
                    "reference_document": random.choice(documents),
                }
            )
        )

    return PromptCollection.create_prompts(
        sys_prompts=sys_prompt,
        user_prompts=user_prompts,
        template_name=template_name,
        contexts=contexts,
        meta_datas=meta_datas,
    )


def create_mock_chatcompletions(
    n: int = 2,
    contents: Optional[list[str]] = None,
    model_name: str = "gpt-4",
    created_ts: int = 1000,
) -> list[ChatCompletion]:
    """Generate n mock ChatCompletion objects using the OpenAI SDK types.

    Args:
        n: Number of ChatCompletion objects to generate.
        contents: Optional list of message contents. If fewer than n, repeated or defaults used.
        model_name: Model name to use.
        created_ts: Unix timestamp of creation.

    Returns:
        list[ChatCompletion]: List of mocked completions.

    """
    if contents is None:
        contents = [f"Response text {i + 1}" for i in range(n)]

    completions: list[ChatCompletion] = []

    for i in range(n):
        content: str = contents[i] if i < len(contents) else contents[-1]
        message: ChatCompletionMessage = ChatCompletionMessage(role="assistant", content=content)
        choice: Choice = Choice(message=message, index=i, finish_reason="stop")

        chat_completion: ChatCompletion = ChatCompletion(
            id=str(uuid.uuid4()),
            choices=[choice],
            created=created_ts,
            model=model_name,
            object="chat.completion",
        )
        completions.append(chat_completion)

    return completions


def create_mock_request_outputs(
    n: int = 2,
    contents: Optional[list[str]] = None,
) -> list[RequestOutput]:
    """Generate n mock RequestOutput objects."""
    completion_outputs: list[CompletionOutput] = []
    completion_outputs = [
        CompletionOutput(
            i,
            text=contents[i] if contents and i < len(contents) else f"Response {i + 1}",
            token_ids=[1, 2, 3],
            cumulative_logprob=None,
            logprobs=None,
        )
        for i in range(n)
    ]

    request_outputs: list[RequestOutput] = []

    for i in range(n):
        request_output: RequestOutput = RequestOutput(
            request_id=str(uuid.uuid4()),
            prompt="",
            prompt_token_ids=[],
            prompt_logprobs=[],
            finished=True,
            outputs=[completion_outputs[i]],
        )
        request_outputs.append(request_output)

    return request_outputs


def create_response_wrapper(prompt_response_pairs: int) -> ResponseWrapper:
    """Create a ResponseWrapper using mock ChatCompletions and a PromptCollection.

    Returns:
        ResponseWrapper: Wrapper containing responses and prompt info.

    """
    prompt_collection: PromptCollection = create_prompt_collection(prompt_response_pairs)
    chat_completions: list[ChatCompletion] = create_mock_chatcompletions(prompt_response_pairs)
    return ResponseWrapper.from_prompt_collection(chat_completions, prompt_collection)
