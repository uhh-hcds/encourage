"""Batch processing handler for OpenAI completions."""

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Iterator

from openai import OpenAI
from tqdm import tqdm

from encourage.prompts.prompt_collection import PromptCollection


def chunks(lst: list[Any], n: int) -> Iterator[list[Any]]:
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def process_batches(
    client: OpenAI,
    prompt_collection: PromptCollection,
    args: dict[str, Any],
    max_workers: int = 4,
    batch_size: int = 50,
) -> list[Any]:
    """Processes batch_messages in sub-batches using ThreadPoolExecutor.

    Args:
        client (OpenAI): OpenAI client
        prompt_collection (PromptCollection): Collection of prompts to process.
        args (dict): Arguments for OpenAI ChatCompletion.
        max_workers (int): Number of threads for the executor.
        batch_size (int): Size of each sub-batch to process.

    Returns:
        list: List of completion results or exceptions for the submitted tasks.

    """
    all_messages = [prompt.conversation.dialog for prompt in prompt_collection.prompts]
    total_samples = len(all_messages)
    num_batches = (total_samples + batch_size - 1) // batch_size
    all_responses = []

    # Process in batches with progress tracking
    with tqdm(total=total_samples, desc="Processing prompts") as bar:
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, total_samples)
            current_batch_size = end_idx - start_idx

            # Get current batch of messages
            batch_messages = all_messages[start_idx:end_idx]
            bar.set_description(
                f"Batch {batch_idx + 1}/{num_batches} ({current_batch_size} prompts)"
            )

            completions: list[Future] = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for sub_batch in chunks(batch_messages, batch_size):
                    for message_list in sub_batch:
                        kwargs_modified = args.copy()
                        kwargs_modified.pop("max_workers", None)
                        kwargs_modified["messages"] = message_list
                        future = executor.submit(
                            client.chat.completions.create,
                            **kwargs_modified,
                        )
                        completions.append(future)

                results = []
                for future in completions:
                    try:
                        results.append(future.result())

                    except Exception as exc:
                        logging.error(
                            f"Exception occurred while processing future: {exc}", exc_info=True
                        )
                        results.append(exc)

            all_responses.extend(results)
            bar.update(current_batch_size)

    return all_responses
