"""Batch processing handler for OpenAI completions."""

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from openai import OpenAI


def process_batches(
    client: OpenAI,
    batch_messages: list[list[dict[str, str]]],
    args: dict[str, Any],
    max_workers: int = 4,
) -> list[Any]:
    """Processes batch_messages in sub-batches using ThreadPoolExecutor.

    Args:
        client (OpenAI): OpenAI client
        batch_messages (list): List of message lists to process.
        args (dict): Arguments for OpenAI ChatCompletion.
        max_workers (int): Number of threads for the executor.

    Returns:
        list: List of completion results or exceptions for the submitted tasks.

    """
    from typing import Iterator

    def chunks(lst: list[Any], n: int) -> Iterator[list[Any]]:
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    completions: list[Future] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        print(f"Processing {len(batch_messages)} batches with {max_workers} workers.")
        for sub_batch in chunks(batch_messages, 100):
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
            logging.error(f"Exception occurred while processing future: {exc}", exc_info=True)
            results.append(exc)

    return results
