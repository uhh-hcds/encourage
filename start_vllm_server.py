"""Main script for starting the VLLM server."""

import argparse
import os
import subprocess

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

API_KEY = os.getenv("VLLM_API_KEY")
CUDA_VISIBLE_DEVICES = os.getenv("CUDA_VISIBLE_DEVICES")
HUGGINGFACE_CACHE_DIR = os.getenv(
    "HUGGINGFACE_CACHE_DIR", os.path.expanduser("~/.cache/huggingface")
)
HUGGING_FACE_HUB_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")

if not API_KEY:
    raise ValueError(
        "API key not found in the .env file. Please add VLLM_API_KEY=<your_api_key> to the file."
    )


def arg_parser() -> argparse.Namespace:
    """Argument parser for the VLLM server."""
    parser = argparse.ArgumentParser(description="VLLM server")
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="The model to use for the VLLM server.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.95,
        help="The fraction of GPU memory to use for the VLLM server.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Main function for starting the VLLM server."""
    # Ensure the Hugging Face cache directory exists
    if not os.path.exists(HUGGINGFACE_CACHE_DIR):
        os.makedirs(HUGGINGFACE_CACHE_DIR, exist_ok=True)

    # Ensure permissions for the cache directory
    os.chmod(HUGGINGFACE_CACHE_DIR, 0o777)

    # Construct the Docker command
    docker_command = [
        "docker run -d",
        "--runtime nvidia",
        f"--gpus '\"device={CUDA_VISIBLE_DEVICES}\"'",
        "-v ~/.cache/huggingface:/root/.cache/huggingface",
        f"--env HUGGING_FACE_HUB_TOKEN={HUGGING_FACE_HUB_TOKEN}",
        "--env HF_HOME=/root/.cache/huggingface/transformers",
        "-p 8000:8000",
        "--ipc=host",
        "--name vllm-server",
        "vllm/vllm-openai:v0.6.4",
        "--model",
        args.model,
        "--dtype",
        "auto",
        "--api-key",
        API_KEY,
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
    ]

    # Execute the Docker command
    try:
        print(f"Running Docker command: {' '.join(docker_command)}")
        subprocess.run(" ".join(docker_command), shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with error: {e}")


if __name__ == "__main__":
    args = arg_parser()
    main(args)
