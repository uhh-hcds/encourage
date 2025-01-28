# **VLLM Server Startup Script**

## **Overview**
This script starts a Docker-based VLLM (Vast Language Model) OpenAI server using the `vllm/vllm-openai:v0.6.4` Docker image. It allows you to specify the model and GPU memory utilization via command-line arguments and loads configuration from a `.env` file.

## **Prerequisites**

- **Docker** with NVIDIA GPU support (`nvidia-docker`)
- **Python 3.x**


## **Environment Variables**

Create a .env file in the root directory of the project with the following environment variables:

```bash
VLLM_API_KEY=your_api_key
CUDA_VISIBLE_DEVICES=0        # Optional: specify GPUs (e.g., 0,1)
HUGGINGFACE_CACHE_DIR=~/.cache/huggingface  # Optional: path to cache
HUGGING_FACE_HUB_TOKEN=your_hugging_face_token
```

## **Usage**
### Command-Line Arguments

- --model: The model to use (e.g., meta-llama/Meta-Llama-3.1-8B-Instruct). Default: meta-llama/Meta-Llama-3.1-8B-Instruct.
- --gpu-memory-utilization: The fraction of GPU memory to use. Default: 0.95.

## **Running the Script**

You can start the VLLM server by running the VSCode configuration `Start VLLM Server` or using the following command:

```bash
python vllm_server.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --gpu-memory-utilization 0.95
```
