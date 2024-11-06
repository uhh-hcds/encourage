# How to use BatchInferenceRunner

This tutorial demonstrates how to use `ChatInferenceRunner` and `BatchInferenceRunner` with different metadata, contexts, and user prompts, utilizing a language model.
`ChatInferenceRunner` is used for running inference on conversational data for Few-Shot settings, while `BatchInferenceRunner` is used for running inference on a batch of prompts indeed it is useful for evaluation from one Zero-shot settings.

Initialize the LLM model and sampling parameters:

```python
from vllm import LLM, SamplingParams
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

llm = LLM(model=model_name, gpu_memory_utilization=0.95)
sampling_params = SamplingParams(temperature=0.5, max_tokens=1000)
```

## BatchInferenceRunner

```python
from encourage.llm import BatchInferenceRunner

# For initializing the ChatInferenceRunner, you need to provide the LLM model and sampling parameters.
runner = BatchInferenceRunner(llm, sampling_params)
```

To create a `PromptCollection`, you need to include a list of `Prompt` objects.
A PromptCollection is a set of Prompt objects that contain all the information needed to run inference tasks. Think of each Prompt as a request for the language model to respond to, along with some additional information like system and user prompts, context, and metadata.

Each Prompt object has the following attributes:

```python
@dataclass
class Prompt:
    """Prompt dataclass to store prompt information."""
    
    sys_prompt: str         # The system's instruction (e.g., how the AI should behave)
    user_prompt: str        # The user's question or request
    id: str                 # A unique identifier for each prompt
    conversation_id: int    # An identifier for tracking conversations
    context: list[dict]     # A list of dictionaries with additional context information
    meta_data: list[dict]   # A list of dictionaries with metadata related to the prompt
    reformated: str         # Reformatted prompt data (used internally)
```

You can create a PromptCollection with the `create_prompts()` method:

```python
# Define the system prompt (AI behavior or persona)
sys_prompts = "You are a helpful AI that only speaks like a pirate"

# List of user prompts (questions or requests for the AI)
user_prompts = ["User prompt 1", "User prompt 2"] * 5

# Context information for each prompt (additional data or background info)
contexts = [{"key1": "value1"}, {"key2": "value2"}] * 5

# Metadata associated with each prompt (e.g., priority, tags)
meta_datas = [{"meta": "data1"}, {"meta": "data2"}] * 5

# Create a PromptCollection using the create_prompts method
prompt_collection = PromptCollection.create_prompts(
    sys_prompts=sys_prompts,  # System prompt or list of system prompts
    user_prompts=user_prompts,  # List of user prompts
    contexts=contexts,  # List of context dictionaries (optional)
    meta_datas=meta_datas,  # List of metadata dictionaries (optional)
    model_name=model_name,  # The name of the model being used (optional)
    template_name="template_name",  # The name of the template being used (optional)
)
```

If you want to run inference on the prompts, you can use the `run()` method:

```python
# Run inference on the prompt collection
responses = runner.run(prompt_collection)
```

The Output format is a list of Response objects that are internally created with all information.
You can access the response data using the `response.response_data` attribute.
If you want to print the response data, you can use the `print_response_summary()` method:

```python
responses.print_response_summary()
```

The output will look like this:

<details>
  <summary>Example Preview:</summary>

```bash
Processed prompts: 100%|██████████| 10/10 [00:04<00:00,  2.40it/s, est. speed input: 93.57 toks/s, output: 181.87 toks/s]
--------------------------------------------------
🧑‍💻 User Prompt:
User prompt 1
📚 Added Context: {'key1': 'value1'} (See Template for details.)

💬 Response:
Yer lookin' fer a treasure of knowledge, eh? Alright then, matey! I'll give ye the value o' "value1". It be a curious term, but I'll do me best to give ye the lowdown.

🤖 System Prompt:
You are an helpful AI that only speaks like a pirat
🗂️ Metadata: {'meta': 'data1'}
🆔 Request ID: 0
🆔 Prompt ID: 091a6f1e-c8ec-44cd-a1ba-7a0c8a88361a
🆔 Conversation ID: 0
⏳ Processing Time: 4.1766 seconds
```

</details>

### BatchInferenceRunner with a structured output format 

If you want to use a structured output format, you can use the `schema` parameter in the `run()` method:

```python
# Run inference on the prompt collection with structured output
from pydantic import BaseModel

class User(BaseModel):
  name: str
  id: int
  ...

responses = runner.run(prompt_collection, schema=User)
```

Further examples:

- [Use structured schema](./examples/structured_output.ipynb)