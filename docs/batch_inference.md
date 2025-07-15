# How to use BatchInferenceRunner

This tutorial demonstrates how to use `BatchInferenceRunner` with different metadata, contexts, and user prompts, utilizing a language model.
`ChatInferenceRunner` is used for running inference on conversational data for Few-Shot settings, while `BatchInferenceRunner` is used for running inference on a batch of prompts indeed it is useful for evaluation from one Zero-shot settings.

To use the `BatchInferenceRunner`, you need to first start the vllm OpenAI server.
You can find more information about that [here](./vllm_server.md).

Initialize the LLM model and sampling parameters:

```python
from vllm import SamplingParams
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

sampling_params = SamplingParams(temperature=0.5, max_tokens=1000)
```

## BatchInferenceRunner

```python
from encourage.llm import BatchInferenceRunner

# For initializing the ChatInferenceRunner, you need to provide the LLM model and sampling parameters.
runner = BatchInferenceRunner(sampling_params, model_name=model_name)
```



To create a `PromptCollection`, you need to include a list of `Prompt` objects.
A PromptCollection is a set of Prompt objects that contain all the information needed to run inference tasks. Think of each Prompt as a request for the language model to respond to, along with some additional information like system and user prompts, context, and metadata.

Each `Prompt` contains:

- `id`: unique identifier (auto-generated)
- `conversation`: stores dialog turns between user and system
- `context`: structured background info (`Context`)
- `meta_data`: tags or attributes for filtering (`MetaData`)

> ✅ `Prompt` objects support serialization via `.to_json()` and deserialization with `Prompt.from_json()`.

---

You typically don’t need to create `Prompt` objects manually.  
Use `PromptCollection.create_prompts()` to generate them easily from lists of inputs:



```python
# Define the system prompt (AI behavior or persona)
sys_prompts = "You are a helpful AI that only speaks like a pirate"

# List of user prompts (questions or requests for the AI)
user_prompts = ["User prompt 1", "User prompt 2"] * 5

# Context information for each prompt (additional data or background info)
contexts = [Context.from_prompt_vars({"key1": "value1"}), Context.from_prompt_vars({"key2": "value2"})] * 5

# # Metadata associated with each prompt (e.g., priority, tags)
meta_datas = [MetaData({"meta": "data1"}), MetaData({"meta": "data2"})] * 5

# Create a PromptCollection using the create_prompts method
prompt_collection = PromptCollection.create_prompts(
    sys_prompts=sys_prompts,  # System prompt or list of system prompts
    user_prompts=user_prompts,  # List of user prompts
    contexts=contexts,  # List of context dictionaries (optional)
    meta_datas=meta_datas,  # List of metadata dictionaries (optional)
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

sampling_params = SamplingParams(temperature=0.5, max_tokens=1000)

responses = runner.run(prompt_collection, User)
```

Further examples:

- [Use structured schema](./examples/structured_output.ipynb)