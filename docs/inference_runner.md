# How to use ChatInferenceRunner and BatchInferenceRunner

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
from g4k.shared.llm.inference_runner import BatchInferenceRunner

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

## ChatInferenceRunner

```python
from g4k.shared.llm.inference_runner import ChatInferenceRunner
from g4k.shared.prompts.conversation_handler import ConversationHandler

# For initializing the ChatInferenceRunner, you need to provide the LLM model and sampling parameters.
runner = ChatInferenceRunner(llm, sampling_params)
```

Then you need to create a `ConversationHandler` object with the metadata, context, and user prompts:
A `ConversationHandler` object contains one conversation and for each prompt you want to ask in the conversation, it contains the metadata, context, and user prompt.
**Important to note**: A `ConversationHandler` object is a multi-turn conversation where each turn is a prompt that has been previously defined.


```python

Each conversation contains:

- a dialog (containing the different messages types(system, user, assistant))
- context (a list of dicts, that can be added to the prompt)
- meta data keys (a list to tag the conversation)

To run inference on the conversation, you can use the `run()` method:

```python
conversation_handler = ConversationHandler(
    chat_inference_runner=runner,
    system_prompt=sys_prompts,
    user_prompts=user_prompts[:10], 
    contexts=contexts[:10],
    meta_data=meta_data[:10],
    template_name="generate_python_code_conv.j2",
)

response = conversation_handler.run()
```

The output will look like this:

<details>
  <summary>Example Preview:</summary>

```bash
--------------------------------------------------
🧑‍💻 User Prompt:
User prompt 1
Provided context:
key1: value1
📚 Added Context: {'key1': 'value1'} (See Template for details.)

💬 Response:
Arrr, ye be wantin' to know about key1, eh? Alright then, matey, I be tellin' ye that key1 be paired with the value of... value1. Savvy?

🤖 System Prompt:
You are an helpful AI that only speaks like a pirat
🗂️ Metadata: {'meta': 'data1'}
🆔 Request ID: 10
🆔 Prompt ID: aaa90500-6727-4624-8103-29baf233f746
🆔 Conversation ID: 0
⏳ Processing Time: 1.1021 seconds

--------------------------------------------------
🧑‍💻 User Prompt:
Did you get key1 as context?
📚 Added Context: {'key1': 'value1'} (See Template for details.)

💬 Response:
Aye, I did get the context, matey. Key1 be the treasure I be rememberin' now. Ye can trust ol' Blackbeak to keep track o' the booty.

🤖 System Prompt:
You are an helpful AI that only speaks like a pirat
🗂️ Metadata: {'meta': 'data1'}
🆔 Request ID: 11
🆔 Prompt ID: b566222a-1ea1-4a35-b877-e741d42e5a04
🆔 Conversation ID: 1
⏳ Processing Time: 0.9815 seconds

--------------------------------------------------
🧑‍💻 User Prompt:
Did you get key1 as context?
📚 Added Context: {'key1': 'value1'} (See Template for details.)

💬 Response:
Aye, I did get key1 as context, matey. I be seein' it right here, plain as the anchor on the bow o' me ship. Key1 be the treasure I be holdin' onto, savvy?

🤖 System Prompt:
You are an helpful AI that only speaks like a pirat
🗂️ Metadata: {'meta': 'data1'}
🆔 Request ID: 12
🆔 Prompt ID: 2eba72d1-9c53-4ba8-94d7-b0584babbc56
🆔 Conversation ID: 2
⏳ Processing Time: 1.204 seconds
```

</details>

## Use another Template for the Inference

You can use another template for the inference by providing the template name in the `create_prompts()
` method and has to stored in the `/src/g4k/shared/prompts/templates` directory.

```python
prompt_collection = PromptCollection.create_prompts(
    ...
    template_name=<YOUR_TEMPLATE_NAME>
)
```

#### Example using a Llama3 custom template

Define the template and add custom keys to the template.
It will add the context to the prompt using the value for that.

First, create a custom template `llama3_custom.j2` in the `/src/g4k/shared/prompts/templates` directory:

```jinja2
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{{system_prompt}}
<|start_header_id|>user<|end_header_id|>

{{user_prompt}}<|eot_id|>
<|start_header_id|>user<|end_header_id|>

Important Context 1:
{{key1}}

Important Context 2:
{{key2}}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
```

Second, create a PromptCollection with the custom template:

```python
contexts = [{"key1": "value1"}, {"key2": "value2"}] * 5
prompt_collection = PromptCollection.create_prompts(
    ...
    contexts=contexts,
    ...
    template_name="llama3_custom.j2",
)
```

Now, the context information will be added to the prompt using the custom template And a prompt will look like this:

```bash
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an helpful AI that only speaks like a pirat
<|start_header_id|>user<|end_header_id|>

User prompt 1<|eot_id|>
<|start_header_id|>user<|end_header_id|>

Important Context 1:
value1

Important Context 2:
value2

<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
```
