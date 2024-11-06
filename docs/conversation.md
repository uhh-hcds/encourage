# How to use ChatInferenceRunner

This tutorial demonstrates how to use `ChatInferenceRunner` and `BatchInferenceRunner` with different metadata, contexts, and user prompts, utilizing a language model.
`ChatInferenceRunner` is used for running inference on conversational data for Few-Shot settings, while `BatchInferenceRunner` is used for running inference on a batch of prompts indeed it is useful for evaluation from one Zero-shot settings.

Initialize the LLM model and sampling parameters:

```python
from vllm import LLM, SamplingParams
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

llm = LLM(model=model_name, gpu_memory_utilization=0.95)
sampling_params = SamplingParams(temperature=0.5, max_tokens=1000)
```
## ChatInferenceRunner

```python
from g4k.shared.llm import ChatInferenceRunner
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
