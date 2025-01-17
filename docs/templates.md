
## Use another Template for the Inference

You can use another template for the inference by providing the template name in the `create_prompts()` method.

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

{{user_prompt}}

Important Context 1:
{{key1}}

Important Context 2:
{{key2}}
```

Second, create a PromptCollection with the custom template:

```python
contexts = [Context.from_prompt_vars({"key1": "value1"}), Context.from_prompt_vars({"key2": "value2"})] * 5
prompt_collection = PromptCollection.create_prompts(
    ...
    contexts=contexts,
    ...
    template_name="llama3_custom.j2",
)
```

Now, the context information will be added to the prompt using the custom template And a prompt will look like this:

```bash
User prompt 1

Important Context 1:
value1

Important Context 2:
value2
```
