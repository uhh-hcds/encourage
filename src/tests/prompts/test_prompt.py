import json
import unittest
from unittest import mock

from encourage.prompts.context import Context
from encourage.prompts.meta_data import MetaData

with mock.patch.dict("sys.modules", {"vllm": mock.MagicMock()}):
    from encourage.prompts.prompt_collection import PromptCollection


class TestPromptCollection(unittest.TestCase):
    def setUp(self):
        self.sys_prompts = ["System prompt 1", "System prompt 2"]
        self.user_prompts = ["User prompt 1", "User prompt 2"]
        self.contexts = [
            Context.from_prompt_vars({"info": "context1"}),
            Context.from_prompt_vars({"info": "context2"}),
        ]
        self.meta_datas = [MetaData({"meta": "data1"}), MetaData({"meta": "data2"})]
        self.template_name = "llama3_conv.j2"
        self.reformated_prompt = [
            "Question: \nUser prompt 1\n\nAnswer: ",
            "Question: \nUser prompt 2\n\nAnswer: ",
        ]

        self.prompt_collection = PromptCollection.create_prompts(
            sys_prompts=self.sys_prompts,
            user_prompts=self.user_prompts,
            contexts=self.contexts,
            meta_datas=self.meta_datas,
            template_name=self.template_name,
        )

    def test_create_prompts_success(self):
        """Test successful creation of PromptCollection with matching prompts."""
        self.assertEqual(len(self.prompt_collection), 2)
        for i, prompt in enumerate(self.prompt_collection):
            print(prompt)
            self.assertEqual(prompt.conversation.sys_prompt, self.sys_prompts[i])
            self.assertEqual(
                prompt.conversation.get_last_message_by_user(),
                self.reformated_prompt[i],
            )
            self.assertEqual(prompt.context, self.contexts[i])
            self.assertEqual(prompt.meta_data, self.meta_datas[i])

    def test_create_prompts_mismatched_lengths(self):
        """Test that ValueError is raised when prompt lists have mismatched lengths."""
        with self.assertRaises(ValueError) as context:
            PromptCollection.create_prompts(sys_prompts=["Sys1"], user_prompts=["User1", "User2"])
        self.assertIn(
            "The number of system prompts must match the number of user prompts.",
            str(context.exception),
        )

    def test_create_prompts_without_contexts_and_meta_datas(self):
        """Test creation of PromptCollection without providing contexts and meta_datas."""
        prompt_collection = PromptCollection.create_prompts(
            sys_prompts=self.sys_prompts,
            user_prompts=self.user_prompts,
            template_name=self.template_name,
        )
        self.assertEqual(len(prompt_collection), 2)
        for i, prompt in enumerate(prompt_collection.prompts):
            self.assertEqual(prompt.context, Context())
            self.assertEqual(prompt.meta_data, MetaData())
            self.assertEqual(
                prompt.conversation.get_last_message_by_user(),
                self.reformated_prompt[i],
            )

    def test_create_prompts_partial_contexts_and_meta_datas(self):
        """Test creation of PromptCollection with partial contexts and meta_datas."""
        # Test with a partial list of contexts, expecting a ValueError for contexts length mismatch
        with self.assertRaises(
            ValueError, msg="The number of contexts must match the number of prompts."
        ):
            PromptCollection.create_prompts(
                sys_prompts=self.sys_prompts,
                user_prompts=self.user_prompts,
                contexts=[
                    Context.from_prompt_vars({"info": "context1"})
                ],  # Partial context (1 item instead of 2)
                meta_datas=[
                    MetaData({"meta": "meta1"}),
                    MetaData({"meta": "meta2"}),
                ],  # Full meta_data (2 items)
            )

        # Test with a partial list of meta_datas, expecting a ValueError for meta_datas mismatch
        with self.assertRaises(
            ValueError, msg="The number of meta_datas must match the number of prompts."
        ):
            PromptCollection.create_prompts(
                sys_prompts=self.sys_prompts,
                user_prompts=self.user_prompts,
                contexts=[
                    Context.from_prompt_vars({"info": "context1"}),
                    Context.from_prompt_vars({"info": "context2"}),
                ],  # Partial context (1 item instead of 2)
                meta_datas=[MetaData({"meta": "meta1"})],  # Partial meta_data (1 item instead of 2)
            )

    def test_from_json(self):
        """Test deserialization of PromptCollection from JSON."""
        json_data = self.prompt_collection.to_json()
        new_collection = PromptCollection.from_json(json_data)
        self.assertEqual(len(new_collection), len(self.prompt_collection))
        for original, deserialized in zip(self.prompt_collection.prompts, new_collection.prompts):
            self.assertEqual(original.id, deserialized.id)
            self.assertEqual(original.conversation, deserialized.conversation)
            self.assertEqual(original.context, deserialized.context)
            self.assertEqual(original.meta_data, deserialized.meta_data)

    def test_to_json(self):
        """Test serialization of PromptCollection to JSON."""
        json_data = self.prompt_collection.to_json()
        expected_json = json.dumps(
            {
                "prompts": [
                    {
                        "id": json.loads(json_data)["prompts"][0]["id"],
                        "conversation": '{"dialog": [{"role": "system", "content": "System prompt 1"}, {"role": "user", "content": "Question: \\nUser prompt 1\\n\\nAnswer: "}]}',  # noqa: E501
                        "context": {"documents": [], "prompt_vars": {"info": "context1"}},
                        "meta_data": {"meta": "data1"},
                    },
                    {
                        "id": json.loads(json_data)["prompts"][1]["id"],
                        "conversation": '{"dialog": [{"role": "system", "content": "System prompt 2"}, {"role": "user", "content": "Question: \\nUser prompt 2\\n\\nAnswer: "}]}',  # noqa: E501
                        "context": {"documents": [], "prompt_vars": {"info": "context2"}},
                        "meta_data": {"meta": "data2"},
                    },
                ]
            }
        )
        self.assertEqual(json.loads(json_data), json.loads(expected_json))

    def test_len(self):
        """Test the __len__ method of PromptCollection."""
        self.assertEqual(len(self.prompt_collection), 2)

    def test_iter(self):
        """Test the __iter__ method of PromptCollection."""
        prompts = list(iter(self.prompt_collection))
        self.assertEqual(prompts, self.prompt_collection.prompts)


if __name__ == "__main__":
    unittest.main()
