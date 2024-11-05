import json
import unittest
from unittest import mock
from unittest.mock import patch

with mock.patch.dict("sys.modules", {"vllm": mock.MagicMock()}):
    from encourage.prompts.prompt_collection import PromptCollection
    from encourage.prompts.prompt_reformatter import PromptReformatter


class TestPromptCollection(unittest.TestCase):
    def setUp(self):
        self.sys_prompts = ["System prompt 1", "System prompt 2"]
        self.user_prompts = ["User prompt 1", "User prompt 2"]
        self.contexts = [[{"info": "context1"}], [{"info": "context2"}]]
        self.meta_datas = [[{"meta": "meta1"}], [{"meta": "meta2"}]]
        self.model_name = "ModelX"
        self.template_name = "TemplateA"

        # Mock the PromptReformatter.reformat method to return a predictable string
        self.reformat_patch = patch.object(
            PromptReformatter, "reformat_prompt", return_value="Reformatted prompt"
        )
        self.mock_reformat = self.reformat_patch.start()

        self.prompt_collection = PromptCollection.create_prompts(
            sys_prompts=self.sys_prompts,
            user_prompts=self.user_prompts,
            contexts=self.contexts,
            meta_datas=self.meta_datas,
            model_name=self.model_name,
            template_name=self.template_name,
        )

    def tearDown(self):
        self.reformat_patch.stop()

    def test_create_prompts_success(self):
        """Test successful creation of PromptCollection with matching prompts."""
        self.assertEqual(len(self.prompt_collection), 2)
        for i, prompt in enumerate(self.prompt_collection.prompts):
            self.assertEqual(prompt.sys_prompt, self.sys_prompts[i])
            self.assertEqual(prompt.user_prompt, self.user_prompts[i])
            self.assertEqual(prompt.context, self.contexts[i])
            self.assertEqual(prompt.meta_data, self.meta_datas[i])
            self.assertEqual(prompt.reformatted, "Reformatted prompt")

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
            sys_prompts=["Sys1", "Sys2"], user_prompts=["User1", "User2"]
        )
        self.assertEqual(len(prompt_collection), 2)
        for prompt in prompt_collection.prompts:
            self.assertEqual(prompt.context, [])
            self.assertEqual(prompt.meta_data, [])
            self.assertEqual(prompt.reformatted, "Reformatted prompt")

    def test_create_prompts_partial_contexts_and_meta_datas(self):
        """Test creation of PromptCollection with partial contexts and meta_datas."""
        # Test with a partial list of contexts, expecting a ValueError for contexts length mismatch
        with self.assertRaises(
            ValueError, msg="The number of contexts must match the number of prompts."
        ):
            PromptCollection.create_prompts(
                sys_prompts=self.sys_prompts,
                user_prompts=self.user_prompts,
                contexts=[{"info": "context1"}],  # Partial context (1 item instead of 2)
                meta_datas=[{"meta": "meta1"}, {"meta": "meta2"}],  # Full meta_data (2 items)
            )

        # Test with a partial list of meta_datas, expecting a ValueError for meta_datas mismatch
        with self.assertRaises(
            ValueError, msg="The number of meta_datas must match the number of prompts."
        ):
            PromptCollection.create_prompts(
                sys_prompts=self.sys_prompts,
                user_prompts=self.user_prompts,
                contexts=[{"info": "context1"}, {"info": "context2"}],  # Full context (2 items)
                meta_datas=[{"meta": "meta1"}],  # Partial meta_data (1 item instead of 2)
            )

    def test_from_json(self):
        """Test deserialization of PromptCollection from JSON."""
        json_data = self.prompt_collection.to_json()
        new_collection = PromptCollection.from_json(json_data)
        self.assertEqual(len(new_collection), len(self.prompt_collection))
        for original, deserialized in zip(self.prompt_collection.prompts, new_collection.prompts):
            self.assertEqual(original.sys_prompt, deserialized.sys_prompt)
            self.assertEqual(original.user_prompt, deserialized.user_prompt)
            self.assertEqual(original.context, deserialized.context)
            self.assertEqual(original.meta_data, deserialized.meta_data)
            self.assertEqual(original.reformatted, deserialized.reformatted)

    def test_to_json(self):
        """Test serialization of PromptCollection to JSON."""
        json_data = self.prompt_collection.to_json()
        expected_json = json.dumps(
            {
                "prompts": [
                    {
                        "id": json.loads(json_data)["prompts"][0]["id"],
                        "sys_prompt": "System prompt 1",
                        "user_prompt": "User prompt 1",
                        "conversation_id": 0,
                        "context": [{"info": "context1"}],
                        "meta_data": [{"meta": "meta1"}],
                        "reformatted": "Reformatted prompt",
                    },
                    {
                        "id": json.loads(json_data)["prompts"][1]["id"],
                        "sys_prompt": "System prompt 2",
                        "user_prompt": "User prompt 2",
                        "conversation_id": 0,
                        "context": [{"info": "context2"}],
                        "meta_data": [{"meta": "meta2"}],
                        "reformatted": "Reformatted prompt",
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
