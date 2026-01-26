import unittest

from encourage.prompts.conversation import Conversation, Role


class TestConversation(unittest.TestCase):
    def test_init_with_empty_user_prompt(self):
        """Test that Conversation doesn't add empty user message."""
        conversation = Conversation(sys_prompt="System message")
        
        # Should only have the system message, no user message
        self.assertEqual(len(conversation.dialog), 1)
        self.assertEqual(conversation.dialog[0]["role"], Role.SYSTEM.value)
        self.assertEqual(conversation.dialog[0]["content"], "System message")
        
        # Verify no user messages exist
        user_messages = conversation.get_messages_by_role(Role.USER)
        self.assertEqual(len(user_messages), 0)

    def test_init_with_non_empty_user_prompt(self):
        """Test that Conversation adds user message when provided."""
        conversation = Conversation(sys_prompt="System message", user_prompt="User question")
        
        # Should have both system and user messages
        self.assertEqual(len(conversation.dialog), 2)
        self.assertEqual(conversation.dialog[0]["role"], Role.SYSTEM.value)
        self.assertEqual(conversation.dialog[0]["content"], "System message")
        self.assertEqual(conversation.dialog[1]["role"], Role.USER.value)
        self.assertEqual(conversation.dialog[1]["content"], "User question")
        
        # Verify user message exists
        user_messages = conversation.get_messages_by_role(Role.USER)
        self.assertEqual(len(user_messages), 1)
        self.assertEqual(user_messages[0]["content"], "User question")

    def test_init_with_empty_strings(self):
        """Test that Conversation with all empty strings has no messages."""
        conversation = Conversation(sys_prompt="", user_prompt="")
        
        # Should have no messages
        self.assertEqual(len(conversation.dialog), 0)
        
        # Verify no user or system messages exist
        user_messages = conversation.get_messages_by_role(Role.USER)
        system_messages = conversation.get_messages_by_role(Role.SYSTEM)
        self.assertEqual(len(user_messages), 0)
        self.assertEqual(len(system_messages), 0)

    def test_add_message_after_empty_init(self):
        """Test adding messages after initializing with empty user prompt."""
        conversation = Conversation(sys_prompt="System message")
        
        # Add a user message
        conversation.add_message(Role.USER.value, "First user message")
        
        # Should now have system + user message
        self.assertEqual(len(conversation.dialog), 2)
        user_messages = conversation.get_messages_by_role(Role.USER)
        self.assertEqual(len(user_messages), 1)
        self.assertEqual(user_messages[0]["content"], "First user message")


if __name__ == "__main__":
    unittest.main()
