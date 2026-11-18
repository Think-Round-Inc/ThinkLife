
import asyncio
from types import SimpleNamespace

from brain.cortex.workflow_engine import WorkflowEngine


async def main():
    engine = WorkflowEngine()

    # Current user message after Task 4 redaction
    request = SimpleNamespace(
        message="My email is [REDACTED_EMAIL]",
        user_context=None
    )

    specs = SimpleNamespace(
        agent_metadata={
            "messages_context": {
                "system_prompt": "You are a helpful assistant.",

                # Previously redacted conversation history
                "conversation_history": [
                    {
                        "role": "user",
                        "content": "My phone is [REDACTED_PHONE]"
                    },
                    {
                        "role": "assistant",
                        "content": "I understand."
                    }
                ],

                # Original message retained in metadata
                # to verify that the workflow does not use it
                "current_message": "My email is john@example.com"
            }
        }
    )

    state = {
        "request": request,
        "specs": specs,
        "execution_steps": [],
        "errors": [],
        "data_source_results": [],
        "tool_results": [],
        "validation_attempts": 0,
        "validation_feedback": None
    }

    # Build AI messages without calling the AI provider
    result = await engine._build_messages_node(state)

    # Verify workflow execution
    assert not result["errors"], result["errors"]

    # Collect user messages sent to the AI
    user_messages = [
        msg["content"]
        for msg in result["messages"]
        if msg["role"] == "user"
    ]

    # Verify historical and current user messages
    assert user_messages == [
        "My phone is [REDACTED_PHONE]",
        "My email is [REDACTED_EMAIL]"
    ], f"Unexpected user messages: {user_messages}"

    # Check the complete AI message payload
    all_message_content = " ".join(
        str(msg.get("content", ""))
        for msg in result["messages"]
    )

    # Original sensitive information must not appear
    assert "john@example.com" not in all_message_content
    assert "806-555-1234" not in all_message_content

    print("PASS: Conversation history remains redacted.")
    print("PASS: Current message remains redacted.")
    print("PASS: No original test PII appears in the AI messages.")


if __name__ == "__main__":
    asyncio.run(main())

