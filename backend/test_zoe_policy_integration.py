
import asyncio
from types import SimpleNamespace
from unittest.mock import Mock, AsyncMock, patch

from agents.zoe.zoe_service import ZoeService
from brain.guardrails.fallback_handler import DEFAULT_FALLBACK


# Use the current fallback message from US 2.2 Task 2.
SAFE_FALLBACK = DEFAULT_FALLBACK


async def run_low_confidence_test(name, original_response, expected_original):
    service = ZoeService()
    service.initialized = True

    # Mock existing US 2.1 security operations.
    security_manager = Mock()

    security_manager.sanitize_input.side_effect = lambda text: text

    security_manager.redact_sensitive_data.side_effect = (
        lambda text: {
            "content": text.replace(
                "john@example.com",
                "[REDACTED_EMAIL]"
            )
        }
    )

    security_manager.filter_content.side_effect = (
        lambda text: {
            "safe": True,
            "content": text
        }
    )

    # Mock the policy decision for controlled integration testing.
    def validate_policy(text):
        unsafe = "You definitely have a disease." in text
        return {"allowed": not unsafe}

    service.policy_validator = Mock()
    service.policy_validator.validate_response.side_effect = validate_policy

    # Mock ZoeCore and conversation history.
    conversation_manager = Mock()

    service.zoe_core = Mock()
    service.zoe_core.conversation_manager = conversation_manager

    service.zoe_core.create_brain_request.return_value = {
        "session_id": "test_session_001",
        "redirect_info": None
    }

    # Mock ZoeAgent and its Cortex security manager.
    service.zoe_agent = Mock()
    service.zoe_agent.cortex.security_manager = security_manager

    agent_response = SimpleNamespace(
        success=True,
        content=original_response,
        session_id="test_session_001",
        metadata={"confidence_score": 0.50},
        processing_time=0.01
    )

    service.zoe_agent.process_request = AsyncMock(
        return_value=agent_response
    )

    # Execute the actual ZoeService.process_message method.
    with patch(
        "brain.specs.BrainRequest",
        return_value=Mock()
    ), patch(
        "agents.zoe.helpers.create_user_context",
        return_value=Mock()
    ):
        result = await service.process_message(
            message="I need some support",
            user_id="test_user",
            session_id="test_session_001"
        )

    assert result["success"] is True, (
        f"{name}: Processing failed: {result}"
    )

    assert result["metadata"]["low_confidence"] is True, (
        f"{name}: Low-confidence path was not used"
    )

    # Retrieve the assistant message stored in conversation history.
    assistant_calls = [
        call
        for call in conversation_manager.add_message.call_args_list
        if call.kwargs.get("role") == "assistant"
    ]

    assert len(assistant_calls) == 1, (
        f"{name}: Expected one assistant history message"
    )

    stored_message = assistant_calls[0].kwargs
    stored_metadata = stored_message["metadata"]

    # Verify the existing metadata key is preserved.
    assert "original_response" in stored_metadata, (
        f"{name}: original_response metadata key is missing"
    )

    # Verify the original response was moderated before storage.
    assert stored_metadata["original_response"] == expected_original, (
        f"{name}: Unexpected stored original response: "
        f"{stored_metadata['original_response']}"
    )

    # Verify the user-facing low-confidence fallback is unchanged.
    expected_fallback = service._get_low_confidence_response(
        "I need some support",
        0.50
    )

    assert result["response"] == expected_fallback, (
        f"{name}: User-facing fallback changed"
    )

    assert stored_message["content"] == result["response"], (
        f"{name}: Conversation history response differs"
    )

    # Verify the existing security operations were executed.
    assert security_manager.sanitize_input.called, (
        f"{name}: Sanitization was not executed"
    )

    assert security_manager.redact_sensitive_data.called, (
        f"{name}: Redaction was not executed"
    )

    assert security_manager.filter_content.called, (
        f"{name}: Content filtering was not executed"
    )

    assert service.policy_validator.validate_response.called, (
        f"{name}: Policy validation was not executed"
    )

    print(f"PASS: {name}")


async def main():
    await run_low_confidence_test(
        name="Safe original response",
        original_response="Here are some healthy ways to manage stress.",
        expected_original="Here are some healthy ways to manage stress."
    )

    await run_low_confidence_test(
        name="PII redaction in original response",
        original_response="Contact john@example.com for support.",
        expected_original="Contact [REDACTED_EMAIL] for support."
    )

    await run_low_confidence_test(
        name="Unsafe original response blocked",
        original_response="You definitely have a disease.",
        expected_original=SAFE_FALLBACK
    )

    print("\n--- INTEGRATION TEST SUMMARY ---")
    print("Passed: 3")
    print("Failed: 0")
    print("Low-confidence metadata integration checks passed.")


if __name__ == "__main__":
    asyncio.run(main())

