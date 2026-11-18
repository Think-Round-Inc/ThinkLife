
"""
US 2.2 Task 4 - Safety redirect hallucination tracking test.

Uses simulated dependencies. No real LLM call is made.
"""

import asyncio
from unittest.mock import Mock, patch

from agents.zoe.zoe_service import ZoeService


async def main():
    service = ZoeService()
    service.initialized = True

    # Simulate the existing security operations.
    security_manager = Mock()
    security_manager.sanitize_input.side_effect = lambda text: text
    security_manager.redact_sensitive_data.side_effect = (
        lambda text: {"content": text}
    )

    service.zoe_agent = Mock()
    service.zoe_agent.cortex.security_manager = security_manager

    # Simulate a safety redirect.
    service.zoe_core = Mock()
    service.zoe_core.create_brain_request.return_value = {
        "session_id": "test_session",
        "redirect_info": {
            "should_redirect": True,
            "is_harmful": True,
            "response": "Please seek appropriate support."
        }
    }

    # Capture observations without writing to the real log.
    service.hallucination_tracker.observe = Mock(
        return_value={"persisted": True}
    )

    # Run the actual ZoeService response path.
    with patch.object(
        service,
        "_moderate_final_response",
        return_value=("Please seek appropriate support.", False)
    ):
        result = await service.process_message(
            message="Test safety redirect",
            user_id="test_user",
            session_id="test_session"
        )

    # Verify the chatbot response is preserved.
    assert result["success"] is True
    assert result["response"] == "Please seek appropriate support."

    # Verify one observation was recorded.
    service.hallucination_tracker.observe.assert_called_once_with(
        status="unevaluated",
        agent_name="zoe",
        groundedness_score=None,
        reason="not_applicable"
    )

    # Verify the redirect was saved to conversation history.
    assistant_calls = [
        call
        for call in service.zoe_core.conversation_manager.add_message.call_args_list
        if call.kwargs.get("role") == "assistant"
    ]

    assert len(assistant_calls) == 1
    assert assistant_calls[0].kwargs["content"] == result["response"]

    print("PASS: Safety redirect response preserved")
    print("PASS: Exactly one hallucination observation recorded")
    print("PASS: Observation status and reason correct")
    print("PASS: Redirect response stored in conversation history")
    print("\nAll 4 safety redirect integration checks passed.")


if __name__ == "__main__":
    asyncio.run(main())
