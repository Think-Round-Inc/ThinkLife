
"""
US 2.2 Task 1 - Service-level metadata security test.

Uses simulated agent responses. No real LLM call is made.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

from agents.zoe.zoe_service import ZoeService


SECRET = "SECRET_INTERNAL_TEST_VALUE"


class FakeSecurityManager:
    def sanitize_input(self, content):
        return content

    def redact_sensitive_data(self, content):
        return {"content": content}

    def filter_content(self, content):
        return {"safe": True, "content": content}


class FakePolicyValidator:
    def validate_response(self, content):
        return {"allowed": True}


class FakeConversationManager:
    def __init__(self):
        self.messages = []

    def add_message(self, **kwargs):
        self.messages.append(kwargs)


class FakeZoeCore:
    def __init__(self):
        self.conversation_manager = FakeConversationManager()

    def create_brain_request(self, **kwargs):
        return {
            "session_id": kwargs["session_id"],
            "redirect_info": None,
        }

    def get_fallback_response(self):
        return "Please try again later."

    def get_error_response(self):
        return "A technical error occurred."


class FakeZoeAgent:
    def __init__(self, response):
        self.response = response
        self.cortex = SimpleNamespace(
            security_manager=FakeSecurityManager()
        )

    async def process_request(self, request):
        return self.response


def create_agent_response(success, confidence):
    return SimpleNamespace(
        success=success,
        content="This is a safe test response.",
        session_id="test_session",
        processing_time=0.25,
        metadata={
            "agent": "zoe",
            "confidence_score": confidence,
            "duration_seconds": 0.25,
            "error": SECRET,
            "api_key": SECRET,
            "execution_plan": {"private": SECRET},
            "errors": [SECRET],
        },
    )


async def check_scenario(name, success, confidence):
    service = ZoeService()
    service.initialized = True
    service.zoe_core = FakeZoeCore()
    service.zoe_agent = FakeZoeAgent(
        create_agent_response(success, confidence)
    )
    service.policy_validator = FakePolicyValidator()

    # Replace request construction only for this test.
    with patch(
        "brain.specs.BrainRequest",
        side_effect=lambda **kwargs: SimpleNamespace(**kwargs),
    ), patch(
        "agents.zoe.helpers.create_user_context",
        side_effect=lambda **kwargs: kwargs,
    ):
        result = await service.process_message(
            message="Hello",
            user_id="test_user",
            session_id="test_session",
        )

    assert result["success"] is success, (
        f"{name}: Unexpected success status"
    )

    assert result["metadata"]["agent"] == "zoe"

    assert (
        result["metadata"]["confidence_score"] == confidence
    )

    # Verify the complete public service response.
    assert SECRET not in str(result), (
        f"{name}: Sensitive metadata leaked in response"
    )

    for field in ("api_key", "execution_plan", "errors"):
        assert field not in result["metadata"], (
            f"{name}: Internal field exposed: {field}"
        )

    # Verify all stored conversation messages.
    history = service.zoe_core.conversation_manager.messages

    assert history, f"{name}: No history was stored"

    assert SECRET not in str(history), (
        f"{name}: Sensitive metadata leaked in history"
    )

    if not success:
        assert result["error"] == "Processing failed"
        assert result["metadata"]["fallback"] is True

    if success and confidence < 0.75:
        assert result["metadata"]["low_confidence"] is True

    print(f"PASS: {name}")


async def main():
    await check_scenario(
        "High-confidence response metadata",
        True,
        0.95,
    )

    await check_scenario(
        "Low-confidence response metadata",
        True,
        0.50,
    )

    await check_scenario(
        "Failure response and history metadata",
        False,
        0.20,
    )

    print()
    print("=" * 55)
    print("USER STORY 2.2 - TASK 1")
    print("ALL 3 SERVICE-LEVEL METADATA CHECKS PASSED")
    print("=" * 55)


if __name__ == "__main__":
    asyncio.run(main())
