
import asyncio
import logging

from agents.zoe.zoe_service import ZoeService


async def main():
    service = ZoeService()

    # Simulate a service that has not initialized its dependencies.
    service.initialized = True
    service.zoe_core = None
    service.zoe_agent = None

    # Suppress the expected exception traceback during this test.
    logging.disable(logging.CRITICAL)

    try:
        result = await service.process_message(
            message="Test exception handling",
            user_id="test_user",
            session_id="test_session"
        )
    finally:
        logging.disable(logging.NOTSET)

    expected_response = (
        "I'm experiencing technical difficulties. "
        "Please try again later."
    )

    assert result["success"] is False
    assert result["response"] == expected_response
    assert result["metadata"]["output_moderation"] == "blocked"

    print("PASS: Exception handled without crashing")
    print("PASS: Safe technical-error response returned")
    print("PASS: Output moderation marked as blocked")
    print("\nAll 3 exception-path checks passed.")


if __name__ == "__main__":
    asyncio.run(main())
