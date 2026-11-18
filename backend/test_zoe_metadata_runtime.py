
"""
User Story 2.2 - Task 1
Runtime metadata filtering verification.
"""

import math
from agents.zoe.zoe_service import ZoeService


def run_test():
    service = ZoeService()

    # Simulate internal metadata containing sensitive information.
    raw_metadata = {
        "agent": "zoe",
        "confidence_score": 0.92,
        "duration_seconds": 1.5,
        "estimated_cost": 0.002,
        "validation_attempts": 2,
        "output_moderation": "passed",
        "error": "Internal database connection failed",
        "execution_plan": {
            "internal_steps": ["private processing details"]
        },
        "api_key": "SECRET_TEST_API_KEY",
        "user_email": "test@example.com",
        "errors": ["Internal system traceback"],
    }

    safe_metadata = service._safe_public_metadata(
        raw_metadata
    )

    # Check approved fields remain available.
    assert safe_metadata["agent"] == "zoe"
    assert safe_metadata["confidence_score"] == 0.92
    assert safe_metadata["duration_seconds"] == 1.5
    assert safe_metadata["estimated_cost"] == 0.002
    assert safe_metadata["validation_attempts"] == 2
    assert safe_metadata["output_moderation"] == "passed"

    print("PASS: Approved metadata preserved.")

    # Check sensitive fields are excluded.
    blocked_fields = [
        "error",
        "execution_plan",
        "api_key",
        "user_email",
        "errors",
    ]

    for field in blocked_fields:
        assert field not in safe_metadata, (
            f"Sensitive field leaked: {field}"
        )

    print("PASS: Sensitive metadata excluded.")

    # Check original internal metadata is unchanged.
    assert raw_metadata["api_key"] == "SECRET_TEST_API_KEY"
    assert "execution_plan" in raw_metadata

    print("PASS: Internal metadata remains unchanged.")

    # Check invalid numeric values are rejected.
    invalid_metadata = {
        "confidence_score": float("nan"),
        "duration_seconds": float("inf"),
        "estimated_cost": True,
        "validation_attempts": "two",
        "output_moderation": "unknown",
    }

    filtered = service._safe_public_metadata(
        invalid_metadata
    )

    assert filtered == {"agent": "zoe"}

    print("PASS: Invalid metadata values rejected.")

    # Check unexpected input does not expose data.
    assert service._safe_public_metadata(None) == {
        "agent": "zoe"
    }

    print("PASS: Unexpected metadata input handled.")

    print()
    print("=" * 55)
    print("USER STORY 2.2 - TASK 1")
    print("ALL 5 METADATA FILTERING CHECKS PASSED")
    print("=" * 55)


if __name__ == "__main__":
    run_test()
