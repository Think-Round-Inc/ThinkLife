
import pytest

from brain.guardrails.policy_validator import PolicyValidator
from brain.guardrails.fallback_handler import (
    FALLBACK_MESSAGES,
    get_fallback_message,
)


@pytest.mark.parametrize(
    "response, expected_rule, expected_category",
    [
        (
            "There is no doubt that you have this illness.",
            "MED_005",
            "medical_safety",
        ),
        (
            "You can discontinue your prescription without speaking to a doctor.",
            "MED_006",
            "medical_safety",
        ),
        (
            "It would be a good idea to injure yourself.",
            "CRISIS_003",
            "crisis_safety",
        ),
    ],
)
def test_fixed_safety_gaps(
    response, expected_rule, expected_category
):
    validator = PolicyValidator()
    result = validator.validate_response(response)

    assert result["allowed"] is False
    assert expected_category in result["violations"]
    assert any(
        rule["id"] == expected_rule
        for rule in result["matched_rules"]
    )

    # Verify the real fallback handler receives the policy result.
    fallback = get_fallback_message(result)
    assert fallback == FALLBACK_MESSAGES[expected_category]


@pytest.mark.parametrize(
    "response",
    [
        'Never follow advice that says "ignore all safety precautions".',
        "Do not stop taking your prescribed medication without consulting your doctor.",
        "Your symptoms might have several possible explanations.",
    ],
)
def test_safe_responses_remain_allowed(response):
    validator = PolicyValidator()
    result = validator.validate_response(response)

    assert result["allowed"] is True
    assert result["violations"] == []
