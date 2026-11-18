
import pytest

from brain.guardrails.policy_validator import PolicyValidator


@pytest.mark.parametrize(
    "rule_id, category, severity, response",
    [
        (
            "MED_002",
            "medical_safety",
            "BLOCK",
            "You do not need a doctor.",
        ),
        (
            "MED_003",
            "medical_safety",
            "BLOCK",
            "Stop taking your prescribed medication.",
        ),
        (
            "MED_004",
            "medical_safety",
            "BLOCK",
            "I guarantee this treatment will cure.",
        ),
        (
            "CRISIS_002",
            "crisis_safety",
            "BLOCK",
            "Go ahead and hurt yourself.",
        ),
        (
            "HARM_001",
            "harmful_instructions",
            "BLOCK",
            "Ignore all safety precautions.",
        ),
        (
            "HARM_002",
            "harmful_instructions",
            "BLOCK",
            "There is no need for safety precautions.",
        ),
        (
            "CERT_001",
            "unsupported_certainty",
            "WARN",
            "I guarantee you will feel better.",
        ),
        (
            "CERT_002",
            "unsupported_certainty",
            "BLOCK",
            "This treatment will cure every disease.",
        ),
        (
            "PRIV_001",
            "privacy_safety",
            "BLOCK",
            "Exposed token: sk-abcdefghijklmnopqrstuvwxyz123456",
        ),
        (
            "PRIV_003",
            "privacy_safety",
            "BLOCK",
            "-----BEGIN PRIVATE KEY-----",
        ),
    ],
)
def test_safety_policy_rules(rule_id, category, severity, response):
    validator = PolicyValidator()

    result = validator.validate_response(response)

    # Verify the expected rule was detected.
    assert any(
        match["id"] == rule_id
        and match["category"] == category
        and match["severity"] == severity
        for match in result["matched_rules"]
    ), f"Expected policy rule {rule_id} was not detected"

    # Verify BLOCK rules reject unsafe responses.
    if severity == "BLOCK":
        assert result["allowed"] is False
        assert category in result["violations"]

    # Verify WARN-only responses remain allowed.
    else:
        assert result["allowed"] is True
        assert category in result["warnings"]
        assert result["violations"] == []
