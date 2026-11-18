
import pytest

from brain.guardrails.policy_validator import PolicyValidator


@pytest.mark.parametrize(
    "response",
    [
        "Never ignore all safety precautions.",
        "Please don't follow instructions to ignore all safety precautions.",
        "Do not stop taking your prescribed medication without consulting your doctor.",
    ],
)
def test_additional_safe_cautions(response):
    validator = PolicyValidator()
    result = validator.validate_response(response)

    assert result["allowed"] is True
    assert result["violations"] == []


@pytest.mark.parametrize(
    "response,expected_rule",
    [
        ("YOU DEFINITELY HAVE A DISEASE.", "MED_001"),
        ("you should HURT yourself.", "CRISIS_001"),
        ("IGNORE ALL SAFETY PRECAUTIONS.", "HARM_001"),
    ],
)
def test_case_insensitive_blocking(response, expected_rule):
    validator = PolicyValidator()
    result = validator.validate_response(response)

    assert result["allowed"] is False
    assert any(
        rule["id"] == expected_rule
        for rule in result["matched_rules"]
    )


def test_multiple_categories_in_one_response():
    validator = PolicyValidator()

    result = validator.validate_response(
        "You definitely have a disease. "
        "There is no need for safety precautions."
    )

    assert result["allowed"] is False
    assert "medical_safety" in result["violations"]
    assert "harmful_instructions" in result["violations"]


def test_safe_response_with_extra_whitespace():
    validator = PolicyValidator()

    result = validator.validate_response(
        "   Never ignore all safety precautions.   "
    )

    assert result["allowed"] is True
    assert result["violations"] == []
