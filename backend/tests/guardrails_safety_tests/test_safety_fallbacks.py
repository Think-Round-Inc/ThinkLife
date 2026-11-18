
import pytest

from brain.guardrails.fallback_handler import (
    FALLBACK_MESSAGES,
    DEFAULT_FALLBACK,
    FALLBACK_PRIORITY,
    get_fallback_message,
)


@pytest.mark.parametrize("category", FALLBACK_PRIORITY)
def test_category_fallback_messages(category):
    result = {
        "allowed": False,
        "violations": [category],
    }

    actual_message = get_fallback_message(result)

    assert actual_message == FALLBACK_MESSAGES[category]
    assert isinstance(actual_message, str)
    assert actual_message.strip()


def test_multiple_violations_follow_priority():
    result = {
        "allowed": False,
        "violations": [
            "professional_boundaries",
            "medical_safety",
            "crisis_safety",
        ],
    }

    assert get_fallback_message(result) == FALLBACK_MESSAGES[
        "crisis_safety"
    ]


def test_priority_not_based_on_violation_order():
    result = {
        "allowed": False,
        "violations": [
            "harmful_instructions",
            "privacy_safety",
        ],
    }

    assert get_fallback_message(result) == FALLBACK_MESSAGES[
        "privacy_safety"
    ]


@pytest.mark.parametrize(
    "policy_result",
    [
        None,
        {},
        {"allowed": True, "violations": ["medical_safety"]},
        {"allowed": False},
        {"allowed": False, "violations": None},
        {"allowed": False, "violations": "medical_safety"},
        {"allowed": False, "violations": []},
        {"allowed": False, "violations": ["unknown_category"]},
    ],
)
def test_default_fallback_for_invalid_or_unknown_results(policy_result):
    assert get_fallback_message(policy_result) == DEFAULT_FALLBACK
