
"""US 2.2 Task 2: Safe fallback messages for blocked output."""

FALLBACK_MESSAGES = {
    "medical_safety": (
        "I can't provide a definite medical diagnosis or recommend "
        "unsafe changes to treatment. A qualified healthcare "
        "professional can help you make informed decisions."
    ),
    "crisis_safety": (
        "I'm concerned about your safety, and you deserve support. "
        "If you're in immediate danger, contact emergency services. "
        "In the U.S., you can call or text 988 for crisis support."
    ),
    "harmful_instructions": (
        "I can't provide instructions that could put someone at risk. "
        "I can help you explore a safer approach."
    ),
    "professional_boundaries": (
        "I'm an AI assistant, not a licensed healthcare professional. "
        "I can offer general information and support, but not "
        "professional diagnosis or treatment."
    ),
    "unsupported_certainty": (
        "I can't guarantee that outcome. I can help you understand "
        "the available information and where uncertainty remains."
    ),
    "privacy_safety": (
        "I can't share a response that may expose sensitive information. "
        "Please avoid sharing passwords, private keys, or other "
        "confidential details."
    ),
}

DEFAULT_FALLBACK = (
    "I'm unable to provide that response safely right now. "
    "Please try rephrasing your message."
)

FALLBACK_PRIORITY = (
    "crisis_safety",
    "medical_safety",
    "privacy_safety",
    "harmful_instructions",
    "professional_boundaries",
    "unsupported_certainty",
)


def get_fallback_message(policy_result: dict) -> str:
    """Select a safe fallback for a blocked policy result."""
    if not isinstance(policy_result, dict):
        return DEFAULT_FALLBACK

    if policy_result.get("allowed") is not False:
        return DEFAULT_FALLBACK

    violations = policy_result.get("violations")

    if not isinstance(violations, list):
        return DEFAULT_FALLBACK

    for category in FALLBACK_PRIORITY:
        if category in violations:
            return FALLBACK_MESSAGES[category]

    return DEFAULT_FALLBACK
