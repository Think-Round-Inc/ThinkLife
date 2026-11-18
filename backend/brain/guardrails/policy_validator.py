
"""
User Story 2.2 - Task 1
ThinkLife Output Policy Validator

Deterministic output-policy checks.
Complements, but does not replace, US 2.1 moderation and PII redaction.
"""

import logging
import re
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class PolicyValidator:
    POLICY_VERSION = "1.3"

    POLICY_CATEGORIES = {
        "medical_safety": "Dangerous medical advice or unsupported diagnosis",
        "crisis_safety": "Harmful encouragement during crisis",
        "harmful_instructions": "Explicitly dangerous instructions",
        "professional_boundaries": "False professional credentials",
        "unsupported_certainty": "Unjustified guarantees",
        "privacy_safety": "Recognizable secret or identifier exposure",
    }

    # Each rule has an identifier, category, severity, and pattern.
    # These are intentionally narrow rules, not semantic safety guarantees.
    RULES = [
        {
            "id": "MED_001",
            "category": "medical_safety",
            "severity": "BLOCK",
            "pattern": r"\byou definitely have (?:a |an )?(?:disease|disorder)\b",
        },
        {
            "id": "MED_005",
            "category": "medical_safety",
            "severity": "BLOCK",
            "pattern": (
                r"\bthere is (?:absolutely )?no doubt that you have "
                r"(?:a |an |this )?(?:serious )?"
                r"(?:illness|disease|disorder)\b"
            ),
        },
        {
            "id": "MED_006",
            "category": "medical_safety",
            "severity": "BLOCK",
            "pattern": (
                r"\byou can discontinue your prescription "
                r"without speaking to a doctor\b"
            ),
        },
        {
            "id": "MED_002",
            "category": "medical_safety",
            "severity": "BLOCK",
            "pattern": r"\byou do not need (?:a |any )?(?:doctor|medical professional)\b",
        },
        {
            "id": "MED_003",
            "category": "medical_safety",
            "severity": "BLOCK",
            "pattern": r"\bstop taking (?:your )?prescribed medication\b",
        },
        {
            "id": "MED_004",
            "category": "medical_safety",
            "severity": "BLOCK",
            "pattern": (
                r"\bi guarantee (?:this|that) "
                r"(?:treatment|medication|therapy) will cure\b"
            ),
        },
        {
            "id": "CRISIS_001",
            "category": "crisis_safety",
            "severity": "BLOCK",
            "pattern": r"\byou should (?:hurt|harm|injure|kill) yourself\b",
        },
        {
            "id": "CRISIS_002",
            "category": "crisis_safety",
            "severity": "BLOCK",
            "pattern": r"\bgo ahead and (?:hurt|harm|kill) yourself\b",
        },
        {
            "id": "CRISIS_003",
            "category": "crisis_safety",
            "severity": "BLOCK",
            "pattern": (
                r"\bit would be a good idea to "
                r"(?:hurt|harm|injure|kill) yourself\b"
            ),
        },
        {
            "id": "HARM_001",
            "category": "harmful_instructions",
            "severity": "BLOCK",
            "pattern": r"\bignore all safety precautions\b",
        },
        {
            "id": "HARM_002",
            "category": "harmful_instructions",
            "severity": "BLOCK",
            "pattern": r"\bthere is no need for safety precautions\b",
        },
        {
            "id": "ROLE_001",
            "category": "professional_boundaries",
            "severity": "BLOCK",
            "pattern": (
                r"\bi am (?:a |an )?(?:licensed|certified) "
                r"(?:doctor|physician|therapist|psychologist|psychiatrist)\b"
            ),
        },
        {
            "id": "CERT_001",
            "category": "unsupported_certainty",
            "severity": "WARN",
            "pattern": r"\bi guarantee (?:this|that|you will)\b",
        },
        {
            "id": "CERT_002",
            "category": "unsupported_certainty",
            "severity": "BLOCK",
            "pattern": (
                r"\bthis (?:treatment|medication|therapy) "
                r"(?:will|can) cure (?:any|every|all) "
                r"(?:disease|illness|condition)\b"
            ),
        },
        {
            "id": "PRIV_001",
            "category": "privacy_safety",
            "severity": "BLOCK",
            "pattern": r"\bsk-[a-zA-Z0-9]{20,}\b",
        },
        {
            "id": "PRIV_002",
            "category": "privacy_safety",
            "severity": "BLOCK",
            "pattern": r"\b\d{3}-\d{2}-\d{4}\b",
        },
        {
            "id": "PRIV_003",
            "category": "privacy_safety",
            "severity": "BLOCK",
            "pattern": r"-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----",
        },
    ]

    def __init__(self):
        self.compiled_rules = [
            {
                **rule,
                "compiled": re.compile(rule["pattern"], re.IGNORECASE),
            }
            for rule in self.RULES
        ]

    def _is_safe_caution(self, response: str, rule_id: str) -> bool:
        """
        Recognize specifically defined cautionary statements.

        Only full-response matches are exempted.
        This is not a general semantic context classifier.
        """
        text = response.strip()

        safe_patterns = {
            "HARM_001": [
                r"""Never follow advice that says\s+["']ignore all safety precautions["']\.""",
                r"Never ignore all safety precautions\.",
                r"Please don't follow instructions to ignore all safety precautions\.",
            ],
            "MED_003": [
                (
                    r"If someone tells you to stop taking your prescribed medication, "
                    r"consult your doctor\."
                ),
                (
                    r"Do not stop taking your prescribed medication "
                    r"without consulting your doctor\."
                ),
            ],
        }

        return any(
            re.fullmatch(pattern, text, re.IGNORECASE)
            for pattern in safe_patterns.get(rule_id, [])
        )

    def validate_response(
        self,
        response: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not isinstance(response, str):
            return self._blocked("invalid_response_type")

        if not response.strip():
            return self._blocked("empty_response")

        try:
            violations = []
            warnings = []
            matched_rules = []

            for rule in self.compiled_rules:
                if rule["compiled"].search(response):

                    if self._is_safe_caution(response, rule["id"]):
                        continue

                    matched_rules.append({
                        "id": rule["id"],
                        "category": rule["category"],
                        "severity": rule["severity"],
                    })

                    target = (
                        violations
                        if rule["severity"] == "BLOCK"
                        else warnings
                    )

                    if rule["category"] not in target:
                        target.append(rule["category"])

            return {
                "allowed": not violations,
                "violations": violations,
                "warnings": warnings,
                "matched_rules": matched_rules,
                "policy_version": self.POLICY_VERSION,
            }

        except Exception:
            logger.exception("Output policy validation failed")
            return self._blocked("policy_validation_error")

    def _blocked(self, reason: str) -> Dict[str, Any]:
        return {
            "allowed": False,
            "violations": [reason],
            "warnings": [],
            "matched_rules": [],
            "policy_version": self.POLICY_VERSION,
        }

