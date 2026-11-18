from typing import Optional
import logging

from presidio_analyzer import AnalyzerEngine


logger = logging.getLogger(__name__)


class PIIDetector:
    """
    Local PII detection layer using Microsoft Presidio.

    This component detects broader personally identifiable
    information without sending text to an external service.
    """

    def __init__(self):
        self.analyzer = AnalyzerEngine()

    def redact(self, text: Optional[str]) -> str:
        """
        Detect and redact security-relevant PII from text while
        safely handling overlapping Presidio detections.
        """

        if not text:
            return text or ""

        try:
            # Only redact entity types that are strongly relevant
            # to sensitive-data protection.
            #
            # We intentionally exclude generic DATE_TIME,
            # LOCATION, and URL entities to reduce false positives
            # in normal chatbot conversations.
            sensitive_entities = [
                "PERSON",
                "EMAIL_ADDRESS",
                "PHONE_NUMBER",
                "CREDIT_CARD",
                "US_SSN",
                "US_PASSPORT",
                "US_DRIVER_LICENSE",
                "US_BANK_NUMBER",
                "US_ITIN",
                "MEDICAL_LICENSE",
                "IBAN_CODE",
                "IP_ADDRESS",
                "MAC_ADDRESS",
                "CRYPTO",
                "UK_NHS",
            ]

            results = self.analyzer.analyze(
                text=text,
                language="en",
                entities=sensitive_entities,
            )

            # Prefer more security-specific entity types when
            # multiple Presidio recognizers detect the same span.
            entity_priority = {
                "US_BANK_NUMBER": 100,
                "US_PASSPORT": 90,
                "US_DRIVER_LICENSE": 85,
                "US_SSN": 85,
                "US_ITIN": 85,
                "CREDIT_CARD": 80,
                "IBAN_CODE": 80,
                "MEDICAL_LICENSE": 80,
                "UK_NHS": 80,
                "CRYPTO": 75,
                "EMAIL_ADDRESS": 70,
                "PHONE_NUMBER": 70,
                "IP_ADDRESS": 70,
                "MAC_ADDRESS": 70,
                "PERSON": 60,
            }

            sorted_results = sorted(
                results,
                key=lambda item: (
                    -entity_priority.get(item.entity_type, 40),
                    -item.score,
                    -(item.end - item.start),
                ),
            )

            selected_results = []

            for result in sorted_results:
                overlaps = any(
                    result.start < selected.end
                    and result.end > selected.start
                    for selected in selected_results
                )

                if not overlaps:
                    selected_results.append(result)

            redacted_text = text

            # Replace detected entities from right to left so
            # the original character indexes stay valid.
            for result in sorted(
                selected_results,
                key=lambda item: item.start,
                reverse=True,
            ):
                replacement = f"[REDACTED_{result.entity_type}]"

                redacted_text = (
                    redacted_text[:result.start]
                    + replacement
                    + redacted_text[result.end:]
                )

            return redacted_text

        except Exception as e:
            logger.warning(
                "Local PII detection failed; returning original text: %s",
                e,
            )
            return text