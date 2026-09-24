"""
Epic 6 / Story 6.1: schema validation + regression set.
Run with: pytest tests/test_release_6_1.py -v
"""

import json
from pathlib import Path

import pytest

from brain.guardrails.policy_validator import PolicyValidator
from brain.guardrails.schema_validator import validate_response_schema

TESTS_DIR = Path(__file__).parent
REGRESSION = json.loads((TESTS_DIR / "regression_set.json").read_text())
EVALUATOR_CASES = json.loads((TESTS_DIR.parent / "evaluators" / "test_cases.json").read_text())


# --- Unit: schema validator -------------------------------------------------

def test_valid_result_has_no_schema_errors():
    assert validate_response_schema(
        {"success": True, "content": "hi", "metadata": {}, "processing_time": 0.4, "confidence": 0.9}
    ) == []


@pytest.mark.parametrize("bad", [
    {"content": "hi", "metadata": {}},                                    # missing success
    {"success": True, "content": None, "metadata": {}},                   # content not a string
    {"success": True, "content": "hi", "metadata": []},                   # metadata not an object
    {"success": True, "content": "hi", "metadata": {}, "confidence": 2},  # confidence out of range
    "just a string",                                                      # not an object
])
def test_malformed_result_reports_errors(bad):
    assert validate_response_schema(bad)


# --- Functional: schema check inside validate_result ------------------------

def test_validate_result_replaces_malformed_result_with_fallback():
    out = PolicyValidator().validate_result({"content": 123})
    assert out["success"] is False
    assert isinstance(out["content"], str)
    assert out["metadata"]["guardrails"]["schema_errors"]


# --- Regression set ---------------------------------------------------------

@pytest.mark.parametrize("case", REGRESSION, ids=lambda c: c["id"])
def test_regression_case(case):
    report = PolicyValidator().validate(case["text"], case.get("context"))
    assert report.is_safe == case["expect_safe"]
    assert sorted(r.policy_name for r in report.failed_policies()) == sorted(case["expect_failed"])


@pytest.mark.parametrize("case", [c for c in EVALUATOR_CASES if c.get("bot_response")], ids=lambda c: c["id"])
def test_evaluator_bot_responses_pass_guardrails(case):
    """Every pre-baked bot response in the offline eval set must clear the guardrails."""
    assert PolicyValidator().validate(case["bot_response"]).is_safe
