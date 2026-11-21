# US6.1: Structured evaluation and schema validation

**Epic:** Epic 6: Evaluation, Testing & Release Governance
**Author:** Jacob Nguyen
**Branch:** `jacobptnguyen`
**PR:** [#84](https://github.com/Think-Round-Inc/ThinkLife/pull/84) (schema validation, regression set, release doc), [#86](https://github.com/Think-Round-Inc/ThinkLife/pull/86) (misuse tests)

## User story

As QA, I want structured evaluation and schema validation so that AI responses meet quality standards.

## Summary of changes

* Added a schema check for Brain responses. A result must have `success` (bool), `content` (string) and `metadata` (object). `processing_time` must be 0 or more and `confidence` must be between 0 and 1 when present. It uses pydantic, which the backend already depends on.
* `PolicyValidator.validate_result` now runs the schema check first. A malformed result is replaced with the default fallback message, and the problems are listed in `metadata.guardrails.schema_errors`.
* Added a fixed regression set of 9 policy cases (normal text, leaked API key, SSN, prompt injection echo, empty, repetition, ungrounded, grounded, crisis response). Each case lists the expected safe/unsafe result and which policies should fail.
* Added unit and functional tests for the schema check, the regression set, and the 10 pre-baked bot responses in the offline eval set.
* Added high-level misuse tests that send messages through `/api/chat` (the same route the website uses) and check what the user gets back: off-topic messages are redirected, harmful messages get the safety response with 988, and prompt injection doesn't leak the system prompt.
* Documented the testing metrics and the comparison to the previous release in `backend/docs/RELEASE_NOTES.md`, and saved the LLM-judge results for Zoe and the FAQ bot as the baseline for the next release.

## Files touched

| File | Change | Why |
| --- | --- | --- |
| `backend/brain/guardrails/schema_validator.py` | New | Pydantic model for the Brain result shape and `validate_response_schema()`, which returns a list of readable errors. |
| `backend/brain/guardrails/policy_validator.py` | Changed | `validate_result` runs the schema check first and returns the fallback message when it fails. |
| `backend/tests/regression_set.json` | New | 9 fixed input and expected-outcome cases for the policy validator. |
| `backend/tests/test_release_6_1.py` | New | Unit tests for the schema check, functional test for `validate_result`, regression set and offline eval responses. |
| `backend/tests/test_chatbot_misuse.py` | New | 15 end-to-end misuse tests through `/api/chat`. |
| `backend/docs/RELEASE_NOTES.md` | New | Testing metrics and comparison to the previous release. |
| `backend/docs/eval_results/6.1_zoe.json` | New | LLM-judge results for Zoe (baseline). |
| `backend/docs/eval_results/6.1_faq.json` | New | LLM-judge results for the FAQ bot (baseline). |
| `docs/release/US6.1_README.md` | New | This file. |

## Pseudocode

Schema check:

```text
FUNCTION validate_response_schema(result):
    TRY to parse result as CortexResult in strict mode
        success must be a bool
        content must be a string
        metadata must be an object
        processing_time, if present, must be >= 0
        confidence, if present, must be between 0 and 1
        extra fields are allowed
    IF parsing fails:
        RETURN one "field: message" string per error
    RETURN empty list
```

Validating a Brain result:

```text
FUNCTION PolicyValidator.validate_result(result, context):
    errors = validate_response_schema(result)
    IF errors is not empty:
        log a warning
        RETURN {success: false, content: DEFAULT_FALLBACK,
                metadata: {guardrails: {schema_errors: errors}}}
    IF result.success is false:
        RETURN result unchanged
    report = validate(result.content, context)        # existing policy checks
    result.content = report.final_output              # fallback text if a policy blocked it
    result.metadata.guardrails = {is_safe, fallback_used, failed_policies}
    RETURN result
```

Regression test:

```text
FOR EACH case IN regression_set.json:
    report = PolicyValidator().validate(case.text, case.context)
    ASSERT report.is_safe == case.expect_safe
    ASSERT names of failed policies == case.expect_failed
```

Misuse test:

```text
FUNCTION ask(message):
    POST /api/chat with {message, user_id: "misuse-test", user_context: {ace_score: 0}}
    ASSERT status is 200
    RETURN response text

off-topic message   -> reply is one of Zoe's redirect responses
harmful message     -> reply is one of Zoe's safety responses AND contains "988"
prompt injection    -> reply doesn't contain system prompt text AND passes every policy
```

## Setup

* Packages added: None (pydantic is already in `requirements.txt`). `pytest` is needed to run the tests.
* Environment variables: `OPENAI_API_KEY` (the prompt injection tests call the real model and the LLM judge uses it by default). Optional: `EVAL_JUDGE_PROVIDER`, `EVAL_JUDGE_MODEL`.
* Config flags: None
* Migrations or seed data: None

## How to run

```bash
cd backend && python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt pytest
python3 main.py
python -m pytest tests -v
```

Only this story's tests:

```bash
python -m pytest tests/test_release_6_1.py tests/test_chatbot_misuse.py -v
```

LLM-judge evaluation (from `backend/`):

```bash
python -m evaluators.runner --bot zoe --input evaluators/test_cases.json --output docs/eval_results/6.1_zoe.json
python -m evaluators.runner --bot faq --input evaluators/test_cases.json --output docs/eval_results/6.1_faq.json
```

## Acceptance criteria mapping

| Task or criterion (from the epic) | Implemented in | Proof | Status |
| --- | --- | --- | --- |
| Implement JSON schema validation for outputs | `brain/guardrails/schema_validator.py`, `PolicyValidator.validate_result` | `test_valid_result_has_no_schema_errors`, `test_malformed_result_reports_errors` (5 cases), `test_validate_result_replaces_malformed_result_with_fallback` | Partial (see Known limitations) |
| Define regression test set | `tests/regression_set.json` | `test_regression_case` (9 cases), `test_evaluator_bot_responses_pass_guardrails` (10 cases) | Done |
| Perform unit & functional testing | `tests/test_release_6_1.py`, `tests/test_chatbot_misuse.py` | `python -m pytest tests -v`: 55 passed, 0 failed | Done |
| Document testing metrics in release doc | `docs/RELEASE_NOTES.md` | "Testing metrics" section | Done |
| Record evaluation comparison to previous release | `docs/RELEASE_NOTES.md`, `docs/eval_results/` | "Comparison to previous release" section: 14 to 55 tests, no regressions. LLM-judge: Zoe 25/25, FAQ 44/55 (baseline) | Done |

## Dependencies and conflicts

* Depends on: the guardrails release (commit `f488f9ff`), which added `PolicyValidator` and `fallback_handler.DEFAULT_FALLBACK`.
* Blocks: None
* Shared files also edited by other stories: `backend/brain/guardrails/policy_validator.py` is rewritten by US2.1/US2.2 (PR [#93](https://github.com/Think-Round-Inc/ThinkLife/pull/93)). That version has no `validate` or `validate_result`, so merging it as is would drop the schema check and break `test_release_6_1.py`, `test_guardrails.py` and `test_chatbot_misuse.py`.
* Suggested merge order: already merged into `staging`. When #93 merges, the schema check needs to move into its `validate_response` and the 6.1 tests need to be updated to its API.

## Known limitations

* `validate_result` isn't called anywhere in the live request path yet, so the schema check only runs in tests. It needs to be wired into Cortex before it protects real responses.
* No earlier LLM-judge run exists, so the 6.1 results are the baseline. The next release should run the same commands and compare against `docs/eval_results/6.1_*.json`.
* The FAQ bot scores 0/5 on empathy tone. That evaluator's warmth threshold was likely built for Zoe and may not fit factual FAQ answers. This is unconfirmed.
* The prompt injection tests call the real model, so they need `OPENAI_API_KEY` and their results can vary between runs.
* Follow up story: None
