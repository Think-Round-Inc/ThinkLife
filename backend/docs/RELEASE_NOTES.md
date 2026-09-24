# Release Notes: Epic 6.1 (Evaluation & Schema Validation)

## What changed
- `brain/guardrails/schema_validator.py`: pydantic schema (no new dependency) for CortexFlow results (`success`, `content`, `metadata`, optional `processing_time`, `confidence` 0-1).
- `PolicyValidator.validate_result` now schema-checks first. A malformed result is replaced with the default fallback message and `metadata.guardrails.schema_errors` lists the problems.
- `tests/regression_set.json`: 9 fixed input/expected-outcome cases for the policy validator.
- `tests/test_release_6_1.py`: unit + functional tests for the above.

## Testing metrics
Run: `cd backend && python -m pytest tests -v`

| Metric | Value |
|---|---|
| Total tests | 40 |
| Passed / failed | 40 / 0 |
| New unit tests (schema validator) | 6 |
| New functional tests (validate_result, regression set, offline eval responses) | 20 |
| Existing guardrail tests | 14 |
| Regression cases | 9 policy cases + 10 pre-baked bot responses from `evaluators/test_cases.json` |

## Comparison to previous release (commit `f488f9ff`, guardrails release)

| Metric | Previous | This release | Change |
|---|---|---|---|
| Tests | 14 | 40 | +26 |
| Pass rate | 14/14 | 40/40 | no regressions |
| Output schema validation | none | yes | new |
| Regression set | none | yes (19 cases) | new |

The 14 previous tests all still pass unchanged.

## LLM-judge evaluation (offline mode, `python -m evaluators.runner`)
Raw results: `docs/eval_results/6.1_zoe.json`, `docs/eval_results/6.1_faq.json`.

| Bot | Cases | Checks passed | Pass rate |
|---|---|---|---|
| Zoe | 5 | 25 / 25 | 100% |
| FAQ | 5 | 44 / 55 | 80% |

FAQ evaluators below 100%: empathytone 0/5 (avg 0.26), citation 3/5 (0.56), faqexactness 3/5 (0.74), instructionfollowingrelevance 4/5 (0.82), policyconsistency 4/5 (0.92). The empathytone failures are likely the factual FAQ style scoring low on a warmth threshold built for Zoe; that is unconfirmed.

**Comparison to previous release:** no earlier LLM-judge run exists, so these results are the baseline. The next release should run the same command and compare against these two files.
