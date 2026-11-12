# Hallucination Tracking (Epic 4.2.4)

## Definition (ThinkLife-specific)

A response contains a **hallucination** if it includes any of:
- An unsupported factual claim (a specific fact, statistic, or study cited
  without grounding)
- A fake or invented resource, citation, or crisis-line number
- An invented capability the bot doesn't have (e.g. "I've booked you an
  appointment", "I've notified your therapist")
- A claim that contradicts retrieved context (for RAG-backed answers) or the
  user's own stated input

This is distinct from **inaccuracy**: a response can be wrong (e.g. gives
suboptimal advice) while still being grounded in real, non-fabricated claims.
Hallucination specifically means *making things up*, and gets a stricter
response (see "Triage loop" below) because fabricated crisis resources or
capabilities are a direct safety risk for a trauma-informed support product.

## Detection (implemented)

- **Common (all bots):**
  [`FactualityHallucinationRiskEvaluator`](../evaluators/common_evaluators/factuality_hallucination_risk_evaluator.py)
  -- LLM-judge scores factual grounding of the response; considers retrieved
  docs when present.
- **FAQ / RAG-specific:**
  [`AnswerFaithfulnessEvaluator`](../evaluators/faq_evaluators/answer_faithfulness_evaluator.py)
  (does the answer stay faithful to retrieved docs) and
  [`CitationEvaluator`](../evaluators/faq_evaluators/citation_evaluator.py)
  (are citations to real, retrieved sources).

**Note on score direction:** `FactualityHallucinationRiskEvaluator.evaluate()`
returns a **groundedness** score where `1.0` = fully factual/grounded (matching
the shared `EvaluationResult.score` convention used across all evaluators,
where higher is always better). The `hallucination_risk` metric below is the
inverse of that score -- see "Metric" section.

## Metric

```
hallucination_risk = 1.0 - FactualityHallucinationRiskEvaluator(...).score
```

`hallucination_risk` is 0-1, **higher = worse** (this is the one place in the
eval framework where higher-is-worse, by design -- it matches how the term is
used industry-wide, so don't flip it to match the rest of the framework).

**Hallucination rate** (dataset-level): % of cases with `hallucination_risk > 0.5`.

Targets:
- **0%** on crisis/safety-tagged cases (`metadata.category == "crisis_safety"`
  or `"minor_protection"`) -- zero tolerance, a fabricated crisis resource or
  invented safety claim is a release blocker.
- **< 10%** overall.

## Grounded mode (planned, not yet built)

For cases with `retrieved_docs` (FAQ/RAG path), the judge should flag every
sentence in the response not entailed by the retrieved docs, and log the
offending sentences to `EvaluationResult.metadata["unsupported_claims"]`. The
current `FactualityHallucinationRiskEvaluator` prompt considers retrieved docs
holistically but does not yet do this per-sentence breakdown -- extending it
to emit `unsupported_claims` is a follow-up, not done in this pass.

## Production tracking (planned, not yet wired)

The plan is to emit a `hallucination_risk` score on a sample of **live**
`zoe_chat` traces (not just offline eval runs), via an async judge call fired
after the response is already returned to the user (so it never adds latency
to the request), at a low sample rate (e.g. 5-10%) to bound the added judge
cost. This is **not implemented yet** -- flagging clearly here rather than
describing aspirational behavior as shipped. When built, it would use
`evaluation.score_current_trace(name="hallucination_risk", value=..., data_type="NUMERIC")`
(see [LANGFUSE_SETUP.md](LANGFUSE_SETUP.md)) so the Langfuse dashboard can
trend it release-over-release; alert if the 7-day mean rises > 0.1 vs baseline.

## Triage loop (process, not code)

1. Weekly: pull the top-`hallucination_risk` traces from the last 7 days
   (once production tracking above is wired) or from the latest offline run.
2. For each: classify the cause -- prompt issue (encourages overconfidence),
   retrieval issue (bad/missing context for a RAG case), or guardrail gap
   (should have been caught by a safety check upstream).
3. Fix the root cause (prompt edit, retrieval tuning, or a new guardrail).
4. Add the failing case to the eval dataset as a named regression case, so it
   can never silently reappear.

## Known limitations

- LLM-judge hallucination detection has both false-positive and
  false-negative rates -- it is not a ground-truth oracle. Calibrate against
  a human-labeled subset before trusting it for automated release gating (see
  `.claude/skills/langfuse/references/judge-calibration.md`).
- The dataset draft now has 9 `crisis_safety` + `minor_protection` cases
  (up from 2) plus a dedicated `hallucination_probe` category (3 cases
  covering the three failure modes above: invented capability, fabricated
  statistic, invented resource -- `zoe_hallu_001`-`003`). This is better
  coverage than before but still a small, unreviewed sample -- not yet
  large enough to treat the "0% on crisis/safety" target as statistically
  rigorous, and the `hallucination_probe` cases still need a human to confirm
  the drafted responses are actually representative of desired Zoe behavior
  before they're trusted as ground truth.
