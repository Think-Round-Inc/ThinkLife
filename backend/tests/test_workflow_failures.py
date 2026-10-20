"""
US 1.1 Task 4 - failure handling transitions. One test per row of the
failure table in docs/workflows/README.md: every failure ends in a retry,
a fallback provider, or an escalation - never in error text shown as an answer.

Run with: pytest tests/test_workflow_failures.py -v
"""

import time

import pytest

from conftest import run
from workflow_fakes import (
    FakeLLM, GOOD_ANSWER, JUDGE_FAIL, JUDGE_PASS, err, install_fakes, make_plan, make_request, make_specs, ok,
)
from brain.cortex.workflow_engine import WorkflowEngine
from brain.cortex.workflow_errors import (
    ProviderCallError, TransientProviderError, classify_provider_error, is_retryable,
)
from brain.guardrails.fallback_handler import DEFAULT_FALLBACK, FALLBACK_MESSAGES

FREE = "openrouter/free"


def execute(monkeypatch, llm, specs=None, **fake_kwargs):
    install_fakes(monkeypatch, llm, **fake_kwargs)
    engine = WorkflowEngine(checkpointer_kind="memory")
    started = time.monotonic()
    result = run(engine.execute_plan(make_plan(specs or make_specs()), make_request()))
    result["_elapsed"] = time.monotonic() - started
    return result


def with_fallback(monkeypatch):
    monkeypatch.setenv("WORKFLOW_FALLBACK_PROVIDERS", f"openrouter:{FREE}")


# ---------------------------------------------------------------------------
# Row 1 - transient errors are retried with backoff
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("error_text", [
    "Error code: 429 - Rate limit reached",
    "Error code: 503 - Service Unavailable",
    "Request timed out.",
    "Connection error.",
])
def test_transient_error_is_retried_then_succeeds(workflow_env, monkeypatch, error_text):
    llm = FakeLLM(scripts={"gpt-4o-mini": [err(error_text), ok(GOOD_ANSWER)]})
    result = execute(monkeypatch, llm)

    assert result["success"] is True
    assert result["status"] == "completed"
    assert result["content"] == GOOD_ANSWER
    assert len(llm.answer_calls()) == 2


def test_empty_completion_counts_as_transient(workflow_env, monkeypatch):
    llm = FakeLLM(scripts={"gpt-4o-mini": [ok("   "), ok(GOOD_ANSWER)]})
    result = execute(monkeypatch, llm)
    assert result["success"] is True
    assert len(llm.answer_calls()) == 2


def test_sdk_retries_are_disabled_so_retries_are_counted_once(workflow_env, monkeypatch):
    llm = FakeLLM()
    execute(monkeypatch, llm)
    assert all(cfg.get("max_retries") == 0 for cfg in llm.configs)


# ---------------------------------------------------------------------------
# Row 2 - provider still down after retries -> fallback provider
# ---------------------------------------------------------------------------

def test_provider_down_falls_back_to_next_model(workflow_env, monkeypatch):
    with_fallback(monkeypatch)
    llm = FakeLLM(scripts={"gpt-4o-mini": [err("Error code: 503 - overloaded")], FREE: [ok("Answer from the fallback model, complete.")]})
    result = execute(monkeypatch, llm)

    assert result["success"] is True
    assert result["content"].startswith("Answer from the fallback model")
    assert result["metadata"]["provider_used"] == f"openrouter:{FREE}"
    steps = result["metadata"]["execution_steps"]
    assert steps.index("call_provider_failed") < steps.index("fallback_provider") < steps.index("check_safety")
    primary = [c for c in llm.answer_calls() if c["model"] == "gpt-4o-mini"]
    assert len(primary) == 3  # profile.provider_retry_attempts
    # The judge used the provider that is actually working
    assert [c["model"] for c in llm.calls if c["judge"]] == [FREE]


def test_provider_that_cannot_initialise_goes_straight_to_fallback(workflow_env, monkeypatch):
    with_fallback(monkeypatch)
    llm = FakeLLM(init_fails={("openai", "gpt-4o-mini")}, scripts={FREE: [ok(GOOD_ANSWER)]})
    result = execute(monkeypatch, llm)

    assert result["success"] is True
    assert result["metadata"]["failures"][0]["kind"] == "provider_unavailable"
    assert not [c for c in llm.answer_calls() if c["model"] == "gpt-4o-mini"]


def test_after_fallback_takes_over_retries_stay_on_it(workflow_env, monkeypatch):
    with_fallback(monkeypatch)
    llm = FakeLLM(
        scripts={"gpt-4o-mini": [err("Error code: 503")], FREE: [ok("First fallback answer, rather weak."), ok(GOOD_ANSWER)]},
        judge=[ok(JUDGE_FAIL), ok(JUDGE_PASS)],
    )
    result = execute(monkeypatch, llm)

    assert result["success"] is True
    assert result["content"] == GOOD_ANSWER
    primary = [c for c in llm.answer_calls() if c["model"] == "gpt-4o-mini"]
    assert len(primary) == 3  # not retried again on the quality loop


# ---------------------------------------------------------------------------
# Row 3 - fallback fails too -> escalate
# ---------------------------------------------------------------------------

def test_fallback_failing_too_escalates(workflow_env, monkeypatch):
    with_fallback(monkeypatch)
    llm = FakeLLM(scripts={"gpt-4o-mini": [err("Error code: 503")], FREE: [err("Error code: 503")]})
    result = execute(monkeypatch, llm)

    assert result["status"] == "escalated"
    assert result["metadata"]["escalation"] == {"reason": "provider_failure", "flagged_for_review": True}
    assert result["success"] is False  # the agent shows its own fallback (Zoe's), as before
    assert result["content"] == DEFAULT_FALLBACK
    assert "Error" not in result["content"]


def test_no_fallback_configured_escalates_after_retries(workflow_env, monkeypatch):
    llm = FakeLLM(scripts={"gpt-4o-mini": [err("Error code: 502 - bad gateway")]})
    result = execute(monkeypatch, llm)
    assert result["status"] == "escalated"
    assert result["metadata"]["escalation"]["reason"] == "provider_failure"
    assert len(llm.answer_calls()) == 3


def test_non_transient_error_is_not_retried(workflow_env, monkeypatch):
    llm = FakeLLM(scripts={"gpt-4o-mini": [err("Error code: 401 - Incorrect API key provided")]})
    result = execute(monkeypatch, llm)
    assert result["status"] == "escalated"
    assert result["metadata"]["failures"][0]["kind"] == "provider_error"
    assert len(llm.answer_calls()) == 1


# ---------------------------------------------------------------------------
# Row 4 - policy block -> escalate, never retried
# ---------------------------------------------------------------------------

def test_policy_block_escalates_without_retry(workflow_env, monkeypatch):
    leaked = "Sure, the key you need is sk-abcdefghijklmnopqrstuvwxyz123456 for the service."
    llm = FakeLLM(scripts={"gpt-4o-mini": [ok(leaked)]})
    result = execute(monkeypatch, llm)

    assert result["status"] == "escalated"
    assert result["metadata"]["escalation"]["reason"] == "policy_block"
    assert result["content"] == FALLBACK_MESSAGES["banned_pattern"]
    assert "sk-" not in result["content"]
    assert result["success"] is True  # the policy message is the answer to show
    assert result["metadata"]["safety"]["failed_policies"] == ["banned_pattern"]
    assert len(llm.answer_calls()) == 1
    assert not any(c["judge"] for c in llm.calls)


def test_safety_checks_follow_the_agent_setting(workflow_env, monkeypatch):
    leaked = "Sure, the key you need is sk-abcdefghijklmnopqrstuvwxyz123456 for the service."
    llm = FakeLLM(scripts={"gpt-4o-mini": [ok(leaked)]})
    result = execute(monkeypatch, llm, specs=make_specs(safety=False))
    assert result["status"] == "completed"
    assert result["metadata"]["safety"]["checked"] is False


# ---------------------------------------------------------------------------
# Rows 5 and 6 - weak answers: retry with feedback, then escalate
# ---------------------------------------------------------------------------

def test_weak_answer_retried_then_accepted(workflow_env, monkeypatch):
    llm = FakeLLM(judge=[ok(JUDGE_FAIL), ok(JUDGE_PASS)])
    result = execute(monkeypatch, llm)
    assert result["success"] is True
    assert result["status"] == "completed"
    assert result["metadata"]["validation_attempts"] == 2


def test_weak_answers_exhaust_iterations_and_escalate(workflow_env, monkeypatch):
    llm = FakeLLM(judge=[ok(JUDGE_FAIL)])
    result = execute(monkeypatch, llm)
    assert result["status"] == "escalated"
    assert result["metadata"]["escalation"]["reason"] == "quality_exhausted"
    assert result["content"].startswith("I'm not confident")
    assert len(llm.answer_calls()) == 3


def test_judge_failure_no_longer_forces_the_fallback_message(workflow_env, monkeypatch):
    # Before US 1.1 a failed judge call scored 0.5, which retried 5 times and
    # then replaced a perfectly good answer with the fallback message.
    llm = FakeLLM(judge=[err("Error code: 429 - rate limit")])
    result = execute(monkeypatch, llm)
    assert result["success"] is True
    assert result["content"] == GOOD_ANSWER
    assert result["metadata"]["validation_mode"] == "judge_failed"
    assert len(llm.answer_calls()) == 1


# ---------------------------------------------------------------------------
# Row 7 - time budget
# ---------------------------------------------------------------------------

def test_no_retry_starts_after_the_time_budget(workflow_env, monkeypatch):
    llm = FakeLLM(judge=[ok(JUDGE_FAIL)], delay=0.1)
    result = execute(monkeypatch, llm, specs=make_specs(timeout_seconds=0.15))
    assert result["status"] == "escalated"
    assert result["metadata"]["escalation"]["reason"] == "timeout"
    assert len(llm.answer_calls()) == 1


def test_hung_provider_is_cut_off(workflow_env, monkeypatch):
    workflow_env.provider_timeout_seconds = 0.2
    llm = FakeLLM(delay=5.0)
    result = execute(monkeypatch, llm, specs=make_specs(timeout_seconds=0.3))
    assert result["status"] == "escalated"
    assert result["success"] is False
    assert result["_elapsed"] < 2.0


# ---------------------------------------------------------------------------
# Row 8 - data source down: retry, then continue without context
# ---------------------------------------------------------------------------

def test_data_source_transient_error_is_retried(workflow_env, monkeypatch):
    llm = FakeLLM()
    result = execute(monkeypatch, llm, specs=make_specs(data_sources=True),
                     vector_behaviour=[ConnectionError("chroma restarting"), [{"content": "Saturdays at 10."}]])
    assert result["status"] == "completed"
    prompt = " ".join(m["content"] for m in llm.answer_calls()[0]["messages"])
    assert "Saturdays at 10." in prompt


def test_data_source_down_continues_without_context(workflow_env, monkeypatch):
    llm = FakeLLM()
    result = execute(monkeypatch, llm, specs=make_specs(data_sources=True),
                     vector_behaviour=[ConnectionError("chroma down")])
    assert result["success"] is True
    assert result["status"] == "completed_with_errors"
    assert result["metadata"]["failures"][0]["node"] == "query_data_sources"
    prompt = " ".join(m["content"] for m in llm.answer_calls()[0]["messages"])
    assert "chroma down" not in prompt  # error text never reaches the model


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text, transient", [
    ("Error code: 429 - Rate limit reached for gpt-4o-mini", True),
    ("Error code: 429 - You exceeded your current quota (insufficient_quota)", False),
    ("Error code: 500 - The server had an error", True),
    ("Request timed out.", True),
    ("Error code: 401 - Incorrect API key provided", False),
    ("Error code: 400 - Invalid model", False),
])
def test_classify_provider_error(text, transient):
    error = classify_provider_error(text)
    assert isinstance(error, TransientProviderError if transient else ProviderCallError)
    assert is_retryable(error) is transient


def test_plain_runtime_errors_are_not_retried():
    assert is_retryable(RuntimeError("bug")) is False
    assert is_retryable(ConnectionError("reset")) is True


def test_many_iterations_do_not_hit_langgraph_step_limit(workflow_env, monkeypatch):
    # Each quality retry costs 4 graph steps; LangGraph's default limit is 25.
    llm = FakeLLM(judge=[ok(JUDGE_FAIL)])
    result = execute(monkeypatch, llm, specs=make_specs(max_iterations=7, data_sources=True, tools=True))
    assert result["status"] == "escalated"
    assert result["metadata"]["escalation"]["reason"] == "quality_exhausted"
    assert len(llm.answer_calls()) == 7
