"""
US 1.1 on top of Epic 4 (observability): the standard workflow keeps reporting
per-stage latency and real token cost, summed across quality retries and
fallback models, and the generic Brain endpoint runs on the same graph.

Run with: pytest tests/test_workflow_observability.py -v
"""

import pytest

from conftest import run
from workflow_fakes import (
    FakeLLM, GOOD_ANSWER, JUDGE_FAIL, JUDGE_PASS, err, install_fakes, make_plan, make_request, make_specs, ok,
)
from brain.cortex.workflow_engine import WorkflowEngine, _add_stage_durations
from evaluation import CostEstimator

USAGE = {"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500}
STAGES = {"total", "retrieval", "llm_generation", "validation"}


def priced(content: str, model: str = "gpt-4o-mini"):
    """A provider reply that carries the model name and token usage, like the real SDKs."""
    return {"success": True, "content": content, "metadata": {"provider": "fake", "model": model, "usage": dict(USAGE)}}


def execute(monkeypatch, llm, specs=None, **fake_kwargs):
    install_fakes(monkeypatch, llm, **fake_kwargs)
    engine = WorkflowEngine(checkpointer_kind="memory")
    return run(engine.execute_plan(make_plan(specs or make_specs()), make_request()))


def test_every_run_reports_a_latency_breakdown(workflow_env, monkeypatch):
    llm = FakeLLM(delay=0.02)
    result = execute(monkeypatch, llm, make_specs(data_sources=True), ds_delay=0.02)

    latency = result["metadata"]["latency_ms"]
    assert set(latency) == STAGES
    assert latency["retrieval"] >= 20
    assert latency["llm_generation"] >= 20
    assert latency["validation"] >= 20  # the judge call
    assert latency["total"] >= latency["retrieval"] + latency["llm_generation"] + latency["validation"] - 1


def test_stage_time_adds_up_across_quality_retries(workflow_env, monkeypatch):
    llm = FakeLLM(delay=0.02, judge=[ok(JUDGE_FAIL), ok(JUDGE_PASS)])
    result = execute(monkeypatch, llm)

    assert result["metadata"]["validation_attempts"] == 2
    latency = result["metadata"]["latency_ms"]
    assert latency["llm_generation"] >= 40  # two answers
    assert latency["validation"] >= 40      # two judge calls


def test_cost_adds_up_across_quality_retries(workflow_env, monkeypatch):
    llm = FakeLLM(scripts={"gpt-4o-mini": [priced(GOOD_ANSWER)]}, judge=[ok(JUDGE_FAIL), ok(JUDGE_PASS)])
    result = execute(monkeypatch, llm)

    one_call = CostEstimator.actual_cost("gpt-4o-mini", USAGE)
    assert one_call and one_call > 0
    assert len(llm.answer_calls()) == 2
    assert result["metadata"]["actual_cost_usd"] == pytest.approx(2 * one_call)


def test_fallback_model_time_and_cost_are_counted(workflow_env, monkeypatch):
    monkeypatch.setenv("WORKFLOW_FALLBACK_PROVIDERS", "openai:gpt-4o")
    llm = FakeLLM(scripts={
        "gpt-4o-mini": [err("Error code: 503 - overloaded")],
        "gpt-4o": [priced("Answer from the fallback model, complete.", model="gpt-4o")],
    })
    result = execute(monkeypatch, llm)

    assert result["metadata"]["provider_used"] == "openai:gpt-4o"
    assert result["metadata"]["actual_cost_usd"] == pytest.approx(CostEstimator.actual_cost("gpt-4o", USAGE))
    assert result["metadata"]["latency_ms"]["llm_generation"] > 0


def test_unknown_model_price_is_reported_as_unknown_not_zero(workflow_env, monkeypatch):
    llm = FakeLLM(scripts={"gpt-4o-mini": [priced(GOOD_ANSWER, model="some/unpriced-model")]})
    result = execute(monkeypatch, llm)
    assert result["success"] is True
    assert result["metadata"]["actual_cost_usd"] is None


def test_escalated_run_still_reports_latency_and_an_error(workflow_env, monkeypatch):
    llm = FakeLLM(scripts={"gpt-4o-mini": [err("Error code: 401 - invalid api key")]})
    result = execute(monkeypatch, llm)

    assert result["status"] == "escalated"
    assert set(result["metadata"]["latency_ms"]) == STAGES
    assert result["metadata"]["error"]  # single-string form read by POST /api/brain
    assert result["metadata"]["actual_cost_usd"] is None


def test_successful_run_has_no_error_string(workflow_env, monkeypatch):
    result = execute(monkeypatch, FakeLLM())
    assert result["success"] is True
    assert result["metadata"]["error"] is None


def test_stage_duration_reducer_sums_per_stage():
    total = _add_stage_durations({"llm_generation": 1.0}, {"llm_generation": 0.5, "validation": 0.25})
    assert total == {"llm_generation": 1.5, "validation": 0.25}
    assert _add_stage_durations(None, None) == {}


def test_generic_brain_endpoint_runs_on_the_standard_workflow(workflow_env, monkeypatch):
    """POST /api/brain (Cortex.process_request, Epic 4) goes through the US 1.1 graph and feeds analytics."""
    monkeypatch.setenv("WORKFLOW_CHECKPOINTER", "memory")
    llm = FakeLLM(scripts={None: [priced(GOOD_ANSWER)]})  # process_request leaves the model to the provider default
    install_fakes(monkeypatch, llm)

    from brain.cortex.cortex import CortexFlow

    async def scenario():
        cortex = CortexFlow()
        try:
            response = await cortex.process_request({
                "message": "When does the program run?",
                "application": "general",
                "user_context": {"user_id": "u1", "is_authenticated": True},
            })
            return response, await cortex.get_health_status(), await cortex.get_analytics()
        finally:
            await cortex.shutdown()

    response, health, analytics = run(scenario())

    assert response["success"] is True
    assert response["message"] == GOOD_ANSWER
    assert response["error"] is None
    meta = response["metadata"]
    assert meta["workflow_version"]
    assert "check_safety" in meta["execution_steps"]
    assert set(meta["latency_ms"]) == STAGES
    assert meta["actual_cost_usd"] == pytest.approx(CostEstimator.actual_cost("gpt-4o-mini", USAGE))

    assert health["components"]["workflow_engine"] is True
    assert analytics["total_requests"] == 1
    assert analytics["priced_requests"] == 1
    assert analytics["total_cost_usd"] == pytest.approx(meta["actual_cost_usd"])
