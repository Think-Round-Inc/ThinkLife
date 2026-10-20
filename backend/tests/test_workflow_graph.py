"""
US 1.1 Task 1 - LangGraph state machine: routing, parallel paths, typed
state, per-agent profiles.

Run with: pytest tests/test_workflow_graph.py -v
"""

import pytest

from conftest import run
from workflow_fakes import (
    FakeLLM, GOOD_ANSWER, JUDGE_FAIL, install_fakes, make_plan, make_request, make_specs, ok,
)
from brain.cortex.workflow_engine import WORKFLOW_VERSION, WorkflowEngine
from brain.cortex.workflow_profiles import resolve_workflow_profile, parse_provider_list
from brain.specs import ProviderSpec


def execute(specs, monkeypatch, llm=None, **fake_kwargs):
    llm = llm or FakeLLM()
    install_fakes(monkeypatch, llm, **fake_kwargs)
    engine = WorkflowEngine(checkpointer_kind="memory")
    result = run(engine.execute_plan(make_plan(specs), make_request()))
    return result, llm


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def test_happy_path_runs_every_stage_once(workflow_env, monkeypatch):
    result, llm = execute(make_specs(), monkeypatch)

    assert result["success"] is True
    assert result["status"] == "completed"
    assert result["content"] == GOOD_ANSWER
    assert result["metadata"]["execution_steps"] == [
        "initialize", "gather_context", "build_messages", "call_provider",
        "check_safety", "validate_response", "finalize",
    ]
    assert result["metadata"]["workflow_version"] == WORKFLOW_VERSION
    assert result["metadata"]["agent_type"] == "test"
    assert result["metadata"]["provider_used"] == "openai:gpt-4o-mini"
    assert len(llm.answer_calls()) == 1


def test_data_sources_and_tools_run_in_parallel(workflow_env, monkeypatch):
    timeline = []
    result, _ = execute(make_specs(data_sources=True, tools=True), monkeypatch,
                        ds_delay=0.2, timeline=timeline)

    steps = result["metadata"]["execution_steps"]
    assert {"query_data_sources", "execute_tools"} <= set(steps)
    # The join runs once, after both branches
    assert steps.count("gather_context") == 1
    assert steps.index("gather_context") > max(steps.index("query_data_sources"), steps.index("execute_tools"))
    assert steps.count("build_messages") == 1
    # Tools did not wait for the slow (0.2 s) data source query. In the old,
    # sequential graph execute_tools only started after query_data_sources ended.
    times = dict(timeline)
    assert times["tools"] < times["ds_end"]


def test_only_data_sources_skips_tools(workflow_env, monkeypatch):
    result, llm = execute(make_specs(data_sources=True), monkeypatch)
    steps = result["metadata"]["execution_steps"]
    assert "query_data_sources" in steps and "execute_tools" not in steps
    # Retrieved context reaches the prompt
    prompt = " ".join(m["content"] for m in llm.answer_calls()[0]["messages"])
    assert "The program runs on Saturdays." in prompt


def test_only_tools_skips_data_sources(workflow_env, monkeypatch):
    result, _ = execute(make_specs(tools=True), monkeypatch)
    steps = result["metadata"]["execution_steps"]
    assert "execute_tools" in steps and "query_data_sources" not in steps


# ---------------------------------------------------------------------------
# Quality loop and profile limits
# ---------------------------------------------------------------------------

def test_max_iterations_comes_from_the_agent_processing_spec(workflow_env, monkeypatch):
    llm = FakeLLM(judge=[ok(JUDGE_FAIL)])
    result, llm = execute(make_specs(max_iterations=2), monkeypatch, llm=llm)

    assert len(llm.answer_calls()) == 2  # not the old hard-coded 5
    assert result["status"] == "escalated"
    assert result["metadata"]["escalation"]["reason"] == "quality_exhausted"
    assert result["metadata"]["validation_attempts"] == 2


def test_retry_sends_validation_feedback_on_agent_prompt_path(workflow_env, monkeypatch):
    # Zoe-style specs (messages_context). Before US 1.1 this path re-sent the
    # identical prompt on retry.
    specs = make_specs(messages_context={"system_prompt": "You are Zoe.", "conversation_history": [],
                                         "current_message": "Hi"})
    llm = FakeLLM(judge=[ok(JUDGE_FAIL), ok("SCORE: 95\nREASONING: good")])
    result, llm = execute(specs, monkeypatch, llm=llm)

    first, second = llm.answer_calls()
    assert not any("Response Improvement Required" in m["content"] for m in first["messages"])
    assert any("Response Improvement Required" in m["content"] for m in second["messages"])
    assert result["success"] is True
    assert result["metadata"]["validation_attempts"] == 2


def test_judge_off_accepts_answers_that_pass_rule_checks(workflow_env, monkeypatch):
    monkeypatch.setenv("WORKFLOW_LLM_VALIDATION", "false")
    result, llm = execute(make_specs(), monkeypatch)
    assert result["success"] is True
    assert result["metadata"]["validation_mode"] == "rules"
    assert not any(c["judge"] for c in llm.calls)  # one LLM call per message instead of two


# ---------------------------------------------------------------------------
# State hygiene
# ---------------------------------------------------------------------------

def test_request_and_specs_are_not_checkpointed(workflow_env, monkeypatch):
    monkeypatch.setenv("WORKFLOW_KEEP_COMPLETED_CHECKPOINTS", "true")
    install_fakes(monkeypatch, FakeLLM())
    engine = WorkflowEngine(checkpointer_kind="memory")

    async def go():
        result = await engine.execute_plan(make_plan(make_specs()), make_request())
        snapshot = await engine.graph.aget_state({"configurable": {"thread_id": result["execution_id"]}})
        return snapshot.values

    values = run(go())
    assert values["status"] == "completed"
    assert "request" not in values and "specs" not in values


# ---------------------------------------------------------------------------
# Profile resolution
# ---------------------------------------------------------------------------

def test_zoe_profile_takes_limits_from_its_processing_spec(workflow_env):
    profile = resolve_workflow_profile(make_specs(agent_type="zoe", max_iterations=2, timeout_seconds=30.0))
    assert profile.agent_type == "zoe"
    assert profile.max_iterations == 2
    assert profile.run_timeout_seconds == 30.0
    assert profile.safety_checks is True


def test_unknown_agent_gets_default_profile(workflow_env):
    profile = resolve_workflow_profile(make_specs(agent_type="bard"))
    assert profile.agent_type == "bard"
    assert profile.provider_retry_attempts == 3


def test_fallback_providers_from_env_inherit_primary_settings(workflow_env, monkeypatch):
    monkeypatch.setenv("WORKFLOW_FALLBACK_PROVIDERS", "openrouter:openrouter/free, anthropic:claude-3-haiku")
    profile = resolve_workflow_profile(make_specs())
    assert [(p.provider_type, p.model) for p in profile.fallback_providers] == [
        ("openrouter", "openrouter/free"), ("anthropic", "claude-3-haiku"),
    ]
    assert profile.fallback_providers[0].max_tokens == 300  # copied from the primary


def test_agent_specific_fallback_env_wins(workflow_env, monkeypatch):
    monkeypatch.setenv("WORKFLOW_FALLBACK_PROVIDERS", "openrouter:openrouter/free")
    monkeypatch.setenv("TEST_FALLBACK_PROVIDERS", "gemini:gemini-1.5-flash")
    profile = resolve_workflow_profile(make_specs())
    assert [(p.provider_type, p.model) for p in profile.fallback_providers] == [("gemini", "gemini-1.5-flash")]


def test_openrouter_key_adds_free_fallback_by_default(workflow_env, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    profile = resolve_workflow_profile(make_specs(provider="openai", model="gpt-4o-mini"))
    assert [(p.provider_type, p.model) for p in profile.fallback_providers] == [("openrouter", "openrouter/free")]

    # ...but never a fallback to the provider that is already failing
    same = resolve_workflow_profile(make_specs(provider="openrouter", model="openrouter/free"))
    assert same.fallback_providers == []


def test_malformed_fallback_entries_are_ignored(workflow_env):
    specs = parse_provider_list("openrouter, :model, openai:gpt-4o-mini", template=ProviderSpec(provider_type="x"))
    assert [(p.provider_type, p.model) for p in specs] == [("openai", "gpt-4o-mini")]
