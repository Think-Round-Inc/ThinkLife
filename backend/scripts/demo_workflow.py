"""
US 1.1 workflow demo: runs the standard Brain workflow through seven
scenarios and prints the path each one takes.

    cd backend
    python scripts/demo_workflow.py            # all scenarios
    python scripts/demo_workflow.py 3 5        # only scenarios 3 and 5

Uses a simulated model (the same fakes as the tests), so it needs no API
keys, network, Docker or Keycloak, and every failure happens on cue.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
for path in (BACKEND_DIR, BACKEND_DIR / "tests"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Keep the demo independent of a developer's .env
for name in ("OPENROUTER_API_KEY", "WORKFLOW_FALLBACK_PROVIDERS", "DEMO_FALLBACK_PROVIDERS",
             "WORKFLOW_LLM_VALIDATION", "LANGFUSE_PUBLIC_KEY"):
    os.environ.pop(name, None)
os.environ["WORKFLOW_CHECKPOINTER"] = "memory"
logging.disable(logging.CRITICAL)

from workflow_fakes import (  # noqa: E402
    FakeLLM, GOOD_ANSWER, JUDGE_FAIL, JUDGE_PASS, err, install_fakes, make_plan, make_request, make_specs, ok,
)
from brain.cortex.workflow_engine import WORKFLOW_VERSION, WorkflowEngine  # noqa: E402
import brain.cortex.workflow_profiles as profiles  # noqa: E402
from brain.specs import WorkflowProfile  # noqa: E402

FREE = "openrouter/free"
LEAKED = "Sure, the key you need is sk-abcdefghijklmnopqrstuvwxyz123456 for the service."


class _Patch:
    """Stand-in for pytest's monkeypatch."""

    def setattr(self, obj, name, value):
        setattr(obj, name, value)


def _demo_profile():
    profiles.BUILTIN_PROFILES["demo"] = WorkflowProfile(
        agent_type="demo", max_iterations=3, run_timeout_seconds=30.0, provider_timeout_seconds=5.0,
        retry_initial_interval=0.05, retry_max_interval=0.2,
    )


def _path(steps):
    """Readable path; the two parallel branches are shown side by side."""
    out, i = [], 0
    while i < len(steps):
        pair = set(steps[i:i + 2])
        if pair == {"query_data_sources", "execute_tools"}:
            out.append("[query_data_sources ‖ execute_tools]")
            i += 2
            continue
        out.append(steps[i])
        i += 1
    return " → ".join(out)


def _show(number, title, what_happens, result, llm):
    md = result["metadata"]
    answers = len(llm.answer_calls())
    judges = sum(1 for c in llm.calls if c["judge"])
    print(f"\n[{number}] {title}")
    print(f"    setup:      {what_happens}")
    print(f"    path:       {_path(md['execution_steps'])}")
    status = result["status"]
    if md.get("escalation"):
        status += f" (reason: {md['escalation']['reason']})"
    print(f"    status:     {status} · success={result['success']}")
    print(f"    model:      {md.get('provider_used') or '—'} · {answers} answer call(s), {judges} scoring call(s)")
    for f in md.get("failures", []):
        print(f"    handled:    {f['node']} → {f['kind']}: {f['detail'][:70]}")
    print(f"    user sees:  \"{result['content'][:90]}\"")


async def _run(llm, specs, env=None, **fake_kwargs):
    for key, value in (env or {}).items():
        os.environ[key] = value
    try:
        install_fakes(_Patch(), llm, **fake_kwargs)
        engine = WorkflowEngine(checkpointer_kind="memory")
        return await engine.execute_plan(make_plan(specs), make_request())
    finally:
        for key in (env or {}):
            os.environ.pop(key, None)


async def scenario_1():
    llm = FakeLLM()
    result = await _run(llm, make_specs(agent_type="demo", data_sources=True, tools=True))
    _show(1, "Normal question", "data sources and tools both enabled; model answers well", result, llm)


async def scenario_2():
    llm = FakeLLM(scripts={"gpt-4o-mini": [err("Error code: 429 - Rate limit reached"), ok(GOOD_ANSWER)]})
    result = await _run(llm, make_specs(agent_type="demo"))
    _show(2, "Rate limit, then success", "first call gets HTTP 429; retried with backoff", result, llm)


async def scenario_3():
    llm = FakeLLM(scripts={"gpt-4o-mini": [err("Error code: 503 - Service Unavailable")], FREE: [ok(GOOD_ANSWER)]})
    result = await _run(llm, make_specs(agent_type="demo"), env={"WORKFLOW_FALLBACK_PROVIDERS": f"openrouter:{FREE}"})
    _show(3, "Main model down → fallback model", "OpenAI returns 503 on every try; fallback is openrouter/free",
          result, llm)


async def scenario_4():
    llm = FakeLLM(scripts={"gpt-4o-mini": [err("Error code: 503")], FREE: [err("Error code: 503")]})
    result = await _run(llm, make_specs(agent_type="demo"), env={"WORKFLOW_FALLBACK_PROVIDERS": f"openrouter:{FREE}"})
    _show(4, "Everything down → escalate", "main and fallback models both fail", result, llm)


async def scenario_5():
    llm = FakeLLM(scripts={"gpt-4o-mini": [ok(LEAKED)]})
    result = await _run(llm, make_specs(agent_type="demo"))
    _show(5, "Unsafe answer → blocked, never retried", "model's answer contains an API key", result, llm)


async def scenario_6():
    llm = FakeLLM(judge=[ok(JUDGE_FAIL)])
    result = await _run(llm, make_specs(agent_type="demo", max_iterations=3))
    _show(6, "Weak answers → retry with feedback → escalate",
          "scoring model rates every answer 20/100; agent allows 3 attempts", result, llm)


async def scenario_7():
    llm = FakeLLM()
    install_fakes(_Patch(), llm)
    engine = WorkflowEngine(checkpointer_kind="memory")
    original = engine._build_messages_node
    calls = {"n": 0}

    async def crash_once(state, runtime):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated crash")
        return await original(state, runtime)

    engine._build_messages_node = crash_once
    specs, request = make_specs(agent_type="demo", data_sources=True), make_request()
    failed = await engine.execute_plan(make_plan(specs), request)
    snapshot = await engine.graph.aget_state({"configurable": {"thread_id": failed["execution_id"]}})
    resumed = await engine.resume_execution(failed["execution_id"], make_plan(specs), request)

    print("\n[7] Crash mid-run → resume from checkpoint")
    print("    setup:      build_messages crashes the first time it runs")
    print(f"    first run:  {failed['status']} · resumable={failed['metadata']['resumable']} · "
          f"checkpoint waiting at {list(snapshot.next)}")
    print(f"    resumed:    {resumed['status']} · success={resumed['success']}")
    print(f"    path:       {_path(resumed['metadata']['execution_steps'])}")
    print(f"    note:       initialize and query_data_sources ran once in total; they were not repeated")


SCENARIOS = [scenario_1, scenario_2, scenario_3, scenario_4, scenario_5, scenario_6, scenario_7]


async def main(selected):
    _demo_profile()
    print(f"ThinkLife Brain workflow v{WORKFLOW_VERSION} — US 1.1 demo (simulated model, no API keys)")
    for number, scenario in enumerate(SCENARIOS, start=1):
        if not selected or number in selected:
            await scenario()
    print()


if __name__ == "__main__":
    asyncio.run(main({int(a) for a in sys.argv[1:] if a.isdigit()}))
