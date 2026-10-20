"""
US 1.1 Task 2 - checkpointing for long-running workflows: a run that crashes
mid-way resumes from its last finished step, including after a restart
(SQLite), and finished runs do not leave user data behind.

Run with: pytest tests/test_workflow_checkpoint.py -v
"""

import pytest

from conftest import run
from workflow_fakes import FakeLLM, GOOD_ANSWER, install_fakes, make_plan, make_request, make_specs
from brain.cortex.workflow_engine import WorkflowEngine


def crash_once(engine: WorkflowEngine, node_attr: str):
    """Make one node raise the first time it runs, as a process crash would."""
    original = getattr(engine, node_attr)
    calls = {"n": 0}

    async def flaky(state, runtime):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated crash")
        return await original(state, runtime)

    setattr(engine, node_attr, flaky)  # before initialize(), so the graph picks it up
    return calls


def thread_state(engine, execution_id):
    return engine.graph.aget_state({"configurable": {"thread_id": execution_id}})


def test_crashed_run_keeps_checkpoint_and_resumes_where_it_stopped(workflow_env, monkeypatch):
    llm = FakeLLM()
    install_fakes(monkeypatch, llm)
    engine = WorkflowEngine(checkpointer_kind="memory")
    calls = crash_once(engine, "_build_messages_node")
    specs, request = make_specs(data_sources=True), make_request()

    async def go():
        failed = await engine.execute_plan(make_plan(specs), request)
        snapshot = await thread_state(engine, failed["execution_id"])
        resumed = await engine.resume_execution(failed["execution_id"], make_plan(specs), request)
        after = await thread_state(engine, failed["execution_id"])
        return failed, snapshot, resumed, after

    failed, snapshot, resumed, after = run(go())

    assert failed["success"] is False
    assert failed["status"] == "failed"
    assert failed["metadata"]["resumable"] is True
    assert snapshot.next == ("build_messages",)

    assert resumed["success"] is True
    assert resumed["content"] == GOOD_ANSWER
    steps = resumed["metadata"]["execution_steps"]
    # Steps finished before the crash were not re-run
    assert steps.count("initialize") == 1
    assert steps.count("query_data_sources") == 1
    assert calls["n"] == 2
    assert len(llm.answer_calls()) == 1
    # The finished run's checkpoint is cleaned up
    assert not after.values


def test_resume_survives_a_restart_with_sqlite(workflow_env, monkeypatch, tmp_path):
    install_fakes(monkeypatch, FakeLLM())
    db = str(tmp_path / "checkpoints.sqlite3")
    specs, request = make_specs(), make_request()

    async def crash_then_restart():
        first = WorkflowEngine(checkpointer_kind="sqlite", checkpoint_path=db)
        crash_once(first, "_build_messages_node")
        failed = await first.execute_plan(make_plan(specs), request)
        await first.shutdown()  # process goes away

        second = WorkflowEngine(checkpointer_kind="sqlite", checkpoint_path=db)
        await second.initialize()
        resumed = await second.resume_execution(failed["execution_id"], make_plan(specs), request)
        await second.shutdown()
        return failed, resumed

    failed, resumed = run(crash_then_restart())
    assert failed["status"] == "failed"
    assert resumed["success"] is True
    assert resumed["metadata"]["execution_steps"].count("initialize") == 1


def test_completed_runs_drop_their_checkpoint_by_default(workflow_env, monkeypatch):
    install_fakes(monkeypatch, FakeLLM())
    engine = WorkflowEngine(checkpointer_kind="memory")

    async def go():
        result = await engine.execute_plan(make_plan(make_specs()), make_request())
        return result, await thread_state(engine, result["execution_id"])

    result, snapshot = run(go())
    assert result["success"] is True
    assert not snapshot.values  # prompt + user context not kept on disk


def test_keep_completed_checkpoints_flag(workflow_env, monkeypatch):
    monkeypatch.setenv("WORKFLOW_KEEP_COMPLETED_CHECKPOINTS", "true")
    install_fakes(monkeypatch, FakeLLM())
    engine = WorkflowEngine(checkpointer_kind="memory")

    async def go():
        result = await engine.execute_plan(make_plan(make_specs()), make_request())
        return await thread_state(engine, result["execution_id"])

    assert run(go()).values["status"] == "completed"


def test_get_execution_summary_has_no_message_content(workflow_env, monkeypatch):
    install_fakes(monkeypatch, FakeLLM())
    engine = WorkflowEngine(checkpointer_kind="memory")

    async def go():
        result = await engine.execute_plan(make_plan(make_specs()), make_request("my private question"))
        return await engine.get_execution(result["execution_id"])

    summary = run(go())
    assert summary["status"] == "completed"
    assert summary["execution_steps"][-1] == "finalize"
    assert "my private question" not in repr(summary)
    assert GOOD_ANSWER not in repr(summary)


def test_resume_unknown_execution_raises(workflow_env, monkeypatch):
    install_fakes(monkeypatch, FakeLLM())
    engine = WorkflowEngine(checkpointer_kind="memory")
    with pytest.raises(KeyError):
        run(engine.resume_execution("does-not-exist", make_plan(make_specs()), make_request()))


def test_checkpointing_can_be_turned_off(workflow_env, monkeypatch):
    install_fakes(monkeypatch, FakeLLM())
    engine = WorkflowEngine(checkpointer_kind="none")
    result = run(engine.execute_plan(make_plan(make_specs()), make_request()))
    assert result["success"] is True
    assert engine.checkpointer is None
