# US1.1: Standardized multi-agent workflows for the ThinkLife Brain

**Epic:** 1 — Multi-Agent Orchestration & Workflow Reliability
**Author:** Naresh Rambhatla
**Branch:** `story/US1.1_agent_workflows`
**PR:** <add link when the PR is opened; base `staging`>

## User story

As a system orchestrator (ThinkLife Brain), I want standardized multi-agent workflows so that Zoe, Bard, and FAQ agents execute tasks predictably and safely.

## Summary of changes

* **One standard LangGraph workflow for every Brain agent.** The engine was a linear graph with fixed limits. It is now one graph that every agent runs, with a `WorkflowProfile` per agent that sets its limits (answer attempts, time budget, retries, safety checks, fallback models). Zoe's profile takes its limits from Zoe's own settings: 2 answer attempts, 30 seconds.
* **Parallel paths and conditional edges.** Knowledge search and tools run at the same time and meet at a new `gather_context` step. After each key step a conditional edge decides where to go: accept the answer, retry with feedback, try a fallback model, or escalate.
* **Checkpointing and resume.** The workflow saves its state after every step (SQLite by default). A run that crashes can be resumed from its last finished step, also after a server restart. A finished run deletes its checkpoint, because the saved prompt can contain user context such as the ACE score.
* **Failure handling.** Every failure ends in a retry, a fallback model, or an escalation with a safe message. Error text is never shown to the user as an answer. Output policy checks (`PolicyValidator`, US 2.1) run as a `check_safety` step; a policy block is never retried.
* **Agent communication pattern.** Inside one run, steps share typed state. Between agents there is no shared state: an agent sends an `AgentMessage` and the `AgentRouter` delivers it by running the recipient's own workflow. Decision and reasons are in ADR-001.
* **Workflow diagrams generated from the code.** A script writes the diagram and the profile table from the graph definition, and a test fails if they go stale. Every response carries the `workflow_version` that produced it.
* **Works with Epic 4 on `staging`.** Per-stage latency, real token cost and the Langfuse span details from Epic 4 are kept and now also cover quality retries, fallback models and the parallel branches.
* **Three bugs fixed on the way:** retries ignored the agent's `max_iterations` and always ran up to 5; a failed LLM judge call replaced good answers with the fallback message; Zoe's retries re-sent the same prompt without the validator's feedback.

Not in this story: Bard and the FAQ bot are not Brain agents yet (see Known limitations), so today the standard workflow runs Zoe and the generic `POST /api/brain` path.

## Files touched

Compared with `staging` (`2bf61a330`). All paths are under `backend/` unless shown otherwise.

| File | Change | Why |
| --- | --- | --- |
| `brain/cortex/workflow_engine.py` | Changed | The standard graph: typed state with reducers, parallel branches, routing, retries, fallback, safety, escalation, resume, diagram export. Keeps Epic 4 latency, cost and tracing. |
| `brain/cortex/workflow_profiles.py` | New | Built-in per-agent profiles and how a profile is resolved (built-in, then the agent's `ProcessingSpec`, then environment). |
| `brain/cortex/workflow_checkpoint.py` | New | Creates the checkpoint store: `sqlite` (default), `memory`, or `none`. |
| `brain/cortex/workflow_errors.py` | New | Error types and the classifier that decides if a provider error is worth retrying. |
| `brain/cortex/agent_router.py` | New | Registry of agents and delivery of `AgentMessage`s between them, with a hop limit. |
| `brain/cortex/cortex.py` | Changed | Adds `resume_execution`, `dispatch_agent_message`, `get_execution`. Epic 4 methods unchanged. |
| `brain/cortex/__init__.py` | Changed | Exports the router and `WORKFLOW_VERSION`. |
| `brain/specs/workflow_specs.py` | Changed | Adds `WorkflowProfile` and the `escalated` / `completed_with_errors` statuses. Removes a duplicate `WorkflowExecution` definition that the later one in the same file silently replaced. |
| `brain/specs/agent_specs.py` | Changed | Adds `AgentMessage` and `MessageIntent`. |
| `brain/specs/__init__.py` | Changed | Exports the new specs. |
| `plugins/zoe_agent.py` | Changed | Passes workflow metadata (steps, failures, escalation reason, execution id) through when Zoe shows her fallback. |
| `requirements.txt` | Changed | `langgraph>=0.6,<2`, `langgraph-checkpoint-sqlite`, `aiosqlite`. |
| `.env.example` | Changed | Workflow settings (all optional, commented out). |
| `scripts/export_workflow_diagrams.py` | New | Writes the diagram and profile table from the code. |
| `scripts/demo_workflow.py` | New | Seven scenarios on a simulated model; needs no API key. |
| `docs/workflows/README.md` | New | Nodes, failure table, configuration, checkpointing, metadata, changelog. |
| `docs/workflows/ADR-001-agent-communication.md` | New | The agent communication decision. |
| `docs/workflows/brain_workflow.md`, `profiles.md` | New | Generated diagram and profile table. |
| `tests/conftest.py`, `tests/workflow_fakes.py` | New | Test setup and a scripted fake LLM, vector store and tool registry. |
| `tests/test_workflow_graph.py` | New | Graph shape, parallel branches, profiles (15 tests). |
| `tests/test_workflow_checkpoint.py` | New | Crash, resume, restart, checkpoint cleanup (7 tests). |
| `tests/test_workflow_failures.py` | New | One test per row of the failure table (29 tests). |
| `tests/test_agent_router.py` | New | Hand-off, chained hand-off, hop limit, refusals (6 tests). |
| `tests/test_workflow_diagrams.py` | New | Generated docs match the code (3 tests). |
| `tests/test_workflow_observability.py` | New | Latency and cost on the new graph; `POST /api/brain` end to end (9 tests). |
| `.gitignore` (repo root) | Changed | Ignores the SQLite side files (`-wal`, `-shm`, `-journal`) of the local checkpoint database; `*.sqlite3` was already ignored. |

## Pseudocode

**One run** (`WorkflowEngine.execute_plan`):

```text
FUNCTION execute_plan(plan, request):
    profile = resolve_workflow_profile(plan.specs)      # built-in -> agent's ProcessingSpec -> env
    state   = new run state (execution_id, deadline = now + profile.run_timeout)
    RUN the graph with thread_id = execution_id          # state is checkpointed after every step:

    initialize
    IF data sources enabled AND tools enabled:
        RUN query_data_sources and execute_tools IN PARALLEL
    ELSE run whichever is enabled, or neither
    gather_context                                       # join point
    build_messages                                       # adds validator feedback on a retry
    call_provider                                        # see below
    check_safety                                         # PolicyValidator; BLOCK -> escalate, never retried
    validate_response                                    # rule checks, then optional LLM judge
    IF confidence >= threshold:            finalize
    ELSE IF attempts left AND time left:   back to build_messages with feedback
    ELSE:                                  escalate -> finalize

    result = content, status, confidence, metadata (steps, failures, latency_ms, actual_cost_usd, ...)
    IF status is not "failed": delete the checkpoint     # it holds the prompt
    RETURN result
```

**Provider call and failure routing** (`_call_provider_node`, `_fallback_provider_node`, `_escalate_node`):

```text
FUNCTION call_provider(state):
    REPEAT up to profile.provider_retry_attempts:
        TRY one call with a per-attempt timeout
        IF it answered: RETURN answer, route = "answered"
        IF the error is transient (timeout, 429, 5xx, dropped connection, empty reply)
           AND the time budget allows: wait (exponential backoff + jitter), try again
        ELSE: stop trying
    IF a fallback model is configured AND time is left: route = "fallback"
    ELSE: route = "escalate"

FUNCTION fallback_provider(state):
    FOR each remaining fallback model in profile order:
        TRY one call; IF it answered: RETURN answer and stay on this model for later retries
    route = "escalate"

FUNCTION escalate(state):
    content = safe message for the reason (policy message, low-confidence message, or default fallback)
    status  = "escalated"; log a warning with the execution id
    success = true only when the safe message is a proper answer (policy block, quality exhausted)
```

**Resume** (`WorkflowEngine.resume_execution`, `CortexFlow.resume_execution`):

```text
FUNCTION resume_execution(execution_id, plan, request):
    snapshot = load checkpoint for thread_id = execution_id
    IF none: raise KeyError
    IF steps remain: reset the time budget, continue the graph from the last finished step
    RETURN result                                        # finished steps are not run again
```

**Agent-to-agent message** (`AgentRouter.dispatch`):

```text
FUNCTION dispatch(message, user_context):
    IF message.hop >= 3 OR recipient unknown: RETURN a refusal (no exception)
    IF intent is "notify": call the recipient's handler, no workflow run
    request = new BrainRequest from payload["message"] and the envelope only   # not the sender's prompt or history
    response = recipient.process_request(request)                              # recipient's own workflow
    IF response hands off again: dispatch the next message, same correlation_id, hop + 1
    RETURN response with correlation_id, handled_by, hop
```

## Setup

* Packages added (versions the tests ran with):
  * `langgraph>=0.6,<2` — was unpinned; tested on 1.2.21
  * `langgraph-checkpoint-sqlite` — 3.1.3
  * `aiosqlite` — 0.22.1
* Python: 3.10 through 3.13. The `langfuse>=3.11` pin already on `staging` needs 3.10 or later.
* Environment variables (all optional): `WORKFLOW_FALLBACK_PROVIDERS`, `<AGENT>_FALLBACK_PROVIDERS` (for example `ZOE_FALLBACK_PROVIDERS`), `WORKFLOW_LLM_VALIDATION`, `WORKFLOW_CHECKPOINTER`, `WORKFLOW_CHECKPOINT_PATH`, `WORKFLOW_KEEP_COMPLETED_CHECKPOINTS`
* Config flags: `WORKFLOW_CHECKPOINTER=sqlite`, `WORKFLOW_LLM_VALIDATION=true`, `WORKFLOW_KEEP_COMPLETED_CHECKPOINTS=false`; no fallback models unless configured
* Migrations or seed data: None. The SQLite checkpoint file is created on first run at `backend/data/workflow_checkpoints.sqlite3` and is git-ignored.

## How to run

```bash
cd backend
pip install -r requirements.txt
python main.py
pytest tests/test_workflow_graph.py tests/test_workflow_checkpoint.py tests/test_workflow_failures.py tests/test_agent_router.py tests/test_workflow_diagrams.py tests/test_workflow_observability.py -v
```

No API key is needed for the tests or for the demo:

```bash
python scripts/demo_workflow.py            # seven scenarios: retry, fallback, escalate, policy block, resume, ...
python scripts/export_workflow_diagrams.py # regenerate docs/workflows/brain_workflow.md and profiles.md
pytest tests                               # whole backend suite: 109 tests
```

## Acceptance criteria mapping

| Task or criterion (from the epic) | Implemented in | Proof | Status |
| --- | --- | --- | --- |
| Design LangGraph state machine for agent workflows (conditional edges, parallel paths) | `workflow_engine.py` (`_build_graph`, routers), `workflow_profiles.py`, `WorkflowProfile` | `test_workflow_graph.py`: `test_happy_path_runs_every_stage_once`, `test_data_sources_and_tools_run_in_parallel`, `test_zoe_profile_takes_limits_from_its_processing_spec`; diagram in `docs/workflows/brain_workflow.md` | Done |
| Implement checkpointing for long-running workflows | `workflow_checkpoint.py`, `WorkflowEngine.resume_execution`, `CortexFlow.resume_execution` | `test_workflow_checkpoint.py`: `test_crashed_run_keeps_checkpoint_and_resumes_where_it_stopped`, `test_resume_survives_a_restart_with_sqlite`, `test_completed_runs_drop_their_checkpoint_by_default`; demo scenario 7 | Done |
| Define agent communication patterns (message passing vs shared state) | `agent_router.py`, `AgentMessage` in `agent_specs.py`, `docs/workflows/ADR-001-agent-communication.md` | `test_agent_router.py`: `test_handoff_runs_recipient_with_only_the_envelope`, `test_chained_handoff_keeps_one_correlation_id`, `test_hop_limit_stops_ping_pong` | Done. Pattern and router are in place; no agent uses a hand-off yet because only Zoe is a Brain agent |
| Implement failure handling transitions (retry / fallback / escalation) | `workflow_engine.py` (`_call_provider_node`, `_fallback_provider_node`, `_check_safety_node`, `_escalate_node`), `workflow_errors.py` | `test_workflow_failures.py`, one test per row of the failure table in `docs/workflows/README.md` (29 tests); demo scenarios 2 to 6 | Done |
| Document workflow diagrams for release traceability | `scripts/export_workflow_diagrams.py`, `docs/workflows/`, `WORKFLOW_VERSION` in response metadata | `test_workflow_diagrams.py`: `test_committed_workflow_docs_match_the_code`, `test_diagram_shows_every_node_and_failure_route` | Done |
| Zoe, Bard, and FAQ agents execute on the standard workflow | Zoe: `plugins/zoe_agent.py` through `CortexFlow` | Zoe and `POST /api/brain`: `test_workflow_observability.py::test_generic_brain_endpoint_runs_on_the_standard_workflow` | Partial. Zoe: done. FAQ and Bard are not Brain agents yet |
| Works with stories already on `staging` (Epic 4, US 6.1) | `workflow_engine.py` (stage timing, cost, spans) | `test_workflow_observability.py` (9 tests); `tests/test_release_6_1.py` and `tests/test_guardrails.py` still pass | Done |

Test totals on this branch: 69 tests for this story, 109 in the whole backend suite, all passing on Python 3.10, 3.11, 3.12 and 3.13.

## Dependencies and conflicts

* Depends on:
  * US 2.1 / 2.2 (guardrails): `check_safety` calls `PolicyValidator` and uses the messages in `fallback_handler.py`. Already on `staging`.
  * US 4.1 / 4.2 (observability): merged into this branch from `staging`; the engine keeps its latency, cost and tracing behaviour.
* Blocks: US 1.2 (same epic), and bringing the FAQ bot and Bard into the Brain, which should use the profiles and router from this story.
* Shared files also edited by other stories:
  * `backend/brain/cortex/workflow_engine.py`, `backend/brain/cortex/cortex.py`, `backend/.env.example` — US 4.1 / 4.2 (on `staging`, already resolved here) and US 5.1 / 5.2 (not on `staging` yet)
  * `backend/brain/specs/__init__.py`, `backend/brain/specs/agent_specs.py`, `backend/brain/specs/workflow_specs.py`, `backend/plugins/zoe_agent.py`, `backend/requirements.txt` — US 5.1 / 5.2
  * `.gitignore` — US 6.2
* Conflict with US 5.1 / 5.2: a trial merge of this branch with the `Shivansh` branch conflicts in `workflow_engine.py`, `cortex.py`, `main.py` and `.env.example`. The `Shivansh` branch conflicts with `staging` in the same four files on its own, so this story adds no new conflicting file.
* Overlap with US 5.2 to agree on: US 5.2 retries provider calls and adds a circuit breaker at the provider boundary (`brain/resilience/`). This story also retries transient provider errors, inside `call_provider`. With both merged, one failing call would be retried by both layers. Proposal: US 5.2 owns retries and the circuit breaker at the provider boundary; this workflow sets `provider_retry_attempts=1` and keeps what happens after those retries are spent (fallback model, escalation, time budget) plus the quality retry loop. An open circuit is treated as "go to the fallback model".
* Suggested merge order: after US 4.x and US 6.1 (done). For US 5.1 / 5.2, either order works, but the second story to merge must port its engine changes onto the first. If US 5.1 / 5.2 merges first, I will port its engine changes into this graph, the same way Epic 4 was ported.

## Known limitations

* Bard is not in the codebase as a Brain agent, and the FAQ bot runs as its own app (`faq-bot-rag/`) outside the Brain. Both get a profile entry and a router registration once they are Brain plugins. No agent is registered with the router at startup yet.
* Tool execution in the Brain is still a placeholder (unchanged by this story); the parallel branch is ready for real tools.
* The free OpenRouter fallback (`openrouter/free` when `OPENROUTER_API_KEY` is set) only switches on once the OpenRouter provider from US 5.x is merged. Until then, set `WORKFLOW_FALLBACK_PROVIDERS` to use a fallback model.
* The LLM judge's token cost is recorded on its Langfuse span but is not included in `actual_cost_usd`, the same as before this story.
* On resume, `latency_ms.total` counts from the original start time, so it includes the time the run was down.
* Checkpoints are stored in a local SQLite file, which suits one backend instance. Several instances would need a shared store (for example Postgres).
* Follow up story: US 1.2; FAQ bot and Bard integration into the Brain.
