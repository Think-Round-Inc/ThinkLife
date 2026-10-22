# ThinkLife Brain workflows

Every Brain agent runs one standard LangGraph workflow, defined in `brain/cortex/workflow_engine.py`. A **workflow profile** per agent sets its limits, fallback providers and safety checks. This folder documents the workflow for US 1.1 (Epic 1 — Multi-Agent Orchestration & Workflow Reliability).

| File | What it is |
| --- | --- |
| [brain_workflow.md](brain_workflow.md) | The workflow diagram, **generated from the code** |
| [profiles.md](profiles.md) | Built-in per-agent profiles, **generated from the code** |
| [ADR-001-agent-communication.md](ADR-001-agent-communication.md) | How agents talk to each other |

The two generated files are checked by `tests/test_workflow_diagrams.py`: change the graph without regenerating them and the tests fail. Regenerate with:

```bash
cd backend
python scripts/export_workflow_diagrams.py
```

## Requirements

Runs on Python 3.9 through 3.13. On Python 3.9, pip installs LangGraph 0.6; on 3.10 and later, LangGraph 1.x. The workflow only uses features both versions have, and the tests pass on both (`requirements.txt` pins `aiosqlite<0.22` on 3.9, which the 3.9 SQLite checkpoint store needs).

## Nodes

| Node | What it does | Added in US 1.1 |
| --- | --- | --- |
| `initialize` | Starts the run; fans out to the enabled branches | |
| `query_data_sources` | Queries the vector DB; retries transient errors; on failure continues without context | |
| `execute_tools` | Runs the agent's tools (still a placeholder in the Brain); runs **in parallel** with `query_data_sources` | |
| `gather_context` | Join point after the parallel branches; prepares retrieved text for the groundedness check | yes |
| `build_messages` | Builds the prompt; on a retry it adds the validator's feedback (also on the agent-prompt path Zoe uses, which ignored feedback before) | |
| `call_provider` | One LLM answer. Retries transient errors with exponential backoff and a per-attempt timeout; if it still fails, routes to `fallback_provider` or `escalate` | |
| `fallback_provider` | Tries each fallback model in the profile once, then escalates | yes |
| `check_safety` | Runs `PolicyValidator` (`brain/guardrails/policy_validator.py`) on the answer; a BLOCK escalates | yes |
| `validate_response` | Rule checks, then an optional LLM judge; accept, retry with feedback, or escalate | |
| `escalate` | Safe message for the user, status `escalated`, reason in metadata, `ESCALATED` warning in the log with the execution id | yes |
| `finalize` | Sets the final status and end time | |

## Failure paths

Every failure ends in a retry, a fallback, or an escalation — never in error text shown as an answer. Each row has a test in `tests/test_workflow_failures.py`.

| Failure | Caught in | Path | `success` | User sees |
| --- | --- | --- | --- | --- |
| Timeout, 429, 5xx, dropped connection, empty completion | `call_provider` | Retried with exponential backoff, up to `provider_retry_attempts` | true | Normal answer, a little slower |
| Provider still failing, or can't start (missing key, bad model) | `call_provider` | `fallback_provider` tries each fallback model once | true | Answer from the fallback model |
| Fallback fails too, or no fallback configured | `fallback_provider` / `call_provider` | `escalate` (`provider_failure`) | **false** | The agent's own fallback (Zoe's), as before |
| Non-retryable provider error (401, 400, billing quota) | `call_provider` | No retry; fallback, else `escalate` | false | The agent's own fallback |
| Policy BLOCK (leaked key, SSN pattern, empty/degenerate output) | `check_safety` | `escalate` (`policy_block`), never retried | true | The policy's message from `fallback_handler.py` |
| Weak answer (score below 0.75) | `validate_response` | Back to `build_messages` with feedback, up to `max_iterations` | true | Improved answer |
| Attempts used up | `validate_response` | `escalate` (`quality_exhausted`) | true | "I'm not confident I can provide an accurate answer…" |
| Time budget spent before a valid answer | any routing step | No new retry or fallback starts; `escalate` (`timeout`) | false | The agent's own fallback |
| Vector DB unavailable | `query_data_sources` | Retry, then continue without context; error recorded | true | Answer without retrieved context |
| Bug in a node (unexpected exception) | the graph | Run stops with status `failed`; checkpoint kept so it can be resumed | false | The agent's own fallback |

`success: false` lets the calling agent show its own fallback, which is what happened before when the provider errored.

The time budget is `ProcessingSpec.timeout_seconds` (Zoe: 30 s). It stops **new** work; a provider call already running may finish. A hard cap of budget + `provider_timeout_seconds` stops a run that hangs.

## Configuration

Profiles resolve in this order, later wins: built-in profile (`brain/cortex/workflow_profiles.py`) → the agent's `ProcessingSpec` → environment.

| Variable | Default | Effect |
| --- | --- | --- |
| `WORKFLOW_FALLBACK_PROVIDERS` | unset | Fallback models for every agent, e.g. `openrouter:openrouter/free,openai:gpt-4o-mini` |
| `<AGENT>_FALLBACK_PROVIDERS` | unset | Same, for one agent, e.g. `ZOE_FALLBACK_PROVIDERS` |
| `OPENROUTER_API_KEY` | unset | When set and no fallback is configured, `openrouter:openrouter/free` becomes the fallback |
| `WORKFLOW_LLM_VALIDATION` | `true` | `false` turns off the LLM judge: one LLM call per message instead of two (useful on the free tier) |
| `WORKFLOW_CHECKPOINTER` | `sqlite` | `sqlite`, `memory`, or `none` |
| `WORKFLOW_CHECKPOINT_PATH` | `backend/data/workflow_checkpoints.sqlite3` | SQLite checkpoint file (git-ignored) |
| `WORKFLOW_KEEP_COMPLETED_CHECKPOINTS` | `false` | Keep checkpoints of finished runs for debugging |

The provider SDKs' own retries are turned off (`max_retries=0`) so retries happen once, in the workflow, where they show up in traces.

## Checkpointing and resume

The graph is compiled with a checkpointer and `thread_id` = execution id, so LangGraph saves the state after every step.

- A run that **fails** (bug, crash) keeps its checkpoint; the response has `metadata.resumable: true`. Resume it with `CortexFlow.resume_execution(execution_id, agent_specs, request)`. Steps that already finished are not re-run.
- A run that **finishes** (completed or escalated) deletes its checkpoint, because the state holds the prompt, which includes user context such as the ACE score.
- The live request and agent specs are passed as LangGraph runtime context, not state, so they are never written to the checkpoint store.
- `CortexFlow.get_execution(execution_id)` returns a summary of a recent run without message content.

## Response metadata

Every workflow response's `metadata` now includes:

| Key | Meaning |
| --- | --- |
| `workflow_version` | Graph version that produced the answer (`WORKFLOW_VERSION`) |
| `agent_type` | Profile used |
| `execution_steps` | Nodes in the order they ran |
| `failures` | `{node, kind, detail}` for each failure handled |
| `escalation` | `{reason, flagged_for_review}` when escalated, else `null` |
| `provider_used` | `type:model` that produced the answer |
| `validation_mode` | `llm_judge`, `rules`, `judge_failed` or `judge_unparsed` |
| `safety` | Policy check result |
| `resumable` | Whether the run can be resumed from its checkpoint |

## Changing the workflow

1. Edit `brain/cortex/workflow_engine.py`. Use the `NODE_*` constants and route with conditional edges (not exceptions), so the diagram picks every route up.
2. Bump `WORKFLOW_VERSION` and add a line to the changelog below.
3. Run `python scripts/export_workflow_diagrams.py` and commit the regenerated files.
4. Run `pytest tests` from `backend/`.

## Changelog

| Version | Change |
| --- | --- |
| 1.1.0 | US 1.1: typed state with reducers; data sources and tools in parallel; per-agent profiles; checkpointing and resume; transient-error retries with backoff; fallback provider; policy check node; escalation node; agent message router; generated diagrams. Fixes: retries now respect the agent's `max_iterations` (was a fixed 5); a failed LLM judge no longer replaces good answers with the fallback message; retries on Zoe's prompt path now include the validator's feedback. |
| 1.0 | Linear graph: data sources → tools → messages → provider → validate, with a quality retry loop. |
