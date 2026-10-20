# ADR-001: How agents in the ThinkLife Brain communicate

- **Status:** Accepted (US 1.1, Epic 1 — Multi-Agent Orchestration & Workflow Reliability)
- **Date:** 2026-10-20
- **Code:** `brain/cortex/agent_router.py`, `AgentMessage` in `brain/specs/agent_specs.py`

## Decision

**Shared state inside a run, message passing between agents.**

1. **Inside one workflow run**, the LangGraph nodes share one typed state (`WorkflowState` in `brain/cortex/workflow_engine.py`). Nodes never call each other; each returns only the keys it changed. List fields use `operator.add` reducers so parallel branches can both write.
2. **Between agents** (Zoe, FAQ, Bard, …) there is no shared state and no direct call. An agent that needs another agent sends an `AgentMessage`; the `AgentRouter` delivers it by running the recipient's own workflow (`agent.process_request`).

## The envelope

| Field | Meaning |
| --- | --- |
| `sender`, `recipient` | Agent names as registered with the router (lowercase) |
| `intent` | `handoff` — recipient answers the user from now on; `query` — sender wants an answer back; `notify` — event, no workflow run |
| `payload` | `payload["message"]` is the text the recipient works on (required for `handoff` and `query`) |
| `correlation_id` | Same on every hop of one user request; defaults to the first `message_id` |
| `hop` | Incremented on each delivery; the router refuses at 3 hops to stop ping-pong |

To hand off again, a recipient returns an `AgentMessage` in its response metadata under `agent_message`; the router follows it and keeps the `correlation_id`.

## Why

- **Privacy.** Zoe's prompt includes user context such as the ACE score. With message passing, a recipient only ever sees the envelope and the message text — never another agent's prompt, history or user profile. Shared state between agents would make that leak the default.
- **Traceability.** One `correlation_id` links every hop in logs and Langfuse traces, so a multi-agent answer can be followed end to end.
- **Independent failure handling.** Each agent's run keeps its own retries, fallback and escalation (see `README.md`). One agent failing does not corrupt another's state.
- **Agents can be added without touching the others.** A new agent registers with the router and gets a `WorkflowProfile`; nothing else changes.

## Alternatives considered

- **One big shared graph with every agent as a subgraph.** Simpler routing, but every agent would see every other agent's state, and one agent's retries would hold up the others.
- **Agents calling each other directly.** No central place for hop limits, tracing or access rules.

## Usage

```python
from brain.cortex import get_agent_router
from brain.specs import AgentMessage, MessageIntent

router = get_agent_router()
router.register("faq", faq_agent)          # once, at startup

response = await router.dispatch(
    AgentMessage(sender="zoe", recipient="faq", intent=MessageIntent.HANDOFF,
                 payload={"message": "When does the program meet?"}),
    user_context,
)
# or: await cortex.dispatch_agent_message(message, user_context)
```

## Open questions (next team meeting)

- **Bard** appears only in `docs/BRAIN_ARCHITECTURE.md`; there is no Bard agent in the code. It gets a profile and a router registration once it exists.
- **FAQ bot** runs as its own FastAPI app (`faq-bot-rag/`, FAISS + OpenAI embeddings) and does not go through the Brain. Should it become a Brain plugin so Zoe can hand off to it?
- Nothing registers agents with the router at startup yet, because only Zoe is a Brain plugin today.
