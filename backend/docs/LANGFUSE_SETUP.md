# Langfuse Setup (Observability -- Epic 4, Phase 1)

ThinkLife uses **Langfuse** (free tier, no card required) for tracing, cost, latency and
prompt-version observability of the Zoe chatbot's multi-agent flow.

## Cloud Setup (free)

1.  **Sign Up:** Go to [cloud.langfuse.com](https://cloud.langfuse.com) (EU) or
    [us.cloud.langfuse.com](https://us.cloud.langfuse.com) (US) and sign up for the free
    **Hobby** plan (~50k units/month, no card).
2.  **Create Project:** e.g. "ThinkLife Zoe -- Dev".
3.  **Get API Keys:** **Settings -> API Keys** -> create new keys.
4.  **(Optional) Configure model pricing:** **Settings -> Models** -- confirm `gpt-4o-mini`,
    `gpt-4o`, `gemini-1.5-flash` exist (Cloud ships these by default). This is what lets
    Langfuse auto-compute $ cost from the token usage every generation span carries.

Self-hosting (fully local, also free) is documented in the official Langfuse repo's
`docker-compose.yml` if you'd rather not use Cloud.

## Environment Variables

Add to `backend/.env` (see `backend/.env.example` for the full template):

```bash
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://us.cloud.langfuse.com   # or http://localhost:3000 if self-hosted
LANGFUSE_TRACING_ENABLED=true
LANGFUSE_RELEASE=zoe-v0.7.0                    # bump per release; groups traces/evals by release
LANGFUSE_SAMPLE_RATE=1.0                       # 1.0 in dev; lower in production if needed
```

Tracing is **always-on at startup** (`backend/main.py` lifespan calls
`initialize_langfuse_client()` unconditionally) -- it is no longer gated behind any
per-request flag. With `LANGFUSE_TRACING_ENABLED` unset/false, or with missing/invalid
keys, tracing simply stays off; every `@observe`/`update_current_*`/`score_*` call is a
safe no-op and the request path is unaffected either way (see "Fail-open design" below).

## Backend Integration

The Langfuse client is centrally managed in `backend/evaluation/evaluation_manager.py`,
which re-exports everything other modules need:

```python
from evaluation import (
    observe,                 # @observe(name=..., as_type=...) decorator
    safe_propagate_attributes,  # context manager: sets user_id/session_id/tags on a trace
    start_span,               # context manager: ad-hoc child span (e.g. per-tool spans)
    update_current_span,      # update the active span's output/metadata/level
    update_current_generation,  # update the active generation's model/usage/cost/metadata
    score_current_trace,      # attach a numeric/categorical score to the current trace
    is_langfuse_enabled, initialize_langfuse_client, flush_langfuse,
)
```

### Trace tree

Every `POST /api/zoe/chat` call produces **one trace**, rooted at `zoe_chat`
(`backend/agents/zoe/zoe_service.py`), with `user_id`/`session_id`/`tags` propagated to
every nested span via `safe_propagate_attributes`:

```
zoe_chat                                  (root; user_id, session_id, tags, output)
└── cortex_process_agent_request
    └── cortex_execute_plan (or _create_reasoned_plan/_create_adaptive_plan -> reasoning_optimize_specs)
        └── workflow_execute_plan
            └── workflow_execute_graph
                ├── workflow_node_initialize
                ├── workflow_node_query_data_sources
                ├── workflow_node_execute_tools
                │   └── tool.<name>            (per tool, via start_span)
                ├── workflow_node_build_messages
                ├── workflow_node_call_provider      [generation: model, usage, cost, prompt version]
                │   └── workflow_validate_response_quality  [generation: judge call]
                ├── workflow_node_validate_response
                └── workflow_node_finalize
```

The `build_messages -> call_provider -> validate_response` sequence repeats up to 5x
(the retry-on-low-confidence loop) within the same trace.

This was verified with an in-memory OTEL span exporter (no live Langfuse account needed)
during development -- a single trace_id, correctly nested, with output/metadata attached
at the right spans.

### Cost tracking

`workflow_node_call_provider`'s generation span carries real `model` + `usage_details`
(`prompt_tokens`/`completion_tokens`/`total_tokens`, normalized from both the OpenAI and
Gemini provider response shapes). Langfuse computes $ cost from this automatically
**once model pricing is configured** (Cloud ships it by default; self-host needs it added
under Settings -> Models).

There's also a **local cost mirror** for `/api/brain/analytics`, independent of Langfuse:
`CostEstimator.actual_cost(model, usage)` (`backend/evaluation/estimators/cost_estimator.py`)
holds a small hardcoded $/1M-token pricing table for the models actually in use here. It
returns `None` (not `$0`) for any model not in that table, so "unknown cost" is never
silently reported as free. `BrainAnalytics` aggregates this into
`total_cost_usd`/`average_cost_usd`/`priced_requests`.

### Latency

- Span durations are automatic (every `@observe`'d node has a start/end time).
- `workflow_engine.py` additionally tracks a `stage_durations` breakdown (summed across
  retries) for `retrieval` / `llm_generation` / `validation`, finalized into
  `state["latency_ms"]` in `_finalize_node` and threaded up into the trace's root-span
  metadata.
- A `latency_seconds` numeric score is emitted on every trace via `score_current_trace`.
- The existing `LatencyEvaluator` (`backend/evaluation/evaluators/latency_evaluator.py`)
  is now fed real request latency on every completed request (see
  `CortexFlow._record_latency_evaluation`), so its rolling p50-style stats reflect
  production traffic instead of staying empty.

### Prompt versioning

Zoe's system prompt is no longer an inline string. It's versioned under
`backend/agents/zoe/prompts/` as `zoe_system.vN.md` (+ `zoe_system_trauma_addendum.vN.md`
for high-ACE-score users), resolved at runtime by `agents/zoe/prompts/registry.py`
(`get_prompt(name)` -- picks the highest `.vN.md` file present, returns
`{name, version, sha256, text}`). Bumping the prompt is: add a new `.vN.md` file, commit
it (git *is* the changelog), redeploy.

`ZoeCore.build_system_prompt_versioned()` returns both the compiled text and the version
metadata; `plugins/zoe_agent.py` threads the version metadata through
`agent_metadata.messages_context.system_prompt_version`, and
`workflow_node_call_provider` attaches it to the generation span's
`metadata.prompt = {name, version, sha256, addendum_version}`. If the prompt files are
ever missing at runtime, `ZoeCore` falls back to a short hardcoded prompt rather than
failing the request.

### Fail-open design

A Langfuse outage, missing credentials, or malformed attribute must never break a chat
response. This is enforced two ways:

1. **SDK-level:** exactly one `Langfuse` client is registered per process (even when
   tracing is off) with `tracing_enabled` baked in at construction time. This matters
   because the SDK's own `get_client()` (used internally by `@observe`) reuses whatever
   client is already registered, or **lazily constructs its own from raw env vars** on
   first use if none exists -- registering one up front avoids ending up with a second,
   unconfigured client that silently attempts exports and fails.
2. **Call-site level:** `update_current_span`/`update_current_generation`/
   `score_current_trace`/`start_span`/`safe_propagate_attributes` all wrap the real SDK
   calls in try/except and no-op on failure, so a Langfuse-side error never propagates
   into the request path.

## GET /api/brain/analytics

`CortexFlow.get_analytics()` (previously missing -- this endpoint 500'd) now returns:
request volume/success/error rate, average response time, provider/application usage,
`total_cost_usd`/`average_cost_usd`/`priced_requests`, and `latency_statistics` (the
`LatencyEvaluator` rolling stats). `GET /api/brain/health` and `POST /api/brain` were
fixed the same way (`get_health_status()`, `process_request()`).

## Monitoring dashboard

A "Zoe Observability (Epic 4.1)" dashboard is configured in the Langfuse project via the
unstable Dashboards/Widgets API (`unstable-dashboard-widgets`, `unstable-dashboards` --
see the `langfuse-cli` skill), with 5 widgets:

- **LLM Cost by Model (USD)** -- `observations` view, `sum(totalCost)` grouped by
  `providedModelName`, filtered to `type=GENERATION`.
- **Zoe Chat Request Volume** -- `observations` view, `count`, filtered to `name=zoe_chat`.
- **Latency per Agent Node (p95)** -- `observations` view, `p95(latency)` grouped by
  span `name` -- breaks down which node (retrieval/LLM call/validation) drives latency.
- **End-to-End Latency (avg, s)** -- `scores-numeric` view, `avg(value)`, filtered to the
  `latency_seconds` score.
- **Response Confidence (avg)** -- `scores-numeric` view, `avg(value)`, filtered to the
  `response_confidence` score (see below). Acceptance threshold is 0.75.

`response_confidence` is emitted via `score_current_trace()` in
`WorkflowEngine._finalize_node` (`backend/brain/cortex/workflow_engine.py`) -- the same
LLM-judge confidence already computed in `_validate_response_quality`, previously only
kept in-memory/metadata and not persisted as a trace score.

Dashboard/widget IDs and creation commands aren't checked in (they're project-specific
Langfuse state, not code) -- recreate via the CLI commands above if you provision a new
Langfuse project.

## Not yet built

- Alert rules on the above panels (e.g. notify if `response_confidence` 7-day mean drops,
  or cost/latency spikes) -- Langfuse alerting is UI/plan-tier dependent and not yet set up.
- A local `/observability` frontend page (optional per the plan) -- not built yet.
- Phase 2 (offline evaluation) is in progress as a separate body of work: the accuracy
  and relevance metrics and a 39-item draft dataset are built (see
  [EVAL_METRICS.md](EVAL_METRICS.md) and [HALLUCINATION_TRACKING.md](HALLUCINATION_TRACKING.md));
  dataset review and the release-gate `compare.py` are not built yet.

## Troubleshooting

*   **"Langfuse not available":** Ensure a supported `langfuse` is installed
    (`pip install 'langfuse>=3.11.0,<5'`). Older versions lack `propagate_attributes`, which
    makes the import fail and silently turns all tracing into no-ops.
*   **No traces appearing:** Check `LANGFUSE_TRACING_ENABLED=true` and your keys in `.env`.
    Startup logs one clear line either way ("Langfuse tracing enabled" / "...disabled"
    / "...requested but ... are not set").
*   **`Failed to export span batch: 401`:** credentials are present but wrong -- traces
    are still attempted (by design, so you get this loud signal) but will keep failing
    until `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY`/`LANGFUSE_HOST` are fixed. The chat
    request itself is unaffected either way (export happens async, off the request path).
