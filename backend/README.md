# ThinkLife Brain

The central AI orchestration system for the ThinkLife platform, featuring a plugin-based architecture and trauma-informed design.

## Overview

The ThinkLife Brain is a generalized AI system that manages all AI operations across the platform. It provides:

- Plugin-based agent system with automatic discovery
- Trauma-informed safety built into interactions
- LangGraph workflow engine for standardized execution
- MCP integration for data source abstraction
- Backward compatibility with existing integrations

## Structure

The system is organized into the following components:

- **brain/**: Core orchestration system
  - **cortex/**: Main orchestration logic (Cortex, Reasoning, Workflow)
  - **data_sources/**: Data abstraction layer
  - **guardrails/**: Security and session management
  - **providers/**: LLM provider implementations (OpenAI, Gemini, Anthropic)
  - **specs/**: Data structures and type definitions
  - **tools/**: Tool definitions and registry
- **agents/**: Domain-specific agent logic (e.g., Zoe)
- **evaluation/**: Estimators and evaluators for performance and quality
- **middleware/**: Authentication and request handling
- **plugins/**: Connector layer between agents and the Brain

## Quick Start

### Basic Usage

```python
from brain import CortexFlow

# Initialize with configuration
brain_config = {
    "providers": {
        "openai": {
            "enabled": True,
            "api_key": "your-api-key",
            "model": "gpt-4o-mini"
        }
    }
}

brain = CortexFlow(brain_config)

# Process requests
request = {
    "message": "Hello, I need support",
    "application": "healing-rooms",
    "user_context": {"user_id": "user123", "ace_score": 2}
}

response = await brain.process_request(request)
```

### Health Monitoring

```python
# Check system health
health = await brain.get_health_status()
print(f"System status: {health['overall']}")

# Get analytics
analytics = await brain.get_analytics()
print(f"Total requests: {analytics['total_requests']}")
```

## Plugin System

To create a new agent:

1. Create your agent logic in `agents/your_agent/` folder.
2. Create a plugin connector in `backend/plugins/your_agent_plugin.py` implementing `IAgent`.
3. Register it in `plugins/registry.py`'s `register_builtin_plugins()` so it's
   discoverable via `get_plugin_registry()` without editing `main.py` at every
   call site. `main.py`'s `lifespan` calls `register_builtin_plugins()` on
   startup.

See `docs/CREATING_AGENTS.md` for detailed instructions, and
`plugins/echo_agent.py` for a minimal reference implementation that exercises
the full `IAgent` contract without external dependencies.

## Configuration

### Environment & Settings

All configuration is centralized in `config/settings.py` (a `pydantic-settings`
`Settings` class) instead of scattered `os.getenv()` calls. It reads from
environment variables and an optional `.env` file in `backend/`:

```python
from config import settings

settings.openai_api_key
settings.confidence_threshold      # default 0.75
settings.max_validation_attempts   # default 5
settings.cors_allow_origins
```

Key environment variables it reads:

| Variable | Default | Used for |
|---|---|---|
| `LLM_PROVIDER_TYPE` | `openai` | Provider used by the built-in agents: `openai`, `anthropic`, `gemini`, `openrouter` |
| `LLM_PROVIDER_MODEL` | provider default | Model for that provider (must be in `provider_registry.py`; openai falls back to `OPENAI_MODEL`) |
| `EMBEDDING_PROVIDER` | `auto` | Knowledge-base embeddings: `auto` (openai if key set, else local MiniLM), `local`, `openai`, `openrouter`, `gemini` |
| `EMBEDDING_MODEL` | provider default | Embedding model; index must be rebuilt after changing it |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GEMINI_API_KEY` / `OPENROUTER_API_KEY` / `TAVILY_API_KEY` | — | Provider & tool credentials |
| `OPENAI_MODEL` | `gpt-4o-mini` | Default OpenAI model |
| `OPENAI_MAX_TOKENS` / `OPENAI_TEMPERATURE` | `2000` / `0.7` | Default OpenAI generation params |
| `ZOE_VECTOR_DB_PATH` | `backend/data/zoe.sqlite3` | Zoe's vector store location |
| `KEYCLOAK_URL` / `KEYCLOAK_REALM` / `KEYCLOAK_CLIENT_ID` | see `settings.py` | Auth |
| `ENVIRONMENT` | `development` | Set to `production` to make missing required config abort startup instead of only warning (see `main.py`'s `lifespan`) |

Note: some offline/tooling scripts outside the live request path
(`evaluation/`, `evaluators/`, `agents/zoe/utils/knowledge_index_builder.py`)
still read their own environment variables directly and were left as-is,
since they run independently of the FastAPI app.

### Switching Providers

Set `LLM_PROVIDER_TYPE` (and optionally `LLM_PROVIDER_MODEL`) plus the matching API key, e.g.
`LLM_PROVIDER_TYPE=openrouter`, `LLM_PROVIDER_MODEL=anthropic/claude-3.5-sonnet`, `OPENROUTER_API_KEY=...`.
`ZoeAgent`/`EchoAgent` read `settings.default_llm()` when building their `ProviderSpec`, and
`validate_required_for_startup()` warns (aborts in production) on an unknown provider, a model
not in `PROVIDER_CONFIGS`, or a missing key. OpenRouter is OpenAI-compatible
(`brain/providers/openrouter.py` subclasses `OpenAIProvider`). Embeddings are chosen separately by
`EMBEDDING_PROVIDER` (`brain/data_sources/embeddings.py`; rebuild the index after changing it). TTS and the evaluators
remain OpenAI-only; TTS switches itself off without `OPENAI_API_KEY`. To add a provider: adapter in `brain/providers/`, entry in `PROVIDER_CONFIGS`,
branch in `create_provider`, key in `Settings.provider_api_key`.

### Provider Configuration

```python
{
    "providers": {
        "openai": {
            "enabled": True,
            "api_key": "your-key",
            "model": "gpt-4o-mini",
            "max_tokens": 2000,
            "temperature": 0.7
        }
    }
}
```

### Security Configuration

Security features are automatically enabled:
- Rate limiting: 60 requests per minute per user
- Content filtering: Trauma-safe mode enabled
- Input sanitization: XSS and injection protection

### Middleware Chain & Observability

Every request passes through this middleware chain, outermost to innermost:

```
CORS -> RequestLoggingMiddleware -> PayloadValidationMiddleware -> KeycloakAuthMiddleware -> route
```

(see `main.py`'s `add_middleware` calls - Starlette treats the *last*-added
middleware as outermost, so they're registered in the reverse of that order.)

- **`RequestLoggingMiddleware`** (`middleware/logging_middleware.py`) tags
  every request with a correlation ID (`X-Correlation-ID` request/response
  header), and propagates it into *every* logger in the app for that
  request's duration via a `ContextVar` + `logging.Filter` - so a single
  request can be traced end-to-end across Cortex/WorkflowEngine/agent logs,
  not just the request/response line itself.
  It also calls `configure_logging()`, which fixes a pre-existing issue:
  this app never called `logging.basicConfig()` anywhere, so every
  `logger.info(...)` call in the codebase was silently dropped (Python's
  root logger defaults to WARNING with no handler). Logs are now visible.
- **`PayloadValidationMiddleware`** (`middleware/validation_middleware.py`)
  rejects oversized (`> settings.max_request_body_bytes`, default 1MB) or
  malformed-JSON request bodies with a 413/400 before they reach any route.
  Field-level validation (message required, non-blank, under 10,000 chars)
  is handled by the `ChatMessageRequest` Pydantic model shared by
  `/api/zoe/chat`, `/api/chat`, and `/api/{app_name}`, which replaced the
  `Dict[str, Any]` bodies those routes used to validate by hand.

### Resource Lifecycle (Providers, Tools & Data Sources)

`brain/lifecycle.py` provides async context managers used by `WorkflowEngine`:

```python
from brain.lifecycle import managed_provider, managed_data_source, run_tool_with_timeout

async with managed_provider("openai", config) as provider:
    response = await provider.generate_response(messages=messages)
# provider.close() is guaranteed here, even if initialize() returned False
# or generate_response() raised.

result = await run_tool_with_timeout(tool_instance, timeout_seconds=15, query="...")
# tool.initialize()/close() are guaranteed to pair up, and a slow tool call
# is cancelled (with cleanup still running) rather than hanging a request.

async with managed_data_source(vector_source, init_config) as source:
    results = await source.query(query="...")
# Same guarantee as managed_provider, for IDataSource implementations.
```

`WorkflowEngine._execute_tools_node` now actually instantiates and runs
tools through this path (it was previously a placeholder that never called a
real tool - see the module docstring in `brain/lifecycle.py` for details).

### Concurrency

- **Independent work runs in parallel.** `_execute_tools_node` and
  `_query_data_sources_node` (`brain/cortex/workflow_engine.py`) run their
  tools/data sources concurrently via `asyncio.gather`, instead of one at a
  time - a request touching multiple tools or data sources no longer pays
  for their latency serially.
- **Blocking SDK calls are offloaded.** `TavilyClient.search()`
  (`brain/tools/tavily_search.py`) and Chroma's `similarity_search()`
  (`brain/data_sources/vector_db.py`) are both synchronous, blocking calls;
  both now run via `asyncio.to_thread` so they don't stall the event loop.
- **Provider calls are bounded.** `WorkflowEngine.provider_semaphore`
  (sized from `settings.max_concurrent_provider_calls`, default 10) caps how
  many LLM provider calls can be in flight across the whole process at once,
  protecting against a burst of concurrent requests overwhelming a
  provider's own rate limits.
- **Shared singleton state is lock-protected.** `CortexFlow` and
  `WorkflowEngine` are both process-wide singletons. `CortexFlow.analytics`
  and `active_executions`, and `WorkflowEngine.executions`, are updated
  under `asyncio.Lock`s (`_analytics_lock`, `_executions_lock`). Two of
  these were previously dead/leaking code, fixed as part of this work:
  `analytics` was declared but never once mutated anywhere (so
  `/api/brain/analytics` always returned all-zero data), and
  `WorkflowEngine.executions` was write-only and grew without bound for the
  life of the process - now capped at `_max_tracked_executions` (500,
  oldest evicted first).
- **The auth chain no longer blocks the event loop.** `SecurityManager.
  validate_user` -> `SessionManager.validate_session` -> `validate_token` ->
  `_get_public_key` (`brain/guardrails/`) is now a fully `async def` chain.
  It previously made a synchronous `requests.get(...)` call to Keycloak on
  the critical path of `KeycloakAuthMiddleware.dispatch` - i.e. on *every*
  authenticated request - which could block the event loop for up to 5s.
  The network call now runs via `asyncio.to_thread`. This also fixed a
  second, unrelated bug found in the process: `_get_public_key` was
  decorated with `@lru_cache(maxsize=1)`, which (being a zero-argument
  instance method) cached its result forever after the first call for a
  given `SessionManager` instance - silently making the manual TTL-based
  cache beneath it (`public_key_cache_expiry`) unreachable dead code. That
  manual cache is what actually governs re-fetching now.

### Resilience (Retry & Circuit Breaker)

`brain/resilience/` (Task 8) wraps every provider created via `create_provider()`
(`brain/providers/__init__.py`) in a `ResilientProvider` that adds retry-with-backoff
and a per-provider-type circuit breaker around `generate_response()` calls -
transparently, with no changes needed in `WorkflowEngine` or `managed_provider`.

This is **distinct** from `WorkflowEngine`'s existing confidence-based
validation retry loop (`settings.max_validation_attempts` /
`confidence_threshold`): that loop retries when a response's *quality* is
too low; `brain/resilience/` retries when a provider call *transport-fails*
(timeout, connection error, or - the key adaptation, since provider adapters
here catch their own exceptions and return `{"success": False, ...}` rather
than raising - a returned failure dict).

```python
from brain.resilience import RetryPolicy, CircuitBreaker, ResilientProvider

# Applied automatically by create_provider(), but can be used standalone:
wrapped = ResilientProvider(some_provider_instance)
response = await wrapped.generate_response(messages=[...])
# Retries up to settings.provider_retry_max_attempts times with exponential
# backoff on transient failures, and short-circuits (fails fast, no call to
# the real provider) once settings.circuit_breaker_failure_threshold
# consecutive failures open the circuit, until
# settings.circuit_breaker_recovery_timeout_seconds elapses.
```

Circuit breaker state is surfaced through the health endpoints (Task 8.4):
`GET /api/brain/health` includes a `circuit_breaker` entry per provider once
that provider has been used at least once (breakers are created lazily on
first real call). The overall health status becomes `degraded` if any
circuit is open.

Tool execution (`WorkflowEngine._execute_one_tool`) also gets retry-with-backoff,
via `brain/resilience/tool_resilience.py`'s `run_tool_with_retry` - the plan's
own wording for this task covers "transient provider **or tool** failures".
Tools do not get a circuit breaker (only providers do): a single flaky tool
call doesn't warrant short-circuiting future calls the way a struggling LLM
provider does. Tool retry uses a smaller budget
(`settings.tool_retry_max_attempts`, default 2) than provider retry, since
each attempt is already wrapped in `run_tool_with_timeout`'s timeout
(Task 6), underneath `WorkflowEngine`'s own 5x confidence-based retry loop -
three retry layers stacking would compound latency badly.

### Background Tasks

`background/task_queue.py` (Task 9) provides `BackgroundTaskQueue`: a bounded
`asyncio.Queue` drained by a small pool of worker tasks, started/stopped in
`main.py`'s `lifespan`. Chosen deliberately over bare FastAPI
`BackgroundTasks`, which has no backpressure control - under a burst of
concurrent requests it would spawn an unbounded number of concurrent
background coroutines, which is exactly the kind of resource-exhaustion risk
Story 5.1 exists to *prevent*. `enqueue()` drops (with a warning, not an
exception) rather than blocking the caller when the queue is full.

```python
from background import get_background_queue

await get_background_queue().enqueue(some_async_func, arg1, arg2, name="my_job")
# Returns almost immediately; some_async_func runs on a worker, with its own
# retry budget (settings.background_task_retry_max_attempts), and a failure
# there is logged and counted, never raised back to the caller.
```

Currently used for: **response evaluation**. `evaluate_response()`
(`evaluation/evaluation_manager.py`) was fully implemented - including a
usage example in its own docstring - but had zero callers anywhere in the
codebase before this. `zoe_chat_endpoint` and `legacy_chat_endpoint`
(`main.py`) now enqueue it after every successful chat response.

**Not backgrounded:** TTS generation in `legacy_chat_endpoint` stays
synchronous. Its output (`audio_data`) is part of that endpoint's response
contract - the client's avatar/TTS UI expects it inline in the same
response - so making it fire-and-forget would mean either breaking that
contract or inventing a new delivery mechanism (polling, webhook, or
WebSocket), which is out of scope for adding a background task layer.

### Streaming (SSE)

`POST /api/zoe/chat/stream` (Task 10) streams a response token-by-token as
Server-Sent Events, implementing the `IStreamingAgent` interface
(`brain/specs/agent_specs.py`) that had existed since the original repo
analysis with zero implementations anywhere. Request body is the same
`ChatMessageRequest` the non-streaming `/api/zoe/chat` uses.

```
data: {"type": "chunk", "content": "Hel"}

data: {"type": "chunk", "content": "lo!"}

data: {"type": "done", "session_id": "..."}
```

(or a single `{"type": "error", "error": "..."}` frame if something failed
- since SSE can't change the HTTP status code after streaming has started,
a mid-stream failure is reported this way rather than as a 5xx).

Implemented with a plain `StreamingResponse(media_type="text/event-stream")`
rather than adding the `sse-starlette` dependency noted as missing in the
original repo analysis - hand-rolled `data: ...\n\n` framing needs no new
dependency for what's a small amount of formatting.

**How a chunk gets from the LLM to the client:**
`OpenAIProvider`/`AnthropicProvider.stream_response()` (Task 10.1, using
each SDK's native streaming) → wrapped by `ResilientProvider.stream_response()`
(Task 8's wrapper - fails fast on an open circuit, but deliberately does
**not** retry mid-stream, since retrying a response a client has partially
seen isn't a sound recovery strategy) → `WorkflowEngine.stream_execution()`
(Task 10.3 - reuses the existing data-source/tool/message-building nodes
unchanged, then streams instead of running the confidence-scoring retry
loop) → `CortexFlow.process_agent_request_streaming()` (same auth/rate-limit
guardrails as the non-streaming path, applied before any token is sent) →
`ZoeAgent.stream_response()` (Task 10.2 - applies `ZoeCore`'s safety
post-processing to the *finalized* transcript as one supplementary chunk
after the model's own output, since that post-processing only ever appends
a closing line, never redacts) → `ZoeService.stream_message()` (mirrors
`process_message`'s safety-redirect check, so a harmful message is still
caught before any LLM call) → the SSE endpoint.

**Providers without streaming support** (only OpenAI and Anthropic got it,
per Task 10.1's scope - not Gemini) degrade gracefully: `stream_execution`
falls back to delivering the full `generate_response()` result as a single
SSE chunk.

**Client disconnect (Task 10.6):** every layer above is (or wraps) an async
generator using the context managers from Task 6 (`managed_provider`, etc.),
so a disconnect mid-stream correctly triggers cleanup - Python delivers
`GeneratorExit` at the suspended `yield` and unwinds through every
`finally`/`async with` layer before the generator is considered closed.

## Documentation

- **[System Architecture](docs/BRAIN_ARCHITECTURE.md)**: Complete system overview
- **[Creating Agents](docs/CREATING_AGENTS.md)**: Guide to build new agents
- **[Workflow Engine](docs/WORKFLOW_ENGINE.md)**: Details on the execution engine
- **[Cortex & Reasoning](docs/CORTEX_REASONING_ARCHITECTURE.md)**: Planning and reasoning logic

## Key Benefits

1. **Modular**: Add new agents without modifying core code.
2. **Trauma-Informed**: Safety features built into the core.
3. **Efficient**: Optimized routing and execution planning.
4. **Scalable**: Architecture supports growth and multiple agents.
5. **Maintainable**: Clear separation of concerns.
