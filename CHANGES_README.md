# Change Summary: Epic 4 (Observability and AI Quality Metrics)

**Branch:** `arvind` (uncommitted working-tree changes on top of `234e54f54`)
**Author:** kavish110
**Date:** 2026-10-22
**Scope:** Zoe chatbot backend. The FAQ bot only shares the offline evaluators.

The plan these changes follow is in [instructions.md](instructions.md). This file lists
**what changed, where it changed, and what each change depends on**.

---

## 1. Summary

| Area | Epic item | What it does now |
|---|---|---|
| Langfuse tracing | 4.1.1 | Every `POST /api/zoe/chat` request produces one trace with nested spans, `user_id`/`session_id`/tags attached. Langfuse starts once at app startup and no longer depends on a per-request `processing.eval` flag. |
| Cost tracking | 4.1.2 | The LLM call span carries the model name and real token usage. The local analytics endpoint also computes a cost in USD. |
| Latency | 4.1.3 | Time is measured for each stage (retrieval, LLM call, validation). Each request gets a `latency_seconds` score. Real request latency feeds `LatencyEvaluator`. |
| Prompt versioning | 4.1.4 | Zoe's system prompt moved out of the code into versioned `.vN.md` files. Each trace records the prompt's name, version and sha256. |
| Quality signal | 4.1 dashboard | Each trace stores the validator's confidence as a `response_confidence` score. |
| Broken Brain endpoints fixed | n/a | `POST /api/brain`, `GET /api/brain/health` and `GET /api/brain/analytics` called Cortex methods that didn't exist, so every call failed with an error. Those methods now exist. |
| Secret masking | 4.1 | API keys and tokens are replaced with `[REDACTED]` before any data is sent to Langfuse. |
| Offline evaluation | 4.2.2–4.2.4 | New `AccuracyEvaluator`. The relevance evaluator now scores 4 separate parts. The judge model can be OpenAI, Gemini or Ollama, so evaluations can run for free. Results are broken down by category. There is a 39-case Zoe draft dataset. |

---

## 2. Modified files

### 2.1 Observability core: `backend/evaluation/`

| File | Changes |
|---|---|
| [backend/evaluation/evaluation_manager.py](backend/evaluation/evaluation_manager.py) | **Main tracing module.** <br>• `initialize_langfuse_client()` always creates exactly one client for the whole process, even when tracing is off. Without it, the SDK would quietly create a second, unconfigured client. Tracing is on only when `LANGFUSE_TRACING_ENABLED=true` **and** both keys are set. `auth_check()` only logs a result; it doesn't turn tracing off. <br>• New `_tracing_active` flag. `is_langfuse_enabled()` now returns this flag instead of checking whether a client exists. <br>• The client now receives `release`, `sample_rate`, `tracing_enabled` and `mask=mask_secrets`. <br>• New `mask_secrets()`: a regex and a list of key names that redact `sk-…`, `pk-lf-…`, `sk-lf-…`, `AIza…` and `Bearer …` values, plus keys such as `api_key`/`password`/`token`. It **does not** redact user message text or ACE context; that is still an open product/compliance decision. <br>• New helpers that can never break a request: `update_current_span`, `update_current_generation`, `score_current_trace`, `start_span` (context manager) and `safe_propagate_attributes` (context manager). If Langfuse fails or is missing, they do nothing. <br>• Imports `propagate_attributes` from `langfuse`, with a do-nothing fallback when Langfuse isn't installed. |
| [backend/evaluation/\_\_init\_\_.py](backend/evaluation/__init__.py) | Re-exports `observe`, `propagate_attributes`, `safe_propagate_attributes`, `start_span`, `update_current_span`, `update_current_generation` and `score_current_trace`. |
| [backend/evaluation/estimators/cost_estimator.py](backend/evaluation/estimators/cost_estimator.py) | New price table `_MODEL_PRICING_PER_1M_TOKENS` (gpt-4o-mini, gpt-4o, gemini-1.5-flash, gemini-2.0-flash). New `CostEstimator.actual_cost(model, usage)` returns the real cost in USD, or `None` (not `0.0`) when the model isn't in the table. This table is used **only** for the local `/api/brain/analytics` totals. The Langfuse dashboard uses Langfuse's own model prices. |

### 2.2 Request path: `backend/brain/`, `backend/agents/`, `backend/plugins/`

| File | Changes |
|---|---|
| [backend/main.py](backend/main.py) | At startup (`lifespan`), always calls `initialize_langfuse_client()` and logs one line saying whether tracing is on or off. At shutdown, calls `flush_langfuse()`. |
| [backend/agents/zoe/zoe_service.py](backend/agents/zoe/zoe_service.py) | `process_message` is now the **root span**: `@observe(name="zoe_chat", as_type="agent", capture_input=False, capture_output=False)`. Its body moved into `_process_message_traced()` and runs inside `safe_propagate_attributes(user_id, session_id, tags=["agent:zoe","app:<app>"], trace_name="zoe.chat")`. The trace input is set to only the user's message. Every return branch (redirect, low confidence, success, LLM failure, exception) records an output, metadata and a level. The success and low-confidence branches also send a `latency_seconds` score. |
| [backend/agents/zoe/zoe_core.py](backend/agents/zoe/zoe_core.py) | The hardcoded system prompt was removed. New `build_system_prompt_versioned(context)` loads `zoe_system` (and `zoe_system_trauma_addendum` when `high_trauma_risk` is set) through `get_prompt()`, and returns `(text, {name, version, sha256, addendum_version})`. `build_system_prompt()` still exists and calls the new method. If the prompt files are missing, a short built-in prompt (`_FALLBACK_BASE_PROMPT`) is used instead. |
| [backend/plugins/zoe_agent.py](backend/plugins/zoe_agent.py) | Calls `build_system_prompt_versioned()` and puts `system_prompt_version` into `messages_context`. The workflow engine later reads it and attaches it to the LLM call span. |
| [backend/brain/cortex/cortex.py](backend/brain/cortex/cortex.py) | • Removed the per-request Langfuse setup that depended on `processing.eval`. A single call to `initialize_langfuse_client()` remains as a safety net for scripts and tests that skip the app's startup code. <br>• New `_record_analytics()`: after each request, updates the running request count, success/error rate, average response time, usage per app and per provider, and cost totals. <br>• New `_record_latency_evaluation()`: feeds real request latency into `LatencyEvaluator`. <br>• **New `process_request(data)`**: handles `POST /api/brain` and its `/api/<app>` aliases. It builds a default direct-execution OpenAI spec from the config. <br>• **New `get_health_status()`** and **`get_analytics()`**, used by `GET /api/brain/health` and `GET /api/brain/analytics`. <br>• Imports `ProviderSpec`, `ProcessingSpec`, `ApplicationType`, `UserContext` and `get_latency_statistics`. |
| [backend/brain/cortex/workflow_engine.py](backend/brain/cortex/workflow_engine.py) | • New `WorkflowState` fields: `stage_durations`, `latency_ms` and `total_cost_usd`. Their values are returned as `metadata.latency_ms` and `metadata.actual_cost_usd`. <br>• `_query_data_sources_node`: marked as a `retriever` span. Records the query text and results, and times the `retrieval` stage. <br>• `_execute_tools_node`: one `tool.<name>` span per tool (`start_span`). <br>• `_call_provider_node`: auto-capture is turned off; otherwise it would log the entire `WorkflowState` and overwrite the clean input/output. Calls `update_current_generation(model, input, output, usage_details, cost_details, metadata{provider, finish_reason, temperature, max_tokens, prompt version})`. Adds up the cost across retries and times the `llm_generation` stage. <br>• `_validate_response_node`: times the `validation` stage. <br>• `_validate_response_quality`: marked as an `evaluator` span. The judge LLM call is wrapped in its own `judge_call` generation span so its token usage is recorded too. <br>• `_finalize_node`: calls `score_current_trace("response_confidence")` and builds the `latency_ms` breakdown (`total`, `retrieval`, `llm_generation`, `validation`). <br>• New helper `_extract_usage_details()` reads `prompt_tokens`, `completion_tokens` and `total_tokens` from the provider's metadata. |
| [backend/brain/cortex/reasoning_engine.py](backend/brain/cortex/reasoning_engine.py) | Added `@observe(name="reasoning_optimize_specs")` to `optimize_execution_specs`. |
| [backend/brain/specs/core_specs.py](backend/brain/specs/core_specs.py) | `BrainAnalytics` gains `total_cost_usd`, `average_cost_usd` and `priced_requests`. |

### 2.3 Offline evaluation: `backend/evaluators/`

| File | Changes |
|---|---|
| [backend/evaluators/base.py](backend/evaluators/base.py) | `LLMJudgeEvaluator` can now use different judge providers: `provider=` / `EVAL_JUDGE_PROVIDER` = `openai` \| `gemini` \| `ollama`, and `model=` / `EVAL_JUDGE_MODEL`. Defaults per provider: `gpt-4o-mini`, `gemini-1.5-flash`, `llama3.1:8b`. New `_call_openai`, `_call_gemini` and `_call_ollama`; each SDK is imported only when its provider is used. A failed judge call is retried 3 times, waiting 2 s, then 4 s. If all 3 fail, the evaluator returns a score of `0.0` with label `"error"` instead of crashing. The old `client` property was **removed**. |
| [backend/evaluators/common_evaluators/instruction_following_relevance_evaluator.py](backend/evaluators/common_evaluators/instruction_following_relevance_evaluator.py) | Now scores 4 parts and combines them with weights: `0.4·directness + 0.3·completeness + 0.2·focus + 0.1·context_fit`. The pass threshold went from **0.7 to 0.75**. The judge prompt now includes the last 6 conversation turns and the user context. The 4 part scores appear in `metadata.sub_scores`. |
| [backend/evaluators/common_evaluators/\_\_init\_\_.py](backend/evaluators/common_evaluators/__init__.py) | Exports `AccuracyEvaluator`. |
| [backend/evaluators/runner.py](backend/evaluators/runner.py) | • The evaluator lists are now built by functions (`_common_evaluators()`, `_faq_only_evaluators()`) instead of once when the module loads, so `--judge` takes effect. `AccuracyEvaluator` was added to the common set. <br>• New command-line options `--judge {openai,gemini,ollama}`, `--judge-model` and `--release`. <br>• `check_api_key()` was replaced by `check_judge_credentials(provider)`. <br>• The output JSON gains `meta.release`, `meta.judge_provider`, `meta.judge_model`, a `by_category` breakdown per evaluator, and a `category` field on each case. |

### 2.4 Docs

| File | Changes |
|---|---|
| [backend/docs/LANGFUSE_SETUP.md](backend/docs/LANGFUSE_SETUP.md) | Rewritten: environment variables, trace tree, cost/latency/prompt-version design, the fail-safe design, `/api/brain/analytics`, dashboard widgets, what's not built yet (Phase 2 is now described as in progress, not "not started"), and troubleshooting (now includes how to fix an unsupported Langfuse version). |
| [backend/requirements.txt](backend/requirements.txt) | `langfuse>=3.7.0` changed to `langfuse>=3.11.0,<5` (see section 4.1). |

---

## 3. New files

| File | Purpose |
|---|---|
| [backend/agents/zoe/prompts/registry.py](backend/agents/zoe/prompts/registry.py) | `get_prompt(name)` finds the highest-numbered `name.vN.md` file in this folder and returns `{name, version, sha256, text}`. It caches results with `lru_cache`, so **a restart is needed to pick up a new version**. To change a prompt, add a new `.vN+1.md` file and commit it. |
| [backend/agents/zoe/prompts/\_\_init\_\_.py](backend/agents/zoe/prompts/__init__.py) | Exports `get_prompt` and `PromptVersion`. |
| [backend/agents/zoe/prompts/zoe_system.v1.md](backend/agents/zoe/prompts/zoe_system.v1.md) | Zoe's base system prompt, moved word for word from `zoe_core.py`. |
| [backend/agents/zoe/prompts/zoe_system_trauma_addendum.v1.md](backend/agents/zoe/prompts/zoe_system_trauma_addendum.v1.md) | The extra prompt text for users with a high ACE score, moved word for word from `zoe_core.py`. |
| [backend/evaluators/common_evaluators/accuracy_evaluator.py](backend/evaluators/common_evaluators/accuracy_evaluator.py) | `AccuracyEvaluator` (Epic 4.2.2) has two modes. **Ground-truth mode** is used when the case has `expected_answer` or `metadata.expected_points`; it passes at 0.8 or higher. **Rubric mode** is used for open-ended support prompts; it scores validity, safety, made-up facts and honesty, and passes at 0.7 or higher. Two fixed checks run regardless of the judge: matching any `must_not_contain` string means automatic failure (score 0.0), and a missing `must_contain` string limits the score to at most 0.6. |
| [backend/evaluators/datasets/zoe_eval_v1.draft.json](backend/evaluators/datasets/zoe_eval_v1.draft.json) | Draft dataset of **39 Zoe cases** across 14 categories: emotional support, crisis/safety, off-topic redirect, minor protection, high-ACE, multi-turn, hallucination probes, hard negatives, and others. 11 cases have an `expected_answer`. The metadata includes `must_contain`, `must_not_contain`, `review_status` and calibration fields. **Still a draft and not yet reviewed.** |
| [backend/docs/EVAL_METRICS.md](backend/docs/EVAL_METRICS.md) | Definitions of the accuracy and relevance metrics, how to set up the judge, and the planned release gate (`compare.py`, not built). It refers to the dataset as the 39-item draft throughout. |
| [backend/docs/HALLUCINATION_TRACKING.md](backend/docs/HALLUCINATION_TRACKING.md) | Hallucination tracking (Epic 4.2.4): ThinkLife's definition, how it's detected, the metric, and the planned grounded mode, production tracking and review process. |
| [backend/.env.example](backend/.env.example) | Template for the backend `.env` file: OpenAI, Chroma, server, CORS, Langfuse (`LANGFUSE_*`) and evaluation judge (`EVAL_JUDGE_*`, `GEMINI_API_KEY`, `OLLAMA_HOST`) settings. |
| [instructions.md](instructions.md) | Implementation plan for Epic 4 (Phase 1 tracing and Phase 2 evaluation). |
| `.claude/skills/langfuse/` | Reference files for the Langfuse agent skill, used during development. These are tooling only; the app doesn't use them at runtime. |

---

## 4. Dependencies

### 4.1 Python packages ([backend/requirements.txt](backend/requirements.txt))

No new packages are needed. The only change is a narrower allowed version range for `langfuse`.

| Package | Used by | Required? |
|---|---|---|
| `langfuse>=3.11.0,<5` (was `>=3.7.0`) | `evaluation_manager.py`: `Langfuse`, `observe`, `propagate_attributes`, `start_as_current_observation`, `update_current_span/generation`, `score_current_trace`, the `mask` parameter | Optional at runtime. If the import fails, everything falls back to doing nothing. **See the version note below.** |
| `openai` | OpenAI provider and the `openai` judge | Required for chat. Needed for evaluation only with `--judge openai`. |
| `google-genai` | `_call_gemini` judge (`google.genai.Client`, `GenerateContentConfig`) and the Gemini provider | Only with `--judge gemini` |
| `httpx` | `_call_ollama` judge | Only with `--judge ollama` |
| `langgraph` | `WorkflowEngine` (unchanged) | Required |
| `fastapi`, `python-dotenv` | `main.py`, `evaluation_manager.py` | Required |

> **Why the version range is `>=3.11.0,<5`:** these limits come from searching the published
> Langfuse packages for the functions and arguments the code uses.
> - Below 3.9.0, `propagate_attributes` doesn't exist. The shared import in `evaluation_manager.py`
>   fails, and **all tracing turns off** (it now logs a warning when this happens).
> - 3.9.x and 3.10.x have `propagate_attributes` but not its `trace_name` argument, which
>   `zoe_service.py` passes. `safe_propagate_attributes` catches the resulting error, so
>   `user_id` and `session_id` are **quietly left off traces**.
> - 3.11.0 is the first version with everything needed.
> - `<5` excludes the 5.x major release, which hasn't been tested. This code was tested with
>   4.14.4 in the local `.venv`.

### 4.2 External APIs and services

| Service | Called from | When |
|---|---|---|
| **Langfuse** (Cloud `https://us.cloud.langfuse.com` by default, or self-hosted) | Langfuse SDK, sending data in the background (OTEL span export) and the `auth_check()` call at startup | Only when `LANGFUSE_TRACING_ENABLED=true` and both keys are set. Sending happens in the background, so a Langfuse outage never blocks a chat request. |
| Langfuse **model price settings** (Settings → Models) | Calculated by Langfuse from `usage_details` | Needed for the cost numbers on the Langfuse dashboard |
| Langfuse **dashboard** "Zoe Observability (Epic 4.1)" | Created through the CLI and the unstable Dashboards/Widgets API | Exists only in the Langfuse project and **isn't checked into the repo**. It must be recreated for a new project (see `LANGFUSE_SETUP.md`). |
| **OpenAI API** | Chat provider, the response validator in the workflow, and the `openai` judge | Chat requests, and evaluation with `--judge openai` |
| **Google Gemini API** | `gemini` judge | `--judge gemini` (free tier) |
| **Ollama** (local, `http://localhost:11434`, `POST /api/chat`) | `ollama` judge | `--judge ollama` (needs `ollama serve` running and the model downloaded) |

### 4.3 Environment variables

| Variable | Default | Read in |
|---|---|---|
| `LANGFUSE_TRACING_ENABLED` | `false` | `evaluation_manager.py`: must be `true` for tracing to run |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | none | `evaluation_manager.py` |
| `LANGFUSE_HOST` | `https://us.cloud.langfuse.com` | `evaluation_manager.py` |
| `LANGFUSE_RELEASE` | none | `evaluation_manager.py`: groups traces by release |
| `LANGFUSE_SAMPLE_RATE` | `1.0` | `evaluation_manager.py` |
| `EVAL_JUDGE_PROVIDER` | `openai` | `evaluators/base.py`, `runner.py` (overridden by `--judge`) |
| `EVAL_JUDGE_MODEL` | provider default | `evaluators/base.py`, `runner.py` (overridden by `--judge-model`) |
| `OPENAI_API_KEY` | none | Providers, OpenAI judge |
| `GEMINI_API_KEY` | none | Gemini judge |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama judge |

All of these are listed in [backend/.env.example](backend/.env.example). Tracing is off there
by default (`LANGFUSE_TRACING_ENABLED=false`).

### 4.4 What the changed code expects from the rest of the codebase

- **Provider response shape:** `workflow_engine.py` expects `provider.generate_response()` to return
  `{"success", "content", "metadata": {"model", "provider", "finish_reason", "usage": {"prompt_tokens", "completion_tokens", "total_tokens"}}}`.
  OpenAI and Gemini already return `metadata.usage`. If a provider leaves out `usage` or `model`, no
  cost is recorded (`actual_cost_usd` is `None`); nothing breaks.
- **Cost table:** `CostEstimator._MODEL_PRICING_PER_1M_TOKENS` must contain the exact model name the
  provider reports, or that request isn't counted in the local cost totals.
- **Prompt version flow:** `ZoeAgent` → `specs.agent_metadata["messages_context"]["system_prompt_version"]`
  → `_call_provider_node`. If an agent doesn't set this key, the trace's `prompt` metadata is `None`.
- **Trace nesting:** `ZoeService.process_message` must stay the outermost function marked with
  `@observe`. `cortex.py` relies on it no longer setting up Langfuse for each request.
- **Prompt files** must be shipped with the backend (`agents/zoe/prompts/*.vN.md`). Otherwise Zoe
  falls back to `_FALLBACK_BASE_PROMPT` and logs an error.
- **Evaluator subclasses:** the constructor is now `LLMJudgeEvaluator(pass_threshold, provider=None, model=None)`.
  All current subclasses pass `pass_threshold` by keyword, so they still work. **Any code that used
  `evaluator.client` will break**, because that property was removed.
- **Dataset fields used by the evaluators:** `metadata.category` (for the `by_category` breakdown), and
  `metadata.expected_points`, `metadata.must_contain` and `metadata.must_not_contain` (used by `AccuracyEvaluator`).

---

## 5. API endpoints affected

| Endpoint | Change |
|---|---|
| `POST /api/zoe/chat` | Same behavior and response. It now produces one Langfuse trace per request when tracing is on. |
| `POST /api/brain`, `POST /api/{app_name}` and the app aliases | **Fixed.** Previously failed because `CortexFlow.process_request` didn't exist. Returns `{success, message, data{execution_id, confidence}, error, metadata, timestamp}`. |
| `GET /api/brain/health` | **Fixed.** Returns `{overall, initialized, components{…}, langfuse_tracing_active, active_executions, uptime_seconds}`. |
| `GET /api/brain/analytics` | **Fixed.** Returns `BrainAnalytics` fields, including the new cost fields, plus `uptime_seconds` and `latency_statistics`. |

Zoe's response body now also includes `metadata.latency_ms` and `metadata.actual_cost_usd`, which
come from the workflow engine.

---

## 6. How to verify

```bash
# 1. Tracing (needs Langfuse keys in backend/.env, plus LANGFUSE_TRACING_ENABLED=true)
cd backend && uvicorn main:app --reload
#    The startup log should say "Langfuse tracing enabled".
#    Send a message to POST /api/zoe/chat, then find the trace "zoe.chat" in Langfuse.

# 2. Fixed endpoints
curl localhost:8000/api/brain/health
curl localhost:8000/api/brain/analytics

# 3. Offline evaluation on a free judge
python -m evaluators.runner --bot zoe \
  --input evaluators/datasets/zoe_eval_v1.draft.json \
  --judge gemini --release zoe-v0.7.0 --output eval_results.json
```

## 7. Not done yet

- `evaluators/compare.py` release gate and `evaluators/reports/baseline.json`.
- Review of the dataset (`zoe_eval_v1.draft.json` is still a draft).
- Langfuse alert rules, and an optional frontend `/observability` page.
- Redacting user message content and ACE context before it is sent to Langfuse (a product/compliance decision).
- Automated tests for the new tracing helpers, `CostEstimator.actual_cost`, the prompt registry and `AccuracyEvaluator`.
