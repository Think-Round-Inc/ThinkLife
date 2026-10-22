"""
Workflow Engine - LangGraph-based orchestrator

US 1.1 (Multi-Agent Orchestration & Workflow Reliability): every agent - Zoe
today, FAQ and Bard once they are Brain plugins - runs this one standard
graph. A WorkflowProfile per agent (brain/cortex/workflow_profiles.py) sets
its limits, fallback providers and safety checks.

    initialize ──┬─> query_data_sources ─┬─> gather_context -> build_messages -> call_provider
                 └─> execute_tools ──────┘          (both in parallel when enabled)
    call_provider  -> check_safety -> validate_response -> finalize
                   ↘ (still failing after retries) fallback_provider -> check_safety
    any unrecoverable failure -> escalate -> finalize

Failure handling (see docs/workflows/README.md for the full table):
  * transient provider errors (timeout, 429, 5xx, connection) are retried with
    exponential backoff inside call_provider
  * a provider still failing after retries goes to fallback_provider, which
    tries each fallback model in the profile once
  * a policy block from check_safety is never retried
  * a weak answer loops back to build_messages with feedback, up to
    ProcessingSpec.max_iterations
  * anything unrecoverable ends in escalate: a safe message, status
    "escalated", and the reason in the response metadata

Checkpointing: the graph is compiled with a checkpointer (SQLite by default,
see workflow_checkpoint.py) and thread_id = execution_id, so an interrupted
run can be resumed with resume_execution().

Compatibility: uses only LangGraph features present in both 0.6 (what
Python 3.9 installs) and 1.x, so it runs on Python 3.9 through 3.13.

Only plain, serialisable data lives in WorkflowState. The live request and
agent specs travel in LangGraph's runtime context (WorkflowContext), which is
not checkpointed.
"""

import asyncio
import logging
import operator
import os
import random
import re
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Annotated, Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from ..specs import WorkflowProfile, WorkflowStatus
from .workflow_checkpoint import create_checkpointer, keep_completed_checkpoints
from .workflow_errors import (
    PolicyBlockedError,
    ProviderUnavailableError,
    TransientProviderError,
    classify_provider_error,
    failure_kind,
    is_retryable,
)
from .workflow_profiles import resolve_workflow_profile

logger = logging.getLogger(__name__)

# Conditional Langfuse import for tracing
try:
    from langfuse import observe
except ImportError:
    def observe(name=None, **kwargs):
        def decorator(func):
            return func
        return decorator


# Bump on any change to the graph's nodes or edges, and regenerate the
# diagrams (python scripts/export_workflow_diagrams.py). Stamped into every
# response's metadata so each answer can be traced to a graph version.
WORKFLOW_VERSION = "1.1.0"

# Node names - one place, used by the graph, the routers and the docs.
NODE_INITIALIZE = "initialize"
NODE_QUERY_DATA_SOURCES = "query_data_sources"
NODE_EXECUTE_TOOLS = "execute_tools"
NODE_GATHER_CONTEXT = "gather_context"
NODE_BUILD_MESSAGES = "build_messages"
NODE_CALL_PROVIDER = "call_provider"
NODE_FALLBACK_PROVIDER = "fallback_provider"
NODE_CHECK_SAFETY = "check_safety"
NODE_VALIDATE_RESPONSE = "validate_response"
NODE_ESCALATE = "escalate"
NODE_FINALIZE = "finalize"

# Routes out of call_provider
PROVIDER_ANSWERED = "answered"
PROVIDER_FALLBACK = "fallback"
PROVIDER_ESCALATE = "escalate"

# Why a run was escalated
ESCALATION_PROVIDER_FAILURE = "provider_failure"
ESCALATION_POLICY_BLOCK = "policy_block"
ESCALATION_QUALITY_EXHAUSTED = "quality_exhausted"
ESCALATION_TIMEOUT = "timeout"

# Escalations where the safe message is a proper answer for the user. For the
# others, success=False so the agent shows its own fallback (e.g. Zoe's), as it
# did before when the provider errored.
_ESCALATIONS_ANSWERED = {ESCALATION_POLICY_BLOCK, ESCALATION_QUALITY_EXHAUSTED}

_LOW_CONFIDENCE_MESSAGE = "I'm not confident I can provide an accurate answer to that question right now."

_RECENT_EXECUTIONS_LIMIT = 200


class WorkflowState(TypedDict, total=False):
    """
    State shared by all nodes of one run. Must stay serialisable (it is
    checkpointed). List fields with operator.add are appended to by each
    node, which is what lets the parallel branches both write.
    """
    # Identity
    execution_id: str
    agent_type: str
    workflow_version: str

    # Routing plan, fixed at initialize
    use_data_sources: bool
    use_tools: bool

    # Intermediate
    data_source_results: Annotated[List[Dict[str, Any]], operator.add]
    tool_results: Annotated[List[Dict[str, Any]], operator.add]
    messages: List[Dict[str, str]]
    context: Dict[str, Any]

    # Output
    llm_response: Optional[Dict[str, Any]]
    final_content: Optional[str]
    provider_used: Optional[str]
    provider_route: Optional[str]  # answered | fallback | escalate, set by call_provider

    # Metadata
    errors: Annotated[List[str], operator.add]
    execution_steps: Annotated[List[str], operator.add]
    status: str
    start_time: float
    deadline: float
    end_time: Optional[float]

    # Quality loop
    validation_attempts: int
    max_validation_runs: int
    validation_feedback: Optional[str]
    validation_mode: Optional[str]
    is_validated: bool
    confidence_score: float

    # Failure handling
    fallbacks_available: int
    active_fallback: Optional[int]  # index into profile.fallback_providers, None = primary
    failures: Annotated[List[Dict[str, Any]], operator.add]
    escalation_reason: Optional[str]
    safety: Optional[Dict[str, Any]]


@dataclass
class WorkflowContext:
    """Per-run objects that are not checkpointed (LangGraph runtime context)."""
    request: Any  # BrainRequest
    specs: Any  # AgentExecutionSpec
    profile: WorkflowProfile


def _deadline_passed(state: WorkflowState) -> bool:
    deadline = state.get("deadline")
    return bool(deadline) and time.time() >= deadline


class WorkflowEngine:
    """LangGraph-based workflow orchestrator"""

    def __init__(self, checkpointer_kind: Optional[str] = None, checkpoint_path: Optional[str] = None):
        self._checkpointer_kind = checkpointer_kind
        self._checkpoint_path = checkpoint_path
        self.checkpointer = None
        self._checkpoint_conn = None
        self._graphs: Dict[tuple, Any] = {}
        self._recent: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._policy_validator = None
        self._initialized = False
        self.graph = None  # graph for the default profile, kept for callers that inspect it

    async def initialize(self):
        if self._initialized:
            return
        self.checkpointer, self._checkpoint_conn = await create_checkpointer(
            self._checkpointer_kind, self._checkpoint_path
        )
        self.graph = self.get_graph(WorkflowProfile())
        self._initialized = True
        logger.info("Workflow Engine initialized (workflow v%s)", WORKFLOW_VERSION)

    # ------------------------------------------------------------------ #
    # Graph construction
    # ------------------------------------------------------------------ #

    def get_graph(self, profile: Optional[WorkflowProfile] = None):
        """
        The compiled graph (built once). The graph is the same for every agent;
        per-agent settings are read from the profile in runtime context.
        `profile` is accepted for callers that pass one.
        """
        if "default" not in self._graphs:
            self._graphs["default"] = self._build_graph()
        return self._graphs["default"]

    def _build_graph(self):
        workflow = StateGraph(WorkflowState, context_schema=WorkflowContext)

        workflow.add_node(NODE_INITIALIZE, self._initialize_node)
        workflow.add_node(NODE_QUERY_DATA_SOURCES, self._query_data_sources_node)
        workflow.add_node(NODE_EXECUTE_TOOLS, self._execute_tools_node)
        workflow.add_node(NODE_GATHER_CONTEXT, self._gather_context_node)
        workflow.add_node(NODE_BUILD_MESSAGES, self._build_messages_node)
        workflow.add_node(NODE_CALL_PROVIDER, self._call_provider_node)
        workflow.add_node(NODE_FALLBACK_PROVIDER, self._fallback_provider_node)
        workflow.add_node(NODE_CHECK_SAFETY, self._check_safety_node)
        workflow.add_node(NODE_VALIDATE_RESPONSE, self._validate_response_node)
        workflow.add_node(NODE_ESCALATE, self._escalate_node)
        workflow.add_node(NODE_FINALIZE, self._finalize_node)

        workflow.add_edge(START, NODE_INITIALIZE)

        # Parallel fan-out: data sources and tools run in the same step when both
        # are enabled, and meet at gather_context.
        workflow.add_conditional_edges(
            NODE_INITIALIZE,
            self._route_after_initialize,
            [NODE_QUERY_DATA_SOURCES, NODE_EXECUTE_TOOLS, NODE_GATHER_CONTEXT],
        )
        workflow.add_edge(NODE_QUERY_DATA_SOURCES, NODE_GATHER_CONTEXT)
        workflow.add_edge(NODE_EXECUTE_TOOLS, NODE_GATHER_CONTEXT)
        workflow.add_edge(NODE_GATHER_CONTEXT, NODE_BUILD_MESSAGES)
        workflow.add_edge(NODE_BUILD_MESSAGES, NODE_CALL_PROVIDER)

        # call_provider retries transient errors itself; if it still fails it
        # routes to a fallback model, or escalates when none is left.
        workflow.add_conditional_edges(
            NODE_CALL_PROVIDER,
            self._route_after_provider,
            {PROVIDER_ANSWERED: NODE_CHECK_SAFETY, PROVIDER_FALLBACK: NODE_FALLBACK_PROVIDER,
             PROVIDER_ESCALATE: NODE_ESCALATE},
        )

        workflow.add_conditional_edges(
            NODE_FALLBACK_PROVIDER,
            self._route_after_fallback,
            {"recovered": NODE_CHECK_SAFETY, "escalate": NODE_ESCALATE},
        )
        workflow.add_conditional_edges(
            NODE_CHECK_SAFETY,
            self._route_after_safety,
            {"passed": NODE_VALIDATE_RESPONSE, "blocked": NODE_ESCALATE},
        )
        workflow.add_conditional_edges(
            NODE_VALIDATE_RESPONSE,
            self._route_after_validation,
            {"retry": NODE_BUILD_MESSAGES, "finalize": NODE_FINALIZE, "escalate": NODE_ESCALATE},
        )
        workflow.add_edge(NODE_ESCALATE, NODE_FINALIZE)
        workflow.add_edge(NODE_FINALIZE, END)

        return workflow.compile(checkpointer=self.checkpointer)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @observe(name="workflow_execute_plan")
    async def execute_plan(self, plan: Any, request: Any) -> Dict[str, Any]:
        if not self._initialized:
            await self.initialize()

        execution_id = str(uuid.uuid4())
        specs = plan.optimized_specs
        profile = resolve_workflow_profile(specs)
        start_time = time.time()

        initial_state: WorkflowState = {
            "execution_id": execution_id,
            "agent_type": profile.agent_type,
            "workflow_version": WORKFLOW_VERSION,
            "use_data_sources": bool(specs.data_sources and any(d.enabled for d in specs.data_sources)),
            "use_tools": bool(specs.tools and any(t.enabled for t in specs.tools)),
            "data_source_results": [],
            "tool_results": [],
            "messages": [],
            "context": {
                "plan_metadata": plan.to_dict() if hasattr(plan, "to_dict") else {},
                "session_id": getattr(getattr(request, "user_context", None), "session_id", None),
            },
            "llm_response": None,
            "final_content": None,
            "provider_used": None,
            "provider_route": None,
            "errors": [],
            "execution_steps": [],
            "status": WorkflowStatus.RUNNING.value,
            "start_time": start_time,
            "deadline": start_time + profile.run_timeout_seconds,
            "end_time": None,
            "validation_attempts": 0,
            "max_validation_runs": profile.max_iterations,
            "validation_feedback": None,
            "validation_mode": None,
            "is_validated": False,
            "confidence_score": 0.0,
            "fallbacks_available": len(profile.fallback_providers),
            "active_fallback": None,
            "failures": [],
            "escalation_reason": None,
            "safety": None,
        }
        self._tag_trace(execution_id, profile)
        final_state = await self._run(
            self.get_graph(profile), initial_state, execution_id,
            WorkflowContext(request=request, specs=specs, profile=profile), profile,
        )
        return await self._complete(execution_id, final_state, profile)

    async def resume_execution(self, execution_id: str, plan: Any, request: Any) -> Dict[str, Any]:
        """
        Resume an interrupted run from its last checkpoint. The caller passes
        the original plan and request again: they are runtime context, not
        checkpointed state.
        """
        if not self._initialized:
            await self.initialize()
        if self.checkpointer is None:
            raise RuntimeError("Checkpointing is disabled (WORKFLOW_CHECKPOINTER=none); cannot resume")

        specs = plan.optimized_specs
        profile = resolve_workflow_profile(specs)
        graph = self.get_graph(profile)
        config = self._config(execution_id, profile)
        snapshot = await graph.aget_state(config)
        if not snapshot or not snapshot.values:
            raise KeyError(f"No checkpoint for execution {execution_id}")

        if snapshot.next:
            logger.info("Resuming workflow %s at %s", execution_id, list(snapshot.next))
            # The time budget restarts on resume; the original run already timed out or crashed.
            await graph.aupdate_state(config, {"deadline": time.time() + profile.run_timeout_seconds})
            final_state = await self._run(
                graph, None, execution_id,
                WorkflowContext(request=request, specs=specs, profile=profile), profile,
            )
        else:
            final_state = snapshot.values
        return await self._complete(execution_id, final_state, profile)

    async def get_execution(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Summary of a recent run (from the checkpoint if kept, else from memory)."""
        if self.checkpointer is not None and self.graph is not None:
            try:
                snapshot = await self.graph.aget_state(self._config(execution_id))
                if snapshot and snapshot.values:
                    return self._summary(snapshot.values, resumable=bool(snapshot.next))
            except Exception:
                logger.debug("No checkpoint for %s", execution_id, exc_info=True)
        return self._recent.get(execution_id)

    async def shutdown(self):
        if self._checkpoint_conn is not None:
            try:
                await self._checkpoint_conn.close()
            except Exception:
                logger.exception("Error closing workflow checkpoint store")
            self._checkpoint_conn = None
        self._initialized = False
        self._graphs.clear()
        logger.info("Workflow Engine shutdown")

    # ------------------------------------------------------------------ #
    # Running and packaging results
    # ------------------------------------------------------------------ #

    def _config(self, execution_id: str, profile: Optional[WorkflowProfile] = None) -> Dict[str, Any]:
        config: Dict[str, Any] = {"configurable": {"thread_id": execution_id}}
        if profile is not None:
            config["metadata"] = {"agent_type": profile.agent_type, "workflow_version": WORKFLOW_VERSION}
            # Each quality retry is 4 steps (build, call, safety, validate); the
            # rest of the run is under 10. A bounded limit turns a future routing
            # bug into a fast GraphRecursionError instead of LangGraph's default
            # of ~10,000 steps.
            config["recursion_limit"] = 4 * max(1, profile.max_iterations) + 20
        return config

    async def _run(self, graph, graph_input, execution_id: str, ctx: WorkflowContext,
                   profile: WorkflowProfile) -> WorkflowState:
        config = self._config(execution_id, profile)
        # Hard wall-clock cap: the run budget plus one in-flight provider call.
        hard_cap = profile.run_timeout_seconds + profile.provider_timeout_seconds
        try:
            return await asyncio.wait_for(graph.ainvoke(graph_input, config, context=ctx), timeout=hard_cap)
        except asyncio.TimeoutError:
            logger.warning("Workflow %s hit its hard time cap (%.0fs); escalating", execution_id, hard_cap)
            state = await self._latest_state(graph, config, graph_input)
            return self._escalated_state(state, ESCALATION_TIMEOUT, f"Run exceeded {hard_cap:.0f}s")
        except Exception as e:
            logger.exception("Graph execution error in %s", execution_id)
            state = await self._latest_state(graph, config, graph_input)
            state = dict(state)
            state["status"] = WorkflowStatus.FAILED.value
            state["errors"] = list(state.get("errors", [])) + [f"Graph error: {e}"]
            state["end_time"] = time.time()
            return state  # type: ignore[return-value]

    async def _latest_state(self, graph, config, fallback) -> Dict[str, Any]:
        if self.checkpointer is not None:
            try:
                snapshot = await graph.aget_state(config)
                if snapshot and snapshot.values:
                    return dict(snapshot.values)
            except Exception:
                logger.debug("Could not read checkpoint", exc_info=True)
        return dict(fallback or {})

    def _escalated_state(self, state: Dict[str, Any], reason: str, detail: str) -> WorkflowState:
        state = dict(state)
        state["escalation_reason"] = reason
        state["final_content"] = self._escalation_message(reason, state)
        state["status"] = WorkflowStatus.ESCALATED.value
        state["errors"] = list(state.get("errors", [])) + [detail]
        state["failures"] = list(state.get("failures", [])) + [{"node": "workflow", "kind": reason, "detail": detail}]
        state["execution_steps"] = list(state.get("execution_steps", [])) + [NODE_ESCALATE]
        state["end_time"] = time.time()
        return state  # type: ignore[return-value]

    async def _complete(self, execution_id: str, final_state: WorkflowState, profile: WorkflowProfile) -> Dict[str, Any]:
        result = self._to_result(execution_id, final_state, profile)
        # Finished runs drop their checkpoints (they hold the prompt, incl. user
        # context); failed runs keep them so they can be resumed.
        if final_state.get("status") != WorkflowStatus.FAILED.value:
            await self._delete_checkpoint(execution_id)
        self._remember(execution_id, self._summary(final_state, resumable=result["metadata"]["resumable"]))
        return result

    async def _delete_checkpoint(self, execution_id: str):
        if self.checkpointer is None or keep_completed_checkpoints():
            return
        try:
            await self.checkpointer.adelete_thread(execution_id)
        except Exception:
            logger.debug("Could not delete checkpoint %s", execution_id, exc_info=True)

    def _to_result(self, execution_id: str, state: WorkflowState, profile: WorkflowProfile) -> Dict[str, Any]:
        status = state.get("status") or WorkflowStatus.FAILED.value
        final_content = state.get("final_content")
        if final_content is None:
            llm_response = state.get("llm_response")
            final_content = llm_response.get("content", "") if isinstance(llm_response, dict) else ""
        has_content = bool(final_content and final_content.strip())
        reason = state.get("escalation_reason")

        if status == WorkflowStatus.FAILED.value:
            success = False
            final_content = f"Workflow execution failed: {(state.get('errors') or ['unknown error'])[-1]}"
        elif status == WorkflowStatus.ESCALATED.value:
            success = has_content and reason in _ESCALATIONS_ANSWERED
        else:
            success = has_content

        start = state.get("start_time") or time.time()
        return {
            "success": success,
            "content": final_content or "",
            "execution_id": execution_id,
            "status": status,
            "confidence": state.get("confidence_score", 0.0),
            "metadata": {
                "workflow_version": state.get("workflow_version", WORKFLOW_VERSION),
                "agent_type": state.get("agent_type", profile.agent_type),
                "execution_steps": state.get("execution_steps", []),
                "errors": state.get("errors", []),
                "failures": state.get("failures", []),
                "duration_seconds": (state.get("end_time") or time.time()) - start,
                "confidence_score": state.get("confidence_score", 0.0),
                "validation_attempts": state.get("validation_attempts", 0),
                "validation_mode": state.get("validation_mode"),
                "provider_used": state.get("provider_used"),
                "escalation": {"reason": reason, "flagged_for_review": True} if reason else None,
                "safety": state.get("safety"),
                "resumable": status == WorkflowStatus.FAILED.value and self.checkpointer is not None,
            },
        }

    def _summary(self, state: Dict[str, Any], resumable: bool = False) -> Dict[str, Any]:
        """Run summary without message content (no user data)."""
        return {
            "execution_id": state.get("execution_id"),
            "agent_type": state.get("agent_type"),
            "workflow_version": state.get("workflow_version"),
            "status": state.get("status"),
            "execution_steps": list(state.get("execution_steps", [])),
            "escalation_reason": state.get("escalation_reason"),
            "failures": [{"node": f.get("node"), "kind": f.get("kind")} for f in state.get("failures", [])],
            "confidence_score": state.get("confidence_score"),
            "provider_used": state.get("provider_used"),
            "duration_seconds": (state.get("end_time") or time.time()) - (state.get("start_time") or time.time()),
            "resumable": resumable,
        }

    def _remember(self, execution_id: str, summary: Dict[str, Any]):
        self._recent[execution_id] = summary
        self._recent.move_to_end(execution_id)
        while len(self._recent) > _RECENT_EXECUTIONS_LIMIT:
            self._recent.popitem(last=False)

    def _tag_trace(self, execution_id: str, profile: WorkflowProfile):
        """Put the workflow version and profile on the Langfuse span, when tracing is on."""
        if not os.getenv("LANGFUSE_PUBLIC_KEY"):
            return
        try:
            from langfuse import get_client
            get_client().update_current_span(
                version=WORKFLOW_VERSION,
                metadata={"execution_id": execution_id, "workflow_profile": profile.to_dict()},
            )
        except Exception:
            logger.debug("Langfuse span tagging skipped", exc_info=True)

    # ------------------------------------------------------------------ #
    # Nodes
    # ------------------------------------------------------------------ #

    @observe(name="workflow_node_initialize")
    async def _initialize_node(self, state: WorkflowState) -> Dict[str, Any]:
        return {"execution_steps": [NODE_INITIALIZE]}

    @observe(name="workflow_node_query_data_sources")
    async def _query_data_sources_node(self, state: WorkflowState, runtime: Runtime[WorkflowContext]) -> Dict[str, Any]:
        """
        Queries each enabled data source, retrying transient errors with the
        same backoff as call_provider. A source that still fails is recorded
        and the run continues without its context.
        """
        request, specs, profile = runtime.context.request, runtime.context.specs, runtime.context.profile
        results: List[Dict[str, Any]] = []
        errors: List[str] = []
        failures: List[Dict[str, Any]] = []

        try:
            from ..data_sources import get_data_source_registry, create_vector_data_source, DataSourceConfig
            from ..specs import DataSourceType
            registry = get_data_source_registry()
        except Exception as e:
            logger.exception("Data source setup failed")
            return {"execution_steps": [NODE_QUERY_DATA_SOURCES], "errors": [f"DS Error: {e}"]}

        for ds_spec in specs.data_sources:
            if not ds_spec.enabled:
                continue
            source_name = str(ds_spec.source_type.value if hasattr(ds_spec.source_type, "value") else ds_spec.source_type)
            if source_name != "vector_db":
                continue
            if not registry.check_data_source_available("vector_db"):
                results.append({"source": source_name, "error": "Not available"})
                continue

            ds_config = ds_spec.config or {}

            async def query_once(ds_spec=ds_spec, ds_config=ds_config, source_name=source_name):
                config = DataSourceConfig(source_id="vector_db", source_type=DataSourceType.VECTOR_DB, config=ds_config)
                vector_source = create_vector_data_source(config)
                if not await vector_source.initialize(ds_config):
                    return {"source": source_name, "error": "Initialization failed"}
                try:
                    query_res = await vector_source.query(
                        query=ds_spec.query or request.message,
                        context={"filter": ds_config.get("filter")},
                        k=ds_spec.limit or 5,
                    )
                finally:
                    await vector_source.close()
                return {"source": source_name, "results": query_res, "count": len(query_res)}

            try:
                results.append(await self._retry_transient(
                    query_once, attempts=profile.data_source_retry_attempts, profile=profile
                ))
            except Exception as e:
                logger.warning("Data source query failed for %s: %s", ds_spec.source_type, e)
                errors.append(f"DS Error: {e}")
                failures.append({"node": NODE_QUERY_DATA_SOURCES, "kind": failure_kind(e), "detail": str(e)[:300]})
                results.append({"source": source_name, "error": str(e)})

        return {
            "execution_steps": [NODE_QUERY_DATA_SOURCES],
            "data_source_results": results,
            "errors": errors,
            "failures": failures,
        }

    @observe(name="workflow_node_execute_tools")
    async def _execute_tools_node(self, state: WorkflowState, runtime: Runtime[WorkflowContext]) -> Dict[str, Any]:
        results: List[Dict[str, Any]] = []
        errors: List[str] = []
        try:
            from ..tools import get_tool_registry
            registry = get_tool_registry()
            for tool in runtime.context.specs.tools:
                if not tool.enabled:
                    continue
                if registry.check_tool_available(tool.name):
                    # Tool execution is still a placeholder in the Brain (unchanged by US 1.1)
                    results.append({"tool": tool.name, "status": "executed", "result": "Tool execution placeholder"})
                else:
                    results.append({"tool": tool.name, "error": "Not available"})
        except Exception as e:
            logger.exception("Tool execution node error")
            errors.append(str(e))
        return {"execution_steps": [NODE_EXECUTE_TOOLS], "tool_results": results, "errors": errors}

    @observe(name="workflow_node_gather_context")
    async def _gather_context_node(self, state: WorkflowState) -> Dict[str, Any]:
        """Join point after the parallel branches; prepares retrieved text for the groundedness check."""
        context = dict(state.get("context") or {})
        retrieved = self._format_data_source_context(state)
        context["retrieved_context"] = retrieved or None
        context["data_sources_ok"] = sum(1 for r in state.get("data_source_results", []) if "error" not in r)
        context["tools_ok"] = sum(1 for r in state.get("tool_results", []) if "error" not in r)
        return {"execution_steps": [NODE_GATHER_CONTEXT], "context": context}

    @observe(name="workflow_node_build_messages")
    async def _build_messages_node(self, state: WorkflowState, runtime: Runtime[WorkflowContext]) -> Dict[str, Any]:
        request, specs = runtime.context.request, runtime.context.specs
        messages: List[Dict[str, str]] = []
        errors: List[str] = []

        try:
            is_retry = state.get("validation_attempts", 0) > 0 and bool(state.get("validation_feedback"))
            retry_guidance = self._retry_guidance(state) if is_retry else None
            user_info = self._extract_user_info_for_llm(request.user_context)
            agent_meta = getattr(specs, "agent_metadata", {}) or {}
            msg_context = agent_meta.get("messages_context", {})

            if msg_context:
                # Agent-provided prompt (Zoe)
                sys_prompt = msg_context.get("system_prompt", "")
                if user_info and not self._has_user_context_in_prompt(sys_prompt):
                    sys_prompt = self._enhance_prompt_with_user_info(sys_prompt, user_info)
                if sys_prompt:
                    messages.append({"role": "system", "content": sys_prompt})
                for msg in msg_context.get("conversation_history", []):
                    messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
                self._inject_context_messages(messages, state)
                # Previously the agent path ignored validation feedback, so a retry
                # re-sent the identical prompt. Now it carries the feedback too.
                if retry_guidance:
                    messages.append({"role": "system", "content": retry_guidance})
                messages.append({"role": "user", "content": msg_context.get("current_message") or request.message})
            else:
                # Generic build
                sys_parts = []
                if user_info:
                    sys_parts.append(self._build_user_context_prompt(user_info))
                ds_ctx = self._format_data_source_context(state)
                if ds_ctx:
                    sys_parts.append(f"Context from data sources:\n{ds_ctx}")
                tool_ctx = self._format_tool_context(state)
                if tool_ctx:
                    sys_parts.append(f"Tool results:\n{tool_ctx}")
                if retry_guidance:
                    sys_parts.append(retry_guidance)
                if sys_parts:
                    messages.append({"role": "system", "content": "\n\n".join(sys_parts)})
                messages.append({"role": "user", "content": request.message})
        except Exception as e:
            logger.exception("Build messages node error")
            errors.append(str(e))
            if not messages:
                messages = [{"role": "user", "content": getattr(request, "message", None) or "Hello"}]

        return {"execution_steps": [NODE_BUILD_MESSAGES], "messages": messages, "errors": errors}

    @observe(name="workflow_node_call_provider", as_type="generation")
    async def _call_provider_node(self, state: WorkflowState, runtime: Runtime[WorkflowContext]) -> Dict[str, Any]:
        """
        One LLM answer. Transient errors (timeout, 429, 5xx, dropped connection,
        empty reply) are retried with exponential backoff, up to
        profile.provider_retry_attempts and never past the run's time budget.
        If it still fails, the route is set to a fallback model or escalation.
        """
        profile = runtime.context.profile
        spec = self._active_provider_spec(state, runtime.context)
        if spec is None:
            return self._provider_failure(state, ProviderUnavailableError("No provider specified"))
        label = f"{spec.provider_type}:{spec.model}"

        async def one_attempt():
            try:
                return await asyncio.wait_for(
                    self._invoke_provider(spec, state.get("messages") or []),
                    timeout=profile.provider_timeout_seconds,
                )
            except asyncio.TimeoutError:
                raise TransientProviderError(
                    f"{label} timed out after {profile.provider_timeout_seconds:g}s",
                    provider=spec.provider_type, model=spec.model,
                )

        try:
            response = await self._retry_transient(
                one_attempt, attempts=profile.provider_retry_attempts, profile=profile, deadline=state.get("deadline")
            )
        except Exception as exc:  # retries spent, not retryable, or a bug in a provider
            return self._provider_failure(state, exc)

        return {
            "execution_steps": [NODE_CALL_PROVIDER],
            "llm_response": response,
            "final_content": response.get("content", ""),
            "provider_used": label,
            "provider_route": PROVIDER_ANSWERED,
        }

    def _provider_failure(self, state: WorkflowState, exc: BaseException) -> Dict[str, Any]:
        """call_provider failed for good: go to the next fallback model, or escalate."""
        kind = failure_kind(exc)
        logger.warning("call_provider failed (%s): %s", kind, exc)
        update: Dict[str, Any] = {
            "execution_steps": [f"{NODE_CALL_PROVIDER}_failed"],
            "failures": [{"node": NODE_CALL_PROVIDER, "kind": kind, "detail": str(exc)[:300]}],
            "errors": [f"Provider failure ({kind}): {exc}"],
        }
        next_fallback = (state.get("active_fallback") if state.get("active_fallback") is not None else -1) + 1
        if next_fallback < state.get("fallbacks_available", 0) and not _deadline_passed(state):
            update["provider_route"] = PROVIDER_FALLBACK
        else:
            update["provider_route"] = PROVIDER_ESCALATE
            update["escalation_reason"] = ESCALATION_TIMEOUT if _deadline_passed(state) else ESCALATION_PROVIDER_FAILURE
        return update

    @observe(name="workflow_node_fallback_provider", as_type="generation")
    async def _fallback_provider_node(self, state: WorkflowState, runtime: Runtime[WorkflowContext]) -> Dict[str, Any]:
        """Try each remaining fallback provider once, in profile order."""
        profile = runtime.context.profile
        start = (state.get("active_fallback") if state.get("active_fallback") is not None else -1) + 1
        failures: List[Dict[str, Any]] = []
        errors: List[str] = []

        for index in range(start, len(profile.fallback_providers)):
            spec = profile.fallback_providers[index]
            label = f"{spec.provider_type}:{spec.model}"
            if _deadline_passed(state):
                break
            try:
                response = await asyncio.wait_for(
                    self._invoke_provider(spec, state.get("messages") or []),
                    timeout=profile.provider_timeout_seconds,
                )
            except Exception as e:
                kind = failure_kind(e)
                logger.warning("Fallback provider %s failed (%s): %s", label, kind, e)
                failures.append({"node": NODE_FALLBACK_PROVIDER, "kind": kind, "detail": f"{label}: {str(e)[:250]}"})
                errors.append(f"Fallback {label} failed: {e}")
                continue
            logger.info("Fallback provider %s answered", label)
            return {
                "execution_steps": [NODE_FALLBACK_PROVIDER],
                "llm_response": response,
                "final_content": response.get("content", ""),
                "provider_used": label,
                "active_fallback": index,
                "failures": failures,
                "errors": errors,
            }

        reason = ESCALATION_TIMEOUT if _deadline_passed(state) else ESCALATION_PROVIDER_FAILURE
        return {
            "execution_steps": [NODE_FALLBACK_PROVIDER],
            "failures": failures,
            "errors": errors,
            "active_fallback": len(profile.fallback_providers) - 1,
            "escalation_reason": reason,
        }

    @observe(name="workflow_node_check_safety")
    async def _check_safety_node(self, state: WorkflowState, runtime: Runtime[WorkflowContext]) -> Dict[str, Any]:
        """
        Output policy checks (Bhargavi's PolicyValidator, US 2.1). A BLOCK
        replaces the answer with the policy's fallback message and escalates;
        WARN-level flags are only recorded.
        """
        profile = runtime.context.profile
        step = {"execution_steps": [NODE_CHECK_SAFETY]}
        if not profile.safety_checks:
            return {**step, "safety": {"checked": False, "reason": "disabled for this agent"}}

        validator = self._get_policy_validator()
        if validator is None:
            return {**step, "safety": {"checked": False, "reason": "policy validator unavailable"}}

        content = state.get("final_content") or ""
        try:
            report = validator.validate(content, context={"retrieved_context": (state.get("context") or {}).get("retrieved_context")})
        except Exception as e:
            logger.exception("Policy validation error")
            return {**step, "safety": {"checked": False, "reason": f"validator error: {e}"}, "errors": [f"Safety check error: {e}"]}

        failed = [r.policy_name for r in report.failed_policies()]
        safety = {"checked": True, "is_safe": report.is_safe, "failed_policies": failed, "fallback_used": report.fallback_used}
        if report.is_safe:
            return {**step, "safety": safety}

        blocked = PolicyBlockedError(f"Blocked by: {', '.join(failed)}")
        logger.warning("Output blocked by policy in %s: %s", state.get("execution_id"), failed)
        return {
            **step,
            "safety": safety,
            "final_content": report.final_output,  # the policy's own fallback message
            "escalation_reason": ESCALATION_POLICY_BLOCK,
            "failures": [{"node": NODE_CHECK_SAFETY, "kind": blocked.kind, "detail": str(blocked)}],
            "errors": [str(blocked)],
        }

    @observe(name="workflow_node_validate_response")
    async def _validate_response_node(self, state: WorkflowState, runtime: Runtime[WorkflowContext]) -> Dict[str, Any]:
        """Score the answer; decide between accept, retry with feedback, or escalate."""
        profile = runtime.context.profile
        attempts = state.get("validation_attempts", 0) + 1
        update: Dict[str, Any] = {"execution_steps": [NODE_VALIDATE_RESPONSE], "validation_attempts": attempts}

        response_content = state.get("final_content") or ""
        try:
            result = await self._validate_response_quality(response_content, runtime.context.request.message, state, runtime.context)
        except Exception as e:
            logger.exception("Validation node error")
            result = {"is_valid": True, "confidence": profile.confidence_threshold, "mode": "validator_error"}
            update["errors"] = [f"Validation error: {e}"]

        confidence = float(result.get("confidence", 0.0))
        update["confidence_score"] = confidence
        update["validation_mode"] = result.get("mode")

        if confidence >= profile.confidence_threshold:
            update["is_validated"] = True
            update["validation_feedback"] = None
            logger.info("Response validated (attempt %d, confidence %.2f)", attempts, confidence)
            return update

        update["is_validated"] = False
        update["validation_feedback"] = result.get("feedback", f"Confidence too low: {confidence:.2f}")
        if attempts >= state.get("max_validation_runs", profile.max_iterations):
            update["escalation_reason"] = ESCALATION_QUALITY_EXHAUSTED
            update["errors"] = update.get("errors", []) + [f"Response validation failed after {attempts} attempts"]
        elif _deadline_passed(state):
            update["escalation_reason"] = ESCALATION_TIMEOUT
            update["errors"] = update.get("errors", []) + ["Time budget spent before a valid answer"]
        logger.warning("Response validation failed (attempt %d, confidence %.2f)", attempts, confidence)
        return update

    async def _validate_response_quality(self, response: str, user_message: str, state: WorkflowState,
                                         ctx: WorkflowContext) -> Dict[str, Any]:
        """
        Rule checks first, then (if the profile allows) an LLM judge.
        Returns {"is_valid", "confidence" 0-1, "feedback"?, "mode"}.

        Change from before US 1.1: when the judge is off or fails, an answer
        that passes the rule checks is accepted at the threshold. Previously a
        failed judge call scored 0.5, which forced retries and then replaced
        every answer with the fallback message.
        """
        profile = ctx.profile
        if len(response.strip()) < 10:
            return {"is_valid": False, "feedback": "Response is too short", "confidence": 0.1, "mode": "rules"}
        if response.count(" ") < 3:
            return {"is_valid": False, "feedback": "Response lacks substance", "confidence": 0.2, "mode": "rules"}
        error_patterns = ["i cannot", "i can't", "i don't have", "i'm unable", "as an ai", "i apologize, but", "i'm sorry, but"]
        if any(p in response.lower() for p in error_patterns) and len(response) < 100:
            return {"is_valid": False, "feedback": "Response appears to be a refusal or error message", "confidence": 0.3, "mode": "rules"}

        rules_pass = {"is_valid": True, "confidence": profile.confidence_threshold}
        if not profile.llm_validation:
            return {**rules_pass, "mode": "rules"}

        spec = self._active_provider_spec(state, ctx)
        if spec is None:
            return {**rules_pass, "mode": "rules"}

        validation_prompt = f"""Evaluate this AI response and provide a confidence score.

User Message: {user_message}

AI Response: {response}

Evaluate:
1. Relevance to the user's question (0-25 points)
2. Helpfulness and informativeness (0-25 points)
3. Accuracy and correctness (0-25 points)
4. Completeness (not cut off, addresses the question) (0-25 points)

Respond in this EXACT format:
SCORE: <number 0-100>
REASONING: <brief explanation>

If score >= 75, the response is acceptable. Below 75, it needs improvement."""

        try:
            judge = await asyncio.wait_for(
                self._invoke_provider(spec, [{"role": "user", "content": validation_prompt}],
                                      overrides={"max_tokens": 200, "temperature": 0.3}),
                timeout=profile.provider_timeout_seconds,
            )
        except Exception as e:
            logger.warning("LLM judge unavailable, using rule checks only: %s", e)
            return {**rules_pass, "mode": "judge_failed"}

        text = (judge.get("content") or "").strip()
        score_match = re.search(r"SCORE:\s*(\d+)", text)
        if not score_match:
            logger.warning("Could not parse judge score; using rule checks only")
            return {**rules_pass, "mode": "judge_unparsed"}

        llm_confidence = min(int(score_match.group(1)) / 100.0, 1.0)
        # Same weighting as before: 30% rule checks (0.5 base), 70% judge
        final_confidence = 0.5 * 0.3 + llm_confidence * 0.7
        reasoning_match = re.search(r"REASONING:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided"
        if final_confidence >= profile.confidence_threshold:
            return {"is_valid": True, "confidence": final_confidence, "reasoning": reasoning, "mode": "llm_judge"}
        return {"is_valid": False, "confidence": final_confidence,
                "feedback": f"Confidence too low ({final_confidence:.2f}): {reasoning}", "mode": "llm_judge"}

    @observe(name="workflow_node_escalate")
    async def _escalate_node(self, state: WorkflowState) -> Dict[str, Any]:
        """
        Terminal failure path: a safe message for the user, status
        "escalated", and a warning log line carrying the execution id so the
        run can be found and reviewed.
        """
        reason = state.get("escalation_reason") or ESCALATION_PROVIDER_FAILURE
        message = self._escalation_message(reason, state)
        logger.warning(
            "ESCALATED workflow execution_id=%s agent=%s reason=%s failures=%s",
            state.get("execution_id"), state.get("agent_type"), reason,
            [f.get("kind") for f in state.get("failures", [])],
        )
        return {"execution_steps": [NODE_ESCALATE], "final_content": message, "escalation_reason": reason}

    @observe(name="workflow_node_finalize")
    async def _finalize_node(self, state: WorkflowState) -> Dict[str, Any]:
        if state.get("escalation_reason"):
            status = WorkflowStatus.ESCALATED.value
        elif state.get("errors"):
            status = WorkflowStatus.COMPLETED_WITH_ERRORS.value
        else:
            status = WorkflowStatus.COMPLETED.value
        logger.info(
            "Workflow %s finished: %s (confidence %.2f, attempts %d)",
            state.get("execution_id"), status, state.get("confidence_score", 0.0), state.get("validation_attempts", 0),
        )
        return {"execution_steps": [NODE_FINALIZE], "status": status, "end_time": time.time()}

    # ------------------------------------------------------------------ #
    # Routing
    # ------------------------------------------------------------------ #

    def _route_after_initialize(self, state: WorkflowState) -> List[str]:
        """Fan out to every enabled branch; both run in the same step."""
        branches = []
        if state.get("use_data_sources"):
            branches.append(NODE_QUERY_DATA_SOURCES)
        if state.get("use_tools"):
            branches.append(NODE_EXECUTE_TOOLS)
        return branches or [NODE_GATHER_CONTEXT]

    def _route_after_provider(self, state: WorkflowState) -> str:
        return state.get("provider_route") or PROVIDER_ESCALATE

    def _route_after_fallback(self, state: WorkflowState) -> str:
        return "escalate" if state.get("escalation_reason") else "recovered"

    def _route_after_safety(self, state: WorkflowState) -> str:
        return "blocked" if state.get("escalation_reason") == ESCALATION_POLICY_BLOCK else "passed"

    def _route_after_validation(self, state: WorkflowState) -> str:
        if state.get("is_validated"):
            return "finalize"
        if state.get("escalation_reason"):
            return "escalate"
        logger.info("Retrying response generation (attempt %d/%d)",
                    state.get("validation_attempts", 0) + 1, state.get("max_validation_runs", 0))
        return "retry"

    # ------------------------------------------------------------------ #
    # Provider helpers
    # ------------------------------------------------------------------ #

    def _active_provider_spec(self, state: WorkflowState, ctx: WorkflowContext):
        """The primary provider, or the fallback that took over earlier in this run."""
        index = state.get("active_fallback")
        if index is not None and 0 <= index < len(ctx.profile.fallback_providers):
            return ctx.profile.fallback_providers[index]
        return getattr(ctx.specs, "provider", None)

    async def _invoke_provider(self, spec: Any, messages: List[Dict[str, str]],
                               overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create, call and close one provider. Raises a workflow error instead of
        returning {"success": False}, so the graph can route the failure.
        """
        from ..providers import create_provider

        config = spec.to_dict() if hasattr(spec, "to_dict") else {}
        p_type = config.pop("provider_type", None) or spec.provider_type
        label = f"{p_type}:{getattr(spec, 'model', None)}"
        # Retries are owned by the workflow (visible in traces, bounded by the
        # profile); stop the SDK from retrying underneath as well.
        config.setdefault("max_retries", 0)
        key_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
        }
        if p_type in key_map and "api_key" not in config:
            config["api_key"] = os.getenv(key_map[p_type])

        try:
            provider = create_provider(p_type, config)
        except Exception as e:
            raise ProviderUnavailableError(f"Cannot create provider {label}: {e}", provider=p_type, model=spec.model) from e
        if not await provider.initialize():
            raise ProviderUnavailableError(f"Failed to initialize provider {label}", provider=p_type, model=spec.model)

        try:
            kwargs: Dict[str, Any] = {}
            if spec.model:
                kwargs["model"] = spec.model
            if spec.temperature is not None:
                kwargs["temperature"] = spec.temperature
            if spec.max_tokens:
                kwargs["max_tokens"] = spec.max_tokens
            if spec.custom_params:
                kwargs.update(spec.custom_params)
            if overrides:
                kwargs.update(overrides)
            response = await provider.generate_response(messages=messages, **kwargs)
        finally:
            await provider.close()

        if not response.get("success"):
            err = (response.get("metadata") or {}).get("error", "Unknown provider error")
            raise classify_provider_error(err, provider=p_type, model=spec.model)
        if not (response.get("content") or "").strip():
            # Free models occasionally return an empty completion; another try usually works.
            raise TransientProviderError(f"Empty response from {label}", provider=p_type, model=spec.model)
        return response

    async def _retry_transient(self, fn, attempts: int, profile: WorkflowProfile,
                               deadline: Optional[float] = None):
        """
        Call `fn` up to `attempts` times, retrying only transient errors, with
        exponential backoff (initial_interval, doubling, capped at
        max_interval, plus up to 25% jitter). No retry starts after `deadline`.
        """
        delay = profile.retry_initial_interval
        for attempt in range(1, max(1, attempts) + 1):
            try:
                return await fn()
            except Exception as e:
                if attempt >= attempts or not is_retryable(e):
                    raise
                wait = min(delay, profile.retry_max_interval)
                if profile.retry_jitter:
                    wait += random.uniform(0, wait * 0.25)
                if deadline and time.time() + wait >= deadline:
                    raise
                logger.info("Transient error (attempt %d/%d), retrying in %.2fs: %s", attempt, attempts, wait, e)
                await asyncio.sleep(wait)
                delay *= 2

    def _get_policy_validator(self):
        if self._policy_validator is None:
            try:
                from ..guardrails.policy_validator import PolicyValidator
                self._policy_validator = PolicyValidator()
            except Exception:
                logger.warning("PolicyValidator unavailable; safety checks skipped", exc_info=True)
                return None
        return self._policy_validator

    def _escalation_message(self, reason: str, state: Dict[str, Any]) -> str:
        if reason == ESCALATION_POLICY_BLOCK and state.get("final_content"):
            return state["final_content"]  # already the policy-specific message from check_safety
        if reason == ESCALATION_QUALITY_EXHAUSTED:
            return _LOW_CONFIDENCE_MESSAGE
        try:
            from ..guardrails.fallback_handler import DEFAULT_FALLBACK
            return DEFAULT_FALLBACK
        except Exception:
            return "I'm not able to provide a response to that right now. Please try again."

    # ------------------------------------------------------------------ #
    # Message helpers (unchanged behaviour)
    # ------------------------------------------------------------------ #

    def _retry_guidance(self, state: WorkflowState) -> str:
        prev_response = state.get("final_content") or ""
        feedback = state.get("validation_feedback") or ""
        return f"""IMPORTANT - Response Improvement Required:
Your previous response was not satisfactory.

Previous response: {prev_response[:500]}{'...' if len(prev_response) > 500 else ''}

Feedback: {feedback}

Please generate a better, more helpful, and complete response that addresses the user's question properly."""

    def _inject_context_messages(self, messages: List[Dict], state: WorkflowState):
        """Helper to inject DS/Tool results into messages"""
        ds_ctx = self._format_data_source_context(state)
        if ds_ctx:
            messages.append({"role": "system", "content": f"Context:\n{ds_ctx}"})
        tool_ctx = self._format_tool_context(state)
        if tool_ctx:
            messages.append({"role": "system", "content": f"Tool Results:\n{tool_ctx}"})

    def _format_data_source_context(self, state: WorkflowState) -> str:
        results = [r for r in state.get("data_source_results", []) if "error" not in r]
        if not results:
            return ""
        return "\n".join(f"Data from {r['source']}: {r.get('message', r.get('results', ''))}" for r in results)

    def _format_tool_context(self, state: WorkflowState) -> str:
        if not state.get("tool_results"):
            return ""
        return "\n".join(f"Tool {r['tool']}: {r.get('result', r.get('error', ''))}" for r in state["tool_results"])

    # User Context Utilities

    def _extract_user_info_for_llm(self, user_context) -> Optional[Dict[str, Any]]:
        if not user_context:
            return None

        from dataclasses import asdict
        ctx = asdict(user_context) if hasattr(user_context, "__dataclass_fields__") else dict(user_context or {})

        if not ctx.get("is_authenticated") and not ctx.get("authenticated"):
            return None

        info = {k: ctx.get(k) for k in ["user_id", "session_id", "name", "email", "roles"] if ctx.get(k)}

        profile = ctx.get("user_profile")
        if profile:
            ace = profile.get("ace_score") if isinstance(profile, dict) else getattr(profile, "ace_score", None)
            if ace is not None:
                info["ace_score"] = ace

        return info if len(info) > 1 else None  # >1 because user_id/session_id almost always exist

    def _has_user_context_in_prompt(self, prompt: str) -> bool:
        if not prompt:
            return False
        indicators = ["user_id", "session_id", "user context", "user information", "authenticated user"]
        prompt_lower = prompt.lower()
        return any(i in prompt_lower for i in indicators)

    def _enhance_prompt_with_user_info(self, prompt: str, user_info: Dict[str, Any]) -> str:
        ctx_str = self._build_user_context_prompt(user_info)
        return f"{ctx_str}\n\n{prompt}" if ctx_str else prompt

    def _build_user_context_prompt(self, info: Dict[str, Any]) -> str:
        if not info:
            return ""
        lines = []
        if info.get("name"):
            lines.append(f"User name: {info['name']}")
        if info.get("email"):
            lines.append(f"User email: {info['email']}")
        if info.get("roles"):
            lines.append(f"User roles: {', '.join(info['roles'])}")
        if info.get("ace_score") is not None:
            lines.append(f"User ACE score: {info['ace_score']}")

        return "User Context:\n" + "\n".join(f"- {l}" for l in lines) if lines else ""


def describe_graph_mermaid(engine: WorkflowEngine, profile: Optional[WorkflowProfile] = None) -> str:
    """
    Mermaid source for the workflow, generated from the graph definition
    itself (US 1.1 release traceability), so the diagram cannot drift from
    the code. Built from the StateGraph's own edges and branches; this
    output is the same on LangGraph 0.6 and 1.x, unlike LangGraph's renderer.

    Solid arrow = always; dotted arrow = conditional, labelled with the route.
    """
    builder = engine.get_graph(profile or WorkflowProfile()).builder

    def node_id(name: str) -> str:
        return {START: "wf_start", END: "wf_end"}.get(name, name)

    lines = [
        "flowchart TD",
        f"    %% ThinkLife Brain workflow v{WORKFLOW_VERSION} - generated by scripts/export_workflow_diagrams.py; do not edit",
        "    wf_start([start])",
        "    wf_end([end])",
    ]
    for name in builder.nodes:
        lines.append(f"    {name}[{name}]")

    edges = []
    for src, dst in sorted(builder.edges):
        edges.append(f"    {node_id(src)} --> {node_id(dst)}")
    for src in builder.branches:
        for branch in builder.branches[src].values():
            for label, dst in branch.ends.items():
                text = label
                if src == NODE_INITIALIZE:
                    text = "no data sources or tools" if dst == NODE_GATHER_CONTEXT else "in parallel"
                elif src == NODE_CALL_PROVIDER and label != PROVIDER_ANSWERED:
                    text = "still failing, " + ("try fallback model" if label == PROVIDER_FALLBACK else "none left")
                edges.append(f"    {node_id(src)} -. {text} .-> {node_id(dst)}")
    lines.extend(sorted(set(edges), key=edges.index))

    new_nodes = (NODE_GATHER_CONTEXT, NODE_FALLBACK_PROVIDER, NODE_CHECK_SAFETY, NODE_ESCALATE)
    lines.append("    classDef added stroke-dasharray: 5 4")
    lines.append(f"    class {','.join(new_nodes)} added")
    return "\n".join(lines) + "\n"


# Singleton
_workflow_engine_instance: Optional[WorkflowEngine] = None


def get_workflow_engine() -> WorkflowEngine:
    global _workflow_engine_instance
    if _workflow_engine_instance is None:
        _workflow_engine_instance = WorkflowEngine()
    return _workflow_engine_instance
