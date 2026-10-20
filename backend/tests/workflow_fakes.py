"""
Test doubles for the Brain workflow (US 1.1): a scriptable fake LLM provider,
a fake vector store and tool registry, and builders for specs and requests.
No network, no API keys.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from brain.specs import (
    AgentExecutionSpec, ApplicationType, BrainRequest, DataSourceSpec, DataSourceType,
    ExecutionPlan, ProcessingSpec, ProviderSpec, UserContext,
)
from brain.specs.tool_specs import ToolSpec

GOOD_ANSWER = "Here is a clear and complete answer to your question about the program schedule."
JUDGE_PASS = "SCORE: 95\nREASONING: Relevant and complete."
JUDGE_FAIL = "SCORE: 20\nREASONING: Off topic."


def ok(content: str) -> Dict[str, Any]:
    return {"success": True, "content": content, "metadata": {"provider": "fake"}}


def err(message: str) -> Dict[str, Any]:
    return {"success": False, "content": "", "metadata": {"error": message, "provider": "fake"}}


@dataclass
class FakeLLM:
    """
    Scripted behaviour per model name. Each script is a list consumed in
    order (the last item repeats). Items: a response dict, an Exception to
    raise, or a callable(messages) -> response dict. Judge calls (the
    validation prompt) are answered from `judge` instead.
    """
    scripts: Dict[str, List[Any]] = field(default_factory=dict)
    judge: List[Any] = field(default_factory=lambda: [ok(JUDGE_PASS)])
    init_fails: set = field(default_factory=set)
    delay: float = 0.0
    calls: List[Dict[str, Any]] = field(default_factory=list)
    configs: List[Dict[str, Any]] = field(default_factory=list)

    def _next(self, queue: List[Any], messages):
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(item, BaseException):
            raise item
        return item(messages) if callable(item) else item

    def factory(self, provider_type: str, config: dict):
        llm = self
        self.configs.append(dict(config))

        class _Provider:
            def __init__(self):
                self.model = config.get("model")

            async def initialize(self):
                return (provider_type, self.model) not in llm.init_fails

            async def generate_response(self, messages, **kwargs):
                model = kwargs.get("model", self.model)
                is_judge = "Evaluate this AI response" in (messages[-1].get("content") or "")
                llm.calls.append({"provider": provider_type, "model": model, "judge": is_judge,
                                  "messages": messages, "kwargs": kwargs})
                if llm.delay:
                    await asyncio.sleep(llm.delay)
                if is_judge:
                    return llm._next(llm.judge, messages)
                return llm._next(llm.scripts.get(model) or [ok(GOOD_ANSWER)], messages)

            async def close(self):
                pass

        return _Provider()

    def answer_calls(self) -> List[Dict[str, Any]]:
        return [c for c in self.calls if not c["judge"]]


class FakeVectorSource:
    def __init__(self, behaviour: List[Any]):
        self.behaviour = behaviour

    async def initialize(self, config):
        return True

    async def query(self, query, context=None, k=5):
        item = self.behaviour.pop(0) if len(self.behaviour) > 1 else self.behaviour[0]
        if isinstance(item, BaseException):
            raise item
        return item

    async def close(self):
        pass


class FakeRegistry:
    def check_data_source_available(self, name):
        return True

    def check_tool_available(self, name):
        return True


def install_fakes(monkeypatch, llm: FakeLLM, vector_behaviour: Optional[List[Any]] = None,
                  ds_delay: float = 0.0, timeline: Optional[List] = None):
    """Route provider, data-source and tool lookups in the workflow to fakes."""
    import time
    import brain.data_sources as ds_mod
    import brain.providers as providers_mod
    import brain.tools as tools_mod

    monkeypatch.setattr(providers_mod, "create_provider", llm.factory)
    behaviour = vector_behaviour if vector_behaviour is not None else [[{"content": "The program runs on Saturdays."}]]

    def create_vector(config):
        source = FakeVectorSource(behaviour)
        original = source.query

        async def timed_query(*a, **kw):
            if timeline is not None:
                timeline.append(("ds_start", time.monotonic()))
            if ds_delay:
                await asyncio.sleep(ds_delay)
            try:
                return await original(*a, **kw)
            finally:
                if timeline is not None:
                    timeline.append(("ds_end", time.monotonic()))

        source.query = timed_query
        return source

    monkeypatch.setattr(ds_mod, "create_vector_data_source", create_vector)
    monkeypatch.setattr(ds_mod, "get_data_source_registry", lambda: FakeRegistry())

    class TimedToolRegistry(FakeRegistry):
        def check_tool_available(self, name):
            if timeline is not None:
                timeline.append(("tools", time.monotonic()))
            return True

    monkeypatch.setattr(tools_mod, "get_tool_registry", lambda: TimedToolRegistry())


def make_specs(*, agent_type: str = "test", provider: str = "openai", model: str = "gpt-4o-mini",
               data_sources: bool = False, tools: bool = False, max_iterations: int = 3,
               timeout_seconds: float = 30.0, safety: bool = True,
               messages_context: Optional[Dict[str, Any]] = None) -> AgentExecutionSpec:
    meta: Dict[str, Any] = {"agent_type": agent_type}
    if messages_context is not None:
        meta["messages_context"] = messages_context
    return AgentExecutionSpec(
        data_sources=[DataSourceSpec(source_type=DataSourceType.VECTOR_DB, query=None, limit=3, enabled=True,
                                     config={"collection_name": "test"})] if data_sources else [],
        provider=ProviderSpec(provider_type=provider, model=model, temperature=0.7, max_tokens=300),
        tools=[ToolSpec(name="tavily_search", enabled=True)] if tools else [],
        processing=ProcessingSpec(max_iterations=max_iterations, timeout_seconds=timeout_seconds,
                                  enable_safety_checks=safety),
        agent_metadata=meta,
    )


def make_plan(specs: AgentExecutionSpec) -> ExecutionPlan:
    return ExecutionPlan(original_specs=specs, optimized_specs=specs, reasoning_applied=False, confidence=1.0)


def make_request(message: str = "When does the program run?") -> BrainRequest:
    return BrainRequest(
        id="req-1",
        application=ApplicationType.GENERAL,
        message=message,
        user_context=UserContext(user_id="u1", session_id="s1", is_authenticated=False),
    )
