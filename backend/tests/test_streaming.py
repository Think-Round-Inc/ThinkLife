"""
Tests for streaming (Epic 5 / Story 5.1, Task 10.6).

Run with: pytest tests/test_streaming.py -v
"""

import asyncio
import json

import pytest
from fastapi.testclient import TestClient


# --- Provider-level streaming (10.1) ---------------------------------------

def test_openai_provider_stream_response_yields_text_deltas(monkeypatch):
    async def run():
        from brain.providers.openai import OpenAIProvider

        class FakeDelta:
            def __init__(self, content):
                self.content = content

        class FakeChoice:
            def __init__(self, content):
                self.delta = FakeDelta(content)

        class FakeChunk:
            def __init__(self, content):
                self.choices = [FakeChoice(content)] if content is not None else []

        class FakeStream:
            def __aiter__(self):
                return self

            def __init__(self):
                self._words = iter(["Hello", " world", None, "!"])

            async def __anext__(self):
                try:
                    word = next(self._words)
                except StopIteration:
                    raise StopAsyncIteration
                return FakeChunk(word)

        class FakeCompletions:
            async def create(self, **params):
                assert params["stream"] is True
                return FakeStream()

        class FakeChat:
            completions = FakeCompletions()

        class FakeClient:
            chat = FakeChat()

        provider = OpenAIProvider({"api_key": "fake", "model": "gpt-4o-mini"})
        provider.client = FakeClient()
        provider._initialized = True

        chunks = []
        async for chunk in provider.stream_response(messages=[{"role": "user", "content": "hi"}]):
            chunks.append(chunk)

        assert chunks == ["Hello", " world", "!"]  # None-content chunk skipped

    asyncio.run(run())


def test_openai_provider_stream_response_raises_if_not_initialized():
    async def run():
        from brain.providers.openai import OpenAIProvider

        provider = OpenAIProvider({})
        with pytest.raises(RuntimeError, match="not initialized"):
            async for _ in provider.stream_response(messages=[]):
                pass

    asyncio.run(run())


# --- ResilientProvider streaming passthrough (10.1/10.3) -------------------

def test_resilient_provider_stream_response_passes_through_chunks():
    async def run():
        from brain.resilience.provider_resilience import ResilientProvider

        class FakeStreamingInner:
            name = "fake"

            async def stream_response(self, messages, **kwargs):
                for c in ["a", "b", "c"]:
                    yield c

        wrapped = ResilientProvider(FakeStreamingInner())
        chunks = [c async for c in wrapped.stream_response(messages=[])]
        assert chunks == ["a", "b", "c"]

    asyncio.run(run())


def test_resilient_provider_stream_response_fails_fast_when_circuit_open():
    async def run():
        from brain.resilience.provider_resilience import ResilientProvider
        from brain.resilience.circuit_breaker import CircuitBreaker, CircuitOpenError

        class FakeStreamingInner:
            name = "fake"
            called = False

            async def stream_response(self, messages, **kwargs):
                FakeStreamingInner.called = True
                yield "should not happen"

        breaker = CircuitBreaker("stream-test", failure_threshold=1, recovery_timeout=60)

        async def fails():
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            await breaker.call(fails)

        wrapped = ResilientProvider(FakeStreamingInner(), circuit_breaker=breaker)
        with pytest.raises(CircuitOpenError):
            async for _ in wrapped.stream_response(messages=[]):
                pass
        assert FakeStreamingInner.called is False

    asyncio.run(run())


def test_resilient_provider_stream_response_raises_not_implemented_without_inner_support():
    async def run():
        from brain.resilience.provider_resilience import ResilientProvider

        class NonStreamingInner:
            name = "fake"

        wrapped = ResilientProvider(NonStreamingInner())
        with pytest.raises(NotImplementedError):
            async for _ in wrapped.stream_response(messages=[]):
                pass

    asyncio.run(run())


# --- WorkflowEngine.stream_execution (10.3) --------------------------------

class _FakeStreamingProvider:
    name = "openai"

    def __init__(self, words, close_tracker=None, hang_after=None):
        self.words = words
        self.close_tracker = close_tracker if close_tracker is not None else []
        self.hang_after = hang_after

    async def initialize(self):
        return True

    async def close(self):
        self.close_tracker.append(1)

    async def stream_response(self, messages, **kwargs):
        for i, word in enumerate(self.words):
            if self.hang_after is not None and i == self.hang_after:
                await asyncio.sleep(10)  # simulate a hung stream; test will cancel before this
            yield word


def test_workflow_engine_stream_execution_yields_chunks_and_records_execution(monkeypatch):
    async def run():
        import brain.providers as providers_pkg
        from brain.cortex.workflow_engine import WorkflowEngine
        from brain.specs import BrainRequest, UserContext, AgentExecutionSpec, ProviderSpec, ProcessingSpec

        fake = _FakeStreamingProvider(["Hi", " there"])
        monkeypatch.setattr(providers_pkg, "create_provider", lambda t, c: fake)

        engine = WorkflowEngine()
        request = BrainRequest(
            id="stream-1", application="general", message="hi",
            user_context=UserContext(user_id="u", session_id="s", is_authenticated=False),
        )
        specs = AgentExecutionSpec(
            data_sources=[], tools=[],
            provider=ProviderSpec(provider_type="openai", model="gpt-4o-mini"),
            processing=ProcessingSpec(execution_strategy="direct"),
        )

        chunks = [c async for c in engine.stream_execution(specs, request)]
        assert chunks == ["Hi", " there"]
        assert fake.close_tracker == [1]  # managed_provider closed it exactly once

        recorded = engine.executions.get("stream-1") or next(
            (v for v in engine.executions.values() if v.get("final_content") == "Hi there"), None
        )
        assert recorded is not None
        assert recorded["final_content"] == "Hi there"
        assert recorded["status"] == "completed"

    asyncio.run(run())


def test_workflow_engine_stream_execution_falls_back_to_single_chunk_without_streaming_support(monkeypatch):
    """Providers without stream_response (e.g. Gemini, per Task 10.1's scope) degrade gracefully."""
    async def run():
        import brain.providers as providers_pkg
        from brain.cortex.workflow_engine import WorkflowEngine
        from brain.specs import BrainRequest, UserContext, AgentExecutionSpec, ProviderSpec, ProcessingSpec

        class NonStreamingProvider:
            name = "gemini"

            async def initialize(self):
                return True

            async def close(self):
                pass

            async def generate_response(self, messages, **kwargs):
                return {"success": True, "content": "full response at once", "metadata": {}}

        monkeypatch.setattr(providers_pkg, "create_provider", lambda t, c: NonStreamingProvider())

        engine = WorkflowEngine()
        request = BrainRequest(
            id="stream-2", application="general", message="hi",
            user_context=UserContext(user_id="u", session_id="s", is_authenticated=False),
        )
        specs = AgentExecutionSpec(
            data_sources=[], tools=[],
            provider=ProviderSpec(provider_type="gemini", model="gemini-pro"),
            processing=ProcessingSpec(execution_strategy="direct"),
        )

        chunks = [c async for c in engine.stream_execution(specs, request)]
        assert chunks == ["full response at once"]

    asyncio.run(run())


def test_workflow_engine_stream_execution_cleans_up_on_client_disconnect(monkeypatch):
    """
    Core Task 10.6 guarantee: if the caller stops consuming the generator
    early (simulating a client disconnecting mid-stream), the provider is
    still closed - Python delivers GeneratorExit at the suspended `yield`
    and unwinds through managed_provider's __aexit__ before the generator
    is considered closed (see WorkflowEngine.stream_execution's docstring).
    """
    async def run():
        import brain.providers as providers_pkg
        from brain.cortex.workflow_engine import WorkflowEngine
        from brain.specs import BrainRequest, UserContext, AgentExecutionSpec, ProviderSpec, ProcessingSpec

        close_tracker = []
        fake = _FakeStreamingProvider(["one", "two", "three", "four"], close_tracker=close_tracker, hang_after=2)
        monkeypatch.setattr(providers_pkg, "create_provider", lambda t, c: fake)

        engine = WorkflowEngine()
        request = BrainRequest(
            id="stream-disconnect", application="general", message="hi",
            user_context=UserContext(user_id="u", session_id="s", is_authenticated=False),
        )
        specs = AgentExecutionSpec(
            data_sources=[], tools=[],
            provider=ProviderSpec(provider_type="openai", model="gpt-4o-mini"),
            processing=ProcessingSpec(execution_strategy="direct"),
        )

        gen = engine.stream_execution(specs, request)
        received = []
        received.append(await gen.__anext__())
        received.append(await gen.__anext__())
        # Simulate the client disconnecting here, before the (hung) third
        # chunk would ever arrive.
        await gen.aclose()

        assert received == ["one", "two"]
        assert close_tracker == [1], "provider.close() must run even though the stream was closed early"

    asyncio.run(run())


# --- ZoeAgent / ZoeService streaming (10.2) --------------------------------

def test_zoe_service_stream_message_respects_safety_redirect(monkeypatch):
    """
    The streaming path must go through the same harmful-content redirect
    check as process_message - never call an LLM at all for a message that
    should be redirected.
    """
    async def run():
        from agents.zoe.zoe_service import ZoeService

        service = ZoeService()
        await service.initialize()
        try:
            llm_was_called = {"value": False}

            original_stream_response = service.zoe_agent.stream_response

            async def spy_stream_response(request):
                llm_was_called["value"] = True
                async for c in original_stream_response(request):  # pragma: no cover
                    yield c

            monkeypatch.setattr(service.zoe_agent, "stream_response", spy_stream_response)
            monkeypatch.setattr(
                service.zoe_core, "create_brain_request",
                lambda **kwargs: {
                    "session_id": kwargs.get("session_id") or "redirect-session",
                    "redirect_info": {
                        "should_redirect": True,
                        "response": "I'm concerned about what you shared - here are some resources.",
                        "is_harmful": True,
                    },
                },
            )

            chunks = [c async for c in service.stream_message(message="something concerning", session_id="redirect-session")]
            assert chunks == ["I'm concerned about what you shared - here are some resources."]
            assert llm_was_called["value"] is False
        finally:
            await service.shutdown()

    asyncio.run(run())


def test_zoe_agent_stream_response_delegates_to_cortex_streaming(monkeypatch):
    async def run():
        from plugins.zoe_agent import ZoeAgent
        from brain.specs import AgentConfig, BrainRequest, UserContext

        agent = ZoeAgent(AgentConfig(agent_id="zoe"))
        await agent.initialize(agent.config)

        async def fake_stream(specs, request):
            for word in ["Hi", " friend"]:
                yield word

        monkeypatch.setattr(agent.cortex, "process_agent_request_streaming", fake_stream)
        monkeypatch.setattr(agent.zoe_core, "post_process_response", lambda text, ctx: text)  # no supplement

        request = BrainRequest(
            id="zoe-stream-1", application="chatbot", message="hi",
            user_context=UserContext(user_id="u", session_id="s", is_authenticated=False),
        )

        chunks = [c async for c in agent.stream_response(request)]
        assert chunks == ["Hi", " friend"]

    asyncio.run(run())


# --- SSE endpoint integration (10.4/10.5/10.6) -----------------------------

def _parse_sse_frames(response):
    frames = []
    for line in response.iter_lines():
        if line and line.startswith("data: "):
            frames.append(json.loads(line[len("data: "):]))
    return frames


def test_sse_endpoint_streams_multiple_chunks_then_done(monkeypatch):
    import main
    import brain.providers as providers_pkg

    class FakeStreamingProvider:
        name = "openai"

        async def initialize(self):
            return True

        async def close(self):
            pass

        async def health_check(self):
            return {"healthy": True}

        async def stream_response(self, messages, **kwargs):
            for word in ["Hello", " there", "!"]:
                yield word

        async def generate_response(self, messages, **kwargs):
            return {"success": True, "content": "Hello there!", "metadata": {}}

    monkeypatch.setattr(providers_pkg, "create_provider", lambda t, c: FakeStreamingProvider())

    with TestClient(main.app) as client:
        with client.stream("POST", "/api/zoe/chat/stream", json={"message": "hi"}) as r:
            assert r.status_code == 200
            assert "text/event-stream" in r.headers["content-type"]
            assert "X-Correlation-ID" in r.headers
            frames = _parse_sse_frames(r)

    chunk_frames = [f for f in frames if f["type"] == "chunk"]
    done_frames = [f for f in frames if f["type"] == "done"]
    assert len(chunk_frames) == 3
    assert "".join(f["content"] for f in chunk_frames) == "Hello there!"
    assert len(done_frames) == 1
    assert done_frames[0]["session_id"]


def test_sse_endpoint_reports_error_frame_on_provider_failure(monkeypatch):
    """
    No provider credentials configured in the test environment -> the
    provider fails to initialize inside managed_provider, which should
    surface as a single 'error' SSE frame (200 status - SSE can't change the
    status code after streaming has started), not an unhandled 500.
    """
    import main

    with TestClient(main.app) as client:
        with client.stream("POST", "/api/zoe/chat/stream", json={"message": "hi"}) as r:
            assert r.status_code == 200
            frames = _parse_sse_frames(r)

    assert len(frames) == 1
    assert frames[0]["type"] == "error"
    assert frames[0]["error"]


def test_sse_endpoint_rejects_ace_score_restricted_chat():
    import main

    with TestClient(main.app) as client:
        response = client.post(
            "/api/zoe/chat/stream",
            json={"message": "hi", "user_context": {"ace_score": 5}},
        )
        assert response.status_code == 403


def test_sse_endpoint_validates_blank_message():
    import main

    with TestClient(main.app) as client:
        response = client.post("/api/zoe/chat/stream", json={"message": "   "})
        assert response.status_code == 422
