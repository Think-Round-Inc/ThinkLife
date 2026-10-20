"""
Shared pytest setup for backend tests.

Lets `pytest` run from either the repo root or backend/, and isolates the
workflow tests from a developer's .env (an OPENROUTER_API_KEY there would
otherwise add a default fallback provider and change what the tests see).
"""

import asyncio
import os
import sys

import pytest

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
for path in (BACKEND_DIR, TESTS_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

_WORKFLOW_ENV = [
    "OPENROUTER_API_KEY", "WORKFLOW_FALLBACK_PROVIDERS", "TEST_FALLBACK_PROVIDERS", "ZOE_FALLBACK_PROVIDERS",
    "WORKFLOW_LLM_VALIDATION", "WORKFLOW_CHECKPOINTER", "WORKFLOW_CHECKPOINT_PATH",
    "WORKFLOW_KEEP_COMPLETED_CHECKPOINTS", "LANGFUSE_PUBLIC_KEY",
]


@pytest.fixture
def workflow_env(monkeypatch):
    """Clean env + a fast 'test' profile (millisecond backoff) for workflow tests."""
    for name in _WORKFLOW_ENV:
        monkeypatch.delenv(name, raising=False)

    from brain.specs import WorkflowProfile
    import brain.cortex.workflow_profiles as profiles

    test_profile = WorkflowProfile(
        agent_type="test",
        max_iterations=3,
        run_timeout_seconds=30.0,
        provider_timeout_seconds=5.0,
        retry_initial_interval=0.001,
        retry_max_interval=0.005,
        retry_jitter=False,
    )
    monkeypatch.setitem(profiles.BUILTIN_PROFILES, "test", test_profile)
    return test_profile


def run(coro):
    """Run a coroutine in a fresh event loop (no pytest-asyncio needed)."""
    return asyncio.run(coro)
