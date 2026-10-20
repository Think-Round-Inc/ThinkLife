"""
Workflow error taxonomy (US 1.1 - failure handling transitions)

Every failure inside the workflow is turned into one of these types so the
graph can pick a path for it:

    TransientProviderError   -> retried with backoff (RetryPolicy on call_provider)
    ProviderUnavailableError -> no retry; fallback provider, else escalate
    ProviderCallError        -> no retry; fallback provider, else escalate
    PolicyBlockedError       -> never retried; escalate with the policy's message
    WorkflowTimeoutError     -> run budget spent; escalate

Providers in brain/providers/ return {"success": False, "metadata": {"error": ...}}
instead of raising, so `classify_provider_error` reads that error text and
decides whether it is worth retrying.
"""

import asyncio
import re
from typing import Optional


class WorkflowError(Exception):
    """Base class for errors the workflow knows how to route."""

    kind = "workflow_error"

    def __init__(self, message: str, *, provider: Optional[str] = None, model: Optional[str] = None):
        super().__init__(message)
        self.provider = provider
        self.model = model


class TransientProviderError(WorkflowError):
    """Timeout, rate limit, 5xx or dropped connection. Safe to retry."""

    kind = "provider_transient"


class ProviderUnavailableError(WorkflowError):
    """Provider could not be created or initialised (missing key, bad model, missing SDK)."""

    kind = "provider_unavailable"


class ProviderCallError(WorkflowError):
    """Provider answered with a non-retryable error (bad request, auth, content filter)."""

    kind = "provider_error"


class PolicyBlockedError(WorkflowError):
    """Output blocked by a guardrail policy. Never retried."""

    kind = "policy_block"


class WorkflowTimeoutError(WorkflowError):
    """The run's time budget (ProcessingSpec.timeout_seconds) is spent."""

    kind = "timeout"


# Error text from the OpenAI / Anthropic / Gemini / OpenRouter SDKs that means
# "try again later". Matched case-insensitively against the provider's error string.
_TRANSIENT_PATTERNS = [
    # "insufficient_quota" (billing) is deliberately NOT here: retrying cannot fix it.
    r"\b429\b", r"rate.?limit", r"too many requests",
    r"\b50[0234]\b", r"overloaded", r"server error", r"service unavailable", r"bad gateway",
    r"timed? ?out", r"timeout", r"connection", r"temporarily", r"try again",
]
_TRANSIENT_RE = re.compile("|".join(_TRANSIENT_PATTERNS), re.IGNORECASE)

# Errors that look transient (OpenAI sends billing problems as HTTP 429) but are not.
_PERMANENT_RE = re.compile(r"insufficient_quota|exceeded your current quota|invalid.{0,10}api.?key|unauthori[sz]ed|\b401\b|\b403\b",
                           re.IGNORECASE)


def is_transient_error_text(text: Optional[str]) -> bool:
    if not text or _PERMANENT_RE.search(text):
        return False
    return bool(_TRANSIENT_RE.search(text))


def classify_provider_error(error_text: str, *, provider: Optional[str] = None,
                            model: Optional[str] = None) -> WorkflowError:
    """Turn a provider's error string into the matching workflow error."""
    cls = TransientProviderError if is_transient_error_text(error_text) else ProviderCallError
    return cls(error_text or "Unknown provider error", provider=provider, model=model)


def is_retryable(exc: BaseException) -> bool:
    """
    retry_on predicate for LangGraph's RetryPolicy.

    LangGraph's default predicate only retries network-style exceptions, so a
    plain RuntimeError would never be retried; this makes the rule explicit.
    """
    try:
        from langgraph.errors import NodeTimeoutError
    except ImportError:  # pragma: no cover - older langgraph
        NodeTimeoutError = ()  # type: ignore[assignment]

    if isinstance(exc, WorkflowError):
        return isinstance(exc, TransientProviderError)
    if NodeTimeoutError and isinstance(exc, NodeTimeoutError):
        return True
    if isinstance(exc, (ConnectionError, asyncio.TimeoutError, TimeoutError)):
        return True
    try:
        import httpx
        if isinstance(exc, (httpx.TransportError, httpx.TimeoutException)):
            return True
    except ImportError:  # pragma: no cover
        pass
    return False


def failure_kind(exc: BaseException) -> str:
    """Short label recorded in state and response metadata."""
    if isinstance(exc, WorkflowError):
        return exc.kind
    if is_retryable(exc):
        return TransientProviderError.kind
    return "unexpected_error"
