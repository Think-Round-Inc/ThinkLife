"""
ThinkLife Evaluation System with LangFuse Observability

Evaluation framework for AI responses with:
- Response latency monitoring
- LangFuse integration for tracing and monitoring (v3.x)

Usage:
    from evaluation.evaluation_manager import evaluate_response, get_evaluation_manager
    
    # Run evaluation
    results = await evaluate_response(
        user_message="I'm feeling anxious",
        bot_message="I hear that you're feeling anxious...",
        start_time=start_timestamp,
        end_time=end_timestamp,
        context={"user_id": "user_123", "session_id": "session_456"}
    )
    
    # Access latency evaluator
    manager = get_evaluation_manager()
    latency_result = await manager.latency_evaluator.evaluate(start_time, end_time)
"""

import logging
import os
import re
import sys
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, Any, Optional
from dotenv import load_dotenv

from .evaluators.latency_evaluator import LatencyEvaluator

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# --- Masking ---
# Redacts credential-shaped strings and well-known secret-bearing keys before
# anything reaches Langfuse. Scoped deliberately narrow: this catches API keys/
# tokens that must never leave the process regardless of policy. It does NOT
# redact user message content, names, or trauma/ACE-score context -- deciding
# what counts as sensitive there is a product/compliance call, not implied by
# the code, and is a documented open question in the observability plan.
_SECRET_KEY_NAMES = {"api_key", "apikey", "secret_key", "secret", "authorization", "password", "token"}
_SECRET_VALUE_PATTERN = re.compile(
    r"(sk-[A-Za-z0-9_-]{16,}|pk-lf-[A-Za-z0-9-]{8,}|sk-lf-[A-Za-z0-9-]{8,}|AIza[0-9A-Za-z_-]{35}|Bearer\s+[A-Za-z0-9._-]+)"
)


def _mask_value(value: Any) -> Any:
    if isinstance(value, str):
        return _SECRET_VALUE_PATTERN.sub("[REDACTED]", value)
    if isinstance(value, dict):
        return {
            k: ("[REDACTED]" if isinstance(k, str) and k.lower() in _SECRET_KEY_NAMES else _mask_value(v))
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        masked = [_mask_value(v) for v in value]
        return type(value)(masked) if isinstance(value, tuple) else masked
    return value


def mask_secrets(*, data: Any, **_kwargs: Any) -> Any:
    """Langfuse `mask` callback -- see Langfuse(mask=mask_secrets) below."""
    try:
        return _mask_value(data)
    except Exception as e:
        logger.debug(f"Masking failed, passing data through unmasked (non-fatal): {e}")
        return data

# --- Langfuse Client Implementation ---

# Conditional import for Langfuse
try:
    from langfuse import Langfuse, observe, propagate_attributes
    LANGFUSE_AVAILABLE = True
    logger.info("Langfuse package imported successfully (v3.11.0+)")
except ImportError as e:
    # Also hit when an older langfuse is installed that lacks propagate_attributes
    # (added in 3.9.0) -- tracing then silently degrades to no-ops, so say so loudly.
    logger.warning(f"Langfuse not available: {e}. Install with: pip install 'langfuse>=3.11.0,<5'")
    LANGFUSE_AVAILABLE = False
    Langfuse = None
    # Create no-op decorator
    def observe(name=None, **kwargs):
        def decorator(func):
            return func
        return decorator

    # No-op context manager fallback for trace-attribute propagation
    @contextmanager
    def propagate_attributes(**kwargs):
        yield

# Global singleton client. `_langfuse_client` is registered even when tracing is
# disabled/unconfigured -- this is deliberate: the Langfuse SDK's own `get_client()`
# (used internally by `@observe` and other module-level helpers) reuses whatever
# client is already registered for the process, or lazily constructs its own
# ad-hoc client from raw env vars on first use otherwise. Always registering one
# client up front avoids ending up with two disconnected clients (ours, silently
# disabled, plus a second one auto-created by the SDK that still tries to export
# spans and fails). `_tracing_active` is the real "is tracing on" flag.
_langfuse_client: Optional[Langfuse] = None
_langfuse_initialized: bool = False
_tracing_active: bool = False


def initialize_langfuse_client() -> bool:
    """
    Initialize Langfuse client singleton for tracing (Langfuse 3.11.0+).

    Always registers a single process-wide client (see note above), even when
    tracing ends up disabled, so no other Langfuse SDK entry point lazily
    creates a second, unconfigured client. Returns whether tracing is active.
    """
    global _langfuse_client, _langfuse_initialized, _tracing_active

    if _langfuse_initialized:
        return _tracing_active

    _langfuse_initialized = True

    if not LANGFUSE_AVAILABLE:
        return False

    tracing_requested = os.getenv("LANGFUSE_TRACING_ENABLED", "false").lower() == "true"
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST", "https://us.cloud.langfuse.com")
    has_credentials = bool(public_key and secret_key)
    should_trace = tracing_requested and has_credentials

    try:
        # NOTE: the SDK caches one client per public_key for the life of the
        # process (LangfuseResourceManager.__new__ returns the cached instance
        # on any later construction with the same public_key, ignoring new
        # kwargs). So `tracing_enabled` must be decided *before* this call --
        # it cannot be flipped after the fact by reconstructing the client.
        _langfuse_client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
            release=os.getenv("LANGFUSE_RELEASE"),
            sample_rate=float(os.getenv("LANGFUSE_SAMPLE_RATE", "1.0")),
            tracing_enabled=should_trace,
            mask=mask_secrets,
        )
    except Exception as e:
        logger.error(f"Langfuse client construction failed: {e}")
        _langfuse_client = None
        return False

    if not tracing_requested:
        logger.info("Langfuse tracing disabled (LANGFUSE_TRACING_ENABLED is not 'true')")
        return False

    if not has_credentials:
        logger.warning("Langfuse tracing requested but LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY are not set")
        return False

    # Diagnostic only -- does NOT gate tracing_enabled (already fixed above).
    try:
        if _langfuse_client.auth_check():
            logger.info(f"Langfuse client authenticated (host: {host})")
        else:
            logger.error(
                "Langfuse authentication check failed -- traces will still be sent but exports "
                "will likely fail. Verify LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY/LANGFUSE_HOST."
            )
    except Exception as e:
        logger.error(
            f"Langfuse authentication check errored ({e}) -- traces will still be sent but exports "
            "will likely fail. Verify LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY/LANGFUSE_HOST."
        )

    _tracing_active = True
    return True


def get_langfuse_client() -> Optional[Langfuse]:
    return _langfuse_client


def is_langfuse_enabled() -> bool:
    return _tracing_active


def flush_langfuse():
    global _langfuse_client
    if _langfuse_client is not None:
        try:
            _langfuse_client.flush()
        except Exception as e:
            logger.error(f"Error flushing Langfuse events: {e}")


def reset_langfuse_client():
    global _langfuse_client, _langfuse_initialized, _tracing_active
    if _langfuse_client is not None:
        try:
            _langfuse_client.flush()
        except Exception:
            pass
    _langfuse_client = None
    _langfuse_initialized = False
    _tracing_active = False


# --- Fail-open tracing helpers ---
# These wrap the raw Langfuse client calls so a Langfuse outage, malformed
# attribute, or missing client never breaks the request path they're called from.

def update_current_span(**kwargs) -> None:
    """Update the currently active Langfuse span/trace. No-op if tracing is unavailable."""
    if _langfuse_client is None:
        return
    try:
        _langfuse_client.update_current_span(**kwargs)
    except Exception as e:
        logger.debug(f"Langfuse update_current_span failed (non-fatal): {e}")


def update_current_generation(**kwargs) -> None:
    """Update the currently active Langfuse generation. No-op if tracing is unavailable."""
    if _langfuse_client is None:
        return
    try:
        _langfuse_client.update_current_generation(**kwargs)
    except Exception as e:
        logger.debug(f"Langfuse update_current_generation failed (non-fatal): {e}")


def score_current_trace(**kwargs) -> None:
    """Attach a numeric/categorical score to the current trace. No-op if tracing is unavailable."""
    if _langfuse_client is None:
        return
    try:
        _langfuse_client.score_current_trace(**kwargs)
    except Exception as e:
        logger.debug(f"Langfuse score_current_trace failed (non-fatal): {e}")


@contextmanager
def start_span(name: str, as_type: str = "span", **kwargs):
    """Fail-open wrapper around Langfuse's start_as_current_observation().

    Yields the Langfuse observation object, or None if tracing is unavailable
    or fails to start. Never raises into the caller.
    """
    if _langfuse_client is None:
        yield None
        return

    try:
        cm = _langfuse_client.start_as_current_observation(name=name, as_type=as_type, **kwargs)
        observation = cm.__enter__()
    except Exception as e:
        logger.debug(f"Langfuse start_span '{name}' failed to start (non-fatal): {e}")
        yield None
        return

    try:
        yield observation
    except BaseException:
        cm.__exit__(*sys.exc_info())
        raise
    else:
        cm.__exit__(None, None, None)


@contextmanager
def safe_propagate_attributes(**kwargs):
    """Fail-open wrapper around propagate_attributes(): never breaks the caller's request."""
    if _langfuse_client is None:
        yield
        return

    try:
        cm = propagate_attributes(**kwargs)
        cm.__enter__()
    except Exception as e:
        logger.debug(f"Langfuse propagate_attributes failed to start (non-fatal): {e}")
        yield
        return

    try:
        yield
    except BaseException:
        cm.__exit__(*sys.exc_info())
        raise
    else:
        cm.__exit__(None, None, None)

# --- End Langfuse Client Implementation ---


class EvaluationManager:
    """
    Manages and coordinates evaluators with LangFuse observability
    Replaces the old EvaluationRegistry
    """

    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.latency_evaluator = LatencyEvaluator()
        logger.info("Evaluation Manager initialized")

    @observe(name="run_full_evaluation")
    async def evaluate_response(
        self,
        user_message: str,
        bot_message: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        context = context or {}
        
        try:
            # Run latency evaluation
            latency_result = None
            if start_time is not None and end_time is not None:
                latency_result = await self.latency_evaluator.evaluate(start_time, end_time, context)
            
            results = {}
            if latency_result:
                results["latency"] = latency_result
            
            # Calculate aggregate scores
            aggregate_scores = self._calculate_aggregate_scores(results)
            results["aggregate_scores"] = aggregate_scores
            
            results["evaluation_metadata"] = {
                "timestamp": datetime.now().isoformat(),
                "evaluators_run": 1 if latency_result else 0,
                "context": context
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {"error": str(e), "success": False}

    def _calculate_aggregate_scores(self, results: Dict[str, Any]) -> Dict[str, Any]:
        score_breakdown = {}
        if "latency" in results and isinstance(results["latency"], dict):
            if "performance_score" in results["latency"]:
                score_breakdown["performance_score"] = results["latency"]["performance_score"]
        
        overall_score = score_breakdown.get("performance_score", 0.0)
        
        return {
            "overall_quality_score": round(overall_score, 3),
            "score_breakdown": score_breakdown,
            "quality_level": self._get_quality_level(overall_score)
        }

    def _get_quality_level(self, score: float) -> str:
        if score >= 0.9: return "excellent"
        elif score >= 0.75: return "good"
        elif score >= 0.6: return "acceptable"
        else: return "needs_improvement"

    def get_latency_statistics(self) -> Dict[str, Any]:
        return self.latency_evaluator.get_summary_statistics()

    def reset_latency_statistics(self):
        self.latency_evaluator.reset_statistics()


# Global state
_evaluation_manager: Optional[EvaluationManager] = None

def initialize_evaluation_manager(llm_client=None) -> EvaluationManager:
    global _evaluation_manager
    if _evaluation_manager is None:
        _evaluation_manager = EvaluationManager(llm_client=llm_client)
    return _evaluation_manager

def get_evaluation_manager() -> EvaluationManager:
    global _evaluation_manager
    if _evaluation_manager is None:
        _evaluation_manager = EvaluationManager()
    return _evaluation_manager

# Convenience wrapper
async def evaluate_response(
    user_message: str,
    bot_message: str,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    context: Optional[Dict[str, Any]] = None,
    llm_client=None,
    enable_tracing: bool = False
) -> Dict[str, Any]:
    if enable_tracing and not is_langfuse_enabled():
        initialize_langfuse_client()
    
    manager = get_evaluation_manager()
    if llm_client:
        manager.llm_client = llm_client
        
    return await manager.evaluate_response(
        user_message, bot_message, start_time, end_time, context
    )

def get_latency_statistics() -> Dict[str, Any]:
    return get_evaluation_manager().get_latency_statistics()

def reset_latency_statistics():
    get_evaluation_manager().reset_latency_statistics()

# Export main functions
__all__ = [
    "evaluate_response",
    "get_evaluation_manager",
    "initialize_evaluation_manager",
    "get_latency_statistics",
    "reset_latency_statistics",
    "initialize_langfuse_client",
    "get_langfuse_client",
    "is_langfuse_enabled",
    "flush_langfuse",
    "reset_langfuse_client",
    "LANGFUSE_AVAILABLE",
    "observe",
    "propagate_attributes",
    "safe_propagate_attributes",
    "start_span",
    "update_current_span",
    "update_current_generation",
    "score_current_trace",
]
