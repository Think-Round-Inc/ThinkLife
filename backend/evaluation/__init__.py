"""
ThinkLife Evaluation Package

Provides comprehensive evaluation and observability for AI responses with Langfuse integration (v3.x)
"""

from .evaluation_manager import (
    evaluate_response,
    get_evaluation_manager,
    initialize_evaluation_manager,
    get_latency_statistics,
    reset_latency_statistics,
    initialize_langfuse_client,
    is_langfuse_enabled,
    get_langfuse_client,
    flush_langfuse,
    reset_langfuse_client,
    LANGFUSE_AVAILABLE,
    observe,
    propagate_attributes,
    safe_propagate_attributes,
    start_span,
    update_current_span,
    update_current_generation,
    score_current_trace,
)

from .estimators import (
    CostEstimator,
    LatencyEstimator
)

__all__ = [
    # Evaluation functions
    "evaluate_response",
    "get_evaluation_manager",
    "initialize_evaluation_manager",
    "get_latency_statistics",
    "reset_latency_statistics",
    # Estimators
    "CostEstimator",
    "LatencyEstimator",
    # Langfuse client functions
    "initialize_langfuse_client",
    "is_langfuse_enabled",
    "get_langfuse_client",
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
