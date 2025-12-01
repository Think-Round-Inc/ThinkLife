"""
ThinkLife Evaluation System with LangFuse Observability

Comprehensive evaluation framework for AI responses with:
- Trauma-informed empathy assessment
- Trigger detection and safety evaluation
- Crisis handling evaluation
- Response latency monitoring
- Language accessibility assessment
- LangFuse integration for tracing and monitoring

Usage:
    from evaluation.evaluation_manager import evaluate_response, get_evaluation_manager
    
    # Run full evaluation
    results = await evaluate_response(
        user_message="I'm feeling anxious",
        bot_message="I hear that you're feeling anxious...",
        start_time=start_timestamp,
        end_time=end_timestamp,
        context={"user_id": "user_123", "session_id": "session_456"}
    )
    
    # Access individual evaluators via registry
    registry = get_evaluation_manager()  # Returns EvaluationRegistry
    empathy_result = await registry.empathy_evaluator.evaluate(user_msg, bot_msg)
"""

import logging
import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Conditional import for LangFuse
try:
    from langfuse import Langfuse
    LANGFUSE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"LangFuse not available: {e}. Continuing without observability.")
    LANGFUSE_AVAILABLE = False
    Langfuse = None

from evaluators import EvaluationRegistry

logger = logging.getLogger(__name__)

# Initialize LangFuse client
langfuse_client = None
if LANGFUSE_AVAILABLE:
    try:
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        
        if not public_key or not secret_key:
            logger.warning("LangFuse credentials not found in environment variables. Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY.")
        else:
            langfuse_client = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host=host
            )
            logger.info(f"LangFuse client initialized successfully (host: {host})")
    except Exception as e:
        logger.error(f"LangFuse initialization failed: {e}. Continuing without LangFuse.", exc_info=True)
        langfuse_client = None
else:
    logger.info("LangFuse not available - continuing without observability")

# Global evaluation registry instance
_evaluation_registry: Optional[EvaluationRegistry] = None


def initialize_evaluation_manager(llm_client=None) -> EvaluationRegistry:
    """
    Initialize the global evaluation registry
    
    Args:
        llm_client: Optional LLM client for evaluators (Gemini or OpenAI)
        
    Returns:
        Initialized EvaluationRegistry instance
    """
    global _evaluation_registry
    
    if _evaluation_registry is None:
        _evaluation_registry = EvaluationRegistry(llm_client=llm_client)
        logger.info("Global evaluation registry initialized")
    
    return _evaluation_registry


def get_evaluation_manager() -> EvaluationRegistry:
    """
    Get the global evaluation registry instance
    
    Returns:
        EvaluationRegistry instance (creates one if not exists)
    """
    global _evaluation_registry
    
    if _evaluation_registry is None:
        _evaluation_registry = EvaluationRegistry()
        logger.info("Created new evaluation registry instance")
    
    return _evaluation_registry


async def evaluate_response(
    user_message: str,
    bot_message: str,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    context: Optional[Dict[str, Any]] = None,
    llm_client=None
) -> Dict[str, Any]:
    """
    Run comprehensive evaluation on a bot response
    
    This is the main entry point for running all evaluations with LangFuse tracing.
    
    Args:
        user_message: User's input message
        bot_message: Bot's response message
        start_time: Optional start timestamp for latency evaluation
        end_time: Optional end timestamp for latency evaluation
        context: Additional context (user_id, session_id, etc.)
        llm_client: Optional LLM client for evaluators
        
    Returns:
        Dict containing:
        - empathy: Empathy assessment results
        - trigger: Trigger detection results
        - crisis: Crisis/safety evaluation results
        - accessibility: Language accessibility results
        - latency: Performance metrics (if timestamps provided)
        - aggregate_scores: Overall quality scores
        - evaluation_metadata: Evaluation metadata
    """
    # Get or create evaluation manager
    manager = get_evaluation_manager()
    
    # If LLM client provided, update the manager's client
    if llm_client:
        manager.llm_client = llm_client
        manager.empathy_evaluator.llm_client = llm_client
        manager.trigger_evaluator.llm_client = llm_client
        manager.crisis_evaluator.llm_client = llm_client
        manager.accessibility_evaluator.llm_client = llm_client
    
    # Run evaluation
    results = await manager.evaluate_response(
        user_message=user_message,
        bot_message=bot_message,
        start_time=start_time,
        end_time=end_time,
        context=context
    )
    
    return results


# Backward compatibility aliases
async def run_evaluation(
    user_message: str,
    bot_message: str,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None
) -> Dict[str, Any]:
    """
    Legacy function for backward compatibility
    
    Use evaluate_response() for new code
    """
    return await evaluate_response(
        user_message=user_message,
        bot_message=bot_message,
        start_time=start_time,
        end_time=end_time
    )


def get_latency_statistics() -> Dict[str, Any]:
    """
    Get summary statistics for latency evaluations
    
    Returns:
        Dict with comprehensive latency statistics
    """
    manager = get_evaluation_manager()
    return manager.get_latency_statistics()


def reset_latency_statistics():
    """Reset all latency statistics"""
    manager = get_evaluation_manager()
    manager.reset_latency_statistics()
    logger.info("Latency statistics reset")


# Export main functions
__all__ = [
    "evaluate_response",
    "run_evaluation",
    "get_evaluation_manager",
    "initialize_evaluation_manager",
    "get_latency_statistics",
    "reset_latency_statistics",
    "langfuse_client"
]
