from typing import Any, Dict, Optional
import logging

# Try to import AgentExecutionSpec, but handle circular imports if necessary
try:
    from brain.specs import AgentExecutionSpec
except ImportError:
    # Fallback for type hinting if imported in a context where brain is not ready
    AgentExecutionSpec = Any

# Conditional Langfuse import (Langfuse 3.7.0+)
try:
    from langfuse import observe
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    # Create no-op decorator
    def observe(name=None, **kwargs):
        def decorator(func):
            return func
        return decorator

logger = logging.getLogger(__name__)

class CostEstimator:
    """Estimates the financial cost of an execution"""

    # Approximate USD price per 1M tokens as (input, output). For the LOCAL
    # /api/brain/analytics rollup only -- Langfuse Cloud's own model pricing
    # (Settings -> Models) is the source of truth for the Langfuse dashboard
    # and is computed server-side from usage_details independently of this
    # table. Update these if provider list pricing changes.
    _MODEL_PRICING_PER_1M_TOKENS = {
        "gpt-4o-mini": (0.15, 0.60),
        "gpt-4o": (2.50, 10.00),
        "gemini-1.5-flash": (0.075, 0.30),
        "gemini-2.0-flash": (0.10, 0.40),
    }

    @staticmethod
    def actual_cost(model: Optional[str], usage: Optional[Dict[str, int]]) -> Optional[float]:
        """Post-call actual cost computed from real token usage. Returns None
        (not 0.0) when the model isn't in the local pricing table, so callers
        can distinguish "$0 spent" from "cost unknown"."""
        if not model or not usage:
            return None

        pricing = CostEstimator._MODEL_PRICING_PER_1M_TOKENS.get(model)
        if not pricing:
            return None

        input_price, output_price = pricing
        prompt_tokens = usage.get("prompt_tokens", 0) or 0
        completion_tokens = usage.get("completion_tokens", 0) or 0

        cost = (prompt_tokens / 1_000_000) * input_price + (completion_tokens / 1_000_000) * output_price
        return round(cost, 6)

    @staticmethod
    @observe(name="estimate_cost")
    def estimate(specs: AgentExecutionSpec) -> float:
        """
        Estimate execution cost based on specs
        
        Args:
            specs: The agent execution specifications
            
        Returns:
            Estimated cost in USD
        """
        cost = 0.0
        
        try:
            if specs.provider:
                # Rough cost estimate based on tokens
                # Handle both object and dict access for flexibility
                if hasattr(specs.provider, 'max_tokens'):
                    max_tokens = specs.provider.max_tokens
                elif isinstance(specs.provider, dict):
                    max_tokens = specs.provider.get('max_tokens')
                else:
                    max_tokens = 2000
                
                estimated_tokens = max_tokens if max_tokens else 2000
                # Placeholder rate: $0.01 per 1K tokens
                cost += estimated_tokens * 0.00001
            
            # Add data source costs (e.g. vector DB read units)
            if specs.data_sources:
                cost += len(specs.data_sources) * 0.001
            
            # Add tool costs (e.g. API calls)
            if specs.tools:
                cost += len(specs.tools) * 0.005
            
            return round(cost, 4)
            
        except Exception as e:
            logger.warning(f"Cost estimation failed: {e}")
            return 0.0
