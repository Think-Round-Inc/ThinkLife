"""
Reasoning Engine
Uses LLMs to decide next steps, choose tools/data sources, refine plans
"""

import logging
from typing import Dict, Any, List, Optional

from ..specs import BrainRequest, ProviderSpec, ToolSpec, DataSourceSpec
from ..providers import create_provider, check_provider_spec_availability
from ..tools import get_tool_registry
from ..data_sources import get_data_source_registry

logger = logging.getLogger(__name__)


class ReasoningEngine:
    """
    LLM-powered reasoning engine that decides:
    - What tools to use
    - What data sources to query
    - How to refine plans
    - Next steps in execution
    """
    
    def __init__(self):
        self.tool_registry = get_tool_registry()
        self.data_source_registry = get_data_source_registry()
    
    async def initialize(self):
        """Initialize reasoning engine"""
        await self.tool_registry.initialize()
        await self.data_source_registry.initialize()
        logger.info("Reasoning Engine initialized")
    
    async def decide_next_step(
        self,
        request: BrainRequest,
        provider_spec: ProviderSpec,
        context: Dict[str, Any],
        execution_history: List[str]
    ) -> Dict[str, Any]:
        """
        Use LLM to decide next step based on:
        - Current request
        - Available tools/data sources
        - Execution history
        - Context data
        """
        # Build reasoning prompt
        reasoning_prompt = self._build_reasoning_prompt(
            request, context, execution_history
        )
        
        # Use LLM to reason
        provider = await self._create_provider(provider_spec)
        if not provider:
            return {"action": "error", "reason": "Provider unavailable"}
        
        try:
            response = await provider.generate(
                messages=[{"role": "user", "content": reasoning_prompt}]
            )
            
            # Parse LLM decision
            decision = self._parse_decision(response.get("content", ""))
            return decision
            
        except Exception as e:
            logger.error(f"Reasoning failed: {e}")
            return {"action": "error", "reason": str(e)}
    
    async def select_tools(
        self,
        request: BrainRequest,
        provider_spec: ProviderSpec,
        available_tools: List[str]
    ) -> List[ToolSpec]:
        """Use LLM to select relevant tools for the task"""
        tool_descriptions = self.tool_registry.get_tool_descriptions()
        
        prompt = f"""
Given the user request: "{request.message}"

Available tools:
{self._format_tools(tool_descriptions)}

Select the most relevant tools to use. Reply with tool names only, one per line.
"""
        
        provider = await self._create_provider(provider_spec)
        if not provider:
            return []
        
        try:
            response = await provider.generate(
                messages=[{"role": "user", "content": prompt}]
            )
            
            selected = self._parse_tool_selection(response.get("content", ""))
            return [ToolSpec(name=tool) for tool in selected if tool in available_tools]
            
        except Exception as e:
            logger.error(f"Tool selection failed: {e}")
            return []
    
    async def refine_plan(
        self,
        original_request: BrainRequest,
        provider_spec: ProviderSpec,
        results_so_far: Dict[str, Any],
        remaining_iterations: int
    ) -> Dict[str, Any]:
        """Use LLM to refine execution plan based on results"""
        prompt = f"""
Original request: "{original_request.message}"

Results so far:
{self._format_results(results_so_far)}

Remaining iterations: {remaining_iterations}

Should we:
1. Continue with current approach
2. Try different tools/data sources
3. Conclude with current results

Provide reasoning and recommendation.
"""
        
        provider = await self._create_provider(provider_spec)
        if not provider:
            return {"action": "continue", "reason": "Provider unavailable"}
        
        try:
            response = await provider.generate(
                messages=[{"role": "user", "content": prompt}]
            )
            
            return self._parse_plan_refinement(response.get("content", ""))
            
        except Exception as e:
            logger.error(f"Plan refinement failed: {e}")
            return {"action": "continue", "reason": str(e)}
    
    def _build_reasoning_prompt(
        self,
        request: BrainRequest,
        context: Dict[str, Any],
        history: List[str]
    ) -> str:
        """Build prompt for reasoning"""
        return f"""
Task: {request.message}

Context: {context}

Execution History:
{chr(10).join(history) if history else "None"}

Based on the above, what should be the next action?
"""
    
    def _parse_decision(self, llm_response: str) -> Dict[str, Any]:
        """Parse LLM decision from response"""
        response_lower = llm_response.lower()
        
        if "tool" in response_lower:
            return {"action": "use_tool", "reason": llm_response}
        elif "data" in response_lower or "search" in response_lower:
            return {"action": "query_data", "reason": llm_response}
        elif "complete" in response_lower or "done" in response_lower:
            return {"action": "complete", "reason": llm_response}
        else:
            return {"action": "continue", "reason": llm_response}
    
    def _parse_tool_selection(self, llm_response: str) -> List[str]:
        """Parse selected tools from LLM response"""
        lines = llm_response.strip().split("\n")
        tools = []
        for line in lines:
            tool = line.strip().strip("-*•").strip()
            if tool and not tool.startswith("#"):
                tools.append(tool)
        return tools
    
    def _parse_plan_refinement(self, llm_response: str) -> Dict[str, Any]:
        """Parse plan refinement from LLM response"""
        response_lower = llm_response.lower()
        
        if "different" in response_lower or "try" in response_lower:
            return {"action": "adjust", "reason": llm_response}
        elif "conclude" in response_lower or "finish" in response_lower:
            return {"action": "conclude", "reason": llm_response}
        else:
            return {"action": "continue", "reason": llm_response}
    
    def _format_tools(self, tool_descriptions: Dict[str, str]) -> str:
        """Format tool descriptions"""
        return "\n".join(f"- {name}: {desc}" for name, desc in tool_descriptions.items())
    
    def _format_results(self, results: Dict[str, Any]) -> str:
        """Format results for prompt"""
        return str(results)
    
    async def _create_provider(self, provider_spec: ProviderSpec):
        """Create provider instance"""
        # Validate provider spec
        is_valid, errors, _ = check_provider_spec_availability(provider_spec)
        if not is_valid:
            logger.error(f"Provider spec invalid: {errors}")
            return None
        
        # Create provider
        try:
            provider = create_provider(
                provider_spec.provider_type,
                {
                    "model": provider_spec.model,
                    "temperature": provider_spec.temperature,
                    "max_tokens": provider_spec.max_tokens,
                    **provider_spec.custom_params
                }
            )
            await provider.initialize()
            return provider
        except Exception as e:
            logger.error(f"Provider creation failed: {e}")
            return None
    
    async def optimize_execution_specs(
        self,
        original_specs: Any,  # AgentExecutionSpec
        request: Any,  # BrainRequest
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Optimize execution specs using LLM reasoning
        
        This is the main method for the two-phase architecture.
        It analyzes the agent's specs and suggests optimizations.
        
        Returns:
            {
                "optimized_specs": AgentExecutionSpec,
                "confidence": float (0.0-1.0),
                "notes": dict,
                "estimated_cost": float,
                "estimated_latency": float
            }
        """
        logger.info("Reasoning engine optimizing execution specs")
        
        try:
            # For high-level implementation, we'll do basic optimization
            # In production, this would use an LLM to analyze and suggest improvements
            
            optimized_specs = original_specs  # Start with original
            confidence = 0.75  # Default confidence
            notes = {}
            
            # Analyze provider selection
            if original_specs.provider:
                provider_notes = await self._analyze_provider(original_specs.provider, request, context)
                notes["provider"] = provider_notes
                
                # Example optimization: suggest cheaper provider for simple queries
                if len(request.message.split()) < 50 and provider_notes.get("can_use_cheaper"):
                    confidence = 0.85
                    notes["suggestion"] = "Simple query - cheaper model recommended"
            
            # Analyze data sources
            if original_specs.data_sources:
                datasource_notes = await self._analyze_data_sources(original_specs.data_sources, request)
                notes["data_sources"] = datasource_notes
            
            # Analyze tools
            if original_specs.tools:
                tool_notes = await self._analyze_tools(original_specs.tools, request)
                notes["tools"] = tool_notes
            
            # Estimate cost and latency
            estimated_cost = self._estimate_spec_cost(optimized_specs)
            estimated_latency = self._estimate_spec_latency(optimized_specs)
            
            return {
                "optimized_specs": optimized_specs,
                "confidence": confidence,
                "notes": notes,
                "estimated_cost": estimated_cost,
                "estimated_latency": estimated_latency
            }
        
        except Exception as e:
            logger.error(f"Error optimizing specs: {e}")
            # Return original specs with low confidence
            return {
                "optimized_specs": original_specs,
                "confidence": 0.0,
                "notes": {"error": str(e)},
                "estimated_cost": 0.0,
                "estimated_latency": 0.0
            }
    
    async def _analyze_provider(
        self,
        provider_spec: Any,
        request: Any,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze provider selection"""
        notes = {
            "original_provider": provider_spec.provider_type,
            "original_model": provider_spec.model if hasattr(provider_spec, 'model') else None,
            "can_use_cheaper": len(request.message.split()) < 50  # Simple heuristic
        }
        return notes
    
    async def _analyze_data_sources(
        self,
        data_sources: List[Any],
        request: Any
    ) -> Dict[str, Any]:
        """Analyze data source selection"""
        notes = {
            "count": len(data_sources),
            "types": [ds.source_type.value if hasattr(ds.source_type, 'value') else str(ds.source_type) for ds in data_sources]
        }
        return notes
    
    async def _analyze_tools(
        self,
        tools: List[Any],
        request: Any
    ) -> Dict[str, Any]:
        """Analyze tool selection"""
        notes = {
            "count": len(tools),
            "names": [tool.name for tool in tools if hasattr(tool, 'name')]
        }
        return notes
    
    def _estimate_spec_cost(self, specs: Any) -> float:
        """Estimate cost of executing specs"""
        cost = 0.0
        
        if specs.provider:
            # Rough estimate based on max_tokens
            tokens = specs.provider.max_tokens if hasattr(specs.provider, 'max_tokens') else 2000
            cost += tokens * 0.00001  # $0.01 per 1K tokens
        
        cost += len(specs.data_sources) * 0.001
        cost += len(specs.tools) * 0.005
        
        return round(cost, 4)
    
    def _estimate_spec_latency(self, specs: Any) -> float:
        """Estimate latency of executing specs"""
        latency = 0.5  # Base
        latency += len(specs.data_sources) * 0.3
        latency += len(specs.tools) * 0.5
        if specs.provider:
            latency += 1.5
        return round(latency, 2)
    
    async def shutdown(self):
        """Shutdown reasoning engine"""
        logger.info("Reasoning Engine shutdown")


# Singleton
_reasoning_engine_instance: Optional[ReasoningEngine] = None


def get_reasoning_engine() -> ReasoningEngine:
    """Get singleton ReasoningEngine instance"""
    global _reasoning_engine_instance
    if _reasoning_engine_instance is None:
        _reasoning_engine_instance = ReasoningEngine()
    return _reasoning_engine_instance

