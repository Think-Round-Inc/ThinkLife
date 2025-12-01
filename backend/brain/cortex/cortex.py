"""
Cortex - Generalized AI orchestration system
"""

import asyncio
import logging
import os
import time
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from dotenv import load_dotenv

# LangFuse tracing
try:
    from langfuse.decorators import observe, langfuse_context
    LANGFUSE_AVAILABLE = True
except (ImportError, Exception) as e:
    LANGFUSE_AVAILABLE = False
    def observe(name=None, **kwargs):
        def decorator(func):
            return func
        return decorator
    class LangfuseContext:
        def update_current_trace(self, **kwargs):
            pass
        def update_current_observation(self, **kwargs):
            pass
    langfuse_context = LangfuseContext()

from ..specs import (
    BrainRequest, BrainResponse, IAgent,
    BrainConfig, BrainAnalytics, PluginInfo, WorkflowExecution, 
    DataSourceInfo, ApplicationType, PluginStatus, WorkflowType,
    AgentExecutionSpec, DataSourceSpec, ProviderSpec, ToolSpec, ProcessingSpec,
    DataSourceType, ExecutionPlan, WorkflowExecution, WorkflowStep
)
from .workflow_engine import WorkflowEngine, get_workflow_engine
from .reasoning_engine import ReasoningEngine, get_reasoning_engine
from ..data_sources import get_data_source_registry
from ..tools import get_tool_registry
from ..providers import get_provider_registry
from ..guardrails import SecurityManager

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class CortexFlow:
    """
    CortexFlow - Two-Phase Execution System
    
    Phase 1: Planning - Agent specs - Optional reasoning
    Phase 2: Execution - Workflow engine executes the plan
    
    Key Features:
    - Two-phase execution (planning - execution)
    - Optional reasoning engine for optimization
    - Plugin-based agent system
    - Workflow engine for standardized execution
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls, config=None):
        """Singleton pattern implementation"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, config: Optional[Union[BrainConfig, Dict[str, Any]]] = None):
        """Initialize the Cortex"""
        if self._initialized:
            return
        
        # Store config
        if isinstance(config, BrainConfig):
            self.config = config
        elif isinstance(config, dict):
            self.config = BrainConfig(**config)
        else:
            self.config = BrainConfig()

        # Core components  
        self.reasoning_engine: Optional[ReasoningEngine] = None
        self.workflow_engine: Optional[WorkflowEngine] = None
        self.provider_registry = get_provider_registry()
        self.tool_registry = get_tool_registry()
        self.data_source_registry = get_data_source_registry()
        self.security_manager: SecurityManager = SecurityManager(self.config.security)
        
        # Analytics and monitoring
        self.analytics = BrainAnalytics()
        self.start_time = datetime.now()
        self.active_executions: Dict[str, WorkflowExecution] = {}
        
        logger.info("Cortex initialized")

    
    async def initialize(self) -> None:
        """Initialize all Cortex components"""
        if self._initialized:
            return
        
        logger.info("Initializing Cortex components...")
        
        try:
            # Initialize reasoning engine
            self.reasoning_engine = get_reasoning_engine()
            await self.reasoning_engine.initialize()
            
            # Initialize workflow engine 
            self.workflow_engine = get_workflow_engine()
            await self.workflow_engine.initialize()
            
            # Initialize guardrails
            self.security_manager = SecurityManager(self.config.security)
            
            self._initialized = True
            logger.info("Cortex initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Cortex: {str(e)}")
            raise
    
    
    async def _ensure_initialized(self):
        """Ensure the Cortex is initialized before processing requests"""
        if not self._initialized:
            await self.initialize()
    
    @observe(name="cortex_process_agent_request")
    async def process_agent_request(
        self,
        agent_specs: AgentExecutionSpec,
        request: BrainRequest
    ) -> Dict[str, Any]:
        """
        Process agent request with agent specs and LLM request
        
        Flow:
        1. Check availability of providers, tools, and data sources via registries
        2. Apply guardrails (rate limiting, content filtering)
        3. Check if reasoning is enabled
        4. If reasoning enabled - invoke reasoning engine (create_execution_plan)
        5. If reasoning not enabled - invoke execute_agent_request (workflow engine)
        6. Return response from workflow engine
        """
        start_time = time.time()
        
        # Update observation metadata (not trace - this is a nested observation)
        langfuse_context.update_current_observation(
            metadata={
                "agent_type": agent_specs.agent_type if hasattr(agent_specs, 'agent_type') else "unknown",
                "provider": agent_specs.provider.provider_type if agent_specs.provider else None,
                "model": agent_specs.provider.model if agent_specs.provider else None,
                "tools_count": len(agent_specs.tools) if agent_specs.tools else 0,
                "data_sources_count": len(agent_specs.data_sources) if agent_specs.data_sources else 0
            }
        )
        
        try:
            # Ensure initialization
            await self._ensure_initialized()
            
            logger.info("Processing agent request - checking availability")
            
            # 1. Check provider availability
            if agent_specs.provider:
                provider_type = agent_specs.provider.provider_type
                model = agent_specs.provider.model
                is_valid, errors, info = self.provider_registry.check_provider_and_model(
                    provider_type, model
                )
                if not is_valid:
                    logger.error(f"Provider validation failed: {errors}")
                    return {
                        "success": False,
                        "content": f"Provider validation failed: {', '.join(errors)}",
                        "metadata": {"errors": errors},
                        "processing_time": time.time() - start_time
                    }
                logger.info(f"Provider validated: {provider_type} - {info.get('model')}")
            
            # 2. Check tools availability
            for tool_spec in agent_specs.tools:
                if tool_spec.enabled:
                    tool_name = tool_spec.name
                    if not self.tool_registry.check_tool_available(tool_name):
                        logger.warning(f"Tool '{tool_name}' not available")
                        return {
                            "success": False,
                            "content": f"Tool '{tool_name}' is not available",
                            "metadata": {"error": f"Tool '{tool_name}' not found"},
                            "processing_time": time.time() - start_time
                        }
                    logger.info(f"Tool validated: {tool_name}")
            
            # 3. Check data sources availability
            for data_source_spec in agent_specs.data_sources:
                if data_source_spec.enabled:
                    # Get source name from enum value
                    source_name = data_source_spec.source_type.value if hasattr(data_source_spec.source_type, 'value') else str(data_source_spec.source_type)
                    # Check if data source is available (currently only vector_db is supported)
                    if source_name == "vector_db":
                        if not self.data_source_registry.check_data_source_available("vector_db"):
                            logger.warning(f"Data source 'vector_db' not available")
                            return {
                                "success": False,
                                "content": f"Data source 'vector_db' is not available",
                                "metadata": {"error": "Data source 'vector_db' not found"},
                                "processing_time": time.time() - start_time
                            }
                        logger.info(f"Data source validated: {source_name}")
                    # Other data source types (conversation_history, etc.) are handled by agents
                    elif source_name not in ["conversation_history"]:
                        logger.info(f"Data source '{source_name}' - validation skipped (handled by agent)")
            
            # 4. Apply guardrails (rate limiting)
            user_id = request.user_context.user_id
            if not self.security_manager.check_rate_limit(user_id):
                logger.warning(f"Rate limit exceeded for user: {user_id}")
                return {
                    "success": False,
                    "content": "Rate limit exceeded. Please try again later.",
                    "metadata": {"error": "Rate limit exceeded"},
                    "processing_time": time.time() - start_time
                }
            
            # 5. Check if reasoning is enabled
            reasoning_enabled = False
            if agent_specs.processing:
                execution_strategy = agent_specs.processing.execution_strategy
                reasoning_enabled = execution_strategy in ["reasoned", "adaptive"]
            
            # 6. Execute based on reasoning flag
            if reasoning_enabled:
                # Use reasoning engine (two-phase: plan then execute)
                logger.info("Reasoning enabled - creating execution plan")
                plan = await self.create_execution_plan(agent_specs, request)
                result = await self.execute_plan(plan, request, None)  # agent not needed for execution
            else:
                # Direct execution via workflow engine
                logger.info("Reasoning disabled - executing directly via workflow engine")
                # Create a direct plan without reasoning
                plan = ExecutionPlan(
                    original_specs=agent_specs,
                    optimized_specs=agent_specs,
                    reasoning_applied=False,
                    confidence=1.0,
                    reasoning_notes={"strategy": "direct"},
                    estimated_cost=self._estimate_cost(agent_specs),
                    estimated_latency=self._estimate_latency(agent_specs)
                )
                result = await self.execute_agent_request(plan, request)
            
            # 7. Return response
            result["processing_time"] = time.time() - start_time
            return result
            
        except Exception as e:
            logger.error(f"Error processing agent request: {str(e)}")
            return {
                "success": False,
                "content": "An error occurred processing your request.",
                "metadata": {"error": str(e)},
                "processing_time": time.time() - start_time
            }

    @observe(name="cortex_create_execution_plan")
    async def create_execution_plan(
        self,
        agent_specs: AgentExecutionSpec,
        request: BrainRequest
    ) -> ExecutionPlan:
        """
        PHASE 1: PLANNING
        
        Create an execution plan from agent specifications.
        Optionally uses reasoning engine to optimize the plan.
        
        Returns:
            ExecutionPlan with optimized specs and metadata
        """
        logger.info(f"Planning phase started - strategy: {agent_specs.processing.execution_strategy}")
        
        strategy = agent_specs.processing.execution_strategy
        
        if strategy == "direct":
            # Skip reasoning - use agent specs directly
            return self._create_direct_plan(agent_specs)
        
        elif strategy == "reasoned":
            # Always use reasoning
            return await self._create_reasoned_plan(agent_specs, request)
        
        elif strategy == "adaptive":
            # Use reasoning, but fallback to agent specs if confidence low
            return await self._create_adaptive_plan(agent_specs, request)
        
        else:
            logger.warning(f"Unknown execution strategy: {strategy}, using direct")
            return self._create_direct_plan(agent_specs)
    
    def _create_direct_plan(self, agent_specs: AgentExecutionSpec) -> ExecutionPlan:
        """Create execution plan without reasoning"""
        logger.info("Creating direct execution plan (no reasoning)")
        
        return ExecutionPlan(
            original_specs=agent_specs,
            optimized_specs=agent_specs,
            reasoning_applied=False,
            confidence=1.0,
            reasoning_notes={"strategy": "direct", "message": "Skipped reasoning as requested"},
            estimated_cost=self._estimate_cost(agent_specs),
            estimated_latency=self._estimate_latency(agent_specs)
        )
    
    async def _create_reasoned_plan(
        self,
        agent_specs: AgentExecutionSpec,
        request: BrainRequest
    ) -> ExecutionPlan:
        """Create execution plan with reasoning (always optimize)"""
        logger.info("Creating reasoned execution plan (with reasoning)")
        
        try:
            # Get reasoning suggestions
            reasoning_result = await self.reasoning_engine.optimize_execution_specs(
                original_specs=agent_specs,
                request=request,
                context=self._get_execution_context()
            )
            
            optimized_specs = reasoning_result.get("optimized_specs", agent_specs)
            confidence = reasoning_result.get("confidence", 0.8)
            
            return ExecutionPlan(
                original_specs=agent_specs,
                optimized_specs=optimized_specs,
                reasoning_applied=True,
                confidence=confidence,
                reasoning_notes=reasoning_result.get("notes", {}),
                estimated_cost=reasoning_result.get("estimated_cost", 0.0),
                estimated_latency=reasoning_result.get("estimated_latency", 0.0)
            )
        
        except Exception as e:
            logger.error(f"Reasoning failed, falling back to direct plan: {e}")
            return self._create_direct_plan(agent_specs)
    
    async def _create_adaptive_plan(
        self,
        agent_specs: AgentExecutionSpec,
        request: BrainRequest
    ) -> ExecutionPlan:
        """Create adaptive plan with reasoning fallback"""
        logger.info("Creating adaptive execution plan (reasoning with confidence threshold)")
        
        try:
            # Get reasoning suggestions
            reasoning_result = await self.reasoning_engine.optimize_execution_specs(
                original_specs=agent_specs,
                request=request,
                context=self._get_execution_context()
            )
            
            confidence = reasoning_result.get("confidence", 0.0)
            threshold = agent_specs.processing.reasoning_threshold
            
            if confidence >= threshold:
                # High confidence - use reasoning suggestions
                logger.info(f"Using reasoning suggestions (confidence: {confidence} >= {threshold})")
                
                return ExecutionPlan(
                    original_specs=agent_specs,
                    optimized_specs=reasoning_result["optimized_specs"],
                    reasoning_applied=True,
                    confidence=confidence,
                    reasoning_notes=reasoning_result.get("notes", {}),
                    estimated_cost=reasoning_result.get("estimated_cost", 0.0),
                    estimated_latency=reasoning_result.get("estimated_latency", 0.0)
                )
            else:
                # Low confidence - stick with agent specs
                logger.info(f"Reasoning confidence too low ({confidence} < {threshold}), using agent specs")
                
                return ExecutionPlan(
                    original_specs=agent_specs,
                    optimized_specs=agent_specs,
                    reasoning_applied=False,
                    confidence=1.0,
                    reasoning_notes={
                        "skipped": f"confidence {confidence} < {threshold}",
                        "reasoning_suggestions": reasoning_result.get("notes", {})
                    },
                    estimated_cost=self._estimate_cost(agent_specs),
                    estimated_latency=self._estimate_latency(agent_specs)
                )
        
        except Exception as e:
            logger.error(f"Adaptive planning failed, falling back to direct plan: {e}")
            return self._create_direct_plan(agent_specs)
    
    def _get_execution_context(self) -> Dict[str, Any]:
        """Get current execution context for reasoning"""
        # Get available providers from PROVIDER_CONFIGS
        from ..providers.provider_registry import PROVIDER_CONFIGS
        from ..tools.tool_registry import TOOLS
        from ..data_sources.data_source_registry import DATA_SOURCES
        
        return {
            "available_providers": list(PROVIDER_CONFIGS.keys()),
            "available_tools": TOOLS,
            "data_sources": DATA_SOURCES,
            "system_load": {
                "total_requests": self.analytics.total_requests,
                "average_response_time": self.analytics.average_response_time
            }
        }
    
    def _estimate_cost(self, specs: AgentExecutionSpec) -> float:
        """Estimate execution cost based on specs"""
        # Placeholder - would implement actual cost estimation
        cost = 0.0
        
        if specs.provider:
            # Rough cost estimate based on tokens
            estimated_tokens = specs.provider.max_tokens if hasattr(specs.provider, 'max_tokens') else 2000
            cost += estimated_tokens * 0.00001  # $0.01 per 1K tokens (placeholder)
        
        # Add data source costs
        cost += len(specs.data_sources) * 0.001
        
        # Add tool costs
        cost += len(specs.tools) * 0.005
        
        return round(cost, 4)
    
    def _estimate_latency(self, specs: AgentExecutionSpec) -> float:
        """Estimate execution latency based on specs"""
        # Placeholder - would implement actual latency estimation
        latency = 0.5  # Base latency
        
        # Add data source latency
        latency += len(specs.data_sources) * 0.3
        
        # Add tool latency
        latency += len(specs.tools) * 0.5
        
        # Add provider latency
        if specs.provider:
            latency += 1.5
        
        return round(latency, 2)
    
    @observe(name="cortex_execute_plan")
    async def execute_plan(
        self,
        plan: ExecutionPlan,
        request: BrainRequest,
        agent: Optional[IAgent] = None
    ) -> Dict[str, Any]:
        """
        PHASE 2: EXECUTION
        
        Execute the execution plan via workflow engine.
        """
        logger.info(f"Execution phase started - reasoning_applied: {plan.reasoning_applied}")
        
        try:
            # Execute via workflow engine
            result = await self.workflow_engine.execute_plan(
                plan=plan,
                request=request
            )
            
            # Add planning metadata to result
            result["execution_plan"] = plan.to_dict()
            
            return result
            
        except Exception as e:
            logger.error(f"Execution failed: {e}")
            raise
    
    @observe(name="cortex_execute_agent_request")
    async def execute_agent_request(
        self,
        plan: ExecutionPlan,
        request: BrainRequest
    ) -> Dict[str, Any]:
        """
        Execute the agent request according to the execution plan.
        """
        return await self.execute_plan(plan, request, None)
    
    async def shutdown(self):
        """Shutdown CortexFlow and all its components"""
        logger.info("Shutting down CortexFlow...")
        
        try:
            # Shutdown workflow engine
            if self.workflow_engine:
                await self.workflow_engine.shutdown()
            
            # Shutdown reasoning engine (if it has a shutdown method)
            if self.reasoning_engine and hasattr(self.reasoning_engine, 'shutdown'):
                await self.reasoning_engine.shutdown()
            
            # Reset initialization state
            self._initialized = False
            
            logger.info("CortexFlow shutdown complete")
        except Exception as e:
            logger.error(f"Error during CortexFlow shutdown: {e}")


