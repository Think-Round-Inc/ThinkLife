"""
ThinkxLife Brain Core - Generalized AI orchestration system with plugin architecture
Main Brain system with backward compatibility for existing integrations
"""

import asyncio
import logging
import os
import time
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from dotenv import load_dotenv

from ..specs import (
    BrainRequest, BrainResponse, IAgent,
    BrainConfig, BrainAnalytics, PluginInfo, WorkflowExecution, 
    DataSourceInfo, ApplicationType, PluginStatus, WorkflowType,
    AgentExecutionSpec, DataSourceSpec, ProviderSpec, ToolSpec, ProcessingSpec,
    DataSourceType, ExecutionPlan
)
from .workflow_engine import WorkflowEngine, get_workflow_engine, WorkflowStep, WorkflowStatus
from .reasoning_engine import ReasoningEngine, get_reasoning_engine
from ..data_sources import get_data_source_registry
from ..guardrails import SecurityManager
from ..providers import create_provider, get_available_providers

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class CortexFlow:
    """
    CortexFlow - Two-Phase Execution System
    
    Phase 1: Planning - Agent specs → Optional reasoning → Execution plan
    Phase 2: Execution - Workflow engine executes the plan
    
    Key Features:
    - Two-phase execution (planning → execution)
    - Optional reasoning engine for optimization
    - Plugin-based agent system with automatic discovery
    - LangGraph workflow engine for standardized execution
    - Real-time analytics and monitoring
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls, config=None):
        """Singleton pattern implementation"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, config: Optional[Union[BrainConfig, Dict[str, Any]]] = None):
        """Initialize the ThinkxLife Brain with plugin-based architecture"""
        if self._initialized:
            return
        

        # Core components  
        self.reasoning_engine: Optional[ReasoningEngine] = None
        self.workflow_engine: Optional[WorkflowEngine] = None
        self.data_source_registry = None
        self.security_manager: SecurityManager = SecurityManager(self.config.security)
        
        # Analytics and monitoring
        self.analytics = BrainAnalytics()
        self.start_time = datetime.now()
        self.active_executions: Dict[str, WorkflowExecution] = {}
        
        # Plugin system is the only supported method
        logger.info("ThinkxLife Brain initialized")

    
    async def initialize(self) -> None:
        """Initialize all Brain components"""
        if self._initialized:
            return
        
        logger.info("Initializing ThinkxLife Brain components...")
        
        try:
            # Initialize reasoning engine (LLM-powered decision making)
            self.reasoning_engine = get_reasoning_engine()
            await self.reasoning_engine.initialize()
            
            # Initialize workflow engine (industrial-grade orchestration)
            self.workflow_engine = get_workflow_engine()
            await self.workflow_engine.initialize()
            
            # Initialize data source registry
            self.data_source_registry = get_data_source_registry()
            await self.data_source_registry.initialize()
            
            self._initialized = True
            
            logger.info("ThinkxLife Brain initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Brain: {str(e)}")
            raise
    
    
    async def _ensure_initialized(self):
        """Ensure the Brain is initialized before processing requests"""
        if not self._initialized:
            await self.initialize()
    
    async def execute_agent_request(
        self,
        specifications: AgentExecutionSpec,
        request: BrainRequest,
        messages: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Execute request according to agent's specifications.
        Agent decides everything, Brain executes.
        
        Args:
            specifications: Agent's specifications for data sources, provider, tools, processing
            request: Original BrainRequest object
            messages: Formatted messages for LLM
            
        Returns:
            Response dictionary with content and metadata
        """
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        try:
            logger.info(f"Executing agent request {request_id} with agent specifications")
            
            # Sub-node 1: Query specified data sources
            context_data = await self._query_specified_data_sources(
                specifications.data_sources,
                request
            )
            
            # Sub-node 2-4: Use Workflow Engine (LangGraph) to orchestrate execution
            orchestration_result = await self.workflow_engine.orchestrate_request(
                messages=messages,
                provider_spec=specifications.provider,
                processing_spec=specifications.processing,
                context_data=context_data,
                tools=specifications.tools
            )
            
            processing_time = time.time() - start_time
            
            return {
                "success": orchestration_result.success,
                "content": orchestration_result.content,
                "metadata": {
                    "request_id": request_id,
                    "processing_time": processing_time,
                    "provider_used": orchestration_result.provider_used,
                    "iterations_used": orchestration_result.iterations_used,
                    "data_sources_queried": len(specifications.data_sources),
                    "tools_applied": len(specifications.tools),
                    **orchestration_result.metadata
                }
            }
            
        except Exception as e:
            logger.error(f"Error executing agent request {request_id}: {str(e)}")
            return {
                "success": False,
                "content": "Failed to process request",
                "error": str(e),
                "metadata": {
                    "request_id": request_id,
                    "processing_time": time.time() - start_time
                }
            }
    
    async def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point for processing Brain requests
        Routes to appropriate agent using the plugin system
        """
        # Ensure initialization for backward compatibility
        await self._ensure_initialized()
        
        start_time = time.time()
        request_id = request_data.get("id", str(uuid.uuid4()))
        
        try:
            # Update analytics
            self.analytics.total_requests += 1
            application = request_data.get("application", "general")
            self.analytics.application_usage[application] = (
                self.analytics.application_usage.get(application, 0) + 1
            )
            
            # Security validation
            if not self._validate_request(request_data):
                return self._create_error_response(
                    request_id, "Request validation failed", start_time
                )
            
            # Rate limiting check
            user_id = request_data.get("user_context", {}).get("user_id", "anonymous")
            if not self.security_manager.check_rate_limit(user_id):
                return self._create_error_response(
                    request_id, "Rate limit exceeded", start_time
                )
            
            # Create BrainRequest object
            brain_request = self._create_brain_request(request_data)
            
        
            
            # For legacy support, create a simple execution
            brain_request = self._create_brain_request(request_data)
            
            # Simple response for legacy API
            response_content = "Brain is ready. Please use plugin-based agents for request processing."
            
            processing_time = time.time() - start_time
            
            return {
                "success": True,
                "message": response_content,
                "data": {},
                "error": None,
                "metadata": {
                    "request_id": request_id,
                    "processing_time": processing_time,
                    "architecture": "plugin_based",
                    "note": "Use plugins to connect agents to Brain"
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing Brain request {request_id}: {str(e)}")
            processing_time = time.time() - start_time
            self._update_analytics(False, processing_time)
            
            return self._create_error_response(
                request_id, f"Internal Brain error: {str(e)}", start_time
            )
    
    def _create_brain_request(self, request_data: Dict[str, Any]) -> BrainRequest:
        """Create BrainRequest object from request data"""
        from .types import UserContext, RequestContext, RequestMetadata
        
        # Extract user context
        user_context_data = request_data.get("user_context", {})
        user_context = UserContext(
            user_id=user_context_data.get("user_id", "anonymous"),
            session_id=request_data.get("session_id", str(uuid.uuid4())),
            is_authenticated=user_context_data.get("is_authenticated", False)
        )
        
        # Set user profile data
        if "ace_score" in user_context_data:
            user_context.ace_score = user_context_data["ace_score"]
        
        # Create request context
        request_context = RequestContext(
            session_id=request_data.get("session_id"),
            user_preferences=None,  # Could be populated from user data
            application_state=request_data.get("metadata", {}),
            brain_context={}
        )
        
        # Create request metadata
        request_metadata = RequestMetadata()
        
        return BrainRequest(
            id=request_data.get("id", str(uuid.uuid4())),
            application=ApplicationType(request_data.get("application", "general")),
            message=request_data["message"],
            user_context=user_context,
            context=request_context,
            metadata=request_metadata
        )
    
    def _validate_request(self, request_data: Dict[str, Any]) -> bool:
        """Validate request data"""
        required_fields = ["message", "application"]
        return all(field in request_data for field in required_fields)
    
    def _convert_agent_response(self, agent_response, request_id: str, processing_time: float) -> Dict[str, Any]:
        """Convert AgentResponse to API response format matching main.py expectations"""
        return {
            "success": agent_response.success,
            "message": agent_response.content,
            "data": agent_response.metadata,
            "error": None if agent_response.success else agent_response.metadata.get("error", "Agent processing failed"),
            "metadata": {
                **agent_response.metadata,
                "processing_time": processing_time,
                "plugin_mode": True,
                "request_id": request_id
            },
            "timestamp": datetime.now().isoformat()
        }
    
    def _create_error_response(self, request_id: str, error_message: str, start_time: float) -> Dict[str, Any]:
        """Create error response matching main.py expectations"""
        return {
            "success": False,
            "message": None,
            "data": None,
            "error": error_message,
            "metadata": {
                "processing_time": time.time() - start_time,
                "request_id": request_id
            },
            "timestamp": datetime.now().isoformat()
        }
    
    def _update_analytics(self, success: bool, processing_time: float, agent_id: str = None, workflow_type: str = None):
        """Update analytics data"""
        
        # Update success rate
        total = self.analytics.total_requests
        if success:
            self.analytics.success_rate = (
                self.analytics.success_rate * (total - 1) + 1.0
            ) / total
        else:
            self.analytics.error_rate = (
                self.analytics.error_rate * (total - 1) + 1.0
            ) / total
        
        # Update response time
        self.analytics.average_response_time = (
            self.analytics.average_response_time * (total - 1) + processing_time
        ) / total
        
        # Update plugin usage
        if agent_id:
            self.analytics.plugin_usage[agent_id] = (
                self.analytics.plugin_usage.get(agent_id, 0) + 1
            )
        
        # Update workflow usage
        if workflow_type:
            self.analytics.workflow_executions[workflow_type] = (
                self.analytics.workflow_executions.get(workflow_type, 0) + 1
            )
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status"""
        await self._ensure_initialized()
        
        overall_status = "healthy"
        
        # Check data source health
        data_source_health = {}
        if self.data_source_registry:
            data_source_health = await self.data_source_registry.health_check_all()
            if any(status.get("status") == "unhealthy" for status in data_source_health.values()):
                overall_status = "degraded"
        
        # System health
        uptime = (datetime.now() - self.start_time).total_seconds()
        system_health = {
            "uptime_seconds": uptime,
            "total_requests": self.analytics.total_requests,
            "success_rate": self.analytics.success_rate,
            "error_rate": self.analytics.error_rate,
            "average_response_time": self.analytics.average_response_time,
            "active_plugins": self.analytics.active_plugins
        }
        
        return {
            "overall": overall_status,
            "data_sources": data_source_health,
            "system": system_health,
            "timestamp": datetime.now().isoformat()
        }
    
    async def get_analytics(self) -> Dict[str, Any]:
        """Get comprehensive analytics"""
        await self._ensure_initialized()
        
        # Update uptime
        uptime = (datetime.now() - self.start_time).total_seconds()
        self.analytics.uptime = uptime / 3600  # Convert to hours
        
        # Get data source information
        data_source_info = {}
        if self.data_source_registry:
            sources_info = self.data_source_registry.list_sources()
            for source_id, info in sources_info.items():
                data_source_info[source_id] = {
                    "type": info["type"],
                    "enabled": info["enabled"],
                    "usage_count": self.analytics.data_source_usage.get(source_id, 0)
                }
        
        return {
            **self.analytics.__dict__,
            "data_sources": data_source_info,
            "workflows": self.analytics.workflow_executions,
            "timestamp": datetime.now().isoformat()
        }
    
    # TWO-PHASE EXECUTION: PLANNING + EXECUTION
    
    async def process_agent_request(
        self,
        request: BrainRequest,
        agent: IAgent,
        execution_specs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process agent request using plugin specs
        
        Flow:
        1. Load agent's core (e.g., ZoeCore from agents/zoe/)
        2. Build prompts using core
        3. Execute LLM with two-phase
        4. Post-process with core
        5. Update conversation with core
        6. Return response
        """
        start_time = time.time()
        session_id = request.user_context.session_id
        
        try:
            # Load agent core (e.g., ZoeCore)
            agent_type = execution_specs.get("metadata", {}).get("agent_type", "unknown")
            logger.info(f"Processing request for agent: {agent_type}")
            
            if agent_type == "zoe":
                from agents.zoe import ZoeCore
                agent_core = ZoeCore()
                
                # Prepare context using ZoeCore
                context = agent_core.prepare_context(
                    message=request.message,
                    user_context={
                        "user_id": request.user_context.user_id,
                        "ace_score": getattr(request.user_context, 'ace_score', None)
                    },
                    session_id=session_id
                )
                
                # Get conversation history
                history = agent_core.get_conversation_history(session_id)
                
                # Build system prompt
                system_prompt = agent_core.build_system_prompt(context)
                
                # Build messages for LLM
                messages = [{"role": "system", "content": system_prompt}]
                
                # Add history
                for msg in history[-10:]:
                    messages.append({
                        "role": msg.get("role", "user"),
                        "content": msg.get("content", "")
                    })
                
                # Add current message
                messages.append({"role": "user", "content": request.message})
                
                # Execute LLM (simplified for now - will use two-phase later)
                llm_specs = execution_specs.get("llm", {})
                provider = create_provider(
                    llm_specs.get("provider", "openai"),
                    {
                        "model": llm_specs.get("model", "gpt-4o-mini"),
                        "temperature": llm_specs.get("temperature", 0.7),
                        "max_tokens": llm_specs.get("max_tokens", 1500),
                        **llm_specs.get("params", {})
                    }
                )
                await provider.initialize()
                
                # Call LLM
                llm_result = await provider.generate_response(messages=messages)
                llm_response = llm_result.get("content", "")
                
                if llm_response:
                    # Post-process with ZoeCore
                    final_response = agent_core.post_process_response(llm_response, context)
                    
                    # Update conversation
                    agent_core.update_conversation(session_id, request.message, final_response)
                    
                    return {
                        "success": True,
                        "content": final_response,
                        "session_id": session_id,
                        "metadata": {
                            "agent": agent_type,
                            "processing_time": time.time() - start_time,
                            "context": context
                        },
                        "processing_time": time.time() - start_time
                    }
                else:
                    # Use fallback
                    fallback = agent_core.get_fallback_response()
                    return {
                        "success": False,
                        "content": fallback,
                        "session_id": session_id,
                        "metadata": {"error": "No LLM response"},
                        "processing_time": time.time() - start_time
                    }
            
            else:
                # Unknown agent type
                return {
                    "success": False,
                    "content": "Unknown agent type",
                    "metadata": {"error": f"Agent type {agent_type} not supported"},
                    "processing_time": time.time() - start_time
                }
                
        except Exception as e:
            logger.error(f"Error processing agent request: {str(e)}")
            
            # Try to get error response from agent core
            try:
                if agent_type == "zoe":
                    from agents.zoe import ZoeCore
                    error_response = ZoeCore().get_error_response()
                else:
                    error_response = "An error occurred processing your request."
            except:
                error_response = "An error occurred processing your request."
            
            return {
                "success": False,
                "content": error_response,
                "metadata": {"error": str(e)},
                "processing_time": time.time() - start_time
            }
    
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
        return {
            "available_providers": get_available_providers(),
            "available_tools": list(self.tool_registry.tools.keys()) if hasattr(self, 'tool_registry') else [],
            "data_sources": self.data_source_registry.list_sources() if self.data_source_registry else {},
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
            estimated_tokens = specs.provider.max_tokens
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
    
    async def execute_plan(
        self,
        plan: ExecutionPlan,
        request: BrainRequest,
        agent: IAgent
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
    
    
    async def _query_specified_data_sources(
        self,
        data_source_specs: List[DataSourceSpec],
        request: BrainRequest
    ) -> Dict[str, Any]:
        """Query data sources as specified by agent"""
        context_data = {}
        
        if not data_source_specs:
            return context_data
        
        for spec in data_source_specs:
            if not spec.enabled:
                continue
                
            try:
                source_type = spec.source_type
                
                # Handle conversation history - agents provide this through specs
                if source_type == DataSourceType.CONVERSATION_HISTORY:
                    # Agents provide conversation history through their specifications
                    # This is now handled by the agent, not Brain
                    logger.debug("Conversation history requested - should be provided by agent in spec.query")
                
                # Handle vector DB or other data sources
                elif source_type in [DataSourceType.VECTOR_DB, DataSourceType.FILE_SYSTEM]:
                    if self.data_source_registry:
                        # Check if this is an external agent-specific data source
                        if spec.config and spec.config.get("db_path"):
                            # Register external data source if needed
                            source_id = await self.data_source_registry.get_or_create_external_source(spec.config)
                            
                            if source_id:
                                # Query the specific external source
                                external_source = self.data_source_registry.get_source(source_id)
                                if external_source:
                                    query = spec.query or request.message
                                    results = await external_source.query(
                                        query,
                                        context=spec.filters,
                                        k=spec.limit
                                    )
                                    context_data[f"external_{source_type.value}_results"] = results
                                    logger.info(f"Queried external data source: {source_id}")
                            else:
                                logger.warning(f"Failed to register external data source from spec: {spec.config}")
                        else:
                            # Use default data source registry
                            query = spec.query or request.message
                            results = await self.data_source_registry.query_best(
                                query,
                                context=spec.filters,
                                k=spec.limit
                            )
                            context_data[f"{source_type.value}_results"] = results
                
                # Handle web search via MCP
                elif source_type == DataSourceType.WEB_SEARCH:
                    # Tools are handled by workflow engine
                    pass
                
                logger.info(f"Queried data source: {source_type.value}")
                
            except Exception as e:
                logger.warning(f"Failed to query data source {spec.source_type}: {str(e)}")
        
        return context_data
    
    async def _get_specified_provider(self, provider_spec: Optional[ProviderSpec]):
        """Get provider as specified by agent"""
        if not provider_spec:
            # No provider specified, return None (agent handles this case)
            return None
        
        provider_type = provider_spec.provider_type
        
        # Check if provider is already in cache
        if provider_type in self.providers_cache:
            return self.providers_cache[provider_type]
        
        # Initialize new provider with agent's configuration
        try:
            provider_config = provider_spec.to_dict()
            provider = create_provider(provider_type, provider_config)
            await provider.initialize()
            
            # Cache for future use
            self.providers_cache[provider_type] = provider
            
            logger.info(f"Initialized provider {provider_type} as specified by agent")
            return provider
            
        except Exception as e:
            logger.error(f"Failed to initialize specified provider {provider_type}: {str(e)}")
            return None
    
    async def _apply_specified_tools(
        self,
        tool_specs: List[ToolSpec],
        context_data: Dict[str, Any],
        request: BrainRequest
    ) -> Dict[str, Any]:
        """Apply tools as specified by agent"""
        enhanced_data = {}
        
        for tool_spec in tool_specs:
            if not tool_spec.enabled:
                continue
            
            try:
                tool_name = tool_spec.name
                tool_config = tool_spec.config
                
                # Apply tool based on name
                # This is where MCP tools would be invoked
                # Tools are handled by workflow engine
                pass
                
                logger.info(f"Applied tool: {tool_name}")
                
            except Exception as e:
                logger.warning(f"Failed to apply tool {tool_spec.name}: {str(e)}")
        
        return enhanced_data
    
    async def _execute_with_provider(
        self,
        provider: Any,
        messages: List[Dict[str, str]],
        context_data: Dict[str, Any],
        processing_spec: ProcessingSpec,
        provider_spec: Optional[ProviderSpec]
    ) -> Dict[str, Any]:
        """Execute LLM request with specified provider and processing configuration"""
        
        if not provider:
            return {
                "success": False,
                "content": "No provider available",
                "metadata": {}
            }
        
        try:
            # Prepare provider parameters from agent specifications
            provider_params = provider_spec.to_dict() if provider_spec else {}
            
            # Remove provider_type as it's not a generation parameter
            provider_params.pop("provider_type", None)
            
            # Execute with iterative processing if specified
            max_iterations = processing_spec.max_iterations
            
            for iteration in range(max_iterations):
                try:
                    # Generate response
                    response = await provider.generate_response(
                        messages=messages,
                        **provider_params
                    )
                    
                    # Check if response is satisfactory
                    if response.get("success", False) and response.get("content"):
                        return {
                            "success": True,
                            "content": response.get("content", ""),
                            "metadata": {
                                "iteration": iteration + 1,
                                **response.get("metadata", {})
                            }
                        }
                    
                    # If not satisfactory and we have more iterations, continue
                    if iteration < max_iterations - 1:
                        logger.info(f"Response not satisfactory, trying iteration {iteration + 2}")
                        continue
                    
                except Exception as e:
                    logger.error(f"Provider execution failed on iteration {iteration + 1}: {str(e)}")
                    if iteration == max_iterations - 1:
                        raise
            
            # If we get here, all iterations failed
            return {
                "success": False,
                "content": "Failed to generate satisfactory response",
                "metadata": {"iterations_attempted": max_iterations}
            }
            
        except Exception as e:
            logger.error(f"Error executing with provider: {str(e)}")
            return {
                "success": False,
                "content": "Provider execution failed",
                "metadata": {"error": str(e)}
            }
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the Brain"""
        logger.info("Shutting down CortexFlow...")
        
        # Shutdown components
        if self.workflow_engine:
            await self.workflow_engine.shutdown()
        
        if self.data_source_registry:
            await self.data_source_registry.shutdown()
        
        logger.info("CortexFlow shutdown complete")


