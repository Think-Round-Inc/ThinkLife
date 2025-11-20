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

from .interfaces import BrainRequest, BrainResponse, IAgent
from .types import (
    BrainConfig, BrainAnalytics, PluginInfo, WorkflowExecution, 
    DataSourceInfo, ApplicationType, PluginStatus, WorkflowType,
    AgentExecutionSpec, DataSourceSpec, ProviderSpec, ToolSpec, ProcessingSpec,
    DataSourceType
)
from .spec_validator import SpecificationValidator, get_spec_validator
from .workflow_engine import WorkflowEngine, get_workflow_engine, OrchestrationResult
from .data_sources import get_data_source_registry
from .security_manager import SecurityManager
from .providers import create_provider, get_available_providers

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class ThinkxLifeBrain:
    """
    ThinkxLife AI Brain that orchestrates agents through a plugin architecture
    
    Key Features:
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
        self.spec_validator: Optional[SpecificationValidator] = None
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
            # Initialize specification validator
            self.spec_validator = get_spec_validator()
            await self.spec_validator.initialize(self.config.__dict__ if hasattr(self.config, '__dict__') else {})
            
            # Initialize workflow engine (LangGraph-based orchestration)
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
            
            # Sub-node 0: Validate specifications
            if self.spec_validator:
                validation_result = await self.spec_validator.validate_execution_spec(specifications)
                if not validation_result.valid:
                    logger.error(f"Specification validation failed: {validation_result.errors}")
                    return {
                        "success": False,
                        "content": "Invalid execution specification",
                        "error": "; ".join(validation_result.errors),
                        "metadata": {
                            "request_id": request_id,
                            "validation_errors": validation_result.errors
                        }
                    }
                
                # Log warnings and suggestions
                if validation_result.warnings:
                    logger.warning(f"Specification warnings: {validation_result.warnings}")
                if validation_result.suggestions:
                    logger.info(f"Specification suggestions: {validation_result.suggestions}")
            
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
    
    # Legacy methods removed - agents now connect via plugins
    
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
        logger.info("Shutting down Generalized Brain...")
        
        # Shutdown components
        if self.workflow_engine:
            await self.workflow_engine.shutdown()
        
        if self.data_source_registry:
            await self.data_source_registry.shutdown()
        
        logger.info("ThinkxLife Brain shutdown complete")


