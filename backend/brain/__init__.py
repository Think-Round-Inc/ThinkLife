"""
ThinkxLife Brain - Generalized AI Orchestration System

This module contains the centralized AI Brain that manages all AI operations
across the ThinkxLife platform using a plugin-based architecture.

Version 2.0 Features:
- Plugin-based agent system with automatic discovery
- LangGraph workflow engine for standardized execution
- MCP integration for data source abstraction
- Trauma-informed safety systems
- Zero-code agent integration
"""

# Import main brain system
from .brain_core import ThinkxLifeBrain
from .types import (
    BrainRequest, BrainResponse, BrainConfig,
    AgentExecutionSpec, DataSourceSpec, ProviderSpec, ToolSpec, ProcessingSpec,
    DataSourceType, ModelInfo, ProviderInfo
)
from .interfaces import IAgent, IAgentPlugin, AgentMetadata, AgentConfig, AgentResponse
from .spec_validator import SpecificationValidator, get_spec_validator, ValidationResult
from .workflow_engine import WorkflowEngine, get_workflow_engine, OrchestrationResult
from .data_sources import (
    DataSourceRegistry, get_data_source_registry
)
from .security_manager import SecurityManager
from .tools import (
    get_tool_registry, ToolRegistry, BaseTool, ToolResult,
    TavilySearchTool, DocumentSummarizerTool,
    create_tool, get_available_tools as get_available_tool_types
)

# Import providers
from . import providers

__version__ = "2.0.0"
    
__all__ = [
    # Main Brain class
    "ThinkxLifeBrain",
    "GeneralizedBrain",
    
    # Core types
    "BrainRequest", 
    "BrainResponse",
    "BrainConfig",
    
    # Execution specifications
    "AgentExecutionSpec",
    "DataSourceSpec",
    "ProviderSpec",
    "ToolSpec",
    "ProcessingSpec",
    "DataSourceType",
    
    # Provider information
    "ModelInfo",
    "ProviderInfo",
    
    # Plugin system
    "IAgent",
    "IAgentPlugin", 
    "AgentMetadata",
    "AgentConfig",
    "AgentResponse",
    
    # Management systems
    "SpecificationValidator",
    "ValidationResult",
    "WorkflowEngine",
    "OrchestrationResult",
    "DataSourceRegistry",
    "SecurityManager",
    
    # Tools
    "ToolRegistry",
    "BaseTool",
    "ToolResult",
    "TavilySearchTool",
    "DocumentSummarizerTool",
    "create_tool",
    "get_available_tool_types",
    
    # Providers
    "providers",
    
    # Utility functions
    "get_spec_validator",
    "get_workflow_engine",
    "get_data_source_registry",
    "get_tool_registry"
] 