"""
Standard interfaces and contracts for ThinkxLife Brain agents and plugins
Defines the contracts that agents must implement to work with Brain
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .types import BrainRequest, BrainResponse, UserContext


class AgentCapability(str, Enum):
    """Agent capabilities"""
    CONVERSATIONAL = "conversational"
    ANALYTICAL = "analytical"
    CREATIVE = "creative"
    SEARCH = "search"
    TOOL_USE = "tool_use"


class DataSourceType(str, Enum):
    """Types of data sources"""
    VECTOR_DB = "vector_db"
    FILE_SYSTEM = "file_system"
    WEB_SEARCH = "web_search"
    MEMORY = "memory"
    CONVERSATION_HISTORY = "conversation_history"


@dataclass
class AgentMetadata:
    """Metadata describing an agent's capabilities"""
    name: str
    version: str
    description: str
    capabilities: List[AgentCapability]
    supported_applications: List[str]
    requires_auth: bool = True
    author: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class AgentConfig:
    """Configuration for agent initialization"""
    agent_id: str
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResponse:
    """Standardized agent response"""
    success: bool
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: Optional[float] = None
    processing_time: float = 0.0
    session_id: Optional[str] = None


# ============================================================================
# Core Interfaces
# ============================================================================


class IDataSource(ABC):
    """Interface for data sources that Brain can use"""
    
    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the data source"""
        pass
    
    @abstractmethod
    async def query(self, query: str, context: Dict[str, Any] = None, **kwargs) -> List[Dict[str, Any]]:
        """Query the data source"""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Check data source health"""
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Clean up resources"""
        pass
    
    @property
    @abstractmethod
    def source_type(self) -> DataSourceType:
        """Return the type of this data source"""
        pass


class IAgent(ABC):
    """
    Base interface that all Brain agents must implement
    
    Agents specify execution requirements via create_execution_specs(),
    and Brain executes according to those specifications.
    """
    
    @property
    @abstractmethod
    def metadata(self) -> AgentMetadata:
        """Return agent metadata"""
        pass
    
    @abstractmethod
    async def initialize(self, config: AgentConfig) -> bool:
        """Initialize the agent"""
        pass
    
    @abstractmethod
    async def create_execution_specs(self, request: BrainRequest) -> 'AgentExecutionSpec':
        """
        Agent specifies how Brain should process the request:
        - Which data sources to query
        - Which LLM provider and configuration to use
        - Which tools to apply
        - Processing requirements (iterations, timeout, etc.)
        
        Brain executes exactly what agent specifies.
        """
        pass
    
    @abstractmethod
    async def process_request(self, request: BrainRequest) -> AgentResponse:
        """
        Process a user request and return response
        
        Agents implement domain logic, Brain handles LLM execution via specifications.
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Check agent health"""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Clean shutdown of agent resources"""
        pass
    
    async def can_handle_request(self, request: BrainRequest) -> float:
        """
        Return confidence score (0.0-1.0) for handling this request
        Default implementation checks supported applications
        """
        application = request.application
        if application in self.metadata.supported_applications:
            return 0.8
        return 0.0


class IConversationalAgent(IAgent):
    """
    Interface for agents that maintain conversation context
    Agents implementing this manage their own conversation history
    """
    
    @abstractmethod
    async def get_conversation_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for a session"""
        pass
    
    @abstractmethod
    async def clear_conversation(self, session_id: str) -> bool:
        """Clear conversation history"""
        pass
    
    @abstractmethod
    async def update_context(self, session_id: str, context: Dict[str, Any]) -> bool:
        """Update conversation context"""
        pass


class ISafetyAwareAgent(IAgent):
    """
    Interface for agents with safety and content filtering
    """
    
    @abstractmethod
    async def assess_content_safety(self, request: BrainRequest) -> Dict[str, Any]:
        """
        Assess request for safety concerns
        
        Returns:
            Dict with keys: safe (bool), concerns (list), severity (str)
        """
        pass
    
    @abstractmethod
    async def apply_content_filters(self, response: AgentResponse) -> AgentResponse:
        """Apply content filters to response"""
        pass


class IStreamingAgent(IAgent):
    """Interface for agents that support streaming responses"""
    
    @abstractmethod
    async def stream_response(self, request: BrainRequest) -> AsyncGenerator[str, None]:
        """Stream response chunks as they're generated"""
        pass


class IAgentPlugin(ABC):
    """
    Interface for agent plugins that can be dynamically loaded
    Plugins create agent instances and provide metadata
    """
    
    @abstractmethod
    def create_agent(self, config: AgentConfig) -> IAgent:
        """Factory method to create agent instance"""
        pass
    
    @abstractmethod
    def get_metadata(self) -> AgentMetadata:
        """Get plugin metadata"""
        pass
    
    @abstractmethod
    def validate_config(self, config: AgentConfig) -> bool:
        """Validate configuration before agent creation"""
        pass
