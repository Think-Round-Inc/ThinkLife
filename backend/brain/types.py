"""
Type definitions for the ThinkxLife Brain system
"""

from typing import Dict, List, Optional, Any, Literal
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class ApplicationType(str, Enum):
    """Available application types"""
    HEALING_ROOMS = "healing-rooms"
    AI_AWARENESS = "inside-our-ai"
    CHATBOT = "chatbot"
    COMPLIANCE = "compliance"
    EXTERIOR_SPACES = "exterior-spaces"
    GENERAL = "general"


class ProviderType(str, Enum):
    """Available AI provider types"""
    OPENAI = "openai"
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"
    GROK = "grok"


@dataclass
class ModelInfo:
    """Information about a specific AI model"""
    name: str
    max_tokens: int
    input_cost_per_1k: Optional[float] = None
    output_cost_per_1k: Optional[float] = None
    description: Optional[str] = None
    capabilities: List[str] = field(default_factory=lambda: ["text", "completion"])
    
    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = ["text", "completion"]


@dataclass
class ProviderInfo:
    """Information about an AI provider"""
    name: str
    available: bool
    models: List[ModelInfo]
    required_fields: List[str]
    optional_fields: List[str]
    defaults: Dict[str, Any]
    max_tokens_limit: int
    min_tokens_limit: int
    supports_streaming: bool = True
    supports_functions: bool = False
    supports_tools: bool = False


class WorkflowType(str, Enum):
    """Available workflow types"""
    CONVERSATIONAL = "conversational"
    TRAUMA_INFORMED = "trauma_informed"
    ANALYTICAL = "analytical"
    CREATIVE = "creative"
    CUSTOM = "custom"


class PluginStatus(str, Enum):
    """Plugin status types"""
    REGISTERED = "registered"
    INITIALIZED = "initialized"
    ACTIVE = "active"
    ERROR = "error"
    DISABLED = "disabled"


class MessageRole(str, Enum):
    """Message roles in conversation"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class BrainConfig:
    """Brain configuration"""
    providers: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    security: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    session: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Message:
    """Chat message"""
    id: str
    role: MessageRole
    content: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class UserProfile:
    """User profile information"""
    id: str
    name: Optional[str] = None
    email: Optional[str] = None
    age: Optional[int] = None
    ace_score: Optional[float] = None
    ai_knowledge_level: Optional[Literal["beginner", "intermediate", "advanced"]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class UserPreferences:
    """User preferences"""
    language: str = "en"
    theme: Literal["light", "dark"] = "light"
    communication_style: Literal["formal", "casual", "empathetic"] = "empathetic"
    ai_personality: Literal["supportive", "educational", "professional"] = "supportive"
    privacy_level: Literal["high", "medium", "low"] = "medium"
    content_filtering: bool = True


@dataclass
class TraumaContext:
    """Trauma-related context for healing rooms"""
    ace_score: float = 0.0
    trauma_types: List[str] = field(default_factory=list)
    trigger_words: List[str] = field(default_factory=list)
    safety_preferences: Dict[str, Any] = field(default_factory=dict)
    healing_goals: List[str] = field(default_factory=list)
    last_assessment: Optional[datetime] = None


@dataclass
class AIKnowledgeContext:
    """AI knowledge context for educational features"""
    level: Literal["beginner", "intermediate", "advanced"] = "beginner"
    completed_modules: List[str] = field(default_factory=list)
    current_interests: List[str] = field(default_factory=list)
    preferred_style: Literal["visual", "textual", "interactive"] = "textual"
    ethics_understanding: float = 50.0
    last_quiz_score: Optional[float] = None


@dataclass
class UserContext:
    """Complete user context"""
    user_id: str
    session_id: str
    is_authenticated: bool
    user_profile: Optional[UserProfile] = None
    permissions: List[str] = field(default_factory=list)
    preferences: UserPreferences = field(default_factory=UserPreferences)
    trauma_context: Optional[TraumaContext] = None
    ai_knowledge_context: Optional[AIKnowledgeContext] = None


@dataclass
class RequestContext:
    """Request-specific context"""
    session_id: Optional[str] = None
    conversation_history: List[Message] = field(default_factory=list)
    user_preferences: Optional[UserPreferences] = None
    application_state: Optional[Dict[str, Any]] = None
    retrieved_docs: Optional[List[Dict[str, Any]]] = None
    brain_context: Optional[Dict[str, Any]] = None


@dataclass
class RequestMetadata:
    """Request metadata"""
    user_agent: Optional[str] = None
    ip_address: Optional[str] = None
    referrer: Optional[str] = None
    device_type: Optional[Literal["desktop", "mobile", "tablet"]] = None
    language: Optional[str] = None


@dataclass
class BrainRequest:
    """Brain request structure"""
    id: str
    application: ApplicationType
    message: str
    user_context: UserContext
    context: Optional[RequestContext] = None
    metadata: Optional[RequestMetadata] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ResponseMetadata:
    """Response metadata"""
    provider: str
    model: str
    tokens_used: Optional[int] = None
    processing_time: float = 0.0
    confidence: Optional[float] = None
    sources: List[str] = field(default_factory=list)


@dataclass
class BrainResponse:
    """Brain response structure"""
    id: Optional[str] = None
    success: bool = True
    message: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Optional[ResponseMetadata] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ProviderConfig:
    """Base provider configuration"""
    enabled: bool = True
    timeout: float = 30.0


@dataclass
class LocalProviderConfig(ProviderConfig):
    """Local provider configuration"""
    endpoint: str = "http://localhost:8000"
    api_key: Optional[str] = None


@dataclass
class OpenAIProviderConfig(ProviderConfig):
    """OpenAI provider configuration"""
    api_key: str = ""
    model: str = "gpt-4o-mini"
    max_tokens: int = 2000
    temperature: float = 0.7
    organization: Optional[str] = None


@dataclass
class AnthropicProviderConfig(ProviderConfig):
    """Anthropic provider configuration"""
    api_key: str = ""
    model: str = "claude-3-sonnet-20240229"
    max_tokens: int = 2000
    temperature: float = 0.7


@dataclass
class GeminiProviderConfig(ProviderConfig):
    """Google Gemini provider configuration"""
    api_key: str = ""
    model: str = "gemini-1.5-flash"
    max_tokens: int = 2000
    temperature: float = 0.7
    safety_settings: Optional[Dict[str, Any]] = None


@dataclass
class GrokProviderConfig(ProviderConfig):
    """Grok (xAI) provider configuration"""
    api_key: str = ""
    model: str = "grok-beta"
    max_tokens: int = 2000
    temperature: float = 0.7
    stream: bool = False


@dataclass
class SecurityConfig:
    """Security configuration"""
    rate_limiting: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "max_requests_per_minute": 60,
        "max_requests_per_hour": 1000
    })
    content_filtering: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "blocked_words": [],
        "trauma_safe_mode": True
    })
    user_validation: Dict[str, Any] = field(default_factory=lambda: {
        "require_auth": True,
        "allow_anonymous": False
    })


@dataclass
class HealthStatus:
    """Provider health status"""
    overall: Literal["healthy", "degraded", "unhealthy"]
    providers: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    system: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PluginInfo:
    """Information about a registered plugin"""
    plugin_id: str
    name: str
    version: str
    description: str
    status: PluginStatus
    capabilities: List[str] = field(default_factory=list)
    supported_applications: List[str] = field(default_factory=list)
    author: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    last_used: Optional[datetime] = None
    usage_count: int = 0
    error_count: int = 0


@dataclass
class WorkflowExecution:
    """Information about workflow execution"""
    execution_id: str
    workflow_type: WorkflowType
    agent_id: str
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "running"  # running, completed, failed
    steps_executed: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class DataSourceInfo:
    """Information about data sources"""
    source_id: str
    source_type: str
    enabled: bool
    priority: int
    health_status: str
    last_query: Optional[datetime] = None
    query_count: int = 0
    error_count: int = 0
    average_response_time: float = 0.0


@dataclass
class BrainAnalytics:
    """Brain analytics data"""
    total_requests: int = 0
    success_rate: float = 0.0
    average_response_time: float = 0.0
    provider_usage: Dict[str, int] = field(default_factory=dict)
    application_usage: Dict[str, int] = field(default_factory=dict)
    user_satisfaction: float = 0.0
    error_rate: float = 0.0
    uptime: float = 0.0
    
    # Plugin system analytics
    active_plugins: int = 0
    plugin_usage: Dict[str, int] = field(default_factory=dict)
    workflow_executions: Dict[str, int] = field(default_factory=dict)
    data_source_usage: Dict[str, int] = field(default_factory=dict)
    
    # RAG analytics
    rag_usage: Dict[str, Any] = field(default_factory=lambda: {
        "queries": 0,
        "successful_retrievals": 0,
        "average_documents_retrieved": 0.0,
        "cache_hit_rate": 0.0
    })


# ============================================================================
# Agent Execution Specifications
# ============================================================================

class DataSourceType(str, Enum):
    """Available data source types"""
    VECTOR_DB = "vector_db"
    FILE_SYSTEM = "file_system"
    WEB_SEARCH = "web_search"
    DATABASE = "database"
    API = "api"
    MEMORY = "memory"
    CONVERSATION_HISTORY = "conversation_history"


@dataclass
class DataSourceSpec:
    """Specification for a data source that agents want to use"""
    source_type: DataSourceType
    query: Optional[str] = None
    filters: Dict[str, Any] = field(default_factory=dict)
    limit: int = 5
    enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "source_type": self.source_type.value if isinstance(self.source_type, Enum) else self.source_type,
            "query": self.query,
            "filters": self.filters,
            "limit": self.limit,
            "enabled": self.enabled,
            "config": self.config
        }


@dataclass
class ProviderSpec:
    """Specification for LLM provider configuration that agents want to use"""
    provider_type: str  # "openai", "gemini", "anthropic", "grok"
    model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2000
    stream: bool = False
    custom_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for provider calls"""
        result = {
            "provider_type": self.provider_type,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": self.stream,
            **self.custom_params
        }
        if self.model:
            result["model"] = self.model
        return result


@dataclass
class ToolSpec:
    """Specification for a tool that agents want to use"""
    name: str
    enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)
    required_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "enabled": self.enabled,
            "config": self.config,
            "required_params": self.required_params
        }


@dataclass
class ProcessingSpec:
    """Specification for how agents want the request to be processed"""
    max_iterations: int = 3
    timeout_seconds: float = 30.0
    enable_context_enhancement: bool = True
    enable_safety_checks: bool = True
    enable_conversation_memory: bool = True
    stream_response: bool = False
    custom_processing: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "max_iterations": self.max_iterations,
            "timeout_seconds": self.timeout_seconds,
            "enable_context_enhancement": self.enable_context_enhancement,
            "enable_safety_checks": self.enable_safety_checks,
            "enable_conversation_memory": self.enable_conversation_memory,
            "stream_response": self.stream_response,
            **self.custom_processing
        }


@dataclass
class AgentExecutionSpec:
    """
    Complete specification from agent to Brain about how to process the request
    
    This is what agents pass to Brain to specify:
    - Which data sources to use
    - Which provider and configuration
    - Which tools to apply
    - How to process the request
    """
    data_sources: List[DataSourceSpec] = field(default_factory=list)
    provider: Optional[ProviderSpec] = None
    tools: List[ToolSpec] = field(default_factory=list)
    processing: ProcessingSpec = field(default_factory=ProcessingSpec)
    
    # Optional agent-specific metadata
    agent_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "data_sources": [ds.to_dict() for ds in self.data_sources],
            "provider": self.provider.to_dict() if self.provider else None,
            "tools": [tool.to_dict() for tool in self.tools],
            "processing": self.processing.to_dict(),
            "agent_metadata": self.agent_metadata
        } 