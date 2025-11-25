"""
Zoe Agent Plugin - Lightweight connector

This plugin:
1. Contains LLM request specs (provider, model, params, tools, data sources)
2. Invokes cortex for agent request processing
3. Returns results

All domain logic (personality, prompts, conversation) stays in agents/zoe/
"""

import logging
from typing import Dict, Any, List, Optional

from brain.specs import (
    IAgent, IAgentPlugin, IConversationalAgent,
    AgentMetadata, AgentConfig, AgentResponse, BrainRequest,
    AgentCapability
)

logger = logging.getLogger(__name__)


class ZoeAgent(IConversationalAgent):
    """
    Lightweight Zoe Plugin
    
    Contains:
    - LLM request specifications (provider, model, params)
    - Data source specifications
    - Tool specifications
    
    Invokes CortexFlow for processing
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self.agent_id = config.agent_id
        self._initialized = False
        
        # LLM Request Specifications
        self.llm_specs = {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "temperature": 0.8,
            "max_tokens": 1500,
            "params": {
                "top_p": 0.9,
                "presence_penalty": 0.3,
                "frequency_penalty": 0.2
            }
        }
        
        # Data source specifications
        self.data_sources = [
            {
                "type": "conversation_history",
                "enabled": True,
                "limit": 10
            },
            {
                "type": "vector_db",
                "enabled": True,
                "limit": 3,
                "filters": {"trauma_informed": True}
            }
        ]
        
        # Tool specifications (none for now)
        self.tools = []
        
        # Processing specifications
        self.processing = {
            "execution_strategy": "adaptive",
            "reasoning_threshold": 0.75,
            "max_iterations": 2,
            "timeout_seconds": 30.0
        }
        
    @property
    def metadata(self) -> AgentMetadata:
        """Return Zoe agent metadata"""
        return AgentMetadata(
            name="Zoe AI Companion",
            version="2.0.0",
            description="Empathetic AI companion with trauma-informed care",
            capabilities=[
                AgentCapability.CONVERSATIONAL,
                AgentCapability.EDUCATIONAL
            ],
            supported_applications=[
                "chatbot",
                "healing-rooms", 
                "general",
                "ai-awareness"
            ],
            author="ThinkxLife Team"
        )

    async def initialize(self, config: AgentConfig) -> bool:
        """Initialize plugin"""
        if self._initialized:
            return True
            
        try:
            logger.info("Initializing Zoe Agent Plugin")
            self._initialized = True
            logger.info("Zoe Agent Plugin initialized")
            return True
                
        except Exception as e:
            logger.error(f"Error initializing Zoe Agent Plugin: {str(e)}")
            return False

    async def create_execution_specs(self, request: BrainRequest) -> Dict[str, Any]:
        """
        Create LLM execution specs from stored configurations
        
        Returns dict with:
        - provider specs
        - data source specs  
        - tool specs
        - processing specs
        """
        return {
            "llm": self.llm_specs,
            "data_sources": self.data_sources,
            "tools": self.tools,
            "processing": self.processing,
            "metadata": {
                "agent_type": "zoe",
                "application": request.application.value,
                "session_id": request.user_context.session_id
            }
        }
    
    async def process_request(self, request: BrainRequest) -> AgentResponse:
        """
        Process request by invoking cortex with LLM specs
        
        Simple flow:
        1. Get execution specs (from stored config)
        2. Invoke cortex for processing
        3. Return response
        """
        if not self._initialized:
            return AgentResponse(
                success=False,
                content="I'm not quite ready yet. Please try again in a moment.",
                metadata={"error": "Plugin not initialized"}
            )
        
        try:
            # Get execution specs
            execution_specs = await self.create_execution_specs(request)
            
            # Invoke cortex for agent request processing
            from brain import CortexFlow
            cortex = CortexFlow()
            
            # Let cortex handle everything
            result = await cortex.process_agent_request(
                request=request,
                agent=self,
                execution_specs=execution_specs
            )
            
            # Return as AgentResponse
            return AgentResponse(
                success=result.get("success", False),
                content=result.get("content", ""),
                metadata=result.get("metadata", {}),
                processing_time=result.get("processing_time", 0.0),
                session_id=result.get("session_id")
            )
            
        except Exception as e:
            logger.error(f"Error in Zoe Agent Plugin: {str(e)}")
            return AgentResponse(
                success=False,
                content="I'm experiencing some technical challenges. Please try again.",
                metadata={"error": str(e)}
            )

    async def health_check(self) -> Dict[str, Any]:
        """Health check for plugin"""
        return {
            "status": "healthy" if self._initialized else "not_initialized",
            "agent_id": self.agent_id,
            "agent_name": "Zoe",
            "initialized": self._initialized,
            "llm_provider": self.llm_specs["provider"],
            "llm_model": self.llm_specs["model"]
        }

    async def shutdown(self) -> None:
        """Shutdown plugin"""
        self._initialized = False
        logger.info("Zoe Agent Plugin shutdown complete")

    async def can_handle_request(self, request: BrainRequest) -> float:
        """Return confidence score for handling this request"""
        # Zoe specializes in healing rooms and general conversation
        if request.application == "healing-rooms":
            return 0.98
        elif request.application == "chatbot":
            return 0.95
        elif request.application == "general":
            return 0.9
        else:
            return 0.0

    # IConversationalAgent methods - handled by cortex/agents/zoe
    async def get_conversation_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get conversation history (handled by cortex)"""
        # Cortex manages conversation via agents/zoe
        return []

    async def clear_conversation(self, session_id: str) -> bool:
        """Clear conversation (handled by cortex)"""
        return True

    async def update_context(self, session_id: str, context: Dict[str, Any]) -> bool:
        """Update context (handled by cortex)"""
        return True


class ZoeAgentPlugin(IAgentPlugin):
    """Plugin factory for Zoe Agent"""

    def create_agent(self, config: AgentConfig) -> IAgent:
        """Factory method to create Zoe agent instance"""
        return ZoeAgent(config)

    def get_metadata(self) -> AgentMetadata:
        """Get Zoe plugin metadata"""
        return AgentMetadata(
            name="Zoe AI Companion",
            version="2.0.0",
            description="ThinkxLife's empathetic AI companion with trauma-informed care",
            capabilities=[
                AgentCapability.CONVERSATIONAL,
                AgentCapability.EDUCATIONAL
            ],
            supported_applications=[
                "chatbot",
                "healing-rooms",
                "general", 
                "ai-awareness"
            ],
            author="ThinkxLife Team"
        )

    def validate_config(self, config: AgentConfig) -> bool:
        """Validate Zoe agent configuration"""
        return True


# Export the plugin for auto-discovery
__all__ = ["ZoeAgent", "ZoeAgentPlugin"]
