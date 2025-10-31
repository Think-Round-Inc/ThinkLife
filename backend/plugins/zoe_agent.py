"""
Zoe Agent Plugin - Simple connector between Brain and Zoe services
This plugin acts as a lightweight bridge, delegating all logic to Zoe
"""

import logging
from typing import Dict, Any, List, Optional

from brain.interfaces import (
    IAgent, IAgentPlugin, IConversationalAgent,
    AgentMetadata, AgentConfig, AgentResponse, BrainRequest,
    AgentCapability
)

logger = logging.getLogger(__name__)


class ZoeAgent(IConversationalAgent):
    """
    Lightweight Zoe Agent Plugin - Simple connector to Zoe services
    All domain logic is handled by Zoe, this just bridges the connection
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self.agent_id = config.agent_id
        self._initialized = False
        self.zoe_interface = None
        
    @property
    def metadata(self) -> AgentMetadata:
        """Return Zoe agent metadata"""
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

    async def initialize(self, config: AgentConfig) -> bool:
        """Initialize Zoe agent by connecting to Zoe services"""
        if self._initialized:
            return True
            
        try:
            logger.info("Initializing Zoe Agent connector")
            
            # Import Zoe components
            from agents.zoe import ZoeCore
            from agents.zoe.brain_interface import ZoeBrainInterface
            
            # Initialize ZoeCore
            zoe_core = ZoeCore()
            
            # Create Zoe's Brain interface
            self.zoe_interface = ZoeBrainInterface(zoe_core)
            
            # Initialize interface
            if await self.zoe_interface.initialize():
                self._initialized = True
                logger.info("Zoe Agent connector initialized successfully")
                return True
            else:
                logger.error("Failed to initialize Zoe interface")
                return False
                
        except Exception as e:
            logger.error(f"Error initializing Zoe Agent connector: {str(e)}")
            return False

    async def process_request(self, request: BrainRequest) -> AgentResponse:
        """Process request by delegating to Zoe interface"""
        if not self._initialized or not self.zoe_interface:
            return AgentResponse(
                success=False,
                content="Zoe is not available right now. Please try again later.",
                metadata={"error": "Agent not initialized"}
            )
        
        try:
            # Delegate to Zoe's Brain interface
            zoe_response = await self.zoe_interface.process_brain_request(request)
            
            # Convert to AgentResponse
            return AgentResponse(
                success=zoe_response.get("success", False),
                content=zoe_response.get("response", ""),
                metadata=zoe_response.get("metadata", {}),
                processing_time=zoe_response.get("processing_time", 0.0),
                session_id=zoe_response.get("session_id")
            )
            
        except Exception as e:
            logger.error(f"Error in Zoe Agent connector: {str(e)}")
            return AgentResponse(
                success=False,
                content="I'm experiencing some technical difficulties. Please try again.",
                metadata={"error": str(e)}
            )

    async def health_check(self) -> Dict[str, Any]:
        """Health check via Zoe interface"""
        base_health = {
            "status": "healthy" if self._initialized else "not_initialized",
            "agent_id": self.agent_id,
            "initialized": self._initialized
        }
        
        if self.zoe_interface:
            zoe_health = await self.zoe_interface.health_check()
            base_health.update(zoe_health)
        
        return base_health

    async def shutdown(self) -> None:
        """Shutdown agent connector"""
        if self.zoe_interface:
            await self.zoe_interface.shutdown()
        
        self.zoe_interface = None
        self._initialized = False
        logger.info("Zoe Agent connector shutdown complete")

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

    # IConversationalAgent methods - delegate to Zoe
    async def get_conversation_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get conversation history via Zoe interface"""
        if self.zoe_interface:
            return await self.zoe_interface.get_conversation_history(session_id)
        return []

    async def clear_conversation(self, session_id: str) -> bool:
        """Clear conversation via Zoe interface"""
        if self.zoe_interface:
            return await self.zoe_interface.clear_conversation(session_id)
        return False

    async def update_context(self, session_id: str, context: Dict[str, Any]) -> bool:
        """Update context via Zoe interface"""
        if self.zoe_interface:
            return await self.zoe_interface.update_context(session_id, context)
        return False


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
