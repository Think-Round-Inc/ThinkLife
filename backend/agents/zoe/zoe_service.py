"""
Zoe Service - Frontend Interface

This service provides the interface between the frontend and the plugin architecture.
All LLM processing goes through: Frontend - ZoeService - Plugin - CortexFlow

Usage:
    from agents.zoe import ZoeService
    
    service = ZoeService()
    await service.initialize()
    
    response = await service.process_message(
        message="I'm feeling anxious",
        user_id="user_123",
        session_id="session_1",
        user_context={"ace_score": 5}
    )
"""

import logging
import uuid
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class ZoeService:
    """
    Zoe Service - Clean interface for frontend
    
    This service:
    1. Receives requests from frontend/API
    2. Uses ZoeCore to prepare context and check safety
    3. Invokes the ZoeAgent plugin for LLM processing
    4. Returns formatted response to frontend
    
    All LLM calls go through the plugin - cortex architecture.
    """
    
    def __init__(self):
        self.zoe_core = None
        self.zoe_agent = None
        self.initialized = False
    
    async def initialize(self) -> bool:
        """Initialize ZoeService"""
        try:
            from .zoe_core import ZoeCore
            from plugins.zoe_agent import ZoeAgent
            from brain.specs import AgentConfig
            
            # Initialize ZoeCore (personality, conversation)
            self.zoe_core = ZoeCore()
            logger.info("ZoeCore initialized")
            
            # Initialize ZoeAgent plugin
            config = AgentConfig(agent_id="zoe_main")
            self.zoe_agent = ZoeAgent(config)
            await self.zoe_agent.initialize(config)
            logger.info("ZoeAgent plugin initialized")
            
            self.initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize ZoeService: {e}")
            return False
    
    async def process_message(
        self,
        message: str,
        user_id: str = "anonymous",
        session_id: Optional[str] = None,
        user_context: Optional[Dict[str, Any]] = None,
        application: str = "chatbot"
    ) -> Dict[str, Any]:
        """
        Process a user message through the plugin architecture
        
        Args:
            message: User's message
            user_id: User identifier
            session_id: Session ID from brain's unified session management
            user_context: Optional user context (ace_score, etc.) - should include session_id
            application: Application type
        
        Returns:
            Dictionary with response and metadata
        """
        # NOTE: Tracing is handled by cortex.py - no decorator here to avoid duplicate traces
        
        if not self.initialized:
            await self.initialize()
        
        try:
            # Ensure user_context is a dict
            if user_context is None:
                user_context = {}

            # Resolve session_id
            if not session_id:
                session_id = user_context.get("session_id")
            
            # Generate session_id if still missing (for conversation tracking)
            if not session_id:
                session_id = str(uuid.uuid4())
                logger.info(f"Generated new conversation session_id: {session_id}")
            
            # Ensure session_id is in user_context for consistency
            user_context["session_id"] = session_id
            
            # 1. Create BrainRequest using ZoeCore helper
            request_data = self.zoe_core.create_brain_request(
                message=message,
                user_context=user_context,
                application=application,
                session_id=session_id,
                user_id=user_id
            )
            
            # 2. Check if message needs redirection (safety)
            redirect_info = request_data.get("redirect_info")
            if redirect_info and redirect_info.get("should_redirect"):
                # Add to conversation history
                self.zoe_core.conversation_manager.add_message(
                    session_id=request_data["session_id"],
                    role="user",
                    content=message
                )
                self.zoe_core.conversation_manager.add_message(
                    session_id=request_data["session_id"],
                    role="assistant",
                    content=redirect_info["response"],
                    metadata={
                        "redirected": True,
                        "harmful": redirect_info.get("is_harmful", False)
                    }
                )
                
                return {
                    "success": True,
                    "response": redirect_info["response"],
                    "session_id": request_data["session_id"],
                    "redirected": True,
                    "metadata": {
                        "safety_redirect": True,
                        "agent": "zoe"
                    },
                    "timestamp": datetime.now().isoformat()
                }
            
            # 3. Create BrainRequest object for plugin
            from brain.specs import BrainRequest, ApplicationType
            from .helpers import create_user_context
            
            brain_request = BrainRequest(
                id=f"zoe_{datetime.now().timestamp()}",
                application=ApplicationType(application),
                message=message,
                user_context=create_user_context(
                    user_id=user_id,
                    session_id=request_data["session_id"],
                    user_context=user_context
                )
            )
            
            # 4. Add user message to conversation history
            self.zoe_core.conversation_manager.add_message(
                session_id=request_data["session_id"],
                role="user",
                content=message,
                metadata={"application": application}
            )
            
            # 5. Process through plugin - cortex
            logger.info(f"Processing message through plugin for session: {request_data['session_id']}")
            agent_response = await self.zoe_agent.process_request(brain_request)
            
            # 6. Format response for frontend
            if agent_response.success:
                return {
                    "success": True,
                    "response": agent_response.content,
                    "session_id": agent_response.session_id,
                    "metadata": {
                        **agent_response.metadata,
                        "processing_time": agent_response.processing_time,
                        "agent": "zoe"
                    },
                    "timestamp": datetime.now().isoformat()
                }
            else:
                # LLM processing failed, use fallback
                fallback_response = self.zoe_core.get_fallback_response()
                
                # Add fallback to conversation
                self.zoe_core.conversation_manager.add_message(
                    session_id=request_data["session_id"],
                    role="assistant",
                    content=fallback_response,
                    metadata={"fallback": True, "error": agent_response.metadata.get("error")}
                )
                
                return {
                    "success": False,
                    "response": fallback_response,
                    "session_id": request_data["session_id"],
                    "error": agent_response.metadata.get("error", "Processing failed"),
                    "metadata": {"agent": "zoe", "fallback": True},
                    "timestamp": datetime.now().isoformat()
                }
        
        except Exception as e:
            logger.exception("Error in ZoeService.process_message")
            
            error_response = self.zoe_core.get_error_response() if self.zoe_core else "I'm experiencing technical difficulties."
            
            return {
                "success": False,
                "response": error_response,
                "error": str(e),
                "metadata": {"agent": "zoe", "error": True},
                "timestamp": datetime.now().isoformat()
            }
    
    def get_conversation_history(self, session_id: str, limit: Optional[int] = None) -> list:
        """Get conversation history for a session"""
        if not self.zoe_core:
            return []
        return self.zoe_core.get_conversation_history(session_id, limit)
    
    def clear_conversation(self, session_id: str) -> bool:
        """Clear conversation history"""
        if not self.zoe_core:
            return False
        return self.zoe_core.clear_conversation(session_id)
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for ZoeService"""
        health = {
            "service": "zoe",
            "initialized": self.initialized,
            "zoe_core_available": self.zoe_core is not None,
            "zoe_agent_available": self.zoe_agent is not None
        }
        
        if self.zoe_core:
            health["zoe_core_health"] = self.zoe_core.health_check()
        
        if self.zoe_agent:
            health["zoe_agent_health"] = await self.zoe_agent.health_check()
        
        return health
    
    async def shutdown(self):
        """Shutdown ZoeService"""
        if self.zoe_core:
            self.zoe_core.shutdown()
        
        if self.zoe_agent:
            await self.zoe_agent.shutdown()
        
        logger.info("ZoeService shutdown complete")


# Singleton instance for easy access
_zoe_service_instance: Optional[ZoeService] = None


def get_zoe_service() -> ZoeService:
    """Get singleton ZoeService instance"""
    global _zoe_service_instance
    if _zoe_service_instance is None:
        _zoe_service_instance = ZoeService()
    return _zoe_service_instance
