"""
Zoe AI Companion - Core Module

Zoe is ThinkLife's empathetic AI companion designed with trauma-informed care principles.
This module integrates with the ThinkLife Brain system and manages conversation context.
"""

import logging
import uuid
from typing import Dict, Optional, Any, List
from datetime import datetime

# ThinkLife Brain imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from brain.brain_core import ThinkLifeBrain
    from brain.context_manager import ContextManager
except ImportError as e:
    logging.warning(f"Brain imports failed: {e}")
    ThinkLifeBrain = None
    ContextManager = None

# Zoe imports
from .personality import ZoePersonality
from .conversation_manager import ZoeConversationManager

logger = logging.getLogger(__name__)


class ZoeCore:
    """
    Zoe AI Companion - Main interface that connects to ThinkLife Brain
    
    This class handles:
    - Personality-based responses
    - Integration with Brain module for LLM calls
    - Conversation context and session management
    - Trauma-informed interaction patterns
    
    Enhanced with full conversation management for contextual responses.
    """
    
    def __init__(self, brain_instance: Optional[ThinkLifeBrain] = None):
        self.brain = brain_instance
        self.personality = ZoePersonality()
        
        # Initialize conversation manager
        # Note: ThinkLifeBrain doesn't have a context_manager attribute
        # So we create a new ContextManager instance for conversation management
        if ContextManager:
            brain_context_manager = ContextManager()
        else:
            brain_context_manager = None
        
        self.conversation_manager = ZoeConversationManager(brain_context_manager)
        
        logger.info("Zoe AI Companion initialized with Brain integration and conversation management")
    
    async def process_message(
        self,
        message: str,
        user_context: Optional[Dict[str, Any]] = None,
        application: str = "chatbot",
        session_id: Optional[str] = None,
        user_id: str = "anonymous"
    ) -> Dict[str, Any]:
        """
        Process a user message through Zoe's personality and Brain system with full conversation context
        
        Args:
            message: User's message
            user_context: User context information (age, ACE score, etc.)
            application: Application type (defaults to chatbot)
            session_id: Optional session ID for conversation continuity
            user_id: User identifier for session management
            
        Returns:
            Dictionary containing Zoe's response and metadata
        """
        try:
            # Get or create conversation session
            session_id, session = self.conversation_manager.get_or_create_session(
                session_id=session_id,
                user_id=user_id,
                user_context=user_context
            )
            
            # Add user message to conversation history
            self.conversation_manager.add_message(
                session_id=session_id,
                role="user",
                content=message,
                metadata={"application": application}
            )
            
            # Check if message needs redirection (off-topic or harmful)
            if self.personality.is_off_topic_request(message):
                # Check if it's a harmful request for safety response
                is_harmful = self.personality._is_harmful_request(message.lower())
                redirect_response = self.personality.get_redirect_response(is_harmful=is_harmful)
                
                # Add assistant response to conversation history
                self.conversation_manager.add_message(
                    session_id=session_id,
                    role="assistant",
                    content=redirect_response,
                    metadata={
                        "redirected": True, 
                        "harmful_request": is_harmful,
                        "application": application
                    }
                )
                
                return {
                    "success": True,
                    "response": redirect_response,
                    "redirected": True,
                    "safety_response": is_harmful,
                    "session_id": session_id,
                    "timestamp": datetime.now().isoformat()
                }
            
            # Get enhanced context with conversation history
            enhanced_context = self._prepare_brain_context_with_history(
                session_id=session_id,
                user_context=user_context or {},
                message=message
            )
            
            # Create Brain request with conversation context
            brain_request_data = {
                "id": str(uuid.uuid4()),
                "message": message,
                "application": application,
                "user_context": enhanced_context,
                "session_id": session_id,
                "metadata": {
                    "source": "zoe",
                    "personality_mode": "empathetic_companion",
                    "conversation_length": len(session.messages),
                    "session_duration": (datetime.now() - session.created_at).total_seconds()
                }
            }
            
            # Process through Brain system
            brain_response = await self.brain.process_request(brain_request_data)
            
            if brain_response.get("success", False):
                ai_response = brain_response.get("message", "")
                
                # Add assistant response to conversation history
                self.conversation_manager.add_message(
                    session_id=session_id,
                    role="assistant",
                    content=ai_response,
                    metadata={
                        "brain_metadata": brain_response.get("metadata", {}),
                        "application": application
                    }
                )
                
                # Apply Zoe's personality post-processing
                final_response = self.personality.post_process_response(
                    ai_response, 
                    enhanced_context
                )
                
                # Update final response in conversation history
                if final_response != ai_response:
                    # Update the last message with personality-processed response
                    if session.messages and session.messages[-1].role == "assistant":
                        session.messages[-1].content = final_response
                        session.messages[-1].metadata["personality_processed"] = True
                
                return {
                    "success": True,
                    "response": final_response,
                    "session_id": session_id,
                    "conversation_stats": self.conversation_manager.get_session_stats(session_id),
                    "metadata": brain_response.get("metadata", {}),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                # Handle Brain processing failure
                error_response = self.personality.get_error_response()
                
                # Add error response to conversation history
                self.conversation_manager.add_message(
                    session_id=session_id,
                    role="assistant",
                    content=error_response,
                    metadata={"error": True, "brain_error": brain_response.get("error")}
                )
                
                return {
                    "success": False,
                    "response": error_response,
                    "session_id": session_id,
                    "error": brain_response.get("error", "Brain processing failed"),
                    "timestamp": datetime.now().isoformat()
                }
        
        except Exception as e:
            logger.error(f"Error in Zoe message processing: {str(e)}")
            
            # Attempt to add error to conversation if session exists
            try:
                if 'session_id' in locals():
                    error_response = self.personality.get_error_response()
                    self.conversation_manager.add_message(
                        session_id=session_id,
                        role="assistant", 
                        content=error_response,
                        metadata={"error": True, "exception": str(e)}
                    )
            except:
                pass
            
            return {
                "success": False,
                "response": "I'm having trouble processing your message right now. Let's try again in a moment.",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _prepare_brain_context_with_history(
        self,
        session_id: str,
        user_context: Dict[str, Any],
        message: str
    ) -> Dict[str, Any]:
        """
        Prepare enhanced context for Brain processing including conversation history
        
        Args:
            session_id: Session identifier
            user_context: Base user context
            message: Current message
            
        Returns:
            Enhanced context dictionary with conversation history
        """
        # Get conversation context from conversation manager
        conversation_context = self.conversation_manager.get_context_for_ai(session_id)
        
        # Merge with user context
        enhanced_context = {
            **user_context,
            **conversation_context,
            "current_message": message,
            "zoe_personality_active": True,
            "trauma_informed_mode": True,
            "empathetic_responses": True
        }
        
        # Add personality-specific context
        personality_context = self.personality.get_context_enhancements(enhanced_context)
        enhanced_context.update(personality_context)
        
        return enhanced_context
    
    def get_conversation_history(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get conversation history for a session"""
        return self.conversation_manager.get_conversation_history(session_id, limit)
    
    def get_session_stats(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a conversation session"""
        return self.conversation_manager.get_session_stats(session_id)
    
    def get_user_sessions(self, user_id: str) -> List[str]:
        """Get all session IDs for a user"""
        return self.conversation_manager.get_user_sessions(user_id)
    
    def end_session(self, session_id: str) -> bool:
        """End a conversation session"""
        return self.conversation_manager.end_session(session_id)
    
    def cleanup_expired_sessions(self) -> int:
        """Clean up expired conversation sessions"""
        return self.conversation_manager.cleanup_expired_sessions()
    
    def export_conversation(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Export a conversation for external use"""
        return self.conversation_manager.export_conversation(session_id)
    
    def update_user_context(self, session_id: str, context_updates: Dict[str, Any]) -> bool:
        """Update user context for a session"""
        return self.conversation_manager.update_user_context(session_id, context_updates) 