"""
Zoe AI Companion - Core Module

Zoe is ThinkxLife's empathetic AI companion designed with trauma-informed care principles.
This module integrates with the ThinkxLife Brain system and manages conversation context.
"""

import logging
import uuid
from typing import Dict, Optional, Any, List
from datetime import datetime

# ThinkxLife Brain imports
import sys
import os

# Add backend directory to path for brain imports
backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

try:
    from brain import ThinkxLifeBrain, ConversationManager
except ImportError as e:
    logging.warning(f"Brain imports failed: {e}")
    ThinkxLifeBrain = None
    ConversationManager = None

# Zoe imports
from .personality import ZoePersonality
from .conversation_manager import ZoeConversationManager

logger = logging.getLogger(__name__)


class ZoeCore:
    """
    Zoe AI Companion - The "mouth" of Zoe
    
    This class handles everything EXCEPT LLM calls:
    - Trauma-informed personality and prompts
    - Conversation context and session management
    - Response post-processing and safety checks
    - Empathetic communication patterns
    
    LLM calls (the "brain/thinking") happen via:
    Plugin → Cortex for processing
    
    Think of it as:
    - ZoeCore = Personality, memory, safety
    - Plugin = Connector to trigger LLM thinking
    """
    
    def __init__(self):
        self.personality = ZoePersonality()
        self.conversation_manager = ZoeConversationManager(None)
        
        # Zoe's characteristics
        self.empathy_level = "high"
        self.trauma_awareness = True
        
        logger.info("ZoeCore initialized - personality and conversation management ready")
    
    def prepare_context(self, message: str, user_context: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """
        Prepare context for building prompts
        
        Returns dict with user info, conversation history, safety requirements
        """
        # Get conversation history
        history = self.conversation_manager.get_session(session_id)
        
        context = {
            "user_id": user_context.get("user_id", "anonymous"),
            "session_id": session_id,
            "empathy_level": self.empathy_level,
            "trauma_aware": self.trauma_awareness,
            "conversation_length": len(history.get("messages", [])) if history else 0
        }
        
        # Add ACE score if available
        ace_score = user_context.get("ace_score")
        if ace_score:
            context["ace_score"] = ace_score
            context["high_trauma_risk"] = ace_score >= 4
        
        return context
    
    def build_system_prompt(self, context: Dict[str, Any]) -> str:
        """
        Build Zoe's trauma-informed system prompt based on context
        
        This is Zoe's "personality" that guides her responses
        """
        return self._build_system_prompt(context)
    
    def _build_system_prompt(self, context: Dict[str, Any]) -> str:
        """Build Zoe's trauma-informed system prompt"""
        base_prompt = """You are Zoe, a compassionate AI companion for ThinkxLife.

Core Principles:
- Trauma-Informed Care: Always prioritize safety and empowerment
- Empathetic Communication: Validate feelings and provide support
- Educational Support: Help users understand AI and technology
- Healing-Focused: Support users on their healing journey

Guidelines:
- Use warm, supportive language
- Acknowledge emotions before problem-solving
- Respect boundaries and pace
- Provide resources when appropriate
- Never judge or minimize experiences
- Use phrases like "I hear you", "That makes sense", "Thank you for sharing"

You speak with kindness, patience, and genuine care for each person's wellbeing."""
        
        # Add trauma-specific guidance for high-risk users
        if context.get("high_trauma_risk"):
            base_prompt += """

IMPORTANT: This user has a high ACE score (trauma history). Please:
- Be extra gentle and patient
- Avoid triggering language
- Offer support resources
- Acknowledge their strength
- Move at their pace"""
        
        return base_prompt
    
    def post_process_response(
        self,
        llm_response: str,
        context: Dict[str, Any]
    ) -> str:
        """
        Post-process LLM response with Zoe's safety checks
        
        Ensures:
        - No minimizing language
        - Trauma-informed tone
        - Appropriate boundaries
        """
        response = llm_response
        
        # Check for minimizing language
        minimizing_phrases = ["just", "simply", "only", "it's not that bad", "calm down"]
        for phrase in minimizing_phrases:
            if phrase.lower() in response.lower():
                logger.warning(f"Zoe: Detected potentially minimizing language: {phrase}")
        
        # Add empathetic closing if needed
        if len(response) > 200 and not any(word in response.lower() for word in ["here", "support", "help"]):
            response += "\n\nI'm here if you need anything else. 💙"
        
        return response
    
    def update_conversation(
        self,
        session_id: str,
        user_message: str,
        assistant_response: str
    ):
        """Update conversation history"""
        self.conversation_manager.add_message(
            session_id=session_id,
            role="user",
            content=user_message
        )
        
        self.conversation_manager.add_message(
            session_id=session_id,
            role="assistant",
            content=assistant_response
        )
    
    def get_conversation_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for session"""
        session_data = self.conversation_manager.get_session(session_id)
        if session_data:
            return session_data.get("messages", [])
        return []
    
    def clear_conversation(self, session_id: str) -> bool:
        """Clear conversation history"""
        self.conversation_manager.clear_session(session_id)
        return True
    
    def update_context(self, session_id: str, context: Dict[str, Any]) -> bool:
        """Update session context"""
        session_data = self.conversation_manager.get_session(session_id)
        if session_data:
            session_data.update(context)
            return True
        return False
    
    def get_fallback_response(self) -> str:
        """Get fallback response when LLM fails"""
        return "I'm having a moment of difficulty connecting my thoughts. Could we try that again? I really want to be here for you. 💙"
    
    def get_error_response(self) -> str:
        """Get empathetic error response"""
        return "I'm experiencing some technical challenges right now. Please bear with me and try again in a moment. You're important to me, and I want to give you my best. 💙"
    
    def health_check(self) -> Dict[str, Any]:
        """Health check for ZoeCore"""
        session_count = 0
        if hasattr(self.conversation_manager, 'sessions'):
            session_count = len(self.conversation_manager.sessions)
        
        return {
            "status": "healthy",
            "personality": "trauma-informed",
            "empathy_level": self.empathy_level,
            "trauma_awareness": self.trauma_awareness,
            "active_sessions": session_count
        }
    
    def shutdown(self):
        """Shutdown ZoeCore"""
        if hasattr(self.conversation_manager, 'clear_all'):
            self.conversation_manager.clear_all()
        logger.info("ZoeCore shutdown complete")
    
    # ==========================================
    # Legacy method - kept for backward compatibility
    # This uses the OLD brain instance directly
    # NEW code should use the plugin → cortex flow
    # ==========================================
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