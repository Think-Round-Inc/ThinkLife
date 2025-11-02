"""
Context Manager for ThinkLife Brain
"""

from typing import Dict, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class ContextManager:
    """
    Manages conversation context and user context across sessions
    """
    
    def __init__(self):
        self.contexts = {}  # session_id -> context
        self.max_history_length = 20
        self.context_retention_hours = 24
    
    def get_context(self, session_id: str) -> Dict[str, Any]:
        """Get context for a session"""
        if session_id not in self.contexts:
            self.contexts[session_id] = {
                "conversation_history": [],
                "user_preferences": {},
                "application_state": {},
                "created_at": datetime.now(),
                "last_accessed": datetime.now()
            }
        
        self.contexts[session_id]["last_accessed"] = datetime.now()
        return self.contexts[session_id]
    
    def update_context(self, session_id: str, updates: Dict[str, Any]):
        """Update context for a session"""
        context = self.get_context(session_id)
        context.update(updates)
        context["last_accessed"] = datetime.now()
    
    def add_message(self, session_id: str, role: str, content: str):
        """Add a message to conversation history"""
        context = self.get_context(session_id)
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        
        context["conversation_history"].append(message)
        
        # Limit history length
        if len(context["conversation_history"]) > self.max_history_length:
            context["conversation_history"] = context["conversation_history"][-self.max_history_length:]
    
    def cleanup_expired_contexts(self):
        """Remove expired contexts"""
        cutoff_time = datetime.now() - timedelta(hours=self.context_retention_hours)
        expired_sessions = [
            session_id for session_id, context in self.contexts.items()
            if context["last_accessed"] < cutoff_time
        ]
        
        for session_id in expired_sessions:
            del self.contexts[session_id]
            logger.info(f"Cleaned up expired context for session {session_id}") 