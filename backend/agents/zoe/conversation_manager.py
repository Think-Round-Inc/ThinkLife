"""
Zoe Conversation Manager

Manages conversation history, sessions, and context for Zoe AI Companion.
Integrates with the Brain's ConversationManager for comprehensive conversation tracking.
"""

import uuid
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass

# Import Brain conversation manager
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from brain.conversation_manager import ConversationManager
except ImportError as e:
    logging.warning(f"Brain conversation manager import failed: {e}")
    ConversationManager = None

logger = logging.getLogger(__name__)


@dataclass
class ConversationMessage:
    """Represents a single message in a conversation"""
    id: str
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ConversationSession:
    """Represents a conversation session"""
    session_id: str
    user_id: str
    created_at: datetime
    last_activity: datetime
    messages: List[ConversationMessage]
    user_context: Dict[str, Any]
    summary: Optional[str] = None
    active: bool = True


class ZoeConversationManager:
    """
    Manages conversations for Zoe AI Companion
    
    Features:
    - Session management with unique IDs
    - Conversation history tracking
    - Context integration with Brain's ConversationManager
    - Automatic session cleanup
    - User preference storage
    - Conversation summarization for long sessions
    """
    
    def __init__(self, brain_conversation_manager: Optional[ConversationManager] = None):
        self.sessions: Dict[str, ConversationSession] = {}
        self.user_sessions: Dict[str, List[str]] = {}  # user_id -> [session_ids]
        
        # Initialize or use provided conversation manager
        if brain_conversation_manager:
            self.conversation_manager = brain_conversation_manager
        elif ConversationManager:
            self.conversation_manager = ConversationManager()
        else:
            # Fallback if ConversationManager is not available
            self.conversation_manager = None
            logger.warning("ConversationManager not available - running without Brain integration")
        
        # Configuration
        self.max_message_history = 50
        self.session_timeout_hours = 24
        self.auto_cleanup_interval = 3600  # 1 hour in seconds
        self.max_sessions_per_user = 10
        
        logger.info("Zoe Conversation Manager initialized")
    
    def create_session(
        self, 
        user_id: str, 
        user_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new conversation session
        
        Args:
            user_id: User identifier
            user_context: Initial user context (age, preferences, etc.)
            
        Returns:
            session_id: Unique session identifier
        """
        session_id = str(uuid.uuid4())
        now = datetime.now()
        
        session = ConversationSession(
            session_id=session_id,
            user_id=user_id,
            created_at=now,
            last_activity=now,
            messages=[],
            user_context=user_context or {},
            active=True
        )
        
        # Store session
        self.sessions[session_id] = session
        
        # Track user sessions
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = []
        self.user_sessions[user_id].append(session_id)
        
        # Limit sessions per user
        self._limit_user_sessions(user_id)
        
        # Initialize Brain conversation if available
        if self.conversation_manager:
            self.conversation_manager.get_conversation(session_id)
        
        logger.info(f"Created new conversation session {session_id} for user {user_id}")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """Get a conversation session by ID"""
        return self.sessions.get(session_id)
    
    def get_or_create_session(
        self, 
        session_id: Optional[str], 
        user_id: str,
        user_context: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, ConversationSession]:
        """
        Get existing session or create new one
        
        Returns:
            Tuple of (session_id, session)
        """
        if session_id and session_id in self.sessions:
            session = self.sessions[session_id]
            if self._is_session_valid(session):
                session.last_activity = datetime.now()
                return session_id, session
        
        # Create new session
        new_session_id = self.create_session(user_id, user_context)
        return new_session_id, self.sessions[new_session_id]
    
    def add_message(
        self, 
        session_id: str, 
        role: str, 
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add a message to the conversation
        
        Args:
            session_id: Session identifier
            role: 'user' or 'assistant'
            content: Message content
            metadata: Optional message metadata
            
        Returns:
            bool: Success status
        """
        session = self.get_session(session_id)
        if not session:
            logger.error(f"Session {session_id} not found")
            return False
        
        # Create message
        message = ConversationMessage(
            id=str(uuid.uuid4()),
            role=role,
            content=content,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        
        # Add to session
        session.messages.append(message)
        session.last_activity = datetime.now()
        
        # Limit message history
        if len(session.messages) > self.max_message_history:
            # Remove oldest messages but keep system/important ones
            session.messages = session.messages[-self.max_message_history:]
        
        # Update Brain conversation manager if available
        if self.conversation_manager:
            self.conversation_manager.add_message(session_id, role, content)
        
        logger.debug(f"Added {role} message to session {session_id}")
        return True
    
    def get_conversation_history(
        self, 
        session_id: str, 
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get conversation history for a session
        
        Args:
            session_id: Session identifier
            limit: Maximum number of messages to return
            
        Returns:
            List of message dictionaries
        """
        session = self.get_session(session_id)
        if not session:
            return []
        
        messages = session.messages
        if limit:
            messages = messages[-limit:]
        
        return [
            {
                "id": msg.id,
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat(),
                "metadata": msg.metadata
            }
            for msg in messages
        ]
    
    def get_context_for_ai(self, session_id: str) -> Dict[str, Any]:
        """
        Get formatted context for AI processing
        
        Returns context suitable for Brain/OpenAI processing
        """
        session = self.get_session(session_id)
        if not session:
            return {}
        
        # Get recent conversation history (last 10 messages for AI context)
        recent_messages = session.messages[-10:] if session.messages else []
        
        conversation_history = [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat()
            }
            for msg in recent_messages
        ]
        
        # Combine with user context
        ai_context = {
            **session.user_context,
            "conversation_history": conversation_history,
            "session_id": session_id,
            "conversation_length": len(session.messages),
            "session_duration": (datetime.now() - session.created_at).total_seconds(),
        }
        
        return ai_context
    
    def update_user_context(
        self, 
        session_id: str, 
        context_updates: Dict[str, Any]
    ) -> bool:
        """Update user context for a session"""
        session = self.get_session(session_id)
        if not session:
            return False
        
        session.user_context.update(context_updates)
        session.last_activity = datetime.now()
        
        # Update Brain conversation manager if available
        if self.conversation_manager:
            self.conversation_manager.update_conversation(session_id, {"user_context": session.user_context})
        
        return True
    
    def end_session(self, session_id: str) -> bool:
        """End a conversation session"""
        session = self.get_session(session_id)
        if not session:
            return False
        
        session.active = False
        session.last_activity = datetime.now()
        
        logger.info(f"Ended conversation session {session_id}")
        return True
    
    def get_user_sessions(self, user_id: str) -> List[str]:
        """Get all session IDs for a user"""
        return self.user_sessions.get(user_id, [])
    
    def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired sessions
        
        Returns:
            Number of sessions cleaned up
        """
        now = datetime.now()
        expired_sessions = []
        
        for session_id, session in self.sessions.items():
            if not self._is_session_valid(session, now):
                expired_sessions.append(session_id)
        
        # Remove expired sessions
        for session_id in expired_sessions:
            session = self.sessions.pop(session_id, None)
            if session:
                # Remove from user sessions
                user_sessions = self.user_sessions.get(session.user_id, [])
                if session_id in user_sessions:
                    user_sessions.remove(session_id)
        
        # Cleanup Brain conversations if available
        if self.conversation_manager:
            self.conversation_manager.cleanup_expired_conversations()
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
        
        return len(expired_sessions)
    
    def get_session_stats(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a session"""
        session = self.get_session(session_id)
        if not session:
            return None
        
        user_messages = [msg for msg in session.messages if msg.role == "user"]
        assistant_messages = [msg for msg in session.messages if msg.role == "assistant"]
        
        return {
            "session_id": session_id,
            "user_id": session.user_id,
            "created_at": session.created_at.isoformat(),
            "last_activity": session.last_activity.isoformat(),
            "total_messages": len(session.messages),
            "user_messages": len(user_messages),
            "assistant_messages": len(assistant_messages),
            "session_duration_minutes": (datetime.now() - session.created_at).total_seconds() / 60,
            "active": session.active
        }
    
    def _is_session_valid(self, session: ConversationSession, now: Optional[datetime] = None) -> bool:
        """Check if a session is still valid"""
        if not session.active:
            return False
        
        if now is None:
            now = datetime.now()
        
        time_since_activity = now - session.last_activity
        return time_since_activity.total_seconds() < (self.session_timeout_hours * 3600)
    
    def _limit_user_sessions(self, user_id: str):
        """Limit the number of sessions per user"""
        user_sessions = self.user_sessions.get(user_id, [])
        
        if len(user_sessions) > self.max_sessions_per_user:
            # Remove oldest sessions
            sessions_to_remove = len(user_sessions) - self.max_sessions_per_user
            oldest_sessions = sorted(
                [(sid, self.sessions[sid].created_at) for sid in user_sessions if sid in self.sessions],
                key=lambda x: x[1]
            )[:sessions_to_remove]
            
            for session_id, _ in oldest_sessions:
                if session_id in self.sessions:
                    del self.sessions[session_id]
                if session_id in user_sessions:
                    user_sessions.remove(session_id)
    
    def export_conversation(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Export a conversation for external use"""
        session = self.get_session(session_id)
        if not session:
            return None
        
        return {
            "session_id": session_id,
            "user_id": session.user_id,
            "created_at": session.created_at.isoformat(),
            "last_activity": session.last_activity.isoformat(),
            "user_context": session.user_context,
            "messages": [
                {
                    "id": msg.id,
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat(),
                    "metadata": msg.metadata
                }
                for msg in session.messages
            ],
            "summary": session.summary,
            "stats": self.get_session_stats(session_id)
        } 