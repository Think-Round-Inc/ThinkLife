"""
Session Manager for Keycloak Authentication
Handles unified session tracking from login to logout
"""

import logging
import os
import uuid
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import jwt
import requests
from functools import lru_cache

logger = logging.getLogger(__name__)


@dataclass
class UserSession:
    """
    Represents a single user session from login to logout
    One session = one login event until logout
    """
    session_id: str
    user_id: str
    login_time: datetime
    last_activity: datetime
    logout_time: Optional[datetime] = None
    status: str = "active"  # active, ended
    token_state: Optional[str] = None  # Keycloak session_state from token
    user_info: Dict[str, Any] = field(default_factory=dict)
    
    def is_active(self) -> bool:
        """Check if session is currently active"""
        return self.status == "active" and self.logout_time is None
    
    def end_session(self):
        """Mark session as ended (logout)"""
        self.status = "ended"
        self.logout_time = datetime.now()
        logger.info(f"Session {self.session_id} ended for user {self.user_id}")
    
    def update_activity(self):
        """Update last activity timestamp"""
        if self.is_active():
            self.last_activity = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary for easy tracking"""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "login_time": self.login_time.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "logout_time": self.logout_time.isoformat() if self.logout_time else None,
            "status": self.status,
            "duration_seconds": (
                (self.logout_time or datetime.now()) - self.login_time
            ).total_seconds(),
            "user_info": self.user_info
        }


class SessionManager:
    """
    Manages user sessions from login to logout
    Simple and trackable session lifecycle management
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.keycloak_url = os.getenv("KEYCLOAK_URL", "http://localhost:8080")
        self.keycloak_realm = os.getenv("KEYCLOAK_REALM", "thinklife")
        self.keycloak_client_id = os.getenv("KEYCLOAK_CLIENT_ID", "thinklife-frontend")
        
        # Session storage: session_id -> UserSession
        self.sessions: Dict[str, UserSession] = {}
        
        # User to active sessions mapping: user_id -> [session_ids]
        self.user_sessions: Dict[str, list] = {}
        
        # Token to session mapping: token_state -> session_id (for quick lookup)
        self.token_to_session: Dict[str, str] = {}
        
        # Public key cache for token validation
        self.public_key_cache = {}
        self.public_key_cache_expiry = {}
        self.cache_ttl = timedelta(hours=1)
        
        logger.info("SessionManager initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default session configuration"""
        return {
            "token_validation": {
                "enabled": True,
                "verify_signature": True,
                "verify_exp": True,
                "verify_iss": True,
            },
            "session_timeout": {
                "enabled": True,
                "max_session_duration_hours": 24,
                "inactivity_timeout_hours": 8,
            },
            "user_context": {
                "include_roles": True,
                "include_email": True,
                "include_profile": True,
            }
        }
    
    @lru_cache(maxsize=1)
    def _get_public_key(self) -> Optional[str]:
        """Get Keycloak realm public key for token verification"""
        try:
            cache_key = f"{self.keycloak_url}/{self.keycloak_realm}"
            if cache_key in self.public_key_cache:
                expiry = self.public_key_cache_expiry.get(cache_key)
                if expiry and datetime.now() < expiry:
                    return self.public_key_cache[cache_key]
            
            url = f"{self.keycloak_url}/realms/{self.keycloak_realm}"
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            
            realm_info = response.json()
            public_key = realm_info.get("public_key")
            
            if public_key:
                self.public_key_cache[cache_key] = public_key
                self.public_key_cache_expiry[cache_key] = datetime.now() + self.cache_ttl
                return public_key
            
            logger.warning("Public key not found in realm info")
            return None
            
        except Exception as e:
            logger.error(f"Failed to fetch Keycloak public key: {str(e)}")
            return None
    
    def validate_token(self, token: str) -> Dict[str, Any]:
        """
        Validate Keycloak JWT token
        
        Returns:
            Dict with validation result and user info if valid
        """
        if not token:
            return {
                "valid": False,
                "error": "No token provided",
                "user": None
            }
        
        try:
            # Decode token without verification first to get header
            unverified = jwt.decode(token, options={"verify_signature": False})
            
            # Get issuer
            issuer = unverified.get("iss")
            expected_issuer = f"{self.keycloak_url}/realms/{self.keycloak_realm}"
            
            if issuer != expected_issuer:
                return {
                    "valid": False,
                    "error": f"Invalid issuer: {issuer}",
                    "user": None
                }
            
            # Get public key
            public_key = self._get_public_key()
            if not public_key:
                logger.warning("Public key not available, decoding without signature verification")
                decoded = jwt.decode(
                    token,
                    options={"verify_signature": False, "verify_exp": True}
                )
            else:
                decoded = jwt.decode(
                    token,
                    f"-----BEGIN PUBLIC KEY-----\n{public_key}\n-----END PUBLIC KEY-----",
                    algorithms=["RS256"],
                    issuer=expected_issuer,
                    options={"verify_exp": True}
                )
            
            # Extract user info
            user_info = {
                "user_id": decoded.get("sub"),
                "email": decoded.get("email"),
                "name": decoded.get("name"),
                "first_name": decoded.get("given_name"),
                "last_name": decoded.get("family_name"),
                "username": decoded.get("preferred_username"),
                "email_verified": decoded.get("email_verified", False),
                "roles": decoded.get("realm_access", {}).get("roles", []),
                "client_roles": decoded.get("resource_access", {}).get(self.keycloak_client_id, {}).get("roles", []),
                "session_state": decoded.get("session_state"),
                "exp": decoded.get("exp"),
                "iat": decoded.get("iat"),
            }
            
            return {
                "valid": True,
                "error": None,
                "user": user_info,
                "token_data": decoded
            }
            
        except jwt.ExpiredSignatureError:
            return {
                "valid": False,
                "error": "Token has expired",
                "user": None
            }
        except jwt.InvalidTokenError as e:
            return {
                "valid": False,
                "error": f"Invalid token: {str(e)}",
                "user": None
            }
        except Exception as e:
            logger.error(f"Token validation error: {str(e)}")
            return {
                "valid": False,
                "error": f"Validation error: {str(e)}",
                "user": None
            }
    
    def create_session(self, token: str, user_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a new session on login
        One session per login event
        
        Args:
            token: Keycloak JWT token
            user_info: Optional user info dict (if already validated)
            
        Returns:
            Dict with session_id and user_context
        """
        # Validate token if user_info not provided
        if not user_info:
            validation_result = self.validate_token(token)
            if not validation_result["valid"]:
                return {
                    "success": False,
                    "error": validation_result.get("error", "Token validation failed"),
                    "session_id": None,
                    "user_context": None
                }
            user_info = validation_result["user"]
        
        user_id = user_info.get("user_id")
        token_state = user_info.get("session_state")
        
        if not user_id:
            return {
                "success": False,
                "error": "User ID not found in token",
                "session_id": None,
                "user_context": None
            }
        
        # Check if user already has an active session with this token_state
        if token_state and token_state in self.token_to_session:
            existing_session_id = self.token_to_session[token_state]
            existing_session = self.sessions.get(existing_session_id)
            
            if existing_session and existing_session.is_active():
                # Return existing active session
                logger.info(f"Using existing active session {existing_session_id} for user {user_id}")
                existing_session.update_activity()
                user_context = self.create_user_context(user_info, existing_session_id)
                return {
                    "success": True,
                    "error": None,
                    "session_id": existing_session_id,
                    "user_context": user_context
                }
        
        # Create new session
        session_id = str(uuid.uuid4())
        now = datetime.now()
        
        session = UserSession(
            session_id=session_id,
            user_id=user_id,
            login_time=now,
            last_activity=now,
            token_state=token_state,
            user_info=user_info
        )
        
        # Store session
        self.sessions[session_id] = session
        
        # Track user sessions
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = []
        self.user_sessions[user_id].append(session_id)
        
        # Map token_state to session_id
        if token_state:
            self.token_to_session[token_state] = session_id
        
        logger.info(f"Created new session {session_id} for user {user_id} (login)")
        
        # Create user context
        user_context = self.create_user_context(user_info, session_id)
        
        return {
            "success": True,
            "error": None,
            "session_id": session_id,
            "user_context": user_context
        }
    
    def end_session(self, session_id: Optional[str] = None, user_id: Optional[str] = None, token_state: Optional[str] = None) -> bool:
        """
        End a session on logout
        Can be called with session_id, user_id, or token_state
        
        Args:
            session_id: Direct session ID
            user_id: End all active sessions for user
            token_state: End session by token state
            
        Returns:
            True if session(s) ended successfully
        """
        if session_id:
            session = self.sessions.get(session_id)
            if session and session.is_active():
                session.end_session()
                # Remove from token mapping
                if session.token_state and session.token_state in self.token_to_session:
                    del self.token_to_session[session.token_state]
                logger.info(f"Ended session {session_id} for user {session.user_id}")
                return True
        
        elif token_state:
            session_id = self.token_to_session.get(token_state)
            if session_id:
                return self.end_session(session_id=session_id)
        
        elif user_id:
            # End all active sessions for user
            active_sessions = self.get_user_active_sessions(user_id)
            for sid in active_sessions:
                self.end_session(session_id=sid)
            return len(active_sessions) > 0
        
        return False
    
    def get_session(self, session_id: str) -> Optional[UserSession]:
        """Get session by ID"""
        return self.sessions.get(session_id)
    
    def get_session_by_token_state(self, token_state: str) -> Optional[UserSession]:
        """Get session by Keycloak token state"""
        session_id = self.token_to_session.get(token_state)
        if session_id:
            return self.sessions.get(session_id)
        return None
    
    def update_session_activity(self, session_id: Optional[str] = None, token_state: Optional[str] = None):
        """
        Update last activity for a session
        Called on each request to track activity
        """
        session = None
        
        if session_id:
            session = self.sessions.get(session_id)
        elif token_state:
            session = self.get_session_by_token_state(token_state)
        
        if session and session.is_active():
            session.update_activity()
    
    def get_user_active_sessions(self, user_id: str) -> list:
        """Get all active session IDs for a user"""
        session_ids = self.user_sessions.get(user_id, [])
        active = []
        for sid in session_ids:
            session = self.sessions.get(sid)
            if session and session.is_active():
                active.append(sid)
        return active
    
    def get_user_sessions(self, user_id: str, include_ended: bool = False) -> list:
        """Get all sessions for a user"""
        session_ids = self.user_sessions.get(user_id, [])
        if include_ended:
            return [self.sessions[sid].to_dict() for sid in session_ids if sid in self.sessions]
        return [self.sessions[sid].to_dict() for sid in session_ids if sid in self.sessions and self.sessions[sid].is_active()]
    
    def create_user_context(self, user_info: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """
        Create user context from validated token user info
        Maps Keycloak user info to UserContext-compatible format
        """
        context = {
            "user_id": user_info.get("user_id", "anonymous"),
            "session_id": session_id,  # Our unified session ID
            "is_authenticated": True,
            "authenticated": True,
            "auth_provider": "keycloak",
        }
        
        # Add optional fields based on config
        config = self.config.get("user_context", {})
        
        if config.get("include_email", True):
            context["email"] = user_info.get("email")
        
        if config.get("include_profile", True):
            context["name"] = user_info.get("name")
            context["first_name"] = user_info.get("first_name")
            context["last_name"] = user_info.get("last_name")
            context["username"] = user_info.get("username")
        
        if config.get("include_roles", True):
            roles = user_info.get("roles", [])
            client_roles = user_info.get("client_roles", [])
            context["roles"] = roles
            context["client_roles"] = client_roles
            context["permissions"] = roles + client_roles
        
        context["email_verified"] = user_info.get("email_verified", False)
        
        return context
    
    def validate_session(self, token: Optional[str] = None, user_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Validate session - checks token and updates session activity
        This is called on each request to validate and track activity
        
        Returns validation result with user context if valid
        """
        # If no token provided, check if user_context has auth info
        if not token and user_context:
            session_id = user_context.get("session_id")
            if session_id:
                session = self.get_session(session_id)
                if session and session.is_active():
                    session.update_activity()
                    return {
                        "valid": True,
                        "user_context": user_context,
                        "error": None
                    }
            return {
                "valid": False,
                "user_context": None,
                "error": "No valid session found"
            }
        
        if not token:
            return {
                "valid": False,
                "user_context": None,
                "error": "No token provided"
            }
        
        # Validate token
        validation_result = self.validate_token(token)
        
        if not validation_result["valid"]:
            return {
                "valid": False,
                "user_context": None,
                "error": validation_result["error"]
            }
        
        user_info = validation_result["user"]
        token_state = user_info.get("session_state")
        
        # Get or create session
        session = None
        session_id = None
        
        if token_state:
            session = self.get_session_by_token_state(token_state)
            if session:
                session_id = session.session_id
                session.update_activity()
        
        # If no session found, create one (login event)
        if not session or not session.is_active():
            create_result = self.create_session(token, user_info)
            if not create_result["success"]:
                return {
                    "valid": False,
                    "user_context": None,
                    "error": create_result.get("error", "Failed to create session")
                }
            session_id = create_result["session_id"]
            user_context = create_result["user_context"]
        else:
            # Update activity and create user context
            user_context = self.create_user_context(user_info, session_id)
        
        return {
            "valid": True,
            "user_context": user_context,
            "error": None
        }
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics about all sessions - useful for tracking"""
        total_sessions = len(self.sessions)
        active_sessions = sum(1 for s in self.sessions.values() if s.is_active())
        ended_sessions = total_sessions - active_sessions
        
        total_users = len(self.user_sessions)
        
        return {
            "total_sessions": total_sessions,
            "active_sessions": active_sessions,
            "ended_sessions": ended_sessions,
            "total_users": total_users,
            "sessions_by_user": {uid: len(sessions) for uid, sessions in self.user_sessions.items()}
        }
    
    def cleanup_old_sessions(self, max_age_hours: int = 168):  # 7 days default
        """Clean up old ended sessions to free memory"""
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        to_remove = []
        
        for session_id, session in self.sessions.items():
            if session.status == "ended" and session.logout_time and session.logout_time < cutoff:
                to_remove.append(session_id)
        
        for session_id in to_remove:
            session = self.sessions.pop(session_id, None)
            if session:
                # Remove from user_sessions mapping
                user_id = session.user_id
                if user_id in self.user_sessions:
                    self.user_sessions[user_id] = [sid for sid in self.user_sessions[user_id] if sid != session_id]
                    if not self.user_sessions[user_id]:
                        del self.user_sessions[user_id]
                
                # Remove from token mapping
                if session.token_state and session.token_state in self.token_to_session:
                    del self.token_to_session[session.token_state]
        
        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} old sessions")
        
        return len(to_remove)
