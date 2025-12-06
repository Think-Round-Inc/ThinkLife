"""
Security Manager for ThinkLife Brain
"""

from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import logging
import re

from .session_manager import SessionManager

logger = logging.getLogger(__name__)


class SecurityManager:
    """
    Manages security, rate limiting, and content filtering
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.rate_limits = {}  # user_id -> rate limit data
        self.blocked_words = self.config.get("content_filtering", {}).get("blocked_words", [])
        self.trauma_safe_mode = self.config.get("content_filtering", {}).get("trauma_safe_mode", True)
        
        # Initialize session manager for Keycloak authentication
        self.session_manager = SessionManager(self.config.get("session", {}))
    
    def _get_default_config(self):
        """Get default security configuration"""
        return {
            "rate_limiting": {
                "enabled": True,
                "max_requests_per_minute": 60,
                "max_requests_per_hour": 1000
            },
            "content_filtering": {
                "enabled": True,
                "blocked_words": [],
                "trauma_safe_mode": True
            },
            "user_validation": {
                "require_auth": True,
                "allow_anonymous": False
            }
        }
    
    def check_rate_limit(self, user_id: str, user_context: Optional[Dict[str, Any]] = None) -> bool:
        """Check if user has exceeded rate limits"""
        if not self.config.get("rate_limiting", {}).get("enabled", True):
            return True
        
        now = datetime.now()
        
        if user_id not in self.rate_limits:
            self.rate_limits[user_id] = {
                "requests_this_minute": [],
                "requests_this_hour": []
            }
        
        user_limits = self.rate_limits[user_id]
        
        # Clean old requests
        minute_ago = now - timedelta(minutes=1)
        hour_ago = now - timedelta(hours=1)
        
        user_limits["requests_this_minute"] = [
            req_time for req_time in user_limits["requests_this_minute"]
            if req_time > minute_ago
        ]
        
        user_limits["requests_this_hour"] = [
            req_time for req_time in user_limits["requests_this_hour"]
            if req_time > hour_ago
        ]
        
        # Check limits
        max_per_minute = self.config.get("rate_limiting", {}).get("max_requests_per_minute", 60)
        max_per_hour = self.config.get("rate_limiting", {}).get("max_requests_per_hour", 1000)
        
        if len(user_limits["requests_this_minute"]) >= max_per_minute:
            logger.warning(f"Rate limit exceeded for user {user_id}: {len(user_limits['requests_this_minute'])} requests per minute")
            return False
        
        if len(user_limits["requests_this_hour"]) >= max_per_hour:
            logger.warning(f"Rate limit exceeded for user {user_id}: {len(user_limits['requests_this_hour'])} requests per hour")
            return False
        
        # Record this request
        user_limits["requests_this_minute"].append(now)
        user_limits["requests_this_hour"].append(now)
        
        # Log rate limit check with user context if available
        if user_context:
            logger.debug(f"Rate limit check for user {user_id} (authenticated: {user_context.get('authenticated', False)})")
        
        return True
    
    def filter_content(self, content: str) -> Dict[str, Any]:
        """Filter content for inappropriate material"""
        if not self.config.get("content_filtering", {}).get("enabled", True):
            return {"safe": True, "content": content}
        
        original_content = content
        filtered_content = content
        flags = []
        
        # Check for blocked words
        for word in self.blocked_words:
            if word.lower() in content.lower():
                flags.append(f"blocked_word: {word}")
                filtered_content = re.sub(
                    re.escape(word), 
                    "*" * len(word), 
                    filtered_content, 
                    flags=re.IGNORECASE
                )
        
        # Trauma safety checks if enabled
        if self.trauma_safe_mode:
            trauma_indicators = [
                "suicide", "self-harm", "abuse", "violence", 
                "trauma", "ptsd", "depression", "anxiety"
            ]
            
            for indicator in trauma_indicators:
                if indicator.lower() in content.lower():
                    flags.append(f"trauma_indicator: {indicator}")
        
        is_safe = len(flags) == 0
        
        return {
            "safe": is_safe,
            "content": filtered_content if is_safe else original_content,
            "flags": flags,
            "original_content": original_content
        }
    
    def validate_user(self, user_context: Dict[str, Any], token: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate user authentication and permissions using unified session tracking
        
        Returns:
            Dict with validation result and updated user_context
        """
        config = self.config.get("user_validation", {})
        
        # If user_context already has authenticated=True (from middleware), trust it
        # but still validate/update session if token is provided
        already_authenticated = user_context.get("authenticated", False) or user_context.get("is_authenticated", False)
        
        # Validate session if token provided
        if token:
            session_result = self.session_manager.validate_session(token, user_context)
            if not session_result["valid"]:
                # If already authenticated from middleware, but token validation fails,
                # this might be a token refresh issue - allow if already authenticated
                if already_authenticated:
                    logger.debug(f"Token validation failed but user already authenticated from middleware")
                    # Keep existing user_context
                elif config.get("require_auth", True) and not config.get("allow_anonymous", False):
                    logger.warning(f"Session validation failed: {session_result.get('error')}")
                    return {
                        "valid": False,
                        "error": session_result.get("error", "Authentication failed"),
                        "user_context": None
                    }
            else:
                # Update user context with validated session info (includes unified session_id)
                user_context = session_result["user_context"]
                already_authenticated = True
        
        # Check if authentication is required
        if config.get("require_auth", True):
            if not already_authenticated and not user_context.get("authenticated", False):
                if not config.get("allow_anonymous", False):
                    logger.warning("Authentication required but user not authenticated")
                    return {
                        "valid": False,
                        "error": "Authentication required",
                        "user_context": None
                    }
        
        # Ensure authenticated flag is set if validation passed
        if not user_context.get("authenticated", False) and already_authenticated:
            user_context["authenticated"] = True
            user_context["is_authenticated"] = True
        
        # Additional validation logic can be added here
        return {
            "valid": True,
            "error": None,
            "user_context": user_context
        }
    
    def sanitize_input(self, input_text: str) -> str:
        """Sanitize user input"""
        # Remove potential script tags and other dangerous content
        sanitized = re.sub(r'<script[^>]*>.*?</script>', '', input_text, flags=re.IGNORECASE | re.DOTALL)
        sanitized = re.sub(r'<[^>]+>', '', sanitized)  # Remove HTML tags
        
        # Limit length
        max_length = 10000
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
        
        return sanitized.strip()
    
    def log_security_event(self, event_type: str, user_id: str, details: Dict[str, Any]):
        """Log security-related events"""
        logger.warning(f"Security event: {event_type} for user {user_id}: {details}")

