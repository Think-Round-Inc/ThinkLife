"""
Security Manager for ThinkxLife Brain
"""

from typing import Dict, Any
from datetime import datetime, timedelta
import logging
import re

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
    
    def check_rate_limit(self, user_id: str) -> bool:
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
    
    def validate_user(self, user_context: Dict[str, Any]) -> bool:
        """Validate user authentication and permissions"""
        config = self.config.get("user_validation", {})
        
        # Check if authentication is required
        if config.get("require_auth", True):
            if not user_context.get("is_authenticated", False):
                if not config.get("allow_anonymous", False):
                    logger.warning("Authentication required but user not authenticated")
                    return False
        
        # Additional validation logic can be added here
        return True
    
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

