"""
Security Manager for ThinkLife Brain
"""

from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import logging
import re
from pathlib import Path

from .session_manager import SessionManager
from .pii_detector import PIIDetector 

logger = logging.getLogger(__name__)


class SecurityManager:
    """Manages security, rate limiting, and content filtering"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.rate_limits = {}
        self.blocked_words = self.config.get("content_filtering", {}).get("blocked_words", [])
        # Load blocked terms from the configuration file.
        blocked_terms_path = Path(__file__).resolve().parents[2] / "config" / "blocked_terms.txt"
        if blocked_terms_path.exists():
            with blocked_terms_path.open("r", encoding="utf-8") as f:
                file_blocked_terms = [
                    line.strip()
                    for line in f
                    if line.strip() and not line.strip().startswith("#")
                ]
                self.blocked_words = list(
                    dict.fromkeys(
                        self.blocked_words + file_blocked_terms
                    )
                )
            
        self.trauma_safe_mode = self.config.get("content_filtering", {}).get("trauma_safe_mode", True)
        self.session_manager = SessionManager(self.config.get("session", {}))
        self.pii_detector = PIIDetector() # Initialize the local PII detector.
    
    def _get_default_config(self):
        return {
            "rate_limiting": {"enabled": True, "max_requests_per_minute": 60, "max_requests_per_hour": 1000},
            "content_filtering": {"enabled": True, "blocked_words": [], "trauma_safe_mode": True},
            "user_validation": {"require_auth": True, "allow_anonymous": False}
        }
    
    def check_rate_limit(self, user_id: str, user_context: Optional[Dict[str, Any]] = None) -> bool:
        if not self.config.get("rate_limiting", {}).get("enabled", True):
            return True
        
        now = datetime.now()
        if user_id not in self.rate_limits:
            self.rate_limits[user_id] = {"minute": [], "hour": []}
        
        limits = self.rate_limits[user_id]
        
        # Clean old
        limits["minute"] = [t for t in limits["minute"] if t > now - timedelta(minutes=1)]
        limits["hour"] = [t for t in limits["hour"] if t > now - timedelta(hours=1)]
        
        max_min = self.config.get("rate_limiting", {}).get("max_requests_per_minute", 60)
        max_hour = self.config.get("rate_limiting", {}).get("max_requests_per_hour", 1000)
        
        if len(limits["minute"]) >= max_min:
            logger.warning(f"Rate limit (min) exceeded for {user_id}")
            return False
        
        if len(limits["hour"]) >= max_hour:
            logger.warning(f"Rate limit (hour) exceeded for {user_id}")
            return False
        
        limits["minute"].append(now)
        limits["hour"].append(now)
        return True
    
    def filter_content(self, content: str) -> Dict[str, Any]:
        """
              Filter user content while allowing trauma-related conversations.
            """
    

        if not self.config.get("content_filtering", {}).get("enabled", True):
            return {
                        "safe": True,
                        "content": content,
                        "flags": [],
                        "trauma_detected": False,
                        "original_content": content,
                    }
        filtered_content = content
        blocked_flags = []
        trauma_flags = []

        # Block configured blocked words
        for word in self.blocked_words:
            if not word or not word.strip():
                continue
            term = word.strip()
            pattern = rf"(?<!\w){re.escape(term)}(?!\w)"
            if re.search(pattern, content, flags=re.IGNORECASE):
                blocked_flags.append(term)
                filtered_content = re.sub(
                    pattern,
                    "*" * len(term),
                    filtered_content,
                    flags=re.IGNORECASE,
                )
         

        # Detect trauma-related topics without blocking them.
        if self.trauma_safe_mode:
            trauma_patterns = {
                "suicide": [
                    r"\bsuicide\b",
                    r"\bsuicidal\b",
                ],
                "self-harm": [
                    r"\bself[-\s]?harm\b",
                    r"\bself[-\s]?harming\b",
                ],
                "abuse": [
                    r"\babuse\b",
                    r"\babused\b",
                    r"\babusive\b",
                ],
                "violence": [
                    r"\bviolence\b",
                    r"\bviolent\b",
                ],
                "trauma": [
                    r"\btrauma\b",
                    r"\btraumatic\b",
                    r"\btraumatized\b",
                ],
                 "ptsd": [
                     r"\bptsd\b",
                     r"\bpost[-\s]?traumatic stress\b",
                 ],
                 "depression": [
                     r"\bdepression\b",
                     r"\bdepressed\b",
                     r"\bdepressive\b",
                 ],
                 "anxiety": [
                     r"\banxiety\b",
                     r"\banxious\b",
                 ],
            }
            for category, patterns in trauma_patterns.items():
                if any(
                    re.search(pattern, content, flags=re.IGNORECASE)
                    for pattern in patterns
                ):
                    trauma_flags.append(category)
                    


        return {
            "safe": len(blocked_flags) == 0,
            "content": filtered_content,
            "flags": blocked_flags,
            "trauma_detected": len(trauma_flags) > 0,
            "trauma_flags": trauma_flags,
            "original_content": content,
        }
      
    def validate_user(self, user_context: Dict[str, Any], token: Optional[str] = None) -> Dict[str, Any]:
        config = self.config.get("user_validation", {})
        auth = user_context.get("authenticated", False) or user_context.get("is_authenticated", False)

        # Validate supplied session tokens and reject invalid tokens. 
        if token:
            res = self.session_manager.validate_session(token, user_context)
            if not res["valid"]:
                return {
                    "valid": False,
                    "error": res.get("error", "Authentication failed"),
                    "user_context": None,
                }
            user_context = res["user_context"]
            auth = True
        
        if config.get("require_auth", True) and not auth and not config.get("allow_anonymous", False):
            return {"valid": False, "error": "Authentication required", "user_context": None}
        
        if auth and not user_context.get("authenticated"):
            user_context["authenticated"] = True
            user_context["is_authenticated"] = True
        
        return {"valid": True, "error": None, "user_context": user_context}
    
    def sanitize_input(self, input_text: str) -> str:
        # Basic sanitization
        s = re.sub(r'<script[^>]*>.*?</script>', '', input_text, flags=re.IGNORECASE | re.DOTALL)
        s = re.sub(r'<[^>]+>', '', s)
        return s[:10000].strip()

    # Detect and redact sensitive information from text.
    def redact_sensitive_data(self, text: str) -> Dict[str, Any]:
        """
        Detect and redact sensitive data from text.

        Returns the redacted text and the types of sensitive
        information that were detected.
        """
        if not text:
            return {
                "content": text,
                "redacted": False,
                "redaction_types": [],
                "redaction_count": 0,
            }

        redacted_text = text
        redaction_types = []
        redaction_count = 0

        patterns = {
            "email": (
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
                "[REDACTED_EMAIL]",
            ),
            "ssn": (
                r"\b\d{3}-\d{2}-\d{4}\b",
                "[REDACTED_SSN]",
            ),
            "credit_card": (
                r"\b(?:\d[ -]*?){13,19}\b",
                "[REDACTED_CARD]",
            ),
            "bearer_token": (
                r"\bBearer\s+[A-Za-z0-9\-._~+/]+=*\b",
                "Bearer [REDACTED_TOKEN]",
            ),
            "api_key": (
                r"\b(?:sk|pk|api)[-_][A-Za-z0-9_-]{16,}\b",
                "[REDACTED_API_KEY]",
            ),
             "phone": (
                 r"(?<!\d)(?:\+?1[-.\s]?)?"
                 r"(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}(?!\d)",
                 "[REDACTED_PHONE]",
             ),
                 
                            
        }
 
        for data_type, (pattern, replacement) in patterns.items():
            redacted_text, matches = re.subn(
                pattern,
                replacement,
                redacted_text,
                flags=re.IGNORECASE,
            )

            if matches > 0:
                redaction_types.append(data_type)
                redaction_count += matches

        # Detect and redact additional PII using Presidio.
        try:
            pii_redacted_text = self.pii_detector.redact(redacted_text)
            if pii_redacted_text != redacted_text:
                redaction_types.append("pii")
                redaction_count += 1
            redacted_text = pii_redacted_text
        except Exception as e:
            logger.warning(
                "PII detector failed; continuing with regex redaction only: %s",
                e,
            )
        

        return {
            "content": redacted_text,
            "redacted": redaction_count > 0,
            "redaction_types": redaction_types,
            "redaction_count": redaction_count,
        }
    def log_security_event(self, event_type: str, user_id: str, details: Dict[str, Any]):
        logger.warning(f"Security event: {event_type} user={user_id} details={details}")
