"""
Guardrails module for ThinkxLife Brain
Handles authentication, rate limiting, content filtering, and security validation
"""

from .security_manager import SecurityManager

__all__ = [
    "SecurityManager",
]

