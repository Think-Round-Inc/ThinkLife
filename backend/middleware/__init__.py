"""
Middleware module for ThinkLife Backend
"""

from .keycloak_auth import (
    KeycloakAuthMiddleware,
    get_user_context,
    get_user_id,
    is_authenticated,
    extract_token_from_header,
    extract_token_from_cookie,
)
from .logging_middleware import (
    RequestLoggingMiddleware,
    configure_logging,
    correlation_id_var,
)
from .validation_middleware import PayloadValidationMiddleware

__all__ = [
    "KeycloakAuthMiddleware",
    "get_user_context",
    "get_user_id",
    "is_authenticated",
    "extract_token_from_header",
    "extract_token_from_cookie",
    "RequestLoggingMiddleware",
    "configure_logging",
    "correlation_id_var",
    "PayloadValidationMiddleware",
]

