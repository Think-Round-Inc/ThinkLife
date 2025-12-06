"""
Keycloak Authentication Middleware
Extracts and validates Keycloak tokens from requests
"""

import logging
from typing import Optional, Dict, Any
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware

from brain.guardrails import SessionManager

logger = logging.getLogger(__name__)


def extract_token_from_header(request: Request) -> Optional[str]:
    """Extract Bearer token from Authorization header"""
    authorization = request.headers.get("Authorization")
    if not authorization:
        return None
    
    if authorization.startswith("Bearer "):
        return authorization[7:]  # Remove "Bearer " prefix
    
    return None


def extract_token_from_cookie(request: Request) -> Optional[str]:
    """Extract token from cookie"""
    return request.cookies.get("keycloak_token") or request.cookies.get("access_token")


class KeycloakAuthMiddleware(BaseHTTPMiddleware):
    """
    Middleware to extract and validate Keycloak tokens
    Adds validated user context to request state
    """
    
    def __init__(self, app, session_manager: Optional[SessionManager] = None):
        super().__init__(app)
        self.session_manager = session_manager or SessionManager()
        # Paths that don't require authentication
        self.public_paths = [
            "/api/health",
            "/docs",
            "/openapi.json",
            "/redoc",
        ]
    
    async def dispatch(self, request: Request, call_next):
        # Skip auth for public paths
        if any(request.url.path.startswith(path) for path in self.public_paths):
            return await call_next(request)
        
        # Extract token from header or cookie
        token = extract_token_from_header(request) or extract_token_from_cookie(request)
        
        # Initialize user context
        user_context = {}
        
        if token:
            # Validate token
            session_result = self.session_manager.validate_session(token)
            
            if session_result["valid"]:
                user_context = session_result["user_context"]
                logger.debug(f"Valid session for user: {user_context.get('user_id')}")
            else:
                logger.warning(f"Invalid token: {session_result.get('error')}")
                # Don't fail the request, just log - let endpoints decide if auth is required
        else:
            logger.debug("No token provided in request")
        
        # Add user context to request state
        request.state.user_context = user_context
        request.state.authenticated = user_context.get("authenticated", False)
        request.state.user_id = user_context.get("user_id", "anonymous")
        request.state.token = token
        
        return await call_next(request)


def get_user_context(request: Request) -> Dict[str, Any]:
    """Helper function to get user context from request"""
    return getattr(request.state, "user_context", {})


def get_user_id(request: Request) -> str:
    """Helper function to get user ID from request"""
    return getattr(request.state, "user_id", "anonymous")


def is_authenticated(request: Request) -> bool:
    """Helper function to check if request is authenticated"""
    return getattr(request.state, "authenticated", False)

