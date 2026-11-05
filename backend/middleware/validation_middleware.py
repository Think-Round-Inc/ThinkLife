"""
Payload validation middleware.

This handles the transport-level half of "reject malformed payloads early
and consistently" (Epic 5 / Story 5.1, Task 5.2): request body too large, or
a JSON Content-Type whose body doesn't actually parse as JSON. Both are
rejected here, before any route handler (or FastAPI's own per-field
validation) runs, with a consistent 400/413 response shape.

The other half - validating the *fields* of a request body (message
required, non-empty, under a max length, etc.) - is handled per-route by
FastAPI's native Pydantic validation. See `ChatMessageRequest` in `main.py`,
which replaced the `Dict[str, Any]` parameters that `zoe_chat_endpoint`,
`legacy_chat_endpoint`, and `generic_app_endpoint` used to accept and
manually (and inconsistently) check by hand.
"""

from __future__ import annotations

import json
import logging

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from config import settings

logger = logging.getLogger(__name__)

# Methods that may carry a body worth validating. GET/HEAD/DELETE bodies are
# not something this app's routes use.
_BODY_METHODS = {"POST", "PUT", "PATCH"}


class PayloadValidationMiddleware(BaseHTTPMiddleware):
    """
    For POST/PUT/PATCH requests:
    - Rejects bodies larger than `settings.max_request_body_bytes` (413).
    - If `Content-Type` claims JSON, rejects bodies that don't actually parse
      as JSON (400) - instead of letting that surface later as a less clear
      error (or, for the `Dict[str, Any]`-typed routes prior to this change,
      an unhandled exception).

    Bodies are read via `await request.body()`, which Starlette caches -
    downstream code (including FastAPI's own request-body parsing) can still
    read the same bytes afterward without them being consumed twice.
    """

    async def dispatch(self, request: Request, call_next):
        if request.method in _BODY_METHODS:
            content_length = request.headers.get("content-length")
            if content_length is not None:
                try:
                    if int(content_length) > settings.max_request_body_bytes:
                        return JSONResponse(
                            status_code=413,
                            content={
                                "success": False,
                                "error": "Request body too large",
                                "max_bytes": settings.max_request_body_bytes,
                            },
                        )
                except ValueError:
                    pass  # malformed header; let downstream handle it

            content_type = request.headers.get("content-type", "")
            if "application/json" in content_type:
                body = await request.body()
                if body:
                    try:
                        json.loads(body)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        logger.info(
                            f"Rejecting malformed JSON body on {request.method} {request.url.path}"
                        )
                        return JSONResponse(
                            status_code=400,
                            content={"success": False, "error": "Request body is not valid JSON"},
                        )

        return await call_next(request)
