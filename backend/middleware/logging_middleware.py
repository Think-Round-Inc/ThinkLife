"""
Request logging middleware + logging configuration.

Two things live here, deliberately together:

1. `RequestLoggingMiddleware` - logs every request/response with a
   correlation ID, method, path, status code, and duration.
2. `configure_logging()` - wires that correlation ID into every OTHER log
   line in the app too (the many `logger.info(...)` calls already scattered
   through `brain/cortex/cortex.py`, `brain/cortex/workflow_engine.py`, etc.)
   via a `logging.Filter`, so a single request can be traced end-to-end.

Why `configure_logging()` matters, concretely: this codebase never called
`logging.basicConfig()` (or any other root logging configuration) anywhere.
Python's root logger defaults to level WARNING with no handler attached in
that situation, which means every `logger.info(...)` call in the codebase -
including the fairly detailed ones in Cortex and the workflow engine - was
being silently dropped and never actually reached stdout/any log sink.
`configure_logging()` fixes that as a side effect of adding correlation IDs.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextvars import ContextVar
from typing import Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from config import settings

logger = logging.getLogger(__name__)

# Holds the current request's correlation ID for the lifetime of that
# request's asyncio task. Starlette/FastAPI runs each request in its own
# task, so contextvars propagate correctly down through Cortex/WorkflowEngine
# without needing to pass a correlation id argument through every call.
correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="-")

CORRELATION_ID_HEADER = "X-Correlation-ID"


class CorrelationIdLogFilter(logging.Filter):
    """Injects the current request's correlation ID into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = correlation_id_var.get()
        return True


def configure_logging(level: Optional[str] = None) -> None:
    """
    Configure root logging once, at app startup.

    Idempotent: safe to call more than once (e.g. once from `main.py` import
    time and again in tests) - it clears any handlers it previously added
    before reinstalling them, rather than stacking duplicate handlers.
    """
    root = logging.getLogger()
    log_level = getattr(logging, (level or settings.log_level).upper(), logging.INFO)
    root.setLevel(log_level)

    # Remove any handler this function previously installed (idempotency),
    # identified by a marker attribute, without touching handlers other code
    # (e.g. uvicorn) may have installed.
    for h in list(root.handlers):
        if getattr(h, "_thinklife_logging_handler", False):
            root.removeHandler(h)

    handler = logging.StreamHandler()
    handler._thinklife_logging_handler = True  # type: ignore[attr-defined]
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s [%(correlation_id)s] %(name)s: %(message)s"
        )
    )
    handler.addFilter(CorrelationIdLogFilter())
    root.addHandler(handler)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Logs one line on request start and one on request completion (or
    failure), tagged with a correlation ID that's also propagated to every
    other logger in the app for the duration of the request (see
    `correlation_id_var` / `CorrelationIdLogFilter` above) and returned to
    the client via the `X-Correlation-ID` response header.

    Deliberately does NOT catch/swallow exceptions: it logs that one
    occurred (with the correlation ID and duration so far) and re-raises, so
    downstream error handling (FastAPI's exception handlers, Starlette's
    `ExceptionMiddleware`) still runs exactly as it did before this
    middleware was added.
    """

    async def dispatch(self, request: Request, call_next):
        incoming_id = request.headers.get(CORRELATION_ID_HEADER)
        correlation_id = incoming_id or str(uuid.uuid4())
        token = correlation_id_var.set(correlation_id)
        request.state.correlation_id = correlation_id

        start = time.perf_counter()
        logger.info(f"--> {request.method} {request.url.path}")

        try:
            response = await call_next(request)
        except Exception:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.exception(
                f"<-- {request.method} {request.url.path} failed after {duration_ms:.1f}ms"
            )
            raise
        else:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.info(
                f"<-- {request.method} {request.url.path} {response.status_code} "
                f"({duration_ms:.1f}ms)"
            )
            response.headers[CORRELATION_ID_HEADER] = correlation_id
            return response
        finally:
            correlation_id_var.reset(token)
