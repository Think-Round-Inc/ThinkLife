"""
ThinkLife Backend with Brain Integration
This is the main FastAPI application that integrates the ThinkLife Brain
system with existing chatbot functionality.
"""

import json
import logging
import os
import sys
import time
import uuid
import warnings
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Any, Optional

# patch for Python < 3.10 to support packages_distributions
# This fixes compatibility issues with google-api-core and other libs on Python 3.9
if sys.version_info < (3, 10):
    try:
        import importlib.metadata
        import importlib_metadata
        if not hasattr(importlib.metadata, "packages_distributions"):
            importlib.metadata.packages_distributions = importlib_metadata.packages_distributions
    except ImportError:
        pass

# Suppress Google API Core Python 3.9 EOL warning
warnings.filterwarnings("ignore", message="You are using a Python version.*past its end of life", category=FutureWarning, module="google.api_core")

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

# Configure root logging (correlation-ID-aware) before anything else logs.
# See middleware/logging_middleware.py for why this matters: without it,
# every logger.info() call in this codebase - including the fairly detailed
# ones in Cortex/WorkflowEngine - was silently dropped (root logger default
# level is WARNING with no handler attached).
from middleware.logging_middleware import configure_logging, RequestLoggingMiddleware
configure_logging()

logger = logging.getLogger(__name__)

# Imports
from config import settings
from brain import CortexFlow
from agents.zoe import ZoeService
from agents.zoe.tts_service import tts_service
from middleware.keycloak_auth import KeycloakAuthMiddleware, get_user_context, extract_token_from_header, extract_token_from_cookie, get_user_id
from middleware.validation_middleware import PayloadValidationMiddleware
from brain.guardrails import SessionManager
from plugins import get_plugin_registry, register_builtin_plugins
from background import get_background_queue
from evaluation import evaluate_response

# Global instances
brain_instance = None
zoe_service_instance = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global brain_instance, zoe_service_instance

    logger.info("Starting ThinkLife Backend...")

    # Fail fast on missing required configuration (see config/settings.py).
    # In production this raises and aborts startup; in development/test it
    # only warns, since local/dev environments frequently run with a subset
    # of provider keys configured.
    startup_problems = settings.validate_required_for_startup()
    for problem in startup_problems:
        logger.warning(f"Startup configuration warning: {problem}")
    if startup_problems and settings.environment == "production":
        raise RuntimeError(
            "Aborting startup due to missing required configuration: "
            + "; ".join(startup_problems)
        )

    # Brain Config
    brain_config = {
        "providers": {
            "openai": settings.openai_provider_config(),
        }
    }

    brain_instance = CortexFlow(brain_config)
    await brain_instance.initialize()
    logger.info("CortexFlow initialized")

    zoe_service_instance = ZoeService()
    await zoe_service_instance.initialize()
    logger.info("ZoeService initialized")

    # Plugin discovery (Task 3 - see plugins/registry.py). Additive: this does
    # not replace ZoeService's existing direct import of ZoeAgent, it makes
    # agents additionally look-up-able by id for future multi-agent routing.
    plugin_registry = register_builtin_plugins()
    logger.info(f"Registered plugins: {plugin_registry.list_plugins()}")

    # Background task queue (Task 9) - started before yield so it's ready
    # for the first request; stopped/drained after yield (Task 9.3).
    background_queue = get_background_queue()
    await background_queue.start()

    yield

    logger.info("Shutting down...")
    await background_queue.stop()
    if brain_instance: await brain_instance.shutdown()
    if zoe_service_instance: await zoe_service_instance.shutdown()
    logger.info("Shutdown complete")


app = FastAPI(
    title="ThinkLife Backend with Brain",
    description="AI-powered backend with centralized Brain orchestration",
    version="2.0.0",
    lifespan=lifespan,
)

# Middleware registration order matters: Starlette treats the LAST-added
# middleware as the OUTERMOST layer (it sees the request first, the response
# last). To get an effective request-time order of
#   CORS -> Logging -> Validation -> Auth
# (CORS outermost so its headers apply even to responses rejected further
# in; Logging next so it captures every request including ones Validation or
# Auth reject; Validation before Auth per the Task 5 plan), middleware is
# *added* in the reverse of that order below.
app.add_middleware(KeycloakAuthMiddleware)
app.add_middleware(PayloadValidationMiddleware)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Models
class APIBrainRequest(BaseModel):
    message: str
    application: str
    user_context: Dict[str, Any] = {}
    session_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class APIBrainResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    brain_status: Dict[str, Any]
    timestamp: str


class ChatMessageRequest(BaseModel):
    """
    Shared request body for the chat-style endpoints (`/api/zoe/chat`,
    `/api/chat`, `/api/{app_name}`), replacing the `Dict[str, Any]` bodies
    those routes used to accept and validate by hand (each with its own
    slightly different, duplicated `if not msg.strip(): ...` checks).

    FastAPI validates this automatically and returns a consistent 422 on bad
    input, before the route handler body runs at all - see Task 5.2.
    """
    message: str = Field(..., min_length=1, max_length=10000)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    user_context: Dict[str, Any] = {}
    metadata: Optional[Dict[str, Any]] = None

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("message must not be blank")
        return v


# Dependencies
def get_brain() -> CortexFlow:
    if not brain_instance: raise HTTPException(503, "Brain not initialized")
    return brain_instance

def get_zoe() -> ZoeService:
    if not zoe_service_instance: raise HTTPException(503, "Zoe not initialized")
    return zoe_service_instance


# Brain API
@app.post("/api/brain", response_model=APIBrainResponse)
async def process_brain_request(
    request: APIBrainRequest, 
    brain: CortexFlow = Depends(get_brain),
    http_request: Request = None
):
    try:
        valid_apps = ["healing-rooms", "inside-our-ai", "chatbot", "compliance", "exterior-spaces", "general"]
        if request.application not in valid_apps:
            raise HTTPException(400, f"Invalid app. Must be: {valid_apps}")

        kc_context = get_user_context(http_request) if http_request else {}
        token = extract_token_from_header(http_request) or extract_token_from_cookie(http_request) if http_request else None
        
        merged_context = {**request.user_context, **kc_context}
        metadata = request.metadata or {}
        if token: metadata["token"] = token
        
        data = {
            "id": request.session_id,
            "message": request.message,
            "application": request.application,
            "user_context": merged_context,
            "metadata": metadata,
        }

        res = await brain.process_request(data)

        return APIBrainResponse(
            success=res.get("success", False),
            message=res.get("message"),
            data=res.get("data"),
            error=res.get("error"),
            metadata=res.get("metadata"),
            timestamp=res.get("timestamp", datetime.now().isoformat()),
        )

    except Exception as e:
        logger.exception("Error processing Brain request")
        raise HTTPException(500, str(e))


@app.get("/api/brain/health", response_model=HealthResponse)
async def get_brain_health(brain: CortexFlow = Depends(get_brain)):
    try:
        status = await brain.get_health_status()
        return HealthResponse(
            status=status.get("overall", "unknown"),
            brain_status=status,
            timestamp=datetime.now().isoformat(),
        )
    except Exception as e:
        logger.error(f"Health check error: {e}")
        raise HTTPException(500, str(e))


@app.get("/api/brain/analytics")
async def get_brain_analytics(brain: CortexFlow = Depends(get_brain)):
    try:
        return {
            "success": True,
            "data": await brain.get_analytics(),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Analytics error: {e}")
        raise HTTPException(500, str(e))


# Session API
@app.post("/api/session/logout")
async def logout_session(http_request: Request, session_id: Optional[str] = None):
    try:
        ctx = get_user_context(http_request)
        manager = SessionManager()
        
        sid = session_id or ctx.get("session_id")
        uid = ctx.get("user_id")
        
        token = getattr(http_request.state, "token", None)
        state = None
        if token:
            try:
                import jwt
                state = jwt.decode(token, options={"verify_signature": False}).get("session_state")
            except: pass
        
        success = manager.end_session(session_id=sid, user_id=uid if not sid else None, token_state=state if not sid else None)
        
        return {
            "success": success,
            "message": "Session ended" if success else "Session not found",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Logout error: {e}")
        raise HTTPException(500, str(e))


@app.get("/api/session/stats")
async def get_session_stats():
    return {"success": True, "data": SessionManager().get_session_stats(), "timestamp": datetime.now().isoformat()}


@app.get("/api/session/user/{user_id}")
async def get_user_sessions(user_id: str, include_ended: bool = False):
    try:
        sessions = SessionManager().get_user_sessions(user_id, include_ended=include_ended)
        return {"success": True, "data": {"user_id": user_id, "sessions": sessions, "count": len(sessions)}, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"Get sessions error: {e}")
        raise HTTPException(500, str(e))


@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    session = SessionManager().get_session(session_id)
    if not session: raise HTTPException(404, "Session not found")
    return {"success": True, "data": session.to_dict(), "timestamp": datetime.now().isoformat()}


# Zoe API
@app.post("/api/zoe/chat")
async def zoe_chat_endpoint(
    request: ChatMessageRequest,
    zoe: ZoeService = Depends(get_zoe),
    http_request: Request = None
):
    try:
        kc_context = get_user_context(http_request) if http_request else {}
        kc_uid = get_user_id(http_request) if http_request else None

        msg = request.message
        uid = kc_uid or request.user_id or "anonymous"
        sid = request.session_id
        ctx = {**request.user_context, **kc_context}

        if ctx.get("ace_score", 0) >= 4:
            raise HTTPException(403, "Chat restricted due to ACE score safety policy.")

        # Note: blank/oversized message is now rejected by ChatMessageRequest's
        # own validation (see Field(..., min_length=1, max_length=10000) and
        # message_must_not_be_blank above) before this handler even runs.

        eval_start = time.time()
        result = await zoe.process_message(
            message=msg, user_id=uid, session_id=sid, user_context=ctx, application="chatbot"
        )

        # Post-hoc response evaluation (Task 9.2): evaluate_response()
        # (evaluation/evaluation_manager.py) was previously fully implemented
        # but never actually called anywhere in the codebase. It's exactly
        # the kind of non-critical-path work Task 9 asks to move to the
        # background layer - the user doesn't wait on it, and a failure here
        # can never affect the chat response already returned below.
        if result.get("success") and result.get("response"):
            await get_background_queue().enqueue(
                evaluate_response,
                user_message=msg,
                bot_message=result["response"],
                start_time=eval_start,
                end_time=time.time(),
                context={"user_id": uid, "session_id": sid},
                name="evaluate_response:zoe_chat",
            )

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Zoe chat error")
        raise HTTPException(500, str(e))


@app.post("/api/zoe/chat/stream")
async def zoe_chat_stream_endpoint(
    request: ChatMessageRequest,
    zoe: ZoeService = Depends(get_zoe),
    http_request: Request = None,
):
    """
    Server-Sent Events streaming counterpart to `/api/zoe/chat` (Task 10.4).

    Uses a plain `StreamingResponse` with `media_type="text/event-stream"`
    rather than adding the `sse-starlette` dependency noted as missing back
    in the original repo analysis - hand-rolled SSE framing (`data: ...\\n\\n`)
    is a well-supported, simple pattern that avoids a new dependency for
    what's a small amount of formatting.

    Frame shapes sent as `data: <json>\\n\\n`:
      - {"type": "chunk", "content": "..."}   - one per token/chunk
      - {"type": "done", "session_id": ...}    - once, at the end
      - {"type": "error", "error": "..."}      - once, if something failed

    Auth/validation/logging (Task 10.5): this endpoint sits behind the same
    middleware chain as every other route (CORS -> RequestLoggingMiddleware
    -> PayloadValidationMiddleware -> KeycloakAuthMiddleware, see
    `main.py`'s `add_middleware` calls) - Starlette's `BaseHTTPMiddleware`
    correctly passes a `StreamingResponse` through without buffering its
    body, so auth/validation still run before any token is sent. One
    semantic difference worth calling out: `RequestLoggingMiddleware`'s
    "request complete" log line reflects when the SSE response *object* is
    handed back (i.e. essentially immediately), not when the last token is
    actually sent - that's normal for streaming responses in any ASGI app,
    not a gap specific to this middleware. The generator below logs its own
    stream-level start/end/error separately for that reason.
    """
    kc_context = get_user_context(http_request) if http_request else {}
    kc_uid = get_user_id(http_request) if http_request else None

    msg = request.message
    uid = kc_uid or request.user_id or "anonymous"
    ctx = {**request.user_context, **kc_context}
    # Resolve the session_id here (matching ZoeService.stream_message's own
    # fallback) rather than leaving it to be generated inside stream_message:
    # that generation happens deep inside a method we only iterate, with no
    # way to learn the value it picked afterward - resolving it up front
    # means the 'done' frame below can always report the real session_id.
    sid = request.session_id or ctx.get("session_id") or str(uuid.uuid4())

    if ctx.get("ace_score", 0) >= 4:
        raise HTTPException(403, "Chat restricted due to ACE score safety policy.")

    async def event_stream():
        chunk_count = 0
        try:
            logger.info(f"SSE stream starting for session={sid}")
            async for chunk in zoe.stream_message(
                message=msg, user_id=uid, session_id=sid, user_context=ctx, application="chatbot"
            ):
                chunk_count += 1
                yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
            logger.info(f"SSE stream completed for session={sid} ({chunk_count} chunk(s))")
            yield f"data: {json.dumps({'type': 'done', 'session_id': sid})}\n\n"
        except Exception as e:
            # A failure here may happen after some chunks were already sent
            # (e.g. the provider call fails partway through) - there's no
            # HTTP status code left to change at this point, so the error is
            # reported as a final SSE frame instead. Cleanup of whatever
            # resources were in use (provider client, tool client, data
            # source connection) still happened via the context managers in
            # WorkflowEngine.stream_execution / brain/lifecycle.py before
            # this exception reached here.
            logger.exception(f"SSE stream failed for session={sid} after {chunk_count} chunk(s)")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/zoe/sessions/{session_id}/history")
async def get_zoe_session_history(session_id: str, zoe: ZoeService = Depends(get_zoe)):
    return {
        "success": True, 
        "history": zoe.get_conversation_history(session_id),
        "session_id": session_id,
        "timestamp": datetime.now().isoformat()
    }


@app.delete("/api/zoe/sessions/{session_id}")
async def end_zoe_session(session_id: str, zoe: ZoeService = Depends(get_zoe)):
    zoe.clear_conversation(session_id)
    return {"success": True, "message": "History cleared", "session_id": session_id, "timestamp": datetime.now().isoformat()}


@app.get("/api/zoe/health")
async def get_zoe_health(zoe: ZoeService = Depends(get_zoe)):
    return await zoe.health_check()


# Legacy / App Endpoints
@app.post("/api/chat")
async def legacy_chat_endpoint(request: ChatMessageRequest, zoe: ZoeService = Depends(get_zoe)):
    try:
        msg = request.message
        uid = request.user_id or "anonymous"
        sid = request.session_id
        ctx = request.user_context

        if ctx.get("ace_score", 0) >= 4:
            return {"success": False, "error": "ACE score restriction", "restricted": True}

        eval_start = time.time()
        res = await zoe.process_message(msg, uid, sid, ctx, "chatbot")

        # TTS generation is deliberately NOT moved to the background queue,
        # unlike evaluation below: its output (`audio_data`) is part of this
        # response's contract (the client's avatar/TTS UI expects it inline
        # in the same response), so making it "fire and forget" would mean
        # either breaking that contract or inventing a new delivery
        # mechanism (polling/webhook/websocket) - out of scope for adding a
        # background task layer. It remains synchronous and on the critical
        # path, same as before this change.
        audio = None
        if (ctx.get("avatar_mode") or ctx.get("test_tts")) and res.get("success"):
            audio = await tts_service.generate_speech(res.get("response", ""))

        # Post-hoc response evaluation (Task 9.2) - see the longer comment
        # in zoe_chat_endpoint above for why this one is safe to background.
        if res.get("success") and res.get("response"):
            await get_background_queue().enqueue(
                evaluate_response,
                user_message=msg,
                bot_message=res["response"],
                start_time=eval_start,
                end_time=time.time(),
                context={"user_id": uid, "session_id": sid},
                name="evaluate_response:legacy_chat",
            )

        return {
            "response": res.get("response", ""),
            "success": res.get("success", False),
            "error": res.get("error"),
            "session_id": res.get("session_id"),
            "audio_data": audio,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.exception("Legacy chat error")
        return {"success": False, "error": str(e), "response": "Technical error"}


# Generic App Handlers
@app.post("/api/{app_name}")
async def generic_app_endpoint(app_name: str, request: ChatMessageRequest, brain: CortexFlow = Depends(get_brain)):
    # Validate app_name against allowed list if needed, or let Brain handle it
    req = APIBrainRequest(
        message=request.message,
        application=app_name,
        user_context=request.user_context,
        session_id=request.session_id,
        metadata=request.metadata or {}
    )
    return await process_brain_request(req, brain)

# Explicit mappings for backward compatibility
@app.post("/api/healing-rooms")
async def healing_rooms(r: ChatMessageRequest, b: CortexFlow = Depends(get_brain)): return await generic_app_endpoint("healing-rooms", r, b)

@app.post("/api/inside-our-ai")
async def inside_ai(r: ChatMessageRequest, b: CortexFlow = Depends(get_brain)): return await generic_app_endpoint("inside-our-ai", r, b)

@app.post("/api/compliance")
async def compliance(r: ChatMessageRequest, b: CortexFlow = Depends(get_brain)): return await generic_app_endpoint("compliance", r, b)

@app.post("/api/exterior-spaces")
async def exterior(r: ChatMessageRequest, b: CortexFlow = Depends(get_brain)): return await generic_app_endpoint("exterior-spaces", r, b)


@app.get("/healthz")
async def liveness_check():
    """
    Liveness probe: answers 'is the process up and able to respond at all?'
    Deliberately does NOT check dependencies (Brain, providers, DB, ...) so a
    slow/degraded dependency never causes an orchestrator to kill and restart
    an otherwise-fine process. Use /api/health for a dependency-aware check.
    """
    return {"status": "alive", "timestamp": datetime.now().isoformat()}


@app.get("/health")
@app.get("/api/health")
async def health_check():
    """
    Aggregated readiness check: combines Brain health (providers, workflow
    engine), ZoeService health, and registered-plugin visibility into a
    single response suitable for a load balancer / readiness probe.

    Fixes two pre-existing issues:
    - This endpoint previously only reported whether the brain/zoe *instances*
      existed, not whether they were actually healthy.
    - `/api/health` was on the auth middleware's public allow-list
      (`middleware/keycloak_auth.py`) but no route existed at that path -
      only `/health` did. Both paths now exist and are public.
    """
    brain_health: Dict[str, Any] = {"available": False}
    zoe_health: Dict[str, Any] = {"available": False}

    if brain_instance is not None:
        try:
            brain_health = await brain_instance.get_health_status()
            brain_health["available"] = True
        except Exception as e:
            logger.error(f"Brain health check failed: {e}")
            brain_health = {"available": True, "overall": "unhealthy", "error": str(e)}

    if zoe_service_instance is not None:
        try:
            zoe_health = await zoe_service_instance.health_check()
            zoe_health["available"] = True
        except Exception as e:
            logger.error(f"Zoe health check failed: {e}")
            zoe_health = {"available": True, "initialized": False, "error": str(e)}

    plugins_health = {
        "registered": get_plugin_registry().list_plugins(),
    }

    brain_ok = brain_health.get("available") and brain_health.get("overall") in ("healthy", "degraded")
    zoe_ok = zoe_health.get("available") and zoe_health.get("initialized", False)
    overall = "healthy" if (brain_ok and zoe_ok) else "degraded" if (brain_ok or zoe_ok) else "unhealthy"

    return {
        "status": overall,
        "timestamp": datetime.now().isoformat(),
        "brain": brain_health,
        "zoe": zoe_health,
        "plugins": plugins_health,
    }

@app.get("/")
async def root():
    return {"message": "ThinkLife Backend", "version": settings.app_version}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, reload_includes=["*.py",".env"])
