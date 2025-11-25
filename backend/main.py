"""
ThinkxLife Backend with Brain Integration

This is the main FastAPI application that integrates the ThinkxLife Brain
system with existing chatbot functionality.
"""

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Any, Optional, List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langfuse import Langfuse, get_client
from evaluation import run_evaluation


# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import Brain system
from brain import CortexFlow

# Import Zoe AI Companion
from agents.zoe import ZoeCore

# Import the Agent Orchestrator
from agents.bard.orchestrator.orchestra import Orchestrator, get_llm

# Import TTS Service
from agents.zoe.tts_service import tts_service

# Global instances
brain_instance = None
zoe_instance = None

langfuse = get_client()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    global brain_instance, zoe_instance, orchestrator_instance

    # Startup
    logger.info("Starting ThinkxLife Backend with Brain and Zoe integration...")

    # Initialize Brain
    brain_config = {
        "providers": {
            "openai": {
                "enabled": bool(os.getenv("OPENAI_API_KEY")),
                "api_key": os.getenv("OPENAI_API_KEY"),
                "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                "max_tokens": int(os.getenv("OPENAI_MAX_TOKENS", "2000")),
                "temperature": float(os.getenv("OPENAI_TEMPERATURE", "0.7")),
            }
        }
    }

    brain_instance = CortexFlow(brain_config)
    logger.info("Brain system initialized")

    # Initialize Zoe with Brain integration
    zoe_instance = ZoeCore(brain_instance)
    logger.info("Zoe AI Companion initialized")

    orchestrator_instance = Orchestrator(llm=get_llm())
    logger.info("Agent Orchestrator initialized")

    yield

    # Shutdown
    logger.info("Shutting down ThinkxLife Backend...")
    if brain_instance:
        await brain_instance.shutdown()
    logger.info("Shutdown complete")

    if orchestrator_instance:
        await orchestrator_instance.close()

    logger.info("Shutdown complete")


# Initialize FastAPI app
app = FastAPI(
    title="ThinkxLife Backend with Brain",
    description="AI-powered backend with centralized Brain orchestration",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "https://thinkxlife.vercel.app",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# LangFuse connection (logging and tracing)
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host="https://cloud.langfuse.com",
)


# Pydantic models for API validation
class APIBrainRequest(BaseModel):
    """API Brain request model"""

    message: str
    application: str
    user_context: Dict[str, Any] = {}
    session_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class APIBrainResponse(BaseModel):
    """API Brain response model"""

    success: bool
    message: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response"""

    status: str
    brain_status: Dict[str, Any]
    timestamp: str


class OrchestratorRequest(BaseModel):
    task: str
    session_id: Optional[str] = None


class OrchestratorResponse(BaseModel):
    output: str
    session_id: str
    agent_name: str
    trace: Optional[Dict[str, Any]] = None
    history: Optional[List[Dict[str, Any]]] = None


def get_brain() -> ThinkxLifeBrain:
    """Get Brain instance"""
    if not brain_instance:
        raise HTTPException(status_code=503, detail="Brain system not initialized")
    return brain_instance


def get_zoe() -> ZoeCore:
    """Get Zoe instance"""
    if not zoe_instance:
        raise HTTPException(status_code=503, detail="Zoe system not initialized")
    return zoe_instance


# Brain API endpoints
@app.options("/api/brain")
async def brain_options():
    """Handle CORS preflight requests for brain endpoint"""
    return {"message": "OK"}


@app.post("/api/brain", response_model=APIBrainResponse)
async def process_brain_request(
    request: APIBrainRequest, brain: ThinkxLifeBrain = Depends(get_brain)
) -> APIBrainResponse:
    """
    Process a request through the ThinkxLife Brain system

    This is the main endpoint that all frontend applications use
    to interact with AI capabilities.
    """
    try:
        # Validate application type
        valid_applications = [
            "healing-rooms",
            "inside-our-ai",
            "chatbot",
            "compliance",
            "exterior-spaces",
            "general",
            "agent_orchestrator",
        ]

        if request.application not in valid_applications:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid application. Must be one of: {valid_applications}",
            )

        # Prepare Brain request
        brain_request_data = {
            "id": request.session_id,
            "message": request.message,
            "application": request.application,
            "user_context": request.user_context,
            "metadata": request.metadata or {},
        }

        # Process with Brain
        response_data = await brain.process_request(brain_request_data)

        # Return formatted response
        return APIBrainResponse(
            success=response_data.get("success", False),
            message=response_data.get("message"),
            data=response_data.get("data"),
            error=response_data.get("error"),
            metadata=response_data.get("metadata"),
            timestamp=response_data.get("timestamp", datetime.now().isoformat()),
        )

    except Exception as e:
        logger.error(f"Error processing Brain request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/brain/health", response_model=HealthResponse)
async def get_brain_health(
    brain: ThinkxLifeBrain = Depends(get_brain),
) -> HealthResponse:
    """Get Brain system health status"""
    try:
        health_status = await brain.get_health_status()

        return HealthResponse(
            status=health_status.get("overall", "unknown"),
            brain_status=health_status,
            timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        logger.error(f"Error getting Brain health: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/brain/analytics")
async def get_brain_analytics(brain: ThinkxLifeBrain = Depends(get_brain)):
    """Get Brain system analytics"""
    try:
        analytics = await brain.get_analytics()
        return {
            "success": True,
            "data": analytics,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error getting Brain analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Zoe AI Companion endpoints
@app.options("/api/zoe/chat")
async def zoe_chat_options():
    """Handle CORS preflight requests for Zoe chat endpoint"""
    return {"message": "OK"}


@app.post("/api/zoe/chat")
async def zoe_chat_endpoint(request: Dict[str, Any], zoe: ZoeCore = Depends(get_zoe)):
    """
    Chat with Zoe AI Companion

    This endpoint provides access to Zoe, the empathetic AI companion
    that integrates with the Brain system for LLM calls.
    """
    try:
        # Validate request structure
        if not isinstance(request, dict):
            raise HTTPException(status_code=400, detail="Invalid request format")

        # Extract request data
        message = request.get("message", "")
        user_id = request.get("user_id", "anonymous")
        session_id = request.get("session_id")
        user_context = request.get("user_context", {})

        # Check ACE score restriction - prevent chat access for scores >= 4
        ace_score = user_context.get("ace_score", 0)
        if ace_score >= 4:
            raise HTTPException(
                status_code=403,
                detail="Chat access is restricted for your safety. Please contact info@thinkround.org to learn more about our Trauma Transformation Training program.",
            )

        # Validate required fields
        if not message or not message.strip():
            raise HTTPException(
                status_code=400, detail="Message is required and cannot be empty"
            )

        if len(message) > 10000:
            raise HTTPException(
                status_code=400, detail="Message too long (max 10,000 characters)"
            )

        with langfuse.start_as_current_span(name="zoe-chat-interaction") as root_span:
            # Run the main processing inside the span for better tracing
            response = await zoe.process_message(
                message=message,
                user_context=user_context,
                application="chatbot",
                session_id=session_id,
                user_id=user_id,
            )

            # Update trace attributes
            root_span.update_trace(
                session_id=session_id,
                input={"user_input": message, "user_context": user_context},
                output={"bot_response": response.get("response", "")},
                metadata={"user_id": user_id},
            )

            # Fetch trace_id and run evaluator inside the active context
            trace_id = langfuse.get_current_trace_id()
            if trace_id:
                eval_results = {}
                user_input = message
                bot_message = response.get("response", "")

                eval_results = await run_evaluation(user_input, bot_message)

                # Optionally log the evaluator results into the same span
                root_span.update_trace(metadata={"evaluation_results": eval_results})

        return response

    except Exception as e:
        logger.error(f"Error in Zoe chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/zoe/sessions/{user_id}")
async def get_zoe_user_sessions(
    user_id: str, limit: int = 10, zoe: ZoeCore = Depends(get_zoe)
):
    """Get recent Zoe sessions for a user"""
    try:
        sessions = await zoe.get_user_sessions(user_id, limit)
        return {
            "success": True,
            "sessions": sessions,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error getting Zoe user sessions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/zoe/sessions/{session_id}/history")
async def get_zoe_session_history(session_id: str, zoe: ZoeCore = Depends(get_zoe)):
    """Get conversation history for a Zoe session"""
    try:
        history = await zoe.get_session_history(session_id)
        return {
            "success": True,
            "history": history,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error getting Zoe session history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/zoe/sessions/{session_id}")
async def end_zoe_session(session_id: str, zoe: ZoeCore = Depends(get_zoe)):
    """End a Zoe conversation session"""
    try:
        await zoe.end_session(session_id)
        return {
            "success": True,
            "message": "Session ended successfully",
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error ending Zoe session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/zoe/health")
async def get_zoe_health(zoe: ZoeCore = Depends(get_zoe)):
    """Get Zoe health status"""
    try:
        health = await zoe.health_check()
        return health
    except Exception as e:
        logger.error(f"Error getting Zoe health: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/session-analytics")
async def get_session_analytics(
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    brain: ThinkxLifeBrain = Depends(get_brain),
    zoe: ZoeCore = Depends(get_zoe),
):
    """Get session analytics and statistics"""
    try:
        # Get Brain analytics
        brain_analytics = await brain.get_analytics()

        # Get session-specific data if requested
        session_data = {}
        if session_id:
            session_data = await zoe.get_session_history(session_id)

        return {
            "success": True,
            "data": {
                "brain_analytics": brain_analytics,
                "session_data": session_data,
                "user_id": user_id,
                "session_id": session_id,
            },
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error getting session analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Legacy chatbot endpoints (redirects to Zoe for backward compatibility)
@app.options("/api/chat")
async def chat_options():
    """Handle CORS preflight requests for legacy chat endpoint"""
    return {"message": "OK"}


@app.post("/api/chat")
async def legacy_chat_endpoint(
    request: Dict[str, Any], zoe: ZoeCore = Depends(get_zoe)
):
    """
    Legacy chat endpoint for backward compatibility

    This endpoint maintains compatibility with existing frontend code
    while routing through Zoe AI Companion with full conversation management.
    """
    try:
        # Extract request data
        message = request.get("message", "")
        user_id = request.get("user_id", "anonymous")
        session_id = request.get("session_id")
        user_context = request.get("user_context", {})

        # Check ACE score restriction - prevent chat access for scores >= 4
        ace_score = user_context.get("ace_score", 0)
        if ace_score >= 4:
            return {
                "response": "Chat access is restricted for your safety. Please contact info@thinkround.org to learn more about our Trauma Transformation Training program.",
                "success": False,
                "error": "ACE score restriction",
                "restricted": True,
                "timestamp": datetime.now().isoformat(),
            }

        # Validate required fields
        if not message or not message.strip():
            return {
                "response": "I didn't receive a message. What would you like to talk about?",
                "success": False,
                "error": "No message provided",
                "timestamp": datetime.now().isoformat(),
            }

        if len(message) > 10000:
            return {
                "response": "Your message is too long. Please keep it under 10,000 characters.",
                "success": False,
                "error": "Message too long",
                "timestamp": datetime.now().isoformat(),
            }

        # Process through Zoe with conversation management
        zoe_response = await zoe.process_message(
            message=message,
            user_context=user_context,
            application="chatbot",
            session_id=session_id,
            user_id=user_id,
        )

        # Generate TTS audio if avatar mode is enabled
        audio_data = None
        avatar_mode = user_context.get("avatar_mode", False)
        test_tts = user_context.get("test_tts", False)

        if (avatar_mode or test_tts) and zoe_response.get("success", False):
            response_text = zoe_response.get("response", "")
            if response_text:
                audio_data = await tts_service.generate_speech(response_text)
                if audio_data:
                    logger.info("TTS audio generated successfully")
                else:
                    logger.warning("TTS audio generation failed")

        # Transform to legacy response format
        response_data = {
            "response": zoe_response.get("response", ""),
            "success": zoe_response.get("success", False),
            "error": zoe_response.get("error"),
            "session_id": zoe_response.get("session_id"),
            "timestamp": zoe_response.get("timestamp", datetime.now().isoformat()),
        }

        # Add audio data if generated
        if audio_data:
            response_data["audio_data"] = audio_data

        return response_data

    except Exception as e:
        logger.error(f"Error in legacy chat endpoint: {str(e)}")
        return {
            "response": "I apologize, but I'm experiencing technical difficulties. I'm still here for you though.",
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


# Helper function for application-specific endpoints
async def create_application_endpoint(
    request: Dict[str, Any],
    application: str,
    brain: ThinkxLifeBrain = Depends(get_brain),
):
    """Generic handler for application-specific endpoints"""
    api_request = APIBrainRequest(
        message=request.get("message", ""),
        application=application,
        user_context=request.get("user_context", {}),
        session_id=request.get("session_id"),
        metadata=request.get("metadata", {}),
    )

    return await process_brain_request(api_request, brain)


# Application-specific endpoints
@app.post("/api/healing-rooms")
async def healing_rooms_endpoint(
    request: Dict[str, Any], brain: ThinkxLifeBrain = Depends(get_brain)
):
    """Healing rooms specific endpoint"""
    return await create_application_endpoint(request, "healing-rooms", brain)


@app.post("/api/inside-our-ai")
async def inside_our_ai_endpoint(
    request: Dict[str, Any], brain: ThinkxLifeBrain = Depends(get_brain)
):
    """Inside our AI specific endpoint"""
    return await create_application_endpoint(request, "inside-our-ai", brain)


@app.post("/api/compliance")
async def compliance_endpoint(
    request: Dict[str, Any], brain: ThinkxLifeBrain = Depends(get_brain)
):
    """Compliance specific endpoint"""
    return await create_application_endpoint(request, "compliance", brain)


@app.post("/api/exterior-spaces")
async def exterior_spaces_endpoint(
    request: Dict[str, Any], brain: ThinkxLifeBrain = Depends(get_brain)
):
    """Exterior spaces specific endpoint"""
    return await create_application_endpoint(request, "exterior-spaces", brain)


# Health check endpoint
@app.get("/health")
async def health_check():
    """Basic health check"""
    return {
        "status": "healthy",
        "service": "ThinkxLife Backend with Brain and Zoe",
        "timestamp": datetime.now().isoformat(),
        "brain_available": brain_instance is not None,
        "zoe_available": zoe_instance is not None,
    }


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "ThinkxLife Backend with Brain and Zoe Integration",
        "version": "1.0.0",
        "brain_enabled": brain_instance is not None,
        "zoe_enabled": zoe_instance is not None,
        "endpoints": {
            "brain": "/api/brain",
            "brain_health": "/api/brain/health",
            "brain_analytics": "/api/brain/analytics",
            "zoe": {
                "chat": "/api/zoe/chat",
                "health": "/api/zoe/health",
                "sessions": "/api/zoe/sessions/{user_id}",
                "session_history": "/api/zoe/sessions/{session_id}/history",
            },
            "applications": {
                "healing_rooms": "/api/healing-rooms",
                "inside_our_ai": "/api/inside-our-ai",
                "compliance": "/api/compliance",
                "exterior_spaces": "/api/exterior-spaces",
                "chatbot": "/api/chat",  # Legacy endpoint, routes to Zoe
            },
        },
    }


# Agent Orchestrator


@app.post("/api/agent/orchestrator/chat", response_model=OrchestratorResponse)
async def orchestrator_chat(request: OrchestratorRequest):
    """
    Main entrypoint to run Agent Orchestrator.
    Handles routing, new/existing sessions, tracing, and history internally.
    """
    if orchestrator_instance is None:
        raise HTTPException(status_code=500, detail="Orchestrator is not initialized")

    try:
        result = await orchestrator_instance.orchestrate(
            Input=request.task,
            session_id=request.session_id,  # orchestrator handles new/existing
        )

        return OrchestratorResponse(**result)

    except Exception as e:
        logger.exception("Error running orchestrator")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/agent/orchestrator/state/{session_id}")
async def orchestrator_state(session_id: str):
    """
    Get the current state of any orchestrated agent session.
    """
    if orchestrator_instance is None:
        raise HTTPException(status_code=500, detail="Orchestrator is not initialized")

    if session_id not in orchestrator_instance.sessions:
        raise HTTPException(status_code=404, detail="Unknown session_id")

    runner = orchestrator_instance.sessions[session_id]
    try:
        snapshot = await runner.get_current_state(session_id)
        return {"session_id": session_id, "state": snapshot}
    except Exception as e:
        logger.exception("Error getting session state")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    # Run the server
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
