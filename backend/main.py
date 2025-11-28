"""
ThinkLife Backend with Brain Integration

This is the main FastAPI application that integrates the ThinkLife Brain
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


# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import Brain system
from brain import CortexFlow

# Import Zoe AI Companion
from agents.zoe import ZoeService

# Import the Agent Orchestrator
from agents.bard.orchestrator.orchestra import Orchestrator, get_llm

# Import TTS Service
from agents.zoe.tts_service import tts_service

# Global instances
brain_instance = None
zoe_service_instance = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    global brain_instance, zoe_service_instance, orchestrator_instance

    # Startup
    logger.info("Starting ThinkLife Backend with Brain and Zoe integration...")

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
    await brain_instance.initialize()
    logger.info("Brain system (CortexFlow) initialized")

    # Initialize Zoe Service
    zoe_service_instance = ZoeService()
    await zoe_service_instance.initialize()
    logger.info("Zoe AI Companion initialized (Plugin architecture)")

    orchestrator_instance = Orchestrator(llm=get_llm())
    logger.info("Agent Orchestrator initialized")

    yield

    # Shutdown
    logger.info("Shutting down ThinkLife Backend...")
    if brain_instance:
        await brain_instance.shutdown()
    
    if zoe_service_instance:
        await zoe_service_instance.shutdown()

    if orchestrator_instance:
        await orchestrator_instance.close()

    logger.info("Shutdown complete")


# Initialize FastAPI app
app = FastAPI(
    title="ThinkLife Backend with Brain",
    description="AI-powered backend with centralized Brain orchestration",
    version="2.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "https://thinklife.vercel.app",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
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


def get_brain() -> CortexFlow:
    """Get Brain instance"""
    if not brain_instance:
        raise HTTPException(status_code=503, detail="Brain system not initialized")
    return brain_instance


def get_zoe() -> ZoeService:
    """Get Zoe service instance"""
    if not zoe_service_instance:
        raise HTTPException(status_code=503, detail="Zoe system not initialized")
    return zoe_service_instance


# Brain API endpoints
@app.options("/api/brain")
async def brain_options():
    """Handle CORS preflight requests for brain endpoint"""
    return {"message": "OK"}


@app.post("/api/brain", response_model=APIBrainResponse)
async def process_brain_request(
    request: APIBrainRequest, brain: CortexFlow = Depends(get_brain)
) -> APIBrainResponse:
    """
    Process a request through the ThinkLife Brain system

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
    brain: CortexFlow = Depends(get_brain),
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
async def get_brain_analytics(brain: CortexFlow = Depends(get_brain)):
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
async def zoe_chat_endpoint(request: Dict[str, Any], zoe: ZoeService = Depends(get_zoe)):
    """
    Chat with Zoe AI Companion

    This endpoint provides access to Zoe, the empathetic AI companion.
    All LLM processing goes through: ZoeService - Plugin - CortexFlow - WorkflowEngine
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

        # Process through plugin architecture
        response = await zoe.process_message(
            message=message,
            user_id=user_id,
            session_id=session_id,
            user_context=user_context,
            application="chatbot"
        )

        return response

    except Exception as e:
        logger.error(f"Error in Zoe chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/zoe/sessions/{user_id}")
async def get_zoe_user_sessions(
    user_id: str, limit: int = 10, zoe: ZoeService = Depends(get_zoe)
):
    """Get recent Zoe sessions for a user"""
    try:
        # Note: This method needs to be implemented in ZoeService if needed
        # For now, return empty list
        return {
            "success": True,
            "sessions": [],
            "message": "Session listing not yet implemented",
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error getting Zoe user sessions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/zoe/sessions/{session_id}/history")
async def get_zoe_session_history(session_id: str, zoe: ZoeService = Depends(get_zoe)):
    """Get conversation history for a Zoe session"""
    try:
        history = zoe.get_conversation_history(session_id)
        return {
            "success": True,
            "history": history,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error getting Zoe session history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/zoe/sessions/{session_id}")
async def end_zoe_session(session_id: str, zoe: ZoeService = Depends(get_zoe)):
    """End a Zoe conversation session"""
    try:
        zoe.clear_conversation(session_id)
        return {
            "success": True,
            "message": "Session ended successfully",
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error ending Zoe session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/zoe/health")
async def get_zoe_health(zoe: ZoeService = Depends(get_zoe)):
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
    brain: CortexFlow = Depends(get_brain),
    zoe: ZoeService = Depends(get_zoe),
):
    """Get session analytics and statistics"""
    try:
        # Get session-specific data if requested
        session_data = {}
        if session_id:
            session_data = zoe.get_conversation_history(session_id)

        return {
            "success": True,
            "data": {
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
    request: Dict[str, Any], zoe: ZoeService = Depends(get_zoe)
):
    """
    Legacy chat endpoint for backward compatibility

    This endpoint maintains compatibility with existing frontend code.
    All LLM processing goes through: ZoeService - Plugin - CortexFlow - WorkflowEngine
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

        # Process through NEW plugin architecture
        zoe_response = await zoe.process_message(
            message=message,
            user_id=user_id,
            session_id=session_id,
            user_context=user_context,
            application="chatbot"
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
    brain: CortexFlow = Depends(get_brain),
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
    request: Dict[str, Any], brain: CortexFlow = Depends(get_brain)
):
    """Healing rooms specific endpoint"""
    return await create_application_endpoint(request, "healing-rooms", brain)


@app.post("/api/inside-our-ai")
async def inside_our_ai_endpoint(
    request: Dict[str, Any], brain: CortexFlow = Depends(get_brain)
):
    """Inside our AI specific endpoint"""
    return await create_application_endpoint(request, "inside-our-ai", brain)


@app.post("/api/compliance")
async def compliance_endpoint(
    request: Dict[str, Any], brain: CortexFlow = Depends(get_brain)
):
    """Compliance specific endpoint"""
    return await create_application_endpoint(request, "compliance", brain)


@app.post("/api/exterior-spaces")
async def exterior_spaces_endpoint(
    request: Dict[str, Any], brain: CortexFlow = Depends(get_brain)
):
    """Exterior spaces specific endpoint"""
    return await create_application_endpoint(request, "exterior-spaces", brain)


# Health check endpoint
@app.get("/health")
async def health_check():
    """Basic health check"""
    return {
        "status": "healthy",
        "service": "ThinkLife Backend with Brain and Zoe",
        "timestamp": datetime.now().isoformat(),
        "brain_available": brain_instance is not None,
        "zoe_available": zoe_service_instance is not None,
    }


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "ThinkLife Backend with Brain and Zoe Integration",
        "version": "1.0.0",
        "brain_enabled": brain_instance is not None,
        "zoe_enabled": zoe_service_instance is not None,
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
