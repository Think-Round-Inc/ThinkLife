
"""
Zoe Service - Frontend Interface

This service provides the interface between the frontend and the plugin architecture.
All LLM processing goes through: Frontend - ZoeService - Plugin - CortexFlow

Usage:
    from agents.zoe import ZoeService

    service = ZoeService()
    await service.initialize()

    response = await service.process_message(
        message="I'm feeling anxious",
        user_id="user_123",
        session_id="session_1",
        user_context={"ace_score": 5}
    )
"""

import logging
import uuid
from typing import Dict, Any, Optional
from datetime import datetime
from brain.guardrails.policy_validator import PolicyValidator  
from brain.guardrails.fallback_handler import get_fallback_message 
from brain.guardrails.hallucination_tracker import HallucinationTracker 


logger = logging.getLogger(__name__)


class ZoeService:
    """
    Zoe Service - Clean interface for frontend

    This service:
    1. Receives requests from frontend/API
    2. Uses ZoeCore to prepare context and check safety
    3. Invokes the ZoeAgent plugin for LLM processing
    4. Returns formatted response to frontend

    All LLM calls go through the plugin - cortex architecture.
    """

    def __init__(self):
        self.zoe_core = None
        self.zoe_agent = None
        self.initialized = False
        # Initialize final-response policy validation.
        self.policy_validator = PolicyValidator() 
        # Initialize hallucination observation tracking for response monitoring.
        self.hallucination_tracker = HallucinationTracker()

    # Record response observations without interrupting message processing.
    def _record_hallucination_observation(
        self,
        reason="insufficient_evidence"
    ) -> None:
        """Record an unevaluated observation without exposing message content."""
        try:
            observation = self.hallucination_tracker.observe(
                status="unevaluated",
                agent_name="zoe",
                groundedness_score=None,
                reason=reason
            )

            if not observation["persisted"]:
                logger.warning(
                    "Hallucination observation was not persisted"
                )

        except Exception:
            logger.exception(
                "Hallucination observation tracking failed"
            )

    # Restrict agent metadata to approved fields to prevent internal data exposure.
    def _safe_public_metadata(self, agent_metadata):
        safe_metadata = {"agent": "zoe"}

        if not isinstance(agent_metadata, dict):
            return safe_metadata

        # Preserve numeric operational metrics only.
        for key in (
            "confidence_score",
            "duration_seconds",
            "estimated_cost",
            "estimated_latency",
            "validation_attempts",
        ):
            value = agent_metadata.get(key)

            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
            ):
                import math

                if math.isfinite(value):
                    safe_metadata[key] = value

        # Preserve moderation status only when recognized.
        moderation = agent_metadata.get("output_moderation")

        if moderation in ("passed", "blocked"):
            safe_metadata["output_moderation"] = moderation

        return safe_metadata

    async def initialize(self) -> bool:
        """Initialize ZoeService"""
        try:
            from .zoe_core import ZoeCore
            from plugins.zoe_agent import ZoeAgent
            from brain.specs import AgentConfig

            # Initialize ZoeCore (personality, conversation)
            self.zoe_core = ZoeCore()
            logger.info("ZoeCore initialized")

            # Initialize ZoeAgent plugin
            config = AgentConfig(agent_id="zoe_main")
            self.zoe_agent = ZoeAgent(config)
            await self.zoe_agent.initialize(config)
            logger.info("ZoeAgent plugin initialized")

            self.initialized = True
            return True

        except Exception as e:
            logger.error(f"Failed to initialize ZoeService: {e}")
            return False

    # Reusable moderation helper
    def _moderate_final_response(
        self,
        response_text: str
    ) -> tuple[str, bool]:
        """
        Apply existing security checks and final-output policy validation.
        """

        safe_replacement = (
            "I'm unable to provide that response safely. "
            "Please try rephrasing your message."
        )

        if not isinstance(response_text, str) or not response_text.strip():
            return safe_replacement, True

        try:
            security_manager = self.zoe_agent.cortex.security_manager

            # Sanitize the generated response before further security checks.
            sanitized_response = security_manager.sanitize_input(
                response_text
            )

            # Redact sensitive information before returning or storing the response.
            redaction_result = security_manager.redact_sensitive_data(
                sanitized_response
            )

            redacted_response = redaction_result["content"]

            # Apply content safety filters to the sanitized response.
            output_check = security_manager.filter_content(
                redacted_response
            )

            if not output_check["safe"]:
                return safe_replacement, True

            # Validate the final response against application safety policies.
            final_response = output_check["content"]

            policy_result = self.policy_validator.validate_response(
                final_response
            )

            if not policy_result["allowed"]:
                return get_fallback_message(policy_result), True 

            return final_response, False

        except Exception:
            logger.exception("Final response moderation failed")
            return safe_replacement, True

    async def process_message(
        self,
        message: str,
        user_id: str = "anonymous",
        session_id: Optional[str] = None,
        user_context: Optional[Dict[str, Any]] = None,
        application: str = "chatbot"
    ) -> Dict[str, Any]:
        """
        Process a user message through the plugin architecture.

        Args:
            message: User's message
            user_id: User identifier
            session_id: Session ID from brain's unified session management
            user_context: Optional user context (ace_score, etc.)
            application: Application type

        Returns:
            Dictionary with response and metadata
        """

        # NOTE:
        # Tracing is handled by cortex.py.
        # No decorator here to avoid duplicate traces.

        if not self.initialized:
            await self.initialize()

        try:
            # Ensure user_context is a dict
            if user_context is None:
                user_context = {}

            # Resolve session_id
            if not session_id:
                session_id = user_context.get("session_id")

            # Generate session_id if still missing
            if not session_id:
                session_id = str(uuid.uuid4())
                logger.info(
                    f"Generated new conversation session_id: {session_id}"
                )

            # Ensure session_id is in user_context
            user_context["session_id"] = session_id

            # 1. Create BrainRequest data using ZoeCore helper
            request_data = self.zoe_core.create_brain_request(
                message=message,
                user_context=user_context,
                application=application,
                session_id=session_id,
                user_id=user_id
            )

            # 2. Check if message needs redirection (safety)
            redirect_info = request_data.get("redirect_info")

            if redirect_info and redirect_info.get("should_redirect"):
                redirect_response, output_blocked = (
                    self._moderate_final_response(
                        redirect_info["response"]
                    )
                )

                # Sanitize and redact the user message before storing it in conversation history.
                security_manager = self.zoe_agent.cortex.security_manager
                sanitized_redirect_message = security_manager.sanitize_input(message)

                redirect_redaction_result = security_manager.redact_sensitive_data(
                    sanitized_redirect_message
                )
                safe_redirect_message = redirect_redaction_result["content"]
                self.zoe_core.conversation_manager.add_message(
                    session_id=request_data["session_id"],
                    role="user",
                    content=safe_redirect_message
                )

                # Add redirected assistant response
                self.zoe_core.conversation_manager.add_message(
                    session_id=request_data["session_id"],
                    role="assistant",
                    content=redirect_response,
                    metadata={
                        "redirected": True,
                        "harmful": redirect_info.get(
                            "is_harmful",
                            False
                        ),
                        "output_moderation": (
                            "blocked"
                            if output_blocked
                            else "passed"
                        )
                    }
                )

                # Record the safety redirect without evaluating response groundedness.
                self._record_hallucination_observation(
                    reason="not_applicable"
                )

                return {
                    "success": True,
                    "response": redirect_response,
                    "session_id": request_data["session_id"],
                    "redirected": True,
                    "metadata": {
                        "safety_redirect": True,
                        "agent": "zoe"
                    },
                    "timestamp": datetime.now().isoformat()
                }

            # 3. Create BrainRequest object for plugin
            from brain.specs import BrainRequest, ApplicationType
            from .helpers import create_user_context

            brain_request = BrainRequest(
                id=f"zoe_{datetime.now().timestamp()}",
                application=ApplicationType(application),
                message=message,
                user_context=create_user_context(
                    user_id=user_id,
                    session_id=request_data["session_id"],
                    user_context=user_context
                )
            )

            # Sanitize and redact user input before storing it in conversation history.
            security_manager = self.zoe_agent.cortex.security_manager
            sanitized_user_message = security_manager.sanitize_input(message)
            user_redaction_result = security_manager.redact_sensitive_data(
                sanitized_user_message
            )
            safe_user_message = user_redaction_result["content"]

            # 4. Add user message to conversation history
            self.zoe_core.conversation_manager.add_message(
                session_id=request_data["session_id"],
                role="user",
                content=safe_user_message,
                metadata={"application": application}
            )

            # 5. Process through plugin - Cortex
            logger.info(
                "Processing message through plugin for session: "
                f"{request_data['session_id']}"
            )

            agent_response = await self.zoe_agent.process_request(
                brain_request
            )

            # 6. Extract confidence score
            confidence = agent_response.metadata.get(
                "confidence_score",
                0.0
            )

            # Preserve security and content-filter failures as errors.
            # Do not convert them into successful low-confidence fallback responses.
            if not agent_response.success:
                fallback_response = (
                    self.zoe_core.get_fallback_response()
                )

                fallback_response, output_blocked = (
                    self._moderate_final_response(
                        fallback_response
                    )
                )

                self.zoe_core.conversation_manager.add_message(
                    session_id=request_data["session_id"],
                    role="assistant",
                    content=fallback_response,
                    metadata={
                        "fallback": True,
                        "error": "Processing failed",
                        "output_moderation": (
                            "blocked"
                            if output_blocked
                            else "passed"
                        ),
                    },
                )

                # Record the failed response without evaluating its groundedness.
                self._record_hallucination_observation(
                    reason="not_applicable"
                )

                return {
                    "success": False,
                    "response": fallback_response,
                    "session_id": request_data["session_id"],

                    # Return a safe error message without exposing internal processing details.
                    "error": "Processing failed",
                    "metadata": {
                        **self._safe_public_metadata(agent_response.metadata),
                        "agent": "zoe",
                        "fallback": True,
                        "output_moderation": (
                            "blocked" if output_blocked else "passed"
                        ),
                    },
                    "timestamp": datetime.now().isoformat(),
                }

            # -----------------------------------------------------
            # LOW-CONFIDENCE SUCCESS RESPONSE
            # -----------------------------------------------------
            if confidence < 0.75:
                logger.info(
                    f"Low confidence response ({confidence:.2f}), "
                    "using Zoe's fallback"
                )

                # Use Zoe's empathetic fallback
                zoe_fallback = self._get_low_confidence_response(
                    message,
                    confidence
                )

                # Final output moderation
                zoe_fallback, output_blocked = (
                    self._moderate_final_response(
                        zoe_fallback
                    )
                )

                # Validate and moderate the original response before including it in metadata.
                safe_original_response, original_response_blocked = (
                    self._moderate_final_response(agent_response.content)
                )

                # Add to conversation history
                self.zoe_core.conversation_manager.add_message(
                    session_id=agent_response.session_id,
                    role="assistant",
                    content=zoe_fallback,
                    metadata={
                        "low_confidence": True,
                        "original_confidence": confidence,

                        # Store only the validated and moderated original response.
                        "original_response": safe_original_response,
                        "output_moderation": (
                            "blocked"
                            if output_blocked
                            else "passed"
                        )
                    },
                )

                # Record the low-confidence response for hallucination monitoring.
                self._record_hallucination_observation(
                    reason="insufficient_evidence"
                )

                return {
                    "success": True,
                    "response": zoe_fallback,
                    "session_id": agent_response.session_id,
                    "confidence": confidence,
                    "metadata": {
                        **self._safe_public_metadata(agent_response.metadata),
                        "low_confidence": True,
                        "processing_time": (
                            agent_response.processing_time
                        ),
                        "agent": "zoe",
                        "output_moderation": (
                            "blocked" if output_blocked else "passed"
                        )
                    },
                    "timestamp": datetime.now().isoformat()
                }

            # -----------------------------------------------------
            # 7. HIGH-CONFIDENCE SUCCESS RESPONSE
            # -----------------------------------------------------
            final_response, output_blocked = (
                self._moderate_final_response(
                    agent_response.content
                )
            )

            # Add final response to conversation history
            self.zoe_core.conversation_manager.add_message(
                session_id=agent_response.session_id,
                role="assistant",
                content=final_response,
                metadata={
                    "confidence": confidence,
                    "output_moderation": (
                        "blocked"
                        if output_blocked
                        else "passed"
                    )
                }
            )

            # Record the successful response for hallucination monitoring.
            self._record_hallucination_observation(
                reason="insufficient_evidence"
            )

            return {
                "success": True,
                "response": final_response,
                "session_id": agent_response.session_id,
                "confidence": confidence,
                "metadata": {
                    **self._safe_public_metadata(agent_response.metadata),
                    "processing_time": (
                        agent_response.processing_time
                    ),
                    "agent": "zoe",
                    "output_moderation": (
                        "blocked"
                        if output_blocked
                        else "passed"
                    )
                },
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.exception(
                "Error in ZoeService.process_message"
            )

            error_response = (
                self.zoe_core.get_error_response()
                if self.zoe_core
                else "I'm experiencing technical difficulties."
            )

            # Final moderation for exception response
            if (
                self.zoe_agent
                and getattr(
                    self.zoe_agent,
                    "cortex",
                    None
                )
            ):
                error_response, output_blocked = (
                    self._moderate_final_response(
                        error_response
                    )
                )
            else:
                error_response = (
                    "I'm experiencing technical difficulties. "
                    "Please try again later."
                )
                output_blocked = True

            # Record the exception response for operational monitoring.
            self._record_hallucination_observation(
                reason="not_applicable"
            )

            return {
                "success": False,
                "response": error_response,

                # Return a generic error response without exposing internal exception details.
                "error": "An internal processing error occurred.",
                "metadata": {
                    "agent": "zoe",
                    "error": True,
                    "output_moderation": (
                        "blocked"
                        if output_blocked
                        else "passed"
                    )
                },
                "timestamp": datetime.now().isoformat()
            }

    def _get_low_confidence_response(
        self,
        message: str,
        confidence: float
    ) -> str:
        """
        Generate empathetic fallback response for
        low-confidence situations.

        Args:
            message: The user's original message
            confidence: The confidence score (0.0-1.0)

        Returns:
            Empathetic fallback response
        """

        message_lower = message.lower()

        # For emotional/support requests
        if any(
            word in message_lower
            for word in [
                "anxious",
                "scared",
                "worried",
                "stressed",
                "upset",
                "sad",
                "hurt",
                "pain"
            ]
        ):
            return (
                "I hear that you're going through something "
                "difficult right now. While I'm not entirely "
                "certain I can give you the best specific advice "
                "on this, I want you to know that your feelings "
                "are valid. Would you like to talk more about "
                "what you're experiencing, or would you prefer "
                "me to suggest some general coping strategies?"
            )

        # For questions requiring specific knowledge
        if any(
            word in message_lower
            for word in [
                "how",
                "what",
                "when",
                "where",
                "why",
                "explain"
            ]
        ):
            return (
                "I'm not completely confident I have the right "
                "information to answer that question accurately. "
                "Rather than give you potentially incorrect "
                "information, I think it would be better if you "
                "could rephrase your question or provide more "
                "context. I'm here to help, and I want to make "
                "sure I give you reliable guidance."
            )

        # For complex or unclear messages
        if len(message.split()) > 50:
            return (
                "I want to make sure I understand you correctly "
                "before responding. Your message covers quite a "
                "bit, and I'd like to give you a thoughtful "
                "answer. Could you help me by sharing what's "
                "most important to you right now, or what you'd "
                "most like support with?"
            )

        # Generic empathetic fallback
        return (
            "I'm here to support you, but I'm not entirely "
            "confident I fully understand what you need right "
            "now. Could you tell me a bit more about what's on "
            "your mind, or help me understand better how I can "
            "help? Your wellbeing matters, and I want to make "
            "sure I give you the support you deserve."
        )

        

    def get_conversation_history(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> list:
        """Get conversation history for a session"""

        if not self.zoe_core:
            return []

        return self.zoe_core.get_conversation_history(
            session_id,
            limit
        )

    def clear_conversation(
        self,
        session_id: str
    ) -> bool:
        """Clear conversation history"""

        if not self.zoe_core:
            return False

        return self.zoe_core.clear_conversation(
            session_id
        )

    async def health_check(self) -> Dict[str, Any]:
        """Health check for ZoeService"""

        health = {
            "service": "zoe",
            "initialized": self.initialized,
            "zoe_core_available": (
                self.zoe_core is not None
            ),
            "zoe_agent_available": (
                self.zoe_agent is not None
            )
        }

        if self.zoe_core:
            health["zoe_core_health"] = (
                self.zoe_core.health_check()
            )

        if self.zoe_agent:
            health["zoe_agent_health"] = (
                await self.zoe_agent.health_check()
            )

        return health

    async def shutdown(self):
        """Shutdown ZoeService"""

        if self.zoe_core:
            self.zoe_core.shutdown()

        if self.zoe_agent:
            await self.zoe_agent.shutdown()

        logger.info("ZoeService shutdown complete")


# Singleton instance for easy access
_zoe_service_instance: Optional[ZoeService] = None


def get_zoe_service() -> ZoeService:
    """Get singleton ZoeService instance"""

    global _zoe_service_instance

    if _zoe_service_instance is None:
        _zoe_service_instance = ZoeService()

    return _zoe_service_instance

