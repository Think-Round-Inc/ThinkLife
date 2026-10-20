"""
Agent Router - message passing between agents (US 1.1)

The communication rule (docs/workflows/ADR-001-agent-communication.md):

  * Inside one workflow run, nodes share the typed WorkflowState.
  * Between agents there is no shared state and no direct calls. An agent
    that needs another one sends an AgentMessage; this router delivers it
    by running the recipient's own workflow (agent.process_request).

The recipient receives only the envelope and payload["message"] - never the
sender's prompt, conversation history or user profile (e.g. Zoe's ACE-score
context stays in Zoe's run). `correlation_id` links every hop in logs and
traces, and `hop` stops agents from bouncing a request back and forth.
"""

import logging
from dataclasses import replace
from typing import Any, Dict, List, Optional

from ..specs import (
    AgentMessage, AgentResponse, ApplicationType, BrainRequest, MessageIntent, RequestContext,
)

logger = logging.getLogger(__name__)

MAX_HOPS = 3


class AgentRouter:
    """Registry of Brain agents plus delivery of AgentMessages between them."""

    def __init__(self, max_hops: int = MAX_HOPS):
        self._agents: Dict[str, Any] = {}
        self.max_hops = max_hops

    def register(self, name: str, agent: Any) -> None:
        """Register an IAgent under a lowercase name ("zoe", "faq", ...)."""
        key = name.lower()
        if key in self._agents:
            logger.warning("Replacing registered agent %r", key)
        self._agents[key] = agent

    def unregister(self, name: str) -> None:
        self._agents.pop(name.lower(), None)

    def registered(self) -> List[str]:
        return sorted(self._agents)

    async def dispatch(self, message: AgentMessage, user_context: Any,
                       application: ApplicationType = ApplicationType.GENERAL) -> AgentResponse:
        """Deliver one message; follows a returned hand-off chain up to max_hops."""
        recipient = message.recipient.lower()
        log_ctx = f"correlation_id={message.correlation_id} {message.sender}->{recipient} intent={message.intent.value} hop={message.hop}"

        if message.hop >= self.max_hops:
            logger.warning("Agent message refused, hop limit reached: %s", log_ctx)
            return self._refusal(message, f"hop limit {self.max_hops} reached")

        agent = self._agents.get(recipient)
        if agent is None:
            logger.warning("Agent message to unknown recipient: %s", log_ctx)
            return self._refusal(message, f"unknown recipient '{recipient}'")

        if message.intent == MessageIntent.NOTIFY:
            handler = getattr(agent, "handle_notification", None)
            if handler is not None:
                await handler(message)
            logger.info("Agent notification delivered: %s", log_ctx)
            return AgentResponse(success=True, content="", metadata=self._trace(message, recipient))

        text = (message.payload or {}).get("message")
        if not text:
            return self._refusal(message, "payload['message'] is required for handoff and query")

        request = BrainRequest(
            id=message.message_id,
            application=application,
            message=text,
            user_context=user_context,
            # Only the envelope travels with the request, not the sender's state.
            context=RequestContext(
                session_id=getattr(user_context, "session_id", None),
                brain_context={"agent_message": {k: v for k, v in message.to_dict().items() if k != "payload"}},
            ),
        )
        logger.info("Agent message delivered: %s", log_ctx)
        response = await agent.process_request(request)
        response.metadata = {**(response.metadata or {}), **self._trace(message, recipient)}

        # The recipient may hand off again by returning an AgentMessage in its metadata.
        onward = (response.metadata or {}).get("agent_message")
        if isinstance(onward, AgentMessage):
            onward = replace(onward, correlation_id=message.correlation_id, hop=message.hop + 1)
            return await self.dispatch(onward, user_context, application)
        return response

    def _trace(self, message: AgentMessage, handled_by: str) -> Dict[str, Any]:
        return {
            "correlation_id": message.correlation_id,
            "message_id": message.message_id,
            "handled_by": handled_by,
            "sent_by": message.sender,
            "hop": message.hop + 1,
        }

    def _refusal(self, message: AgentMessage, reason: str) -> AgentResponse:
        return AgentResponse(
            success=False,
            content="",
            metadata={**self._trace(message, message.recipient), "error": reason},
        )


_router: Optional[AgentRouter] = None


def get_agent_router() -> AgentRouter:
    global _router
    if _router is None:
        _router = AgentRouter()
    return _router
