"""
EchoAgent - a minimal, dependency-free reference implementation of `IAgent`.

Purpose: prove that the `IAgent` plugin contract (`brain/specs/agent_specs.py`)
is genuinely implementable by more than one agent, not just `ZoeAgent`. This
is the practical validation called for by Epic 5 / Story 5.2
"""

import logging
import time
from typing import Any, Dict

from config import settings
from brain.specs import (
    IAgent, AgentMetadata, AgentConfig, AgentResponse, BrainRequest,
    AgentCapability, AgentExecutionSpec, ProviderSpec, ProcessingSpec,
)

logger = logging.getLogger(__name__)


class EchoAgent(IAgent):
    """Reference/test-double agent - echoes the user's message back."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.initialized = False

    @property
    def metadata(self) -> AgentMetadata:
        return AgentMetadata(
            name="Echo",
            version="1.0.0",
            description="Minimal reference agent used to validate the IAgent plugin contract",
            capabilities=[AgentCapability.CONVERSATIONAL],
            supported_applications=["general"],
            requires_auth=False,
            author="ThinkLife",
        )

    async def initialize(self, config: AgentConfig) -> bool:
        self.config = config
        self.initialized = True
        logger.info("EchoAgent initialized")
        return True

    async def create_execution_specs(self, request: BrainRequest) -> AgentExecutionSpec:
        # A minimal but fully-valid spec, demonstrating the contract even
        # though EchoAgent's process_request() below doesn't route through
        # Cortex/WorkflowEngine.
        return AgentExecutionSpec(
            data_sources=[],
            provider=ProviderSpec(provider_type=settings.default_llm()[0], model=settings.default_llm()[1]),
            tools=[],
            processing=ProcessingSpec(execution_strategy="direct", eval=False),
            agent_metadata={"agent_type": "echo"},
        )

    async def process_request(self, request: BrainRequest) -> AgentResponse:
        if not self.initialized:
            await self.initialize(self.config)

        start_time = time.time()
        session_id = request.user_context.session_id

        return AgentResponse(
            success=True,
            content=f"Echo: {request.message}",
            metadata={"agent": "echo"},
            confidence=1.0,
            processing_time=time.time() - start_time,
            session_id=session_id,
        )

    async def health_check(self) -> Dict[str, Any]:
        return {
            "agent": "echo",
            "status": "healthy" if self.initialized else "not_initialized",
            "initialized": self.initialized,
        }

    async def shutdown(self) -> None:
        self.initialized = False
        logger.info("EchoAgent shutdown")


def create_echo_agent(config: AgentConfig) -> EchoAgent:
    return EchoAgent(config)
