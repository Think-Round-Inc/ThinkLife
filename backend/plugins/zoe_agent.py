"""
Zoe's Plugin - Lightweight connector

All domain logic (personality, prompts, conversation) stays in agents/zoe/

The plugin is responsible for:
1. Creating the LLM request specifications (provider, model, params, tools, data sources)
2. Invoking the cortex to process the request
3. Post-processing response with ZoeCore
4. Returning the final response
"""

import logging
import sys
import os
from typing import Dict, Any, List, Optional

# Add backend directory to path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from brain.specs import (
    IAgent, AgentMetadata, AgentConfig, AgentResponse, BrainRequest,
    AgentCapability, AgentExecutionSpec, DataSourceSpec, ProviderSpec, 
    ToolSpec, ProcessingSpec, DataSourceType, ApplicationType
)
from brain.cortex import CortexFlow
from agents.zoe import ZoeCore

logger = logging.getLogger(__name__)


class ZoeAgent(IAgent):
    """
    Zoe Agent - Lightweight connector to CortexFlow
    
    Flow:
    1. Get conversation history and context from ZoeCore
    2. Build system prompt with ZoeCore
    3. Create AgentExecutionSpec (provider, tools, data sources, processing)
    4. Invoke CortexFlow to process the request
    5. Post-process LLM response with ZoeCore (safety, empathy)
    6. Update conversation history with ZoeCore
    7. Return final response
    """
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.initialized = False
        self.zoe_core = None
        self.cortex = None
    
    @property
    def metadata(self) -> AgentMetadata:
        """Return agent metadata"""
        return AgentMetadata(
            name="Zoe",
            version="1.0.0",
            description="Empathetic AI companion with trauma-informed care",
            capabilities=[
                AgentCapability.CONVERSATIONAL,
                AgentCapability.ANALYTICAL
            ],
            supported_applications=["chatbot", "general"],
            requires_auth=False,
            author="ThinkLife"
        )
    
    async def initialize(self, config: AgentConfig) -> bool:
        """Initialize the agent"""
        try:
            self.config = config
            
            # Initialize ZoeCore (personality, conversation management)
            self.zoe_core = ZoeCore()
            logger.info("ZoeCore initialized")
            
            # Get CortexFlow singleton
            self.cortex = CortexFlow()
            await self.cortex.initialize()
            logger.info("CortexFlow initialized")
            
            self.initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize ZoeAgent: {e}")
            return False
    
    async def create_execution_specs(self, request: BrainRequest) -> AgentExecutionSpec:
        """
        Create execution specs for the request.
        This is where Zoe defines how the Brain should process her request.
        """
        # 1. Prepare context using ZoeCore
        user_context = {
            "user_id": request.user_context.user_id,
            "ace_score": getattr(request.user_context.user_profile, 'ace_score', None) if request.user_context.user_profile else None
        }
        
        context = self.zoe_core.prepare_context(
            message=request.message,
            user_context=user_context,
            session_id=request.user_context.session_id
        )
        
        # 2. Get conversation history from ZoeCore
        conversation_history = self.zoe_core.get_conversation_history(request.user_context.session_id)
        
        # 3. Build system prompt with ZoeCore
        system_prompt = self.zoe_core.build_system_prompt(context)
        
        # 4. Build messages list (system prompt + history + current message)
        messages_context = {
            "system_prompt": system_prompt,
            "conversation_history": conversation_history[-10:],  # Last 10 messages
            "current_message": request.message
        }
        
        # 5. Create and return AgentExecutionSpec
        return AgentExecutionSpec(
            # Data sources: vector_db for Zoe's knowledge base
            data_sources=[
                DataSourceSpec(
                    source_type=DataSourceType.VECTOR_DB,
                    query=request.message,
                    limit=5,
                    enabled=True,
                    config={"collection_name": "zoe_knowledge"}
                ),
            ],
            
            # Provider: OpenAI with Zoe's preferred settings
            provider=ProviderSpec(
                provider_type="openai",
                model="gpt-4o-mini",
                temperature=0.7,
                max_tokens=1500,
                custom_params={"presence_penalty": 0.1, "frequency_penalty": 0.1}
            ),
            
            # Tools: Empty for now (can add tavily_search if needed)
            tools=[],
            
            # Processing config
            processing=ProcessingSpec(
                max_iterations=2,
                timeout_seconds=30.0,
                enable_safety_checks=True,
                enable_conversation_memory=True,
                execution_strategy="direct"  # No reasoning for Zoe (personality-driven)
            ),
            
            # Agent metadata (for workflow tracking)
            agent_metadata={
                "agent_type": "zoe",
                "agent_name": "Zoe AI Companion",
                "personality": "empathetic",
                "trauma_informed": True,
                "messages_context": messages_context  # Pass context to workflow
            }
        )
    
    async def process_request(self, request: BrainRequest) -> AgentResponse:
        """
        Process a user request through CortexFlow
        
        This is the main entry point for processing requests
        """
        if not self.initialized:
            await self.initialize(self.config)
        
        session_id = request.user_context.session_id
        start_time = 0.0
        
        try:
            import time
            start_time = time.time()
            
            # 1. Prepare context using ZoeCore
            user_context = {
                "user_id": request.user_context.user_id,
                "ace_score": getattr(request.user_context, 'ace_score', None)
            }
            
            context = self.zoe_core.prepare_context(
                message=request.message,
                user_context=user_context,
                session_id=session_id
            )
            
            # 2. Get conversation history from ZoeCore
            conversation_history = self.zoe_core.get_conversation_history(session_id)
            
            # 3. Build system prompt with ZoeCore
            system_prompt = self.zoe_core.build_system_prompt(context)
            
            # 4. Create AgentExecutionSpec using the abstract method
            agent_specs = await self.create_execution_specs(request)
            
            # 5. Invoke CortexFlow to process the request
            logger.info(f"Invoking CortexFlow for session: {session_id}")
            cortex_response = await self.cortex.process_agent_request(
                agent_specs=agent_specs,
                request=request
            )
            
            # 7. Extract LLM response
            if cortex_response.get("success"):
                llm_response = cortex_response.get("content", "")
                
                # 8. Post-process with ZoeCore (safety checks, empathy enhancements)
                final_response = self.zoe_core.post_process_response(
                    llm_response=llm_response,
                    context=context
                )
                
                # 9. Update conversation history with ZoeCore
                self.zoe_core.update_conversation(
                    session_id=session_id,
                    user_message=request.message,
                    assistant_response=final_response
                )
                
                # 10. Return successful response
                processing_time = time.time() - start_time
                
                return AgentResponse(
                    success=True,
                    content=final_response,
                    metadata={
                        **cortex_response.get("metadata", {}),
                        "agent": "zoe",
                        "session_id": session_id,
                        "personality_processed": True,
                        "trauma_informed": True
                    },
                    confidence=0.9,
                    processing_time=processing_time,
                    session_id=session_id
                )
            
            else:
                # Cortex processing failed
                error_msg = cortex_response.get("content", "Processing failed")
                fallback_response = self.zoe_core.get_fallback_response()
                
                processing_time = time.time() - start_time
                
                return AgentResponse(
                    success=False,
                    content=fallback_response,
                    metadata={
                        "error": error_msg,
                        "agent": "zoe",
                        "session_id": session_id
                    },
                    processing_time=processing_time,
                    session_id=session_id
                )
        
        except Exception as e:
            logger.error(f"Error in ZoeAgent.process_request: {e}")
            
            # Get empathetic error response from ZoeCore
            error_response = self.zoe_core.get_error_response() if self.zoe_core else "I'm having technical difficulties. Please try again."
            
            import time
            processing_time = time.time() - start_time if start_time else 0.0
            
            return AgentResponse(
                success=False,
                content=error_response,
                metadata={
                    "error": str(e),
                    "agent": "zoe",
                    "session_id": session_id
                },
                processing_time=processing_time,
                session_id=session_id
            )
    
    async def health_check(self) -> Dict[str, Any]:
        """Check agent health"""
        health = {
            "agent": "zoe",
            "status": "healthy" if self.initialized else "not_initialized",
            "initialized": self.initialized,
            "cortex_available": self.cortex is not None,
            "zoe_core_available": self.zoe_core is not None
        }
        
        if self.zoe_core:
            zoe_health = self.zoe_core.health_check()
            health["zoe_core"] = zoe_health
        
        return health
    
    async def shutdown(self) -> None:
        """Clean shutdown of agent resources"""
        if self.zoe_core:
            self.zoe_core.shutdown()
        
        if self.cortex:
            await self.cortex.shutdown()
        
        logger.info("ZoeAgent shutdown complete")


# Plugin factory function
def create_zoe_agent(config: AgentConfig) -> ZoeAgent:
    """Factory function to create Zoe agent"""
    return ZoeAgent(config)
