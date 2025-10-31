"""
Example Agent Plugin - Demonstrates how to create a Brain-compatible agent
This serves as a template for developers to create their own agents
"""

import asyncio
import logging
import time
from typing import Dict, Any, List
from datetime import datetime

from brain.interfaces import (
    IAgent, IAgentPlugin, IConversationalAgent, ITraumaInformedAgent,
    AgentMetadata, AgentConfig, AgentResponse, BrainRequest, 
    AgentCapability, UserContext
)
from brain.types import (
    AgentExecutionSpec, DataSourceSpec, ProviderSpec, ToolSpec, ProcessingSpec,
    DataSourceType
)

logger = logging.getLogger(__name__)


class ExampleAgent(IAgent, IConversationalAgent, ITraumaInformedAgent):
    """
    Example agent implementation showing all the key interfaces
    This agent provides basic conversational capabilities with trauma-informed features
    """
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.agent_id = config.agent_id
        self._initialized = False
        
        # Conversation storage (in production, use proper database)
        self.conversations: Dict[str, List[Dict[str, Any]]] = {}
        self.contexts: Dict[str, Dict[str, Any]] = {}
        
        # Configuration
        self.max_history = config.config.get("max_history", 20)
        self.response_style = config.config.get("response_style", "empathetic")
        
    @property
    def metadata(self) -> AgentMetadata:
        """Return agent metadata"""
        return AgentMetadata(
            name="Example Agent",
            version="1.0.0",
            description="Example agent demonstrating Brain plugin capabilities",
            capabilities=[
                AgentCapability.CONVERSATIONAL,
                AgentCapability.THERAPEUTIC,
                AgentCapability.EDUCATIONAL
            ],
            supported_applications=[
                "chatbot",
                "healing-rooms",
                "general"
            ],
            trauma_informed=True,
            requires_auth=False,
            max_concurrent_sessions=50,
            average_response_time=1.5,
            author="ThinkxLife Team"
        )
    
    async def initialize(self, config: AgentConfig) -> bool:
        """Initialize the agent"""
        try:
            logger.info(f"Initializing Example Agent {self.agent_id}")
            
            # Perform any initialization tasks here
            # e.g., load models, connect to databases, etc.
            
            self._initialized = True
            logger.info(f"Example Agent {self.agent_id} initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Example Agent {self.agent_id}: {str(e)}")
            return False
    
    async def create_execution_specs(self, request: BrainRequest) -> AgentExecutionSpec:
        """
        Specify how Brain should execute this request
        
        Agent decides:
        - Which data sources to query
        - Which LLM provider and configuration
        - Which tools to apply
        - Processing requirements
        """
        # Determine data sources based on application type
        data_sources = []
        
        # Always include conversation history for conversational agents
        data_sources.append(
            DataSourceSpec(
                source_type=DataSourceType.CONVERSATION_HISTORY,
                enabled=True
            )
        )
        
        # For healing rooms, query vector DB with healing-specific content
        if request.application == "healing-rooms":
            data_sources.append(
                DataSourceSpec(
                    source_type=DataSourceType.VECTOR_DB,
                    query=request.message,
                    filters={"application": "healing-rooms", "category": "therapeutic"},
                    limit=5,
                    enabled=True
                )
            )
        
        # Specify provider configuration
        provider = ProviderSpec(
            provider_type="openai",
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=1500,
            custom_params={
                "top_p": 0.9,
                "presence_penalty": 0.1
            }
        )
        
        # Specify processing configuration
        processing = ProcessingSpec(
            max_iterations=2,
            timeout_seconds=25.0,
            enable_context_enhancement=True,
            enable_safety_checks=True,
            enable_conversation_memory=True,
            stream_response=False
        )
        
        return AgentExecutionSpec(
            data_sources=data_sources,
            provider=provider,
            tools=[],  # No tools for this example
            processing=processing,
            agent_metadata={
                "agent_type": "example",
                "response_style": self.response_style
            }
        )
    
    async def process_request(self, request: BrainRequest) -> AgentResponse:
        """Process a user request"""
        start_time = time.time()
        
        try:
            if not self._initialized:
                return AgentResponse(
                    success=False,
                    content="Agent not initialized",
                    processing_time=time.time() - start_time
                )
            
            # Extract request information
            message = request.message
            user_context = request.user_context
            session_id = request.context.session_id if request.context else "default"
            
            # Perform safety assessment for trauma-informed care
            safety_assessment = await self.assess_safety(request)
            if not safety_assessment.get("safe", True):
                return await self._handle_unsafe_request(request, safety_assessment)
            
            # Load conversation context
            conversation_history = await self.get_conversation_history(session_id)
            
            # Generate response based on application type
            if request.application == "healing-rooms":
                response_content = await self._generate_healing_response(
                    message, user_context, conversation_history
                )
            elif request.application == "chatbot":
                response_content = await self._generate_conversational_response(
                    message, user_context, conversation_history
                )
            else:
                response_content = await self._generate_general_response(
                    message, user_context, conversation_history
                )
            
            # Create response
            response = AgentResponse(
                success=True,
                content=response_content,
                metadata={
                    "agent": self.metadata.name,
                    "application": request.application,
                    "session_id": session_id,
                    "safety_checked": True
                },
                processing_time=time.time() - start_time,
                session_id=session_id
            )
            
            # Apply safety filters
            filtered_response = await self.apply_safety_filters(response)
            
            # Update conversation history
            await self._update_conversation_history(session_id, message, filtered_response.content)
            
            return filtered_response
            
        except Exception as e:
            logger.error(f"Error processing request in Example Agent: {str(e)}")
            return AgentResponse(
                success=False,
                content="I apologize, but I encountered an error processing your request.",
                metadata={"error": str(e)},
                processing_time=time.time() - start_time
            )
    
    async def _generate_healing_response(
        self, 
        message: str, 
        user_context: UserContext, 
        history: List[Dict[str, Any]]
    ) -> str:
        """Generate trauma-informed healing response"""
        
        ace_score = user_context.ace_score or 0
        
        # Trauma-informed response templates
        if ace_score >= 4:
            # High ACE score - extra gentle approach
            responses = [
                f"Thank you for sharing that with me. I can hear the strength in your words, even when things feel difficult.",
                f"What you've experienced matters, and so do you. I'm here to listen without judgment.",
                f"It takes courage to reach out. You're taking an important step by being here."
            ]
        else:
            # Standard empathetic responses
            responses = [
                f"I appreciate you sharing that with me. How are you feeling about it?",
                f"That sounds important to you. Would you like to explore that further?",
                f"I'm here to support you. What would be most helpful right now?"
            ]
        
        # Simple response selection (in production, use more sophisticated NLP)
        import random
        base_response = random.choice(responses)
        
        # Add context-aware elements
        if "sad" in message.lower() or "depressed" in message.lower():
            base_response += " Remember that it's okay to feel sad sometimes, and these feelings can change."
        elif "anxious" in message.lower() or "worried" in message.lower():
            base_response += " Anxiety can feel overwhelming, but you're not alone in this."
        
        return base_response
    
    async def _generate_conversational_response(
        self, 
        message: str, 
        user_context: UserContext, 
        history: List[Dict[str, Any]]
    ) -> str:
        """Generate general conversational response"""
        
        # Simple conversational responses (in production, integrate with LLM)
        if "hello" in message.lower() or "hi" in message.lower():
            return "Hello! I'm here to help and support you. What would you like to talk about today?"
        elif "how are you" in message.lower():
            return "Thank you for asking! I'm here and ready to help. How are you doing today?"
        elif "help" in message.lower():
            return "I'm here to support you. You can talk to me about anything that's on your mind, and I'll do my best to listen and provide helpful responses."
        else:
            return f"I hear what you're saying about '{message}'. Can you tell me more about how you're feeling about this?"
    
    async def _generate_general_response(
        self, 
        message: str, 
        user_context: UserContext, 
        history: List[Dict[str, Any]]
    ) -> str:
        """Generate general purpose response"""
        return f"Thank you for your message. I'm here to help with whatever you need. Could you tell me more about what you're looking for?"
    
    async def _handle_unsafe_request(self, request: BrainRequest, safety_assessment: Dict[str, Any]) -> AgentResponse:
        """Handle requests that fail safety assessment"""
        
        crisis_detected = safety_assessment.get("crisis_detected", False)
        
        if crisis_detected:
            crisis_resources = self.get_crisis_resources(request.user_context)
            
            return AgentResponse(
                success=True,
                content="I'm concerned about your safety. Please reach out to a mental health professional or crisis hotline immediately. You don't have to go through this alone.",
                metadata={
                    "crisis_intervention": True,
                    "resources": crisis_resources
                },
                safety_flags=["crisis_detected"]
            )
        else:
            return AgentResponse(
                success=False,
                content="I'm not able to respond to that request. Let's talk about something else that might be helpful.",
                metadata={"safety_block": True},
                safety_flags=["content_blocked"]
            )
    
    async def _update_conversation_history(self, session_id: str, user_message: str, agent_response: str) -> None:
        """Update conversation history"""
        if session_id not in self.conversations:
            self.conversations[session_id] = []
        
        # Add user message
        self.conversations[session_id].append({
            "role": "user",
            "content": user_message,
            "timestamp": datetime.now().isoformat()
        })
        
        # Add agent response
        self.conversations[session_id].append({
            "role": "assistant",
            "content": agent_response,
            "timestamp": datetime.now().isoformat()
        })
        
        # Limit history length
        if len(self.conversations[session_id]) > self.max_history * 2:  # *2 for user+assistant pairs
            self.conversations[session_id] = self.conversations[session_id][-self.max_history * 2:]
    
    async def health_check(self) -> Dict[str, Any]:
        """Check agent health"""
        return {
            "status": "healthy" if self._initialized else "not_initialized",
            "agent_id": self.agent_id,
            "active_sessions": len(self.conversations),
            "total_conversations": sum(len(conv) for conv in self.conversations.values()),
            "initialized": self._initialized,
            "timestamp": datetime.now().isoformat()
        }
    
    async def shutdown(self) -> None:
        """Shutdown the agent"""
        logger.info(f"Shutting down Example Agent {self.agent_id}")
        
        # Clean up resources
        self.conversations.clear()
        self.contexts.clear()
        self._initialized = False
        
        logger.info(f"Example Agent {self.agent_id} shutdown complete")
    
    async def can_handle_request(self, request: BrainRequest) -> float:
        """Return confidence score for handling this request"""
        
        # Check if application is supported
        if request.application not in self.metadata.supported_applications:
            return 0.0
        
        # Higher confidence for applications we specialize in
        if request.application == "healing-rooms":
            return 0.9
        elif request.application == "chatbot":
            return 0.8
        else:
            return 0.6
    
    # IConversationalAgent implementation
    
    async def get_conversation_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for a session"""
        return self.conversations.get(session_id, [])
    
    async def clear_conversation(self, session_id: str) -> bool:
        """Clear conversation history for a session"""
        try:
            if session_id in self.conversations:
                del self.conversations[session_id]
            if session_id in self.contexts:
                del self.contexts[session_id]
            return True
        except Exception as e:
            logger.error(f"Error clearing conversation {session_id}: {str(e)}")
            return False
    
    async def update_context(self, session_id: str, context: Dict[str, Any]) -> bool:
        """Update conversation context"""
        try:
            if session_id not in self.contexts:
                self.contexts[session_id] = {}
            
            self.contexts[session_id].update(context)
            return True
        except Exception as e:
            logger.error(f"Error updating context for {session_id}: {str(e)}")
            return False
    
    # ITraumaInformedAgent implementation
    
    async def assess_safety(self, request: BrainRequest) -> Dict[str, Any]:
        """Assess request for trauma triggers and safety concerns"""
        
        message = request.message.lower()
        user_context = request.user_context
        
        # Crisis indicators
        crisis_words = ["suicide", "kill myself", "end it all", "not worth living"]
        crisis_detected = any(word in message for word in crisis_words)
        
        # Trauma triggers
        trigger_words = ["abuse", "violence", "trauma", "hurt me"]
        triggers_present = [word for word in trigger_words if word in message]
        
        # Check ACE score for additional context
        ace_score = user_context.ace_score or 0
        high_risk = ace_score >= 4
        
        return {
            "safe": not crisis_detected,
            "crisis_detected": crisis_detected,
            "triggers_present": triggers_present,
            "high_risk_user": high_risk,
            "recommendations": self._get_safety_recommendations(crisis_detected, triggers_present, high_risk)
        }
    
    def _get_safety_recommendations(self, crisis_detected: bool, triggers_present: List[str], high_risk: bool) -> List[str]:
        """Get safety recommendations based on assessment"""
        recommendations = []
        
        if crisis_detected:
            recommendations.append("immediate_intervention")
            recommendations.append("crisis_resources")
        
        if triggers_present:
            recommendations.append("gentle_response")
            recommendations.append("validation_focus")
        
        if high_risk:
            recommendations.append("extra_empathy")
            recommendations.append("professional_support_suggestion")
        
        return recommendations
    
    async def apply_safety_filters(self, response: AgentResponse) -> AgentResponse:
        """Apply trauma-informed safety filters to response"""
        
        content = response.content
        
        # Remove potentially triggering language
        trigger_replacements = {
            "you should": "you might consider",
            "you must": "it could be helpful to",
            "you need to": "one option might be to"
        }
        
        for trigger, replacement in trigger_replacements.items():
            content = content.replace(trigger, replacement)
        
        # Add safety disclaimers for certain topics
        safety_topics = ["medication", "therapy", "treatment"]
        if any(topic in content.lower() for topic in safety_topics):
            content += "\n\nPlease remember that I'm not a substitute for professional medical or therapeutic advice."
        
        # Update response
        response.content = content
        response.safety_flags = response.safety_flags or []
        response.safety_flags.append("safety_filtered")
        
        return response
    
    def get_crisis_resources(self, user_context: UserContext) -> List[Dict[str, Any]]:
        """Get appropriate crisis resources for user"""
        
        # Default crisis resources
        resources = [
            {
                "name": "National Suicide Prevention Lifeline",
                "phone": "988",
                "description": "24/7 crisis support",
                "type": "crisis_hotline"
            },
            {
                "name": "Crisis Text Line",
                "contact": "Text HOME to 741741",
                "description": "24/7 text-based crisis support",
                "type": "text_support"
            },
            {
                "name": "SAMHSA National Helpline",
                "phone": "1-800-662-4357",
                "description": "Treatment referral and information service",
                "type": "treatment_referral"
            }
        ]
        
        # Add location-specific resources if available
        # This would be enhanced with actual location-based resource lookup
        
        return resources


class ExampleAgentPlugin(IAgentPlugin):
    """Plugin class for the Example Agent"""
    
    def create_agent(self, config: AgentConfig) -> IAgent:
        """Factory method to create agent instance"""
        return ExampleAgent(config)
    
    def get_metadata(self) -> AgentMetadata:
        """Get plugin metadata"""
        return AgentMetadata(
            name="Example Agent",
            version="1.0.0",
            description="Example agent demonstrating Brain plugin capabilities",
            capabilities=[
                AgentCapability.CONVERSATIONAL,
                AgentCapability.THERAPEUTIC,
                AgentCapability.EDUCATIONAL
            ],
            supported_applications=[
                "chatbot",
                "healing-rooms",
                "general"
            ],
            trauma_informed=True,
            requires_auth=False,
            max_concurrent_sessions=50,
            average_response_time=1.5,
            author="ThinkxLife Team"
        )
    
    def validate_config(self, config: AgentConfig) -> bool:
        """Validate configuration before agent creation"""
        
        # Check required configuration
        if not config.agent_id:
            logger.error("Agent ID is required")
            return False
        
        # Validate optional configuration
        max_history = config.config.get("max_history", 20)
        if not isinstance(max_history, int) or max_history < 1:
            logger.error("max_history must be a positive integer")
            return False
        
        response_style = config.config.get("response_style", "empathetic")
        valid_styles = ["empathetic", "professional", "casual"]
        if response_style not in valid_styles:
            logger.error(f"response_style must be one of: {valid_styles}")
            return False
        
        return True
