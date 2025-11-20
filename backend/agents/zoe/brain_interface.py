"""
Zoe Brain Interface - Handles integration between Brain plugin system and Zoe services
This service provides Zoe's dedicated interface to the Brain system
"""

import logging
import time
from typing import Dict, Any, List, Optional
from datetime import datetime

# Import Brain's shared knowledge system
try:
    from brain.data_sources import get_data_source_registry
except ImportError:
    # Fallback if brain not available
    get_data_source_registry = None

logger = logging.getLogger(__name__)


class ZoeBrainInterface:
    """
    Zoe's dedicated interface to the Brain plugin system
    Handles all integration logic while keeping the plugin lightweight
    """
    
    def __init__(self, zoe_core=None):
        self.zoe_core = zoe_core
        self._initialized = False
        self.personality_style = "empathetic_companion"
        
        # Access Brain's shared knowledge system
        self.data_registry = get_data_source_registry() if get_data_source_registry else None
        
    async def initialize(self) -> bool:
        """Initialize Zoe's Brain interface service"""
        try:
            if not self.zoe_core:
                logger.error("ZoeCore not provided to interface")
                return False
                
            self._initialized = True
            logger.info("Zoe Brain Interface initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Zoe Brain Interface: {str(e)}")
            return False
    
    async def process_brain_request(self, brain_request) -> Dict[str, Any]:
        """
        Process a Brain request using Zoe's trauma-informed workflow
        This implements Zoe's own domain-specific processing logic
        """
        start_time = time.time()
        
        try:
            if not self._initialized or not self.zoe_core:
                return {
                    "success": False,
                    "content": "Zoe is not available right now. Please try again later.",
                    "error": "Service not initialized",
                    "processing_time": time.time() - start_time
                }
            
            # Extract request information
            message = brain_request.message
            user_context = brain_request.user_context
            session_id = brain_request.context.session_id if brain_request.context else "default"
            application = brain_request.application
            
            # Convert Brain request to Zoe format
            zoe_user_context = self._convert_user_context(user_context, application)
            
            # Use Zoe's trauma-informed workflow
            if application in ["healing-rooms"] or self._is_high_trauma_risk(zoe_user_context):
                return await self._process_trauma_informed_request(
                    message, zoe_user_context, application, session_id, user_context.user_id, start_time
                )
            else:
                return await self._process_standard_request(
                    message, zoe_user_context, application, session_id, user_context.user_id, start_time
                )
            
        except Exception as e:
            logger.error(f"Error processing Brain request in Zoe interface: {str(e)}")
            return {
                "success": False,
                "content": "I'm experiencing some technical difficulties. I'm still here for you though.",
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    def _convert_user_context(self, user_context, application) -> Dict[str, Any]:
        """Convert Brain UserContext to Zoe format"""
        ace_score = 0
        if user_context.trauma_context:
            ace_score = user_context.trauma_context.ace_score
        
        return {
            "ace_score": ace_score,
            "age": getattr(user_context, 'age', 25),
            "name": getattr(user_context, 'name', "User"),
            "application": str(application),
            "user_id": user_context.user_id
        }
    
    async def _process_message_direct(self, message: str, user_context: dict, application: str, session_id: str, user_id: str, knowledge_context: List[Dict[str, Any]] = None) -> dict:
        """Process message directly without calling Brain (to avoid circular dependency)"""
        try:
            # Add user message to conversation
            self.zoe_core.conversation_manager.add_message(
                session_id=session_id,
                role="user", 
                content=message,
                metadata={"application": application}
            )
            
            # Get conversation context
            session_id, session = self.zoe_core.conversation_manager.get_or_create_session(
                session_id=session_id,
                user_id=user_id
            )
            
            # Generate response using Zoe's personality system + shared knowledge
            personality_context = {
                "user_context": user_context,
                "conversation_length": len(session.messages),
                "application": application,
                "ace_score": user_context.get("ace_score", 0),
                "shared_knowledge": knowledge_context or []
            }
            
            # Generate base response enhanced with shared knowledge
            base_response = self._generate_empathetic_response(message, personality_context)
            response_content = self.zoe_core.personality.post_process_response(
                base_response, personality_context
            )
            
            # Add assistant response to conversation
            self.zoe_core.conversation_manager.add_message(
                session_id=session_id,
                role="assistant",
                content=response_content,
                metadata={"agent": "zoe", "application": application}
            )
            
            return {
                "success": True,
                "response": response_content,
                "session_id": session_id,
                "metadata": {
                    "agent": "zoe_interface",
                    "application": application,
                    "conversation_length": len(session.messages)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in direct Zoe processing: {str(e)}")
            return {
                "success": False,
                "response": "I'm experiencing some technical difficulties, but I'm still here for you.",
                "error": str(e),
                "session_id": session_id
            }
    
    def _generate_empathetic_response(self, message: str, context: dict) -> str:
        """Generate a basic empathetic response based on message content and context"""
        
        # Extract context information
        ace_score = context.get("ace_score", 0)
        application = context.get("application", "general")
        conversation_length = context.get("conversation_length", 0)
        shared_knowledge = context.get("shared_knowledge", [])
        
        # Extract relevant information from shared knowledge
        knowledge_summary = self._summarize_shared_knowledge(shared_knowledge, message)
        
        # Determine response tone based on context
        if ace_score > 4:
            # High trauma score - extra gentle
            if any(word in message.lower() for word in ["anxious", "worried", "scared", "afraid"]):
                return "I can hear that you're feeling anxious right now, and that takes courage to share. You're not alone in this feeling. What you're experiencing is valid, and I'm here to support you through it. Would it help to talk about what's making you feel this way?"
            elif any(word in message.lower() for word in ["sad", "depressed", "down", "upset"]):
                return "I can sense the sadness in your words, and I want you to know that it's okay to feel this way. Your feelings matter, and you matter. Sometimes just acknowledging these feelings can be the first step. I'm here with you, and we can take this one moment at a time."
            else:
                return "Thank you for sharing with me. I can sense that you're going through something difficult right now. Please know that you're safe here, and I'm honored that you trust me with your thoughts. How are you feeling in this moment?"
        
        elif application == "healing-rooms":
            # Healing rooms - trauma-informed but not as intensive
            if any(word in message.lower() for word in ["hello", "hi", "hey"]):
                return "Hello, and welcome to this safe space. I'm Zoe, and I'm here to support you with warmth and understanding. How are you feeling today?"
            elif any(word in message.lower() for word in ["help", "support", "need"]):
                base_response = "I'm here to help and support you in whatever way I can. This is a safe space where you can share what's on your mind."
                if knowledge_summary:
                    base_response += f" I also have some resources that might be helpful: {knowledge_summary}"
                return base_response + " What would feel most helpful for you right now?"
            else:
                return "I hear you, and I want you to know that your feelings and experiences are valid. You've taken a brave step by reaching out. How can I best support you today?"
        
        else:
            # General conversation - warm but less clinical
            if any(word in message.lower() for word in ["hello", "hi", "hey"]):
                return "Hello! I'm Zoe, and I'm so glad you're here. How are you doing today?"
            elif any(word in message.lower() for word in ["how are you", "how's it going"]):
                return "Thank you for asking! I'm doing well and I'm here, ready to listen and support you. How are you feeling today?"
            else:
                return "I appreciate you sharing that with me. It sounds like there's a lot on your mind. I'm here to listen and support you. What would be most helpful for you right now?"
    
    def _convert_to_brain_response(self, zoe_response: dict, start_time: float) -> Dict[str, Any]:
        """Convert Zoe response to Brain AgentResponse format"""
        processing_time = time.time() - start_time
        
        if zoe_response.get("success", False):
            return {
                "success": True,
                "content": zoe_response.get("response", ""),
                "metadata": {
                    "agent": "Zoe AI Companion",
                    "application": zoe_response.get("metadata", {}).get("application", "unknown"),
                    "session_id": zoe_response.get("session_id", ""),
                    "zoe_metadata": zoe_response.get("metadata", {}),
                    "personality_style": self.personality_style
                },
                "processing_time": processing_time,
                "session_id": zoe_response.get("session_id", "")
            }
        else:
            return {
                "success": False,
                "content": zoe_response.get("response", "I'm having trouble right now. Please try again."),
                "metadata": {
                    "agent": "Zoe AI Companion",
                    "error": zoe_response.get("error"),
                    "zoe_metadata": zoe_response.get("metadata", {})
                },
                "processing_time": processing_time
            }
    
    # Trauma-informed methods for Brain plugin interface
    
    async def assess_safety(self, brain_request) -> Dict[str, Any]:
        """Assess safety for trauma-informed care"""
        try:
            message = brain_request.message.lower()
            user_context = brain_request.user_context
            
            # Crisis indicators
            crisis_words = ["suicide", "kill myself", "end it all", "not worth living"]
            crisis_detected = any(word in message for word in crisis_words)
            
            # Self-harm indicators  
            self_harm_words = ["cut myself", "hurt myself", "self harm", "self-harm"]
            self_harm_detected = any(word in message for word in self_harm_words)
            
            # Substance abuse indicators
            substance_words = ["drunk", "high", "overdose", "pills", "drugs"]
            substance_detected = any(word in message for word in substance_words)
            
            # Violence indicators
            violence_words = ["hurt someone", "kill someone", "violence", "weapon"]
            violence_detected = any(word in message for word in violence_words)
            
            # Check ACE score for additional context
            ace_score = 0
            if user_context.trauma_context:
                ace_score = user_context.trauma_context.ace_score
            high_risk = ace_score >= 4
            
            # Overall safety assessment
            any_crisis = crisis_detected or self_harm_detected or substance_detected or violence_detected
            
            return {
                "status": "crisis" if any_crisis else "safe",
                "crisis_detected": crisis_detected,
                "self_harm_detected": self_harm_detected,
                "substance_detected": substance_detected,
                "violence_detected": violence_detected,
                "high_risk_user": high_risk,
                "ace_score": ace_score,
                "requires_intervention": any_crisis,
                "safety_level": "high_risk" if any_crisis else "moderate_risk" if high_risk else "low_risk"
            }
            
        except Exception as e:
            logger.error(f"Error in safety assessment: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "requires_intervention": False
            }
    
    async def apply_safety_filters(self, response_data) -> Dict[str, Any]:
        """Apply trauma-informed safety filters to response"""
        try:
            content = response_data.get("content", "")
            
            # Apply Zoe's personality filtering if available
            if self.zoe_core and hasattr(self.zoe_core, 'personality'):
                filtered_content = self.zoe_core.personality.filter_response(
                    content, {"trauma_safe_mode": True}
                )
                
                return {
                    "success": response_data.get("success", True),
                    "content": filtered_content,
                    "metadata": {
                        **response_data.get("metadata", {}),
                        "safety_filtered": True
                    },
                    "processing_time": response_data.get("processing_time", 0)
                }
            
            return response_data
            
        except Exception as e:
            logger.error(f"Error applying safety filters: {str(e)}")
            return response_data
    
    def get_crisis_resources(self, user_context) -> Dict[str, Any]:
        """Get crisis intervention resources"""
        return {
            "crisis_hotline": "988 Suicide & Crisis Lifeline",
            "text_line": "Text HOME to 741741",
            "emergency": "Call 911 for immediate emergency",
            "online_chat": "suicidepreventionlifeline.org/chat",
            "message": "Your safety is important. Please reach out to these resources if you need immediate help."
        }
    
    # Conversational methods for Brain plugin interface
    
    async def get_conversation_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get conversation history from Zoe system"""
        try:
            if self.zoe_core:
                history = self.zoe_core.get_conversation_history(session_id)
                return history if isinstance(history, list) else []
            return []
        except Exception as e:
            logger.error(f"Error getting conversation history: {str(e)}")
            return []
    
    async def clear_conversation(self, session_id: str) -> bool:
        """Clear conversation history in Zoe system"""
        try:
            if self.zoe_core:
                await self.zoe_core.end_session(session_id)
                return True
            return False
        except Exception as e:
            logger.error(f"Error clearing conversation {session_id}: {str(e)}")
            return False
    
    async def update_context(self, session_id: str, context: Dict[str, Any]) -> bool:
        """Update conversation context in Zoe system"""
        try:
            # Zoe's conversation manager handles context updates internally
            return True
        except Exception as e:
            logger.error(f"Error updating context for {session_id}: {str(e)}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Zoe Brain interface health"""
        return {
            "status": "healthy" if self._initialized else "not_initialized",
            "interface_initialized": self._initialized,
            "zoe_core_available": self.zoe_core is not None,
            "personality_style": self.personality_style,
            "service": "zoe_brain_interface"
        }
    
    def _is_high_trauma_risk(self, user_context: Dict[str, Any]) -> bool:
        """Check if user is at high trauma risk"""
        ace_score = user_context.get("ace_score", 0)
        return ace_score >= 4
    
    async def _process_trauma_informed_request(
        self, message: str, user_context: Dict[str, Any], application: str, 
        session_id: str, user_id: str, start_time: float
    ) -> Dict[str, Any]:
        """
        Process request using Zoe's trauma-informed workflow
        Steps: Safety Assessment → Context Loading → Processing → Safety Filtering → Response
        """
        try:
            # Step 1: Safety Assessment
            safety_assessment = await self._assess_safety(message, user_context)
            
            if safety_assessment.get("status") == "crisis":
                return await self._handle_crisis(user_context, start_time)
            elif safety_assessment.get("status") == "blocked":
                return self._create_blocked_response(start_time)
            
            # Step 2: Load conversation context
            await self._load_conversation_context(session_id, user_id)
            
            # Step 3: Enhance with shared knowledge (if available)
            knowledge_context = await self._get_shared_knowledge(message, application, user_context)
            
            # Step 4: Process with trauma-informed care + shared knowledge
            zoe_response = await self._process_message_direct(
                message=message,
                user_context=user_context,
                application=application,
                session_id=session_id,
                user_id=user_id,
                knowledge_context=knowledge_context
            )
            
            # Step 4: Apply safety filters
            if zoe_response.get("success"):
                filtered_content = await self._apply_safety_filters(
                    zoe_response.get("response", ""), user_context
                )
                zoe_response["response"] = filtered_content
            
            # Step 5: Add trauma-informed metadata
            zoe_response["metadata"] = zoe_response.get("metadata", {})
            zoe_response["metadata"].update({
                "workflow": "trauma_informed",
                "safety_assessment": safety_assessment,
                "trauma_risk_level": "high" if self._is_high_trauma_risk(user_context) else "low"
            })
            
            return self._convert_to_brain_response(zoe_response, start_time)
            
        except Exception as e:
            logger.error(f"Error in trauma-informed workflow: {str(e)}")
            return {
                "success": False,
                "content": "I'm here for you, even though I'm having some technical difficulties right now.",
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    async def _process_standard_request(
        self, message: str, user_context: Dict[str, Any], application: str,
        session_id: str, user_id: str, start_time: float
    ) -> Dict[str, Any]:
        """Process request using standard conversational workflow"""
        try:
            # Load conversation context
            await self._load_conversation_context(session_id, user_id)
            
            # Enhance with shared knowledge (if available)
            knowledge_context = await self._get_shared_knowledge(message, application, user_context)
            
            # Process with Zoe + shared knowledge
            zoe_response = await self._process_message_direct(
                message=message,
                user_context=user_context,
                application=application,
                session_id=session_id,
                user_id=user_id,
                knowledge_context=knowledge_context
            )
            
            # Add standard metadata
            zoe_response["metadata"] = zoe_response.get("metadata", {})
            zoe_response["metadata"].update({
                "workflow": "conversational"
            })
            
            return self._convert_to_brain_response(zoe_response, start_time)
            
        except Exception as e:
            logger.error(f"Error in standard workflow: {str(e)}")
            return {
                "success": False,
                "content": "I'm experiencing some technical difficulties. Please try again.",
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    async def _assess_safety(self, message: str, user_context: Dict[str, Any]) -> Dict[str, Any]:
        """Assess message for trauma triggers and safety concerns"""
        try:
            message_lower = message.lower()
            
            # Use Zoe's personality for crisis detection
            crisis_detected = self.zoe_core.personality.detect_crisis(message_lower)
            self_harm_detected = self.zoe_core.personality.detect_self_harm(message_lower)
            substance_detected = self.zoe_core.personality.detect_substance_abuse(message_lower)
            
            # Additional safety checks
            violence_words = ["hurt someone", "kill someone", "violence", "weapon"]
            violence_detected = any(word in message_lower for word in violence_words)
            
            ace_score = user_context.get("ace_score", 0)
            high_risk = ace_score >= 4
            
            any_crisis = crisis_detected or self_harm_detected or substance_detected or violence_detected
            
            return {
                "status": "crisis" if any_crisis else "safe",
                "crisis_detected": any_crisis,
                "self_harm_risk": self_harm_detected,
                "substance_abuse_risk": substance_detected,
                "violence_risk": violence_detected,
                "trauma_risk_score": ace_score,
                "high_trauma_risk": high_risk,
                "message": "Crisis detected, intervention needed." if any_crisis else "No immediate crisis detected."
            }
            
        except Exception as e:
            logger.error(f"Error in safety assessment: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    async def _handle_crisis(self, user_context: Dict[str, Any], start_time: float) -> Dict[str, Any]:
        """Handle crisis intervention with appropriate resources"""
        try:
            crisis_resources = self.zoe_core.personality.get_crisis_resources(user_context)
            
            crisis_content = (
                "I'm concerned about your safety and wellbeing. You're not alone in this. "
                "Please reach out to a mental health professional or crisis hotline immediately. "
                "Your life has value and there are people who want to help."
            )
            
            return {
                "success": True,
                "response": crisis_content,
                "metadata": {
                    "crisis_detected": True,
                    "resources": crisis_resources,
                    "workflow": "crisis_intervention"
                },
                "processing_time": time.time() - start_time,
                "safety_flags": ["crisis_intervention"]
            }
            
        except Exception as e:
            logger.error(f"Error in crisis handling: {str(e)}")
            return {
                "success": False,
                "response": "I'm concerned about you. Please reach out to a mental health professional.",
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    def _create_blocked_response(self, start_time: float) -> Dict[str, Any]:
        """Create response for blocked content"""
        return {
            "success": True,
            "response": "I understand you're going through something difficult. Let's talk about something that might be more helpful for you right now.",
            "metadata": {"workflow": "content_blocked"},
            "processing_time": time.time() - start_time
        }
    
    async def _load_conversation_context(self, session_id: str, user_id: str) -> None:
        """Load conversation context for continuity"""
        try:
            # This is handled by Zoe's conversation manager
            # Just ensure session exists
            self.zoe_core.conversation_manager.get_or_create_session(session_id, user_id)
        except Exception as e:
            logger.warning(f"Failed to load conversation context: {str(e)}")
    
    async def _apply_safety_filters(self, content: str, user_context: Dict[str, Any]) -> str:
        """Apply Zoe's trauma-informed safety filters to response"""
        try:
            # Use Zoe's personality for content filtering
            filtered_content = self.zoe_core.personality.filter_response(
                response=content,
                user_context=user_context,
                conversation_context=user_context  # Use user_context for both parameters
            )
            return filtered_content
        except Exception as e:
            logger.error(f"Error applying safety filters: {str(e)}")
            return content  # Return original if filtering fails
    
    async def _get_shared_knowledge(self, message: str, application: str, user_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get relevant knowledge from Brain's shared data sources"""
        if not self.data_registry:
            return []
        
        try:
            # Query shared knowledge based on message and context
            context = {
                "application": application,
                "user_id": user_context.get("user_id"),
                "ace_score": user_context.get("ace_score", 0)
            }
            
            # For healing-rooms, prioritize trauma resources
            if application == "healing-rooms":
                # Look for trauma support resources
                trauma_knowledge = await self.data_registry.query_best(
                    query=f"trauma support resources healing {message}",
                    context={**context, "type": "trauma_support"},
                    max_results=2
                )
                
                # Also get general knowledge
                general_knowledge = await self.data_registry.query_best(
                    query=message,
                    context=context,
                    max_results=1
                )
                
                return trauma_knowledge + general_knowledge
            
            else:
                # For other applications, get general knowledge
                return await self.data_registry.query_best(
                    query=message,
                    context=context,
                    max_results=3
                )
                
        except Exception as e:
            logger.warning(f"Failed to get shared knowledge: {str(e)}")
            return []
    
    def _summarize_shared_knowledge(self, knowledge: List[Dict[str, Any]], message: str) -> str:
        """Summarize shared knowledge relevant to the message"""
        if not knowledge:
            return ""
        
        try:
            # Extract key information from knowledge sources
            relevant_info = []
            for item in knowledge[:3]:  # Limit to top 3 items
                content = item.get("content", "")
                source = item.get("source", "knowledge base")
                
                if content and len(content) > 20:  # Only include substantial content
                    # Truncate long content
                    if len(content) > 200:
                        content = content[:200] + "..."
                    relevant_info.append(f"From {source}: {content}")
            
            return " | ".join(relevant_info) if relevant_info else ""
            
        except Exception as e:
            logger.warning(f"Failed to summarize shared knowledge: {str(e)}")
            return ""
    
    async def shutdown(self) -> None:
        """Shutdown Zoe Brain interface"""
        self.zoe_core = None
        self._initialized = False
        logger.info("Zoe Brain Interface shutdown complete")
