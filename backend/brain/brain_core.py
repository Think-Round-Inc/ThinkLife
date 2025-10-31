"""
ThinkxLife Brain Core - Central AI orchestration system
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime

# Types are used in other modules but not directly in brain_core
# Providers are imported dynamically in _initialize_providers()

logger = logging.getLogger(__name__)


class ThinkxLifeBrain:
    """
    Central AI Brain that orchestrates all AI operations across ThinkxLife platform.
    
    This class manages:
    - OpenAI provider integration
    - Application-specific routing and context
    - Security and rate limiting
    - Health monitoring
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls, config=None):
        """Singleton pattern implementation"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, config=None):
        """Initialize the Brain with configuration"""
        if self._initialized:
            return
            
        self.config = config or self._get_default_config()
        self.providers = {}
        
        # Analytics
        self.analytics = {
            "total_requests": 0,
            "success_rate": 0.0,
            "average_response_time": 0.0,
            "provider_usage": {},
            "application_usage": {},
            "error_rate": 0.0,
            "uptime": 0.0
        }
        self.start_time = datetime.now()
        
        # Initialize providers
        self._initialize_providers()
        
        self._initialized = True
        logger.info("ThinkxLife Brain initialized successfully")
    
    def _get_default_config(self):
        """Get default Brain configuration"""
        return {
            "providers": {
                "openai": {
                    "enabled": True,
                    "model": "gpt-4o-mini",
                    "max_tokens": 2000,
                    "temperature": 0.7
                }
            },
            "security": {
                "rate_limiting": {
                    "enabled": True,
                    "max_requests_per_minute": 60,
                    "max_requests_per_hour": 1000
                },
                "content_filtering": {
                    "enabled": True,
                    "trauma_safe_mode": True
                },
                "user_validation": {
                    "require_auth": True,
                    "allow_anonymous": False
                }
            },
            "context": {
                "max_history_length": 20,
                "context_retention_hours": 24,
                "enable_personalization": True
            },
            "session": {
                "timeout_minutes": 30,
                "max_concurrent_sessions": 5,
                "enable_session_analytics": True
            }
        }
    
    def _initialize_providers(self):
        """Initialize AI providers based on configuration"""
        provider_configs = self.config["providers"]
        
        # Initialize only OpenAI Provider
        try:
            if provider_configs.get("openai", {}).get("enabled", False):
                from .providers.openai import OpenAIProvider
                self.providers["openai"] = OpenAIProvider(provider_configs["openai"])
                logger.info("OpenAI provider initialized")
        except ImportError:
            logger.warning("OpenAI provider not available")
    
    async def process_request(self, request_data):
        """
        Main entry point for processing Brain requests
        
        Args:
            request_data: Dictionary containing the request data
            
        Returns:
            Dictionary with the AI's response
        """
        start_time = time.time()
        request_id = request_data.get("id", str(uuid.uuid4()))
        
        try:
            # Update analytics
            self.analytics["total_requests"] += 1
            application = request_data.get("application", "general")
            self.analytics["application_usage"][application] = \
                self.analytics["application_usage"].get(application, 0) + 1
            
            # Basic security validation
            if not self._validate_request(request_data):
                return {
                    "id": request_id,
                    "success": False,
                    "error": "Request validation failed",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Route to appropriate handler
            response = await self._route_request(request_data)
            
            # Update analytics
            processing_time = time.time() - start_time
            self.analytics["average_response_time"] = (
                (self.analytics["average_response_time"] * (self.analytics["total_requests"] - 1) + processing_time) 
                / self.analytics["total_requests"]
            )
            
            if response.get("success", False):
                self.analytics["success_rate"] = (
                    (self.analytics["success_rate"] * (self.analytics["total_requests"] - 1) + 1.0) 
                    / self.analytics["total_requests"]
                )
            else:
                self.analytics["error_rate"] = (
                    (self.analytics["error_rate"] * (self.analytics["total_requests"] - 1) + 1.0) 
                    / self.analytics["total_requests"]
                )
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing Brain request {request_id}: {str(e)}")
            self.analytics["error_rate"] = (
                (self.analytics["error_rate"] * (self.analytics["total_requests"] - 1) + 1.0) 
                / self.analytics["total_requests"]
            )
            
            return {
                "id": request_id,
                "success": False,
                "error": f"Internal Brain error: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    def _validate_request(self, request_data):
        """Basic request validation"""
        required_fields = ["message", "application", "user_context"]
        return all(field in request_data for field in required_fields)
    
    async def _route_request(self, request_data):
        """Route request to appropriate application handler"""
        
        application = request_data.get("application", "general")
        
        handlers = {
            "healing-rooms": self._handle_healing_rooms,
            "inside-our-ai": self._handle_ai_awareness,
            "chatbot": self._handle_chatbot,
            "compliance": self._handle_compliance,
            "exterior-spaces": self._handle_exterior_spaces,
            "general": self._handle_general
        }
        
        handler = handlers.get(application, self._handle_general)
        return await handler(request_data)
    
    async def _handle_healing_rooms(self, request_data):
        """Handle healing rooms requests with trauma-informed approach"""
        
        # Use trauma-safe prompting
        system_prompt = self._get_healing_rooms_prompt(request_data.get("user_context", {}))
        
        # Select appropriate provider (prefer local for sensitive content)
        provider = self._select_provider("healing-rooms")
        
        # Process with trauma context
        enhanced_request = {
            "message": request_data["message"],
            "system_prompt": system_prompt,
            "user_context": request_data.get("user_context", {}),
            "application": "healing-rooms",
            "trauma_safe": True
        }
        
        response = await provider.process_request(enhanced_request)
        
        # Post-process for trauma safety
        safe_response = await self._ensure_trauma_safety(response, request_data.get("user_context", {}))
        
        return safe_response
    
    async def _handle_ai_awareness(self, request_data):
        """Handle Inside our AI showcase requests"""
        
        # Educational system prompt
        system_prompt = self._get_ai_awareness_prompt(request_data.get("user_context", {}))
        
        # Select provider based on educational needs
        provider = self._select_provider("inside-our-ai")
        
        # Enhance with educational context
        enhanced_request = {
            "message": request_data["message"],
            "system_prompt": system_prompt,
            "user_context": request_data.get("user_context", {}),
            "application": "inside-our-ai",
            "educational": True
        }
        
        response = await provider.process_request(enhanced_request)
        
        # Add educational metadata
        if response.get("success") and "metadata" in response:
            response["metadata"]["sources"] = ["ThinkxLife AI Ethics Database", "Educational Content"]
        
        return response
    
    async def _handle_chatbot(self, request_data):
        """Handle general chatbot requests"""
        
        system_prompt = self._get_chatbot_prompt(request_data.get("user_context", {}))
        provider = self._select_provider("chatbot")
        
        # Use existing chatbot core logic
        enhanced_request = {
            "message": request_data["message"],
            "system_prompt": system_prompt,
            "user_context": request_data.get("user_context", {}),
            "application": "chatbot"
        }
        
        return await provider.process_request(enhanced_request)
    
    async def _handle_compliance(self, request_data):
        """Handle compliance and regulatory requests"""
        
        system_prompt = self._get_compliance_prompt(request_data.get("user_context", {}))
        provider = self._select_provider("compliance")
        
        enhanced_request = {
            "message": request_data["message"],
            "system_prompt": system_prompt,
            "user_context": request_data.get("user_context", {}),
            "application": "compliance",
            "regulatory_focus": True
        }
        
        response = await provider.process_request(enhanced_request)
        
        # Add compliance metadata
        if response.get("success") and "metadata" in response:
            response["metadata"]["sources"] = ["Regulatory Database", "Compliance Guidelines"]
        
        return response
    
    async def _handle_exterior_spaces(self, request_data):
        """Handle exterior spaces creative AI requests"""
        
        system_prompt = self._get_exterior_spaces_prompt(request_data.get("user_context", {}))
        provider = self._select_provider("exterior-spaces")
        
        enhanced_request = {
            "message": request_data["message"],
            "system_prompt": system_prompt,
            "user_context": request_data.get("user_context", {}),
            "application": "exterior-spaces",
            "creative": True
        }
        
        return await provider.process_request(enhanced_request)
    
    async def _handle_general(self, request_data):
        """Handle general requests"""
        
        system_prompt = self._get_general_prompt(request_data.get("user_context", {}))
        provider = self._select_provider("general")
        
        enhanced_request = {
            "message": request_data["message"],
            "system_prompt": system_prompt,
            "user_context": request_data.get("user_context", {}),
            "application": "general"
        }
        
        return await provider.process_request(enhanced_request)
    
    def _select_provider(self, application):
        """Select the best provider for the request"""
        
        # Use OpenAI provider
        if "openai" in self.providers:
            return self.providers["openai"]
        
        raise RuntimeError("No available providers")
    
    def _get_healing_rooms_prompt(self, user_context):
        """Get trauma-informed system prompt for healing rooms"""
        ace_score = user_context.get("ace_score", 0.0)
        ace_details = user_context.get("ace_details", [])
        
        # Create sensitive context about specific trauma areas without being explicit
        trauma_context = ""
        if ace_details:
            trauma_areas = []
            for detail in ace_details:
                if "swear at you" in detail or "insult you" in detail:
                    trauma_areas.append("emotional abuse")
                elif "push, grab, slap" in detail:
                    trauma_areas.append("physical abuse")
                elif "touch, fondle" in detail:
                    trauma_areas.append("sexual abuse")
                elif "no one in your family loved you" in detail:
                    trauma_areas.append("emotional neglect")
                elif "didn't have enough to eat" in detail:
                    trauma_areas.append("physical neglect")
                elif "parents ever separated" in detail:
                    trauma_areas.append("family dysfunction")
                elif "mother or stepmother" in detail and "pushed" in detail:
                    trauma_areas.append("domestic violence")
                elif "problem drinker" in detail or "street drugs" in detail:
                    trauma_areas.append("household substance abuse")
                elif "depressed, mentally ill" in detail:
                    trauma_areas.append("household mental illness")
                elif "jail or prison" in detail:
                    trauma_areas.append("household member incarceration")
            
            if trauma_areas:
                unique_areas = list(set(trauma_areas))
                trauma_context = f"\n- Specific areas of concern include: {', '.join(unique_areas)}"
        
        return f"""You are Zoe, an empathetic AI companion for ThinkxLife's healing rooms. 
        
        Guidelines:
        - Always prioritize user safety and emotional well-being
        - Use gentle, non-triggering language that validates their experiences
        - Acknowledge trauma without re-traumatizing or being explicit
        - Focus on healing, growth, and resilience
        - Never provide medical or therapeutic advice
        - Encourage professional help when appropriate
        - Be especially gentle and validating given their trauma history
        
        User Context:
        - ACE Score: {ace_score}
        - This indicates {'higher' if ace_score > 4 else 'moderate' if ace_score > 1 else 'lower'} trauma exposure{trauma_context}
        - Respond with extra empathy, validation, and hope
        - Acknowledge their strength in surviving and seeking support
        
        Remember: This person has shown courage by sharing their story. Treat them with the utmost care and respect."""
    
    def _get_ai_awareness_prompt(self, user_context):
        """Get system prompt for Inside our AI showcase"""
        
        return f"""You are an AI representative showcasing how Think Round Inc uses AI to enhance their programs.
        
        Guidelines:
        - Explain how Think Round integrates AI into healing rooms, arts programs, and community initiatives
        - Focus on AI as an enhancement tool, not replacement for human connection
        - Emphasize trauma-informed, culturally sensitive AI applications
        - Share examples of AI supporting creativity, healing, and community building
        - Maintain Think Round's values of human dignity and cultural authenticity
        
        Your role is to demonstrate Think Round's thoughtful AI integration across their various programs."""
    
    def _get_chatbot_prompt(self, user_context):
        """Get general chatbot system prompt"""
        return """You are ThinkxLife's AI assistant, focused on ethical AI, healing, and human wellbeing.
        
        Guidelines:
        - Be helpful, empathetic, and ethical
        - Respect user privacy and boundaries
        - Promote positive mental health and wellbeing
        - Showcase responsible AI integration within Think Round programs
        - Never provide medical, legal, or financial advice
        
        You represent ThinkxLife's values of ethical AI and human-centered technology."""
    
    def _get_compliance_prompt(self, user_context):
        """Get compliance-focused system prompt"""
        return """You are a compliance-focused AI assistant for ThinkxLife.
        
        Guidelines:
        - Provide general information about AI regulations
        - Focus on GDPR, AI Act, and ethical AI frameworks
        - Never provide legal advice
        - Encourage consultation with legal professionals
        - Emphasize responsible AI practices
        
        Help users understand AI compliance landscape."""
    
    def _get_exterior_spaces_prompt(self, user_context):
        """Get creative system prompt for exterior spaces"""
        return """You are a creative AI assistant for ThinkxLife's exterior spaces platform.
        
        Guidelines:
        - Inspire creativity in outdoor and architectural design
        - Consider sustainability and environmental impact
        - Promote inclusive and accessible design
        - Encourage connection with nature
        - Balance aesthetics with functionality
        
        Help users envision beautiful, sustainable exterior spaces."""
    
    def _get_general_prompt(self, user_context):
        """Get general system prompt"""
        ace_score = user_context.get("ace_score", 0.0)
        
        # Add trauma-sensitive context if ACE score is present
        trauma_sensitivity = ""
        if ace_score > 0:
            trauma_sensitivity = """
        - Use trauma-informed language and approaches
        - Be especially gentle and validating
        - Acknowledge the user's strength and resilience
        - Avoid triggering language or assumptions"""
        
        return f"""You are Zoe, ThinkxLife's empathetic AI assistant.
        
        Guidelines:
        - Be helpful, empathetic, and supportive
        - Maintain ethical AI principles
        - Respect user privacy and boundaries
        - Promote wellbeing and positive mental health
        - Stay within your knowledge and capabilities{trauma_sensitivity}
        
        Assist users with warmth and understanding while maintaining appropriate boundaries."""
    
    async def _ensure_trauma_safety(self, response, user_context):
        """Ensure response is trauma-safe"""
        # Implement trauma safety checks
        if response.get("success") and response.get("message"):
            # Basic trauma safety filtering
            trigger_words = ["suicide", "self-harm", "abuse", "violence"]
            message_lower = response["message"].lower()
            
            for trigger in trigger_words:
                if trigger in message_lower:
                    # Add safety disclaimer
                    response["message"] += "\n\n⚠️ If you're experiencing crisis thoughts, please contact a mental health professional or crisis hotline immediately."
                    break
        
        return response
    
    async def get_health_status(self):
        """Get overall Brain health status"""
        
        provider_health = {}
        overall_status = "healthy"
        
        # Check each provider
        for name, provider in self.providers.items():
            try:
                health = await provider.health_check()
                provider_health[name] = health
                
                if health.get("status") != "healthy":
                    overall_status = "degraded"
                    
            except Exception as e:
                provider_health[name] = {"status": "unhealthy", "error": str(e)}
                overall_status = "unhealthy"
        
        # System health
        uptime = (datetime.now() - self.start_time).total_seconds()
        system_health = {
            "uptime_seconds": uptime,
            "total_requests": self.analytics["total_requests"],
            "success_rate": self.analytics["success_rate"],
            "error_rate": self.analytics["error_rate"],
            "average_response_time": self.analytics["average_response_time"]
        }
        
        return {
            "overall": overall_status,
            "providers": provider_health,
            "system": system_health,
            "timestamp": datetime.now().isoformat()
        }
    
    async def get_analytics(self):
        """Get Brain analytics"""
        
        # Update uptime
        uptime = (datetime.now() - self.start_time).total_seconds()
        self.analytics["uptime"] = uptime / 3600  # Convert to hours
        
        return self.analytics
    
    async def shutdown(self):
        """Gracefully shutdown the Brain"""
        
        logger.info("Shutting down ThinkxLife Brain...")
        
        # Close provider connections
        for provider in self.providers.values():
            if hasattr(provider, 'close'):
                await provider.close()
        
        logger.info("ThinkxLife Brain shutdown complete") 