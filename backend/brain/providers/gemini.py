"""
Gemini Provider - Optional external AI provider for ThinkxLife Brain
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import os

logger = logging.getLogger(__name__)

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("Google Generative AI package not available. Install with: pip install google-generativeai")


class GeminiProvider:
    """
    Gemini provider for external AI capabilities.
    
    This provider integrates with Google's Gemini API to provide additional
    AI capabilities beyond the local chatbot system.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the Gemini provider"""
        if not GEMINI_AVAILABLE:
            raise ImportError("Google Generative AI package not available. Install with: pip install google-generativeai")
        
        load_dotenv()  # Ensure .env is loaded for API key
        
        self.config = config
<<<<<<< HEAD
<<<<<<< Updated upstream
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model = config.get("model", "gemini-1.5-flash")
=======
        self.api_key = config.get("api_key") or os.getenv("GEMINI_API_KEY")
        self.model_name = os.getenv("GEMINI_MODEL")
>>>>>>> Stashed changes
=======
        self.api_key = config.get("api_key") or os.getenv("GEMINI_API_KEY")
        self.model_name = config.get("model", "gemini-1.5-flash")
>>>>>>> main
        self.max_tokens = config.get("max_tokens", 2000)
        self.temperature = config.get("temperature", 0.7)
        self.timeout = config.get("timeout", 30.0)
        self.enabled = config.get("enabled", False)
        
        if not self.api_key:
            logger.warning("Gemini API key not found - provider will be disabled")
            self.enabled = False
            self.model = None
            return
            
        # Configure the SDK and initialize model
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
            logger.info(f"Gemini provider initialized with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {str(e)}")
            self.enabled = False
            self.model = None
    
    async def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a request using Gemini API
        
        Args:
            request_data: Enhanced request data from Brain
            
        Returns:
            Dictionary with response data
        """
        start_time = time.time()
        
        try:
            if not self.enabled or not self.model:
                return {
                    "success": False,
                    "error": "Gemini provider is disabled or not configured",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Extract request components
            message = request_data.get("message", "")
            system_prompt = request_data.get("system_prompt", "")
            user_context = request_data.get("user_context", {})
            application = request_data.get("application", "general")
            response_format = request_data.get("response_format", "json")
            
            # Build full prompt for Gemini (since it doesn't support multi-turn chat natively)
            full_prompt = ""
            if system_prompt:
                full_prompt += f"{system_prompt}\n\n"
            
            # Add conversation history if available
            history = user_context.get("conversation_history", [])
            for msg in history[-10:]:  # Last 10 messages
                role = msg.get("role", "user")
                content = msg.get("content", "")
                full_prompt += f"{role.capitalize()}: {content}\n"
            
            # Add current message
            full_prompt += f"User: {message}"
            
            # Prepare generation config
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=self.max_tokens,
                temperature=self.temperature)
                
            # Enable JSON output if requested (use snake_case for Python SDK)
            if response_format == "json":
                generation_config.response_mime_type = "application/json"
                # Optional: Add schema if needed for controlled JSON
                # generation_config.response_schema = {...}
            
            # Make API call (use asyncio.to_thread since generate_content is sync)
             # Use asyncio.wait_for to prevent leaking TimerHandles
            try:
                response = await asyncio.wait_for(
                    self.model.generate_content_async(
                        contents=full_prompt,
                        generation_config=generation_config
                    ),
                    timeout=self.timeout
                )
            except asyncio.TimeoutError:
                return {
                    "success": False,
                    "error": "Gemini request timed out",
                    "timestamp": datetime.now().isoformat(),
                    "metadata": {"provider": "gemini"}
                }
    
                
            # Extract response
            ai_message = response.text
            tokens_used = response.usage_metadata.total_token_count if hasattr(response, 'usage_metadata') else None
            
            # Build Brain response
            brain_response = {
                "success": True,
                "message": ai_message,
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "provider": "gemini",
                    "model": self.model_name,
                    "tokens_used": tokens_used,
                    "processing_time": time.time() - start_time,
                    "application": application,
                    "sources": ["Gemini API"]
                }
            }
            
            # Add application-specific metadata
            self._add_application_metadata(brain_response, application)
            
            return brain_response
            
        except Exception as e:
            logger.error(f"Error in Gemini provider: {str(e)}")
            return {
                "success": False,
                "error": f"Gemini provider error: {str(e)}",
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "provider": "gemini",
                    "processing_time": time.time() - start_time
                }
    
    }
    
    def _add_application_metadata(self, response: Dict[str, Any], application: str):
        """Add application-specific metadata"""
        
        if application == "healing-rooms":
            response["metadata"]["trauma_informed"] = True
            response["metadata"]["safety_checked"] = True
        elif application == "inside-our-ai":
            response["metadata"]["educational"] = True
            response["metadata"]["ethics_focused"] = True
        elif application == "compliance":
            response["metadata"]["regulatory"] = True
            response["metadata"]["compliance_checked"] = True
        elif application == "exterior-spaces":
            response["metadata"]["creative"] = True
            response["metadata"]["design_focused"] = True
    
    async def health_check(self) -> Dict[str, Any]:
        """Check the health of the Gemini provider"""
        
        try:
            if not self.enabled or not self.model:
                return {
                    "status": "disabled",
                    "message": "Gemini provider is disabled or not configured"
                }
            
            # Test API connectivity asynchronously
            test_response = await self.model.generate_content_async("Health check")
            
            if test_response.text:
                return {
                    "status": "healthy",
                    "message": "Gemini provider is operational",
                    "model": self.model,
                    "api_status": "connected"
                }
            else:
                return {
                    "status": "degraded",
                    "message": "Gemini API responding but with issues"
                }
                
        except Exception as e:
            logger.error(f"Gemini health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "message": f"Gemini health check failed: {str(e)}"
            }
    
    async def close(self):
        """Close the Gemini provider connection"""
        
        try:
            # Gemini doesn't have explicit close, so pass
            pass
                
            logger.info("Gemini provider closed successfully")
            
        except Exception as e:
            logger.error(f"Error closing Gemini provider: {str(e)}")
    
    def get_config(self) -> Dict[str, Any]:
        """Get provider configuration"""
        return {
            "provider": "gemini",
            "enabled": self.enabled,
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "timeout": self.timeout,
            "api_key_configured": bool(self.api_key)
        }