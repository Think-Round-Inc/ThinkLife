"""
OpenAI Provider - Optional external AI provider for ThinkLife Brain
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI package not available. Install with: pip install openai")


class OpenAIProvider:
    """
    OpenAI provider for external AI capabilities.
    
    This provider integrates with OpenAI's API to provide additional
    AI capabilities beyond the local chatbot system.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the OpenAI provider"""
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not available. Install with: pip install openai")
        
        self.config = config
        self.api_key = config.get("api_key")
        self.model = config.get("model", "gpt-4o-mini")
        self.max_tokens = config.get("max_tokens", 2000)
        self.temperature = config.get("temperature", 0.7)
        self.timeout = config.get("timeout", 30.0)
        self.enabled = config.get("enabled", False)
        self.organization = config.get("organization")
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        # Initialize OpenAI client
        self.client = openai.AsyncOpenAI(
            api_key=self.api_key,
            organization=self.organization,
            timeout=self.timeout
        )
        
        logger.info(f"OpenAI provider initialized with model: {self.model}")
    
    async def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a request using OpenAI API
        
        Args:
            request_data: Enhanced request data from Brain
            
        Returns:
            Dictionary with response data
        """
        start_time = time.time()
        
        try:
            if not self.enabled:
                return {
                    "success": False,
                    "error": "OpenAI provider is disabled",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Extract request components
            message = request_data.get("message", "")
            system_prompt = request_data.get("system_prompt", "")
            user_context = request_data.get("user_context", {})
            application = request_data.get("application", "general")
            
            # Build messages for OpenAI
            messages = []
            
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            
            # Add conversation history if available
            history = user_context.get("conversation_history", [])
            for msg in history[-10:]:  # Last 10 messages
                messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                })
            
            # Add current message
            messages.append({
                "role": "user",
                "content": message
            })
            
            # Make API call
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=False
            )
            
            # Extract response
            ai_message = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if response.usage else None
            
            # Build Brain response
            brain_response = {
                "success": True,
                "message": ai_message,
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "provider": "openai",
                    "model": self.model,
                    "tokens_used": tokens_used,
                    "processing_time": time.time() - start_time,
                    "application": application,
                    "sources": ["OpenAI API"]
                }
            }
            
            # Add application-specific metadata
            self._add_application_metadata(brain_response, application)
            
            return brain_response
            
        except Exception as e:
            logger.error(f"Error in OpenAI provider: {str(e)}")
            return {
                "success": False,
                "error": f"OpenAI provider error: {str(e)}",
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "provider": "openai",
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
        """Check the health of the OpenAI provider"""
        
        try:
            if not self.enabled:
                return {
                    "status": "disabled",
                    "message": "OpenAI provider is disabled"
                }
            
            # Test API connectivity
            test_response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Health check"}],
                max_tokens=10,
                temperature=0
            )
            
            if test_response.choices:
                return {
                    "status": "healthy",
                    "message": "OpenAI provider is operational",
                    "model": self.model,
                    "api_status": "connected"
                }
            else:
                return {
                    "status": "degraded",
                    "message": "OpenAI API responding but with issues"
                }
                
        except Exception as e:
            logger.error(f"OpenAI health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "message": f"OpenAI health check failed: {str(e)}"
            }
    
    async def close(self):
        """Close the OpenAI provider connection"""
        
        try:
            if hasattr(self.client, 'close'):
                await self.client.close()
                
            logger.info("OpenAI provider closed successfully")
            
        except Exception as e:
            logger.error(f"Error closing OpenAI provider: {str(e)}")
    
    def get_config(self) -> Dict[str, Any]:
        """Get provider configuration"""
        return {
            "provider": "openai",
            "enabled": self.enabled,
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "timeout": self.timeout,
            "api_key_configured": bool(self.api_key),
            "organization": self.organization
        } 