"""
Anthropic Provider - Integration with Anthropic Claude models
"""

import logging
import os
import time
from typing import Dict, Any, List, Optional, Union


logger = logging.getLogger(__name__)

try:
    import anthropic
    from anthropic import AsyncAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    logger.warning("Anthropic library not available. Install with: pip install anthropic")
    ANTHROPIC_AVAILABLE = False
    AsyncAnthropic = None


class AnthropicProvider:
    """
    Anthropic provider - focused on initialization and request processing
    Configuration validation is handled by the provider registry
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize Anthropic provider with pre-validated configuration"""
        self.config = config or {}
        self.name = "anthropic"
        self._initialized = False
        self.client = None
        
        logger.info(f"Initializing {self.name} provider")
    
    
    async def initialize(self) -> bool:
        """Initialize the provider client - config already validated by registry"""
        try:
            if not ANTHROPIC_AVAILABLE:
                logger.error("Anthropic library not available")
                return False
            
            await self._initialize_provider()
            self._initialized = True
            logger.info("Anthropic provider initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic provider: {str(e)}")
            return False
    
    async def _initialize_provider(self) -> None:
        """Initialize Anthropic client"""
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic library not available")
        
        client_config = {
            "api_key": self.config["api_key"],
            "timeout": self.config.get("timeout", 30.0),
            "max_retries": self.config.get("max_retries", 3),
        }
        
        self.client = AsyncAnthropic(**client_config)
        logger.info(f"Anthropic client initialized with model: {self.config['model']}")
    
    async def generate_response(
        self, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate response using Anthropic Claude
        
        Args:
            messages: List of message dictionaries
            **kwargs: Additional parameters (overrides config)
        
        Returns:
            Dictionary with response content and metadata
        """
        if not self._initialized:
            raise RuntimeError("Provider not initialized")
        
        try:
            # Separate system message from regular messages
            system_message = kwargs.get("system", self.config.get("system"))
            user_messages = []
            
            for msg in messages:
                if msg.get("role") == "system" and not system_message:
                    system_message = msg["content"]
                elif msg.get("role") != "system":
                    user_messages.append(msg)
            
            # Merge kwargs with config (kwargs take precedence)
            request_params = {
                "model": kwargs.get("model", self.config["model"]),
                "max_tokens": kwargs.get("max_tokens", self.config["max_tokens"]),
                "temperature": kwargs.get("temperature", self.config["temperature"]),
                "top_p": kwargs.get("top_p", self.config["top_p"]),
                "messages": user_messages,
                "stream": kwargs.get("stream", self.config["stream"]),
            }
            
            # Add system message if present
            if system_message:
                request_params["system"] = system_message
            
            # Add optional parameters if provided in kwargs or config
            optional_params = [
                "top_k", "stop_sequences", "metadata", "stop_reason", 
                "tools", "tool_choice", "extra_headers", "extra_query", "extra_body"
            ]
            
            for param in optional_params:
                if param in kwargs:
                    request_params[param] = kwargs[param]
                elif self.config.get(param) is not None:
                    request_params[param] = self.config[param]
            
            # Remove None values
            request_params = {k: v for k, v in request_params.items() if v is not None}
            
            # Make the request
            response = await self.client.messages.create(**request_params)
            
            # Extract response
            if hasattr(response, 'content') and response.content:
                # Get text content
                text_content = ""
                for content_block in response.content:
                    if content_block.type == "text":
                        text_content += content_block.text
                
                # Build metadata
                metadata = {
                    "model": response.model,
                    "usage": response.usage.dict() if response.usage else {},
                    "stop_reason": response.stop_reason,
                    "stop_sequence": response.stop_sequence,
                    "provider": "anthropic"
                }
                
                # Add tool use info if present
                tool_uses = []
                for content_block in response.content:
                    if content_block.type == "tool_use":
                        tool_uses.append({
                            "id": content_block.id,
                            "name": content_block.name,
                            "input": content_block.input
                        })
                
                if tool_uses:
                    metadata["tool_uses"] = tool_uses
                
                return {
                    "content": text_content,
                    "metadata": metadata,
                    "success": True
                }
            else:
                return {
                    "content": "",
                    "metadata": {"error": "No response content from Anthropic"},
                    "success": False
                }
                
        except Exception as e:
            logger.error(f"Anthropic request failed: {str(e)}")
            return {
                "content": "",
                "metadata": {"error": str(e), "provider": "anthropic"},
                "success": False
            }
    
    async def generate_streaming_response(
        self, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate streaming response using Anthropic Claude
        
        Args:
            messages: List of message dictionaries
            **kwargs: Additional parameters
        
        Returns:
            List of response chunks
        """
        if not self._initialized:
            raise RuntimeError("Provider not initialized")
        
        try:
            # Set streaming to True
            kwargs["stream"] = True
            response_stream = await self.generate_response(messages, **kwargs)
            
            # Handle streaming response
            chunks = []
            async for chunk in response_stream:
                if hasattr(chunk, 'delta') and chunk.delta.content:
                    chunks.append({
                        "content": chunk.delta.content,
                        "metadata": {
                            "stop_reason": chunk.delta.stop_reason if hasattr(chunk.delta, 'stop_reason') else None,
                            "provider": "anthropic"
                        }
                    })
            
            return chunks
            
        except Exception as e:
            logger.error(f"Anthropic streaming request failed: {str(e)}")
            return [{"content": "", "metadata": {"error": str(e)}, "success": False}]
    
    async def health_check(self) -> Dict[str, Any]:
        """Check provider health"""
        try:
            start_time = time.time()
            
            # Simple test message
            test_messages = [{"role": "user", "content": "Test"}]
            response = await self.generate_response(test_messages, max_tokens=1)
            
            response_time = time.time() - start_time
            
            return {
                "healthy": True,
                "response_time": response_time,
                "provider": self.name,
                "model": self.config.get("model", "unknown")
            }
            
        except Exception as e:
            logger.error(f"Health check failed for Anthropic provider: {str(e)}")
            return {
                "healthy": False,
                "error": str(e),
                "provider": self.name
            }
    
    async def close(self) -> None:
        """Close provider connections"""
        await self._close_provider()
        self._initialized = False
        logger.info(f"Anthropic provider closed")
    
    
    async def _close_provider(self) -> None:
        """Close Anthropic client"""
        if hasattr(self, 'client'):
            # Anthropic async client doesn't need explicit closing
            pass


# Factory function for easy instantiation
def create_anthropic_provider(config: Optional[Dict[str, Any]] = None) -> AnthropicProvider:
    """Create Anthropic provider instance"""
    if config is None:
        config = {}
    return AnthropicProvider(config)
