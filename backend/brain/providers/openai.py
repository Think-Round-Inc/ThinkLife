"""
OpenAI Provider - Integration with OpenAI GPT models
"""

import logging
import os
import time
from typing import Dict, Any, List, Optional, Union
import asyncio

logger = logging.getLogger(__name__)

try:
    import openai
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    logger.warning("OpenAI library not available. Install with: pip install openai")
    OPENAI_AVAILABLE = False
    AsyncOpenAI = None


class OpenAIProvider:
    """
    OpenAI provider - focused on initialization and request processing
    Configuration validation is handled by the provider registry
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize OpenAI provider with pre-validated configuration"""
        self.config = config or {}
        self.name = "openai"
        self._initialized = False
        self.client = None
        
        logger.info(f"Initializing {self.name} provider")
    
    
    async def initialize(self) -> bool:
        """Initialize the provider client - config already validated by registry"""
        try:
            if not OPENAI_AVAILABLE:
                logger.error("OpenAI library not available")
                return False
            
            await self._initialize_provider()
            self._initialized = True
            logger.info("OpenAI provider initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI provider: {str(e)}")
            return False
    
    
    async def _initialize_provider(self) -> None:
        """Initialize OpenAI client"""
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not available")
        
        client_config = {
            "api_key": self.config["api_key"],
            "timeout": self.config.get("timeout", 30.0),
            "max_retries": self.config.get("max_retries", 3),
        }
        
        # Add optional configs
        if self.config.get("organization"):
            client_config["organization"] = self.config["organization"]
        
        if self.config.get("base_url"):
            client_config["base_url"] = self.config["base_url"]
        
        self.client = AsyncOpenAI(**client_config)
        logger.info(f"OpenAI client initialized with model: {self.config['model']}")
    
    async def generate_response(
        self, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate response using OpenAI
        
        Args:
            messages: List of message dictionaries
            **kwargs: Additional parameters (overrides config)
        
        Returns:
            Dictionary with response content and metadata
        """
        if not self._initialized:
            raise RuntimeError("Provider not initialized")
        
        try:
            # Merge kwargs with config (kwargs take precedence)
            request_params = {
                "model": kwargs.get("model", self.config["model"]),
                "messages": messages,
                "max_tokens": kwargs.get("max_tokens", self.config["max_tokens"]),
                "temperature": kwargs.get("temperature", self.config["temperature"]),
                "top_p": kwargs.get("top_p", self.config["top_p"]),
                "frequency_penalty": kwargs.get("frequency_penalty", self.config["frequency_penalty"]),
                "presence_penalty": kwargs.get("presence_penalty", self.config["presence_penalty"]),
                "stream": kwargs.get("stream", self.config["stream"]),
            }
            
            # Add optional parameters if provided in kwargs or config
            optional_params = [
                "stop", "user", "logit_bias", "functions", "function_call", 
                "tools", "tool_choice", "response_format", "seed", "extra_headers",
                "extra_query", "extra_body"
            ]
            
            for param in optional_params:
                if param in kwargs:
                    request_params[param] = kwargs[param]
                elif self.config.get(param) is not None:
                    request_params[param] = self.config[param]
            
            # Remove None values
            request_params = {k: v for k, v in request_params.items() if v is not None}
            
            # Make the request
            response = await self.client.chat.completions.create(**request_params)
            
            # Extract response
            if hasattr(response, 'choices') and response.choices:
                choice = response.choices[0]
                content = choice.message.content or ""
                
                # Build metadata
                metadata = {
                    "model": response.model,
                    "usage": response.usage.dict() if response.usage else {},
                    "finish_reason": choice.finish_reason,
                    "index": choice.index,
                    "provider": "openai"
                }
                
                # Add function call info if present
                if hasattr(choice.message, 'function_call') and choice.message.function_call:
                    metadata["function_call"] = choice.message.function_call
                
                if hasattr(choice.message, 'tool_calls') and choice.message.tool_calls:
                    metadata["tool_calls"] = [call.dict() for call in choice.message.tool_calls]
                
                return {
                    "content": content,
                    "metadata": metadata,
                    "success": True
                }
            else:
                return {
                    "content": "",
                    "metadata": {"error": "No response from OpenAI"},
                    "success": False
                }
                
        except Exception as e:
            logger.error(f"OpenAI request failed: {str(e)}")
            return {
                "content": "",
                "metadata": {"error": str(e), "provider": "openai"},
                "success": False
            }
    
    async def generate_embeddings(
        self, 
        input_text: Union[str, List[str]], 
        model: str = "text-embedding-ada-002",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate embeddings using OpenAI
        
        Args:
            input_text: Text or list of texts to embed
            model: Embedding model to use
            **kwargs: Additional parameters
        """
        if not self._initialized:
            raise RuntimeError("Provider not initialized")
        
        try:
            request_params = {
                "model": model,
                "input": input_text,
                **kwargs
            }
            
            response = await self.client.embeddings.create(**request_params)
            
            return {
                "embeddings": [data.embedding for data in response.data],
                "usage": response.usage.dict() if response.usage else {},
                "model": response.model,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"OpenAI embeddings request failed: {str(e)}")
            return {
                "embeddings": [],
                "error": str(e),
                "success": False
            }
    
    async def generate_image(
        self,
        prompt: str,
        model: str = "dall-e-3",
        size: str = "1024x1024",
        quality: str = "standard",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate image using DALL-E
        
        Args:
            prompt: Image description
            model: Image generation model
            size: Image size
            quality: Image quality
            **kwargs: Additional parameters
        """
        if not self._initialized:
            raise RuntimeError("Provider not initialized")
        
        try:
            request_params = {
                "model": model,
                "prompt": prompt,
                "size": size,
                "quality": quality,
                **kwargs
            }
            
            response = await self.client.images.generate(**request_params)
            
            return {
                "images": [image.url for image in response.data],
                "metadata": {
                    "model": response.model,
                    "created": response.created,
                    "provider": "openai"
                },
                "success": True
            }
            
        except Exception as e:
            logger.error(f"OpenAI image generation failed: {str(e)}")
            return {
                "images": [],
                "error": str(e),
                "success": False
            }
    
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
            logger.error(f"Health check failed for OpenAI provider: {str(e)}")
            return {
                "healthy": False,
                "error": str(e),
                "provider": self.name
            }
    
    async def close(self) -> None:
        """Close provider connections"""
        await self._close_provider()
        self._initialized = False
        logger.info(f"OpenAI provider closed")
    
    
    async def _close_provider(self) -> None:
        """Close OpenAI client"""
        if hasattr(self, 'client'):
            # OpenAI async client doesn't need explicit closing
            pass


# Factory function for easy instantiation
def create_openai_provider(config: Optional[Dict[str, Any]] = None) -> OpenAIProvider:
    """Create OpenAI provider instance"""
    if config is None:
        config = {}
    return OpenAIProvider(config)
