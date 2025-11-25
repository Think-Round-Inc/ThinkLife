"""
OpenAI Provider - Integration with OpenAI GPT models
"""

import logging
from typing import Dict, Any, List, Optional, Union

from ..specs import ProviderSpec
from .provider_registry import check_provider_spec_availability

logger = logging.getLogger(__name__)

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    logger.warning("OpenAI library not available. Install with: pip install openai")
    OPENAI_AVAILABLE = False
    AsyncOpenAI = None


class OpenAIProvider:
    """OpenAI provider for GPT models"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.name = "openai"
        self._initialized = False
        self.client = None
    
    async def initialize(self) -> bool:
        """Validate config and initialize OpenAI client"""
            if not OPENAI_AVAILABLE:
                logger.error("OpenAI library not available")
                return False
            
        # Validate with registry
        if not self._validate_config():
            return False
        
        # Initialize client
        try:
            self.client = AsyncOpenAI(
                api_key=self.config["api_key"],
                timeout=self.config.get("timeout", 30.0),
                max_retries=self.config.get("max_retries", 3),
                organization=self.config.get("organization"),
                base_url=self.config.get("base_url"),
            )
            self._initialized = True
            logger.info(f"OpenAI initialized: {self.config.get('model')}")
            return True
        except Exception as e:
            logger.error(f"OpenAI initialization failed: {e}")
            return False
    
    def _validate_config(self) -> bool:
        """Validate configuration using provider registry"""
        excluded_keys = {"api_key", "model", "temperature", "max_tokens", "stream", 
                        "enabled", "timeout", "max_retries", "organization", "base_url"}
        
        spec = ProviderSpec(
            provider_type="openai",
            model=self.config.get("model"),
            temperature=self.config.get("temperature"),
            max_tokens=self.config.get("max_tokens"),
            stream=self.config.get("stream", False),
            custom_params={k: v for k, v in self.config.items() if k not in excluded_keys}
        )
        
        is_valid, errors, _ = check_provider_spec_availability(spec)
        if not is_valid:
            logger.error(f"Validation failed: {'; '.join(errors)}")
            return False
        return True
    
    async def generate_response(
        self, 
        messages: List[Dict[str, str]], 
        **kwargs: Any) -> Dict[str, Any]:
        """Generate response from OpenAI"""
        if not self._initialized:
            raise RuntimeError("Provider not initialized")
        
        try:
            params = self._build_request_params(kwargs)
            params["messages"] = messages
            
            response = await self.client.chat.completions.create(**params)
            
            if not response.choices:
                return self._error_response("No response from OpenAI")
            
                choice = response.choices[0]
                metadata = {
                    "model": response.model,
                    "usage": response.usage.dict() if response.usage else {},
                    "finish_reason": choice.finish_reason,
                    "provider": "openai"
                }
                
            # Add tool/function calls if present
                if hasattr(choice.message, 'tool_calls') and choice.message.tool_calls:
                    metadata["tool_calls"] = [call.dict() for call in choice.message.tool_calls]
            elif hasattr(choice.message, 'function_call') and choice.message.function_call:
                metadata["function_call"] = choice.message.function_call
                
                return {
                "content": choice.message.content or "",
                    "metadata": metadata,
                    "success": True
                }
        except Exception as e:
            logger.error(f"OpenAI request failed: {e}")
            return self._error_response(str(e))
    
    def _build_request_params(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Build request parameters from config and kwargs"""
        params = {
            "model": kwargs.get("model", self.config.get("model")),
            "max_tokens": kwargs.get("max_tokens", self.config.get("max_tokens")),
            "temperature": kwargs.get("temperature", self.config.get("temperature")),
            "top_p": kwargs.get("top_p", self.config.get("top_p")),
            "frequency_penalty": kwargs.get("frequency_penalty", self.config.get("frequency_penalty")),
            "presence_penalty": kwargs.get("presence_penalty", self.config.get("presence_penalty")),
            "stream": kwargs.get("stream", self.config.get("stream", False)),
        }
        
        # Add optional params
        optional = ["stop", "user", "logit_bias", "functions", "function_call", 
                   "tools", "tool_choice", "response_format", "seed"]
        for key in optional:
            if key in kwargs:
                params[key] = kwargs[key]
            elif key in self.config:
                params[key] = self.config[key]
        
        # Remove None values
        return {k: v for k, v in params.items() if v is not None}
    
    def _error_response(self, error: str) -> Dict[str, Any]:
        """Create standardized error response"""
            return {
                "content": "",
            "metadata": {"error": error, "provider": "openai"},
                "success": False
            }
    
    async def generate_embeddings(
        self, 
        input_text: Union[str, List[str]], 
        model: str = "text-embedding-ada-002",
        **kwargs: Any) -> Dict[str, Any]:
        """Generate embeddings"""
        if not self._initialized:
            raise RuntimeError("Provider not initialized")
        
        try:
            response = await self.client.embeddings.create(
                model=model,
                input=input_text,
                **kwargs
            )
            return {
                "embeddings": [data.embedding for data in response.data],
                "usage": response.usage.dict() if response.usage else {},
                "model": response.model,
                "success": True
            }
        except Exception as e:
            logger.error(f"Embeddings failed: {e}")
            return {"embeddings": [], "error": str(e), "success": False}
    
    async def generate_image(
        self,
        prompt: str,
        model: str = "dall-e-3",
        size: str = "1024x1024",
        quality: str = "standard",
        **kwargs: Any) -> Dict[str, Any]:
        """Generate image with DALL-E"""
        if not self._initialized:
            raise RuntimeError("Provider not initialized")
        
        try:
            response = await self.client.images.generate(
                model=model,
                prompt=prompt,
                size=size,
                quality=quality,
                **kwargs
            )
            return {
                "images": [image.url for image in response.data],
                "metadata": {"model": response.model, "provider": "openai"},
                "success": True
            }
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            return {"images": [], "error": str(e), "success": False}
    
    async def health_check(self) -> Dict[str, Any]:
        """Check provider health"""
        try:
            import time
            start = time.time()
            response = await self.generate_response(
                [{"role": "user", "content": "Test"}], 
                max_tokens=1
            )
            return {
                "healthy": response.get("success", False),
                "response_time": time.time() - start,
                "provider": self.name,
                "model": self.config.get("model", "unknown")
            }
        except Exception as e:
            return {"healthy": False, "error": str(e), "provider": self.name}
    
    async def close(self) -> None:
        """Close provider"""
        self._initialized = False
        self.client = None
        logger.info("OpenAI provider closed")
    
    
def create_openai_provider(config: Optional[Dict[str, Any]] = None) -> OpenAIProvider:
    """Create OpenAI provider instance"""
    return OpenAIProvider(config or {})
