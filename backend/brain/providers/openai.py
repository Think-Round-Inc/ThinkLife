"""
OpenAI Provider - Integration with OpenAI GPT models
"""

import logging
import os
from typing import Dict, Any, List, Optional, Union

logger = logging.getLogger(__name__)

# Optional LangFuse imports - gracefully handle compatibility issues
try:
    from langfuse.decorators import observe, langfuse_context
    LANGFUSE_AVAILABLE = True
except (ImportError, Exception) as e:
    logger.warning(f"LangFuse not available: {e}. Continuing without observability.")
    LANGFUSE_AVAILABLE = False
    
    # Create no-op decorators and context
    def observe(name=None, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    class LangfuseContext:
        def update_current_trace(self, **kwargs):
            pass
        def update_current_observation(self, **kwargs):
            pass
        def observe_llm_call(self, **kwargs):
            class NoOpContext:
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    pass
                def update(self, **kwargs):
                    pass
            return NoOpContext()
    
    langfuse_context = LangfuseContext()

from ..specs import ProviderSpec
from .provider_registry import get_provider_registry

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
            # Get API key from config or environment
            api_key = self.config.get("api_key") or os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.error("OpenAI API key not found in config or OPENAI_API_KEY environment variable")
                return False
            
            self.client = AsyncOpenAI(
                api_key=api_key,
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
        registry = get_provider_registry()
        provider_type = "openai"
        model = self.config.get("model")
        
        is_valid, errors, _ = registry.check_provider_and_model(provider_type, model)
        if not is_valid:
            logger.error(f"Validation failed: {'; '.join(errors)}")
            return False
        return True
    
    @observe(name="openai_generate_response")
    async def generate_response(
        self, 
        messages: List[Dict[str, str]], 
        **kwargs: Any) -> Dict[str, Any]:
        """Generate response from OpenAI with LangFuse tracing"""
        if not self._initialized:
            raise RuntimeError("Provider not initialized")
        
        try:
            params = self._build_request_params(kwargs)
            # Ensure messages and model are in params (required by OpenAI API)
            params["messages"] = messages
            model = params.get("model")
            if not model:
                model = self.config.get("model")
                if not model:
                    raise ValueError("Model not specified in config or kwargs")
                params["model"] = model
            
            # Track in LangFuse
            try:
                langfuse_context.update_current_trace(
                    name="openai_provider_call",
                    metadata={
                        "provider": "openai",
                        "model": model,
                        "message_count": len(messages),
                        "temperature": params.get("temperature"),
                        "max_tokens": params.get("max_tokens")
                    }
                )
            except Exception as lf_error:
                logger.warning(f"LangFuse trace update error (non-fatal): {lf_error}")
            
            # Make OpenAI API call (LangFuse observation is handled by @observe decorator on the method)
            response = await self.client.chat.completions.create(**params)
            
            if not response.choices:
                return self._error_response("No response from OpenAI")
            
            choice = response.choices[0]
            content = choice.message.content or ""
            
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
            
            # Update LangFuse with usage metrics
            if response.usage:
                langfuse_context.update_current_observation(
                    metadata={
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                        "finish_reason": choice.finish_reason
                    }
                )
            
            return {
                "content": content,
                "metadata": metadata,
                "success": True
            }
        except Exception as e:
            logger.error(f"OpenAI request failed: {e}")
            langfuse_context.update_current_observation(
                level="ERROR",
                status_message=str(e),
                metadata={"error": str(e)}
            )
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
