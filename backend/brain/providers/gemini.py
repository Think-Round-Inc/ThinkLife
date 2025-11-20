"""
Google Gemini Provider - Integration with Google's Gemini models
"""

import logging
from typing import Dict, Any, List, Optional

from ..types import ProviderSpec
from .provider_registry import check_provider_spec_availability

logger = logging.getLogger(__name__)

try:
    import google.generativeai as genai
    from google.generativeai import GenerativeModel
    GEMINI_AVAILABLE = True
except ImportError:
    logger.warning("Google Generative AI library not available. Install with: pip install google-generativeai")
    GEMINI_AVAILABLE = False
    genai = None
    GenerativeModel = None


class GeminiProvider:
    """Google Gemini provider"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.name = "gemini"
        self._initialized = False
        self.model = None
    
    async def initialize(self) -> bool:
        """Validate config and initialize Gemini client"""
        if not GEMINI_AVAILABLE:
            logger.error("Gemini library not available")
            return False
        
        # Validate with registry
        if not self._validate_config():
            return False
        
        # Initialize client
        try:
            genai.configure(api_key=self.config["api_key"])
            model_name = self.config.get("model", "gemini-1.5-flash")
            self.model = GenerativeModel(model_name)
            self._initialized = True
            logger.info(f"Gemini initialized: {model_name}")
            return True
        except Exception as e:
            logger.error(f"Gemini initialization failed: {e}")
            return False
    
    def _validate_config(self) -> bool:
        """Validate configuration using provider registry"""
        excluded_keys = {"api_key", "model", "temperature", "max_tokens", "stream",
                        "enabled", "timeout", "max_retries", "safety_settings", "generation_config"}
        
        spec = ProviderSpec(
            provider_type="gemini",
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
        """Generate response from Gemini"""
        if not self._initialized:
            raise RuntimeError("Provider not initialized")
        
        try:
            # Get last user message
            user_message = next(
                (msg["content"] for msg in reversed(messages) if msg.get("role") == "user"),
                None
            )
            
            if not user_message:
                return self._error_response("No user message found")
            
            # Build generation config
            gen_config = self._build_request_params(kwargs)
            
            # Generate response
            chat = self.model.start_chat(history=[])
            response = await chat.send_message_async(user_message, generation_config=gen_config)
            
            return {
                "content": response.text if hasattr(response, 'text') else "",
                "metadata": {
                    "model": self.config.get("model", "gemini-1.5-flash"),
                    "provider": "gemini"
                },
                "success": True
            }
        except Exception as e:
            logger.error(f"Gemini request failed: {e}")
            return self._error_response(str(e))
    
    def _build_request_params(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Build request parameters from config and kwargs"""
        gen_config = {
            "temperature": kwargs.get("temperature", self.config.get("temperature", 0.7)),
            "max_output_tokens": kwargs.get("max_tokens", self.config.get("max_tokens", 2000)),
        }
        
        # Add optional params
        for key in ["top_p", "top_k"]:
            if key in kwargs:
                gen_config[key] = kwargs[key]
            elif key in self.config:
                gen_config[key] = self.config[key]
        
        # Remove None values
        return {k: v for k, v in gen_config.items() if v is not None}
    
    def _error_response(self, error: str) -> Dict[str, Any]:
        """Create standardized error response"""
        return {
            "content": "",
            "metadata": {"error": error, "provider": "gemini"},
            "success": False
        }
    
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
        self.model = None
        logger.info("Gemini provider closed")


def create_gemini_provider(config: Optional[Dict[str, Any]] = None) -> GeminiProvider:
    """Create Gemini provider instance"""
    return GeminiProvider(config or {})
