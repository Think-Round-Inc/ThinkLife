"""
Provider Registry - Central registry of available providers, models, and configurations
Used by the specification validator to validate provider specifications
"""

import logging
import os
from typing import Dict, List, Any, Optional, Set, Tuple

from ..types import ModelInfo, ProviderInfo

logger = logging.getLogger(__name__)


class ProviderRegistry:
    """
    Central registry for all LLM providers
    
    Provides information about:
    - Available providers and their status
    - Supported models per provider
    - Configuration options and limits
    - Default settings
    """
    
    def __init__(self):
        self._providers: Dict[str, ProviderInfo] = {}
        self._initialized = False
        
        logger.info("Provider Registry initialized")
    
    async def initialize(self) -> None:
        """Initialize the provider registry"""
        if self._initialized:
            return
        
        # Register all providers
        self._register_openai_provider()
        self._register_anthropic_provider()
        self._register_gemini_provider()
        self._register_grok_provider()
        
        self._initialized = True
        
        available_count = sum(1 for p in self._providers.values() if p.available)
        total_models = sum(len(p.models) for p in self._providers.values())
        
        logger.info(
            f"Provider Registry initialized: {available_count}/{len(self._providers)} providers available, "
            f"{total_models} total models"
        )
    
    def _register_openai_provider(self) -> None:
        """Register OpenAI provider"""
        try:
            # Check if OpenAI library is available
            import openai
            available = bool(os.getenv("OPENAI_API_KEY"))
        except ImportError:
            available = False
        
        # Static model definitions - registry contains this information
        models = [
            ModelInfo("gpt-4o", 128000, description="Most capable GPT-4 model"),
            ModelInfo("gpt-4o-mini", 128000, description="Faster and cheaper than GPT-4o"),
            ModelInfo("gpt-4-turbo", 128000, description="Latest GPT-4 Turbo model"),
            ModelInfo("gpt-4", 8192, description="GPT-4 model"),
            ModelInfo("gpt-3.5-turbo", 16385, description="Fast and efficient model"),
            ModelInfo("gpt-3.5-turbo-16k", 16385, description="Extended context GPT-3.5"),
            ModelInfo("text-embedding-ada-002", 8191, description="Ada embedding model"),
            ModelInfo("text-embedding-3-small", 8191, description="Small embedding model"),
            ModelInfo("text-embedding-3-large", 8191, description="Large embedding model"),
            ModelInfo("dall-e-3", 128000, description="Latest DALL-E image generation"),
            ModelInfo("dall-e-2", 128000, description="DALL-E 2 image generation"),
        ]
        
        # Static configuration info - registry defines these
        required_fields = ["api_key", "model"]
        optional_fields = [
            "max_tokens", "temperature", "top_p", "frequency_penalty",
            "presence_penalty", "stop", "stream", "organization",
            "base_url", "timeout", "max_retries", "user", "logit_bias",
            "functions", "function_call", "tools", "tool_choice",
            "response_format", "seed", "extra_headers", "extra_query", "extra_body"
        ]
        
        defaults = {
            "enabled": True,
            "api_key": os.getenv("OPENAI_API_KEY", ""),
            "model": "gpt-4o-mini",
            "max_tokens": 2000,
            "temperature": 0.7,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "stream": False,
            "timeout": 30.0,
            "max_retries": 3,
            "extra_headers": {},
            "extra_query": {},
            "extra_body": {}
        }
        
        self._providers["openai"] = ProviderInfo(
            name="openai",
            available=available,
            models=models,
            required_fields=required_fields,
            optional_fields=optional_fields,
            defaults=defaults,
            max_tokens_limit=128000,
            min_tokens_limit=1,
            supports_streaming=True,
            supports_functions=True,
            supports_tools=True
        )
    
    def _register_anthropic_provider(self) -> None:
        """Register Anthropic provider"""
        try:
            # Check if Anthropic library is available
            import anthropic
            available = bool(os.getenv("ANTHROPIC_API_KEY"))
        except ImportError:
            available = False
        
        # Static model definitions - registry contains this information
        models = [
            ModelInfo("claude-3-opus-20240229", 200000, description="Most powerful Claude model"),
            ModelInfo("claude-3-sonnet-20240229", 200000, description="Balanced performance and speed"),
            ModelInfo("claude-3-haiku-20240307", 200000, description="Fastest Claude model"),
            ModelInfo("claude-3-5-sonnet-20241022", 200000, description="Latest Sonnet model"),
            ModelInfo("claude-3-5-haiku-20241022", 200000, description="Latest Haiku model"),
        ]
        
        # Static configuration info - registry defines these
        required_fields = ["api_key", "model"]
        optional_fields = [
            "max_tokens", "temperature", "top_p", "top_k",
            "stop_sequences", "system", "stream", "metadata",
            "stop_reason", "timeout", "max_retries", "tools",
            "tool_choice", "extra_headers", "extra_query", "extra_body"
        ]
        
        defaults = {
            "enabled": True,
            "api_key": os.getenv("ANTHROPIC_API_KEY", ""),
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 2000,
            "temperature": 0.7,
            "top_p": 1.0,
            "stream": False,
            "timeout": 30.0,
            "max_retries": 3,
            "extra_headers": {},
            "extra_query": {},
            "extra_body": {}
        }
        
        self._providers["anthropic"] = ProviderInfo(
            name="anthropic",
            available=available,
            models=models,
            required_fields=required_fields,
            optional_fields=optional_fields,
            defaults=defaults,
            max_tokens_limit=200000,
            min_tokens_limit=1,
            supports_streaming=True,
            supports_functions=False,
            supports_tools=True
        )
    
    def _register_gemini_provider(self) -> None:
        """Register Gemini provider"""
        try:
            # Check if Gemini library is available
            import google.generativeai
            available = bool(os.getenv("GEMINI_API_KEY"))
        except ImportError:
            available = False
        
        # Static model definitions - registry contains this information
        models = [
            ModelInfo("gemini-1.5-flash", 1000000, description="Fast and efficient model"),
            ModelInfo("gemini-1.5-pro", 2000000, description="Most capable Gemini model"),
            ModelInfo("gemini-1.0-pro", 30000, description="Previous generation Pro model"),
            ModelInfo("gemini-1.5-flash-8b", 1000000, description="Lightweight 8B parameter model"),
            ModelInfo("gemini-1.5-pro-002", 2000000, description="Latest Pro model variant"),
        ]
        
        # Static configuration info - registry defines these
        required_fields = ["api_key", "model"]
        optional_fields = [
            "max_tokens", "temperature", "top_p", "top_k", "candidate_count",
            "stop_sequences", "safety_settings", "generation_config",
            "stream", "timeout", "max_retries"
        ]
        
        defaults = {
            "enabled": True,
            "api_key": os.getenv("GEMINI_API_KEY", ""),
            "model": "gemini-1.5-flash",
            "max_tokens": 2000,
            "temperature": 0.7,
            "top_p": 1.0,
            "stream": False,
            "timeout": 30.0,
            "max_retries": 3
        }
        
        self._providers["gemini"] = ProviderInfo(
            name="gemini",
            available=available,
            models=models,
            required_fields=required_fields,
            optional_fields=optional_fields,
            defaults=defaults,
            max_tokens_limit=2000000,
            min_tokens_limit=1,
            supports_streaming=True,
            supports_functions=False,
            supports_tools=False
        )
    
    def _register_grok_provider(self) -> None:
        """Register Grok provider"""
        try:
            # Check if Grok library is available (assuming aiohttp for API calls)
            import aiohttp
            available = bool(os.getenv("GROK_API_KEY"))
        except ImportError:
            available = False
        
        # Static model definitions - registry contains this information
        models = [
            ModelInfo("grok-beta", 32768, description="Beta version of Grok"),
            ModelInfo("grok-2", 128000, description="Latest Grok model"),
            ModelInfo("grok-vision-beta", 32768, description="Multimodal Grok with vision"),
        ]
        
        # Static configuration info - registry defines these
        required_fields = ["api_key", "model"]
        optional_fields = [
            "max_tokens", "temperature", "top_p", "stream", "base_url",
            "timeout", "max_retries", "user", "stop", "presence_penalty",
            "frequency_penalty", "logit_bias", "seed", "response_format",
            "tools", "tool_choice", "extra_headers", "extra_query", "extra_body"
        ]
        
        defaults = {
            "enabled": True,
            "api_key": os.getenv("GROK_API_KEY", ""),
            "model": "grok-2",
            "max_tokens": 2000,
            "temperature": 0.7,
            "top_p": 1.0,
            "stream": False,
            "timeout": 30.0,
            "max_retries": 3,
            "base_url": "https://api.x.ai/v1",
            "extra_headers": {},
            "extra_query": {},
            "extra_body": {}
        }
        
        self._providers["grok"] = ProviderInfo(
            name="grok",
            available=available,
            models=models,
            required_fields=required_fields,
            optional_fields=optional_fields,
            defaults=defaults,
            max_tokens_limit=128000,
            min_tokens_limit=1,
            supports_streaming=True,
            supports_functions=False,
            supports_tools=True
        )
    
    def get_available_providers(self) -> Set[str]:
        """Get set of available provider names"""
        return {name for name, info in self._providers.items() if info.available}
    
    def get_all_providers(self) -> Dict[str, ProviderInfo]:
        """Get all providers (available and unavailable)"""
        return self._providers.copy()
    
    def get_provider_info(self, provider_name: str) -> Optional[ProviderInfo]:
        """Get information about a specific provider"""
        return self._providers.get(provider_name)
    
    def is_provider_available(self, provider_name: str) -> bool:
        """Check if a provider is available"""
        provider_info = self._providers.get(provider_name)
        return provider_info is not None and provider_info.available
    
    def get_available_models(self, provider_name: str) -> List[ModelInfo]:
        """Get list of available models for a provider (only if provider is available)"""
        provider_info = self._providers.get(provider_name)
        if not provider_info or not provider_info.available:
            return []
        return provider_info.models.copy()
    
    def get_all_models(self, provider_name: str) -> List[ModelInfo]:
        """Get all models for a provider regardless of availability"""
        provider_info = self._providers.get(provider_name)
        if not provider_info:
            return []
        return provider_info.models.copy()
    
    def get_model_info(self, provider_name: str, model_name: str) -> Optional[ModelInfo]:
        """Get information about a specific model"""
        provider_info = self._providers.get(provider_name)
        if not provider_info:
            return None
        
        for model in provider_info.models:
            if model.name == model_name:
                return model
        
        return None
    
    def validate_model_for_provider(self, provider_name: str, model_name: str) -> Tuple[bool, str]:
        """Validate if a model is supported by a provider"""
        provider_info = self._providers.get(provider_name)
        if not provider_info:
            return False, f"Provider '{provider_name}' not found"
        
        if not provider_info.available:
            return False, f"Provider '{provider_name}' is not available (missing API key or dependencies)"
        
        model_info = self.get_model_info(provider_name, model_name)
        if not model_info:
            available_models = [m.name for m in provider_info.models]
            return False, f"Model '{model_name}' not supported by {provider_name}. Available: {available_models}"
        
        return True, "Model is valid"
    
    def validate_config_for_provider(self, provider_name: str, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate configuration for a specific provider"""
        provider_info = self._providers.get(provider_name)
        if not provider_info:
            return False, [f"Provider '{provider_name}' not found"]
        
        if not provider_info.available:
            return False, [f"Provider '{provider_name}' is not available"]
        
        errors = []
        
        # Check required fields
        for field in provider_info.required_fields:
            if field not in config or not config[field]:
                errors.append(f"Missing required field: {field}")
        
        # Check max_tokens limits
        if "max_tokens" in config:
            max_tokens = config["max_tokens"]
            if max_tokens > provider_info.max_tokens_limit:
                errors.append(
                    f"max_tokens {max_tokens} exceeds provider limit {provider_info.max_tokens_limit}"
                )
            if max_tokens < provider_info.min_tokens_limit:
                errors.append(
                    f"max_tokens {max_tokens} below provider minimum {provider_info.min_tokens_limit}"
                )
        
        # Check temperature range (most providers use 0.0-2.0)
        if "temperature" in config:
            temp = config["temperature"]
            if temp < 0.0 or temp > 2.0:
                errors.append(f"Temperature {temp} is outside valid range (0.0-2.0)")
        
        # Validate model if specified
        if "model" in config:
            valid, model_error = self.validate_model_for_provider(provider_name, config["model"])
            if not valid:
                errors.append(f"Model validation failed: {model_error}")
        
        return len(errors) == 0, errors
    
    def get_provider_defaults(self, provider_name: str) -> Dict[str, Any]:
        """Get default configuration for a provider"""
        provider_info = self._providers.get(provider_name)
        if not provider_info:
            return {}
        return provider_info.defaults.copy()
    
    def get_provider_limits(self, provider_name: str) -> Dict[str, Any]:
        """Get limits for a provider"""
        provider_info = self._providers.get(provider_name)
        if not provider_info:
            return {}
        
        return {
            "max_tokens": provider_info.max_tokens_limit,
            "min_tokens": provider_info.min_tokens_limit,
            "supports_streaming": provider_info.supports_streaming,
            "supports_functions": provider_info.supports_functions,
            "supports_tools": provider_info.supports_tools,
        }
    
    def get_suggested_model_for_use_case(self, provider_name: str, use_case: str) -> Optional[str]:
        """Get suggested model for a specific use case"""
        provider_info = self._providers.get(provider_name)
        if not provider_info or not provider_info.available:
            return None
        
        # Simple heuristics for model selection
        use_case_lower = use_case.lower()
        
        if "fast" in use_case_lower or "quick" in use_case_lower:
            # Return the fastest model available
            if provider_name == "openai":
                return "gpt-3.5-turbo"
            elif provider_name == "anthropic":
                return "claude-3-haiku-20240307"
            elif provider_name == "gemini":
                return "gemini-1.5-flash"
            elif provider_name == "grok":
                return "grok-beta"
        
        elif "powerful" in use_case_lower or "best" in use_case_lower:
            # Return the most capable model
            if provider_name == "openai":
                return "gpt-4o"
            elif provider_name == "anthropic":
                return "claude-3-opus-20240229"
            elif provider_name == "gemini":
                return "gemini-1.5-pro"
            elif provider_name == "grok":
                return "grok-2"
        
        elif "cheap" in use_case_lower or "cost" in use_case_lower:
            # Return the most cost-effective model
            if provider_name == "openai":
                return "gpt-4o-mini"
            elif provider_name == "anthropic":
                return "claude-3-haiku-20240307"
            elif provider_name == "gemini":
                return "gemini-1.5-flash"
            elif provider_name == "grok":
                return "grok-beta"
        
        # Default to first available model
        if provider_info.models:
            return provider_info.models[0].name
        
        return None


# Singleton instance
_provider_registry_instance: Optional[ProviderRegistry] = None


def get_provider_registry() -> ProviderRegistry:
    """Get singleton ProviderRegistry instance"""
    global _provider_registry_instance
    if _provider_registry_instance is None:
        _provider_registry_instance = ProviderRegistry()
    return _provider_registry_instance
