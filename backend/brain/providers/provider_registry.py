"""
Provider Registry - Generic registry that auto-configures from provider list
To add a new provider: Just add it to the PROVIDERS list below
"""

import logging
import os
from typing import Dict, List, Any, Optional, Set, Tuple

from ..specs import ModelInfo, ProviderInfo, ProviderSpec

logger = logging.getLogger(__name__)



PROVIDERS = [
    {
        "name": "openai",
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"],
        "env_key": "OPENAI_API_KEY",
        "library": "openai",
        "default_model": "gpt-4o-mini",
        "default_max_tokens": 128000,
    },
    {
        "name": "anthropic",
        "models": ["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022", "claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"],
        "env_key": "ANTHROPIC_API_KEY",
        "library": "anthropic",
        "default_model": "claude-3-5-sonnet-20241022",
        "default_max_tokens": 200000,
    },
    {
        "name": "gemini",
        "models": ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro", "gemini-1.5-flash-8b", "gemini-1.5-pro-002"],
        "env_key": "GEMINI_API_KEY",
        "library": "google.generativeai",
        "default_model": "gemini-1.5-flash",
        "default_max_tokens": 1000000,
    },
]

# Common parameter validation rules (same for all providers)
COMMON_PARAM_RULES = {
    "temperature": {"min": 0.0, "max": 2.0, "type": "float", "default": 0.7},
    "top_p": {"min": 0.0, "max": 1.0, "type": "float", "default": 1.0},
    "top_k": {"min": 0, "max": 500, "type": "int", "default": 40},
    "max_tokens": {"min": 1, "max": 2000000, "type": "int", "default": 2000},
    "frequency_penalty": {"min": -2.0, "max": 2.0, "type": "float", "default": 0.0},
    "presence_penalty": {"min": -2.0, "max": 2.0, "type": "float", "default": 0.0},
    "stream": {"values": [True, False], "type": "bool", "default": False},
    "timeout": {"min": 1.0, "max": 300.0, "type": "float", "default": 30.0},
    "max_retries": {"min": 0, "max": 10, "type": "int", "default": 3},
}

# Model-specific token limits (if not specified, uses default_max_tokens from provider)
MODEL_TOKEN_LIMITS = {
    "openai": {
        "gpt-4o": 128000, "gpt-4o-mini": 128000, "gpt-4-turbo": 128000,
        "gpt-4": 8192, "gpt-3.5-turbo": 16385,
    },
    "anthropic": {
        "claude-3-opus-20240229": 200000, "claude-3-sonnet-20240229": 200000,
        "claude-3-haiku-20240307": 200000, "claude-3-5-sonnet-20241022": 200000,
        "claude-3-5-haiku-20241022": 200000,
    },
    "gemini": {
        "gemini-1.5-flash": 1000000, "gemini-1.5-pro": 2000000,
        "gemini-1.0-pro": 30000, "gemini-1.5-flash-8b": 1000000,
        "gemini-1.5-pro-002": 2000000,
    },
}


class ProviderRegistry:
    """Generic provider registry - auto-configures from PROVIDERS list"""
    
    def __init__(self):
        self._providers: Dict[str, ProviderInfo] = {}
        self._configs: Dict[str, Dict[str, Any]] = {}
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the provider registry by auto-registering all providers"""
        if self._initialized:
            return
        
        # Auto-register all providers from the PROVIDERS list
        for provider_config in PROVIDERS:
            self._register_provider(provider_config)
        
        self._initialized = True
        available = sum(1 for p in self._providers.values() if p.available)
        total_models = sum(len(p.models) for p in self._providers.values())
        logger.info(f"Provider Registry initialized: {available}/{len(self._providers)} providers, {total_models} models")
    
    def _register_provider(self, provider_config: Dict[str, Any]) -> None:
        """Auto-register a provider from minimal config"""
        provider_name = provider_config["name"]
        
        # Check if provider is available
        available = self._check_availability(provider_config)
        
        # Get model token limits
        token_limits = MODEL_TOKEN_LIMITS.get(provider_name, {})
        default_tokens = provider_config.get("default_max_tokens", 8192)
        
        # Create model info
        models = [
            ModelInfo(
                name=model,
                max_tokens=token_limits.get(model, default_tokens),
                description=f"{provider_name} model: {model}"
            )
            for model in provider_config["models"]
        ]
        
        # Build defaults
        defaults = {
            "enabled": True,
            "model": provider_config.get("default_model", provider_config["models"][0]),
            "api_key": os.getenv(provider_config["env_key"], ""),
        }
        for param, rules in COMMON_PARAM_RULES.items():
            defaults[param] = rules["default"]
        
        # Store config for validation
        self._configs[provider_name] = {
            "models": provider_config["models"],
            "env_key": provider_config["env_key"],
            "library": provider_config["library"],
            "token_limits": token_limits,
            "default_max_tokens": default_tokens,
        }
        
        # Register provider info
        self._providers[provider_name] = ProviderInfo(
            name=provider_name,
            available=available,
            models=models,
            required_fields=["api_key", "model"],
            optional_fields=list(COMMON_PARAM_RULES.keys()) + ["stop", "system", "tools", "tool_choice"],
            defaults=defaults,
            max_tokens_limit=max(token_limits.values()) if token_limits else default_tokens,
            min_tokens_limit=1,
            supports_streaming=True,
            supports_functions=True,
            supports_tools=True
        )
    
    def _check_availability(self, provider_config: Dict[str, Any]) -> bool:
        """Check if provider library is installed and API key is present"""
        library = provider_config.get("library")
        
        # Check library
        if library:
            try:
                __import__(library)
        except ImportError:
                return False
        
        # Check API key
        return bool(os.getenv(provider_config["env_key"]))
    
    def validate_arg_value(self, provider_name: str, arg_name: str, value: Any) -> Tuple[bool, str]:
        """Validate if a value is acceptable for a provider argument"""
        # Use common parameter rules
        if arg_name not in COMMON_PARAM_RULES:
            return True, "Custom argument"  # Allow custom fields
        
        rules = COMMON_PARAM_RULES[arg_name]
        arg_type = rules["type"]
        
        # Type checking
        type_checkers = {
            "float": lambda v: isinstance(v, (int, float)),
            "int": lambda v: isinstance(v, int),
            "bool": lambda v: isinstance(v, bool),
            "str": lambda v: isinstance(v, str)
        }
        
        if arg_type not in type_checkers or not type_checkers[arg_type](value):
            return False, f"{arg_name} must be {arg_type}"
        
        # Min/max validation for numeric types
        if arg_type in ("float", "int"):
            min_val, max_val = rules.get("min"), rules.get("max")
            if min_val is not None and value < min_val:
                return False, f"{arg_name} {value} below minimum {min_val}"
            if max_val is not None and value > max_val:
                return False, f"{arg_name} {value} exceeds maximum {max_val}"
        
        # Allowed values for bool/str
        if arg_type in ("bool", "str"):
            allowed = rules.get("values")
            if allowed and value not in allowed:
                return False, f"{arg_name} must be one of {allowed}"
        
        return True, "Valid"
    
    def check_provider_spec_availability(self, provider_spec: ProviderSpec) -> Tuple[bool, List[str], Optional[Dict[str, Any]]]:
        """Main function to check if a ProviderSpec is available and valid"""
        # Auto-register if not found
        if provider_spec.provider_type not in self._providers:
            provider_config = next((p for p in PROVIDERS if p["name"] == provider_spec.provider_type), None)
            if provider_config:
                self._register_provider(provider_config)
        
        errors = []
        provider_name = provider_spec.provider_type
        provider_config = self._configs.get(provider_name)
        
        if not provider_config:
            available_providers = [p["name"] for p in PROVIDERS]
            return False, [f"Provider '{provider_name}' not found. Available: {available_providers}"], None
        
        provider_info = self._providers.get(provider_name)
        
        # Check availability
        if not provider_info or not provider_info.available:
            env_key = provider_config.get("env_key", "N/A")
            errors.append(f"Provider '{provider_name}' not available. Check library and API key ({env_key})")
        
        # Validate model
        if provider_spec.model:
            if provider_spec.model not in provider_config["models"]:
                errors.append(f"Model '{provider_spec.model}' not supported. Available: {provider_config['models']}")
            elif provider_info:
                model_info = self.get_model_info(provider_name, provider_spec.model)
                if model_info and provider_spec.max_tokens and provider_spec.max_tokens > model_info.max_tokens:
                    errors.append(f"max_tokens {provider_spec.max_tokens} exceeds model limit {model_info.max_tokens}")
        
        # Validate parameters
        for param in ["temperature", "max_tokens", "stream", "top_p", "top_k"]:
            value = getattr(provider_spec, param, None)
            if value is not None:
                valid, error_msg = self.validate_arg_value(provider_name, param, value)
                if not valid:
                    errors.append(f"{param}: {error_msg}")
        
        # Check max_tokens limits
        if provider_spec.max_tokens and provider_info:
            if provider_spec.max_tokens > provider_info.max_tokens_limit:
                errors.append(f"max_tokens {provider_spec.max_tokens} exceeds limit {provider_info.max_tokens_limit}")
            if provider_spec.max_tokens < provider_info.min_tokens_limit:
                errors.append(f"max_tokens {provider_spec.max_tokens} below minimum {provider_info.min_tokens_limit}")
        
        # Validate custom params
        if provider_spec.custom_params:
            for param_name, param_value in provider_spec.custom_params.items():
                valid, error_msg = self.validate_arg_value(provider_name, param_name, param_value)
                if not valid:
                    errors.append(f"Custom parameter '{param_name}': {error_msg}")
        
        if errors:
            return False, errors, None
        
        # Build success info
        defaults = provider_info.defaults if provider_info else {}
        return True, [], {
            "provider": provider_name,
            "available": True,
            "model": provider_spec.model or defaults.get("model"),
            "available_models": provider_config["models"],
            "parameters": {
                "temperature": provider_spec.temperature or defaults.get("temperature"),
                "max_tokens": provider_spec.max_tokens or defaults.get("max_tokens"),
                "stream": provider_spec.stream if provider_spec.stream is not None else defaults.get("stream", False),
            },
            "limits": {
                "max_tokens": provider_info.max_tokens_limit if provider_info else provider_config["default_max_tokens"],
                "min_tokens": 1,
            }
        }
    
    # Simple getter methods
    def get_available_providers(self) -> Set[str]:
        return {name for name, info in self._providers.items() if info.available}
    
    def get_all_providers(self) -> Dict[str, ProviderInfo]:
        return self._providers.copy()
    
    def get_provider_info(self, provider_name: str) -> Optional[ProviderInfo]:
        return self._providers.get(provider_name)
    
    def is_provider_available(self, provider_name: str) -> bool:
        info = self._providers.get(provider_name)
        return info is not None and info.available
    
    def get_available_models(self, provider_name: str) -> List[ModelInfo]:
        info = self._providers.get(provider_name)
        return info.models.copy() if info and info.available else []
    
    def get_all_models(self, provider_name: str) -> List[ModelInfo]:
        info = self._providers.get(provider_name)
        return info.models.copy() if info else []
    
    def get_model_info(self, provider_name: str, model_name: str) -> Optional[ModelInfo]:
        info = self._providers.get(provider_name)
        if not info:
            return None
        return next((m for m in info.models if m.name == model_name), None)
    
    def validate_model_for_provider(self, provider_name: str, model_name: str) -> Tuple[bool, str]:
        """Validate if a model is supported by a provider"""
        info = self._providers.get(provider_name)
        if not info:
            return False, f"Provider '{provider_name}' not found"
        if not info.available:
            return False, f"Provider '{provider_name}' not available"
        
        model_info = self.get_model_info(provider_name, model_name)
        if not model_info:
            available = [m.name for m in info.models]
            return False, f"Model '{model_name}' not supported. Available: {available}"
        return True, "Model is valid"
    
    def get_provider_defaults(self, provider_name: str) -> Dict[str, Any]:
        info = self._providers.get(provider_name)
        return info.defaults.copy() if info else {}
    
    def get_suggested_model_for_use_case(self, provider_name: str, use_case: str) -> Optional[str]:
        """Get suggested model for use case"""
        config = self._configs.get(provider_name)
        if not config:
            return None
        
        models = config["models"]
        use_case_lower = use_case.lower()
        
        # Simple heuristics based on model names
        if "fast" in use_case_lower or "cheap" in use_case_lower:
            # Return smallest/fastest model
            for model in models:
                if any(word in model.lower() for word in ["mini", "flash", "haiku", "3.5"]):
                    return model
        elif "powerful" in use_case_lower or "best" in use_case_lower:
            # Return most powerful model
            for model in models:
                if any(word in model.lower() for word in ["opus", "pro", "gpt-4o"]):
                    return model
        
        # Default: return first model
        return models[0] if models else None


# Singleton
_provider_registry_instance: Optional[ProviderRegistry] = None


def get_provider_registry() -> ProviderRegistry:
    """Get singleton ProviderRegistry instance"""
    global _provider_registry_instance
    if _provider_registry_instance is None:
        _provider_registry_instance = ProviderRegistry()
    return _provider_registry_instance


def check_provider_spec_availability(provider_spec: ProviderSpec) -> Tuple[bool, List[str], Optional[Dict[str, Any]]]:
    """Convenience function to check if a ProviderSpec is available and valid"""
    return get_provider_registry().check_provider_spec_availability(provider_spec)
