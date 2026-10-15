"""
OpenRouter Provider - OpenAI-compatible gateway to many model vendors.

OpenRouter exposes an OpenAI-compatible REST API (https://openrouter.ai/api/v1),
so this just points the existing `AsyncOpenAI` client (already used by
OpenAIProvider) at OpenRouter's base_url with an OpenRouter API key - no new
SDK dependency needed.
"""

import logging
import os
from typing import Any, Dict, List, Optional

from .openai import OpenAIProvider
from .provider_registry import get_provider_registry

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterProvider(OpenAIProvider):
    """OpenRouter provider - reuses OpenAIProvider's OpenAI-SDK-based client."""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.name = "openrouter"

    async def initialize(self) -> bool:
        self.config.setdefault("base_url", OPENROUTER_BASE_URL)
        if not self.config.get("api_key"):
            self.config["api_key"] = os.getenv("OPENROUTER_API_KEY")
        if not self.config["api_key"]:
            # Fail here rather than delegating to super(): OpenAIProvider's own
            # initialize() falls back to OPENAI_API_KEY when its resolved key is
            # falsy, which would silently use the wrong key against OpenRouter.
            logger.error("OpenRouter API key missing")
            return False
        return await super().initialize()

    def _validate_config(self) -> bool:
        registry = get_provider_registry()
        is_valid, errors, _ = registry.check_provider_and_model("openrouter", self.config.get("model"))
        if not is_valid:
            logger.error(f"Validation failed: {'; '.join(errors)}")
            return False
        return True

    async def generate_response(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, Any]:
        result = await super().generate_response(messages, **kwargs)
        if result.get("metadata", {}).get("provider") == "openai":
            result["metadata"]["provider"] = "openrouter"
        return result

    def _error_response(self, error: str) -> Dict[str, Any]:
        return {"content": "", "metadata": {"error": error, "provider": "openrouter"}, "success": False}


def create_openrouter_provider(config: Optional[Dict[str, Any]] = None) -> OpenRouterProvider:
    return OpenRouterProvider(config or {})
