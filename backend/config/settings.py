"""
Centralized application settings.

Replaces the ~57 scattered `os.getenv()` call sites that previously lived across
`main.py`, `brain/cortex/*.py`, `brain/providers/*.py`, `brain/guardrails/*.py`,
`agents/zoe/*.py`, and `plugins/*.py` with a single, typed, validated source of
truth. Values are still sourced from environment variables / a `.env` file — this
module does not change *how* configuration is supplied, only how it is read.

Usage:
    from config import settings
    settings.openai_api_key
    settings.confidence_threshold

For testing, override values via environment variables or by constructing a
`Settings(...)` instance directly and passing it where needed (e.g.
`get_settings.cache_clear()` after monkeypatching `os.environ`).
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

try:
    # Pydantic v2 split BaseSettings into a separate package.
    from pydantic_settings import BaseSettings, SettingsConfigDict
    _PYDANTIC_V2 = True
except ImportError:  # pragma: no cover - fallback for pydantic v1 environments
    from pydantic import BaseSettings  # type: ignore[no-redef]
    SettingsConfigDict = None  # type: ignore[assignment, misc]
    _PYDANTIC_V2 = False

logger = logging.getLogger(__name__)

_BACKEND_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """
    Single source of truth for backend configuration.

    Field names intentionally mirror the environment variable names (in
    lower_snake_case) that were previously read ad hoc via `os.getenv()`, so the
    migration is a mechanical find-and-replace at call sites.
    """

    # --- LLM Providers -----------------------------------------------------
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o-mini"
    openai_max_tokens: int = 2000
    openai_temperature: float = 0.7

    # Which provider/model the built-in agents (Zoe, Echo) use. Switch providers
    # by changing LLM_PROVIDER_TYPE (+ the matching *_API_KEY); no code change needed.
    llm_provider_type: str = "openai"  # openai | anthropic | gemini | openrouter
    llm_provider_model: Optional[str] = None  # None -> OPENAI_MODEL for openai, else the registry default

    # Embeddings (knowledge-base search). Independent of LLM_PROVIDER_TYPE.
    # auto = openai if OPENAI_API_KEY is set, else local (no key, downloads MiniLM once).
    embedding_provider: str = "auto"  # auto | local | openai | openrouter | gemini
    embedding_model: Optional[str] = None  # None -> provider default

    anthropic_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    openrouter_api_key: Optional[str] = None
    tavily_api_key: Optional[str] = None

    # --- Zoe agent -----------------------------------------------------
    zoe_vector_db_path: Optional[str] = None  # falls back to a computed default below

    # --- Keycloak / Auth -----------------------------------------------------
    keycloak_url: str = "http://localhost:8080"
    keycloak_realm: str = "thinklife"
    keycloak_client_id: str = "thinklife-frontend"

    # --- CORS -----------------------------------------------------
    cors_allow_origins: List[str] = [
        "http://localhost:3000",
        "http://localhost:3001",
        "https://thinklife.vercel.app",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
    ]

    # --- Workflow engine tuning -----------------------------------------------------
    # These were previously magic numbers inline in brain/cortex/workflow_engine.py.
    confidence_threshold: float = 0.75
    max_validation_attempts: int = 5

    # --- App metadata -----------------------------------------------------
    app_version: str = "0.7.0"
    environment: str = "development"
    log_level: str = "INFO"

    # --- Middleware -----------------------------------------------------
    max_request_body_bytes: int = 1_000_000  # 1 MB

    # --- Concurrency -----------------------------------------------------
    # Bounds how many provider (LLM) calls can be in flight at once across
    # the whole process, so a burst of concurrent agent requests can't
    # overwhelm a provider's own rate limits. See brain/cortex/cortex.py.
    max_concurrent_provider_calls: int = 10

    # --- Resilience -----------------------------------------------------
    # Retry policy for transient provider call failures (network errors,
    # timeouts, or a provider adapter returning success=False) - distinct
    # from WorkflowEngine's confidence-based validation retry loop
    # (max_validation_attempts above), which retries on low response
    # *quality*, not transport failures. See brain/resilience/.
    provider_retry_max_attempts: int = 3
    provider_retry_base_delay_seconds: float = 0.5
    provider_retry_max_delay_seconds: float = 8.0
    # Tool retry uses a smaller budget than provider retry
    tool_retry_max_attempts: int = 2
    tool_retry_base_delay_seconds: float = 0.3
    tool_retry_max_delay_seconds: float = 2.0

    # --- Background tasks -----------------------------------------------------
    background_queue_max_size: int = 1000
    background_queue_num_workers: int = 2
    # Retry budget for a background job itself (e.g. an evaluation call that
    # transiently fails) - separate from provider/tool retry, and smaller
    # since background work is best-effort by definition.
    background_task_retry_max_attempts: int = 2
    background_task_retry_base_delay_seconds: float = 0.5
    # Circuit breaker: opens after this many consecutive failures for a
    # given provider, short-circuiting further calls until the recovery
    # timeout elapses.
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_timeout_seconds: float = 30.0

    if _PYDANTIC_V2:
        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            case_sensitive=False,
            extra="ignore",
        )
    else:  # pragma: no cover
        class Config:
            env_file = ".env"
            env_file_encoding = "utf-8"
            case_sensitive = False
            extra = "ignore"

    @property
    def zoe_vector_db_path_resolved(self) -> str:
        """Resolved path to Zoe's vector DB, matching the previous default in zoe_agent.py."""
        if self.zoe_vector_db_path:
            return self.zoe_vector_db_path
        return str(_BACKEND_DIR / "data" / "zoe.sqlite3")

    def openai_provider_config(self) -> dict:
        """Convenience accessor matching the shape previously built inline in main.py's lifespan."""
        return {
            "enabled": bool(self.openai_api_key),
            "api_key": self.openai_api_key,
            "model": self.openai_model,
            "max_tokens": self.openai_max_tokens,
            "temperature": self.openai_temperature,
        }

    def default_llm(self) -> tuple:
        """(provider, model) for agents, driven by LLM_PROVIDER_TYPE / LLM_PROVIDER_MODEL."""
        provider = self.llm_provider_type.lower()
        if self.llm_provider_model:
            return provider, self.llm_provider_model
        if provider == "openai":
            return provider, self.openai_model
        from brain.providers.provider_registry import PROVIDER_CONFIGS
        return provider, PROVIDER_CONFIGS.get(provider, {}).get("default_model", "")

    def provider_api_key(self, provider_type: str) -> Optional[str]:
        """
        Look up an API key by provider type, replacing the `key_map` dict that was
        duplicated in two places in workflow_engine.py.
        """
        mapping = {
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key,
            "gemini": self.gemini_api_key,
            "openrouter": self.openrouter_api_key,
        }
        return mapping.get(provider_type)

    def validate_required_for_startup(self) -> List[str]:
        """
        Fail-fast check used during app startup (see main.py's `lifespan`).
        Returns a list of human-readable problems; empty list means OK.
        """
        problems: List[str] = []
        provider, model = self.default_llm()
        from brain.providers.provider_registry import get_provider_registry
        ok, errors, _ = get_provider_registry().check_provider_and_model(provider, model)
        if not ok:
            problems.append(f"LLM_PROVIDER_TYPE/LLM_PROVIDER_MODEL invalid: {'; '.join(errors)}")
        elif not self.provider_api_key(provider):
            problems.append(f"LLM_PROVIDER_TYPE={provider} but its API key ({provider.upper()}_API_KEY) is not set.")
        if not any([self.openai_api_key, self.anthropic_api_key, self.gemini_api_key, self.openrouter_api_key]):
            problems.append(
                "No LLM provider API key configured (OPENAI_API_KEY / ANTHROPIC_API_KEY / "
                "GEMINI_API_KEY / OPENROUTER_API_KEY) - the Brain will not be able to call any provider."
            )
        return problems


@lru_cache
def get_settings() -> Settings:
    """
    Cached settings accessor. Using `lru_cache` means the environment is only read
    once per process; call `get_settings.cache_clear()` in tests if you need to
    re-read environment variables after monkeypatching `os.environ`.
    """
    return Settings()


# Module-level singleton for the common case of `from config import settings`.
settings = get_settings()
