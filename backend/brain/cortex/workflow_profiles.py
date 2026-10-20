"""
Workflow profiles (US 1.1)

One LangGraph workflow serves every agent; a WorkflowProfile tunes it per
agent. Resolution order, later wins:

    1. BUILTIN_PROFILES[agent_type]  (or "default")
    2. the agent's own ProcessingSpec (max_iterations, timeout_seconds,
       enable_safety_checks) - what the agent asked for
    3. environment overrides, for local runs and the free OpenRouter tier:
         WORKFLOW_LLM_VALIDATION=false          turn off the LLM judge
         <AGENT>_FALLBACK_PROVIDERS=type:model,...   e.g. ZOE_FALLBACK_PROVIDERS
         WORKFLOW_FALLBACK_PROVIDERS=type:model,...  for every agent

Agents named in the user story that are not in the codebase yet (Bard; the
FAQ bot, which runs as its own app in faq-bot-rag/) get a profile as soon as
they become Brain plugins - add an entry to BUILTIN_PROFILES.
"""

import copy
import logging
import os
from typing import Any, Dict, List, Optional

from ..specs import WorkflowProfile, ProviderSpec

logger = logging.getLogger(__name__)

DEFAULT_AGENT_TYPE = "default"

BUILTIN_PROFILES: Dict[str, WorkflowProfile] = {
    DEFAULT_AGENT_TYPE: WorkflowProfile(agent_type=DEFAULT_AGENT_TYPE),
    # Zoe: empathetic companion. Keeps the LLM judge (existing behaviour) and
    # the safety checks; its ProcessingSpec sets max_iterations=2, timeout 30s.
    "zoe": WorkflowProfile(
        agent_type="zoe",
        max_iterations=2,
        run_timeout_seconds=30.0,
        llm_validation=True,
        safety_checks=True,
    ),
}

_FALSE = {"0", "false", "no", "off"}


def parse_provider_list(raw: Optional[str], template: Optional[Any] = None) -> List[ProviderSpec]:
    """
    Parse "openrouter:openrouter/free,openai:gpt-4o-mini" into ProviderSpecs.
    Temperature, max_tokens and custom params are copied from `template`
    (the agent's primary provider) so a fallback answers in the same style.
    """
    specs: List[ProviderSpec] = []
    for item in (raw or "").split(","):
        item = item.strip()
        if not item:
            continue
        provider_type, _, model = item.partition(":")
        if not provider_type or not model:
            logger.warning("Ignoring malformed fallback provider %r (expected type:model)", item)
            continue
        specs.append(ProviderSpec(
            provider_type=provider_type.strip(),
            model=model.strip(),
            temperature=getattr(template, "temperature", 0.7),
            max_tokens=getattr(template, "max_tokens", 2000),
            custom_params=dict(getattr(template, "custom_params", {}) or {}),
        ))
    return specs


def _default_fallbacks(primary: Optional[Any]) -> List[ProviderSpec]:
    """
    With no fallback configured: if an OpenRouter key is present and the
    primary is not already the free router, fall back to openrouter/free.
    Costs nothing and covers the "OpenAI key missing or rate-limited" case.
    """
    if not os.getenv("OPENROUTER_API_KEY") or primary is None:
        return []
    if getattr(primary, "provider_type", None) == "openrouter" and getattr(primary, "model", None) == "openrouter/free":
        return []
    return parse_provider_list("openrouter:openrouter/free", template=primary)


def resolve_workflow_profile(specs: Any) -> WorkflowProfile:
    """Build the effective profile for one request from the agent's specs."""
    agent_meta = getattr(specs, "agent_metadata", None) or {}
    agent_type = str(agent_meta.get("agent_type") or DEFAULT_AGENT_TYPE).lower()

    base = BUILTIN_PROFILES.get(agent_type) or BUILTIN_PROFILES[DEFAULT_AGENT_TYPE]
    profile = copy.deepcopy(base)
    profile.agent_type = agent_type

    # 2. What the agent asked for in its ProcessingSpec
    processing = getattr(specs, "processing", None)
    if processing is not None:
        if getattr(processing, "max_iterations", None):
            profile.max_iterations = max(1, int(processing.max_iterations))
        if getattr(processing, "timeout_seconds", None):
            profile.run_timeout_seconds = float(processing.timeout_seconds)
        if getattr(processing, "enable_safety_checks", None) is not None:
            profile.safety_checks = bool(processing.enable_safety_checks)

    # 3. Environment overrides
    if os.getenv("WORKFLOW_LLM_VALIDATION", "").strip().lower() in _FALSE:
        profile.llm_validation = False

    primary = getattr(specs, "provider", None)
    raw = os.getenv(f"{agent_type.upper()}_FALLBACK_PROVIDERS") or os.getenv("WORKFLOW_FALLBACK_PROVIDERS")
    if raw is not None:
        profile.fallback_providers = parse_provider_list(raw, template=primary)
    elif not profile.fallback_providers:
        profile.fallback_providers = _default_fallbacks(primary)

    # Never "fall back" to the exact provider that just failed
    if primary is not None:
        profile.fallback_providers = [
            p for p in profile.fallback_providers
            if (p.provider_type, p.model) != (getattr(primary, "provider_type", None), getattr(primary, "model", None))
        ]
    return profile
