"""
Structural typing contract for LLM provider adapters.
"""

from __future__ import annotations

from typing import Any, Dict, List, Protocol, runtime_checkable


@runtime_checkable
class LLMProviderProtocol(Protocol):
    """
    The minimal async interface every LLM provider adapter must implement to
    be usable by `WorkflowEngine` and `Cortex`.
    """

    name: str

    async def initialize(self) -> bool:
        """Set up any client/connection needed before use. Returns success."""
        ...

    async def generate_response(
        self, messages: List[Dict[str, str]], **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Generate a completion for the given chat messages.
        Expected to return a dict containing at least `success` (bool) and
        `content` (str), matching how `WorkflowEngine` reads the result today.
        """
        ...

    async def health_check(self) -> Dict[str, Any]:
        """Report provider health, expected to include a `healthy` (bool) key."""
        ...

    async def close(self) -> None:
        """Release any held resources (HTTP clients, sockets, etc.)."""
        ...
