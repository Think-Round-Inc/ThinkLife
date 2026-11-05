"""
Async context managers for provider and tool resource lifecycle.

Epic 5 / Story 5.2, Task 6 ("Introduce context managers for tool
execution"). Two problems this fixes:

1. `WorkflowEngine._call_provider_node` previously hand-rolled its own
   `try: ... finally: await provider.close()` block around each provider
   call. That block had a gap: if `provider.initialize()` returned False
   (rather than raising), the code raised *before* entering the `finally`
   that closes the provider - so a provider that had already allocated
   resources in `__init__`/partial `initialize()` never got `close()`
   called. `managed_provider()` below closes the provider in every case.

2. `WorkflowEngine._execute_tools_node` never actually instantiated or ran a
   tool at all - it was a placeholder that just checked availability via
   `ToolRegistry` and returned a canned "Tool execution placeholder" result
   (verified by reading the pre-change code directly). `managed_tool()` /
   `run_tool_with_timeout()` give it a safe way to do the real thing:
   guaranteed `initialize()`/`close()` pairing plus a timeout, so a slow or
   hanging tool call can't block a request indefinitely or leak a client.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, Optional

if TYPE_CHECKING:
    from .providers.protocol import LLMProviderProtocol
    from .tools.base_tool import BaseTool, ToolResult
    from .specs import IDataSource

logger = logging.getLogger(__name__)

DEFAULT_TOOL_TIMEOUT_SECONDS = 15.0


@asynccontextmanager
async def managed_provider(
    provider_type: str, config: Optional[Dict[str, Any]] = None
) -> AsyncIterator["LLMProviderProtocol"]:
    """
    Create, initialize, and guarantee cleanup of an LLM provider client.

    Usage:
        async with managed_provider("openai", config) as provider:
            response = await provider.generate_response(messages=...)
        # provider.close() has already run here, even if the block raised.
    """
    from .providers import create_provider

    provider = create_provider(provider_type, config or {})
    try:
        ok = await provider.initialize()
        if not ok:
            raise RuntimeError(f"Failed to initialize provider '{provider_type}'")
        yield provider
    finally:
        try:
            await provider.close()
        except Exception:
            logger.exception(f"Error closing provider '{provider_type}'")


@asynccontextmanager
async def managed_tool(tool: "BaseTool") -> AsyncIterator["BaseTool"]:
    """
    Guarantee a tool's `initialize()` / `close()` pairing around its use,
    regardless of what happens in between - including a timeout raised by
    `run_tool_with_timeout()` below, or any other exception.
    """
    tool_name = getattr(tool, "name", tool.__class__.__name__)
    try:
        await tool.initialize()
        yield tool
    finally:
        try:
            await tool.close()
        except Exception:
            logger.exception(f"Error closing tool '{tool_name}'")


@asynccontextmanager
async def managed_data_source(
    data_source: "IDataSource", init_config: Optional[Dict[str, Any]] = None
) -> AsyncIterator["IDataSource"]:
    """
    Guarantee an `IDataSource`'s `initialize()` / `close()` pairing.

    Added during the audit that followed Task 6's original scope
    (`_execute_tools_node` / providers only): `WorkflowEngine.
    _query_data_sources_node` had the exact same class of bug fixed
    elsewhere in this module - it called `vector_source.close()` only after
    a *successful* `query()`, so a data source whose `query()` raised (or
    whose `initialize()` succeeded but something after it failed) never got
    closed. This closes the same gap for data sources.
    """
    if not await data_source.initialize(init_config or {}):
        # Nothing to yield into a failed-to-initialize source; still attempt
        # close() in case initialize() partially allocated something before
        # returning False.
        try:
            await data_source.close()
        except Exception:
            logger.exception("Error closing data source that failed to initialize")
        raise RuntimeError("Failed to initialize data source")

    try:
        yield data_source
    finally:
        try:
            await data_source.close()
        except Exception:
            logger.exception("Error closing data source")


async def run_tool_with_timeout(
    tool: "BaseTool",
    timeout_seconds: float = DEFAULT_TOOL_TIMEOUT_SECONDS,
    **kwargs: Any,
) -> "ToolResult":
    """
    Run `tool.execute(**kwargs)` inside `managed_tool`, bounded by a timeout.

    On timeout, `asyncio.wait_for` cancels the underlying task, and
    `managed_tool`'s `finally` still runs `tool.close()` before the
    `asyncio.TimeoutError` propagates to the caller - the tool's resources
    are released even though its work didn't finish.
    """
    async with managed_tool(tool) as t:
        return await asyncio.wait_for(t.execute(**kwargs), timeout=timeout_seconds)
