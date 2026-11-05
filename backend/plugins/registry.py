"""
Plugin Registry - agent discovery and registration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from brain.specs import AgentConfig, AgentMetadata, IAgent

logger = logging.getLogger(__name__)

AgentFactory = Callable[[AgentConfig], IAgent]


@dataclass
class RegisteredPlugin:
    plugin_id: str
    factory: AgentFactory
    metadata: AgentMetadata


class PluginRegistry:
    """Central registry mapping a plugin id to an agent factory + metadata."""

    def __init__(self) -> None:
        self._plugins: Dict[str, RegisteredPlugin] = {}

    def register(self, plugin_id: str, factory: AgentFactory, metadata: AgentMetadata) -> None:
        if plugin_id in self._plugins:
            logger.warning(f"Plugin '{plugin_id}' already registered; overwriting")
        self._plugins[plugin_id] = RegisteredPlugin(plugin_id=plugin_id, factory=factory, metadata=metadata)
        logger.info(f"Registered plugin: {plugin_id}")

    def create(self, plugin_id: str, config: AgentConfig) -> IAgent:
        """Instantiate a fresh agent for `plugin_id` via its registered factory."""
        if plugin_id not in self._plugins:
            raise KeyError(f"No plugin registered as '{plugin_id}'. Available: {self.list_plugins()}")
        return self._plugins[plugin_id].factory(config)

    def get_metadata(self, plugin_id: str) -> AgentMetadata:
        if plugin_id not in self._plugins:
            raise KeyError(f"No plugin registered as '{plugin_id}'. Available: {self.list_plugins()}")
        return self._plugins[plugin_id].metadata

    def list_plugins(self) -> List[str]:
        return list(self._plugins.keys())

    def all_metadata(self) -> Dict[str, AgentMetadata]:
        return {pid: p.metadata for pid, p in self._plugins.items()}

    def is_registered(self, plugin_id: str) -> bool:
        return plugin_id in self._plugins


_plugin_registry_instance: Optional[PluginRegistry] = None


def get_plugin_registry() -> PluginRegistry:
    """Get the singleton PluginRegistry instance."""
    global _plugin_registry_instance
    if _plugin_registry_instance is None:
        _plugin_registry_instance = PluginRegistry()
    return _plugin_registry_instance


def register_builtin_plugins(registry: Optional[PluginRegistry] = None) -> PluginRegistry:
    """
    Register every built-in agent plugin shipped with the backend.

    Imports are done lazily
    """
    reg = registry or get_plugin_registry()

    # --- Zoe (production agent) ---
    if not reg.is_registered("zoe"):
        from plugins.zoe_agent import create_zoe_agent

        zoe_config = AgentConfig(agent_id="zoe")
        zoe_instance = create_zoe_agent(zoe_config)
        reg.register("zoe", create_zoe_agent, zoe_instance.metadata)

    # --- Echo (reference / contract-test agent, see plugins/echo_agent.py) ---
    if not reg.is_registered("echo"):
        from plugins.echo_agent import create_echo_agent

        echo_config = AgentConfig(agent_id="echo")
        echo_instance = create_echo_agent(echo_config)
        reg.register("echo", create_echo_agent, echo_instance.metadata)

    return reg
