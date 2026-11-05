"""
ThinkLife Agent Plugins

Plugin directory for agent implementations that connect to Brain.
Each agent in agents/ folder should have a corresponding plugin here.

Plugins specify:
- Execution specifications (data sources, providers, tools, processing)
- Request processing logic
- Brain integration

"""

from .registry import PluginRegistry, get_plugin_registry, register_builtin_plugins

__all__ = ["PluginRegistry", "get_plugin_registry", "register_builtin_plugins"]

