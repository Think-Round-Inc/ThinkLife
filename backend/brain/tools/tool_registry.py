"""
Tool Registry - Checks if tool files are available
"""

import logging
from pathlib import Path
from typing import Iterable

from ..registry_base import FileBackedRegistry

logger = logging.getLogger(__name__)


# Available tools list
TOOLS = ["tavily_search"]


class ToolRegistry(FileBackedRegistry[str]):
    """
    Registry for tools

    Checks if tool files exist. Shares its lookup logic with
    `ProviderRegistry` and `DataSourceRegistry` via `FileBackedRegistry`
    (see `brain/registry_base.py`).
    """

    resource_kind = "Tool"

    def _allowed_names(self) -> Iterable[str]:
        return TOOLS

    def _module_dir(self) -> Path:
        return Path(__file__).parent

    def check_tool_available(self, tool_name: str) -> bool:
        """
        Check if tool file exists

        Checks if the tool_name exists in TOOLS list and if the
        corresponding .py file exists in the tools directory.

        Args:
            tool_name: Name of the tool (e.g., "tavily_search")

        Returns:
            True if tool is in TOOLS and file exists, False otherwise
        """
        return self.is_available(tool_name)


# Singleton
_tool_registry_instance = None


def get_tool_registry() -> ToolRegistry:
    """Get singleton ToolRegistry instance"""
    global _tool_registry_instance
    if _tool_registry_instance is None:
        _tool_registry_instance = ToolRegistry()
    return _tool_registry_instance
