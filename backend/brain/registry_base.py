"""
Generic registry base class.

`ProviderRegistry`, `ToolRegistry`, and `DataSourceRegistry` (see
`brain/providers/provider_registry.py`, `brain/tools/tool_registry.py`,
`brain/data_sources/data_source_registry.py`) previously each hand-rolled the
same two checks:

  1. Is `name` in a known allow-list?
  2. Does a `<name>.py` file exist alongside the registry module?

This module factors that out into one generic, typed base class so the three
registries share one implementation instead of three near-identical copies.
Each subclass only needs to say what its allow-list is and where its `.py`
files live; `T` is the type of the resource identifier (currently `str` for
all three, but kept generic so a richer descriptor type can be substituted
later without changing the base class).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Generic, Iterable, List, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=str)


class FileBackedRegistry(Generic[T]):
    """
    Base class for registries that validate a resource name against an
    allow-list and confirm a corresponding `<name>.py` file exists on disk.

    Subclasses set `resource_kind` (used only for log messages) and implement
    `_allowed_names()` / `_module_dir()`.
    """

    resource_kind: str = "resource"

    def _allowed_names(self) -> Iterable[T]:  # pragma: no cover - overridden
        raise NotImplementedError

    def _module_dir(self) -> Path:  # pragma: no cover - overridden
        raise NotImplementedError

    def is_available(self, name: T) -> bool:
        """
        True if `name` is in the allow-list AND its backing `.py` file exists.
        This is the shared logic previously duplicated across
        `ProviderRegistry.check_provider_and_model`,
        `ToolRegistry.check_tool_available`, and
        `DataSourceRegistry.check_data_source_available`.
        """
        allowed: List[T] = list(self._allowed_names())
        if name not in allowed:
            logger.warning(f"{self.resource_kind} '{name}' not in allowed list: {allowed}")
            return False

        file_path = self._module_dir() / f"{name}.py"
        if not file_path.exists():
            logger.warning(f"{self.resource_kind} file missing: {file_path.name}")
            return False

        return True
