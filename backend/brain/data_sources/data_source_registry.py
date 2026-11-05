"""
Data Source Registry - Checks if data source files are available
"""

import logging
from pathlib import Path
from typing import Iterable

from ..registry_base import FileBackedRegistry

logger = logging.getLogger(__name__)

DATA_SOURCES = ["vector_db"]


class DataSourceRegistry(FileBackedRegistry[str]):
    """
    Registry for data sources.

    Shares its lookup logic with `ProviderRegistry` and `ToolRegistry` via
    `FileBackedRegistry` (see `brain/registry_base.py`).
    """

    resource_kind = "Data source"

    def _allowed_names(self) -> Iterable[str]:
        return DATA_SOURCES

    def _module_dir(self) -> Path:
        return Path(__file__).parent

    def check_data_source_available(self, source_name: str) -> bool:
        return self.is_available(source_name)


_data_source_registry_instance = None

def get_data_source_registry() -> DataSourceRegistry:
    global _data_source_registry_instance
    if _data_source_registry_instance is None:
        _data_source_registry_instance = DataSourceRegistry()
    return _data_source_registry_instance
