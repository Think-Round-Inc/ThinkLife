"""
Data Sources Package
Provides pluggable data sources for ThinkxLife Brain with central registry and management
"""

from .base_data_source import DataSourceConfig
from .vector_db import VectorDataSource, create_vector_data_source
from .filesystem import FileSystemDataSource, create_filesystem_data_source
from .api import APIDataSource, create_api_data_source
from .memory import MemoryDataSource, create_memory_data_source
from .data_source_manager import DataSourceManager, get_data_source_manager
from .data_source_registry import DataSourceRegistry, DataSourceInfo, get_data_source_registry

__all__ = [
    # Configuration
    "DataSourceConfig",
    
    # Data Source Types
    "VectorDataSource",
    "FileSystemDataSource",
    "APIDataSource",
    "MemoryDataSource",
    
    # Factory Functions
    "create_vector_data_source",
    "create_filesystem_data_source",
    "create_api_data_source",
    "create_memory_data_source",
    
    # Manager
    "DataSourceManager",
    "get_data_source_manager",
    
    # Registry
    "DataSourceRegistry",
    "DataSourceInfo",
    "get_data_source_registry",
]

