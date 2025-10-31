"""
Data Source Registry
Central registry for managing and discovering available data sources
"""

import logging
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field

from ..interfaces import DataSourceType
from .base_data_source import DataSourceConfig

logger = logging.getLogger(__name__)


@dataclass
class DataSourceInfo:
    """Information about a data source type"""
    name: str
    available: bool
    description: str
    required_config: List[str]
    optional_config: List[str]
    supported_operations: List[str]
    can_be_external: bool = False


class DataSourceRegistry:
    """Registry for managing data source types and configurations"""
    
    def __init__(self):
        self._sources: Dict[str, DataSourceInfo] = {}
        self._initialized = False
        logger.info("Data Source Registry initialized")
    
    async def initialize(self) -> None:
        """Initialize the data source registry"""
        if self._initialized:
            return
        
        # Register all built-in data source types
        self._register_vector_db()
        self._register_filesystem()
        self._register_api()
        self._register_memory()
        self._register_conversation_history()
        
        self._initialized = True
        available_count = sum(1 for info in self._sources.values() if info.available)
        logger.info(
            f"Data Source Registry initialized: {available_count}/{len(self._sources)} source types available"
        )
    
    def _register_vector_db(self) -> None:
        """Register vector database data source type"""
        try:
            # Check if required libraries are available
            import chromadb
            import langchain_chroma
            available = True
        except ImportError:
            available = False
        
        self._sources["vector_db"] = DataSourceInfo(
            name="vector_db",
            available=available,
            description="Vector database for semantic search using embeddings",
            required_config=[],
            optional_config=["db_path", "collection_name", "embedding_model", "persist_directory"],
            supported_operations=["query", "similarity_search", "health_check"],
            can_be_external=True
        )
    
    def _register_filesystem(self) -> None:
        """Register filesystem data source type"""
        self._sources["filesystem"] = DataSourceInfo(
            name="filesystem",
            available=True,  # Always available
            description="File system access for document retrieval",
            required_config=[],
            optional_config=["allowed_paths", "file_extensions", "max_files"],
            supported_operations=["query", "read", "search", "health_check"],
            can_be_external=False
        )
    
    def _register_api(self) -> None:
        """Register API data source type"""
        try:
            import httpx
            available = True
        except ImportError:
            available = False
        
        self._sources["api"] = DataSourceInfo(
            name="api",
            available=available,
            description="External API integration for data retrieval",
            required_config=["base_url"],
            optional_config=["api_key", "headers", "endpoint", "timeout"],
            supported_operations=["query", "get", "post", "health_check"],
            can_be_external=False
        )
    
    def _register_memory(self) -> None:
        """Register memory data source type"""
        self._sources["memory"] = DataSourceInfo(
            name="memory",
            available=True,  # Always available
            description="In-memory caching and temporary storage",
            required_config=[],
            optional_config=["cache_ttl", "max_size"],
            supported_operations=["query", "store", "retrieve", "delete", "health_check"],
            can_be_external=False
        )
    
    def _register_conversation_history(self) -> None:
        """Register conversation history data source type"""
        self._sources["conversation_history"] = DataSourceInfo(
            name="conversation_history",
            available=True,  # Always available (agents provide this)
            description="Conversation history provided by agents",
            required_config=[],
            optional_config=["max_messages", "include_system"],
            supported_operations=["query", "retrieve"],
            can_be_external=False
        )
    
    def get_available_sources(self) -> Set[str]:
        """Get set of available data source type names"""
        return {name for name, info in self._sources.items() if info.available}
    
    def get_all_sources(self) -> Dict[str, DataSourceInfo]:
        """Get all registered data source types"""
        return self._sources.copy()
    
    def get_source_info(self, source_type: str) -> Optional[DataSourceInfo]:
        """Get information about a specific data source type"""
        return self._sources.get(source_type)
    
    def is_source_available(self, source_type: str) -> bool:
        """Check if a data source type is available"""
        source_info = self._sources.get(source_type)
        return source_info is not None and source_info.available
    
    def validate_source_config(self, source_type: str, config: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate configuration for a data source type
        
        Returns:
            (is_valid, error_messages)
        """
        source_info = self._sources.get(source_type)
        if not source_info:
            return False, [f"Unknown data source type: {source_type}"]
        
        if not source_info.available:
            return False, [f"Data source type '{source_type}' is not available (missing dependencies)"]
        
        errors = []
        
        # Check required config fields
        for required_field in source_info.required_config:
            if required_field not in config:
                errors.append(f"Missing required field: {required_field}")
        
        # Check for external path if applicable
        if source_type == "vector_db" and config.get("db_path"):
            import os
            db_path = config["db_path"]
            if not os.path.exists(db_path):
                errors.append(f"External database path does not exist: {db_path}")
        
        return len(errors) == 0, errors
    
    def supports_external_sources(self, source_type: str) -> bool:
        """Check if a data source type supports external/agent-specific sources"""
        source_info = self._sources.get(source_type)
        return source_info is not None and source_info.can_be_external
    
    def get_source_requirements(self, source_type: str) -> Dict[str, Any]:
        """Get requirements and configuration info for a data source type"""
        source_info = self._sources.get(source_type)
        if not source_info:
            return {}
        
        return {
            "available": source_info.available,
            "required_config": source_info.required_config,
            "optional_config": source_info.optional_config,
            "supported_operations": source_info.supported_operations,
            "can_be_external": source_info.can_be_external
        }


# Singleton instance
_registry_instance: Optional[DataSourceRegistry] = None


def get_data_source_registry() -> DataSourceRegistry:
    """Get the singleton data source registry instance"""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = DataSourceRegistry()
    return _registry_instance

