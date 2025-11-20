"""
Data Source Registry - Central registry of available data sources and configurations
Used to validate data source specifications from plugins and manage data source instances
"""

import asyncio
import logging
import os
from typing import Dict, List, Any, Optional, Set, Tuple

from ..types import DataSourceSpec, DataSourceType
from ..interfaces import IDataSource
from .base_data_source import DataSourceConfig
from .vector_db import VectorDataSource

logger = logging.getLogger(__name__)


# Data source configuration dictionaries
DATA_SOURCE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "vector_db": {
        "description": "Vector database for semantic search using embeddings",
        "required_config": [],
        "optional_config": ["db_path", "collection_name", "embedding_model", "persist_directory", "k"],
        "supported_operations": ["query", "similarity_search", "health_check"],
        "can_be_external": True,
        "defaults": {
            "enabled": True,
            "k": 5,
            "collection_name": "default_collection",
            "embedding_model": "text-embedding-ada-002"
        },
        "library": "chromadb",
        "embedding_path": "data/embeddings"
    },
}


class DataSourceRegistry:
    """Central registry for all data sources - manages instances and validates specs"""
    
    def __init__(self):
        self._sources: Dict[str, Dict[str, Any]] = {}
        self._instances: Dict[str, IDataSource] = {}
        self._configs: Dict[str, DataSourceConfig] = {}
        self._initialized = False
    
    async def initialize(self, vectorstore=None) -> None:
        """Initialize the data source registry"""
        if self._initialized:
            return
        
        # Register source types
        for source_name in DATA_SOURCE_CONFIGS:
            self._register_source(source_name)
        
        # Initialize default vector source
        if vectorstore:
            await self.register_vector_source(vectorstore)
        else:
            await self.register_default_vector_source()
        
        self._initialized = True
        available = sum(1 for s in self._sources.values() if s.get("available", False))
        logger.info(f"Data Source Registry initialized: {available}/{len(self._sources)} sources available")
    
    def _register_source(self, source_name: str) -> None:
        """Register a data source type from DATA_SOURCE_CONFIGS"""
        config = DATA_SOURCE_CONFIGS.get(source_name)
        if not config:
            return
        
        available = self._check_availability(source_name, config)
        self._sources[source_name] = {**config, "available": available, "name": source_name}
    
    def _check_availability(self, source_name: str, config: Dict[str, Any]) -> bool:
        """Check if data source library is available"""
        library = config.get("library")
        if not library:
            return True
        
        library_map = {"chromadb": "chromadb", "httpx": "httpx"}
        if library in library_map:
            try:
                __import__(library_map[library])
                return True
            except ImportError:
                return False
        return True
    
    async def register_vector_source(self, vectorstore) -> str:
        """Register vector database source with provided vectorstore"""
        config = DataSourceConfig(
            source_id="vector_db",
            source_type=DataSourceType.VECTOR_DB,
            priority=10,
            config={}
        )
        
        source = VectorDataSource(config)
        await source.initialize({"vectorstore": vectorstore})
        
        self._instances["vector_db"] = source
        self._configs["vector_db"] = config
        logger.info("Registered vector database source")
        return "vector_db"
    
    async def register_default_vector_source(self) -> str:
        """Register default vector database source"""
        config = DataSourceConfig(
            source_id="vector_db",
            source_type=DataSourceType.VECTOR_DB,
            priority=10,
            config={}
        )
        
        source = VectorDataSource(config)
        await source.initialize({})
        
        self._instances["vector_db"] = source
        self._configs["vector_db"] = config
        logger.info("Registered default vector database source")
        return "vector_db"
    
    async def register_external_vector_source(self, db_path: str, source_id: str = None) -> Optional[str]:
        """Register an external agent-specific vector database source"""
        try:
            config = DataSourceConfig(
                source_id=source_id or f"external_vector_{db_path.split('/')[-1]}",
                source_type=DataSourceType.VECTOR_DB,
                priority=10,
                config={"db_path": db_path}
            )
            
            source = VectorDataSource(config)
            success = await source.initialize({"db_path": db_path})
            
            if success:
                final_source_id = config.source_id
                self._instances[final_source_id] = source
                self._configs[final_source_id] = config
                logger.info(f"Registered external vector source: {final_source_id} from {db_path}")
                return final_source_id
            else:
                logger.error(f"Failed to initialize external vector source from: {db_path}")
                return None
        except Exception as e:
            logger.error(f"Error registering external vector source from {db_path}: {str(e)}")
            return None
    
    async def get_or_create_external_source(self, spec_config: Dict[str, Any]) -> Optional[str]:
        """Get existing or create external data source based on specification config"""
        db_path = spec_config.get("db_path")
        if not db_path:
            logger.warning("No db_path provided in external source specification")
            return None
        
        source_id = spec_config.get("source_id")
        
        # Check if source already registered
        if source_id and source_id in self._instances:
            logger.debug(f"External source {source_id} already registered")
            return source_id
        
        # Register new external source
        return await self.register_external_vector_source(db_path, source_id)
    
    async def query_all(self, query: str, context: Dict[str, Any] = None, **kwargs) -> Dict[str, List[Dict[str, Any]]]:
        """Query all enabled data sources"""
        results = {}
        
        # Query all sources in parallel
        tasks = []
        source_ids = []
        
        for source_id, source in self._instances.items():
            config = self._configs[source_id]
            if config.enabled:
                tasks.append(source.query(query, context, **kwargs))
                source_ids.append(source_id)
        
        if tasks:
            source_results = await asyncio.gather(*tasks, return_exceptions=True)
            for source_id, result in zip(source_ids, source_results):
                if isinstance(result, Exception):
                    logger.error(f"Query failed for source {source_id}: {str(result)}")
                    results[source_id] = []
                else:
                    results[source_id] = result or []
        
        return results
    
    async def query_best(self, query: str, context: Dict[str, Any] = None, **kwargs) -> List[Dict[str, Any]]:
        """Query and return best results from highest priority sources"""
        all_results = await self.query_all(query, context, **kwargs)
        
        # Sort sources by priority
        sorted_sources = sorted(
            self._configs.items(),
            key=lambda x: x[1].priority,
            reverse=True
        )
        
        # Collect results from highest priority sources first
        best_results = []
        max_results = kwargs.get("max_results", 10)
        
        for source_id, config in sorted_sources:
            if source_id in all_results and config.enabled:
                source_results = all_results[source_id]
                remaining = max_results - len(best_results)
                if remaining > 0:
                    best_results.extend(source_results[:remaining])
                if len(best_results) >= max_results:
                    break
        
        return best_results
    
    def check_data_source_spec_availability(
        self, 
        data_source_spec: DataSourceSpec
    ) -> Tuple[bool, List[str], Optional[Dict[str, Any]]]:
        """Main function to check if a DataSourceSpec is available and valid"""
        # Ensure source is registered
        if data_source_spec.source_type.value not in self._sources:
            self._register_source(data_source_spec.source_type.value)
        
        errors = []
        source_type = data_source_spec.source_type.value
        source_config = DATA_SOURCE_CONFIGS.get(source_type)
        
        if not source_config:
            return False, [f"Data source '{source_type}' not found. Available: {list(DATA_SOURCE_CONFIGS.keys())}"], None
        
        source_info = self._sources.get(source_type)
        
        # Check availability
        if not source_info or not source_info.get("available", False):
            errors.append(f"Data source '{source_type}' not available. Check if library is installed ({source_config.get('library', 'N/A')})")
        
        # Validate config
        if data_source_spec.config:
            required = source_config.get("required_config", [])
            for field in required:
                if field not in data_source_spec.config:
                    errors.append(f"Missing required config field: {field}")
            
            # Validate db_path if provided
            if source_type == "vector_db" and "db_path" in data_source_spec.config:
                db_path = data_source_spec.config["db_path"]
                if not os.path.exists(db_path):
                    errors.append(f"Database path does not exist: {db_path}")
        
        # Validate limit
        if data_source_spec.limit and data_source_spec.limit < 1:
            errors.append(f"Limit must be at least 1, got {data_source_spec.limit}")
        if data_source_spec.limit and data_source_spec.limit > 100:
            errors.append(f"Limit exceeds maximum 100, got {data_source_spec.limit}")
        
        if errors:
            return False, errors, None
        
        # Build success info
        defaults = source_config.get("defaults", {})
        return True, [], {
            "source_type": source_type,
            "available": True,
            "config": data_source_spec.config or defaults,
            "limit": data_source_spec.limit or defaults.get("k", 5),
            "enabled": data_source_spec.enabled,
            "supported_operations": source_config.get("supported_operations", []),
            "can_be_external": source_config.get("can_be_external", False)
        }
    
    # Getter methods
    def get_source_config(self, source_type: str) -> Optional[Dict[str, Any]]:
        return DATA_SOURCE_CONFIGS.get(source_type)
    
    def get_available_sources(self) -> Set[str]:
        return {name for name, info in self._sources.items() if info.get("available", False)}
    
    def get_all_sources(self) -> Dict[str, Dict[str, Any]]:
        return self._sources.copy()
    
    def get_source_info(self, source_type: str) -> Optional[Dict[str, Any]]:
        return self._sources.get(source_type)
    
    def is_source_available(self, source_type: str) -> bool:
        info = self._sources.get(source_type)
        return info is not None and info.get("available", False)
    
    def supports_external_sources(self, source_type: str) -> bool:
        info = self._sources.get(source_type)
        return info is not None and info.get("can_be_external", False)
    
    def get_source(self, source_id: str) -> Optional[IDataSource]:
        """Get data source instance by ID"""
        return self._instances.get(source_id)
    
    def list_sources(self) -> Dict[str, Dict[str, Any]]:
        """List all registered source instances"""
        return {
            source_id: {
                "type": config.source_type.value,
                "enabled": config.enabled,
                "priority": config.priority,
                "config": config.config
            }
            for source_id, config in self._configs.items()
        }
    
    async def health_check_all(self) -> Dict[str, Dict[str, Any]]:
        """Health check all data source instances"""
        results = {}
        for source_id, source in self._instances.items():
            results[source_id] = await source.health_check()
        return results
    
    async def shutdown(self) -> None:
        """Shutdown all data source instances"""
        logger.info("Shutting down Data Source Registry...")
        for source in self._instances.values():
            await source.close()
        self._instances.clear()
        self._configs.clear()
        logger.info("Data Source Registry shutdown complete")


# Singleton
_data_source_registry_instance: Optional[DataSourceRegistry] = None


def get_data_source_registry() -> DataSourceRegistry:
    """Get singleton DataSourceRegistry instance"""
    global _data_source_registry_instance
    if _data_source_registry_instance is None:
        _data_source_registry_instance = DataSourceRegistry()
    return _data_source_registry_instance


def check_data_source_spec_availability(data_source_spec: DataSourceSpec) -> Tuple[bool, List[str], Optional[Dict[str, Any]]]:
    """Convenience function to check if a DataSourceSpec is available and valid"""
    return get_data_source_registry().check_data_source_spec_availability(data_source_spec)
