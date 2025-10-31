"""
Data Source Manager
Central manager for all data sources with intelligent routing and caching
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

from ..interfaces import IDataSource, DataSourceType
from .base_data_source import DataSourceConfig
from .vector_db import VectorDataSource
from .filesystem import FileSystemDataSource
from .api import APIDataSource
from .memory import MemoryDataSource

logger = logging.getLogger(__name__)


class DataSourceManager:
    """Manager for all data sources with intelligent routing and caching"""
    
    def __init__(self):
        self.data_sources: Dict[str, IDataSource] = {}
        self.configs: Dict[str, DataSourceConfig] = {}
        self.cache: MemoryDataSource = None
        self._initialized = False
    
    async def initialize(self, vectorstore=None) -> None:
        """Initialize data source manager"""
        if self._initialized:
            return
        
        # Initialize cache
        cache_config = DataSourceConfig(
            source_id="cache",
            source_type=DataSourceType.MEMORY,
            config={}
        )
        self.cache = MemoryDataSource(cache_config)
        await self.cache.initialize({})
        
        # Register default data sources
        if vectorstore:
            await self.register_vector_source(vectorstore)
        
        await self.register_filesystem_source()
        
        self._initialized = True
        logger.info("Data Source Manager initialized")
    
    async def register_vector_source(self, vectorstore) -> str:
        """Register vector database source"""
        config = DataSourceConfig(
            source_id="vector_db",
            source_type=DataSourceType.VECTOR_DB,
            priority=10,  # High priority
            config={}
        )
        
        source = VectorDataSource(vectorstore, config)
        await source.initialize({})
        
        self.data_sources["vector_db"] = source
        self.configs["vector_db"] = config
        
        logger.info("Registered vector database source")
        return "vector_db"
    
    async def register_filesystem_source(self, allowed_paths: List[str] = None) -> str:
        """Register filesystem source"""
        if allowed_paths is None:
            allowed_paths = ["./data", "./docs"]
        
        config = DataSourceConfig(
            source_id="filesystem",
            source_type=DataSourceType.FILE_SYSTEM,
            priority=5,
            config={"allowed_paths": allowed_paths}
        )
        
        source = FileSystemDataSource(config)
        await source.initialize({})
        
        self.data_sources["filesystem"] = source
        self.configs["filesystem"] = config
        
        logger.info("Registered filesystem source")
        return "filesystem"
    
    async def register_api_source(self, source_id: str, base_url: str, api_key: str = None) -> str:
        """Register API source"""
        config = DataSourceConfig(
            source_id=source_id,
            source_type=DataSourceType.API,
            priority=3,
            config={
                "base_url": base_url,
                "api_key": api_key
            }
        )
        
        source = APIDataSource(config)
        await source.initialize({})
        
        self.data_sources[source_id] = source
        self.configs[source_id] = config
        
        logger.info(f"Registered API source: {source_id}")
        return source_id
    
    async def register_custom_source(self, source_id: str, source: IDataSource, config: DataSourceConfig) -> str:
        """Register custom data source"""
        await source.initialize(config.config)
        
        self.data_sources[source_id] = source
        self.configs[source_id] = config
        
        logger.info(f"Registered custom source: {source_id}")
        return source_id
    
    async def register_external_vector_source(self, db_path: str, source_id: str = None) -> Optional[str]:
        """
        Register an external agent-specific vector database source
        
        Args:
            db_path: Path to the ChromaDB (chroma.sqlite3 file or directory)
            source_id: Optional identifier for this source
            
        Returns:
            Source ID if successful, None otherwise
        """
        try:
            # Create VectorDataSource from external path
            source = await VectorDataSource.create_from_path(db_path, source_id)
            
            if source:
                final_source_id = source.config.source_id
                self.data_sources[final_source_id] = source
                self.configs[final_source_id] = source.config
                
                logger.info(f"Registered external vector source: {final_source_id} from {db_path}")
                return final_source_id
            else:
                logger.error(f"Failed to create external vector source from: {db_path}")
                return None
                
        except Exception as e:
            logger.error(f"Error registering external vector source from {db_path}: {str(e)}")
            return None
    
    async def get_or_create_external_source(self, spec_config: Dict[str, Any]) -> Optional[str]:
        """
        Get existing or create external data source based on specification config
        
        Args:
            spec_config: Configuration from DataSourceSpec.config containing:
                - db_path: Path to the database
                - source_id: Optional identifier
                
        Returns:
            Source ID if successful, None otherwise
        """
        db_path = spec_config.get("db_path")
        if not db_path:
            logger.warning("No db_path provided in external source specification")
            return None
        
        source_id = spec_config.get("source_id")
        
        # Check if source already registered
        if source_id and source_id in self.data_sources:
            logger.debug(f"External source {source_id} already registered")
            return source_id
        
        # Register new external source
        return await self.register_external_vector_source(db_path, source_id)
    
    async def query_all(self, query: str, context: Dict[str, Any] = None, **kwargs) -> Dict[str, List[Dict[str, Any]]]:
        """Query all enabled data sources"""
        results = {}
        
        # Check cache first
        cache_key = f"query:{hash(query + str(context) + str(kwargs))}"
        cached_result = await self.cache.retrieve(cache_key)
        if cached_result:
            return cached_result
        
        # Query all sources in parallel
        tasks = []
        source_ids = []
        
        for source_id, source in self.data_sources.items():
            config = self.configs[source_id]
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
        
        # Cache results
        await self.cache.store(cache_key, results, {
            "timestamp": datetime.now().isoformat(),
            "query": query
        })
        
        return results
    
    async def query_best(self, query: str, context: Dict[str, Any] = None, **kwargs) -> List[Dict[str, Any]]:
        """Query and return best results from highest priority sources"""
        all_results = await self.query_all(query, context, **kwargs)
        
        # Sort sources by priority
        sorted_sources = sorted(
            self.configs.items(),
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
    
    async def health_check_all(self) -> Dict[str, Dict[str, Any]]:
        """Health check all data sources"""
        results = {}
        
        for source_id, source in self.data_sources.items():
            results[source_id] = await source.health_check()
        
        # Include cache health
        if self.cache:
            results["cache"] = await self.cache.health_check()
        
        return results
    
    def get_source(self, source_id: str) -> Optional[IDataSource]:
        """Get data source by ID"""
        return self.data_sources.get(source_id)
    
    def list_sources(self) -> Dict[str, Dict[str, Any]]:
        """List all registered sources"""
        return {
            source_id: {
                "type": config.source_type.value,
                "enabled": config.enabled,
                "priority": config.priority,
                "config": config.config
            }
            for source_id, config in self.configs.items()
        }
    
    async def shutdown(self) -> None:
        """Shutdown all data sources"""
        logger.info("Shutting down Data Source Manager...")
        
        for source in self.data_sources.values():
            await source.close()
        
        if self.cache:
            await self.cache.close()
        
        self.data_sources.clear()
        self.configs.clear()
        
        logger.info("Data Source Manager shutdown complete")


# Global data source manager instance
_data_source_manager: Optional[DataSourceManager] = None


def get_data_source_manager() -> DataSourceManager:
    """Get the global data source manager instance"""
    global _data_source_manager
    if _data_source_manager is None:
        _data_source_manager = DataSourceManager()
    return _data_source_manager

