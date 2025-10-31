"""
Memory Data Source
Provides in-memory caching and temporary data storage
"""

import logging
from typing import Dict, Any, List, Optional

from ..interfaces import IDataSource, DataSourceType
from .base_data_source import DataSourceConfig

logger = logging.getLogger(__name__)


class MemoryDataSource(IDataSource):
    """In-memory data source for caching and temporary data"""
    
    def __init__(self, config: DataSourceConfig):
        self.config = config
        self.data: Dict[str, Any] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self._initialized = False
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize memory data source"""
        self._initialized = True
        logger.info(f"Memory data source {self.config.source_id} initialized")
        return True
    
    async def query(self, query: str, context: Dict[str, Any] = None, **kwargs) -> List[Dict[str, Any]]:
        """Query in-memory data"""
        if not self._initialized:
            return []
        
        try:
            results = []
            query_lower = query.lower()
            
            for key, value in self.data.items():
                # Simple text matching
                if query_lower in key.lower() or query_lower in str(value).lower():
                    results.append({
                        "content": str(value),
                        "metadata": {
                            "key": key,
                            **self.metadata.get(key, {})
                        },
                        "source": "memory"
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Memory query failed: {str(e)}")
            return []
    
    async def store(self, key: str, value: Any, metadata: Dict[str, Any] = None) -> bool:
        """Store data in memory"""
        try:
            self.data[key] = value
            if metadata:
                self.metadata[key] = metadata
            return True
        except Exception as e:
            logger.error(f"Memory store failed: {str(e)}")
            return False
    
    async def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve data from memory"""
        return self.data.get(key)
    
    async def delete(self, key: str) -> bool:
        """Delete data from memory"""
        try:
            if key in self.data:
                del self.data[key]
            if key in self.metadata:
                del self.metadata[key]
            return True
        except Exception as e:
            logger.error(f"Memory delete failed: {str(e)}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Check memory data source health"""
        return {
            "status": "healthy",
            "data_count": len(self.data),
            "memory_usage": len(str(self.data)),
            "initialized": self._initialized
        }
    
    async def close(self) -> None:
        """Close memory data source"""
        self.data.clear()
        self.metadata.clear()
        self._initialized = False
    
    @property
    def source_type(self) -> DataSourceType:
        return DataSourceType.MEMORY


# Factory function
def create_memory_data_source(config: DataSourceConfig = None) -> MemoryDataSource:
    """Create a memory data source instance"""
    if config is None:
        config = DataSourceConfig(
            source_id="memory_cache",
            source_type=DataSourceType.MEMORY,
            priority=1
        )
    return MemoryDataSource(config)

