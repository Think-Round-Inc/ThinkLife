"""
Filesystem Data Source
Provides file system integration for document retrieval
"""

import os
import logging
from typing import Dict, Any, List

from ..interfaces import IDataSource, DataSourceType
from .base_data_source import DataSourceConfig

logger = logging.getLogger(__name__)


class FileSystemDataSource(IDataSource):
    """File system data source implementation"""
    
    def __init__(self, config: DataSourceConfig):
        self.config = config
        self.allowed_paths = config.config.get("allowed_paths", ["./data"])
        self._initialized = False
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize filesystem data source"""
        # Ensure allowed paths exist
        for path in self.allowed_paths:
            os.makedirs(path, exist_ok=True)
        
        self._initialized = True
        logger.info(f"Filesystem data source {self.config.source_id} initialized")
        return True
    
    async def query(self, query: str, context: Dict[str, Any] = None, **kwargs) -> List[Dict[str, Any]]:
        """Query filesystem for relevant files"""
        if not self._initialized:
            return []
        
        try:
            results = []
            file_extensions = kwargs.get("extensions", [".txt", ".md", ".json"])
            max_files = kwargs.get("max_files", 10)
            
            for allowed_path in self.allowed_paths:
                if not os.path.exists(allowed_path):
                    continue
                
                for root, dirs, files in os.walk(allowed_path):
                    for file in files:
                        if any(file.endswith(ext) for ext in file_extensions):
                            file_path = os.path.join(root, file)
                            
                            # Simple relevance check based on filename
                            if query.lower() in file.lower():
                                try:
                                    with open(file_path, 'r', encoding='utf-8') as f:
                                        content = f.read()
                                    
                                    results.append({
                                        "content": content[:1000],  # Truncate for preview
                                        "metadata": {
                                            "file_path": file_path,
                                            "file_name": file,
                                            "file_size": os.path.getsize(file_path)
                                        },
                                        "source": "filesystem"
                                    })
                                    
                                    if len(results) >= max_files:
                                        break
                                        
                                except Exception as e:
                                    logger.warning(f"Could not read file {file_path}: {str(e)}")
            
            return results
            
        except Exception as e:
            logger.error(f"Filesystem query failed: {str(e)}")
            return []
    
    async def health_check(self) -> Dict[str, Any]:
        """Check filesystem health"""
        try:
            accessible_paths = []
            for path in self.allowed_paths:
                if os.path.exists(path) and os.access(path, os.R_OK):
                    accessible_paths.append(path)
            
            return {
                "status": "healthy" if accessible_paths else "degraded",
                "accessible_paths": accessible_paths,
                "total_paths": len(self.allowed_paths),
                "initialized": self._initialized
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "initialized": self._initialized
            }
    
    async def close(self) -> None:
        """Close filesystem data source"""
        self._initialized = False
    
    @property
    def source_type(self) -> DataSourceType:
        return DataSourceType.FILE_SYSTEM


# Factory function
def create_filesystem_data_source(allowed_paths: List[str] = None, config: DataSourceConfig = None) -> FileSystemDataSource:
    """Create a filesystem data source instance"""
    if config is None:
        config = DataSourceConfig(
            source_id="filesystem",
            source_type=DataSourceType.FILE_SYSTEM,
            priority=5,
            config={"allowed_paths": allowed_paths or ["./data", "./docs"]}
        )
    return FileSystemDataSource(config)

