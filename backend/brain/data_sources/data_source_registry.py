"""
Data Source Registry - Checks if data source files are available
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


# Available data sources list
DATA_SOURCES = ["vector_db"]


class DataSourceRegistry:
    """
    Registry for data sources
    
    Checks if data source files exist
    """
    
    def check_data_source_available(self, source_name: str) -> bool:
        """
        Check if data source file exists
        
        Checks if the source_name exists in DATA_SOURCES list and if the
        corresponding .py file exists in the data_sources directory.
        
        Args:
            source_name: Name of the data source (e.g., "vector_db")
            
        Returns:
            True if source is in DATA_SOURCES and file exists, False otherwise
        """
        # Check if source is in DATA_SOURCES list
        if source_name not in DATA_SOURCES:
            logger.warning(f"Data source '{source_name}' not found in DATA_SOURCES list")
            return False
        
        # Check if .py file exists
        file_path = Path(__file__).parent / f"{source_name}.py"
        exists = file_path.exists()
        
        if not exists:
            logger.warning(f"Data source file not found: {source_name}.py")
        
        return exists


# Singleton
_data_source_registry_instance = None


def get_data_source_registry() -> DataSourceRegistry:
    """Get singleton DataSourceRegistry instance"""
    global _data_source_registry_instance
    if _data_source_registry_instance is None:
        _data_source_registry_instance = DataSourceRegistry()
    return _data_source_registry_instance
