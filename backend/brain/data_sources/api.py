"""
API Data Source
Provides external API integration for data retrieval
"""

import logging
from typing import Dict, Any, List

from ..interfaces import IDataSource, DataSourceType
from .base_data_source import DataSourceConfig

logger = logging.getLogger(__name__)


class APIDataSource(IDataSource):
    """API-based data source implementation"""
    
    def __init__(self, config: DataSourceConfig):
        self.config = config
        self.base_url = config.config.get("base_url")
        self.api_key = config.config.get("api_key")
        self.headers = config.config.get("headers", {})
        self._initialized = False
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize API data source"""
        if not self.base_url:
            logger.error(f"API data source {self.config.source_id} missing base_url")
            return False
        
        self._initialized = True
        logger.info(f"API data source {self.config.source_id} initialized")
        return True
    
    async def query(self, query: str, context: Dict[str, Any] = None, **kwargs) -> List[Dict[str, Any]]:
        """Query API endpoint"""
        if not self._initialized:
            return []
        
        try:
            import httpx
            
            headers = self.headers.copy()
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            endpoint = kwargs.get("endpoint", "/search")
            params = {
                "query": query,
                "limit": kwargs.get("limit", 10)
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}{endpoint}",
                    params=params,
                    headers=headers,
                    timeout=self.config.timeout
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Normalize response format
                    if isinstance(data, list):
                        results = data
                    elif isinstance(data, dict) and "results" in data:
                        results = data["results"]
                    else:
                        results = [data]
                    
                    return [
                        {
                            "content": str(result),
                            "metadata": {"api_source": self.config.source_id},
                            "source": "api"
                        }
                        for result in results
                    ]
                else:
                    logger.error(f"API query failed with status {response.status_code}")
                    return []
                    
        except Exception as e:
            logger.error(f"API query failed: {str(e)}")
            return []
    
    async def health_check(self) -> Dict[str, Any]:
        """Check API health"""
        try:
            import httpx
            
            headers = self.headers.copy()
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/health",
                    headers=headers,
                    timeout=5.0
                )
                
                return {
                    "status": "healthy" if response.status_code == 200 else "degraded",
                    "status_code": response.status_code,
                    "initialized": self._initialized
                }
                
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "initialized": self._initialized
            }
    
    async def close(self) -> None:
        """Close API data source"""
        self._initialized = False
    
    @property
    def source_type(self) -> DataSourceType:
        return DataSourceType.API


# Factory function
def create_api_data_source(base_url: str, api_key: str = None, config: DataSourceConfig = None) -> APIDataSource:
    """Create an API data source instance"""
    if config is None:
        config = DataSourceConfig(
            source_id=f"api_{base_url.split('//')[1].split('/')[0]}",  # Extract domain
            source_type=DataSourceType.API,
            priority=3,
            config={"base_url": base_url, "api_key": api_key}
        )
    return APIDataSource(config)

