"""
Tavily Search Tool - Web search using Tavily API
"""

import os
import logging
from typing import Dict, Any

from .base_tool import BaseTool, ToolResult

logger = logging.getLogger(__name__)

# Check for Tavily availability
try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False
    logger.warning("Tavily not available. Install with: pip install tavily-python")


class TavilySearchTool(BaseTool):
    """
    Tavily Search Tool - Web search using Tavily API
    
    Best for:
    - Real-time information retrieval
    - Current events and news
    - Fact-checking
    - Research and discovery
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.api_key = config.get("api_key") if config else None
        if not self.api_key:
            self.api_key = os.getenv("TAVILY_API_KEY")
        self.client = None
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default Tavily configuration"""
        return {
            "api_key": os.getenv("TAVILY_API_KEY", ""),
            "max_results": 5,
            "search_depth": "basic",  # "basic" or "advanced"
            "include_answer": True,
            "include_raw_content": False,
            "include_images": False,
            "include_domains": [],
            "exclude_domains": []
        }
    
    async def initialize(self) -> bool:
        """Initialize Tavily client"""
        if not TAVILY_AVAILABLE:
            logger.error("Tavily library not available")
            return False
        
        if not self.api_key:
            logger.error("Tavily API key not found")
            return False
        
        try:
            self.client = TavilyClient(api_key=self.api_key)
            self._initialized = True
            logger.info("Tavily Search Tool initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Tavily: {str(e)}")
            return False
    
    async def execute(
        self,
        query: str,
        max_results: int = 5,
        search_depth: str = "basic",
        include_answer: bool = True,
        include_raw_content: bool = False,
        **kwargs: Any) -> ToolResult:
        """
        Execute web search using Tavily
        
        Args:
            query: Search query
            max_results: Maximum number of results (default: 5)
            search_depth: "basic" or "advanced" (default: "basic")
            include_answer: Include AI-generated answer (default: True)
            include_raw_content: Include raw page content (default: False)
            
        Returns:
            ToolResult with search results
        """
        if not self._initialized:
            await self.initialize()
        
        if not self.client:
            return ToolResult(
                tool_name="TavilySearch",
                success=False,
                content=None,
                metadata={},
                error="Tavily client not initialized"
            )
        
        try:
            logger.info(f"Tavily search: {query}")
            
            # Execute search
            response = self.client.search(
                query=query,
                max_results=max_results,
                search_depth=search_depth,
                include_answer=include_answer,
                include_raw_content=include_raw_content
            )
            
            # Format results
            results = {
                "query": query,
                "answer": response.get("answer", ""),
                "results": []
            }
            
            for item in response.get("results", []):
                results["results"].append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "content": item.get("content", ""),
                    "score": item.get("score", 0)
                })
            
            logger.info(f"Tavily found {len(results['results'])} results")
            
            return ToolResult(
                tool_name="TavilySearch",
                success=True,
                content=results,
                metadata={
                    "num_results": len(results["results"]),
                    "search_depth": search_depth
                }
            )
            
        except Exception as e:
            logger.error(f"Tavily search error: {str(e)}")
            return ToolResult(
                tool_name="TavilySearch",
                success=False,
                content=None,
                metadata={},
                error=str(e)
            )
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get tool parameters schema"""
        return {
            "query": {
                "type": "string",
                "description": "Search query",
                "required": True
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results",
                "default": 5
            },
            "search_depth": {
                "type": "string",
                "description": "Search depth: basic or advanced",
                "default": "basic",
                "enum": ["basic", "advanced"]
            },
            "include_answer": {
                "type": "boolean",
                "description": "Include AI-generated answer",
                "default": True
            }
        }
    
    async def close(self) -> None:
        """Close Tavily client"""
        self.client = None
        self._initialized = False
        logger.debug("Tavily Search Tool closed")


def create_tavily_search_tool(config: Dict[str, Any] = None) -> TavilySearchTool:
    """Factory function to create Tavily Search Tool"""
    return TavilySearchTool(config)

