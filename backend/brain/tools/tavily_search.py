"""
Tavily Search Tool - Web search using Tavily API
"""

import asyncio
import logging
from typing import Dict, Any

from .base_tool import BaseTool, ToolResult
from config import settings

logger = logging.getLogger(__name__)

try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False


class TavilySearchTool(BaseTool):
    """
    Tavily Search Tool - Web search using Tavily API
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.api_key = (config or {}).get("api_key") or settings.tavily_api_key
        self.client = None
    
    def get_default_config(self) -> Dict[str, Any]:
        return {
            "api_key": settings.tavily_api_key or "",
            "max_results": 5,
            "search_depth": "basic",
            "include_answer": True,
            "include_raw_content": False
        }
    
    async def initialize(self) -> bool:
        if not TAVILY_AVAILABLE:
            logger.error("Tavily library not available")
            return False
        
        if not self.api_key:
            logger.error("Tavily API key missing")
            return False
        
        try:
            self.client = TavilyClient(api_key=self.api_key)
            self._initialized = True
            logger.info("Tavily Search Tool initialized")
            return True
        except Exception:
            logger.exception("Failed to initialize Tavily")
            return False
    
    async def execute(self, query: str, max_results: int = 5, search_depth: str = "basic", **kwargs) -> ToolResult:
        if not self._initialized:
            await self.initialize()
        
        if not self.client:
            return ToolResult("TavilySearch", False, None, {}, "Client not initialized")
        
        try:
            logger.info(f"Searching: {query}")

            # TavilyClient.search() is a synchronous, blocking network call -
            # asyncio.to_thread keeps it from blocking the event loop while
            # it runs (Epic 5 / Story 5.1, Task 7.4).
            resp = await asyncio.to_thread(
                self.client.search,
                query=query,
                max_results=max_results,
                search_depth=search_depth,
                include_answer=kwargs.get("include_answer", True),
                include_raw_content=kwargs.get("include_raw_content", False)
            )
            
            results = {
                "query": query,
                "answer": resp.get("answer", ""),
                "results": [
                    {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "content": item.get("content", ""),
                    "score": item.get("score", 0)
                    }
                    for item in resp.get("results", [])
                ]
            }
            
            return ToolResult("TavilySearch", True, results, {"count": len(results["results"])})
            
        except Exception as e:
            logger.exception("Tavily search error")
            return ToolResult("TavilySearch", False, None, {}, str(e))
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            "query": {"type": "string", "required": True},
            "max_results": {"type": "integer", "default": 5},
            "search_depth": {"type": "string", "enum": ["basic", "advanced"], "default": "basic"}
        }
    
    async def close(self) -> None:
        self.client = None
        self._initialized = False


def create_tavily_search_tool(config: Dict[str, Any] = None) -> TavilySearchTool:
    return TavilySearchTool(config)
