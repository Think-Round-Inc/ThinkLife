"""
Tool Registry - Manages and provides access to all available tools
"""

import logging
from typing import Dict, List, Optional

from .base_tool import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Registry for managing available tools"""
    
    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize all tools"""
        if self._initialized:
            return
        
        # Import and register available tools
        try:
            from .tavily_search import TavilySearchTool
            self.register_tool("tavily_search", TavilySearchTool())
        except ImportError as e:
            logger.warning(f"Tavily Search Tool not available: {e}")
        
        try:
            from .document_summarizer import DocumentSummarizerTool
            self.register_tool("document_summarizer", DocumentSummarizerTool())
        except ImportError as e:
            logger.warning(f"Document Summarizer Tool not available: {e}")
        
        # Initialize each tool
        for name, tool in self.tools.items():
            try:
                await tool.initialize()
                logger.info(f"Initialized tool: {name}")
            except Exception as e:
                logger.warning(f"Failed to initialize tool {name}: {str(e)}")
        
        self._initialized = True
        logger.info(f"Tool Registry initialized with {len(self.tools)} tools")
    
    def register_tool(self, name: str, tool: BaseTool) -> None:
        """Register a tool"""
        self.tools[name] = tool
        logger.debug(f"Registered tool: {name}")
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name"""
        return self.tools.get(name)
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tool names"""
        return list(self.tools.keys())
    
    def get_tool_descriptions(self) -> Dict[str, str]:
        """Get descriptions of all tools"""
        return {
            name: tool.get_description()
            for name, tool in self.tools.items()
        }
    
    def get_tool_parameters(self, tool_name: str) -> Dict[str, any]:
        """Get parameters for a specific tool"""
        tool = self.get_tool(tool_name)
        if tool:
            return tool.get_parameters()
        return {}
    
    async def execute_tool(
        self,
        tool_name: str,
        **kwargs
    ) -> ToolResult:
        """Execute a tool by name"""
        tool = self.get_tool(tool_name)
        
        if not tool:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                content=None,
                metadata={},
                error=f"Tool '{tool_name}' not found"
            )
        
        return await tool.execute(**kwargs)
    
    async def shutdown(self) -> None:
        """Shutdown all tools"""
        logger.info("Shutting down Tool Registry...")
        
        for name, tool in self.tools.items():
            try:
                await tool.close()
                logger.debug(f"Closed tool: {name}")
            except Exception as e:
                logger.error(f"Error closing tool {name}: {str(e)}")
        
        self.tools.clear()
        self._initialized = False
        logger.info("Tool Registry shutdown complete")


# Singleton instance
_tool_registry_instance: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    """Get singleton ToolRegistry instance"""
    global _tool_registry_instance
    if _tool_registry_instance is None:
        _tool_registry_instance = ToolRegistry()
    return _tool_registry_instance

