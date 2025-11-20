"""
Tool Registry - Generic registry that auto-discovers tools from tool list
To add a new tool: Just add the tool name to the TOOLS list below
"""

import logging
import importlib
from typing import Dict, List, Optional

from .base_tool import BaseTool, ToolResult

logger = logging.getLogger(__name__)


TOOLS = [
    "tavily_search",
    # Add new tool names
]



class ToolRegistry:
    """Generic tool registry - auto-discovers tools from TOOLS list"""
    
    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize all tools by auto-discovery"""
        if self._initialized:
            return
        
        # Auto-discover and register tools from TOOLS list
        for tool_name in TOOLS:
            await self._discover_and_register_tool(tool_name)
        
        self._initialized = True
        logger.info(f"Tool Registry initialized with {len(self.tools)} tools")
    
    async def _discover_and_register_tool(self, tool_name: str) -> None:
        """Auto-discover and register a tool by name"""
        try:
            # Convert tool_name to module name (e.g., "tavily_search" -> "tavily_search")
            module_name = tool_name
            
            # Convert to class name (e.g., "tavily_search" -> "TavilySearchTool")
            class_name = self._tool_name_to_class_name(tool_name)
            
            # Import the module dynamically
            module = importlib.import_module(f".{module_name}", package="brain.tools")
            
            # Get the tool class
            tool_class = getattr(module, class_name)
            
            # Instantiate the tool
            tool_instance = tool_class()
            
            # Initialize the tool
            await tool_instance.initialize()
            
            # Register the tool
            self.tools[tool_name] = tool_instance
            logger.info(f"Auto-registered and initialized tool: {tool_name}")
            
        except ImportError as e:
            logger.warning(f"Tool '{tool_name}' module not found: {e}")
        except AttributeError as e:
            logger.warning(f"Tool class '{class_name}' not found in module '{tool_name}': {e}")
        except Exception as e:
            logger.warning(f"Failed to initialize tool '{tool_name}': {str(e)}")
    
    def _tool_name_to_class_name(self, tool_name: str) -> str:
        """Convert tool_name to expected class name"""
        # Convert snake_case to PascalCase and add "Tool" suffix
        # e.g., "tavily_search" -> "TavilySearchTool"
        parts = tool_name.split('_')
        class_name = ''.join(word.capitalize() for word in parts) + 'Tool'
        return class_name
    
    def register_tool(self, name: str, tool: BaseTool) -> None:
        """Manually register a tool (for custom tools)"""
        self.tools[name] = tool
        logger.debug(f"Manually registered tool: {name}")
    
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
    
    def is_tool_available(self, tool_name: str) -> bool:
        """Check if a tool is available"""
        return tool_name in self.tools
    
    async def execute_tool(self, tool_name: str, **kwargs) -> ToolResult:
        """Execute a tool by name"""
        tool = self.get_tool(tool_name)
        
        if not tool:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                content=None,
                metadata={},
                error=f"Tool '{tool_name}' not found. Available: {self.get_available_tools()}"
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
