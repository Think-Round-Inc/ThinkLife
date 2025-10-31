"""
Workflow Engine - LangGraph-based orchestration for agentic execution flow
Handles the iterative loop of provider, data sources, tools, and processing
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, TypedDict, Annotated
from dataclasses import dataclass
from operator import add

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .types import (
    BrainRequest, ProviderSpec, ToolSpec, ProcessingSpec, DataSourceSpec
)
from .providers import create_provider
from .tools import get_tool_registry, ToolRegistry, ToolResult

logger = logging.getLogger(__name__)


class WorkflowState(TypedDict):
    """State for the agentic workflow execution"""
    # Input
    messages: List[Dict[str, str]]
    provider_spec: Optional[Dict[str, Any]]  # Serializable dict instead of ProviderSpec
    processing_spec: Dict[str, Any]  # Serializable dict instead of ProcessingSpec
    context_data: Dict[str, Any]
    tools: List[Dict[str, Any]]  # Serializable dict instead of ToolSpec list
    
    # Execution state
    current_iteration: int
    max_iterations: int
    provider_type: Optional[str]  # Just store provider type, not instance
    enhanced_messages: List[Dict[str, str]]
    tool_results: Dict[str, Any]
    
    # Output
    response: Optional[str]
    success: bool
    error: Optional[str]
    metadata: Dict[str, Any]
    
    # History
    execution_history: Annotated[List[str], add]


@dataclass
class OrchestrationResult:
    """Result of workflow orchestration"""
    success: bool
    content: str
    metadata: Dict[str, Any]
    iterations_used: int
    processing_time: float
    provider_used: str
    error: Optional[str] = None


class WorkflowEngine:
    """
    LangGraph-based workflow engine for agentic execution
    
    Orchestrates the iterative flow:
    1. Initialize Provider
    2. Enhance Context with Data Sources
    3. Apply Tools (if any)
    4. Execute LLM
    5. Evaluate Response
    6. Loop or Return
    
    Uses LangGraph's state management for execution flow
    """
    
    def __init__(self):
        self.providers_cache: Dict[str, Any] = {}
        self.tool_registry: Optional[ToolRegistry] = None
        self.workflow: Optional[StateGraph] = None
        self.checkpointer = MemorySaver()
        self._initialized = False
        
        logger.info("Workflow Engine initialized")
    
    async def initialize(self) -> None:
        """Initialize the workflow engine and build execution graph"""
        if self._initialized:
            return
        
        # Initialize tool registry
        self.tool_registry = get_tool_registry()
        await self.tool_registry.initialize()
        
        # Build LangGraph workflow
        self.workflow = self._build_agentic_workflow()
        
        self._initialized = True
        logger.info("Workflow Engine ready with LangGraph orchestration and tools")
    
    def _build_agentic_workflow(self) -> StateGraph:
        """
        Build the agentic workflow using LangGraph
        
        Flow:
        START → initialize_provider → enhance_context → apply_tools → 
        execute_llm → evaluate_response → [continue/end]
        """
        workflow = StateGraph(WorkflowState)
        
        # Add nodes for each step
        workflow.add_node("initialize_provider", self._initialize_provider_node)
        workflow.add_node("enhance_context", self._enhance_context_node)
        workflow.add_node("apply_tools", self._apply_tools_node)
        workflow.add_node("execute_llm", self._execute_llm_node)
        workflow.add_node("evaluate_response", self._evaluate_response_node)
        
        # Define edges
        workflow.set_entry_point("initialize_provider")
        workflow.add_edge("initialize_provider", "enhance_context")
        workflow.add_edge("enhance_context", "apply_tools")
        workflow.add_edge("apply_tools", "execute_llm")
        workflow.add_edge("execute_llm", "evaluate_response")
        
        # Conditional edge: continue or end
        workflow.add_conditional_edges(
            "evaluate_response",
            self._should_continue,
            {
                "continue": "enhance_context",  # Loop back for next iteration
                "end": END
            }
        )
        
        return workflow.compile(checkpointer=self.checkpointer)
    
    async def orchestrate_request(
        self,
        messages: List[Dict[str, str]],
        provider_spec: Optional[ProviderSpec],
        processing_spec: ProcessingSpec,
        context_data: Dict[str, Any],
        tools: List[ToolSpec] = None
    ) -> OrchestrationResult:
        """
        Orchestrate request using LangGraph workflow
        
        Args:
            messages: Conversation messages
            provider_spec: Provider configuration
            processing_spec: Processing configuration
            context_data: Context from data sources
            tools: Tools to apply (optional)
            
        Returns:
            OrchestrationResult
        """
        start_time = time.time()
        
        try:
            # Initialize workflow state (convert specs to serializable dicts)
            initial_state = WorkflowState(
                messages=messages,
                provider_spec=provider_spec.to_dict() if provider_spec else None,
                processing_spec=processing_spec.__dict__ if hasattr(processing_spec, '__dict__') else processing_spec,
                context_data=context_data,
                tools=[tool.__dict__ if hasattr(tool, '__dict__') else tool for tool in (tools or [])],
                current_iteration=0,
                max_iterations=processing_spec.max_iterations,
                provider_type=provider_spec.provider_type if provider_spec else None,
                enhanced_messages=[],
                tool_results={},
                response=None,
                success=False,
                error=None,
                metadata={},
                execution_history=[]
            )
            
            # Execute workflow
            config = {"configurable": {"thread_id": "workflow_" + str(time.time())}}
            final_state = await self.workflow.ainvoke(initial_state, config)
            
            processing_time = time.time() - start_time
            
            return OrchestrationResult(
                success=final_state.get("success", False),
                content=final_state.get("response", ""),
                metadata={
                    **final_state.get("metadata", {}),
                    "execution_history": final_state.get("execution_history", []),
                    "context_data_size": len(str(context_data)),
                    "tools_applied": len(tools) if tools else 0
                },
                iterations_used=final_state.get("current_iteration", 0),
                processing_time=processing_time,
                provider_used=provider_spec.provider_type if provider_spec else "none"
            )
            
        except Exception as e:
            logger.error(f"Workflow orchestration failed: {str(e)}")
            return OrchestrationResult(
                success=False,
                content="",
                metadata={},
                iterations_used=0,
                processing_time=time.time() - start_time,
                provider_used="unknown",
                error=str(e)
            )
    
    # ============================================================================
    # LangGraph Nodes - Each node is a step in the agentic workflow
    # ============================================================================
    
    async def _initialize_provider_node(self, state: WorkflowState) -> Dict[str, Any]:
        """Node: Initialize or get cached provider"""
        logger.debug("Node: initialize_provider")
        
        provider_spec_dict = state.get("provider_spec")
        
        if not provider_spec_dict:
            return {
                "provider_type": None,
                "execution_history": ["Initialized with no provider"]
            }
        
        provider_type = provider_spec_dict.get("provider_type")
        # Get provider (already cached if available)
        provider = await self._get_provider_from_dict(provider_spec_dict)
        
        return {
            "provider_type": provider_type if provider else None,
            "execution_history": [f"Initialized provider: {provider_type}"]
        }
    
    async def _enhance_context_node(self, state: WorkflowState) -> Dict[str, Any]:
        """Node: Enhance messages with context from data sources"""
        logger.debug(f"Node: enhance_context (iteration {state['current_iteration']})")
        
        messages = state.get("messages", [])
        context_data = state.get("context_data", {})
        
        enhanced = self._enhance_messages_with_context(messages, context_data)
        
        return {
            "enhanced_messages": enhanced,
            "execution_history": [f"Enhanced messages with {len(context_data)} context sources"]
        }
    
    async def _apply_tools_node(self, state: WorkflowState) -> Dict[str, Any]:
        """
        Node: Apply tools based on specifications
        
        Logic:
        - If tools explicitly disabled (enabled=False) → skip
        - If specific tools specified → use those tools
        - If no tools specified → auto-select tools based on query
        """
        logger.debug("Node: apply_tools")
        
        tools_specs = state.get("tools", [])
        messages = state.get("messages", [])
        
        # Check if tools are explicitly disabled
        if tools_specs and len(tools_specs) == 1:
            first_tool = tools_specs[0]
            if isinstance(first_tool, dict) and not first_tool.get("enabled", True):
                return {
                    "tool_results": {},
                    "execution_history": ["Tools explicitly disabled"]
                }
        
        # Determine which tools to use
        if tools_specs and len(tools_specs) > 0:
            # Specific tools specified - use them
            selected_tools = tools_specs
            selection_method = "specified"
        else:
            # No tools specified - auto-select based on query
            selected_tools = await self._auto_select_tools(messages)
            selection_method = "auto-selected"
        
        if not selected_tools:
            return {
                "tool_results": {},
                "execution_history": ["No tools needed for this query"]
            }
        
        # Execute selected tools
        tool_results = await self._execute_tools(selected_tools, messages)
        
        # Add tool results to enhanced messages
        enhanced_messages = state.get("enhanced_messages", [])
        if tool_results:
            enhanced_messages = self._add_tool_results_to_messages(
                enhanced_messages, tool_results
            )
        
        return {
            "tool_results": tool_results,
            "enhanced_messages": enhanced_messages,
            "execution_history": [
                f"{selection_method.capitalize()}: {len(selected_tools)} tools, "
                f"{len([r for r in tool_results.values() if r.get('success')])} successful"
            ]
        }
    
    async def _execute_llm_node(self, state: WorkflowState) -> Dict[str, Any]:
        """Node: Execute LLM with provider"""
        iteration = state.get("current_iteration", 0)
        logger.debug(f"Node: execute_llm (iteration {iteration})")
        
        provider_type = state.get("provider_type")
        if not provider_type:
            return {
                "success": False,
                "error": "No provider available",
                "execution_history": ["LLM execution failed: No provider"]
            }
        
        # Get provider from cache
        provider = self.providers_cache.get(provider_type)
        if not provider:
            return {
                "success": False,
                "error": "Provider not initialized",
                "execution_history": ["LLM execution failed: Provider not in cache"]
            }
        
        enhanced_messages = state.get("enhanced_messages", [])
        provider_spec_dict = state.get("provider_spec", {})
        
        try:
            # Prepare provider parameters
            provider_params = provider_spec_dict.copy()
            provider_params.pop("provider_type", None)
            
            # Execute LLM
            response = await provider.generate_response(
                messages=enhanced_messages,
                **provider_params
            )
            
            if response.get("success") and response.get("content"):
                return {
                    "response": response.get("content", ""),
                    "success": True,
                    "metadata": {
                        **state.get("metadata", {}),
                        **response.get("metadata", {})
                    },
                    "execution_history": [f"LLM executed successfully (iteration {iteration})"]
                }
            else:
                return {
                    "success": False,
                    "error": "Empty or failed response",
                    "execution_history": [f"LLM returned empty response (iteration {iteration})"]
                }
                
        except Exception as e:
            logger.error(f"LLM execution error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "execution_history": [f"LLM error: {str(e)}"]
            }
    
    async def _evaluate_response_node(self, state: WorkflowState) -> Dict[str, Any]:
        """Node: Evaluate response quality and decide if iteration needed"""
        logger.debug("Node: evaluate_response")
        
        response = state.get("response", "")
        current_iteration = state.get("current_iteration", 0)
        max_iterations = state.get("max_iterations", 1)
        success = state.get("success", False)
        
        # Increment iteration counter
        new_iteration = current_iteration + 1
        
        # Check if response is satisfactory
        is_satisfactory = success and len(response) > 10
        
        evaluation_msg = f"Iteration {new_iteration}: "
        if is_satisfactory:
            evaluation_msg += "Response satisfactory"
        else:
            evaluation_msg += "Response needs improvement"
        
        return {
            "current_iteration": new_iteration,
            "execution_history": [evaluation_msg]
        }
    
    def _should_continue(self, state: WorkflowState) -> str:
        """Conditional edge: Determine if workflow should continue or end"""
        
        current_iteration = state.get("current_iteration", 0)
        max_iterations = state.get("max_iterations", 1)
        success = state.get("success", False)
        response = state.get("response", "")
        
        # End conditions
        if current_iteration >= max_iterations:
            logger.debug(f"Ending: Max iterations reached ({max_iterations})")
            return "end"
        
        if success and len(response) > 10:
            logger.debug("Ending: Satisfactory response received")
            return "end"
        
        # Continue for another iteration
        logger.debug(f"Continuing: Iteration {current_iteration}/{max_iterations}")
        return "continue"
    
    # ============================================================================
    # Helper Methods
    # ============================================================================
    
    async def _get_provider_from_dict(self, provider_spec_dict: Dict[str, Any]) -> Optional[Any]:
        """Get or initialize provider from spec dictionary"""
        provider_type = provider_spec_dict.get("provider_type")
        
        # Check cache
        if provider_type in self.providers_cache:
            logger.debug(f"Using cached provider: {provider_type}")
            return self.providers_cache[provider_type]
        
        # Initialize new provider
        try:
            provider_config = provider_spec_dict.copy()
            provider = create_provider(provider_type, provider_config)
            
            if await provider.initialize():
                self.providers_cache[provider_type] = provider
                logger.info(f"Initialized provider: {provider_type}")
                return provider
            else:
                logger.error(f"Failed to initialize provider: {provider_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error initializing provider {provider_type}: {str(e)}")
            return None
    
    def _enhance_messages_with_context(
        self,
        messages: List[Dict[str, str]],
        context_data: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Enhance messages with context from data sources"""
        enhanced = messages.copy()
        context_parts = []
        
        # Add vector DB results
        if "vector_db_results" in context_data:
            results = context_data["vector_db_results"]
            if results:
                context_parts.append("Relevant information:")
                for idx, result in enumerate(results[:5], 1):
                    content = result.get("content", "")
                    if content:
                        context_parts.append(f"{idx}. {content[:200]}...")
        
        # Add external vector DB results
        if "external_vector_db_results" in context_data:
            results = context_data["external_vector_db_results"]
            if results:
                context_parts.append("\nAdditional context:")
                for idx, result in enumerate(results[:3], 1):
                    content = result.get("content", "")
                    if content:
                        context_parts.append(f"{idx}. {content[:200]}...")
        
        # Add conversation history
        if "conversation_history" in context_data:
            history = context_data["conversation_history"]
            if history and len(history) > 0:
                context_parts.append(f"\nConversation history: {len(history)} messages")
        
        # Add context as system message
        if context_parts:
            context_text = "\n".join(context_parts)
            enhanced.insert(0, {
                "role": "system",
                "content": f"Context:\n\n{context_text}"
            })
        
        return enhanced
    
    async def _auto_select_tools(self, messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Auto-select tools based on user query
        
        Logic:
        - Check if query needs web search (current events, facts, research)
        - Check if query needs summarization (long text, summary request)
        """
        if not messages:
            return []
        
        # Get last user message
        user_message = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_message = msg.get("content", "").lower()
                break
        
        if not user_message:
            return []
        
        selected_tools = []
        
        # Keywords that indicate need for web search
        search_keywords = [
            "search", "find", "latest", "current", "news", "recent", 
            "today", "what is", "who is", "research", "look up"
        ]
        
        # Keywords that indicate need for summarization
        summary_keywords = [
            "summarize", "summary", "tldr", "key points", "brief",
            "condense", "main ideas", "overview"
        ]
        
        # Check for search need
        if any(keyword in user_message for keyword in search_keywords):
            selected_tools.append({
                "name": "tavily_search",
                "enabled": True,
                "config": {
                    "max_results": 5,
                    "search_depth": "basic"
                }
            })
            logger.debug("Auto-selected: Tavily Search")
        
        # Check for summarization need
        if any(keyword in user_message for keyword in summary_keywords):
            selected_tools.append({
                "name": "document_summarizer",
                "enabled": True,
                "config": {
                    "summary_type": "brief"
                }
            })
            logger.debug("Auto-selected: Document Summarizer")
        
        return selected_tools
    
    async def _execute_tools(
        self,
        tool_specs: List[Dict[str, Any]],
        messages: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Execute specified tools"""
        if not self.tool_registry:
            logger.warning("Tool registry not initialized")
            return {}
        
        tool_results = {}
        
        # Extract query from messages
        query = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                query = msg.get("content", "")
                break
        
        for tool_spec in tool_specs:
            tool_name = tool_spec.get("name")
            enabled = tool_spec.get("enabled", True)
            config = tool_spec.get("config", {})
            
            if not enabled:
                continue
            
            try:
                logger.info(f"Executing tool: {tool_name}")
                
                # Prepare tool parameters
                params = {"query": query} if query else {}
                params.update(config)
                
                # Execute tool
                result = await self.tool_registry.execute_tool(tool_name, **params)
                
                tool_results[tool_name] = {
                    "success": result.success,
                    "content": result.content,
                    "metadata": result.metadata,
                    "error": result.error
                }
                
                if result.success:
                    logger.info(f"Tool {tool_name} executed successfully")
                else:
                    logger.warning(f"Tool {tool_name} failed: {result.error}")
                
            except Exception as e:
                logger.error(f"Error executing tool {tool_name}: {str(e)}")
                tool_results[tool_name] = {
                    "success": False,
                    "content": None,
                    "metadata": {},
                    "error": str(e)
                }
        
        return tool_results
    
    def _add_tool_results_to_messages(
        self,
        messages: List[Dict[str, str]],
        tool_results: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Add tool results to messages with formatted content"""
        if not tool_results:
            return messages
        
        enhanced = messages.copy()
        results_parts = ["Tool execution results:\n"]
        
        for tool_name, result in tool_results.items():
            if not result.get("success"):
                continue
            
            content = result.get("content")
            
            # Format based on tool type
            if tool_name == "tavily_search":
                results_parts.append(f"\n🔍 Web Search Results:")
                if isinstance(content, dict):
                    # Add AI answer if available
                    if content.get("answer"):
                        results_parts.append(f"\nAnswer: {content['answer']}\n")
                    
                    # Add search results
                    for idx, item in enumerate(content.get("results", [])[:3], 1):
                        results_parts.append(f"{idx}. {item.get('title', 'No title')}")
                        results_parts.append(f"   {item.get('content', '')[:200]}...")
                        results_parts.append(f"   Source: {item.get('url', '')}\n")
            
            elif tool_name == "document_summarizer":
                results_parts.append(f"\n📄 Document Summary:")
                if isinstance(content, str):
                    results_parts.append(f"{content}\n")
            
            else:
                # Generic formatting
                results_parts.append(f"\n{tool_name}: {str(content)[:500]}\n")
        
        if len(results_parts) > 1:  # More than just the header
            results_text = "\n".join(results_parts)
            enhanced.insert(0, {
                "role": "system",
                "content": results_text
            })
        
        return enhanced
    
    async def shutdown(self) -> None:
        """Shutdown workflow engine and clean up providers"""
        logger.info("Shutting down Workflow Engine...")
        
        # Close all cached providers
        for provider_type, provider in self.providers_cache.items():
            try:
                await provider.close()
                logger.debug(f"Closed provider: {provider_type}")
            except Exception as e:
                logger.error(f"Error closing provider {provider_type}: {str(e)}")
        
        self.providers_cache.clear()
        self._initialized = False
        
        logger.info("Workflow Engine shutdown complete")


# Singleton instance
_workflow_engine_instance: Optional[WorkflowEngine] = None


def get_workflow_engine() -> WorkflowEngine:
    """Get singleton WorkflowEngine instance"""
    global _workflow_engine_instance
    if _workflow_engine_instance is None:
        _workflow_engine_instance = WorkflowEngine()
    return _workflow_engine_instance
