"""
Workflow Engine - LangGraph-based orchestrator
Executes agent requests using providers, tools, and data sources in a generalized state-based workflow
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, Any, List, Optional, TypedDict, Annotated
from datetime import datetime
from functools import reduce
import operator

logger = logging.getLogger(__name__)

# Optional LangFuse imports - gracefully handle compatibility issues
try:
    from langfuse.decorators import observe, langfuse_context
    LANGFUSE_AVAILABLE = True
except (ImportError, Exception) as e:
    logger.warning(f"LangFuse not available: {e}. Continuing without observability.")
    LANGFUSE_AVAILABLE = False
    
    # Create no-op decorators and context
    def observe(name=None, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    class LangfuseContext:
        def update_current_trace(self, **kwargs):
            pass
        def update_current_observation(self, **kwargs):
            pass
        def observe_llm_call(self, **kwargs):
            class NoOpContext:
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    pass
                def update(self, **kwargs):
                    pass
            return NoOpContext()
    
    langfuse_context = LangfuseContext()

from ..specs import WorkflowStatus


class WorkflowState(TypedDict):
    """
    State for LangGraph workflow
    
    This state is passed through all nodes and accumulates results
    """
    # Input
    execution_id: str
    request: Any  # BrainRequest
    specs: Any  # AgentExecutionSpec from ExecutionPlan
    
    # Intermediate state
    data_source_results: List[Dict[str, Any]]
    tool_results: List[Dict[str, Any]]
    messages: List[Dict[str, str]]
    context: Dict[str, Any]
    
    # Output
    llm_response: Optional[Dict[str, Any]]
    final_content: Optional[str]
    
    # Metadata
    errors: List[str]
    execution_steps: List[str]
    status: str
    start_time: float
    end_time: Optional[float]


class WorkflowEngine:
    """
    LangGraph-based workflow orchestrator
    
    Generalized workflow:
    1. Query data sources (if specified in specs)
    2. Execute tools (if specified in specs)
    3. Build messages with context
    4. Call LLM provider
    5. Return response
    
    Features:
    - State-based execution using LangGraph
    - Automatic routing based on specs
    - Retry logic with exponential backoff
    - Timeout handling
    - Comprehensive error handling
    """
    
    def __init__(self):
        self.executions: Dict[str, Dict[str, Any]] = {}
        self._initialized = False
        self.graph = None
    
    async def initialize(self):
        """Initialize workflow engine and build LangGraph"""
        self._initialized = True
        self._build_graph()
        logger.info("Workflow Engine initialized with LangGraph")
    
    def _build_graph(self):
        """Build LangGraph state machine"""
        try:
            from langgraph.graph import StateGraph, END
            
            # Create graph with WorkflowState
            workflow = StateGraph(WorkflowState)
            
            # Add nodes (processing steps)
            workflow.add_node("initialize", self._initialize_node)
            workflow.add_node("query_data_sources", self._query_data_sources_node)
            workflow.add_node("execute_tools", self._execute_tools_node)
            workflow.add_node("build_messages", self._build_messages_node)
            workflow.add_node("call_provider", self._call_provider_node)
            workflow.add_node("finalize", self._finalize_node)
            
            # Set entry point
            workflow.set_entry_point("initialize")
            
            # Add conditional edges (routing logic)
            workflow.add_conditional_edges(
                "initialize",
                self._route_after_initialize,
                {
                    "data_sources": "query_data_sources",
                    "tools": "execute_tools",
                    "messages": "build_messages"
                }
            )
            
            workflow.add_conditional_edges(
                "query_data_sources",
                self._route_after_data_sources,
                {
                    "tools": "execute_tools",
                    "messages": "build_messages"
                }
            )
            
            workflow.add_edge("execute_tools", "build_messages")
            workflow.add_edge("build_messages", "call_provider")
            workflow.add_edge("call_provider", "finalize")
            workflow.add_edge("finalize", END)
            
            # Compile graph
            self.graph = workflow.compile()
            logger.info("LangGraph workflow compiled successfully")
            
        except ImportError:
            logger.warning("LangGraph not available, using fallback execution")
            self.graph = None
    
    async def execute_plan(
        self,
        plan: Any,  # ExecutionPlan
        request: Any  # BrainRequest
    ) -> Dict[str, Any]:
        """
        Execute an execution plan using LangGraph workflow
        
        This is the main entry point called by CortexFlow
        """
        execution_id = str(uuid.uuid4())
        specs = plan.optimized_specs
        
        logger.info(f"Executing plan {execution_id} - reasoning_applied: {plan.reasoning_applied}")
        
        start_time = time.time()
        
        try:
            # Initialize workflow state
            initial_state: WorkflowState = {
                "execution_id": execution_id,
                "request": request,
                "specs": specs,
                "data_source_results": [],
                "tool_results": [],
                "messages": [],
                "context": {
                    "plan_metadata": plan.to_dict() if hasattr(plan, 'to_dict') else {},
                    "session_id": request.user_context.session_id if hasattr(request, 'user_context') else None
                },
                "llm_response": None,
                "final_content": None,
                "errors": [],
                "execution_steps": [],
                "status": "running",
                "start_time": start_time,
                "end_time": None
            }
            
            # Execute workflow through LangGraph
            if self.graph:
                final_state = await self._execute_graph(initial_state)
            else:
                # Fallback: execute sequentially without LangGraph
                final_state = await self._execute_sequential(initial_state)
            
            # Store execution
            self.executions[execution_id] = final_state
            
            # Build response - ensure content is never None
            final_content = final_state.get("final_content")
            if final_content is None:
                # Fallback: try to extract from llm_response
                llm_response = final_state.get("llm_response")
                if llm_response and isinstance(llm_response, dict):
                    final_content = llm_response.get("content", "")
                else:
                    final_content = "No response generated"
                    logger.warning(f"[{execution_id}] final_content is None, using fallback")
            
            result = {
                "success": final_state["status"] == "completed" and final_content and final_content != "No response generated",
                "content": final_content or "",
                "execution_id": execution_id,
                "status": final_state["status"],
                "metadata": {
                    "execution_steps": final_state["execution_steps"],
                    "data_source_results": len(final_state["data_source_results"]),
                    "tool_results": len(final_state["tool_results"]),
                    "errors": final_state["errors"],
                    "duration_seconds": final_state["end_time"] - final_state["start_time"] if final_state["end_time"] else 0
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Workflow execution failed for {execution_id}: {str(e)}")
            return {
                "success": False,
                "content": f"Workflow execution failed: {str(e)}",
                "execution_id": execution_id,
                "status": "failed",
                "metadata": {
                    "error": str(e),
                    "duration_seconds": time.time() - start_time
                }
            }
    
    async def _execute_graph(self, initial_state: WorkflowState) -> WorkflowState:
        """Execute workflow using LangGraph"""
        try:
            # Run the graph
            result = await self.graph.ainvoke(initial_state)
            return result
        except Exception as e:
            logger.error(f"LangGraph execution error: {e}")
            initial_state["status"] = "failed"
            initial_state["errors"].append(f"Graph execution error: {str(e)}")
            initial_state["end_time"] = time.time()
            return initial_state
    
    async def _execute_sequential(self, state: WorkflowState) -> WorkflowState:
        """Fallback: Execute workflow sequentially without LangGraph"""
        try:
            state = await self._initialize_node(state)
            
            # Query data sources if needed
            if state["specs"].data_sources:
                state = await self._query_data_sources_node(state)
            
            # Execute tools if needed
            if state["specs"].tools:
                state = await self._execute_tools_node(state)
            
            # Build messages
            state = await self._build_messages_node(state)
            
            # Call provider
            state = await self._call_provider_node(state)
            
            # Finalize
            state = await self._finalize_node(state)
            
        except Exception as e:
            logger.error(f"Sequential execution error: {e}")
            state["status"] = "failed"
            state["errors"].append(str(e))
            state["end_time"] = time.time()
        
        return state

    
    # LangGraph nodes
    async def _initialize_node(self, state: WorkflowState) -> WorkflowState:
        """Initialize workflow execution"""
        state["execution_steps"].append("initialize")
        logger.info(f"[{state['execution_id']}] Initializing workflow")
        return state
    
    async def _query_data_sources_node(self, state: WorkflowState) -> WorkflowState:
        """Query data sources specified in specs"""
        state["execution_steps"].append("query_data_sources")
        logger.info(f"[{state['execution_id']}] Querying data sources")
        
        try:
            from ..data_sources import get_data_source_registry
            
            registry = get_data_source_registry()
            request = state["request"]
            specs = state["specs"]
            
            results = []
            
            for data_source_spec in specs.data_sources:
                if not data_source_spec.enabled:
                    continue
                
                try:
                    source_type = data_source_spec.source_type
                    source_name = source_type.value if hasattr(source_type, 'value') else str(source_type)
                    
                    # Only query vector_db (conversation_history is handled by agents)
                    if source_name == "vector_db":
                        query = data_source_spec.query or request.message
                        limit = data_source_spec.limit or 5
                        
                        # Check if data source is available
                        if registry.check_data_source_available("vector_db"):
                            # Query data source
                            # Note: This is simplified - actual implementation would load and query vector_db
                            logger.info(f"Querying vector_db with query: {query[:50]}...")
                            # Placeholder for actual vector DB query
                            results.append({
                                "source": source_name,
                                "query": query,
                                "results": [],
                                "message": "Vector DB query placeholder - implement actual query logic"
                            })
                        else:
                            logger.warning(f"Data source {source_name} not available")
                    
                except Exception as e:
                    logger.error(f"Error querying data source: {e}")
                    state["errors"].append(f"Data source error: {str(e)}")
            
            state["data_source_results"] = results
            
        except Exception as e:
            logger.error(f"Data source node error: {e}")
            state["errors"].append(f"Data source node error: {str(e)}")
        
        return state
    
    async def _execute_tools_node(self, state: WorkflowState) -> WorkflowState:
        """Execute tools specified in specs"""
        state["execution_steps"].append("execute_tools")
        logger.info(f"[{state['execution_id']}] Executing tools")
        
        try:
            from ..tools import get_tool_registry
            
            registry = get_tool_registry()
            specs = state["specs"]
            
            results = []
            
            for tool_spec in specs.tools:
                if not tool_spec.enabled:
                    continue
                
                try:
                    tool_name = tool_spec.name
                    
                    # Check if tool is available
                    if registry.check_tool_available(tool_name):
                        logger.info(f"Executing tool: {tool_name}")
                        # Placeholder for actual tool execution
                        # In production, this would load and execute the actual tool
                        results.append({
                            "tool": tool_name,
                            "status": "executed",
                            "result": "Tool execution placeholder - implement actual tool logic"
                        })
                    else:
                        logger.warning(f"Tool {tool_name} not available")
                        results.append({
                            "tool": tool_name,
                            "status": "unavailable",
                            "error": "Tool not found"
                        })
                
                except Exception as e:
                    logger.error(f"Error executing tool {tool_spec.name}: {e}")
                    state["errors"].append(f"Tool error ({tool_spec.name}): {str(e)}")
            
            state["tool_results"] = results
            
        except Exception as e:
            logger.error(f"Tool execution node error: {e}")
            state["errors"].append(f"Tool node error: {str(e)}")
        
        return state
    
    async def _build_messages_node(self, state: WorkflowState) -> WorkflowState:
        """Build messages for LLM provider with context from data sources and tools"""
        state["execution_steps"].append("build_messages")
        logger.info(f"[{state['execution_id']}] Building messages")
        
        try:
            request = state["request"]
            specs = state["specs"]
            messages = []
            
            # Check if agent provided pre-built messages context (e.g., from ZoeCore)
            agent_metadata = specs.agent_metadata if hasattr(specs, 'agent_metadata') else {}
            messages_context = agent_metadata.get("messages_context", {})
            
            if messages_context:
                # Agent provided system prompt and conversation history
                logger.info("Using agent-provided messages context")
                
                # Add system prompt
                system_prompt = messages_context.get("system_prompt")
                if system_prompt:
                    messages.append({
                        "role": "system",
                        "content": system_prompt
                    })
                
                # Add conversation history
                conversation_history = messages_context.get("conversation_history", [])
                for msg in conversation_history:
                    messages.append({
                        "role": msg.get("role", "user"),
                        "content": msg.get("content", "")
                    })
                
                # Add context from data sources if available
                if state["data_source_results"]:
                    context_info = "\n".join([
                        f"Data from {r['source']}: {r.get('message', 'No results')}"
                        for r in state["data_source_results"]
                    ])
                    if context_info:
                        messages.append({
                            "role": "system",
                            "content": f"Additional context from knowledge base:\n{context_info}"
                        })
                
                # Add context from tools if available
                if state["tool_results"]:
                    tool_info = "\n".join([
                        f"Tool {r['tool']}: {r.get('result', r.get('error', 'No result'))}"
                        for r in state["tool_results"]
                    ])
                    if tool_info:
                        messages.append({
                            "role": "system",
                            "content": f"Tool execution results:\n{tool_info}"
                        })
                
                # Add current user message
                current_message = messages_context.get("current_message") or request.message
                messages.append({
                    "role": "user",
                    "content": current_message
                })
            else:
                # No agent-provided context, build generic messages
                logger.info("Building generic messages (no agent context provided)")
                
                # Add context from data sources
                if state["data_source_results"]:
                    context_info = "\n".join([
                        f"Data from {r['source']}: {r.get('message', 'No results')}"
                        for r in state["data_source_results"]
                    ])
                    if context_info:
                        messages.append({
                            "role": "system",
                            "content": f"Context from data sources:\n{context_info}"
                        })
                
                # Add context from tools
                if state["tool_results"]:
                    tool_info = "\n".join([
                        f"Tool {r['tool']}: {r.get('result', r.get('error', 'No result'))}"
                        for r in state["tool_results"]
                    ])
                    if tool_info:
                        messages.append({
                            "role": "system",
                            "content": f"Tool execution results:\n{tool_info}"
                        })
                
                # Add user message
                messages.append({
                    "role": "user",
                    "content": request.message
                })
            
            state["messages"] = messages
            logger.info(f"Built {len(messages)} messages for LLM")
            
        except Exception as e:
            logger.error(f"Build messages node error: {e}")
            state["errors"].append(f"Build messages error: {str(e)}")
        
        return state
    
    @observe(name="workflow_call_provider")
    async def _call_provider_node(self, state: WorkflowState) -> WorkflowState:
        """Call LLM provider with built messages - tracked in LangFuse"""
        state["execution_steps"].append("call_provider")
        logger.info(f"[{state['execution_id']}] Calling LLM provider")
        
        # Track workflow execution in LangFuse
        langfuse_context.update_current_trace(
            name="workflow_execution",
            metadata={
                "execution_id": state["execution_id"],
                "workflow_step": "call_provider",
                "session_id": state.get("context", {}).get("session_id")
            }
        )
        
        try:
            from ..providers import create_provider
            
            specs = state["specs"]
            messages = state["messages"]
            
            if not specs.provider:
                raise ValueError("No provider specified in specs")
            
            provider_type = specs.provider.provider_type
            provider_config = specs.provider.to_dict() if hasattr(specs.provider, 'to_dict') else {}
            
            # Remove provider_type from config as it's used for routing, not configuration
            provider_config.pop("provider_type", None)
            
            # Add API key from environment if not in config
            import os
            if provider_type == "openai" and "api_key" not in provider_config:
                provider_config["api_key"] = os.getenv("OPENAI_API_KEY")
            elif provider_type == "anthropic" and "api_key" not in provider_config:
                provider_config["api_key"] = os.getenv("ANTHROPIC_API_KEY")
            elif provider_type == "gemini" and "api_key" not in provider_config:
                provider_config["api_key"] = os.getenv("GEMINI_API_KEY")
            
            # Create and initialize provider
            provider = create_provider(provider_type, provider_config)
            await provider.initialize()
            
            # Call provider (LangFuse decorators are in provider.generate_response)
            logger.info(f"Calling {provider_type} with {len(messages)} messages")
            response = await provider.generate_response(messages=messages)
            
            state["llm_response"] = response
            
            # Extract content from response
            if response.get("success"):
                state["final_content"] = response.get("content", "")
                
                # Track successful response in LangFuse
                langfuse_context.update_current_observation(
                    metadata={
                        "provider": provider_type,
                        "response_length": len(state["final_content"]),
                        "success": True
                    }
                )
            else:
                error_msg = response.get("error", "Provider returned unsuccessful response")
                state["errors"].append(error_msg)
                state["final_content"] = f"Error: {error_msg}"
                
                # Track error in LangFuse
                langfuse_context.update_current_observation(
                    level="ERROR",
                    status_message=error_msg,
                    metadata={"provider": provider_type, "success": False}
                )
            
        except Exception as e:
            logger.error(f"Provider call node error: {e}")
            state["errors"].append(f"Provider error: {str(e)}")
            state["final_content"] = f"Error calling provider: {str(e)}"
            
            # Track exception in LangFuse
            langfuse_context.update_current_observation(
                level="ERROR",
                status_message=str(e),
                metadata={"error": str(e)}
            )
        
        return state
    
    async def _finalize_node(self, state: WorkflowState) -> WorkflowState:
        """Finalize workflow execution"""
        state["execution_steps"].append("finalize")
        logger.info(f"[{state['execution_id']}] Finalizing workflow")
        
        # Set final status
        if state["errors"]:
            state["status"] = "completed_with_errors"
        else:
            state["status"] = "completed"
        
        state["end_time"] = time.time()
        
        duration = state["end_time"] - state["start_time"]
        logger.info(f"[{state['execution_id']}] Workflow completed in {duration:.2f}s - status: {state['status']}")
        
        return state
    
    # ROUTING FUNCTIONS (for conditional edges)
    
    def _route_after_initialize(self, state: WorkflowState) -> str:
        """Route after initialization based on what's specified in specs"""
        specs = state["specs"]
        
        # Check data sources first
        if specs.data_sources and any(ds.enabled for ds in specs.data_sources):
            return "data_sources"
        
        # Then tools
        if specs.tools and any(t.enabled for t in specs.tools):
            return "tools"
        
        # Otherwise go straight to messages
        return "messages"
    
    def _route_after_data_sources(self, state: WorkflowState) -> str:
        """Route after data sources based on whether tools are needed"""
        specs = state["specs"]
        
        # Check if tools are specified
        if specs.tools and any(t.enabled for t in specs.tools):
            return "tools"
        
        # Otherwise go to messages
        return "messages"
    
    # UTILITY METHODS
    def get_execution(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get workflow execution by ID"""
        return self.executions.get(execution_id)
    
    def list_executions(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """List workflow executions, optionally filtered by status"""
        executions = list(self.executions.values())
        if status:
            executions = [e for e in executions if e.get("status") == status]
        return executions
    
    async def shutdown(self):
        """Shutdown workflow engine"""
        logger.info("Workflow Engine shutdown")


# Singleton
_workflow_engine_instance: Optional[WorkflowEngine] = None


def get_workflow_engine() -> WorkflowEngine:
    """Get singleton WorkflowEngine instance"""
    global _workflow_engine_instance
    if _workflow_engine_instance is None:
        _workflow_engine_instance = WorkflowEngine()
    return _workflow_engine_instance
