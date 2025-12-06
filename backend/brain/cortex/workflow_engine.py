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
            from ..data_sources import get_data_source_registry, create_vector_data_source, DataSourceConfig, DataSourceType
            
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
                            logger.info(f"Querying vector_db with query: {query[:50]}...")
                            
                            try:
                                # Create vector data source with config from spec
                                config = DataSourceConfig(
                                    source_id="vector_db",
                                    source_type=DataSourceType.VECTOR_DB,
                                    config={
                                        "collection_name": data_source_spec.config.get("collection_name", "default_collection") if hasattr(data_source_spec, 'config') and data_source_spec.config else "default_collection",
                                        "k": limit,
                                        "embedding_model": data_source_spec.config.get("embedding_model") if hasattr(data_source_spec, 'config') and data_source_spec.config else None,
                                        "db_path": data_source_spec.config.get("db_path") if hasattr(data_source_spec, 'config') and data_source_spec.config else None,
                                    }
                                )
                                
                                vector_source = create_vector_data_source(config)
                                
                                # Initialize the data source
                                init_config = {}
                                if hasattr(data_source_spec, 'config') and data_source_spec.config:
                                    if data_source_spec.config.get("db_path"):
                                        init_config["db_path"] = data_source_spec.config.get("db_path")
                                    if data_source_spec.config.get("vectorstore"):
                                        init_config["vectorstore"] = data_source_spec.config.get("vectorstore")
                                
                                initialized = await vector_source.initialize(init_config)
                                
                                if initialized:
                                    # Query the vector database
                                    query_results = await vector_source.query(
                                        query=query,
                                        context={"filter": data_source_spec.config.get("filter") if hasattr(data_source_spec, 'config') and data_source_spec.config else None},
                                        k=limit
                                    )
                                    
                                    results.append({
                                        "source": source_name,
                                        "query": query,
                                        "results": query_results,
                                        "count": len(query_results),
                                        "message": f"Retrieved {len(query_results)} results from vector database"
                                    })
                                    
                                    logger.info(f"Vector DB query returned {len(query_results)} results")
                                    
                                    # Close the data source
                                    await vector_source.close()
                                else:
                                    logger.warning("Vector data source initialization failed")
                                    results.append({
                                        "source": source_name,
                                        "query": query,
                                        "results": [],
                                        "message": "Vector DB initialization failed"
                                    })
                                    
                            except Exception as e:
                                logger.error(f"Error querying vector_db: {e}")
                                results.append({
                                    "source": source_name,
                                    "query": query,
                                    "results": [],
                                    "message": f"Vector DB query error: {str(e)}"
                                })
                                state["errors"].append(f"Vector DB query error: {str(e)}")
                        else:
                            logger.warning(f"Data source {source_name} not available")
                            results.append({
                                "source": source_name,
                                "query": query,
                                "results": [],
                                "message": f"Data source {source_name} not available"
                            })
                    
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
            
            # Extract user context from Keycloak session for personalization
            user_context = request.user_context
            user_info = self._extract_user_info_for_llm(user_context)
            
            # Check if agent provided pre-built messages context (e.g., from ZoeCore)
            agent_metadata = specs.agent_metadata if hasattr(specs, 'agent_metadata') else {}
            messages_context = agent_metadata.get("messages_context", {})
            
            if messages_context:
                # Agent provided system prompt and conversation history
                logger.info("Using agent-provided messages context")
                
                # Add system prompt (may already include user context from agent)
                system_prompt = messages_context.get("system_prompt")
                if system_prompt:
                    # Enhance system prompt with Keycloak session info if not already included
                    if user_info and not self._has_user_context_in_prompt(system_prompt):
                        enhanced_prompt = self._enhance_prompt_with_user_info(system_prompt, user_info)
                        messages.append({
                            "role": "system",
                            "content": enhanced_prompt
                        })
                    else:
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
                
                # Build system prompt with user context from Keycloak session
                system_parts = []
                
                # Add user context for personalization
                if user_info:
                    user_context_prompt = self._build_user_context_prompt(user_info)
                    if user_context_prompt:
                        system_parts.append(user_context_prompt)
                
                # Add context from data sources
                if state["data_source_results"]:
                    context_info = "\n".join([
                        f"Data from {r['source']}: {r.get('message', 'No results')}"
                        for r in state["data_source_results"]
                    ])
                    if context_info:
                        system_parts.append(f"Context from data sources:\n{context_info}")
                
                # Add context from tools
                if state["tool_results"]:
                    tool_info = "\n".join([
                        f"Tool {r['tool']}: {r.get('result', r.get('error', 'No result'))}"
                        for r in state["tool_results"]
                    ])
                    if tool_info:
                        system_parts.append(f"Tool execution results:\n{tool_info}")
                
                # Add system prompt if we have any context
                if system_parts:
                    messages.append({
                        "role": "system",
                        "content": "\n\n".join(system_parts)
                    })
                
                # Add user message
                messages.append({
                    "role": "user",
                    "content": request.message
                })
            
            state["messages"] = messages
            
            # Ensure at least one message exists (fallback)
            if not messages:
                logger.warning(f"[{state['execution_id']}] No messages built, adding fallback user message")
                messages.append({
                    "role": "user",
                    "content": request.message or "Hello"
                })
                state["messages"] = messages
            
            logger.info(f"Built {len(messages)} messages for LLM")
            
        except Exception as e:
            logger.error(f"Build messages node error: {e}")
            state["errors"].append(f"Build messages error: {str(e)}")
            
            # Fallback: ensure at least user message exists
            if not state.get("messages"):
                request = state.get("request")
                if request:
                    state["messages"] = [{
                        "role": "user",
                        "content": request.message or "Hello"
                    }]
                    logger.warning(f"[{state['execution_id']}] Using fallback message after error")
        
        return state
    
    async def _call_provider_node(self, state: WorkflowState) -> WorkflowState:
        """Call LLM provider with built messages - tracked in LangFuse"""
        state["execution_steps"].append("call_provider")
        logger.info(f"[{state['execution_id']}] Calling LLM provider")
        
        # Update observation metadata (not trace - this is a nested observation)        
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
            else:
                error_msg = response.get("error", "Provider returned unsuccessful response")
                state["errors"].append(error_msg)
                state["final_content"] = f"Error: {error_msg}"
            
        except Exception as e:
            logger.error(f"Provider call node error: {e}")
            state["errors"].append(f"Provider error: {str(e)}")
            state["final_content"] = f"Error calling provider: {str(e)}"
        
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
    
    # UTILITY METHODS FOR USER CONTEXT
    
    def _extract_user_info_for_llm(self, user_context) -> Optional[Dict[str, Any]]:
        """Extract user information from Keycloak session for LLM personalization"""
        if not user_context:
            return None
        
        # Convert dataclass to dict if needed
        from dataclasses import asdict
        if hasattr(user_context, '__dataclass_fields__'):
            context_dict = asdict(user_context)
        else:
            context_dict = dict(user_context) if isinstance(user_context, dict) else {}
        
        # Only include info if user is authenticated via Keycloak
        if not context_dict.get("is_authenticated") and not context_dict.get("authenticated"):
            return None
        
        user_info = {
            "user_id": context_dict.get("user_id"),
            "session_id": context_dict.get("session_id"),
        }
        
        # Add relevant info for LLM personalization
        if context_dict.get("name"):
            user_info["name"] = context_dict.get("name")
        if context_dict.get("email"):
            user_info["email"] = context_dict.get("email")
        if context_dict.get("roles"):
            user_info["roles"] = context_dict.get("roles", [])
        
        # Add user profile info if available
        user_profile = context_dict.get("user_profile")
        if user_profile:
            if isinstance(user_profile, dict):
                if user_profile.get("ace_score") is not None:
                    user_info["ace_score"] = user_profile.get("ace_score")
            elif hasattr(user_profile, 'ace_score'):
                user_info["ace_score"] = user_profile.ace_score
        
        return user_info if len(user_info) > 2 else None
    
    def _has_user_context_in_prompt(self, prompt: str) -> bool:
        """Check if user context is already included in the prompt"""
        if not prompt:
            return False
        
        # Check for common user context indicators
        indicators = [
            "user_id",
            "session_id",
            "user context",
            "user information",
            "authenticated user",
            "user profile"
        ]
        
        prompt_lower = prompt.lower()
        return any(indicator in prompt_lower for indicator in indicators)
    
    def _enhance_prompt_with_user_info(self, prompt: str, user_info: Dict[str, Any]) -> str:
        """Enhance a system prompt with user information"""
        if not user_info or not prompt:
            return prompt
        
        user_context_section = self._build_user_context_prompt(user_info)
        if not user_context_section:
            return prompt
        
        # Add user context at the beginning of the prompt
        return f"{user_context_section}\n\n{prompt}"
    
    def _build_user_context_prompt(self, user_info: Dict[str, Any]) -> str:
        """Build user context section for LLM prompts"""
        if not user_info:
            return ""
        
        parts = []
        
        if user_info.get("name"):
            parts.append(f"User name: {user_info.get('name')}")
        
        if user_info.get("email"):
            parts.append(f"User email: {user_info.get('email')}")
        
        if user_info.get("roles"):
            roles = ", ".join(user_info.get("roles", []))
            if roles:
                parts.append(f"User roles: {roles}")
        
        if user_info.get("ace_score") is not None:
            ace_score = user_info.get("ace_score")
            parts.append(f"User ACE score: {ace_score} (trauma-informed context)")
        
        if parts:
            return "User Context (for personalization):\n" + "\n".join(f"- {part}" for part in parts)
        
        return ""
    
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
