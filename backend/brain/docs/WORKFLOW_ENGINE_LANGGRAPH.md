# 🔄 Workflow Engine - LangGraph-Based Orchestration

## Overview

The Workflow Engine is a **LangGraph-powered orchestration system** that handles the agentic execution loop for all LLM requests in the Brain system.

## Architecture

### LangGraph State Machine

```
START
  ↓
initialize_provider
  ↓
enhance_context
  ↓
apply_tools
  ↓
execute_llm
  ↓
evaluate_response
  ↓
[Decision: continue or end?]
  ├─ continue → back to enhance_context (next iteration)
  └─ end → RETURN result
```

### Key Components

#### 1. **WorkflowState** (Serializable State)
```python
class WorkflowState(TypedDict):
    # Input
    messages: List[Dict[str, str]]
    provider_spec: Optional[Dict[str, Any]]  # Serializable
    processing_spec: Dict[str, Any]           # Serializable
    context_data: Dict[str, Any]
    tools: List[Dict[str, Any]]              # Serializable
    
    # Execution state
    current_iteration: int
    max_iterations: int
    provider_type: Optional[str]              # Not the instance
    enhanced_messages: List[Dict[str, str]]
    tool_results: Dict[str, Any]
    
    # Output
    response: Optional[str]
    success: bool
    error: Optional[str]
    metadata: Dict[str, Any]
    
    # History
    execution_history: Annotated[List[str], add]
```

#### 2. **LangGraph Nodes** (Execution Steps)

- **initialize_provider**: Get or cache LLM provider
- **enhance_context**: Enrich messages with data sources
- **apply_tools**: Execute MCP tools if specified
- **execute_llm**: Generate response with provider
- **evaluate_response**: Check quality and increment iteration

#### 3. **Conditional Edges** (Decision Points)

```python
def _should_continue(state: WorkflowState) -> str:
    if current_iteration >= max_iterations:
        return "end"
    if success and len(response) > 10:
        return "end"
    return "continue"  # Loop back for another iteration
```

## Usage

### From Brain Core

```python
# Brain Core calls Workflow Engine
orchestration_result = await self.workflow_engine.orchestrate_request(
    messages=messages,
    provider_spec=specifications.provider,
    processing_spec=specifications.processing,
    context_data=context_data,
    tools=specifications.tools,
    mcp_manager=self.mcp_manager
)
```

### Direct Usage (Advanced)

```python
from brain import get_workflow_engine, ProviderSpec, ProcessingSpec

workflow = get_workflow_engine()
await workflow.initialize()

result = await workflow.orchestrate_request(
    messages=[{"role": "user", "content": "Hello"}],
    provider_spec=ProviderSpec(
        provider_type="openai",
        model="gpt-4o-mini",
        temperature=0.7
    ),
    processing_spec=ProcessingSpec(
        max_iterations=3,
        timeout_seconds=30.0
    ),
    context_data={}
)
```

## Benefits

### 🔄 **State Management**
- LangGraph handles all state transitions
- Automatic checkpointing with MemorySaver
- Full execution history tracking

### 🔁 **Iterative Processing**
- Configurable max iterations
- Automatic quality evaluation
- Context enhancement between iterations

### 🔌 **Provider Caching**
- Providers initialized once and cached
- Reused across iterations
- Efficient resource management

### 🛠️ **Tool Integration**
- MCP tool application before LLM
- Tool results added to context
- Seamless tool chaining

### 📊 **Execution Tracking**
- Every step logged in execution_history
- Full observability of workflow
- Easy debugging

## Example Execution Flow

```
Input: "Explain LangGraph"
├─ Step 1: Initialize provider: openai
├─ Step 2: Enhanced messages with 0 context sources
├─ Step 3: No tools to apply
├─ Step 4: LLM executed successfully (iteration 0)
├─ Step 5: Iteration 1: Response satisfactory
└─ Output: "LangGraph is a framework that combines..."
```

## Serialization Design

### Why Everything is Serializable?

LangGraph's checkpointing requires all state to be msgpack-serializable. We achieve this by:

1. **Specs → Dicts**: Convert dataclasses to dictionaries
2. **Provider Instance → Provider Type**: Store only the type string
3. **Tools → Dicts**: Serialize tool specifications

### Provider Management

```python
# Provider not stored in state (not serializable)
provider_type: str  # ✅ Serializable

# Provider cached separately in WorkflowEngine
self.providers_cache[provider_type] = provider_instance
```

## Integration Points

### With Brain Core
- Brain validates specs
- Brain queries data sources
- Workflow orchestrates LLM execution
- Result returned to Brain

### With Providers
- Workflow initializes providers
- Cached for reuse
- Closed on shutdown

### With MCP
- Tools applied before LLM
- Results integrated into context
- Full MCP integration support

## Monitoring

```python
result.metadata = {
    "execution_history": [
        "Initialized provider: openai",
        "Enhanced messages with 5 context sources",
        "Applied 2 tools",
        "LLM executed successfully",
        "Iteration 1: Response satisfactory"
    ],
    "context_data_size": 1024,
    "tools_applied": 2
}
```

## Advanced Features

### Custom Workflows (Future)
LangGraph allows extending with custom nodes:
```python
workflow.add_node("custom_processing", my_custom_node)
workflow.add_edge("execute_llm", "custom_processing")
```

### Parallel Tool Execution (Future)
LangGraph supports parallel execution:
```python
workflow.add_conditional_edges(
    "apply_tools",
    tool_selector,
    {
        "tool_a": "execute_tool_a",
        "tool_b": "execute_tool_b"
    }
)
```

## Summary

**Workflow Engine = LangGraph + Agentic Loop + Provider Orchestration**

- ✅ State management via LangGraph
- ✅ Iterative LLM execution
- ✅ Conditional flow control
- ✅ Provider caching
- ✅ Tool integration
- ✅ Full execution tracking

**Result**: Clean, maintainable, and powerful orchestration for all LLM requests!
