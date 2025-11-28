# ThinkLife Brain Architecture

## 🧠 Overview

The Brain is organized into two core engines that work together to process agent requests:

```
┌─────────────────────────────────────────────────────────────┐
│                      BRAIN CORE                             │
│               (Processes Agent Requests)                    │
└───────────────────┬─────────────────────┬───────────────────┘
                    │                     │
        ┌───────────▼──────────┐  ┌──────▼────────────┐
        │  REASONING ENGINE    │  │ WORKFLOW ENGINE   │
        │  "Thinking"          │  │  "Orchestrator"   │
        └───────────┬──────────┘  └──────┬────────────┘
                    │                     │
        ┌───────────▼─────────────────────▼───────────┐
        │       REGISTRIES (Tools, Data, Providers)   │
        └─────────────────────────────────────────────┘
```

---

## 🎯 Reasoning Engine
**"Brain inside the Brain"**

Uses LLMs to make intelligent decisions about next steps.

### Responsibilities
- 🤔 **Decide next steps** based on context and history
- 🛠️ **Choose tools** appropriate for the task
- 📊 **Select data sources** to query
- 🔄 **Refine plans** based on intermediate results

### Characteristics
- ⚡ **Short-lived** - Per-request lifetime
- 🎯 **Context-driven** - Strongly tied to prompts and context
- 🧠 **LLM-powered** - Uses provider registry for LLM access

### Location
```
backend/brain/reasoning/
├── __init__.py
└── reasoning_engine.py
```

### Usage
```python
from brain.cortex import get_reasoning_engine

reasoning = get_reasoning_engine()
await reasoning.initialize()

# Decide next step
decision = await reasoning.decide_next_step(
    request=brain_request,
    provider_spec=provider_spec,
    context=context,
    execution_history=history
)

# Select tools
tools = await reasoning.select_tools(
    request=brain_request,
    provider_spec=provider_spec,
    available_tools=["tavily_search", "document_summarizer"]
)

# Refine plan
refinement = await reasoning.refine_plan(
    original_request=brain_request,
    provider_spec=provider_spec,
    results_so_far=results,
    remaining_iterations=3
)
```

---

## ⚙️ Workflow Engine
**"Industrial-grade orchestrator"**

Ensures reliable execution with enterprise-grade features.

### Responsibilities
- 🔄 **Retry logic** with exponential backoff
- ⏱️ **Timeout handling** for long-running tasks
- 📅 **Scheduling** and task queuing
- 🔐 **Idempotency** support
- 📊 **State management** (durable state machine)
- 🎯 **DAG execution** with workers

### Characteristics
- 🏗️ **Long-running** - Supports multi-hour workflows
- 🛡️ **Fault-tolerant** - Auto-retry on failures
- 📈 **Scalable** - Worker-based execution
- 💾 **Durable** - State persistence

### Location
```
backend/brain/workflow/
├── __init__.py
└── workflow_engine.py
```

### Usage
```python
from brain.cortex import get_workflow_engine, WorkflowStep

workflow = get_workflow_engine()
await workflow.initialize()

# Define workflow steps
steps = [
    WorkflowStep(
        step_id="reason",
        name="Decide next action",
        action="reason",
        params={"decision": "next_step"},
        max_retries=3,
        timeout=30.0
    ),
    WorkflowStep(
        step_id="query",
        name="Query data",
        action="query_data",
        params={"query": "search term", "limit": 5},
        max_retries=2,
        timeout=15.0
    ),
    WorkflowStep(
        step_id="tool",
        name="Use tool",
        action="use_tool",
        params={"tool_name": "tavily_search", "tool_params": {}},
        max_retries=3,
        timeout=45.0
    )
]

# Execute workflow
execution = await workflow.execute_workflow(
    workflow_name="agent_request_processing",
    steps=steps,
    context={"request_id": "123"},
    idempotency_key="unique-key-123"
)

# Check status
print(execution.status)  # COMPLETED, FAILED, etc.
print(execution.results)  # Results from each step
```

---

## 📋 Registries

Both engines use these registries to access resources:

### Provider Registry
Validates and manages LLM providers (OpenAI, Anthropic, Gemini).

```python
from brain.providers import check_provider_spec_availability

is_valid, errors, info = check_provider_spec_availability(provider_spec)
```

### Tool Registry
Auto-discovers and manages tools.

```python
from brain.tools import get_tool_registry

registry = get_tool_registry()
result = await registry.execute_tool("tavily_search", query="AI research")
```

### Data Source Registry
Manages and queries data sources (Vector DB, etc.).

```python
from brain.data_sources import get_data_source_registry

registry = get_data_source_registry()
results = await registry.query_best("semantic search query", k=5)
```

---

## 🔄 Request Flow

```
1. Plugin sends AgentExecutionSpec to Brain Core
   ↓
2. Brain Core initializes Reasoning + Workflow Engines
   ↓
3. Reasoning Engine decides execution plan
   ↓
4. Workflow Engine executes plan reliably
   ├→ Query data sources (via registry)
   ├→ Use tools (via registry)
   └→ Call LLM providers (via registry)
   ↓
5. Return results to plugin
```

---

## 📁 Directory Structure

```
backend/brain/
├── cortex/                 # Cortex - Central orchestrator + engines
│   ├── __init__.py
│   ├── cortex.py           # Main orchestrator
│   ├── reasoning_engine.py # Reasoning Engine
│   └── workflow_engine.py   # Workflow Engine
│
├── specs/                  # All specifications & types
│   ├── __init__.py
│   ├── core_specs.py       # Brain, request/response, user context
│   ├── provider_specs.py   # Provider types & configs
│   ├── tool_specs.py       # Tool specifications
│   ├── data_source_specs.py # Data source types & interfaces
│   ├── guardrails_specs.py # Security specifications
│   ├── workflow_specs.py   # Workflow types
│   ├── reasoning_specs.py  # Reasoning types
│   └── agent_specs.py      # Agent interfaces & specs
│
├── providers/              # Provider Registry + Implementations
│   ├── provider_registry.py
│   ├── openai.py
│   ├── anthropic.py
│   └── gemini.py
│
├── data_sources/           # Data Source Registry + Connectors
│   ├── data_source_registry.py
│   └── vector_db.py
│
├── tools/                  # Tool Registry + Implementations
│   ├── tool_registry.py
│   ├── base_tool.py
│   └── tavily_search.py
│
└── guardrails/            # Guardrails & Security Management
    └── security_manager.py

```

---

## 🎯 Key Benefits

### Reasoning Engine
✅ **Smart decisions** - LLM-powered reasoning  
✅ **Context-aware** - Uses full conversation history  
✅ **Adaptive** - Can adjust plans based on results

### Workflow Engine
✅ **Reliable** - Automatic retries with backoff  
✅ **Timeout-safe** - No hanging requests  
✅ **Idempotent** - Safe to retry  
✅ **Durable** - Long-running workflow support  
✅ **Observable** - Full execution history

---

## 🚀 Getting Started

```python
from brain import CortexFlow
from brain.specs import AgentExecutionSpec, ProviderSpec, BrainRequest

# Initialize CortexFlow
cortex = CortexFlow()
await brain.initialize()

# Create execution spec from agent
spec = AgentExecutionSpec(
    provider=ProviderSpec(provider_type="openai", model="gpt-4o-mini"),
    data_sources=[...],
    tools=[...],
    processing=ProcessingSpec(max_iterations=3)
)

# Execute through Brain Core
# Reasoning Engine decides what to do
# Workflow Engine ensures reliable execution
result = await brain.execute_agent_request(spec, request, messages)
```

---

## 📖 Philosophy

**Reasoning Engine** = Intelligence  
**Workflow Engine** = Reliability  
**Brain Core** = Orchestration  

Together they provide an **intelligent, reliable, scalable** AI system.

