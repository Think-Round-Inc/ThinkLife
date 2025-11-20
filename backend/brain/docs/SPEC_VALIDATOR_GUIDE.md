# 🛡️ Specification Validator Guide

## Overview

The **Specification Validator** validates execution specifications before Brain executes them. Since plugins communicate directly with Brain Core through specifications (no automatic agent routing), the validator ensures that specifications are valid and executable.

### Purpose

**Previously**: Agent Registry discovered and routed requests to agents  
**Now**: Specification Validator validates specs from plugins before execution

---

## Why the Change?

### Previous Architecture (Agent Registry)

```
User Request → Brain → Agent Registry → Find Best Agent → Execute
```

**Issues**:
- No all-in-one chat interface (each plugin is called directly)
- Agent routing not needed (frontend routes to specific plugins)
- Registry was discovering agents that are never auto-selected
- Overhead without benefit

### Current Architecture (Spec Validator)

```
Plugin → Specifications → Validator → Brain → Execute
```

**Benefits**:
- ✅ Validates specs before execution
- ✅ Catches errors early
- ✅ Provides helpful feedback
- ✅ No unnecessary agent discovery
- ✅ Lightweight and focused

---

## How It Works

### Step 1: Plugin Creates Specifications

```python
# In your plugin (e.g., zoe_agent.py)
async def create_execution_specs(self, request: BrainRequest):
    return AgentExecutionSpec(
        data_sources=[
            DataSourceSpec(
                source_type=DataSourceType.VECTOR_DB,
                config={"db_path": "agents/zoe/chroma_db/chroma.sqlite3"},
                query=request.message,
                limit=5
            )
        ],
        provider=ProviderSpec(
            provider_type="openai",
            model="gpt-4o-mini",
            temperature=0.7
        ),
        processing=ProcessingSpec(
            max_iterations=3,
            timeout_seconds=30.0
        )
    )
```

### Step 2: Brain Validates Specifications

```python
# Brain automatically validates before execution
validation_result = await validator.validate_execution_spec(specs)

if not validation_result.valid:
    # Return error with specific validation issues
    return {"success": False, "errors": validation_result.errors}
```

### Step 3: Brain Executes (if valid)

```python
# If validation passes, Brain executes
response = await brain.execute_agent_request(specs, request, messages)
```

---

## What Gets Validated

### 1. Data Sources ✓

**Checks**:
- Source type is supported
- Query limits are within bounds
- External db_path exists (if specified)
- Required fields are present

**Example Error**:
```
Data source 0: External db_path not found: agents/missing/chroma.sqlite3
```

### 2. Provider ✓

**Checks**:
- Provider type is available (openai, gemini, anthropic)
- Temperature is reasonable (0.0-2.0)
- max_tokens is within limits
- Model is specified

**Example Error**:
```
Provider 'invalid_provider' not available. Available: ['openai', 'gemini', 'anthropic']
```

### 3. Tools ✓

**Checks**:
- Tool type is valid
- Configuration is provided
- Number of tools is reasonable

**Example Warning**:
```
Many tools specified (8), may impact performance
```

### 4. Processing ✓

**Checks**:
- max_iterations within limits (default max: 10)
- timeout_seconds within limits (default max: 300s)
- Values are positive

**Example Error**:
```
max_iterations 100 exceeds limit (10)
```

---

## Validation Modes

### Full Validation (Default)

```python
result = await validator.validate_execution_spec(spec)

# Returns ValidationResult with:
# - valid: bool
# - errors: List[str]
# - warnings: List[str]
# - suggestions: List[str]
```

**Use for**: Development, debugging, detailed feedback

### Strict Validation

```python
result = await validator.validate_execution_spec(spec, strict=True)

# Warnings become errors in strict mode
```

**Use for**: Production, critical paths

### Quick Validation

```python
is_valid = validator.validate_quick(spec)

# Returns: bool (True if executable, False otherwise)
# Skips warnings and suggestions
```

**Use for**: Fast checks, high-throughput scenarios

---

## Validation Results

### ValidationResult Structure

```python
@dataclass
class ValidationResult:
    valid: bool                  # Can Brain execute this?
    errors: List[str]           # Critical issues (prevent execution)
    warnings: List[str]         # Non-critical issues (should review)
    suggestions: List[str]      # Optimization tips
```

### Example Results

#### Valid Specification

```
✅ Specification is valid and executable

💡 Suggestions (1):
  • Data source 0: Using external database at agents/zoe/chroma_db/
```

#### Invalid Specification

```
❌ Specification has errors and cannot be executed

❌ Errors (3):
  • Provider 'invalid_provider' not available
  • max_iterations 100 exceeds limit (10)
  • Data source 0: External db_path not found

⚠️  Warnings (1):
  • Temperature 1.5 is high (typically 0.0-1.0)
```

---

## Configuration Limits

Default limits (configurable):

```python
{
    "max_iterations": 10,          # Maximum processing iterations
    "max_timeout": 300.0,          # Maximum timeout (5 minutes)
    "max_tokens": 4000,            # Maximum token limit
    "max_data_sources": 10,        # Maximum data sources per request
    "max_tools": 5,                # Maximum tools per request
    "max_query_limit": 100         # Maximum query result limit
}
```

### Customizing Limits

```python
# In Brain initialization
brain_config = {
    "max_iterations": 20,    # Increase iteration limit
    "max_timeout": 600.0     # Increase timeout to 10 minutes
}

validator = get_spec_validator()
await validator.initialize(brain_config)
```

---

## Usage Examples

### Example 1: Validate Before Sending to Brain

```python
# In your plugin
async def process_request(self, request: BrainRequest):
    # Create specs
    specs = await self.create_execution_specs(request)
    
    # Validate locally (optional)
    from brain import get_spec_validator
    validator = get_spec_validator()
    result = await validator.validate_execution_spec(specs)
    
    if not result.valid:
        # Handle validation errors
        return AgentResponse(
            success=False,
            content="Invalid configuration",
            metadata={"errors": result.errors}
        )
    
    # Send to Brain (which also validates)
    # ...
```

### Example 2: Get Validation Report

```python
result = await validator.validate_execution_spec(specs)
report = validator.get_validation_report(result)
print(report)
```

### Example 3: Handle Validation Errors

```python
validation_result = await validator.validate_execution_spec(specs)

if not validation_result.valid:
    # Log specific errors
    for error in validation_result.errors:
        logger.error(f"Validation error: {error}")
    
    # Fix common issues
    if "Provider" in str(validation_result.errors):
        # Fallback to default provider
        specs.provider.provider_type = "openai"
```

---

## Benefits

### For Developers

✓ **Early Error Detection** - Catch issues before execution  
✓ **Clear Feedback** - Specific error messages  
✓ **Helpful Suggestions** - Optimization tips  
✓ **Fast Development** - Fail fast with details  

### For the System

✓ **Prevent Invalid Execution** - Don't waste resources on bad specs  
✓ **Better Error Messages** - Users see meaningful errors  
✓ **Resource Protection** - Enforce limits on iterations, timeouts  
✓ **Path Validation** - Ensure external paths exist  

### For Users

✓ **Faster Responses** - Errors caught immediately  
✓ **Clearer Messages** - "Provider not available" vs "Internal error"  
✓ **Better Experience** - No mystery failures  

---

## Comparison: Registry vs Validator

### Agent Registry (Old)

**Purpose**: Discover and route to best agent  
**When**: Request arrives → find agent → execute  
**Use Case**: All-in-one chat interface with automatic routing  

```
Good for:
- Multi-agent systems with auto-selection
- Chat interfaces that pick best agent
- Dynamic agent discovery

Not needed when:
- Frontend routes to specific plugins
- No automatic agent selection
- Direct plugin communication
```

### Specification Validator (New)

**Purpose**: Validate execution specifications  
**When**: Plugin creates specs → validate → execute  
**Use Case**: Direct plugin-to-Brain communication  

```
Good for:
- Direct plugin communication
- Spec validation before execution
- Early error detection
- Configuration validation

Perfect when:
- Each plugin is called directly
- No need for agent routing
- Want fast validation
- Focus on spec correctness
```

---

## Migration from Agent Registry

If you were using Agent Registry, here's what changed:

### Before (Agent Registry)

```python
# Brain automatically discovered and routed
from brain import get_agent_registry

registry = get_agent_registry()
await registry.initialize()  # Discover all plugins

# Brain.process_request() used registry to find agents
```

### After (Spec Validator)

```python
# Brain validates specifications
from brain import get_spec_validator

validator = get_spec_validator()
await validator.initialize()  # Setup validation rules

# Brain.execute_agent_request() validates before executing
```

**What stays the same**:
- Plugin structure (IAgentPlugin, IAgent)
- Specification format (AgentExecutionSpec)
- Brain execution (execute_agent_request)

**What's different**:
- No automatic agent discovery
- No agent routing/selection
- Validation instead of discovery

---

## Best Practices

### 1. Validate in Development

```python
# Always validate during development
result = await validator.validate_execution_spec(specs)
logger.info(validator.get_validation_report(result))
```

### 2. Use Quick Validation in Production

```python
# Fast validation for production
if not validator.validate_quick(specs):
    # Handle error
    pass
```

### 3. Handle External Paths Carefully

```python
# Always use absolute paths for external data sources
import os
DataSourceSpec(
    config={"db_path": os.path.abspath("agents/zoe/chroma_db/")}
)
```

### 4. Set Reasonable Limits

```python
# Don't set iterations too high
ProcessingSpec(
    max_iterations=3,  # Good
    # max_iterations=50,  # Too high
)
```

### 5. Check Warnings

```python
# Warnings are important!
if result.warnings:
    for warning in result.warnings:
        logger.warning(f"Consider fixing: {warning}")
```

---

## Summary

The **Specification Validator** replaces the Agent Registry for systems where:
- Plugins communicate directly with Brain
- No automatic agent routing needed
- Want validation before execution
- Focus on specification correctness

It provides:
- ✅ Spec validation before execution
- ✅ Clear error messages
- ✅ Configuration limit enforcement
- ✅ External path validation
- ✅ Helpful suggestions
- ✅ Lightweight and focused

**Result**: Faster, clearer, more robust specification-based execution!

