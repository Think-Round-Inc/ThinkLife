# Zoe Agent - Refactored Structure

## Overview

Zoe is ThinkLife's trauma-informed empathetic AI companion. All LLM processing goes through the centralized CortexFlow architecture.

## Architecture

```
Frontend → ZoeService → ZoeAgent (plugin) → CortexFlow → WorkflowEngine → Provider
```

## File Structure

### Core Files

- **`zoe_core.py`** - Domain logic handler
  - Personality and prompt building
  - Conversation context preparation
  - Response post-processing
  - Safety checks

- **`zoe_service.py`** - Frontend interface
  - Receives requests from frontend/API
  - Coordinates ZoeCore and ZoeAgent plugin
  - Returns formatted responses

- **`personality.py`** - Trauma-informed personality system
  - Empathetic communication patterns
  - Safety checks and redirects
  - Trauma-aware responses

- **`conversation_manager.py`** - Session and conversation management
  - Session creation and tracking
  - Conversation history
  - Session cleanup

### Utilities

- **`helpers.py`** - Utility functions
  - `create_user_context()` - Creates UserContext for BrainRequest

- **`utils/knowledge_index_builder.py`** - Knowledge base index builder
  - Builds ChromaDB index from knowledge files
  - Previously `build_index.py`

### Services

- **`tts_service.py`** - Text-to-speech service
  - Converts text to speech for audio responses

## Usage

```python
from agents.zoe import ZoeService

# Initialize service
service = ZoeService()
await service.initialize()

# Process message
response = await service.process_message(
    message="I'm feeling anxious",
    user_id="user_123",
    user_context={"ace_score": 5}
)
```

## Removed Files

- **`brain_interface.py`** - Deprecated, removed (functionality moved to plugin)

## Key Principles

1. **Separation of Concerns**: Domain logic (ZoeCore) separate from LLM processing (Plugin)
2. **Centralized Processing**: All LLM calls go through CortexFlow
3. **Trauma-Informed**: All responses consider trauma and safety
4. **Clean Interfaces**: Simple, clear APIs for frontend integration

