# ThinkxLife Brain v2.0

The central AI orchestration system for ThinkxLife platform with plugin-based architecture and trauma-informed design.

## 🧠 Overview

The ThinkxLife Brain is a generalized AI system that manages all AI operations across the platform. It provides:

- **Plugin-based agent system** with automatic discovery
- **Trauma-informed safety** built into every interaction
- **LangGraph workflow engine** for standardized execution
- **MCP integration** for data source abstraction
- **100% backward compatibility** with existing integrations

## 📁 Structure

```
brain/
├── README.md                 # This file
├── __init__.py              # Main module exports
├── brain_core.py            # Core Brain orchestration system
├── types.py                 # Data structures and type definitions
├── interfaces.py            # Agent contracts and interfaces
├── agent_registry.py        # Plugin discovery and management
├── workflow_engine.py       # LangGraph execution engine
├── data_sources.py          # Data source abstraction layer
├── mcp_integration.py       # Model Context Protocol integration
├── conversation_manager.py  # Conversation history and session management
├── security_manager.py      # Security and rate limiting
├── providers/               # AI provider implementations
│   ├── __init__.py
│   ├── openai.py           # OpenAI provider
│   ├── gemini.py           # Google Gemini provider
│   └── anthropic.py        # Anthropic provider
├── docs/                    # Documentation
│   ├── ARCHITECTURE.md      # System architecture guide
│   └── CREATING_AGENTS.md   # Agent development guide
└── tests/                   # Test files
    └── test_integration.py  # Integration tests

../plugins/                  # Agent plugins (outside brain folder)
├── __init__.py
├── example_agent.py        # Example plugin template
└── zoe_agent.py            # Zoe AI plugin
```

## 🚀 Quick Start

### Basic Usage

```python
from brain import ThinkxLifeBrain

# Initialize with config (same format as before)
brain_config = {
    "providers": {
        "openai": {
            "enabled": True,
            "api_key": "your-api-key",
            "model": "gpt-4o-mini"
        }
    }
}

brain = ThinkxLifeBrain(brain_config)

# Process requests (same API as before)
request = {
    "message": "Hello, I need support",
    "application": "healing-rooms",
    "user_context": {"user_id": "user123", "ace_score": 2}
}

response = await brain.process_request(request)
```

### Health Monitoring

```python
# Check system health
health = await brain.get_health_status()
print(f"System status: {health['overall']}")

# Get analytics
analytics = await brain.get_analytics()
print(f"Total requests: {analytics['total_requests']}")
```

## 🔌 Plugin System

### Quick Agent Creation

1. **Create your agent** in `agents/your_agent/` folder
2. **Create a plugin connector** in `backend/plugins/your_agent_plugin.py`
3. **The Brain automatically discovers and loads your agent!**

For detailed instructions, see the [Creating Agents Guide](docs/CREATING_AGENTS.md).

## 🛡️ Trauma-Informed Features

- **Crisis Detection**: Automatic detection of crisis indicators
- **Safety Filtering**: Trauma-safe language processing
- **Crisis Resources**: Comprehensive crisis support resources
- **Validation**: Empathetic and validating responses

## 🔧 Configuration

### Provider Configuration

```python
{
    "providers": {
        "openai": {
            "enabled": True,
            "api_key": "your-key",
            "model": "gpt-4o-mini",
            "max_tokens": 2000,
            "temperature": 0.7
        },
        "gemini": {
            "enabled": True,
            "api_key": "your-key",
            "model": "gemini-1.5-flash"
        }
    }
}
```

### Security Configuration

Security features are automatically enabled with sensible defaults:
- Rate limiting: 60 requests per minute per user
- Content filtering: Trauma-safe mode enabled
- Input sanitization: XSS and injection protection

## 📊 Monitoring

### Health Endpoints

- System health: `await brain.get_health_status()`
- Analytics: `await brain.get_analytics()`
- Individual agent health: Automatic monitoring

### Metrics Tracked

- Total requests processed
- Success/failure rates
- Response times
- Plugin usage statistics
- Workflow execution counts

## 🧪 Testing

Run integration tests:

```bash
cd backend/brain/tests
python test_integration.py
```

## 📚 Documentation

- **[System Architecture](docs/ARCHITECTURE.md)**: Complete system overview and design principles
- **[Creating Agents](docs/CREATING_AGENTS.md)**: Step-by-step guide to build new agents

## 🔄 Backward Compatibility

The Brain v2.0 maintains 100% backward compatibility:

- ✅ Same constructor signature
- ✅ Same method names and signatures
- ✅ Same request/response formats
- ✅ Same configuration format
- ✅ No changes needed in existing code

## 🎯 Key Benefits

1. **Zero-Code Agent Addition**: Add new agents without touching core code
2. **Trauma-Informed by Design**: Safety built into every interaction
3. **Highly Efficient**: Plugin-based routing with confidence scoring
4. **Scalable**: Modular architecture supports horizontal scaling
5. **Maintainable**: Clear separation of concerns and standardized interfaces

---

**ThinkxLife Brain v2.0 - Empowering ethical AI with trauma-informed care** 💙
