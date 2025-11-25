"""
Test script for refactored Zoe architecture

Tests the flow:
Plugin (specs) → Cortex (processing) → ZoeCore (personality) → Response
"""

import asyncio
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from plugins.zoe_agent import ZoeAgent
from brain.specs import BrainRequest, AgentConfig, UserContext, ApplicationType

async def test_zoe_chat():
    """Test Zoe chat flow"""
    print("=" * 60)
    print("Testing Zoe Refactored Architecture")
    print("=" * 60)
    
    # Initialize plugin
    print("\n1. Initializing Zoe Plugin...")
    config = AgentConfig(agent_id="zoe_test")
    zoe_plugin = ZoeAgent(config)
    await zoe_plugin.initialize(config)
    print("✓ Plugin initialized")
    
    # Check plugin specs
    print("\n2. Checking Plugin Specs...")
    print(f"   LLM Provider: {zoe_plugin.llm_specs['provider']}")
    print(f"   LLM Model: {zoe_plugin.llm_specs['model']}")
    print(f"   Temperature: {zoe_plugin.llm_specs['temperature']}")
    print(f"   Data Sources: {len(zoe_plugin.data_sources)}")
    print(f"   Tools: {len(zoe_plugin.tools)}")
    print("✓ Specs loaded")
    
    # Create request
    print("\n3. Creating test request...")
    request = BrainRequest(
        id="test_001",
        application=ApplicationType.CHATBOT,
        message="Hi Zoe, how are you today?",
        user_context=UserContext(
            user_id="test_user",
            session_id="test_session_001",
            is_authenticated=True
        )
    )
    print(f"✓ Request created: '{request.message}'")
    
    # Process request
    print("\n4. Processing request through plugin...")
    print("   Plugin → Cortex → ZoeCore → LLM → ZoeCore → Response")
    
    try:
        response = await zoe_plugin.process_request(request)
        
        print("\n5. Response received:")
        print(f"   Success: {response.success}")
        print(f"   Content: {response.content[:100]}..." if len(response.content) > 100 else f"   Content: {response.content}")
        print(f"   Processing time: {response.processing_time:.2f}s")
        print(f"   Session ID: {response.session_id}")
        
        if response.success:
            print("\n✓ Test PASSED - Zoe responded successfully!")
        else:
            print(f"\n✗ Test FAILED - Error: {response.metadata.get('error')}")
    
    except Exception as e:
        print(f"\n✗ Test FAILED with exception: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Test Complete")
    print("=" * 60)

async def test_conversation():
    """Test multi-turn conversation"""
    print("\n\n" + "=" * 60)
    print("Testing Multi-Turn Conversation")
    print("=" * 60)
    
    # Initialize
    config = AgentConfig(agent_id="zoe_conversation_test")
    zoe_plugin = ZoeAgent(config)
    await zoe_plugin.initialize(config)
    
    session_id = "conversation_session_001"
    messages = [
        "Hi Zoe, I'm feeling stressed today.",
        "Thank you. Can you help me understand why I feel this way?",
        "That makes sense. What can I do about it?"
    ]
    
    for i, message in enumerate(messages, 1):
        print(f"\n{i}. User: {message}")
        
        request = BrainRequest(
            id=f"conv_{i}",
            application=ApplicationType.HEALING_ROOMS,
            message=message,
            user_context=UserContext(
                user_id="conv_user",
                session_id=session_id,
                is_authenticated=True
            )
        )
        
        response = await zoe_plugin.process_request(request)
        
        print(f"   Zoe: {response.content[:150]}..." if len(response.content) > 150 else f"   Zoe: {response.content}")
        
        if not response.success:
            print(f"   Error: {response.metadata.get('error')}")
            break
    
    print("\n✓ Conversation test complete")

if __name__ == "__main__":
    print("\n🤖 Zoe Refactored Architecture Test\n")
    print("Architecture:")
    print("  Plugin: Contains LLM specs (provider, model, params)")
    print("  Cortex: Processes requests using specs")
    print("  ZoeCore: Personality, prompts, conversation")
    print()
    
    # Run tests
    asyncio.run(test_zoe_chat())
    asyncio.run(test_conversation())
    
    print("\n✨ All tests complete!\n")

