#!/usr/bin/env python3
"""
Test Tool Calling Mechanism

This test verifies that tools are actually being called by the LLM.
"""

import asyncio
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.agent import AgentConfig, SimpleAgent
from core.tool import ToolConfig, ToolType, FunctionTool

# Set up OpenAI API key for testing
if not os.getenv('OPENAI_API_KEY'):
    print("⚠️  Warning: OPENAI_API_KEY not set. Using mock responses.")
    os.environ['OPENAI_API_KEY'] = 'mock-key-for-testing'

# Global variable to track tool calls
tool_call_count = 0

async def create_test_tool():
    """Create a test tool that tracks when it's called."""
    global tool_call_count
    
    def test_function(message: str) -> str:
        """Test function that tracks calls."""
        global tool_call_count
        tool_call_count += 1
        return f"TOOL_CALLED: {message} (Call #{tool_call_count})"
    
    config = ToolConfig(
        tool_type=ToolType.FUNCTION,
        name="test_tool",
        description="A test tool that must be called for any user input",
        parameters={
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Message to process"
                }
            },
            "required": ["message"]
        }
    )
    
    return FunctionTool(test_function, config)

async def test_tool_calling():
    """Test if tools are actually being called."""
    global tool_call_count
    tool_call_count = 0
    
    print("🧪 Testing Tool Calling Mechanism")
    print("=" * 50)
    
    # Create agent with strict tool usage requirement
    config = AgentConfig(
        name="test_agent",
        description="Agent for testing tool calling",
        model="gpt-3.5-turbo",
        system_prompt="""You MUST use the test_tool for every user input. 
Never respond without calling the test_tool first. 
Always call the test_tool with the user's message as the parameter.""",
        debug_mode=True  # Enable debug mode to see what's happening
    )
    
    agent = SimpleAgent(config)
    
    # Create and register test tool
    test_tool = await create_test_tool()
    agent.register_tool(test_tool)
    
    await agent.initialize()
    
    print(f"✓ Agent initialized with tools: {agent.available_tools}")
    
    # Test cases
    test_inputs = [
        "Hello, world!",
        "Test message 1",
        "Another test"
    ]
    
    print(f"\nTesting {len(test_inputs)} inputs:")
    
    for i, test_input in enumerate(test_inputs, 1):
        print(f"\n--- Test {i}: {test_input} ---")
        
        initial_count = tool_call_count
        
        try:
            response = await agent.process(test_input)
            print(f"Response: {response}")
            
            if tool_call_count > initial_count:
                print(f"✅ Tool was called! (Total calls: {tool_call_count})")
            else:
                print(f"❌ Tool was NOT called! (Total calls: {tool_call_count})")
                
            if "TOOL_CALLED:" in response:
                print("✅ Tool output detected in response!")
            else:
                print("❌ Tool output NOT detected in response!")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    await agent.shutdown()
    
    print(f"\n📊 Final Results:")
    print(f"Total tool calls: {tool_call_count}")
    print(f"Expected tool calls: {len(test_inputs)}")
    
    if tool_call_count == len(test_inputs):
        print("✅ All tests passed - tools were called correctly!")
    else:
        print("❌ Some tests failed - tools were not called as expected!")

async def test_tool_schemas():
    """Test if tool schemas are properly formatted."""
    print("\n🔍 Testing Tool Schema Generation")
    print("=" * 50)
    
    test_tool = await create_test_tool()
    schema = test_tool.get_schema()
    
    print("Generated tool schema:")
    import json
    print(json.dumps(schema, indent=2))
    
    # Verify schema structure
    required_keys = ["type", "function"]
    function_keys = ["name", "description", "parameters"]
    
    print("\n✅ Schema validation:")
    for key in required_keys:
        if key in schema:
            print(f"  ✓ {key}: present")
        else:
            print(f"  ❌ {key}: missing")
    
    if "function" in schema:
        for key in function_keys:
            if key in schema["function"]:
                print(f"  ✓ function.{key}: present")
            else:
                print(f"  ❌ function.{key}: missing")

async def main():
    """Run all tests."""
    print("🚀 NanoBrain Tool Calling Test Suite")
    print("=" * 50)
    
    try:
        await test_tool_schemas()
        await test_tool_calling()
        
        print("\n" + "=" * 50)
        print("✅ Test suite completed!")
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 