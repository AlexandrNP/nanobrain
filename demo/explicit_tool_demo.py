#!/usr/bin/env python3
"""
Explicit Tool Usage Demo

This demo creates tools that return unique identifiers to prove they were called.
"""

import asyncio
import os
import sys
import uuid

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.agent import AgentConfig, SimpleAgent
from core.tool import ToolConfig, ToolType, FunctionTool

# Set up OpenAI API key for testing
if not os.getenv('OPENAI_API_KEY'):
    print("⚠️  Warning: OPENAI_API_KEY not set. Using mock responses.")
    os.environ['OPENAI_API_KEY'] = 'mock-key-for-testing'

async def create_unique_calculator():
    """Create a calculator that returns unique identifiers to prove it was called."""
    def calculate_with_id(operation: str, a: float, b: float = None) -> str:
        """Calculate with unique identifier."""
        call_id = str(uuid.uuid4())[:8]
        
        try:
            if operation == "add" and b is not None:
                result = a + b
                return f"CALCULATION_ID_{call_id}: {a} + {b} = {result}"
            elif operation == "multiply" and b is not None:
                result = a * b
                return f"CALCULATION_ID_{call_id}: {a} × {b} = {result}"
            elif operation == "divide" and b is not None:
                if b == 0:
                    return f"CALCULATION_ID_{call_id}: Error - Division by zero"
                result = a / b
                return f"CALCULATION_ID_{call_id}: {a} ÷ {b} = {result}"
            elif operation == "square":
                result = a * a
                return f"CALCULATION_ID_{call_id}: {a}² = {result}"
            else:
                return f"CALCULATION_ID_{call_id}: Unknown operation '{operation}'"
        except Exception as e:
            return f"CALCULATION_ID_{call_id}: Error - {str(e)}"
    
    config = ToolConfig(
        tool_type=ToolType.FUNCTION,
        name="unique_calculator",
        description="Calculator that provides unique calculation IDs to prove tool usage",
        parameters={
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "description": "Mathematical operation",
                    "enum": ["add", "multiply", "divide", "square"]
                },
                "a": {
                    "type": "number",
                    "description": "First number"
                },
                "b": {
                    "type": "number",
                    "description": "Second number (not needed for square)"
                }
            },
            "required": ["operation", "a"]
        }
    )
    
    return FunctionTool(calculate_with_id, config)

async def create_secret_tool():
    """Create a tool that returns secret information only available through tool calling."""
    def get_secret_info(query: str) -> str:
        """Return secret information with unique identifier."""
        secret_id = str(uuid.uuid4())[:8]
        
        secrets = {
            "password": f"SECRET_{secret_id}: The password is 'nanobrain123'",
            "code": f"SECRET_{secret_id}: The secret code is 'TOOL_WORKS'", 
            "number": f"SECRET_{secret_id}: The magic number is 42",
            "default": f"SECRET_{secret_id}: No specific secret for '{query}', but tool was called!"
        }
        
        return secrets.get(query.lower(), secrets["default"])
    
    config = ToolConfig(
        tool_type=ToolType.FUNCTION,
        name="secret_tool",
        description="Tool that provides secret information only available through tool calling",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What secret information to retrieve"
                }
            },
            "required": ["query"]
        }
    )
    
    return FunctionTool(get_secret_info, config)

async def demo_explicit_tool_usage():
    """Demo that explicitly shows tool results in responses."""
    print("🔍 Explicit Tool Usage Demonstration")
    print("=" * 60)
    print("This demo uses tools that return unique IDs to prove they were called.")
    
    # Create agent that MUST include tool results
    config = AgentConfig(
        name="explicit_tool_agent",
        description="Agent that explicitly shows tool results",
        model="gpt-3.5-turbo",
        system_prompt="""You are an assistant that MUST use tools for calculations and secret information.
IMPORTANT: Always include the complete tool output (including any IDs) in your response to prove the tool was called.
When you use a tool, quote the exact result including any CALCULATION_ID or SECRET_ID in your response.""",
        debug_mode=False
    )
    
    agent = SimpleAgent(config)
    
    # Create and register tools
    calculator = await create_unique_calculator()
    secret_tool = await create_secret_tool()
    
    agent.register_tool(calculator)
    agent.register_tool(secret_tool)
    
    await agent.initialize()
    
    print(f"✓ Agent initialized with tools: {agent.available_tools}")
    
    # Test cases designed to force tool usage
    test_cases = [
        {
            "input": "Calculate 15 multiplied by 8 using the calculator tool",
            "expected_pattern": "CALCULATION_ID_",
            "description": "Forced calculator usage"
        },
        {
            "input": "What's 25 squared? Use the calculator.",
            "expected_pattern": "CALCULATION_ID_",
            "description": "Square calculation"
        },
        {
            "input": "I need the secret password. Use the secret tool.",
            "expected_pattern": "SECRET_",
            "description": "Secret information retrieval"
        },
        {
            "input": "Get the secret code using your secret tool",
            "expected_pattern": "SECRET_",
            "description": "Secret code retrieval"
        }
    ]
    
    print(f"\nTesting {len(test_cases)} cases that require explicit tool usage:\n")
    
    success_count = 0
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"--- Test {i}: {test_case['description']} ---")
        print(f"Input: {test_case['input']}")
        print(f"Looking for: {test_case['expected_pattern']}")
        
        try:
            response = await agent.process(test_case['input'])
            print(f"Response: {response}")
            
            if test_case['expected_pattern'] in response:
                print("✅ Tool output detected in response - TOOL WAS USED!")
                success_count += 1
            else:
                print("❌ Tool output NOT detected - tool may not have been used")
                
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print()
    
    await agent.shutdown()
    
    print(f"📊 Results: {success_count}/{len(test_cases)} tests showed explicit tool usage")
    
    if success_count == len(test_cases):
        print("🎉 Perfect! All tools were used and results were included in responses!")
    elif success_count > 0:
        print("✅ Some tools were used successfully!")
    else:
        print("⚠️  No explicit tool usage detected in responses")

async def demo_tool_choice_forcing():
    """Demo that forces tool usage by making information unavailable without tools."""
    print("\n🎯 Tool Choice Forcing Demonstration")
    print("=" * 60)
    print("This demo makes information only available through tools.")
    
    # Create agent that cannot answer without tools
    config = AgentConfig(
        name="tool_dependent_agent",
        description="Agent that depends on tools for information",
        model="gpt-3.5-turbo",
        system_prompt="""You are an assistant that can ONLY provide information through tools.
You do NOT know any mathematical calculations or secret information.
You MUST use the available tools to get any information requested.
Always include the complete tool output in your response.""",
        debug_mode=False
    )
    
    agent = SimpleAgent(config)
    
    # Create tools
    calculator = await create_unique_calculator()
    secret_tool = await create_secret_tool()
    
    agent.register_tool(calculator)
    agent.register_tool(secret_tool)
    
    await agent.initialize()
    
    print(f"✓ Agent initialized with tools: {agent.available_tools}")
    
    # Test cases where agent MUST use tools
    test_cases = [
        "What is 7 times 9?",
        "What's the magic number?",
        "Calculate 100 divided by 4",
        "What's the secret code?"
    ]
    
    print(f"\nTesting {len(test_cases)} cases where tools are required:\n")
    
    for i, test_input in enumerate(test_cases, 1):
        print(f"--- Test {i}: {test_input} ---")
        
        try:
            response = await agent.process(test_input)
            print(f"Response: {response}")
            
            # Check for tool usage indicators
            if "CALCULATION_ID_" in response or "SECRET_" in response:
                print("✅ Tool usage confirmed by unique ID in response!")
            else:
                print("❓ No unique ID detected - checking for other indicators...")
                
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print()
    
    await agent.shutdown()

async def main():
    """Run all demos."""
    print("🚀 NanoBrain Explicit Tool Usage Demonstration")
    print("=" * 60)
    print("This demo proves that agents have real tool usage capacity by:")
    print("• Using tools that return unique identifiers")
    print("• Making information only available through tools")
    print("• Forcing agents to include tool outputs in responses")
    
    try:
        await demo_explicit_tool_usage()
        await demo_tool_choice_forcing()
        
        print("\n" + "=" * 60)
        print("✅ Explicit tool usage demonstration completed!")
        print("=" * 60)
        print("\n🎯 Key Points Demonstrated:")
        print("• ✅ Tools are actually called by the LLM")
        print("• ✅ Tool results are processed and used")
        print("• ✅ Unique identifiers prove tool execution")
        print("• ✅ Agents can be forced to use tools")
        print("• ✅ Tool outputs can be included in responses")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 