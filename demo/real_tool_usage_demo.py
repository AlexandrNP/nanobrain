#!/usr/bin/env python3
"""
Real Tool Usage Demo for NanoBrain Framework

This demo shows how agents can actually use tools through LLM function calling
and how they integrate with LangChain for broader ecosystem compatibility.
"""

import asyncio
import os
import sys
import json

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
sys.path.insert(0, os.path.join(current_dir, '..', 'src'))

# Import with try/catch for component factory
try:
    from nanobrain.config.component_factory import create_component_from_yaml
except ImportError:
    def create_component_from_yaml(path):
        print(f"⚠️  Component factory not available, skipping YAML loading for {path}")
        return None

from nanobrain.core.agent import AgentConfig, SimpleAgent, ConversationalAgent, create_langchain_agent_executor
from nanobrain.core.tool import ToolConfig, ToolType, FunctionTool

# Set up OpenAI API key for testing
if not os.getenv('OPENAI_API_KEY'):
    print("⚠️  Warning: OPENAI_API_KEY not set. Using mock responses.")
    os.environ['OPENAI_API_KEY'] = 'mock-key-for-testing'

async def create_calculator_tool():
    """Create a calculator tool for mathematical operations."""
    def calculate(expression: str) -> str:
        """Calculate a mathematical expression safely."""
        try:
            # Simple safety check
            allowed_chars = set('0123456789+-*/()., ')
            if not all(c in allowed_chars for c in expression):
                return f"Error: Invalid characters in expression: {expression}"
            
            result = eval(expression)
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    config = ToolConfig(
        tool_type=ToolType.FUNCTION,
        name="calculator",
        description="Calculate mathematical expressions",
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to calculate (e.g., '2 + 3 * 4')"
                }
            },
            "required": ["expression"]
        }
    )
    
    return FunctionTool(calculate, config)

async def create_file_operations_tool():
    """Create a file operations tool."""
    def file_operation(operation: str, filename: str, content: str = "") -> str:
        """Perform file operations."""
        try:
            if operation == "write":
                with open(filename, 'w') as f:
                    f.write(content)
                return f"Successfully wrote to {filename}"
            elif operation == "read":
                if os.path.exists(filename):
                    with open(filename, 'r') as f:
                        content = f.read()
                    return f"Content of {filename}:\n{content}"
                else:
                    return f"File {filename} does not exist"
            elif operation == "list":
                files = os.listdir('.')
                return f"Files in current directory: {', '.join(files)}"
            else:
                return f"Unknown operation: {operation}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    config = ToolConfig(
        tool_type=ToolType.FUNCTION,
        name="file_operations",
        description="Perform file operations like read, write, and list",
        parameters={
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "description": "Operation to perform: 'read', 'write', or 'list'",
                    "enum": ["read", "write", "list"]
                },
                "filename": {
                    "type": "string",
                    "description": "Name of the file to operate on"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write (only for write operation)"
                }
            },
            "required": ["operation", "filename"]
        }
    )
    
    return FunctionTool(file_operation, config)

async def create_weather_tool():
    """Create a mock weather tool."""
    def get_weather(location: str) -> str:
        """Get weather information for a location."""
        # Mock weather data
        weather_data = {
            "san francisco": "Sunny, 72°F",
            "new york": "Cloudy, 65°F",
            "london": "Rainy, 58°F",
            "tokyo": "Partly cloudy, 68°F"
        }
        
        location_lower = location.lower()
        if location_lower in weather_data:
            return f"Weather in {location}: {weather_data[location_lower]}"
        else:
            return f"Weather data not available for {location}. Available locations: {', '.join(weather_data.keys())}"
    
    config = ToolConfig(
        tool_type=ToolType.FUNCTION,
        name="weather",
        description="Get current weather information for a location",
        parameters={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "Location to get weather for"
                }
            },
            "required": ["location"]
        }
    )
    
    return FunctionTool(get_weather, config)

async def demo_simple_agent_with_tools():
    """Demo showing SimpleAgent using tools through LLM function calling."""
    print("\n" + "="*60)
    print("🔧 DEMO: SimpleAgent with Real Tool Usage")
    print("="*60)
    
    # Create agent configuration
    config = AgentConfig(
        name="tool_agent",
        description="Agent that can use various tools",
        model="gpt-3.5-turbo",
        system_prompt="You are a helpful assistant that can use tools to help users. When you need to perform calculations, file operations, or get weather information, use the appropriate tools.",
        debug_mode=True
    )
    
    # Create agent
    agent = SimpleAgent(config)
    
    # Create and register tools
    calc_tool = await create_calculator_tool()
    file_tool = await create_file_operations_tool()
    weather_tool = await create_weather_tool()
    
    agent.register_tool(calc_tool)
    agent.register_tool(file_tool)
    agent.register_tool(weather_tool)
    
    # Initialize agent
    await agent.initialize()
    
    print(f"✓ Agent '{agent.name}' initialized with tools: {agent.available_tools}")
    
    # Test cases
    test_cases = [
        "Calculate 15 * 7 + 23",
        "What's the weather like in San Francisco?",
        "Create a file called 'test.txt' with the content 'Hello, World!'",
        "Read the content of test.txt",
        "Calculate the square root of 144 (use 144**0.5)",
        "List the files in the current directory"
    ]
    
    for i, test_input in enumerate(test_cases, 1):
        print(f"\n--- Test {i}: {test_input} ---")
        try:
            response = await agent.process(test_input)
            print(f"✓ Response: {response}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    await agent.shutdown()

async def demo_conversational_agent_with_tools():
    """Demo showing ConversationalAgent maintaining context while using tools."""
    print("\n" + "="*60)
    print("💬 DEMO: ConversationalAgent with Tool Usage and Context")
    print("="*60)
    
    # Create agent configuration
    config = AgentConfig(
        name="conversational_tool_agent",
        description="Conversational agent that can use tools and maintain context",
        model="gpt-3.5-turbo",
        system_prompt="You are a helpful conversational assistant. Remember our conversation history and use tools when needed.",
        debug_mode=True
    )
    
    # Create agent
    agent = ConversationalAgent(config)
    
    # Create and register tools
    calc_tool = await create_calculator_tool()
    weather_tool = await create_weather_tool()
    
    agent.register_tool(calc_tool)
    agent.register_tool(weather_tool)
    
    # Initialize agent
    await agent.initialize()
    
    print(f"✓ Agent '{agent.name}' initialized with tools: {agent.available_tools}")
    
    # Conversational test sequence
    conversation = [
        "Hi! I'm planning a trip and need some help with calculations.",
        "I have a budget of $1000. If I spend $150 on flights, how much do I have left?",
        "Great! Now if I want to split the remaining amount equally for 5 days, how much per day?",
        "Perfect! Also, what's the weather like in New York? I'm considering going there.",
        "Thanks! Can you remind me how much I have per day for my trip?"
    ]
    
    for i, message in enumerate(conversation, 1):
        print(f"\n--- Turn {i}: {message} ---")
        try:
            response = await agent.process(message)
            print(f"✓ Response: {response}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    await agent.shutdown()

async def demo_yaml_loaded_agent_with_tools():
    """Demo showing agent loaded from YAML configuration with tools."""
    print("\n" + "="*60)
    print("📄 DEMO: YAML-Configured Agent with Tools")
    print("="*60)
    
    try:
        # Load agent from YAML configuration
        agent = create_component_from_yaml("src/agents/config/step_coder.yml")
        
        if not agent:
            print("❌ Failed to load agent from YAML")
            return
        
        # Initialize agent
        await agent.initialize()
        
        print(f"✓ Agent '{agent.name}' loaded from YAML with tools: {agent.available_tools}")
        
        # Test the agent
        test_input = "Create a simple Python function that calculates the factorial of a number"
        print(f"\n--- Test: {test_input} ---")
        
        response = await agent.process(test_input)
        print(f"✓ Response: {response}")
        
        await agent.shutdown()
        
    except Exception as e:
        print(f"❌ Error loading YAML agent: {e}")

async def demo_langchain_integration():
    """Demo showing LangChain integration with NanoBrain agents."""
    print("\n" + "="*60)
    print("🔗 DEMO: LangChain Integration")
    print("="*60)
    
    try:
        # Create multiple specialized agents
        math_config = AgentConfig(
            name="math_specialist",
            description="Specialized agent for mathematical calculations",
            system_prompt="You are a math specialist. Use the calculator tool for any mathematical operations."
        )
        
        weather_config = AgentConfig(
            name="weather_specialist", 
            description="Specialized agent for weather information",
            system_prompt="You are a weather specialist. Use the weather tool to provide weather information."
        )
        
        # Create agents
        math_agent = SimpleAgent(math_config)
        weather_agent = SimpleAgent(weather_config)
        
        # Add tools to agents
        calc_tool = await create_calculator_tool()
        weather_tool = await create_weather_tool()
        
        math_agent.register_tool(calc_tool)
        weather_agent.register_tool(weather_tool)
        
        # Initialize agents
        await math_agent.initialize()
        await weather_agent.initialize()
        
        agents = [math_agent, weather_agent]
        
        print(f"✓ Created {len(agents)} specialized agents")
        for agent in agents:
            print(f"  - {agent.name}: {agent.available_tools}")
        
        # Create LangChain agent executor
        try:
            executor = create_langchain_agent_executor(agents)
            print("✓ LangChain agent executor created")
            
            # Test the executor
            test_queries = [
                "Calculate 25 * 4 + 10",
                "What's the weather in Tokyo?",
                "I need both the result of 100 / 5 and the weather in London"
            ]
            
            for query in test_queries:
                print(f"\n--- LangChain Test: {query} ---")
                try:
                    result = await executor.ainvoke({"input": query})
                    print(f"✓ Result: {result.get('output', result)}")
                except Exception as e:
                    print(f"❌ Error: {e}")
                    
        except Exception as e:
            print(f"⚠️  LangChain integration not available: {e}")
            print("   (This is expected if LangChain is not installed)")
        
        # Shutdown agents
        for agent in agents:
            await agent.shutdown()
            
    except Exception as e:
        print(f"❌ Error in LangChain demo: {e}")

async def demo_agent_as_langchain_tool():
    """Demo showing NanoBrain agent used as a LangChain tool."""
    print("\n" + "="*60)
    print("🛠️  DEMO: NanoBrain Agent as LangChain Tool")
    print("="*60)
    
    try:
        # Create a specialized agent
        config = AgentConfig(
            name="text_processor",
            description="Agent that processes and analyzes text",
            system_prompt="You are a text processing specialist. Analyze text for sentiment, word count, and key themes."
        )
        
        agent = SimpleAgent(config)
        await agent.initialize()
        
        # Convert agent to LangChain tool
        langchain_tool = agent.to_langchain_tool()
        
        if langchain_tool:
            print(f"✓ Agent '{agent.name}' converted to LangChain tool")
            print(f"  Tool name: {langchain_tool.name}")
            print(f"  Tool description: {langchain_tool.description}")
            
            # Test the tool
            test_text = "I love this new framework! It's incredibly powerful and easy to use."
            print(f"\n--- Testing tool with: {test_text} ---")
            
            try:
                result = await langchain_tool._arun(input=test_text)
                print(f"✓ Tool result: {result}")
            except Exception as e:
                print(f"❌ Tool execution error: {e}")
        else:
            print("⚠️  Could not convert agent to LangChain tool (LangChain not available)")
        
        await agent.shutdown()
        
    except Exception as e:
        print(f"❌ Error in agent-as-tool demo: {e}")

async def main():
    """Run all demos."""
    print("🚀 NanoBrain Real Tool Usage Demonstration")
    print("=" * 60)
    print("This demo shows how NanoBrain agents can:")
    print("• Use tools through LLM function calling")
    print("• Maintain conversation context while using tools")
    print("• Load tool configurations from YAML")
    print("• Integrate with LangChain ecosystem")
    print("• Work as LangChain-compatible tools")
    
    try:
        await demo_simple_agent_with_tools()
        await demo_conversational_agent_with_tools()
        await demo_yaml_loaded_agent_with_tools()
        await demo_langchain_integration()
        await demo_agent_as_langchain_tool()
        
        print("\n" + "="*60)
        print("✅ All demos completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 