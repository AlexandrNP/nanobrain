#!/usr/bin/env python3
"""
LangChain Integration Demo

This demo shows how NanoBrain agents can be used as tools in LangChain agent executors.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.agent import (
    Agent, AgentConfig, SimpleAgent, ConversationalAgent,
    create_langchain_agent_executor, initialize_agents_for_langchain,
    LANGCHAIN_AVAILABLE
)
from config.component_factory import create_component_from_yaml

async def demo_langchain_integration():
    """Demonstrate NanoBrain agents as LangChain tools."""
    print("🔗 LangChain Integration Demo")
    print("=" * 50)
    
    if not LANGCHAIN_AVAILABLE:
        print("❌ LangChain is not available. Install with:")
        print("   pip install langchain langchain-openai")
        return
    
    try:
        # Create specialized NanoBrain agents
        print("\n1. Creating specialized NanoBrain agents...")
        
        # Code Writer Agent
        code_writer_config = AgentConfig(
            name="code_writer",
            description="Specialized agent for writing Python code and scripts",
            model="gpt-3.5-turbo",
            system_prompt="You are a code writing specialist. Generate clean, well-documented Python code."
        )
        code_writer = SimpleAgent(code_writer_config)
        
        # Math Agent
        math_config = AgentConfig(
            name="math_solver",
            description="Specialized agent for solving mathematical problems and calculations",
            model="gpt-3.5-turbo",
            system_prompt="You are a mathematics expert. Solve problems step by step with clear explanations."
        )
        math_agent = SimpleAgent(math_config)
        
        # Text Analyzer Agent
        text_config = AgentConfig(
            name="text_analyzer",
            description="Specialized agent for analyzing and summarizing text content",
            model="gpt-3.5-turbo",
            system_prompt="You are a text analysis expert. Provide detailed analysis and summaries."
        )
        text_agent = SimpleAgent(text_config)
        
        # Initialize all agents
        agents = [code_writer, math_agent, text_agent]
        print(f"✓ Created {len(agents)} specialized agents")
        
        print("\n2. Initializing agents for LangChain...")
        initialized_agents = await initialize_agents_for_langchain(agents)
        print(f"✓ Initialized {len(initialized_agents)} agents")
        
        print("\n3. Creating LangChain agent executor...")
        agent_executor = create_langchain_agent_executor(initialized_agents)
        print("✓ LangChain agent executor created")
        
        print("\n4. Testing LangChain integration...")
        
        # Test 1: Code generation task
        print("\n--- Test 1: Code Generation ---")
        code_request = "Write a Python function that calculates the factorial of a number"
        print(f"Request: {code_request}")
        
        try:
            response = agent_executor.invoke({"input": code_request})
            print(f"✓ Response received: {len(response['output'])} characters")
            print(f"Preview: {response['output'][:100]}...")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        # Test 2: Math problem
        print("\n--- Test 2: Math Problem ---")
        math_request = "Solve the quadratic equation: 2x² + 5x - 3 = 0"
        print(f"Request: {math_request}")
        
        try:
            response = agent_executor.invoke({"input": math_request})
            print(f"✓ Response received: {len(response['output'])} characters")
            print(f"Preview: {response['output'][:100]}...")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        # Test 3: Text analysis
        print("\n--- Test 3: Text Analysis ---")
        text_request = "Analyze this text: 'The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet.'"
        print(f"Request: {text_request[:50]}...")
        
        try:
            response = agent_executor.invoke({"input": text_request})
            print(f"✓ Response received: {len(response['output'])} characters")
            print(f"Preview: {response['output'][:100]}...")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        # Test 4: Multi-step task requiring multiple agents
        print("\n--- Test 4: Multi-Agent Task ---")
        complex_request = "First calculate 15 * 23, then write Python code to verify this calculation"
        print(f"Request: {complex_request}")
        
        try:
            response = agent_executor.invoke({"input": complex_request})
            print(f"✓ Response received: {len(response['output'])} characters")
            print(f"Preview: {response['output'][:100]}...")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        # Cleanup
        print("\n5. Cleaning up...")
        for agent in agents:
            await agent.shutdown()
        print("✓ All agents shut down")
        
        print("\n🎉 LangChain integration demo completed!")
        print("\nKey achievements:")
        print("✓ NanoBrain agents work as LangChain tools")
        print("✓ LangChain agent executor can orchestrate multiple specialized agents")
        print("✓ Seamless integration between frameworks")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


async def demo_simple_langchain_usage():
    """Demonstrate simple usage of NanoBrain agents as LangChain tools."""
    print("\n" + "=" * 50)
    print("🔧 Simple LangChain Usage Demo")
    print("=" * 50)
    
    if not LANGCHAIN_AVAILABLE:
        print("❌ LangChain is not available")
        return
    
    try:
        # Create a single agent
        config = AgentConfig(
            name="helper",
            description="A helpful assistant agent",
            model="gpt-3.5-turbo",
            system_prompt="You are a helpful assistant."
        )
        agent = SimpleAgent(config)
        await agent.initialize()
        
        print("✓ Created and initialized helper agent")
        
        # Test direct usage as LangChain tool
        print("\n--- Testing direct LangChain tool usage ---")
        
        # Test synchronous call
        result = agent._run("What is 2 + 2?")
        print(f"Sync result: {result[:50]}...")
        
        # Test asynchronous call
        result = await agent._arun("What is the capital of France?")
        print(f"Async result: {result[:50]}...")
        
        await agent.shutdown()
        print("✓ Simple usage demo completed")
        
    except Exception as e:
        print(f"❌ Simple demo failed: {e}")


async def main():
    """Main demo function."""
    print("🧠 NanoBrain ↔️ LangChain Integration Demo")
    print("=" * 60)
    print("This demo shows how NanoBrain agents can be used as tools")
    print("in LangChain agent executors for powerful multi-agent workflows.")
    
    # Run demos
    await demo_simple_langchain_usage()
    await demo_langchain_integration()
    
    print("\n" + "=" * 60)
    print("Demo completed! NanoBrain agents are now LangChain-compatible.")


if __name__ == "__main__":
    asyncio.run(main()) 