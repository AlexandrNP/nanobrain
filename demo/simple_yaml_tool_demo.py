#!/usr/bin/env python3
"""
Simple YAML Tool Loading Demo

This demo shows the basic YAML tool loading functionality without complex prompt templates.
"""

import asyncio
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.agent import Agent, AgentConfig


class SimpleAgent(Agent):
    """Simple agent for demonstration."""
    
    async def process(self, input_text: str, **kwargs) -> str:
        """Simple processing that just echoes the input with available tools."""
        tools_info = f"Available tools: {', '.join(self.available_tools)}"
        return f"Processed: {input_text}\n{tools_info}"


async def main():
    """Main demo function."""
    print("Simple YAML Tool Loading Demo")
    print("="*50)
    
    # Create agent with YAML tool configuration
    config = AgentConfig(
        name="SimpleAgent",
        description="Simple agent with YAML tools",
        tools_config_path="tools.yml",
        debug_mode=True
    )
    
    agent = SimpleAgent(config)
    
    try:
        # Initialize agent (loads tools from YAML)
        print("\nInitializing agent with YAML tool configuration...")
        await agent.initialize()
        
        # Show loaded tools
        print(f"Tools loaded: {agent.available_tools}")
        
        # Process a simple input
        result = await agent.execute("Hello, world!")
        print(f"\nResult: {result}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        await agent.shutdown()
    
    print("\nDemo completed!")


if __name__ == "__main__":
    asyncio.run(main()) 