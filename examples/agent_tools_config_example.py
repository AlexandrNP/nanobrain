#!/usr/bin/env python3
"""
Example script for using Agent with tools loaded from config.

This script demonstrates how to create an Agent instance that loads
tools from a YAML configuration file.
"""

import os
import sys
import asyncio

# Add the parent directory to the path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import setup_paths to ensure paths are set up correctly
try:
    import setup_paths
    setup_paths.verify_paths()
except ImportError:
    print("Could not import setup_paths module.")

from src.Agent import Agent
from src.ExecutorBase import ExecutorBase


async def main():
    """
    Demonstrate using Agent with tools loaded from config.
    """
    print("Agent Tools Config Demo")
    print("----------------------")
    
    # Create an ExecutorBase instance
    executor = ExecutorBase()
    
    try:
        # Create an Agent instance with tools loaded from config
        print("\nCreating Agent instance with tools from config:")
        agent = Agent(
            executor=executor,
            model_name="gpt-3.5-turbo",
            tools_config_path="default_configs/AgentTools.yml"
        )
        
        # Print loaded tools
        print(f"\nLoaded {len(agent.tools)} tools:")
        for tool in agent.tools:
            print(f"  - {tool.__class__.__name__}")
        
        # Print tool details
        print(f"\nTool details:")
        for tool in agent.tools:
            print(f"  - {tool.name}: {tool.description}")
        
        print("\nSuccess! All tools loaded correctly.")
        
        # Test processing with tools
        print("\nTesting processing with tools:")
        result = await agent.process_with_tools(["Tell me about the tools you have available."])
        print(f"\nAgent response: {result}")
        
    except Exception as e:
        print(f"Error: {type(e).__name__} - {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main()) 