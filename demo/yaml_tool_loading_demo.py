#!/usr/bin/env python3
"""
YAML Tool Loading Demo for NanoBrain Framework

This demo showcases how agents can load other agents as tools via YAML configuration
instead of programmatic registration.
"""

import asyncio
import json
import sys
import os

# Add the src directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
sys.path.insert(0, os.path.join(current_dir, '..', 'src'))

from nanobrain.core.agent import AgentConfig
import sys
from pathlib import Path
current_file = Path(__file__) if '__file__' in globals() else Path.cwd() / 'demo' / 'yaml_tool_loading_demo.py'
sys.path.insert(0, str(current_file.parent.parent / "library"))
from agents.specialized import CodeWriterAgent, FileWriterAgent


async def demonstrate_yaml_tool_loading():
    """Demonstrate YAML-based tool loading."""
    print("="*60)
    print("YAML TOOL LOADING DEMONSTRATION")
    print("="*60)
    
    # Create CodeWriterAgent with YAML tool configuration
    code_writer_config = AgentConfig(
        name="CodeWriterAgent",
        description="Generates code and uses file writer tool via YAML config",
        model="gpt-4",
        temperature=0.2,
        tools_config_path="tools.yml",  # This will load tools from YAML
        debug_mode=True,
        enable_logging=True,
        log_conversations=True,
        log_tool_calls=True
    )
    
    code_writer = CodeWriterAgent(code_writer_config)
    
    try:
        # Initialize the agent (this will load tools from YAML)
        print("\nInitializing CodeWriterAgent with YAML tool configuration...")
        await code_writer.initialize()
        
        # Check what tools were loaded
        available_tools = code_writer.available_tools
        print(f"\nTools loaded from YAML configuration: {available_tools}")
        
        # Execute a task that should use the file writer tool
        task = """
        Create a simple Python function that calculates the Fibonacci sequence.
        Save it to a file called 'fibonacci.py' with proper documentation.
        Use the file_writer tool to save the file.
        """
        
        print(f"\nExecuting task: {task}")
        result = await code_writer.execute(task)
        print(f"\nTask result: {result}")
        
        # Get performance statistics
        stats = code_writer.get_performance_stats()
        print(f"\nPerformance Stats:")
        print(json.dumps(stats, indent=2))
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Shutdown the agent
        await code_writer.shutdown()


async def demonstrate_yaml_only_approach():
    """Demonstrate the YAML-only tool loading approach."""
    print("\n" + "="*60)
    print("YAML TOOL LOADING - THE ONLY APPROACH")
    print("="*60)
    
    print("\nYAML-based tool loading is now the only way to configure agent tools.")
    print("Programmatic tool registration methods have been removed.")
    
    # YAML approach
    print("\nYAML APPROACH:")
    
    code_writer_yaml = CodeWriterAgent(AgentConfig(
        name="CodeWriterYAML",
        description="Uses YAML tool configuration",
        tools_config_path="tools.yml",
        debug_mode=True
    ))
    
    try:
        await code_writer_yaml.initialize()
        print(f"Tools loaded from YAML: {code_writer_yaml.available_tools}")
        
        # Verify programmatic methods are gone
        print(f"\nProgrammatic methods removed:")
        print(f"- register_agent_tool: {hasattr(code_writer_yaml, 'register_agent_tool')}")
        print(f"- register_function_tool: {hasattr(code_writer_yaml, 'register_function_tool')}")
        print(f"- register_file_writer_tool: {hasattr(code_writer_yaml, 'register_file_writer_tool')}")
        
    finally:
        await code_writer_yaml.shutdown()


async def main():
    """Main demo function."""
    print("NanoBrain Framework - YAML Tool Loading Demo")
    print("This demo shows how to configure agent tools via YAML instead of code.")
    
    try:
        await demonstrate_yaml_tool_loading()
        await demonstrate_yaml_only_approach()
        
        print("\n" + "="*60)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("="*60)
        print("\nKey Benefits of YAML-Only Tool Loading:")
        print("- Declarative configuration")
        print("- Clean agent interface without programmatic methods")
        print("- Easy to modify tool configurations")
        print("- Better separation of concerns")
        print("- Supports dynamic tool loading")
        print("- Eliminates code complexity from tool registration")
        
    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main()) 