#!/usr/bin/env python3
"""
YAML Component Factory Demo for NanoBrain Framework

Demonstrates how to create steps, agents, and complete workflows from YAML configurations.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append('src')

from src.config import ComponentFactory, ComponentType, get_factory
from nanobrain.core.logging_system import get_logger, set_debug_mode

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = get_logger("yaml_factory_demo")
set_debug_mode(True)


async def demo_single_agent_creation():
    """Demonstrate creating a single agent from YAML configuration."""
    print("\n" + "="*60)
    print("DEMO 1: Creating Single Agent from YAML")
    print("="*60)
    
    factory = get_factory()
    
    try:
        # Load agent from YAML template
        agent = factory.create_from_yaml_file(
            "nanobrain/src/agents/config/step_coder.yml",
            component_name="yaml_code_writer"
        )
        
        print(f"✅ Created agent: {agent.config.name}")
        print(f"   Description: {agent.config.description}")
        print(f"   Model: {agent.config.model}")
        print(f"   Temperature: {agent.config.temperature}")
        
        # Test the agent (if OpenAI API key is available)
        try:
            response = await agent.process("Write a simple hello world function in Python")
            print(f"   Agent response: {response[:100]}...")
        except Exception as e:
            print(f"   Note: Agent test skipped (likely missing API key): {e}")
        
        return agent
        
    except Exception as e:
        print(f"❌ Error creating agent: {e}")
        return None


async def demo_single_step_creation():
    """Demonstrate creating a single step from YAML configuration."""
    print("\n" + "="*60)
    print("DEMO 2: Creating Single Step from YAML")
    print("="*60)
    
    factory = get_factory()
    
    try:
        # Create a file writer step from YAML template
        step = factory.create_from_yaml_file(
            "nanobrain/src/agents/config/step_file_writer.yml",
            component_name="my_file_writer"
        )
        
        print(f"✅ Created step: {step.config.name}")
        print(f"   Description: {step.config.description}")
        print(f"   Input configs: {list(step.config.input_configs.keys()) if step.config.input_configs else 'None'}")
        print(f"   Output config: {step.config.output_config.name if step.config.output_config else 'None'}")
        
        return step
        
    except Exception as e:
        print(f"❌ Error creating step: {e}")
        return None


async def demo_workflow_creation():
    """Demonstrate creating a complete workflow from YAML configuration."""
    print("\n" + "="*60)
    print("DEMO 3: Creating Complete Workflow from YAML")
    print("="*60)
    
    factory = get_factory()
    
    try:
        # Create complete workflow from YAML
        workflow_components = factory.create_workflow_from_yaml(
            "nanobrain/src/config/templates/workflow_example.yml"
        )
        
        print("✅ Created complete workflow with components:")
        
        for name, component in workflow_components.items():
            print(f"     - {name}: {type(component).__name__}")
        
        # Show total component count
        total_components = len(workflow_components)
        print(f"\n   Total components created: {total_components}")
        
        return workflow_components
        
    except Exception as e:
        print(f"❌ Error creating workflow: {e}")
        return None


async def demo_custom_yaml_config():
    """Demonstrate creating components from custom YAML configurations."""
    print("\n" + "="*60)
    print("DEMO 4: Creating Components from Custom YAML")
    print("="*60)
    
    factory = get_factory()
    
    # Custom agent configuration
    custom_agent_config = {
        "name": "CustomAnalyzer",
        "description": "Custom data analysis agent",
        "class": "SimpleAgent",
        "config": {
            "name": "CustomAnalyzer",
            "description": "Analyzes data and provides insights",
            "model": "gpt-3.5-turbo",
            "temperature": 0.3,
            "max_tokens": 2000,
            "system_prompt": "You are a data analysis expert. Analyze data and provide clear insights."
        }
    }
    
    # Custom step configuration
    custom_step_config = {
        "name": "DataProcessingStep",
        "description": "Custom data processing step",
        "class": "SimpleStep",
        "config": {
            "name": "DataProcessingStep",
            "description": "Processes input data",
            "debug_mode": True
        },
        "input_configs": {
            "data": {
                "data_type": "memory",  # Fixed: lowercase
                "name": "data",
                "description": "Input data to process"
            }
        },
        "output_config": {
            "data_type": "memory",  # Fixed: lowercase
            "name": "output",
            "description": "Processed data"
        }
    }
    
    try:
        # Create agent from custom config
        agent = factory.create_component(
            ComponentType.AGENT,
            custom_agent_config,
            "custom_analyzer"
        )
        
        print(f"✅ Created custom agent: {agent.config.name}")
        
        # Create step from custom config
        step = factory.create_component(
            ComponentType.STEP,
            custom_step_config,
            "data_processing_step"
        )
        
        print(f"✅ Created custom step: {step.config.name}")
        
        return {"agent": agent, "step": step}
        
    except Exception as e:
        print(f"❌ Error creating custom components: {e}")
        return None


async def demo_component_registry():
    """Demonstrate the component registry functionality."""
    print("\n" + "="*60)
    print("DEMO 5: Component Registry and Management")
    print("="*60)
    
    factory = get_factory()
    
    # List all created components
    component_names = factory.list_components()
    
    print("📋 Components in registry:")
    for name in component_names:
        component = factory.get_component(name)
        print(f"   - {name}: {type(component).__name__}")
    
    # Get a specific component
    if component_names:
        component_name = component_names[0]
        component = factory.get_component(component_name)
        print(f"\n🔍 Retrieved component '{component_name}':")
        print(f"   Type: {type(component).__name__}")
        print(f"   Config: {getattr(component, 'config', 'No config attribute')}")


async def demo_error_handling():
    """Demonstrate error handling in the factory system."""
    print("\n" + "="*60)
    print("DEMO 6: Error Handling")
    print("="*60)
    
    factory = get_factory()
    
    # Test with non-existent file
    try:
        factory.create_from_yaml_file("non_existent_file.yml")
    except FileNotFoundError as e:
        print(f"✅ Correctly handled missing file: {e}")
    
    # Test with invalid configuration
    try:
        invalid_config = {
            "name": "InvalidAgent",
            "class": "NonExistentAgentClass",
            "config": {}
        }
        factory.create_component(ComponentType.AGENT, invalid_config)
    except Exception as e:
        print(f"✅ Correctly handled invalid config: {e}")
    
    # Test with malformed YAML (simulate by passing invalid dict)
    try:
        malformed_config = {
            "config": {
                "temperature": "not_a_number"  # Should be float
            }
        }
        factory.create_component(ComponentType.AGENT, malformed_config)
    except Exception as e:
        print(f"✅ Correctly handled malformed config: {e}")


async def main():
    """Run all demos."""
    print("🧠 NanoBrain YAML Component Factory Demo")
    print("=" * 60)
    print("This demo shows how to create NanoBrain components from YAML configurations.")
    
    try:
        # Run all demos
        await demo_single_agent_creation()
        await demo_single_step_creation()
        await demo_workflow_creation()
        await demo_custom_yaml_config()
        await demo_component_registry()
        await demo_error_handling()
        
        print("\n" + "="*60)
        print("✅ All demos completed successfully!")
        print("="*60)
        
        # Final component summary
        factory = get_factory()
        component_names = factory.list_components()
        total_components = len(component_names)
        
        print(f"\n📊 Final Summary:")
        print(f"   Total components created: {total_components}")
        print(f"   Components: {component_names}")
        
        # Cleanup
        print("\n🧹 Cleaning up components...")
        factory.shutdown_components()
        print("   All components shutdown successfully.")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\n❌ Demo failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    # Run the demo
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 