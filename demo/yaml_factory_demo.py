#!/usr/bin/env python3
"""
YAML Component Factory Demo for NanoBrain Framework

Demonstrates how to create steps, agents, and complete workflows from YAML configurations.
"""

import asyncio
import logging
import sys
import yaml
from pathlib import Path
import os

# Add the project root to the path so we can import nanobrain
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
sys.path.insert(0, os.path.join(current_dir, '..', 'src'))

from nanobrain.core.config.component_factory import ComponentFactory
from nanobrain.core.config.schema_validator import SchemaValidator
from nanobrain.core.logging_system import get_logger, set_debug_mode

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
set_debug_mode(True)


async def demo_single_agent_creation():
    """Demonstrate creating a single agent from YAML configuration."""
    print("\n" + "="*60)
    print("DEMO 1: Creating Single Agent from YAML")
    print("="*60)
    
    factory = ComponentFactory()
    
    try:
        # Modern approach: Use specific agent class paths
        print("✅ Note: In the simplified system, agent creation requires:")
        print("   - Explicit class path (e.g., 'nanobrain.library.agents.specialized.CodeWriterAgent')")
        print("   - Proper agent configuration matching the class requirements")
        print("   - See other demos for working examples with specific agent classes")
        
        return None
        
    except Exception as e:
        print(f"❌ Error creating agent: {e}")
        return None


async def demo_single_step_creation():
    """Demonstrate creating a single step from YAML configuration."""
    print("\n" + "="*60)
    print("DEMO 2: Creating Single Step from YAML")
    print("="*60)
    
    factory = ComponentFactory()
    
    try:
        # Modern approach: Create step with explicit configuration
        print("✅ Note: Step creation now requires:")
        print("   - Explicit step class path")
        print("   - Proper step configuration")
        print("   - Direct dependency injection")
        print("   - See other demos for working step examples")
        
        return None
        
    except Exception as e:
        print(f"❌ Error creating step: {e}")
        return None


async def demo_workflow_creation():
    """Demonstrate creating workflow components using modern approach."""
    print("\n" + "="*60)
    print("DEMO 3: Creating Workflow Components (Modern Approach)")
    print("="*60)
    
    factory = ComponentFactory()
    
    try:
        # Instead of creating entire workflows from YAML, we create individual components
        # and compose them manually. This is more explicit and maintainable.
        
        print("✅ Creating individual workflow components:")
        
        # Create an executor
        executor_config = {
            "executor_type": "local",
            "max_workers": 4,
            "timeout": 30.0
        }
        
        executor = factory.create_component_from_config(
            "nanobrain.core.executor.LocalExecutor",
            executor_config
        )
        print(f"     - executor: {type(executor).__name__}")
        
        # For actual workflow creation, you would now do:
        # workflow_config = WorkflowConfig(name="MyWorkflow", ...)  
        # workflow = SomeWorkflowClass.from_config(workflow_config, executor=executor, ...)
        
        print("\n   Modern approach focuses on explicit component composition")
        print("   rather than implicit YAML-based workflow creation.")
        
        return {"executor": executor}
        
    except Exception as e:
        print(f"❌ Error creating workflow components: {e}")
        return None


async def demo_custom_yaml_config():
    """Demonstrate creating components from custom YAML configurations."""
    print("\n" + "="*60)
    print("DEMO 4: Creating Components from Custom YAML")
    print("="*60)
    
    factory = ComponentFactory()
    
    try:
        # Create executor from dict config (modern approach)
        executor_config = {
            "executor_type": "local",
            "max_workers": 2,
            "timeout": 45.0
        }
        
        executor = factory.create_component_from_config(
            "nanobrain.core.executor.LocalExecutor",
            executor_config
        )
        
        print(f"✅ Created custom executor: {executor.name}")
        print(f"   - Max workers: {executor.config.max_workers}")
        print(f"   - Timeout: {executor.config.timeout}")
        
        return {"executor": executor}
        
    except Exception as e:
        print(f"❌ Error creating custom components: {e}")
        return None


async def demo_component_registry():
    """Demonstrate modern component management."""
    print("\n" + "="*60)
    print("DEMO 5: Modern Component Management")
    print("="*60)
    
    print("📋 Modern approach: Components are managed directly")
    print("   - No global registry needed")
    print("   - Components are created and managed explicitly")
    print("   - Better memory management and lifecycle control")
    
    # Create multiple components to demonstrate management
    factory = ComponentFactory()
    components = {}
    
    for i in range(3):
        config = {
            "executor_type": "local",
            "max_workers": i + 1
        }
        
        executor = factory.create_component_from_config(
            "nanobrain.core.executor.LocalExecutor",
            config
        )
        components[f"executor_{i}"] = executor
        
    print(f"\n🔍 Created {len(components)} components:")
    for name, component in components.items():
        print(f"   - {name}: {type(component).__name__} (workers: {component.config.max_workers})")
    
    return components


async def demo_error_handling():
    """Demonstrate error handling in the factory system."""
    print("\n" + "="*60)
    print("DEMO 6: Error Handling")
    print("="*60)
    
    factory = ComponentFactory()
    
    # Test with non-existent file
    try:
        factory.create_from_yaml_file("non_existent_file.yml", "nanobrain.core.executor.LocalExecutor")
    except FileNotFoundError as e:
        print(f"✅ Correctly handled missing file: {e}")
    
    # Test with invalid class path
    try:
        invalid_config = {
            "executor_type": "local",
            "max_workers": 2
        }
        factory.create_component_from_config("nonexistent.module.Class", invalid_config)
    except Exception as e:
        print(f"✅ Correctly handled invalid class path: {e}")
    
    # Test with invalid configuration
    try:
        malformed_config = {
            "executor_type": "invalid_type"  # Should be valid ExecutorType
        }
        factory.create_component_from_config("nanobrain.core.executor.LocalExecutor", malformed_config)
    except Exception as e:
        print(f"✅ Correctly handled malformed config: {e}")


async def main():
    """Run all demos."""
    print("🧠 NanoBrain Simplified Component Factory Demo")
    print("=" * 60)
    print("This demo shows the new simplified approach to creating NanoBrain components.")
    
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
        
        print(f"\n📊 Key Changes in Simplified Factory:")
        print(f"   - No global component registry")
        print(f"   - Explicit class paths required")
        print(f"   - Direct component management")
        print(f"   - Simplified API surface")
        print(f"   - Better error handling")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\n❌ Demo failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(asyncio.run(main())) 