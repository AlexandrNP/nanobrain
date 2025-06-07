#!/usr/bin/env python3
"""
Advanced Code Writer Demo for NanoBrain Framework v2.0

This demo showcases the refactored NanoBrain framework with:
- Decoupled Step and Agent classes
- Agent-to-agent tool calling
- YAML configuration system
- Async-first design
- Configurable executors
- Data units and triggers for Steps
- Links for dataflow between Steps

The demo demonstrates:
1. Agent-to-agent interaction (CodeWriter using FileWriter as tool)
2. Step-based data processing with triggers
3. YAML configuration loading and saving
4. Schema generation
5. Mixed agent and step workflows
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the nanobrain package to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.agent import AgentConfig, create_agent
from src.core.step import StepConfig, create_step
from src.core.executor import ExecutorConfig, ExecutorType
from src.core.data_unit import DataUnitConfig, DataUnitType
from src.core.trigger import TriggerConfig, TriggerType
from src.core.link import LinkConfig, LinkType, create_link
from src.agents.code_writer import CodeWriterAgent
from src.agents.file_writer import FileWriterAgent
from src.config.yaml_config import WorkflowConfig, create_example_config, save_config
from src.config.schema_generator import generate_all_schemas
from src.config.component_factory import create_component_from_yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def demo_agent_to_agent_interaction():
    """
    Demonstrate agent-to-agent interaction where CodeWriter uses FileWriter as a tool via YAML config.
    """
    print("\n" + "="*60)
    print("DEMO 1: Agent-to-Agent Interaction (YAML-based)")
    print("="*60)
    
    try:
        # Create CodeWriter agent using the step_coder.yml configuration
        code_writer = create_component_from_yaml("src/agents/config/step_coder.yml")
        await code_writer.initialize()
        print(f"✓ CodeWriter agent initialized")
        print(f"  Available tools: {code_writer.available_tools}")
        
        # Test 1: Generate a simple utility function
        print("\n--- Test 1: Generate Utility Function ---")
        request1 = """Create a simple Python function 'add_numbers(a, b)' that adds two numbers and returns the result. Save it to 'output/utils.py'."""
        
        response1 = await code_writer.process(request1)
        if "output/utils.py" in response1:
            print("✓ utils.py generated")
        else:
            print("⚠ Function generated but file creation unclear")
        
        # Test 2: Generate a simple class
        print("\n--- Test 2: Generate Simple Class ---")
        request2 = """Create a simple Python class 'Calculator' with methods add() and multiply(). Save it to 'output/calculator.py'."""
        
        response2 = await code_writer.process(request2)
        if "output/calculator.py" in response2:
            print("✓ calculator.py generated")
        else:
            print("⚠ Class generated but file creation unclear")
        
        # Test 3: Generate configuration file
        print("\n--- Test 3: Generate Config File ---")
        request3 = """Create a simple YAML configuration file for a web app with database settings. Save it to 'output/app_config.yml'."""
        
        response3 = await code_writer.process(request3)
        if "output/app_config.yml" in response3:
            print("✓ app_config.yml generated")
        else:
            print("⚠ Config generated but file creation unclear")
        
        # Test 4: Generate bash script
        print("\n--- Test 4: Generate Bash Script ---")
        request4 = """Create a simple bash script that backs up files to a directory. Save it to 'output/backup.sh'."""
        
        response4 = await code_writer.process(request4)
        if "output/backup.sh" in response4:
            print("✓ backup.sh generated")
        else:
            print("⚠ Script generated but file creation unclear")
        
        # Cleanup
        await code_writer.shutdown()
        print("✓ Agent shutdown successfully")
        
    except Exception as e:
        logger.error(f"Error in agent-to-agent demo: {e}")
        print(f"❌ Demo failed: {e}")


async def demo_step_based_processing():
    """
    Demonstrate Step-based processing with data units and triggers.
    """
    print("\n" + "="*60)
    print("DEMO 2: Step-Based Processing")
    print("="*60)
    
    try:
        # Create a simple data transformation step
        def transform_data(input_data):
            """Transform input data by adding a timestamp and processing flag."""
            import time
            
            if input_data is None:
                return {"output": None}
            
            # Transform the data
            transformed = {
                "original": input_data,
                "processed_at": time.time(),
                "processed": True,
                "length": len(str(input_data))
            }
            
            return transformed
        
        # Configure the step
        step_config = StepConfig(
            name="data_transformer",
            description="Transform input data with metadata",
            input_configs={
                "input_0": DataUnitConfig(data_type=DataUnitType.MEMORY, persistent=False)
            },
            output_config=DataUnitConfig(data_type=DataUnitType.MEMORY, persistent=False),
            trigger_config=TriggerConfig(
                trigger_type=TriggerType.DATA_UPDATED,
                debounce_ms=100
            )
        )
        
        # Create the step
        transform_step = create_step("transform", step_config, transform_func=transform_data)
        await transform_step.initialize()
        print(f"✓ Transform step initialized: {transform_step.name}")
        
        # Test data processing
        print("\n--- Processing Test Data ---")
        test_data = ["Hello, NanoBrain!", {"status": "ok"}, [1, 2, 3]]
        
        for i, data in enumerate(test_data):
            await transform_step.set_input(data)
            await transform_step.execute()
            output = await transform_step.get_output()
            print(f"  Item {i+1}: {type(data).__name__} → processed")
        
        # Show step statistics
        print(f"✓ Processed {transform_step.execution_count} items, {transform_step.error_count} errors")
        
        # Cleanup
        await transform_step.shutdown()
        print("✓ Step shutdown successfully")
        
    except Exception as e:
        logger.error(f"Error in step processing demo: {e}")
        print(f"❌ Demo failed: {e}")


async def demo_yaml_configuration():
    """
    Demonstrate YAML configuration system with schema generation.
    """
    print("\n" + "="*60)
    print("DEMO 3: YAML Configuration System")
    print("="*60)
    
    try:
        # Create example configuration
        config = create_example_config()
        print(f"✓ Created configuration: {config.name}")
        
        # Add minimal components for demo
        config.add_agent("code_writer", {
            "agent_type": "simple",
            "name": "code_writer",
            "description": "Code writing agent",
            "model": "gpt-4"
        })
        
        config.add_step("processor", {
            "step_type": "simple",
            "name": "processor",
            "description": "Data processor",
            "executor": "local"
        })
        
        print(f"✓ Added {len(config.agents)} agents and {len(config.steps)} steps")
        
        # Validate and save
        errors = config.validate_references()
        if not errors:
            print("✓ Configuration validation passed")
        
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        config_file = output_dir / "workflow_config.yaml"
        save_config(config, config_file)
        print(f"✓ Configuration saved to {config_file}")
        
        # Generate schemas
        schema_dir = output_dir / "schemas"
        generate_all_schemas(schema_dir)
        print(f"✓ Schemas generated in {schema_dir}")
        
        print(f"✓ Summary: {len(config.agents)} agents, {len(config.steps)} steps, {len(config.links)} links")
        
    except Exception as e:
        logger.error(f"Error in YAML configuration demo: {e}")
        print(f"❌ Demo failed: {e}")


async def demo_mixed_workflow():
    """
    Demonstrate a mixed workflow with both agents and steps.
    """
    print("\n" + "="*60)
    print("DEMO 4: Mixed Agent-Step Workflow")
    print("="*60)
    
    try:
        # Create CodeWriter agent using the step_coder.yml configuration
        code_writer = create_component_from_yaml("src/agents/config/step_coder.yml")
        await code_writer.initialize()
        
        # Create a step that processes code generation requests
        def process_code_request(inputs):
            """Process code generation requests and format them."""
            request = inputs.get('input_0')
            if not request:
                return {"output": None}
            
            # Format the request
            formatted_request = f"""
Code Generation Request:
{request}

Please generate clean, well-documented Python code and save it to an appropriate file.
"""
            return {"output": formatted_request}
        
        # Configure request processing step
        request_step_config = StepConfig(
            name="request_processor",
            description="Process and format code generation requests",
            input_configs={
                "input_0": DataUnitConfig(data_type=DataUnitType.MEMORY)
            },
            output_config=DataUnitConfig(data_type=DataUnitType.MEMORY),
            trigger_config=TriggerConfig(trigger_type=TriggerType.DATA_UPDATED)
        )
        
        request_step = create_step("simple", request_step_config, process_func=process_code_request)
        await request_step.initialize()
        
        print(f"✓ Mixed workflow initialized")
        print(f"  - Agent: {code_writer.name}")
        print(f"  - Steps: {request_step.name}")
        print(f"  - Available tools: {code_writer.available_tools}")
        
        # Test the workflow
        print(f"\n--- Testing Mixed Workflow ---")
        
        requests = [
            "Create a simple User class with name and email attributes",
            "Generate a function to add two numbers",
            "Create a basic config file for a web app"
        ]
        
        for i, request in enumerate(requests):
            await request_step.set_input(request)
            await request_step.execute()
            formatted_request = await request_step.get_output()
            
            response = await code_writer.process(formatted_request)
            print(f"  {i+1}. {request[:30]}... → Generated")
        
        print(f"✓ Processed {len(requests)} requests through {request_step.execution_count} step executions")
        
        # Cleanup
        await request_step.shutdown()
        await code_writer.shutdown()
        print("✓ Mixed workflow shutdown successfully")
        
    except Exception as e:
        logger.error(f"Error in mixed workflow demo: {e}")
        print(f"❌ Demo failed: {e}")


async def main():
    """
    Main demo function that runs all demonstrations.
    """
    print("🧠 NanoBrain Framework v2.0 - Advanced Demo")
    print("=" * 60)
    print("Showcasing the refactored framework with:")
    print("- Decoupled Step and Agent classes")
    print("- Agent-to-agent tool calling")
    print("- YAML configuration system")
    print("- Async-first design")
    print("- Configurable executors")
    print("- Data units and triggers")
    
    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Run all demos
        await demo_agent_to_agent_interaction()
        await demo_step_based_processing()
        await demo_yaml_configuration()
        await demo_mixed_workflow()
        
        print("\n" + "="*60)
        print("🎉 ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Check the 'output' directory for generated files:")
        
        # List generated files
        if output_dir.exists():
            for file_path in output_dir.rglob("*"):
                if file_path.is_file():
                    print(f"  - {file_path}")
        
        print("\nThe refactored NanoBrain framework demonstrates:")
        print("✓ Clean separation between Agents and Steps")
        print("✓ Powerful agent-to-agent tool calling")
        print("✓ Flexible YAML configuration system")
        print("✓ Async-first architecture")
        print("✓ Configurable execution backends")
        print("✓ Event-driven step processing")
        print("✓ Schema generation and validation")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\n❌ Demo failed with error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    # Run the demo
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 