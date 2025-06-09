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
        print(f"✓ CodeWriter agent initialized with YAML configuration")
        print(f"  Available tools: {code_writer.available_tools}")
        
        # Test 1: Generate a simple Python function
        print("\n--- Test 1: Generate Python Function ---")
        request1 = """
        Generate a Python function called 'fibonacci' that calculates the nth Fibonacci number.
        The function should:
        - Take an integer parameter 'n'
        - Return the nth Fibonacci number
        - Include proper error handling for negative numbers
        - Save the function to 'output/fibonacci.py'
        """
        
        response1 = await code_writer.process(request1)
        print(f"Response: {response1[:200]}...")
        
        # Test 2: Generate a NanoBrain Step class
        print("\n--- Test 2: Generate NanoBrain Step ---")
        response2 = await code_writer.generate_nanobrain_step(
            step_name="DataProcessorStep",
            description="A step that processes and transforms input data",
            input_types=["Dict[str, Any]"],
            output_types=["Dict[str, Any]"]
        )
        print(f"Generated step class: {len(response2)} characters")
        
        # Save the generated step
        save_result = await code_writer.write_code_to_file(
            response2,
            "output/data_processor_step.py",
            "Generated NanoBrain Step class"
        )
        print(f"Save result: {save_result}")
        
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
        test_data = [
            "Hello, NanoBrain!",
            {"message": "Framework refactored", "version": "2.0"},
            [1, 2, 3, 4, 5]
        ]
        
        for i, data in enumerate(test_data):
            print(f"\nProcessing item {i+1}: {data}")
            
            # Set input data
            await transform_step.set_input(data)
            
            # Execute step
            result = await transform_step.execute()
            
            # Get output
            output = await transform_step.get_output()
            print(f"Result: {output}")
        
        # Show step statistics
        print(f"\n--- Step Statistics ---")
        print(f"Executions: {transform_step.execution_count}")
        print(f"Errors: {transform_step.error_count}")
        print(f"Running: {transform_step.is_running}")
        
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
        print(f"✓ Created example configuration: {config.name}")
        
        # Add custom agents to the configuration
        config.add_agent("code_writer", {
            "agent_type": "simple",
            "name": "code_writer",
            "description": "Specialized code writing agent",
            "model": "gpt-4",
            "system_prompt": "You are a code writing assistant.",
            "tools": [
                {
                    "tool_type": "agent",
                    "name": "file_writer",
                    "description": "File writing tool"
                }
            ]
        })
        
        config.add_agent("file_writer", {
            "agent_type": "simple",
            "name": "file_writer", 
            "description": "File operations agent",
            "model": "gpt-3.5-turbo",
            "system_prompt": "You handle file operations."
        })
        
        # Add custom steps
        config.add_step("code_generator", {
            "step_type": "simple",
            "name": "code_generator",
            "description": "Generate code based on specifications",
            "executor": "local",
            "input_data_units": [{"data_type": "memory"}],
            "output_data_units": [{"data_type": "file", "file_path": "/tmp/generated_code.py"}],
            "trigger_config": {"trigger_type": "data_updated"}
        })
        
        # Add links between components
        config.add_link({
            "link_type": "direct",
            "source": "data_processor",
            "target": "code_generator",
            "name": "processor_to_generator"
        })
        
        print(f"✓ Configuration updated with {len(config.agents)} agents and {len(config.steps)} steps")
        
        # Validate configuration
        errors = config.validate_references()
        if errors:
            print(f"⚠️  Configuration validation errors:")
            for error in errors:
                print(f"   - {error}")
        else:
            print("✓ Configuration validation passed")
        
        # Save configuration to YAML
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        config_file = output_dir / "workflow_config.yaml"
        save_config(config, config_file)
        print(f"✓ Configuration saved to {config_file}")
        
        # Generate schemas
        schema_dir = output_dir / "schemas"
        generate_all_schemas(schema_dir)
        print(f"✓ Schemas generated in {schema_dir}")
        
        # Show configuration summary
        print(f"\n--- Configuration Summary ---")
        print(f"Name: {config.name}")
        print(f"Version: {config.version}")
        print(f"Agents: {list(config.agents.keys())}")
        print(f"Steps: {list(config.steps.keys())}")
        print(f"Executors: {list(config.executors.keys())}")
        print(f"Links: {len(config.links)}")
        
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
        
        raw_requests = [
            "Create a class for managing user sessions with login/logout methods",
            "Generate a function to calculate compound interest",
            "Build a simple REST API endpoint for user registration"
        ]
        
        for i, request in enumerate(raw_requests):
            print(f"\n{i+1}. Processing: {request[:50]}...")
            
            # Step 1: Process request through step
            await request_step.set_input(request)
            await request_step.execute()
            formatted_request = await request_step.get_output()
            
            # Step 2: Generate code using agent
            response = await code_writer.process(formatted_request)
            print(f"   ✓ Code generated ({len(response)} chars)")
        
        # Show workflow statistics
        print(f"\n--- Workflow Statistics ---")
        print(f"Request Step - Executions: {request_step.execution_count}")
        print(f"Code Writer - Executions: {code_writer.execution_count}")
        
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