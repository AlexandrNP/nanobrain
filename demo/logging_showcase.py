#!/usr/bin/env python3
"""
NanoBrain Framework Logging Showcase

This demo demonstrates the comprehensive logging and monitoring capabilities
of the enhanced NanoBrain framework, including:

1. Agent activity logging with conversation tracking
2. Step execution logging with data flow monitoring
3. Tool call logging with parameter and result tracking
4. Performance metrics and statistics
5. Error handling and debugging information
6. Execution tracing with unique request IDs
"""

import asyncio
import json
import time
from pathlib import Path

# Import NanoBrain components
from core import (
    # Core components
    Agent, SimpleAgent, ConversationalAgent, AgentConfig,
    Step, SimpleStep, TransformStep, StepConfig,
    DataUnitMemory, DataUnitConfig, DataUnitType,
    DataUpdatedTrigger, TriggerConfig, TriggerType,
    DirectLink, LinkConfig, LinkType,
    LocalExecutor, ExecutorConfig,
    
    # Logging system
    NanoBrainLogger, get_logger, set_debug_mode, OperationType,
    trace_function_calls, LogLevel
)

from agents import CodeWriterAgent, FileWriterAgent


async def demonstrate_agent_logging():
    """Demonstrate comprehensive agent logging capabilities."""
    print("\n" + "="*60)
    print("AGENT LOGGING DEMONSTRATION")
    print("="*60)
    
    # Enable debug mode for detailed logging
    set_debug_mode(True)
    
    # Create agents with logging enabled
    code_writer_config = AgentConfig(
        name="CodeWriterAgent",
        description="Generates Python code with comprehensive logging",
        model="gpt-3.5-turbo",
        temperature=0.7,
        debug_mode=True,
        enable_logging=True,
        log_conversations=True,
        log_tool_calls=True
    )
    
    file_writer_config = AgentConfig(
        name="FileWriterAgent", 
        description="Handles file operations with detailed logging",
        debug_mode=True,
        enable_logging=True,
        log_conversations=True,
        log_tool_calls=True
    )
    
    # Create agents
    code_writer = CodeWriterAgent(code_writer_config)
    file_writer = FileWriterAgent(file_writer_config)
    
    # Register file writer as a tool for code writer
    code_writer.register_agent_tool(
        file_writer, 
        name="file_writer",
        description="Write code to files"
    )
    
    try:
        # Initialize agents (this will be logged)
        await code_writer.initialize()
        await file_writer.initialize()
        
        # Execute a complex task that involves multiple tool calls
        task = """
        Create a simple Python function that calculates the factorial of a number.
        Save it to a file called 'factorial.py' and include proper documentation.
        """
        
        print(f"\nExecuting task: {task}")
        result = await code_writer.execute(task)
        print(f"Task result: {result}")
        
        # Get performance statistics
        code_writer_stats = code_writer.get_performance_stats()
        file_writer_stats = file_writer.get_performance_stats()
        
        print(f"\nCode Writer Performance Stats:")
        print(json.dumps(code_writer_stats, indent=2))
        
        print(f"\nFile Writer Performance Stats:")
        print(json.dumps(file_writer_stats, indent=2))
        
        # Get conversation history
        logger = get_logger()
        conversations = logger.get_conversation_history("CodeWriterAgent")
        print(f"\nConversation History ({len(conversations)} entries):")
        for i, conv in enumerate(conversations[-2:]):  # Show last 2 conversations
            print(f"  {i+1}. Input: {conv.input_text[:100]}...")
            print(f"     Response: {conv.response_text[:100]}...")
            print(f"     Tool calls: {len(conv.tool_calls)}")
            print(f"     Duration: {conv.duration_ms}ms")
        
        # Get tool call history
        tool_calls = logger.get_tool_call_history()
        print(f"\nTool Call History ({len(tool_calls)} calls):")
        for i, call in enumerate(tool_calls[-3:]):  # Show last 3 calls
            print(f"  {i+1}. Tool: {call.tool_name}")
            print(f"     Parameters: {str(call.parameters)[:100]}...")
            print(f"     Duration: {call.duration_ms}ms")
            print(f"     Success: {call.error is None}")
        
    finally:
        # Shutdown agents (this will be logged with final statistics)
        await code_writer.shutdown()
        await file_writer.shutdown()


async def demonstrate_step_logging():
    """Demonstrate comprehensive step logging capabilities."""
    print("\n" + "="*60)
    print("STEP LOGGING DEMONSTRATION")
    print("="*60)
    
    # Create data units with logging
    input_config = DataUnitConfig(
        name="input_data",
        data_type=DataUnitType.MEMORY,
        description="Input data for processing"
    )
    
    output_config = DataUnitConfig(
        name="output_data",
        data_type=DataUnitType.MEMORY,
        description="Processed output data"
    )
    
    # Create trigger configuration
    trigger_config = TriggerConfig(
        name="data_trigger",
        trigger_type=TriggerType.DATA_UPDATED,
        description="Trigger when input data is updated"
    )
    
    # Create step configurations with logging enabled
    transform_config = StepConfig(
        name="DataTransformStep",
        description="Transforms input data with comprehensive logging",
        input_configs={"input": input_config},
        output_config=output_config,
        trigger_config=trigger_config,
        debug_mode=True,
        enable_logging=True,
        log_data_transfers=True,
        log_executions=True
    )
    
    # Custom transformation function
    def uppercase_transform(data):
        """Transform data to uppercase with metadata."""
        return {
            "original": data,
            "transformed": str(data).upper(),
            "timestamp": time.time(),
            "transform_type": "uppercase"
        }
    
    # Create step with custom transform
    transform_step = TransformStep(transform_config, transform_func=uppercase_transform)
    
    try:
        # Initialize step (this will be logged)
        await transform_step.initialize()
        
        # Create input data unit for the step
        input_data_unit = DataUnitMemory(input_config)
        await input_data_unit.initialize()
        transform_step.register_input_data_unit("input", input_data_unit)
        
        # Set up some test data
        test_data = [
            "hello world",
            {"message": "test data", "value": 42},
            ["item1", "item2", "item3"]
        ]
        
        # Process each piece of test data
        for i, data in enumerate(test_data):
            print(f"\nProcessing test data {i+1}: {data}")
            
            # Write data to input (this triggers execution if trigger is set up)
            await input_data_unit.write(data)
            
            # Execute step manually to demonstrate logging
            result = await transform_step.execute()
            print(f"Transform result: {result}")
        
        # Get performance statistics
        step_stats = transform_step.get_performance_stats()
        print(f"\nStep Performance Stats:")
        print(json.dumps(step_stats, indent=2))
        
        # Get the last result
        last_result = transform_step.get_result()
        print(f"\nLast execution result: {last_result}")
        
    finally:
        # Shutdown step (this will be logged with final statistics)
        await transform_step.shutdown()


async def demonstrate_performance_monitoring():
    """Demonstrate performance monitoring and metrics collection."""
    print("\n" + "="*60)
    print("PERFORMANCE MONITORING DEMONSTRATION")
    print("="*60)
    
    # Get the global logger
    logger = get_logger("performance_demo")
    
    # Create a function with tracing
    @trace_function_calls(logger)
    async def sample_processing_function(data, delay=0.1):
        """Sample function that simulates processing with delay."""
        await asyncio.sleep(delay)
        return f"Processed: {data}"
    
    # Execute the function multiple times to generate metrics
    print("Executing traced function multiple times...")
    
    for i in range(5):
        result = await sample_processing_function(f"data_{i}", delay=0.05 * (i + 1))
        print(f"  Result {i+1}: {result}")
    
    # Get performance summary
    performance_summary = logger.get_performance_summary()
    print(f"\nPerformance Summary:")
    print(json.dumps(performance_summary, indent=2))
    
    # Demonstrate execution context tracking
    print(f"\nDemonstrating execution context tracking...")
    
    async with logger.async_execution_context(
        OperationType.WORKFLOW_RUN,
        "sample_workflow",
        workflow_type="demonstration",
        steps_count=3
    ) as context:
        
        # Simulate nested operations
        async with logger.async_execution_context(
            OperationType.STEP_EXECUTE,
            "step_1",
            parent_context=context.request_id
        ):
            await asyncio.sleep(0.1)
            logger.info("Step 1 completed")
        
        async with logger.async_execution_context(
            OperationType.STEP_EXECUTE,
            "step_2",
            parent_context=context.request_id
        ):
            await asyncio.sleep(0.15)
            logger.info("Step 2 completed")
        
        async with logger.async_execution_context(
            OperationType.STEP_EXECUTE,
            "step_3",
            parent_context=context.request_id
        ):
            await asyncio.sleep(0.08)
            logger.info("Step 3 completed")
        
        context.metadata['total_steps'] = 3
        context.metadata['workflow_status'] = 'completed'
    
    print("Workflow execution completed with full tracing")


async def demonstrate_error_handling_and_debugging():
    """Demonstrate error handling and debugging capabilities."""
    print("\n" + "="*60)
    print("ERROR HANDLING AND DEBUGGING DEMONSTRATION")
    print("="*60)
    
    # Create an agent that will encounter errors
    error_config = AgentConfig(
        name="ErrorProneAgent",
        description="Agent designed to demonstrate error handling",
        debug_mode=True,
        enable_logging=True,
        log_conversations=True,
        log_tool_calls=True
    )
    
    error_agent = SimpleAgent(error_config)
    
    # Register a tool that will fail
    def failing_tool(message: str):
        """A tool that always fails for demonstration."""
        raise ValueError(f"Intentional failure with message: {message}")
    
    error_agent.register_function_tool(
        failing_tool,
        name="failing_tool",
        description="A tool that always fails",
        parameters={
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "Message to include in failure"}
            },
            "required": ["message"]
        }
    )
    
    try:
        await error_agent.initialize()
        
        # Try to execute a task that will fail
        print("Attempting to execute a task that will cause errors...")
        
        try:
            result = await error_agent.execute("Use the failing_tool with message 'test error'")
            print(f"Unexpected success: {result}")
        except Exception as e:
            print(f"Expected error caught: {e}")
        
        # Get error statistics
        stats = error_agent.get_performance_stats()
        print(f"\nAgent Error Statistics:")
        print(f"  Execution count: {stats['execution_count']}")
        print(f"  Error count: {stats['error_count']}")
        print(f"  Success rate: {stats['success_rate']:.2%}")
        
    finally:
        await error_agent.shutdown()


async def main():
    """Main demonstration function."""
    print("NanoBrain Framework - Comprehensive Logging Showcase")
    print("=" * 60)
    
    # Set up logging to file
    log_file = Path("nanobrain_demo.log")
    logger = get_logger("nanobrain_demo", log_file=log_file, debug_mode=True)
    
    logger.info("Starting NanoBrain logging demonstration")
    
    try:
        # Run all demonstrations
        await demonstrate_agent_logging()
        await demonstrate_step_logging()
        await demonstrate_performance_monitoring()
        await demonstrate_error_handling_and_debugging()
        
        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETE")
        print("="*60)
        
        # Final performance summary
        final_summary = logger.get_performance_summary()
        print(f"\nFinal Performance Summary:")
        print(json.dumps(final_summary, indent=2))
        
        print(f"\nDetailed logs have been written to: {log_file}")
        print("Check the log file for complete execution traces and debugging information.")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}", error_type=type(e).__name__)
        raise
    
    finally:
        logger.info("NanoBrain logging demonstration completed")


if __name__ == "__main__":
    asyncio.run(main()) 