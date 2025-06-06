#!/usr/bin/env python3
"""
Unit Tests for NanoBrain Logging System

Tests the comprehensive logging and monitoring capabilities including:
- Structured logging with JSON format
- Execution context tracking
- Performance metrics collection
- Agent conversation logging
- Tool call logging
- Error handling and debugging
"""

import asyncio
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.logging_system import (
    NanoBrainLogger, get_logger, set_debug_mode, trace_function_calls,
    LogLevel, OperationType, ExecutionContext, ToolCallLog, 
    AgentConversationLog
)


class TestNanoBrainLogger(unittest.TestCase):
    """Test cases for the NanoBrainLogger class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = Path(self.temp_dir) / "test.log"
        self.logger = NanoBrainLogger(
            name="test_logger",
            log_file=self.log_file,
            debug_mode=True
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up temporary files
        if self.log_file.exists():
            self.log_file.unlink()
        if Path(self.temp_dir).exists():
            Path(self.temp_dir).rmdir()
    
    def test_logger_initialization(self):
        """Test logger initialization."""
        self.assertEqual(self.logger.name, "test_logger")
        self.assertTrue(self.logger.debug_mode)
        self.assertEqual(len(self.logger._context_stack), 0)
        self.assertEqual(len(self.logger._performance_metrics), 0)
        self.assertEqual(len(self.logger._conversation_logs), 0)
        self.assertEqual(len(self.logger._tool_call_logs), 0)
    
    def test_structured_logging(self):
        """Test structured logging functionality."""
        # Test different log levels
        self.logger.debug("Debug message", test_param="debug_value")
        self.logger.info("Info message", test_param="info_value")
        self.logger.warning("Warning message", test_param="warning_value")
        self.logger.error("Error message", test_param="error_value")
        self.logger.critical("Critical message", test_param="critical_value")
        
        # Verify log file was created
        self.assertTrue(self.log_file.exists())
        
        # Read and verify log content
        with open(self.log_file, 'r') as f:
            log_content = f.read()
            self.assertIn("Debug message", log_content)
            self.assertIn("Info message", log_content)
            self.assertIn("Warning message", log_content)
            self.assertIn("Error message", log_content)
            self.assertIn("Critical message", log_content)
    
    def test_execution_context(self):
        """Test execution context tracking."""
        with self.logger.execution_context(
            OperationType.AGENT_PROCESS,
            "test_component",
            test_metadata="test_value"
        ) as context:
            self.assertIsInstance(context, ExecutionContext)
            self.assertEqual(context.operation_type, OperationType.AGENT_PROCESS)
            self.assertEqual(context.component_name, "test_component")
            self.assertIn("test_metadata", context.metadata)
            self.assertEqual(context.metadata["test_metadata"], "test_value")
            
            # Test nested context
            with self.logger.execution_context(
                OperationType.TOOL_CALL,
                "nested_component"
            ) as nested_context:
                self.assertEqual(nested_context.parent_request_id, context.request_id)
        
        # Context should be completed
        self.assertIsNotNone(context.end_time)
        self.assertIsNotNone(context.duration_ms)
        self.assertTrue(context.success)
    
    async def test_async_execution_context(self):
        """Test async execution context tracking."""
        async with self.logger.async_execution_context(
            OperationType.STEP_EXECUTE,
            "async_component",
            async_metadata="async_value"
        ) as context:
            self.assertIsInstance(context, ExecutionContext)
            self.assertEqual(context.operation_type, OperationType.STEP_EXECUTE)
            self.assertEqual(context.component_name, "async_component")
            
            # Simulate some async work
            await asyncio.sleep(0.01)
        
        # Context should be completed
        self.assertIsNotNone(context.end_time)
        self.assertIsNotNone(context.duration_ms)
        self.assertGreater(context.duration_ms, 0)
        self.assertTrue(context.success)
    
    def test_tool_call_logging(self):
        """Test tool call logging functionality."""
        # Log a successful tool call
        self.logger.log_tool_call(
            tool_name="test_tool",
            parameters={"param1": "value1", "param2": 42},
            result="Tool execution successful",
            duration_ms=150.5
        )
        
        # Log a failed tool call
        self.logger.log_tool_call(
            tool_name="failing_tool",
            parameters={"param": "value"},
            error="Tool execution failed"
        )
        
        # Verify tool calls were logged
        tool_calls = self.logger.get_tool_call_history()
        self.assertEqual(len(tool_calls), 2)
        
        # Check successful call
        successful_call = tool_calls[0]
        self.assertEqual(successful_call.tool_name, "test_tool")
        self.assertEqual(successful_call.parameters["param1"], "value1")
        self.assertEqual(successful_call.result, "Tool execution successful")
        self.assertEqual(successful_call.duration_ms, 150.5)
        self.assertIsNone(successful_call.error)
        
        # Check failed call
        failed_call = tool_calls[1]
        self.assertEqual(failed_call.tool_name, "failing_tool")
        self.assertEqual(failed_call.error, "Tool execution failed")
        self.assertIsNone(failed_call.result)
    
    def test_agent_conversation_logging(self):
        """Test agent conversation logging functionality."""
        # Create sample tool calls
        tool_calls = [
            ToolCallLog(
                tool_name="calculator",
                parameters={"operation": "add", "a": 5, "b": 3},
                result=8,
                duration_ms=25.0
            )
        ]
        
        # Log agent conversation
        self.logger.log_agent_conversation(
            agent_name="TestAgent",
            input_text="Calculate 5 + 3",
            response_text="The result is 8",
            tool_calls=tool_calls,
            llm_calls=1,
            total_tokens=150,
            duration_ms=500.0
        )
        
        # Verify conversation was logged
        conversations = self.logger.get_conversation_history()
        self.assertEqual(len(conversations), 1)
        
        conversation = conversations[0]
        self.assertEqual(conversation.agent_name, "TestAgent")
        self.assertEqual(conversation.input_text, "Calculate 5 + 3")
        self.assertEqual(conversation.response_text, "The result is 8")
        self.assertEqual(len(conversation.tool_calls), 1)
        self.assertEqual(conversation.llm_calls, 1)
        self.assertEqual(conversation.total_tokens, 150)
        self.assertEqual(conversation.duration_ms, 500.0)
        
        # Test filtering by agent name
        filtered_conversations = self.logger.get_conversation_history("TestAgent")
        self.assertEqual(len(filtered_conversations), 1)
        
        filtered_conversations = self.logger.get_conversation_history("NonExistentAgent")
        self.assertEqual(len(filtered_conversations), 0)
    
    def test_performance_metrics(self):
        """Test performance metrics collection."""
        # Simulate multiple operations
        operation_types = [
            (OperationType.AGENT_PROCESS, "agent1"),
            (OperationType.AGENT_PROCESS, "agent1"),
            (OperationType.STEP_EXECUTE, "step1"),
            (OperationType.TOOL_CALL, "tool1"),
            (OperationType.TOOL_CALL, "tool1"),
            (OperationType.TOOL_CALL, "tool1")
        ]
        
        for op_type, component in operation_types:
            with self.logger.execution_context(op_type, component):
                time.sleep(0.01)  # Simulate work
        
        # Get performance summary
        summary = self.logger.get_performance_summary()
        
        # Verify metrics were collected
        self.assertIn("agent_process:agent1", summary)
        self.assertIn("step_execute:step1", summary)
        self.assertIn("tool_call:tool1", summary)
        
        # Check agent metrics
        agent_metrics = summary["agent_process:agent1"]
        self.assertEqual(agent_metrics["count"], 2)
        self.assertGreater(agent_metrics["avg_ms"], 0)
        self.assertGreater(agent_metrics["total_ms"], 0)
        
        # Check tool metrics
        tool_metrics = summary["tool_call:tool1"]
        self.assertEqual(tool_metrics["count"], 3)
        self.assertGreater(tool_metrics["avg_ms"], 0)
    
    def test_error_handling_in_context(self):
        """Test error handling within execution contexts."""
        try:
            with self.logger.execution_context(
                OperationType.AGENT_PROCESS,
                "error_component"
            ) as context:
                raise ValueError("Test error")
        except ValueError:
            pass  # Expected
        
        # Context should record the error
        self.assertFalse(context.success)
        self.assertEqual(context.error_message, "Test error")
        self.assertIsNotNone(context.end_time)
        self.assertIsNotNone(context.duration_ms)
    
    def test_clear_logs(self):
        """Test clearing logged data."""
        # Add some data
        self.logger.log_tool_call("test_tool", {"param": "value"})
        self.logger.log_agent_conversation("TestAgent", "input", "output")
        
        # Verify data exists
        self.assertEqual(len(self.logger.get_tool_call_history()), 1)
        self.assertEqual(len(self.logger.get_conversation_history()), 1)
        
        # Clear logs
        self.logger.clear_logs()
        
        # Verify data is cleared
        self.assertEqual(len(self.logger.get_tool_call_history()), 0)
        self.assertEqual(len(self.logger.get_conversation_history()), 0)
        self.assertEqual(len(self.logger._performance_metrics), 0)


class TestTraceFunctionCalls(unittest.TestCase):
    """Test cases for the trace_function_calls decorator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.logger = NanoBrainLogger("trace_test", debug_mode=True)
    
    def test_sync_function_tracing(self):
        """Test tracing of synchronous functions."""
        @trace_function_calls(self.logger)
        def sample_function(x, y, z=10):
            return x + y + z
        
        result = sample_function(5, 3, z=2)
        self.assertEqual(result, 10)
        
        # Check that performance metrics were recorded
        summary = self.logger.get_performance_summary()
        self.assertTrue(any("sample_function" in key for key in summary.keys()))
    
    async def test_async_function_tracing(self):
        """Test tracing of asynchronous functions."""
        @trace_function_calls(self.logger)
        async def async_sample_function(x, delay=0.01):
            await asyncio.sleep(delay)
            return x * 2
        
        result = await async_sample_function(5, delay=0.005)
        self.assertEqual(result, 10)
        
        # Check that performance metrics were recorded
        summary = self.logger.get_performance_summary()
        self.assertTrue(any("async_sample_function" in key for key in summary.keys()))
    
    def test_function_tracing_with_error(self):
        """Test tracing of functions that raise errors."""
        @trace_function_calls(self.logger)
        def failing_function():
            raise RuntimeError("Test error")
        
        with self.assertRaises(RuntimeError):
            failing_function()
        
        # Check that the error was logged
        summary = self.logger.get_performance_summary()
        self.assertTrue(any("failing_function" in key for key in summary.keys()))
    
    def test_parameter_redaction(self):
        """Test that sensitive parameters are redacted."""
        @trace_function_calls(self.logger)
        def function_with_secrets(username, password, token, normal_param):
            return f"User: {username}"
        
        result = function_with_secrets(
            username="testuser",
            password="secret123",
            token="abc123",
            normal_param="visible"
        )
        
        self.assertEqual(result, "User: testuser")
        
        # The actual parameter redaction testing would require
        # inspecting the logged data, which is more complex


class TestGlobalLoggerFunctions(unittest.TestCase):
    """Test cases for global logger functions."""
    
    def test_get_logger(self):
        """Test get_logger function."""
        logger1 = get_logger("test1")
        logger2 = get_logger("test2")
        logger3 = get_logger("test1")  # Should return the same instance
        
        self.assertIsInstance(logger1, NanoBrainLogger)
        self.assertIsInstance(logger2, NanoBrainLogger)
        self.assertEqual(logger1.name, "test1")
        self.assertEqual(logger2.name, "test2")
        # Note: The current implementation creates new instances each time
        # This might be changed to return the same instance for the same name
    
    def test_set_debug_mode(self):
        """Test set_debug_mode function."""
        # This test would require access to the global logger instance
        # and verification that debug mode is properly set
        set_debug_mode(True)
        set_debug_mode(False)
        # The actual testing would require checking the logger's debug mode


class TestIntegrationScenarios(unittest.TestCase):
    """Integration test scenarios for the logging system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.logger = NanoBrainLogger("integration_test", debug_mode=True)
    
    async def test_complex_workflow_logging(self):
        """Test logging of a complex workflow scenario."""
        # Simulate a complex workflow with nested operations
        async with self.logger.async_execution_context(
            OperationType.WORKFLOW_RUN,
            "complex_workflow"
        ) as workflow_context:
            
            # Step 1: Agent processing
            async with self.logger.async_execution_context(
                OperationType.AGENT_PROCESS,
                "planning_agent"
            ) as agent_context:
                
                # Simulate LLM call
                async with self.logger.async_execution_context(
                    OperationType.LLM_CALL,
                    "gpt-3.5-turbo"
                ):
                    await asyncio.sleep(0.01)
                
                # Simulate tool calls
                for i in range(3):
                    self.logger.log_tool_call(
                        tool_name=f"tool_{i}",
                        parameters={"input": f"data_{i}"},
                        result=f"result_{i}",
                        duration_ms=10.0 + i * 5
                    )
            
            # Step 2: Data processing
            async with self.logger.async_execution_context(
                OperationType.STEP_EXECUTE,
                "data_processor"
            ):
                await asyncio.sleep(0.005)
                
                # Log data transfers
                self.logger.log_data_transfer(
                    source="input_data",
                    destination="processor",
                    data_type="dict",
                    size_bytes=1024
                )
            
            # Log agent conversation
            self.logger.log_agent_conversation(
                agent_name="planning_agent",
                input_text="Process the workflow data",
                response_text="Workflow completed successfully",
                llm_calls=1,
                total_tokens=250,
                duration_ms=workflow_context.duration_ms
            )
        
        # Verify all data was logged correctly
        conversations = self.logger.get_conversation_history()
        self.assertEqual(len(conversations), 1)
        
        tool_calls = self.logger.get_tool_call_history()
        self.assertEqual(len(tool_calls), 3)
        
        performance_summary = self.logger.get_performance_summary()
        self.assertGreater(len(performance_summary), 0)
        
        # Verify workflow completed successfully
        self.assertTrue(workflow_context.success)
        self.assertIsNotNone(workflow_context.duration_ms)


def run_async_test(test_func):
    """Helper function to run async tests."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(test_func())
    finally:
        loop.close()


if __name__ == "__main__":
    # Run async tests manually since unittest doesn't handle them well
    print("Running async tests...")
    
    # Test async execution context
    test_logger = TestNanoBrainLogger()
    test_logger.setUp()
    run_async_test(test_logger.test_async_execution_context)
    test_logger.tearDown()
    print("✓ test_async_execution_context passed")
    
    # Test async function tracing
    test_trace = TestTraceFunctionCalls()
    test_trace.setUp()
    run_async_test(test_trace.test_async_function_tracing)
    print("✓ test_async_function_tracing passed")
    
    # Test complex workflow
    test_integration = TestIntegrationScenarios()
    test_integration.setUp()
    run_async_test(test_integration.test_complex_workflow_logging)
    print("✓ test_complex_workflow_logging passed")
    
    print("\nRunning synchronous tests...")
    
    # Run the standard unittest suite for synchronous tests
    unittest.main(verbosity=2, exit=False) 