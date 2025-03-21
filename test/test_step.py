import unittest
import asyncio
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Add src directory to Python path
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))

from src.Step import Step
from src.ExecutorBase import ExecutorBase
from src.LinkBase import LinkBase
from src.DataUnitBase import DataUnitBase
from src.enums import ComponentState
from src.concurrency import CircuitBreaker
from src.TriggerAllDataReceived import TriggerAllDataReceived


class TestStep(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        """Set up the test environment."""
        # Create mock objects
        self.mock_executor = MagicMock()
        self.mock_executor.execute = MagicMock(return_value="test_result")
        
        self.mock_input = MagicMock()
        self.mock_input.get = MagicMock(return_value="test_input")
        
        self.mock_output = MagicMock()
        
        self.mock_circuit_breaker = MagicMock()
        self.mock_circuit_breaker.allow_execution = MagicMock(return_value=True)
        self.mock_circuit_breaker.record_success = MagicMock()
        self.mock_circuit_breaker.record_failure = MagicMock()
        
        self.mock_trigger = MagicMock()
        self.mock_trigger.monitor = MagicMock(return_value=True)
        
        # Create step
        self.step = Step(name="TestStep", executor=self.mock_executor)
        self.step.circuit_breaker = self.mock_circuit_breaker
        self.step.trigger = self.mock_trigger
        
        # Register input and output
        self.step.input_sources = {"test_link": self.mock_input}
        self.step.output = self.mock_output
        
    def test_initialization(self):
        """Test that Step initializes correctly with proper inheritance."""
        # Verify attributes are set correctly
        self.assertEqual(self.step.input_sources, {"test_link": self.mock_input})
        self.assertEqual(self.step.output, self.mock_output)
        self.assertIsNone(self.step.result)
        self.assertEqual(self.step.state, ComponentState.INACTIVE)
        self.assertFalse(self.step.running)
        
        # Verify inheritance from PackageBase
        self.assertTrue(hasattr(self.step, 'directory_tracer'))
        self.assertTrue(hasattr(self.step, 'config_manager'))
        
    async def test_process_method(self):
        """Test the basic process method."""
        # Run the process method with test input
        result = await self.step.process({"test_link": "test_input"})
        
        # Verify it returns the first input
        self.assertEqual(result, "test_input")
        
        # Test with empty input
        result = await self.step.process({})
        self.assertIsNone(result)
    
    def test_get_result(self):
        """Test the get_result method."""
        # Set a result and verify get_result returns it
        self.step.result = "test_result"
        self.assertEqual(self.step.get_result(), "test_result")
    
    async def test_execute(self):
        """Test the execute method."""
        # Instead of using AsyncMock which creates coroutines, use a regular sync mock
        def sync_process(inputs):
            return "mocked_result"
    
        self.step.process = MagicMock(side_effect=sync_process)
        
        # Set up execute to use our mocked process directly
        self.step.executor = None
        
        # Execute the step (execute is a synchronous method)
        result = self.step.execute()
    
        # Verify the result
        self.assertEqual(result, "mocked_result")
        
        # Verify process was called
        self.step.process.assert_called_once()
    
    async def test_execute_with_circuit_breaker_block(self):
        """Test execute with circuit breaker blocking."""
        # Configure circuit breaker to block execution
        self.mock_circuit_breaker.allow_execution.return_value = False
    
        # Execute the step (execute is a synchronous method)
        result = self.step.execute()
    
        # Verify execution was blocked
        self.assertIsNone(result)
        self.assertEqual(self.step.state, ComponentState.BLOCKED)
    
    async def test_execute_with_exception(self):
        """Test execute with an exception during processing."""
        # Configure process to raise an exception
        async def async_process_exception(inputs):
            raise Exception("Test exception")
    
        self.step.process = AsyncMock(side_effect=async_process_exception)
        self.step.executor = None
        
        # Create a mock that will actually properly intercept the call to process
        # in the execute method and raise an exception
        def mock_run_until_complete(coro):
            raise Exception("Test exception")
            
        # Patch asyncio.run to simulate exception during process execution
        with patch('asyncio.run', side_effect=mock_run_until_complete):
            # Execute should now raise an exception
            with self.assertRaises(Exception):
                self.step.execute()
        
            # Verify the step is in ERROR state
            self.assertEqual(self.step.state, ComponentState.ERROR)
            
            # Verify the circuit breaker recorded a failure
            self.mock_circuit_breaker.record_failure.assert_called_once()
    
    async def test_invoke_alias(self):
        """Test that invoke calls execute."""
        # Mock the execute method
        self.step.execute = AsyncMock(return_value="test_result")
        
        # Call invoke
        result = await self.step.invoke()
        
        # Verify execute was called and result was returned
        self.step.execute.assert_called_once()
        self.assertEqual(result, "test_result")
    
    def test_register_input_source(self):
        """Test registering an input source."""
        new_input = MagicMock(spec=DataUnitBase)
        self.step.register_input_source("new_link", new_input)
        self.assertEqual(self.step.input_sources["new_link"], new_input)
    
    def test_register_output(self):
        """Test registering an output."""
        new_output = MagicMock(spec=DataUnitBase)
        self.step.register_output(new_output)
        self.assertEqual(self.step.output, new_output)

    def test_name_attribute(self):
        """Test that the name attribute is properly set."""
        # Test with a custom name
        custom_name = "CustomStepName"
        step_with_name = Step(self.mock_executor, name=custom_name)
        self.assertEqual(step_with_name.name, custom_name, "Step should use the custom name provided in constructor")
        
        # Test without a name (should default to class name)
        step_without_name = Step(self.mock_executor)
        self.assertEqual(step_without_name.name, "Step", "Step should use class name as default when no name is provided")

    async def test_run_method(self):
        """Test the _run method required by BaseTool."""
        # Since _run calls process, we need to mock the process method
        # to return a predictable result without actually awaiting
        def sync_process(inputs):
            return "test_result"
    
        # Replacing the async method with a sync version for testing
        self.step.process = MagicMock(side_effect=sync_process)
    
        # Call _run - It should run the process method
        result = self.step._run("test_input")
    
        # Verify the result
        self.assertEqual(result, "test_result")
        
        # Verify process was called with the expected input
        self.step.process.assert_called_once()
        # Check that the input was correctly formatted
        call_args = self.step.process.call_args[0][0]
        self.assertEqual(call_args["input_0"], "test_input")

    async def test_arun_method(self):
        """Test the _arun method required by BaseTool."""
        # Mock the process method
        self.step.process = AsyncMock(return_value="test_result")
        
        # Call _arun
        result = await self.step._arun("test_input")
        
        # Verify result
        self.assertEqual(result, "test_result")


if __name__ == '__main__':
    unittest.main() 