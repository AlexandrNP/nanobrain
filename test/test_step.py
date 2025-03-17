import unittest
import asyncio
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

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


class TestStep(unittest.TestCase):
    def setUp(self):
        # Create mock executor
        self.executor = MagicMock(spec=ExecutorBase)
        self.executor.can_execute.return_value = True
        
        # Create mock input and output data units
        self.input_data = MagicMock(spec=DataUnitBase)
        self.input_data.get.return_value = "test_input"
        self.output_data = MagicMock(spec=DataUnitBase)
        
        # Create a mock CircuitBreaker
        self.mock_circuit_breaker = MagicMock(spec=CircuitBreaker)
        self.mock_circuit_breaker.can_execute.return_value = True
        
        # Create a mock TriggerAllDataReceived
        self.mock_trigger = MagicMock(spec=TriggerAllDataReceived)
        self.mock_trigger.monitor.return_value = True
        
        # Create step instance with patched CircuitBreaker
        with patch('src.Step.CircuitBreaker', return_value=self.mock_circuit_breaker), \
             patch('src.Step.TriggerAllDataReceived', return_value=self.mock_trigger):
            self.step = Step(
                executor=self.executor,
                input_sources={"test_link": self.input_data},
                output=self.output_data
            )
    
    def test_initialization(self):
        """Test that Step initializes correctly with proper inheritance."""
        # Verify attributes are set correctly
        self.assertEqual(self.step.input_sources, {"test_link": self.input_data})
        self.assertEqual(self.step.output, self.output_data)
        self.assertIsNone(self.step.result)
        self.assertEqual(self.step.state, ComponentState.INACTIVE)
        self.assertFalse(self.step.running)
        
        # Verify inheritance from PackageBase
        self.assertTrue(hasattr(self.step, 'directory_tracer'))
        self.assertTrue(hasattr(self.step, 'config_manager'))
        
    def test_process_method(self):
        """Test the basic process method."""
        # Run the process method with test input
        result = asyncio.run(self.step.process({"test_link": "test_input"}))
        
        # Verify it returns the first input
        self.assertEqual(result, "test_input")
        
        # Test with empty input
        result = asyncio.run(self.step.process({}))
        self.assertIsNone(result)
    
    def test_get_result(self):
        """Test the get_result method."""
        # Set a result and verify get_result returns it
        self.step.result = "test_result"
        self.assertEqual(self.step.get_result(), "test_result")
    
    async def async_test_execute(self):
        """Test the execute method."""
        # Execute the step
        result = await self.step.execute()
        
        # Verify the result is correct
        self.assertEqual(result, "test_input")
        
        # Verify the output was set
        self.output_data.set.assert_called_once_with("test_input")
    
    def test_execute(self):
        """Run the async test."""
        asyncio.run(self.async_test_execute())
    
    async def async_test_execute_with_circuit_breaker_block(self):
        """Test execute with circuit breaker blocking."""
        # Configure circuit breaker to block execution
        self.mock_circuit_breaker.can_execute.return_value = False
        
        # Execute the step
        result = await self.step.execute()
        
        # Verify execution was blocked
        self.assertIsNone(result)
        self.assertEqual(self.step.state, ComponentState.INACTIVE)
    
    def test_execute_with_circuit_breaker_block(self):
        """Run the async test for circuit breaker blocking."""
        asyncio.run(self.async_test_execute_with_circuit_breaker_block())
    
    async def async_test_execute_with_exception(self):
        """Test execute with an exception during processing."""
        # Configure process to raise an exception
        self.step.process = MagicMock(side_effect=Exception("Test exception"))
        
        # Execute the step and expect an exception to be raised
        with self.assertRaises(Exception):
            await self.step.execute()
        
        # Verify the step state after the exception
        self.assertEqual(self.step.state, ComponentState.ERROR)
        self.assertFalse(self.step.running)
    
    def test_execute_with_exception(self):
        """Run the async test for exception handling."""
        asyncio.run(self.async_test_execute_with_exception())
    
    def test_invoke_alias(self):
        """Test that invoke calls execute."""
        # Mock the execute method
        self.step.execute = MagicMock()
        
        # Create a mock coroutine for execute
        async def mock_execute():
            return "test_result"
        
        self.step.execute.return_value = mock_execute()
        
        # Call invoke
        asyncio.run(self.step.invoke())
        
        # Verify execute was called
        self.step.execute.assert_called_once()
    
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
        step_with_name = Step(self.executor, name=custom_name)
        self.assertEqual(step_with_name.name, custom_name, "Step should use the custom name provided in constructor")
        
        # Test without a name (should default to class name)
        step_without_name = Step(self.executor)
        self.assertEqual(step_without_name.name, "Step", "Step should use class name as default when no name is provided")


if __name__ == '__main__':
    unittest.main() 