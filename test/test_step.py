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


class TestStep(unittest.TestCase):
    def setUp(self):
        # Create mock executor
        self.executor = MagicMock(spec=ExecutorBase)
        self.executor.can_execute.return_value = True
        
        # Create mock input and output data units
        self.input_data = MagicMock(spec=DataUnitBase)
        self.output_data = MagicMock(spec=DataUnitBase)
        
        # Create mock input and output links
        self.input_link = MagicMock(spec=LinkBase)
        # Configure the mock to have an output attribute that returns another mock
        self.input_link.output = MagicMock()
        self.input_link.output.get.return_value = "test_input"
        self.output_link = MagicMock(spec=LinkBase)
        self.output_link.input = MagicMock()
        
        # Create a mock CircuitBreaker
        self.mock_circuit_breaker = MagicMock(spec=CircuitBreaker)
        self.mock_circuit_breaker.can_execute.return_value = True
        
        # Create step instance with patched CircuitBreaker
        with patch('src.Step.CircuitBreaker', return_value=self.mock_circuit_breaker):
            self.step = Step(
                executor=self.executor,
                input_sources=[self.input_link],
                output_sink=self.output_link
            )
    
    def test_initialization(self):
        """Test that Step initializes correctly with proper inheritance."""
        # Verify attributes are set correctly
        self.assertEqual(self.step.input_sources, [self.input_link])
        self.assertEqual(self.step.output_sink, self.output_link)
        self.assertIsNone(self.step.result)
        self.assertEqual(self.step.state, ComponentState.INACTIVE)
        self.assertFalse(self.step.running)
        
        # Verify inheritance from PackageBase
        self.assertTrue(hasattr(self.step, 'directory_tracer'))
        self.assertTrue(hasattr(self.step, 'config_manager'))
        
    def test_process_method(self):
        """Test the basic process method."""
        # Run the process method with test input
        result = asyncio.run(self.step.process(["test_input"]))
        
        # Verify it returns the first input
        self.assertEqual(result, "test_input")
        
        # Test with empty input
        result = asyncio.run(self.step.process([]))
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
        
        # Verify the input link's transfer method was called
        self.input_link.transfer.assert_called_once()
        
        # Verify the output was set on the output link
        self.output_link.input.set.assert_called_once_with("test_input")
        
        # Verify the output link's transfer method was called
        self.output_link.transfer.assert_called_once()
        
        # Verify the result is correct
        self.assertEqual(result, "test_input")
        self.assertEqual(self.step.result, "test_input")
        
        # Verify state changes
        self.assertEqual(self.step.state, ComponentState.INACTIVE)
        self.assertFalse(self.step.running)
        
        # Verify circuit breaker interaction
        self.mock_circuit_breaker.record_success.assert_called_once()
    
    def test_execute(self):
        """Run the async test."""
        asyncio.run(self.async_test_execute())
    
    async def async_test_execute_with_circuit_breaker_block(self):
        """Test execute method when circuit breaker blocks execution."""
        # Configure circuit breaker to block execution
        self.mock_circuit_breaker.can_execute.return_value = False
        
        # Execute the step
        result = await self.step.execute()
        
        # Verify execution was blocked
        self.assertIsNone(result)
        self.assertEqual(self.step.state, ComponentState.INACTIVE)
        
        # Verify input link was not accessed
        self.input_link.transfer.assert_not_called()
    
    def test_execute_with_circuit_breaker_block(self):
        """Run the async test for circuit breaker blocking."""
        asyncio.run(self.async_test_execute_with_circuit_breaker_block())
    
    async def async_test_execute_with_exception(self):
        """Test execute method when an exception occurs."""
        # Configure input link to raise an exception
        self.input_link.transfer.side_effect = Exception("Test exception")
        
        # Execute the step and expect an exception
        with self.assertRaises(Exception):
            await self.step.execute()
        
        # Verify state changes
        self.assertEqual(self.step.state, ComponentState.ERROR)
        self.assertFalse(self.step.running)
        
        # Verify circuit breaker interaction
        self.mock_circuit_breaker.record_failure.assert_called_once()
    
    def test_execute_with_exception(self):
        """Run the async test for exception handling."""
        asyncio.run(self.async_test_execute_with_exception())
    
    def test_invoke_alias(self):
        """Test that invoke is an alias for execute."""
        # Create a mock for the execute method
        original_execute = self.step.execute
        
        # Configure the mock to be awaitable and return a value
        async def mock_execute():
            return "test_result"
        
        self.step.execute = mock_execute
        
        # Call invoke
        result = asyncio.run(self.step.invoke())
        
        # Verify result
        self.assertEqual(result, "test_result")
        
        # Restore original method
        self.step.execute = original_execute


if __name__ == '__main__':
    unittest.main() 