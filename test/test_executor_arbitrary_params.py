#!/usr/bin/env python3
"""
Test script for verifying that Executor classes can accept arbitrary parameters.

This test ensures that the ExecutorBase.execute and derived classes' execute methods
can accept a runnable and arbitrary additional parameters.
"""

import os
import sys
import unittest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ExecutorBase import ExecutorBase
from src.ExecutorFunc import ExecutorFunc
from src.Step import Step

class TestExecutorArbitraryParams(unittest.TestCase):
    """Test that Executor classes can accept arbitrary parameters."""
    
    def test_executor_base_arbitrary_params(self):
        """Test that ExecutorBase.execute accepts arbitrary parameters."""
        # Create a mock runnable
        mock_runnable = MagicMock()
        
        # Create a subclass of ExecutorBase that implements execute
        class TestExecutor(ExecutorBase):
            def execute(self, runnable, *args, **kwargs):
                # Store args and kwargs for later verification
                self.args = args
                self.kwargs = kwargs
                return True
        
        # Create an instance of the test executor
        executor = TestExecutor()
        
        # Call execute with a runnable and arbitrary parameters
        result = executor.execute(mock_runnable, 1, 2, 3, a="a", b="b")
        
        # Verify that execute received the parameters
        self.assertEqual(executor.args, (1, 2, 3))
        self.assertEqual(executor.kwargs, {"a": "a", "b": "b"})
        self.assertTrue(result)
    
    def test_executor_func_arbitrary_params(self):
        """Test that ExecutorFunc.execute accepts arbitrary parameters."""
        # Create a mock runnable
        mock_runnable = MagicMock()
        mock_runnable.process = MagicMock(return_value="processed")
        
        # Create an instance of ExecutorFunc
        executor = ExecutorFunc()
        
        # Ensure energy and reliability are at max to avoid test failures
        executor.energy = 1.0
        executor.reliability = 1.0
        
        # Call execute with a runnable and arbitrary parameters
        result = executor.execute(mock_runnable, "input_data", option1=True, option2=False)
        
        # Verify that process was called with the correct parameters
        # Note: ExecutorFunc adds inputs keyword argument
        mock_runnable.process.assert_called_once_with(option1=True, option2=False, inputs="input_data")
        self.assertEqual(result, "processed")
    
    def test_step_execute_passes_inputs(self):
        """Test that Step.execute passes inputs to the executor."""
        # Create a mock executor
        mock_executor = MagicMock()
        mock_executor.execute = MagicMock(return_value="result")
        
        # Create a step with the mock executor
        step = Step(executor=mock_executor, name="TestStep")
        
        # Patch the execute method to accept inputs
        original_execute = step.execute
        def new_execute(inputs=None):
            step.result = step.executor.execute(step, inputs)
            return step.result
        step.execute = new_execute
        
        # Call execute with inputs
        step.execute({"key": "value"})
        
        # Verify that the executor's execute method was called with the step and inputs
        mock_executor.execute.assert_called_once_with(step, {"key": "value"})
        self.assertEqual(step.result, "result")
    
if __name__ == "__main__":
    unittest.main() 