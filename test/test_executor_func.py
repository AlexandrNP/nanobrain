#!/usr/bin/env python3
"""
Unit tests for the ExecutorFunc class.

These tests verify both synchronous and asynchronous execution methods
after the refactoring changes.
"""

import unittest
import pytest
import asyncio
import sys
import os
from unittest.mock import MagicMock, patch

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.ExecutorFunc import ExecutorFunc
from src.ExecutorBase import ExecutorBase

class TestExecutorFunc(unittest.TestCase):
    """Test suite for ExecutorFunc class."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_function = lambda x: f"Processed: {x}"
        self.executor = ExecutorFunc(function=self.test_function)
        self.executor.base_executor.runnable_types.add('str')
        self.executor.base_executor.energy_level = 1.0
        self.executor.base_executor.energy_per_execution = 0.1

    # Helper method for running async functions in tests
    async def async_helper(self, coroutine):
        return await coroutine

    def test_execute(self):
        """Test the synchronous execute method."""
        # Test successful execution
        result = self.executor.execute("Test input")
        self.assertEqual(result, "Processed: Test input")
        
        # Test execution with an unsupported type
        self.executor.base_executor.runnable_types.clear()
        result = self.executor.execute("Test input")
        self.assertIsNone(result)
        
        # Reset runnable types for other tests
        self.executor.base_executor.runnable_types.add('str')

    def test_execute_with_no_function(self):
        """Test execute with no function provided."""
        executor = ExecutorFunc()  # No function provided
        executor.base_executor.runnable_types.add('str')
        executor.base_executor.energy_level = 1.0
        
        result = executor.execute("Test input")
        self.assertIsNone(result)

    def test_execute_with_exception(self):
        """Test execute with a function that raises an exception."""
        def failing_function(x):
            raise ValueError("Test exception")
            
        executor = ExecutorFunc(function=failing_function)
        executor.base_executor.runnable_types.add('str')
        executor.base_executor.energy_level = 1.0
        
        with self.assertRaises(ValueError):
            executor.execute("Test input")

    @pytest.mark.asyncio
    async def test_execute_async(self):
        """Test the asynchronous execute_async method."""
        # Test successful async execution
        result = await self.executor.execute_async("Test async input")
        self.assertEqual(result, "Processed: Test async input")
        
        # Test async execution with complex input
        complex_input = [{"role": "user", "content": "Hello, world!"}]
        
        # Add list type to runnable types
        self.executor.base_executor.runnable_types.add('list')
        
        result = await self.executor.execute_async(complex_input)
        # The result should be "Processed: " followed by the string representation of complex_input
        self.assertEqual(result, f"Processed: {complex_input}")

    def test_reliability_threshold(self):
        """Test the reliability threshold feature."""
        # Mock the base executor's get_modulator_effect to return low reliability
        self.executor.base_executor.get_modulator_effect = MagicMock(return_value=0.1)
        
        # Set a very high threshold to ensure execution fails
        self.executor.reliability_threshold = 0.9
        
        # Patch random.random to return a value that will cause execution to fail
        with patch('random.random', return_value=0.9):
            result = self.executor.execute("Test input")
            self.assertIsNone(result)

    def test_property_delegation(self):
        """Test property delegation to base executor."""
        # Test getter/setter for energy_level
        self.executor.energy_level = 0.5
        self.assertEqual(self.executor.energy_level, 0.5)
        
        # Test getter/setter for energy_per_execution
        self.executor.energy_per_execution = 0.2
        self.assertEqual(self.executor.energy_per_execution, 0.2)
        
        # Test getter/setter for recovery_rate
        self.executor.recovery_rate = 0.3
        self.assertEqual(self.executor.recovery_rate, 0.3)
        
        # Test runnable_types property
        self.assertIn('str', self.executor.runnable_types)

    def test_method_delegation(self):
        """Test method delegation to base executor."""
        # Test can_execute delegation
        self.assertTrue(self.executor.can_execute('str'))
        
        # Test recover_energy delegation
        original_energy = self.executor.energy_level
        self.executor.energy_level = 0.5
        self.executor.recover_energy()
        self.assertGreater(self.executor.energy_level, 0.5)
        
        # Reset energy level
        self.executor.energy_level = original_energy
        
        # Test get_modulator_effect delegation
        self.executor.base_executor.get_modulator_effect = MagicMock(return_value=0.7)
        self.assertEqual(self.executor.get_modulator_effect("reliability"), 0.7)
        
        # Test get_config delegation
        with patch.object(self.executor.base_executor, 'get_config', return_value={"key": "value"}):
            config = self.executor.get_config()
            self.assertEqual(config, {"key": "value"})
        
        # Test update_config delegation
        with patch.object(self.executor.base_executor, 'update_config', return_value=True):
            result = self.executor.update_config({"key": "new_value"})
            self.assertTrue(result)

if __name__ == '__main__':
    unittest.main() 