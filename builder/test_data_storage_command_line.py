"""
Unit tests for DataStorageCommandLine.

This module contains tests for the DataStorageCommandLine class.
"""

import unittest
import asyncio
import time
from unittest.mock import MagicMock, AsyncMock, patch
from io import StringIO

from src.ExecutorBase import ExecutorBase
from src.DataStorageCommandLine import DataStorageCommandLine
from src.DataUnitString import DataUnitString


class TestDataStorageCommandLine(unittest.IsolatedAsyncioTestCase):
    """Test cases for the DataStorageCommandLine class."""
    
    def setUp(self):
        """Set up test environment."""
        self.executor = MagicMock()
        self.storage = DataStorageCommandLine(executor=self.executor)
        self.storage.output = DataUnitString()
    
    def test_initialization(self):
        """Test that the storage initializes correctly."""
        self.assertFalse(self.storage._monitoring)
        self.assertIsNone(self.storage._monitor_task)
        self.assertIsNone(self.storage.agent_builder)
        self.assertEqual(self.storage.history, [])
        self.assertEqual(self.storage.history_size, 20)  # Default value
        self.assertEqual(self.storage.prompt, '> ')  # Default value
    
    @patch('builtins.print')
    def test_display_output(self, mock_print):
        """Test output display functionality."""
        test_output = "Test output message"
        # Since there's no display_response method, we'll test _force_output_change
        self.storage._force_output_change(test_output)
        
        # Check that the output was set
        self.assertEqual(self.storage.output.get(), test_output)
    
    @patch('asyncio.get_event_loop')
    async def test_input_processing(self, mock_get_loop):
        """Test input processing."""
        # Mock the run_in_executor to return a test input
        mock_loop = AsyncMock()
        mock_loop.run_in_executor.return_value = "test input"
        mock_get_loop.return_value = mock_loop
        
        # Test processing directly
        result = await self.storage.process("test input")
        
        # Check result is processed correctly
        self.assertEqual(result, "test input")  # Default behavior is to return query
    
    async def test_process_query(self):
        """Test query processing."""
        test_input = "test input"
        # Test the _process_query method directly
        result = self.storage._process_query(test_input)
        
        # Default behavior is to return the query
        self.assertEqual(result, test_input)
        
        # Test class pattern detection
        class_input = "<TestClass>>Create a test class"
        result = self.storage._process_query(class_input)
        
        # Should format a prompt for generating class code
        self.assertIn("Generate a class named TestClass", result)
        self.assertIn("Create a test class", result)
        
        # Test with more complex instructions
        complex_input = "<ComplexClass>>Create a class with methods for data processing and validation. Include error handling."
        result = self.storage._process_query(complex_input)
        
        # Should format a prompt with the complex instructions
        self.assertIn("Generate a class named ComplexClass", result)
        self.assertIn("Create a class with methods for data processing and validation", result)
        self.assertIn("Include error handling", result)
    
    def test_add_to_history(self):
        """Test history update functionality."""
        test_query = "test query"
        test_response = "test response"
        
        # Add to history
        self.storage._add_to_history(test_query, test_response)
        
        # Check history was updated
        self.assertEqual(len(self.storage.history), 1)
        self.assertEqual(self.storage.history[0]['query'], test_query)
        self.assertEqual(self.storage.history[0]['response'], test_response)
        self.assertIn('timestamp', self.storage.history[0])
    
    def test_get_history(self):
        """Test history retrieval."""
        # Add test entries
        self.storage._add_to_history("query1", "response1")
        self.storage._add_to_history("query2", "response2")
        
        # Get history
        history = self.storage.get_history()
        
        # Check history
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]['query'], "query1")
        self.assertEqual(history[1]['query'], "query2")
    
    async def test_force_output_change(self):
        """Test the force_output_change method."""
        test_data = "test data"
        
        # Call force_output_change
        result = self.storage._force_output_change(test_data)
        
        # Check result
        self.assertTrue(result)
        self.assertEqual(self.storage.output.get(), test_data)
        
        # Test with direct agent_builder reference
        self.storage.agent_builder = AsyncMock()
        self.storage.agent_builder.process = AsyncMock()
        
        # Call force_output_change again
        self.storage._force_output_change(test_data)
        
        # Check that agent_builder.process was called
        self.storage.agent_builder.process.assert_called_once()
    
    def test_start_stop_monitoring(self):
        """Test monitoring start/stop functionality."""
        # We can't actually test the full monitoring loop easily, 
        # but we can test the state changes
        
        # Check initial state
        self.assertFalse(self.storage._monitoring)
        
        # Stop monitoring (should be safe when not monitoring)
        self.storage.stop_monitoring()
        self.assertFalse(self.storage._monitoring)


if __name__ == '__main__':
    unittest.main() 