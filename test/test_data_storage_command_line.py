#!/usr/bin/env python3
"""
Test script for DataStorageCommandLine class.
"""

import os
import sys
import asyncio
import unittest
import time
from unittest.mock import patch, MagicMock, AsyncMock
from io import StringIO

# Add the parent directory to the path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.DataStorageCommandLine import DataStorageCommandLine
from src.ExecutorBase import ExecutorBase
from src.DataUnitString import DataUnitString


class TestDataStorageCommandLine(unittest.IsolatedAsyncioTestCase):
    """Test cases for DataStorageCommandLine class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.executor = ExecutorBase()
        self.storage = DataStorageCommandLine(
            executor=self.executor,
            prompt="test> ",
            exit_command="quit",
            welcome_message="Welcome to test!",
            goodbye_message="Goodbye from test!"
        )
        self.storage.output = DataUnitString()
    
    def test_initialization(self):
        """Test initialization of DataStorageCommandLine."""
        self.assertEqual(self.storage.prompt, "test> ")
        self.assertEqual(self.storage.exit_command, "quit")
        self.assertEqual(self.storage.welcome_message, "Welcome to test!")
        self.assertEqual(self.storage.goodbye_message, "Goodbye from test!")
        self.assertFalse(self.storage.monitoring)
        self.assertIsNone(self.storage.monitor_task)
        self.assertIsNone(self.storage.agent_builder)
        self.assertEqual(self.storage.history, [])
    
    @patch('builtins.print')
    def test_process_query(self, mock_print):
        """Test processing a query."""
        # Test the _process_query method directly
        result = self.storage._process_query("test command")
        
        # Check the response
        self.assertEqual(result, "test command")
        
        # Test class pattern detection
        class_input = "<TestClass>>Create a test class"
        result = self.storage._process_query(class_input)
        
        # Should format a prompt for generating class code
        self.assertIn("Generate a class named TestClass", result)
        self.assertIn("Create a test class", result)
    
    @patch('asyncio.get_event_loop')
    async def test_start_monitoring(self, mock_get_loop):
        """Test the start_monitoring method."""
        # Mock the loop and run_in_executor to provide input
        mock_loop = AsyncMock()
        mock_loop.run_in_executor.side_effect = ["command1", "quit"]
        mock_get_loop.return_value = mock_loop
        
        # Mock the process method
        self.storage.process = AsyncMock()
        
        # Start monitoring (will exit after processing "quit")
        await self.storage.start_monitoring()
        
        # Check that the process method was called with the first command
        self.storage.process.assert_called_with("command1")
    
    @patch('asyncio.get_event_loop')
    async def test_keyboard_interrupt(self, mock_get_loop):
        """Test handling of KeyboardInterrupt."""
        # Mock the loop and run_in_executor to raise KeyboardInterrupt
        mock_loop = AsyncMock()
        mock_loop.run_in_executor.side_effect = KeyboardInterrupt
        mock_get_loop.return_value = mock_loop
        
        # Patch print
        with patch('builtins.print') as mock_print:
            # Start monitoring (will exit after KeyboardInterrupt)
            await self.storage.start_monitoring()
            
            # Check that the interrupt message was printed
            mock_print.assert_any_call("\nInterrupt detected. Exiting.")
    
    async def test_stop_monitoring(self):
        """Test the stop_monitoring method."""
        # Start monitoring
        self.storage.monitoring = True
        
        # Create a mock task
        mock_task = MagicMock()
        mock_task.cancel = MagicMock()
        self.storage.monitor_task = mock_task
        
        # Stop monitoring
        self.storage.stop_monitoring()
        
        # Check that monitoring was stopped
        self.assertFalse(self.storage.monitoring)
        mock_task.cancel.assert_called_once()
    
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
    
    @patch('builtins.print')
    async def test_force_output_change(self, mock_print):
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

    def test_name_attribute(self):
        """Test that the name attribute is properly set."""
        # Test with the default name
        dscl_default = DataStorageCommandLine(executor=self.executor)
        self.assertEqual(dscl_default.name, "CommandLine", "DataStorageCommandLine should use default name when none provided")
        
        # Test with a custom name
        custom_name = "CustomCommandLine"
        dscl_custom = DataStorageCommandLine(executor=self.executor, name=custom_name)
        self.assertEqual(dscl_custom.name, custom_name, "DataStorageCommandLine should use the custom name provided in constructor")


async def demo():
    """Demo function to show use of the DataStorageCommandLine class."""
    # Create an instance of the class
    executor = ExecutorBase()
    storage = DataStorageCommandLine(executor=executor)
    
    # Start monitoring (will exit when the user types "exit")
    await storage.start_monitoring()


if __name__ == "__main__":
    # Run the demo
    asyncio.run(demo()) 