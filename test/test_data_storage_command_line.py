#!/usr/bin/env python3
"""
Test script for DataStorageCommandLine class.
"""

import os
import sys
import asyncio
import unittest
from unittest.mock import patch, MagicMock

# Add the parent directory to the path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.DataStorageCommandLine import DataStorageCommandLine
from src.ExecutorBase import ExecutorBase


class TestDataStorageCommandLine(unittest.TestCase):
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
    
    def test_initialization(self):
        """Test initialization of DataStorageCommandLine."""
        self.assertEqual(self.storage.prompt, "test> ")
        self.assertEqual(self.storage.exit_command, "quit")
        self.assertEqual(self.storage.welcome_message, "Welcome to test!")
        self.assertEqual(self.storage.goodbye_message, "Goodbye from test!")
        self.assertFalse(self.storage.running)
        self.assertIsNone(self.storage.last_input)
        self.assertIsNone(self.storage.last_output)
    
    @patch('builtins.input', return_value="test command")
    @patch('builtins.print')
    def test_process_query(self, mock_print, mock_input):
        """Test processing a query."""
        # Create a test function
        async def run_test():
            # Run the process_query method
            response = await self.storage._process_query("test command")
            
            # Check the response
            self.assertEqual(response, "test command")
            self.assertEqual(self.storage.last_input, "test command")
            self.assertEqual(self.storage.last_output, "test command")
            
            # Check that the response was displayed
            mock_print.assert_called_with("test command")
        
        # Run the test
        asyncio.run(run_test())
    
    @patch('builtins.input', side_effect=["command1", "quit"])
    @patch('builtins.print')
    def test_start_monitoring(self, mock_print, mock_input):
        """Test the start_monitoring method."""
        # Create a future in the current event loop
        async def run_test():
            # Mock the process method
            future = asyncio.Future()
            future.set_result(None)
            self.storage.process = MagicMock(return_value=future)
            
            # Run the start_monitoring method
            await self.storage.start_monitoring()
            
            # Check that the welcome message was displayed
            mock_print.assert_any_call("Welcome to test!")
            
            # Check that the process method was called with the command
            self.storage.process.assert_called_with(["command1"])
            
            # Check that the goodbye message was displayed
            mock_print.assert_any_call("Goodbye from test!")
        
        # Run the test
        asyncio.run(run_test())
    
    @patch('builtins.input', side_effect=KeyboardInterrupt)
    @patch('builtins.print')
    def test_keyboard_interrupt(self, mock_print, mock_input):
        """Test handling of KeyboardInterrupt."""
        # Create a test function
        async def run_test():
            # Run the start_monitoring method
            await self.storage.start_monitoring()
            
            # Check that the welcome message was displayed
            mock_print.assert_any_call("Welcome to test!")
            
            # Check that the keyboard interrupt message was displayed
            mock_print.assert_any_call("\nOperation cancelled by user.")
            
            # Check that the goodbye message was displayed
            mock_print.assert_any_call("Goodbye from test!")
        
        # Run the test
        asyncio.run(run_test())
    
    def test_stop_monitoring(self):
        """Test the stop_monitoring method."""
        # Create a test function
        async def run_test():
            # Set running to True
            self.storage.running = True
            
            # Run the stop_monitoring method
            await self.storage.stop_monitoring()
            
            # Check that running is now False
            self.assertFalse(self.storage.running)
        
        # Run the test
        asyncio.run(run_test())


# Simple demonstration of DataStorageCommandLine
async def demo():
    """Demonstrate the DataStorageCommandLine class."""
    executor = ExecutorBase()
    storage = DataStorageCommandLine(
        executor=executor,
        prompt="nanobrain> ",
        exit_command="exit",
        welcome_message="Welcome to NanoBrain Command Line Interface!",
        goodbye_message="Thank you for using NanoBrain. Goodbye!"
    )
    
    # Start monitoring for user input
    await storage.start_monitoring()


if __name__ == "__main__":
    # If run directly, run the demo
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        asyncio.run(demo())
    # Otherwise, run the tests
    else:
        unittest.main() 