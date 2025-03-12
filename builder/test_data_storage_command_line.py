"""
Unit tests for DataStorageCommandLine.

This module contains tests for the DataStorageCommandLine class.
"""

import unittest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from io import StringIO

from src.ExecutorBase import ExecutorBase
from src.DataStorageCommandLine import DataStorageCommandLine


class TestDataStorageCommandLine(unittest.IsolatedAsyncioTestCase):
    """Test cases for the DataStorageCommandLine class."""
    
    def setUp(self):
        """Set up test environment."""
        self.executor = MagicMock()
        self.storage = DataStorageCommandLine(executor=self.executor)
    
    def test_initialization(self):
        """Test that the storage initializes correctly."""
        self.assertEqual(self.storage.input, None)
        self.assertEqual(self.storage.output, None)
        self.assertEqual(self.storage.trigger, None)
        self.assertEqual(self.storage.last_query, None)
        self.assertEqual(self.storage.last_response, None)
        self.assertEqual(self.storage.processing_history, [])
        self.assertEqual(self.storage.max_history_size, 10)
        self.assertEqual(self.storage.last_input, None)
        self.assertEqual(self.storage.last_output, None)
    
    @patch('sys.stdout', new_callable=StringIO)
    def test_display_response(self, mock_stdout):
        """Test response display functionality."""
        test_output = "Test output message"
        self.storage.display_response(test_output)
        
        self.assertEqual(mock_stdout.getvalue().strip(), test_output)
    
    @patch('builtins.input', return_value="test input")
    async def test_get_user_input(self, mock_input):
        """Test input retrieval."""
        result = await self.storage._get_user_input()
        
        mock_input.assert_called_once_with(self.storage.prompt)
        self.assertEqual(result, "test input")
    
    async def test_process_query(self):
        """Test query processing."""
        test_input = "test command"
        
        # Mock _handle_command
        self.storage._handle_command = AsyncMock(return_value="test response")
        
        # Process the query
        result = await self.storage._process_query(test_input)
        
        # Verify the processing
        self.assertEqual(self.storage.last_input, test_input)
        self.assertEqual(self.storage.last_output, "test response")
        self.assertEqual(result, "test response")
    
    async def test_handle_command(self):
        """Test command handling."""
        # Test empty command
        result = await self.storage._handle_command("")
        self.assertIsNone(result)
        
        # Test help command
        result = await self.storage._handle_command("help")
        self.assertIn("Available commands:", result)
        
        # Test unknown command
        result = await self.storage._handle_command("unknown")
        self.assertEqual(result, "unknown")
    
    def test_display_help(self):
        """Test help display."""
        help_text = self.storage._display_help()
        self.assertIn("Available commands:", help_text)
        self.assertIn("help - Display this help message", help_text)
        self.assertIn("exit - Exit the current session", help_text)
    
    @patch('builtins.input', side_effect=KeyboardInterrupt)
    async def test_keyboard_interrupt_handling(self, mock_input):
        """Test handling of keyboard interrupts."""
        with self.assertRaises(KeyboardInterrupt):
            await self.storage._get_user_input()
        mock_input.assert_called_once()
    
    @patch('builtins.input', side_effect=EOFError)
    async def test_eof_handling(self, mock_input):
        """Test handling of EOF."""
        with self.assertRaises(EOFError):
            await self.storage._get_user_input()
        mock_input.assert_called_once()
    
    def test_update_history(self):
        """Test history update functionality."""
        test_query = "test query"
        test_response = "test response"
        
        self.storage._update_history(test_query, test_response)
        
        self.assertEqual(len(self.storage.processing_history), 1)
        entry = self.storage.processing_history[0]
        self.assertEqual(entry['query'], test_query)
        self.assertEqual(entry['response'], test_response)
        self.assertIn('timestamp', entry)
    
    async def test_start_stop_monitoring(self):
        """Test monitoring start/stop functionality."""
        # Mock _get_user_input to return exit command
        self.storage._get_user_input = AsyncMock(return_value="exit")
        
        # Start monitoring
        await self.storage.start_monitoring()
        
        # Verify state after monitoring stops
        self.assertFalse(self.storage.running)
        
        # Test stop monitoring
        await self.storage.stop_monitoring()
        self.assertFalse(self.storage.running)


if __name__ == '__main__':
    unittest.main() 