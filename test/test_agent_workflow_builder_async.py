#!/usr/bin/env python3
"""
Unit tests for AgentWorkflowBuilder's _safe_execute method.

These tests verify that the _safe_execute method correctly handles
executors with and without execute_async methods.
"""

import unittest
import pytest
import asyncio
import sys
import os
from unittest.mock import MagicMock, AsyncMock, patch

from builder.AgentWorkflowBuilder import AgentWorkflowBuilder

class TestAgentWorkflowBuilderAsync(unittest.TestCase):
    """Test suite for AgentWorkflowBuilder's async functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock executor
        self.mock_executor = MagicMock()
        
        # Create a mock input storage
        self.mock_input_storage = MagicMock()
        self.mock_input_storage.process = AsyncMock(return_value="Mock input response")
        
        # Create the AgentWorkflowBuilder with proper initialization
        with patch('src.Agent.Agent._initialize_llm') as mock_initialize_llm:
            mock_llm = MagicMock()
            mock_llm.invoke = MagicMock(return_value="Mocked response")
            mock_initialize_llm.return_value = mock_llm
            
            # Create the AgentWorkflowBuilder with minimal initialization
            self.builder = AgentWorkflowBuilder(
                executor=self.mock_executor, 
                input_storage=self.mock_input_storage,
                debug_mode=False
            )
            
            # Set the executor directly to ensure it's the mock
            self.builder.executor = self.mock_executor
    
    @pytest.mark.asyncio
    async def test_safe_execute_with_execute_async(self):
        """Test _safe_execute when executor has execute_async method."""
        # Set up the mock executor with an execute_async method
        async_result = {"choices": [{"message": {"content": "Test response"}}]}
        self.mock_executor.execute_async = AsyncMock(return_value=async_result)
        
        # Test messages
        messages = [{"role": "user", "content": "Test message"}]
        
        # Call _safe_execute
        result = await self.builder._safe_execute(self.mock_executor, messages)
        
        # Verify that execute_async was called
        self.mock_executor.execute_async.assert_called_once_with(messages)
        
        # Verify the result
        self.assertEqual(result, async_result)
        
        # Verify that execute was not called
        if hasattr(self.mock_executor, 'execute'):
            self.mock_executor.execute.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_safe_execute_without_execute_async(self):
        """Test _safe_execute when executor doesn't have execute_async method."""
        # Remove execute_async from the mock executor if it exists
        if hasattr(self.mock_executor, 'execute_async'):
            delattr(self.mock_executor, 'execute_async')
        
        # Set up the mock executor with a regular execute method that returns a coroutine
        async def mock_execute(messages):
            return {"choices": [{"message": {"content": "Test response from execute"}}]}
        
        self.mock_executor.execute = MagicMock(side_effect=mock_execute)
        
        # Test messages
        messages = [{"role": "user", "content": "Test message"}]
        
        # Call _safe_execute
        result = await self.builder._safe_execute(self.mock_executor, messages)
        
        # Verify that execute was called
        self.mock_executor.execute.assert_called_once_with(messages)
        
        # Verify the result
        self.assertEqual(result["choices"][0]["message"]["content"], "Test response from execute")
    
    @pytest.mark.asyncio
    async def test_safe_execute_with_all_failures(self):
        """Test _safe_execute when all execution attempts fail."""
        # Set up the mock executor to raise TypeError on all calls
        self.mock_executor.execute_async = MagicMock(side_effect=TypeError("Test error"))
        self.mock_executor.execute = MagicMock(side_effect=TypeError("Test error"))
        
        # Test messages
        messages = [{"role": "user", "content": "Test message"}]
        
        # Call _safe_execute with parameters
        with patch('builtins.print'):  # Suppress print statements
            result = await self.builder._safe_execute(
                self.mock_executor, messages, max_tokens=100, temperature=0.7
            )
        
        # Verify the result is an error string
        self.assertTrue(isinstance(result, str))
        self.assertIn("Error processing request", result)
    
    @pytest.mark.asyncio
    async def test_safe_execute_with_none_executor(self):
        """Test _safe_execute with None executor."""
        # Call _safe_execute with None executor
        result = await self.builder._safe_execute(None, [])
        
        # Verify the result
        self.assertEqual(result, "Error: Executor is not available.")

if __name__ == '__main__':
    unittest.main() 