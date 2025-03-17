#!/usr/bin/env python3
"""
Test script for testing the execute_async method in ExecutorFunc.
"""

import asyncio
import pytest
import os
import sys

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.ExecutorFunc import ExecutorFunc

def process_input(runnable):
    """Test function that returns a string with the input."""
    if hasattr(runnable, 'get'):
        return f"Processed: {runnable.get()}"
    return f"Processed: {str(runnable)}"

@pytest.mark.asyncio
async def test_executor_async():
    """Test the execute_async method in ExecutorFunc."""
    # Create an ExecutorFunc with our test function
    executor = ExecutorFunc(function=process_input)
    
    # Add types that can be executed
    executor.base_executor.runnable_types.add('str')
    executor.base_executor.runnable_types.add('list')
    executor.base_executor.runnable_types.add('dict')
    
    # Set energy level
    executor.base_executor.energy_level = 1.0
    executor.base_executor.energy_per_execution = 0.1
    
    # Test the synchronous execute method
    sync_result = executor.execute("Test input")
    assert sync_result == "Processed: Test input"
    
    # Test the async execute_async method
    async_result = await executor.execute_async("Test input async")
    assert async_result == "Processed: Test input async"
    
    # Test that the original problem scenario works
    message = [{"role": "user", "content": "Hello, world!"}]
    async_complex_result = await executor.execute_async(message)
    assert async_complex_result == f"Processed: {message}"
    
    return True

if __name__ == "__main__":
    pytest.main(["-v", "--asyncio-mode=strict", __file__]) 