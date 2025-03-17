"""
Unit tests for AgentWorkflowBuilder.

This module contains tests for the AgentWorkflowBuilder class.
"""

import unittest
import asyncio
import os
from unittest.mock import MagicMock, AsyncMock, patch
from pathlib import Path

from src.ExecutorBase import ExecutorBase
from src.DataUnitBase import DataUnitBase
from src.Agent import Agent
from src.Step import Step
from src.Workflow import Workflow
from builder.AgentWorkflowBuilder import AgentWorkflowBuilder
from builder.AgentCodeWriter import AgentCodeWriter
from langchain_core.prompts import PromptTemplate

# Set testing mode
os.environ['NANOBRAIN_TESTING'] = '1'
# Mock OpenAI API key for tests that need it
os.environ['OPENAI_API_KEY'] = 'test_key'

# Helper function to run async tests with proper cleanup
def run_async_test(coroutine):
    """Run an async test with proper cleanup of pending tasks."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(coroutine)
        
        # Ensure all pending tasks are complete
        pending = asyncio.all_tasks(loop)
        for task in pending:
            if not task.done():
                try:
                    loop.run_until_complete(asyncio.wait_for(task, timeout=0.5))
                except asyncio.TimeoutError:
                    task.cancel()
                    try:
                        # Give it one more chance to cleanup
                        loop.run_until_complete(asyncio.wait_for(task, timeout=0.1))
                    except:
                        pass
        
        return result
    finally:
        loop.close()
        asyncio.set_event_loop(None)


class TestAgentWorkflowBuilder(unittest.TestCase):
    """Test suite for AgentWorkflowBuilder."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock executor
        self.executor = MagicMock()
        self.executor.execute = AsyncMock(return_value="Mocked response")
        
        # Create the builder with test parameters
        self.builder = AgentWorkflowBuilder(
            executor=self.executor, 
            use_code_writer=True, 
            _debug_mode=True
        )
        
        # Mock the _provide_guidance method to avoid real API calls
        self.builder._provide_guidance = AsyncMock(return_value="Mocked guidance")
        
        # Create a mock code writer if needed
        if self.builder.code_writer:
            self.builder.code_writer.process = AsyncMock(return_value="Mocked code response")
            if hasattr(self.builder.code_writer, '_find_existing_class'):
                self.builder.code_writer._find_existing_class = MagicMock(return_value=(None, None))
    
    def tearDown(self):
        """Clean up after tests."""
        self.builder = None
        self.executor = None
    
    async def test_process(self):
        """Test the main process method."""
        # Test with a normal guidance request
        result = await self.builder.process(["How does NanoBrain work?"])
        self.assertIsNotNone(result)
        
        # Verify _provide_guidance was called
        self.builder._provide_guidance.assert_called_once()
    
    async def test_is_requesting_new_class(self):
        """Test that _is_requesting_new_class correctly identifies requests for new classes."""
        # Test with explicit request for new class
        result = self.builder._is_requesting_new_class("Create a new class from scratch")
        self.assertTrue(result)
        
        # Test with request that doesn't specify new class
        result = self.builder._is_requesting_new_class("How do I create a step?")
        self.assertFalse(result)
        
        # Test with request that explicitly mentions custom implementation
        result = self.builder._is_requesting_new_class("I need a custom class implementation")
        self.assertTrue(result)
    
    async def test_should_generate_code(self):
        """Test that _should_generate_code correctly identifies code generation requests."""
        # Test with explicit code generation request
        result = self.builder._should_generate_code("Generate code for a step")
        self.assertTrue(result)
        
        # Test with implicit code generation request
        result = self.builder._should_generate_code("Implement a step that processes text")
        self.assertTrue(result)
        
        # Test with guidance request (no code generation)
        result = self.builder._should_generate_code("What is a NanoBrain step?")
        self.assertFalse(result)
    
    async def test_component_reuse_functionality(self):
        """Test that the builder prioritizes reusing existing components."""
        # Make sure code_writer exists and is properly mocked
        if not self.builder.code_writer:
            self.builder._init_code_writer(self.executor, True)
            self.builder.code_writer.process = AsyncMock(return_value="Mocked code response")
        
        # Mock the _provide_guidance method to call _generate_code_from_user_input
        # This is needed because process() now calls _provide_guidance
        original_provide_guidance = self.builder._provide_guidance
        
        async def mock_provide_guidance(user_input):
            if self.builder._should_generate_code(user_input):
                return await self.builder._generate_code_from_user_input(user_input)
            return "Mocked guidance"
        
        self.builder._provide_guidance = mock_provide_guidance
        
        # Create a request that should trigger code generation
        result = await self.builder.process(["Generate code for a link between steps"])
        
        # Verify the code writer's process method was called at least once
        self.assertTrue(self.builder.code_writer.process.called)
            
        # Verify a response was returned
        self.assertIsNotNone(result)
    
    async def test_suggest_implementation(self):
        """Test the suggest_implementation method with mocked existing class."""
        # Make sure code_writer exists and is properly mocked
        if not self.builder.code_writer:
            self.builder._init_code_writer(self.executor, True)
        
        # Mock the code writer's process method to return a response
        self.builder.code_writer.process = AsyncMock(return_value="Mocked code response")
        
        # Call suggest_implementation
        result = await self.builder.suggest_implementation('TestStep', 'A test step that processes data')
        
        # Verify the builder.code_writer.process was called at least once
        self.assertTrue(self.builder.code_writer.process.called)
        
        # The first call should be for the code generation
        first_call_args = self.builder.code_writer.process.call_args_list[0][0][0]
        self.assertIsInstance(first_call_args, list)
        self.assertIn("TestStep", first_call_args[0])
        self.assertIn("A test step that processes data", first_call_args[0])
        
        # Verify the method returns the same response as code_writer.process
        self.assertEqual(result, "Mocked code response")
    
    def test_process_sync(self):
        """Synchronous wrapper for test_process."""
        run_async_test(self.test_process())
    
    def test_is_requesting_new_class_sync(self):
        """Synchronous wrapper for test_is_requesting_new_class."""
        run_async_test(self.test_is_requesting_new_class())
    
    def test_should_generate_code_sync(self):
        """Synchronous wrapper for test_should_generate_code."""
        run_async_test(self.test_should_generate_code())
    
    def test_component_reuse_functionality_sync(self):
        """Synchronous wrapper for test_component_reuse_functionality."""
        run_async_test(self.test_component_reuse_functionality())
    
    def test_suggest_implementation_sync(self):
        """Synchronous wrapper for test_suggest_implementation."""
        run_async_test(self.test_suggest_implementation())


if __name__ == '__main__':
    unittest.main() 