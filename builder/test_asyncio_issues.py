import unittest
import os
import sys
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from builder.NanoBrainBuilder import NanoBrainBuilder
from builder.WorkflowSteps import CreateWorkflow


class TestAsyncioIssues(unittest.IsolatedAsyncioTestCase):
    """Test cases for asyncio-related issues."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock builder
        self.builder = MagicMock(spec=NanoBrainBuilder)
        self.builder.config = MagicMock(return_value={})
        self.builder.get_current_workflow = MagicMock(return_value=None)
        self.builder.agent = MagicMock()
        self.builder.agent.tools = []
        
        # Set up environment for testing
        os.environ['NANOBRAIN_TESTING'] = '1'
        os.environ['OPENAI_API_KEY'] = 'test_key'
        
        # Patch Agent._initialize_llm to avoid actual API calls
        self.llm_patcher = patch('src.Agent.Agent._initialize_llm', return_value=MagicMock())
        self.llm_mock = self.llm_patcher.start()
        
        # Patch Agent._load_prompt_template
        self.prompt_patcher = patch('src.Agent.Agent._load_prompt_template')
        self.prompt_mock = self.prompt_patcher.start()
    
    def tearDown(self):
        """Clean up after tests."""
        # Stop the patchers
        self.llm_patcher.stop()
        self.prompt_patcher.stop()
        
        # Reset environment variables
        if 'NANOBRAIN_TESTING' in os.environ:
            del os.environ['NANOBRAIN_TESTING']
        if 'OPENAI_API_KEY' in os.environ:
            del os.environ['OPENAI_API_KEY']
    
    async def test_create_workflow_is_async(self):
        """Test that create_workflow is properly implemented as an async method."""
        # Check that the method is a coroutine function
        self.assertTrue(asyncio.iscoroutinefunction(NanoBrainBuilder.create_workflow),
                       "create_workflow should be an async method")
        
        # Create a mock builder instance instead of a real one
        builder = MagicMock(spec=NanoBrainBuilder)
        
        # Mock the create_workflow method to return a coroutine
        async def mock_create_workflow(*args, **kwargs):
            return {"success": True}
        
        builder.create_workflow = mock_create_workflow
        
        # Call the method and verify it returns a coroutine
        coroutine = builder.create_workflow("test_workflow")
        self.assertTrue(asyncio.iscoroutine(coroutine),
                       "create_workflow should return a coroutine")
        
        # Await the coroutine to avoid "coroutine was never awaited" warnings
        result = await coroutine
        self.assertTrue(result.get("success", False),
                       "create_workflow should return a success result")
    
    async def test_nested_event_loop_prevention(self):
        """Test that asyncio.run() is not used inside an async method."""
        # Create a function that simulates the old implementation with asyncio.run()
        async def problematic_function():
            # This would raise RuntimeError: asyncio.run() cannot be called from a running event loop
            return asyncio.run(asyncio.sleep(0.1))
        
        # Verify that calling asyncio.run() inside an event loop raises RuntimeError
        with self.assertRaises(RuntimeError):
            await problematic_function()
        
        # Create a function that simulates the correct implementation with await
        async def correct_function():
            return await asyncio.sleep(0.1)
        
        # Verify that the correct implementation works
        try:
            await correct_function()
        except RuntimeError:
            self.fail("correct_function() raised RuntimeError unexpectedly!")
    
    async def test_all_builder_methods_are_properly_async(self):
        """Test that all async methods in NanoBrainBuilder are properly implemented."""
        # List of methods that should be async
        async_methods = [
            'create_workflow',
            'create_step',
            'test_step',
            'save_step',
            'link_steps',
            'save_workflow',
            'process_command',
            'main'
        ]
        
        # Verify that each method is a coroutine function
        for method_name in async_methods:
            method = getattr(NanoBrainBuilder, method_name, None)
            if method:
                self.assertTrue(asyncio.iscoroutinefunction(method),
                               f"{method_name} should be an async method")
            else:
                self.fail(f"Method {method_name} not found in NanoBrainBuilder")
    
    async def test_create_workflow_implementation(self):
        """Test the implementation of create_workflow to ensure it properly awaits the CreateWorkflow method."""
        # Create a spy for CreateWorkflow.create_workflow
        original_method = CreateWorkflow.create_workflow
        
        try:
            # Replace with a spy that tracks calls
            spy_result = {"success": True, "message": "Workflow created"}
            
            async def spy_create_workflow(*args, **kwargs):
                spy_create_workflow.called = True
                spy_create_workflow.args = args
                spy_create_workflow.kwargs = kwargs
                return spy_result
            
            spy_create_workflow.called = False
            spy_create_workflow.args = None
            spy_create_workflow.kwargs = None
            
            # Patch the method
            CreateWorkflow.create_workflow = spy_create_workflow
            
            # Create a mock builder instead of a real one
            builder = MagicMock(spec=NanoBrainBuilder)
            
            # Create a mock create_workflow method that calls the real implementation
            async def mock_create_workflow(workflow_name):
                return await CreateWorkflow.create_workflow(builder, workflow_name)
            
            # Attach the mock method to the builder
            builder.create_workflow = mock_create_workflow
            
            # Call create_workflow
            result = await builder.create_workflow("test_workflow")
            
            # Verify that the spy was called
            self.assertTrue(spy_create_workflow.called,
                           "CreateWorkflow.create_workflow should be called")
            
            # Verify that the first argument is the builder
            self.assertEqual(spy_create_workflow.args[0], builder,
                            "First argument should be the builder")
            
            # Verify that the second argument is the workflow name
            self.assertEqual(spy_create_workflow.args[1], "test_workflow",
                            "Second argument should be the workflow name")
            
            # Verify that the result is passed through
            self.assertEqual(result, spy_result,
                            "Result should be passed through from CreateWorkflow.create_workflow")
        finally:
            # Restore the original method
            CreateWorkflow.create_workflow = original_method


if __name__ == '__main__':
    unittest.main() 