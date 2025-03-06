"""
Tests for the NanoBrainBuilder with actual implementations.
"""

import os
import sys
import shutil
import unittest
import asyncio
from unittest.mock import patch, MagicMock

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import mock classes
from test.mock_executor import MockExecutorBase
from test.mock_builder import NanoBrainBuilder
from test.mock_tools import (
    StepFileWriter, 
    StepPlanner, 
    StepCoder, 
    StepGitInit, 
    StepContextSearch, 
    StepWebSearch
)

# Helper function to run async tests
def async_test(coro):
    def wrapper(*args, **kwargs):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro(*args, **kwargs))
        finally:
            loop.close()
    return wrapper


class TestNanoBrainBuilderActual(unittest.TestCase):
    """Test case for the NanoBrainBuilder class with actual implementations."""
    
    def setUp(self):
        """Set up the test environment."""
        # Set up mocks
        self.setup_mocks()
        
        self.executor = MockExecutorBase()
        self.builder = NanoBrainBuilder(executor=self.executor)
        
        # Create a temporary test directory
        self.test_dir = os.path.join(os.getcwd(), "test_workflow")
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir)
    
    def setup_mocks(self):
        """Set up mocks for the tests."""
        # Mock subprocess
        subprocess = MagicMock()
        subprocess.run.return_value.returncode = 0
        sys.modules['subprocess'] = subprocess
    
    def tearDown(self):
        """Clean up the test environment."""
        # Remove the temporary test directory
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_initialization(self):
        """Test that the builder initializes correctly."""
        self.assertIsInstance(self.builder, NanoBrainBuilder)
        self.assertEqual(self.builder.executor, self.executor)
        self.assertIsNotNone(self.builder.agent)
        self.assertEqual(len(self.builder._workflow_stack), 0)
        
        # Check that tools are initialized
        self.assertGreaterEqual(len(self.builder.agent.tools), 6)
    
    def test_workflow_stack(self):
        """Test the workflow stack operations."""
        # Test push_workflow
        self.builder.push_workflow("workflow1")
        self.assertEqual(len(self.builder._workflow_stack), 1)
        self.assertEqual(self.builder.get_current_workflow(), "workflow1")
        
        # Test push_workflow again
        self.builder.push_workflow("workflow2")
        self.assertEqual(len(self.builder._workflow_stack), 2)
        self.assertEqual(self.builder.get_current_workflow(), "workflow2")
        
        # Test pop_workflow
        popped = self.builder.pop_workflow()
        self.assertEqual(popped, "workflow2")
        self.assertEqual(len(self.builder._workflow_stack), 1)
        self.assertEqual(self.builder.get_current_workflow(), "workflow1")
        
        # Test pop_workflow again
        popped = self.builder.pop_workflow()
        self.assertEqual(popped, "workflow1")
        self.assertEqual(len(self.builder._workflow_stack), 0)
        self.assertIsNone(self.builder.get_current_workflow())
        
        # Test pop_workflow on empty stack
        popped = self.builder.pop_workflow()
        self.assertIsNone(popped)
    
    @async_test
    async def test_workflow_creation(self):
        """Test creating a workflow."""
        # Create a workflow
        workflow_name = "test_workflow"
        result = await self.builder.create_workflow(workflow_name)
        
        # Check the result
        self.assertTrue(result["success"], msg=f"Failed with error: {result.get('error', 'Unknown error')}")
        self.assertIn("message", result)
        self.assertIn("workflow_path", result)
    
    @async_test
    async def test_step_creation(self):
        """Test creating a step within a workflow."""
        # Create a step
        step_name = "test"
        result = await self.builder.create_step(step_name)
        
        # Check the result
        self.assertTrue(result["success"], msg=f"Failed with error: {result.get('error', 'Unknown error')}")
        self.assertIn("message", result)
        self.assertIn("step_path", result)
        self.assertIn("step_class_name", result)
    
    @async_test
    async def test_step_linking(self):
        """Test linking steps together."""
        # Link steps
        source_step = "source"
        target_step = "target"
        result = await self.builder.link_steps(source_step, target_step)
        
        # Check the result
        self.assertTrue(result["success"], msg=f"Failed with error: {result.get('error', 'Unknown error')}")
        self.assertIn("message", result)
        self.assertIn("link_file", result)
    
    @async_test
    async def test_save_workflow(self):
        """Test saving a workflow."""
        # Save the workflow
        result = await self.builder.save_workflow()
        
        # Check the result
        self.assertTrue(result["success"], msg=f"Failed with error: {result.get('error', 'Unknown error')}")
        self.assertIn("message", result)
    
    @async_test
    async def test_complete_workflow(self):
        """Test a complete workflow creation and execution."""
        # Create a workflow
        workflow_name = "test_workflow"
        workflow_result = await self.builder.create_workflow(workflow_name)
        
        # Create input step
        input_step = "input"
        input_result = await self.builder.create_step(input_step, description="Input step for test workflow")
        
        # Create processing step
        processing_step = "processing"
        processing_result = await self.builder.create_step(processing_step, description="Processing step for test workflow")
        
        # Create output step
        output_step = "output"
        output_result = await self.builder.create_step(output_step, description="Output step for test workflow")
        
        # Link input to processing
        link1_result = await self.builder.link_steps(input_step, processing_step)
        
        # Link processing to output
        link2_result = await self.builder.link_steps(processing_step, output_step)
        
        # Save the workflow
        save_result = await self.builder.save_workflow()
        
        # Verify all operations were successful
        self.assertTrue(workflow_result["success"])
        self.assertTrue(input_result["success"])
        self.assertTrue(processing_result["success"])
        self.assertTrue(output_result["success"])
        self.assertTrue(link1_result["success"])
        self.assertTrue(link2_result["success"])
        self.assertTrue(save_result["success"])


if __name__ == "__main__":
    unittest.main() 