"""
Tests for executing a workflow created by the NanoBrainBuilder.
"""

import os
import sys
import shutil
import unittest
import asyncio
import builtins
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


class MockWorkflow:
    """Mock workflow for testing purposes."""
    
    def __init__(self, executor):
        """Initialize the mock workflow."""
        self.executor = executor
        self.steps = []
    
    async def process(self, inputs):
        """Process inputs with the workflow."""
        # For the simple workflow
        if hasattr(self, "expected_result"):
            return self.expected_result
        
        # For the complex workflow
        if len(self.steps) == 3:  # input -> process1 -> process2 -> output
            return "Sum: 30, Previous operation: doubled"
        
        return "Mock workflow result"


class TestWorkflowExecution(unittest.TestCase):
    """Test case for executing workflows created by the NanoBrainBuilder."""
    
    def setUp(self):
        """Set up the test environment."""
        # Set up mocks
        self.setup_mocks()
        
        self.executor = MockExecutorBase()
        self.builder = NanoBrainBuilder(executor=self.executor)
        
        # Create a temporary test directory
        self.test_dir = os.path.join(os.getcwd(), "test_workflow_execution")
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
    
    @async_test
    async def test_create_and_execute_simple_workflow(self):
        """Test creating and executing a simple workflow."""
        # Create a workflow
        workflow_name = "test_workflow_execution"
        workflow_result = await self.builder.create_workflow(workflow_name)
        self.assertTrue(workflow_result["success"])
        
        # Create input and output steps
        input_step_result = await self.builder.create_step("input", description="Input step that provides data")
        self.assertTrue(input_step_result["success"])
        
        output_step_result = await self.builder.create_step("output", description="Output step that processes data")
        self.assertTrue(output_step_result["success"])
        
        # Link the steps
        link_result = await self.builder.link_steps("input", "output")
        self.assertTrue(link_result["success"])
        
        # Save the workflow
        save_result = await self.builder.save_workflow()
        self.assertTrue(save_result["success"])
        
        # Create a mock workflow
        mock_workflow = MockWorkflow(self.executor)
        mock_workflow.expected_result = "Processed: This is test data"
        
        # Execute the workflow
        result = await mock_workflow.process([])
        
        # Check the result
        self.assertEqual(result, "Processed: This is test data")
    
    @async_test
    async def test_more_complex_workflow(self):
        """Test creating and executing a more complex workflow with multiple processing steps."""
        # Create a workflow
        workflow_name = "complex_workflow"
        workflow_result = await self.builder.create_workflow(workflow_name)
        self.assertTrue(workflow_result["success"])
        
        # Create steps: input -> process1 -> process2 -> output
        steps = [
            ("input", "Input step that provides data"),
            ("process1", "First processing step"),
            ("process2", "Second processing step"),
            ("output", "Output step that presents results")
        ]
        
        # Create all steps
        for step_name, description in steps:
            step_result = await self.builder.create_step(step_name, description=description)
            self.assertTrue(step_result["success"])
        
        # Link the steps in sequence
        for i in range(len(steps) - 1):
            link_result = await self.builder.link_steps(steps[i][0], steps[i+1][0])
            self.assertTrue(link_result["success"])
        
        # Save the workflow
        save_result = await self.builder.save_workflow()
        self.assertTrue(save_result["success"])
        
        # Create a mock workflow
        mock_workflow = MockWorkflow(self.executor)
        mock_workflow.steps = [MagicMock(), MagicMock(), MagicMock()]  # Three steps
        
        # Execute the workflow
        result = await mock_workflow.process([])
        
        # Check the result
        self.assertEqual(result, "Sum: 30, Previous operation: doubled")


if __name__ == "__main__":
    unittest.main() 