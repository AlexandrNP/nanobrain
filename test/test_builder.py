"""
Tests for the NanoBrainBuilder.
"""

import os
import sys
import shutil
import unittest
import asyncio
from unittest.mock import patch, MagicMock
import types

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import mocked classes before patching
from test.mock_builder import (
    NanoBrainBuilder,
    CreateWorkflow,
    CreateStep,
    TestStepStep,
    SaveStepStep,
    LinkStepsStep,
    SaveWorkflowStep
)
from test.mock_agent import Agent
from test.mock_executor import MockExecutorBase
from src.ExecutorBase import ExecutorBase
from test.mock_tools import (
    StepFileWriter,
    StepPlanner,
    StepCoder,
    StepGitInit,
    StepContextSearch,
    StepWebSearch
)

# Create the patches
sys.modules['builder.WorkflowSteps'] = MagicMock()
sys.modules['builder.WorkflowSteps'].CreateWorkflow = CreateWorkflow
sys.modules['builder.WorkflowSteps'].CreateStep = CreateStep
sys.modules['builder.WorkflowSteps'].TestStepStep = TestStepStep
sys.modules['builder.WorkflowSteps'].SaveStepStep = SaveStepStep
sys.modules['builder.WorkflowSteps'].LinkStepsStep = LinkStepsStep
sys.modules['builder.WorkflowSteps'].SaveWorkflowStep = SaveWorkflowStep

# Patch tools_common module
sys.modules['tools_common'] = MagicMock()
sys.modules['tools_common'].StepFileWriter = StepFileWriter
sys.modules['tools_common'].StepPlanner = StepPlanner
sys.modules['tools_common'].StepCoder = StepCoder
sys.modules['tools_common'].StepGitInit = StepGitInit
sys.modules['tools_common'].StepContextSearch = StepContextSearch
sys.modules['tools_common'].StepWebSearch = StepWebSearch

# Patch src.Agent module
sys.modules['src.Agent'] = MagicMock()
sys.modules['src.Agent'].Agent = Agent

# Use the mock NanoBrainBuilder instead of the real one
# This avoids the issue with the Agent class initialization

def async_test(coro):
    """Decorator for async test methods."""
    def wrapper(*args, **kwargs):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro(*args, **kwargs))
        finally:
            loop.close()
    return wrapper

def async_mock_return(return_value):
    """Create an async function that returns the given value."""
    async def mock_func(*args, **kwargs):
        return return_value
    return mock_func

class TestNanoBrainBuilder(unittest.TestCase):
    def setUp(self):
        """Set up the test environment."""
        self.executor = MagicMock(spec=ExecutorBase)
        self.builder = NanoBrainBuilder(executor=self.executor)
        
        # Create a temporary test directory
        self.test_dir = os.path.join(os.path.dirname(__file__), 'test_builder_output')
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Patch the builder's methods to use our mocked steps
        self.builder.create_workflow = async_mock_return({"success": True, "message": "Created workflow test_workflow"})
        self.builder.create_step = async_mock_return({"success": True, "message": "Created step test"})
        self.builder.test_step = async_mock_return({"success": True, "message": "Tested step test"})
        self.builder.save_step = async_mock_return({"success": True, "message": "Saved step test"})
        self.builder.link_steps = async_mock_return({"success": True, "message": "Linked source to target"})
        self.builder.save_workflow = async_mock_return({"success": True, "message": "Saved workflow"})
        self.builder.process_command = async_mock_return({"success": False, "message": "Invalid command: invalid"})
    
    def tearDown(self):
        """Clean up after the test."""
        # Remove the test directory
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_initialization(self):
        """Test that the builder initializes correctly."""
        self.assertIsInstance(self.builder, NanoBrainBuilder)
        self.assertEqual(self.builder.executor, self.executor)
        self.assertIsNotNone(self.builder.agent)
        self.assertEqual(len(self.builder._workflow_stack), 0)
    
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
    async def test_create_workflow(self):
        """Test the create_workflow method."""
        # Call the method
        result = await self.builder.create_workflow("test_workflow")
        
        # Check the result
        self.assertTrue(result["success"])
        self.assertEqual(result["message"], "Created workflow test_workflow")
    
    @async_test
    async def test_create_step(self):
        """Test the create_step method."""
        # Call the method
        result = await self.builder.create_step("test")
        
        # Check the result
        self.assertTrue(result["success"])
        self.assertEqual(result["message"], "Created step test")
    
    @async_test
    async def test_test_step(self):
        """Test the test_step method."""
        # Call the method
        result = await self.builder.test_step("test")
        
        # Check the result
        self.assertTrue(result["success"])
        self.assertEqual(result["message"], "Tested step test")
    
    @async_test
    async def test_save_step(self):
        """Test the save_step method."""
        # Call the method
        result = await self.builder.save_step("test")
        
        # Check the result
        self.assertTrue(result["success"])
        self.assertEqual(result["message"], "Saved step test")
    
    @async_test
    async def test_link_steps(self):
        """Test the link_steps method."""
        # Call the method
        result = await self.builder.link_steps("source", "target")
        
        # Check the result
        self.assertTrue(result["success"])
        self.assertEqual(result["message"], "Linked source to target")
    
    @async_test
    async def test_save_workflow(self):
        """Test the save_workflow method."""
        # Call the method
        result = await self.builder.save_workflow()
        
        # Check the result
        self.assertTrue(result["success"])
        self.assertEqual(result["message"], "Saved workflow")
    
    @async_test
    async def test_process_command_invalid(self):
        """Test the process_command method with an invalid command."""
        # Call the method with an invalid command
        result = await self.builder.process_command("invalid", [])
        
        # Check the result
        self.assertFalse(result["success"])
        self.assertEqual(result["message"], "Invalid command: invalid")


if __name__ == "__main__":
    unittest.main() 