"""
Tests for the NanoBrainBuilder.
"""

import os
import sys
import shutil
import unittest
import asyncio
from unittest.mock import patch, MagicMock

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import mocked classes before patching
from test.mock_builder import (
    CreateWorkflowStep, 
    CreateStepStep, 
    TestStepStep, 
    SaveStepStep, 
    LinkStepsStep, 
    SaveWorkflowStep
)
from test.mock_agent import Agent
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
sys.modules['builder.WorkflowSteps'].CreateWorkflowStep = CreateWorkflowStep
sys.modules['builder.WorkflowSteps'].CreateStepStep = CreateStepStep
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

from builder import NanoBrainBuilder
from src.ExecutorBase import ExecutorBase


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


class TestNanoBrainBuilder(unittest.TestCase):
    """Test case for the NanoBrainBuilder class."""
    
    def setUp(self):
        """Set up the test environment."""
        self.executor = MagicMock(spec=ExecutorBase)
        self.builder = NanoBrainBuilder(executor=self.executor)
        
        # Create a temporary test directory
        self.test_dir = os.path.join(os.getcwd(), "test_workflow")
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir)
    
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
        self.assertEqual(result["message"], "Created step StepTest")
    
    @async_test
    async def test_test_step(self):
        """Test the test_step method."""
        # Call the method
        result = await self.builder.test_step("test")
        
        # Check the result
        self.assertTrue(result["success"])
        self.assertEqual(result["message"], "Tests for StepTest passed")
    
    @async_test
    async def test_save_step(self):
        """Test the save_step method."""
        # Call the method
        result = await self.builder.save_step("test")
        
        # Check the result
        self.assertTrue(result["success"])
        self.assertEqual(result["message"], "Saved step StepTest")
    
    @async_test
    async def test_link_steps(self):
        """Test the link_steps method."""
        # Call the method
        result = await self.builder.link_steps("source", "target")
        
        # Check the result
        self.assertTrue(result["success"])
        self.assertEqual(result["message"], "Created link from StepSource to StepTarget")
    
    @async_test
    async def test_save_workflow(self):
        """Test the save_workflow method."""
        # Call the method
        result = await self.builder.save_workflow()
        
        # Check the result
        self.assertTrue(result["success"])
        self.assertEqual(result["message"], "Saved workflow at test_workflow")
    
    @async_test
    async def test_process_command_invalid(self):
        """Test the process_command method with an invalid command."""
        # Call the method with an invalid command
        result = await self.builder.process_command("invalid", [])
        
        # Check the result
        self.assertFalse(result["success"])
        self.assertEqual(result["error"], "Unknown command: invalid")


if __name__ == "__main__":
    unittest.main() 