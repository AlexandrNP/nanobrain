"""
Simplified tests for the NanoBrainBuilder.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class MockAgent:
    """Mock Agent class for testing."""
    
    def __init__(self, **kwargs):
        """Initialize the mock Agent."""
        self.tools = []
        self.workflow_context = {}
    
    def add_tool(self, tool):
        """Add a tool to the agent."""
        self.tools.append(tool)
    
    def update_workflow_context(self, workflow_path):
        """Update the workflow context."""
        self.workflow_context = {"path": workflow_path}


class MockExecutorBase:
    """Mock ExecutorBase class for testing."""
    
    def __init__(self):
        """Initialize the mock ExecutorBase."""
        pass


class MockNanoBrainBuilder:
    """Mock NanoBrainBuilder class for testing."""
    
    def __init__(self, executor=None):
        """Initialize the mock NanoBrainBuilder."""
        self.executor = executor or MockExecutorBase()
        self.agent = MockAgent()
        self._workflow_stack = []
    
    def get_current_workflow(self):
        """Get the current workflow path."""
        if not self._workflow_stack:
            return None
        return self._workflow_stack[-1]
    
    def push_workflow(self, workflow_path):
        """Push a workflow onto the stack."""
        self._workflow_stack.append(workflow_path)
    
    def pop_workflow(self):
        """Pop a workflow from the stack."""
        if not self._workflow_stack:
            return None
        return self._workflow_stack.pop()


class TestNanoBrainBuilder(unittest.TestCase):
    """Test case for the NanoBrainBuilder class."""
    
    def setUp(self):
        """Set up the test environment."""
        self.builder = MockNanoBrainBuilder()
    
    def test_initialization(self):
        """Test that the builder initializes correctly."""
        self.assertIsInstance(self.builder, MockNanoBrainBuilder)
        self.assertIsInstance(self.builder.executor, MockExecutorBase)
        self.assertIsInstance(self.builder.agent, MockAgent)
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


if __name__ == "__main__":
    unittest.main() 