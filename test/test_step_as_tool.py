import unittest
import asyncio
import sys
import os
from unittest.mock import MagicMock, patch

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.Step import Step
from src.ExecutorBase import ExecutorBase
from langchain_core.tools import BaseTool
from src.enums import ComponentState

class TestStepAsTool(unittest.TestCase):
    """Test the refactored Step class that inherits from BaseTool."""

    def setUp(self):
        """Set up test fixtures."""
        self.executor = MagicMock(spec=ExecutorBase)
        # Patch the DirectoryTracer to avoid file system interactions
        with patch('src.DirectoryTracer.DirectoryTracer'):
            self.step = Step(executor=self.executor, name="TestStep", description="A test step")
        
    def test_initialization(self):
        """Test that the Step is properly initialized with BaseTool attributes."""
        self.assertIsInstance(self.step, BaseTool)
        self.assertEqual(self.step.name, "TestStep")
        self.assertEqual(self.step.description, "A test step")
        self.assertFalse(self.step.return_direct)
        self.assertEqual(self.step.state, ComponentState.INACTIVE)
        self.assertFalse(self.step.running)
        
    def test_run_method(self):
        """Test the _run method implementation."""
        # Mock the process method to return a test value
        self.step.process = MagicMock(return_value="Test result")
        
        # Call the _run method
        result = self.step._run("test input")
        
        # Check that process was called with the correct arguments
        self.step.process.assert_called_once()
        # The input should be converted to a dictionary with auto-generated keys
        self.assertIn("input_0", self.step.process.call_args[0][0])
        self.assertEqual(self.step.process.call_args[0][0]["input_0"], "test input")
        
    def test_arun_method(self):
        """Test the _arun method implementation."""
        # Create an async mock for the process method
        async def mock_process(inputs):
            return "Async test result"
        
        self.step.process = mock_process
        
        # Run the _arun method in an async context
        result = asyncio.run(self.step._arun("test input"))
        
        # Check the result
        self.assertEqual(result, "Async test result")
        
    def test_tool_compatibility(self):
        """Test that the Step can be used as a Tool with LangChain."""
        # Test the run method
        self.step.process = MagicMock(return_value="Test run result")
        result = self.step.run("test input")
        self.assertEqual(result, "Test run result")
        
        # Test the arun method in an async context
        async def mock_process(inputs):
            return "Async run result"
        
        self.step.process = mock_process
        result = asyncio.run(self.step.arun("test input"))
        self.assertEqual(result, "Async run result")
        
    def test_multiple_inheritance(self):
        """Test that the Step maintains functionality from all parent classes."""
        # Test PackageBase functionality
        self.assertEqual(self.step.executor, self.executor)
        
        # Test IRunnable functionality
        self.assertTrue(hasattr(self.step, 'invoke'))
        
        # Test BaseTool functionality
        self.assertTrue(hasattr(self.step, 'run'))
        self.assertTrue(hasattr(self.step, 'arun'))
        
    def test_custom_step_subclass(self):
        """Test creating a custom Step subclass."""
        # Patch the DirectoryTracer to avoid file system interactions
        with patch('src.DirectoryTracer.DirectoryTracer'):
            class CustomStep(Step):
                """A custom step for testing."""
                
                async def process(self, inputs):
                    return f"Custom step processed: {inputs}"
            
            custom_step = CustomStep(self.executor, name="CustomStep")
            
            # Test as a BaseTool
            self.assertIsInstance(custom_step, BaseTool)
            self.assertEqual(custom_step.name, "CustomStep")
            self.assertEqual(custom_step.description, "A custom step for testing.")
            
            # Test the process method
            result = asyncio.run(custom_step.process({"input": "test"}))
            self.assertEqual(result, "Custom step processed: {'input': 'test'}")
            
            # Test the run method
            async def run_test():
                return await custom_step.arun("test input")
            
            result = asyncio.run(run_test())
            self.assertTrue("Custom step processed" in result)
            self.assertTrue("input_0" in result)

if __name__ == '__main__':
    unittest.main() 