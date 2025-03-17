#!/usr/bin/env python3
"""
Test script to execute the TestStep we created.

This script imports and runs the step to verify it works correctly.
"""

import os
import sys
import asyncio
import pytest
from pathlib import Path

# Add the project root to the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Add the workflow src directory to the path
workflow_src = os.path.join(project_root, 'workflows', 'test_workflow', 'src')
if workflow_src not in sys.path:
    sys.path.insert(0, workflow_src)

# Import the step
from workflows.test_workflow.src.StepTestStep.StepTestStep import StepTestStep
from src.ExecutorBase import ExecutorBase
from src.DataUnitBase import DataUnitBase

class MockDataUnit(DataUnitBase):
    """Mock data unit for testing."""
    
    def __init__(self, initial_data=None):
        """Initialize the mock data unit."""
        self.data = initial_data
    
    def get(self):
        """Get the data."""
        return self.data
    
    def set(self, data):
        """Set the data."""
        self.data = data
        return True

class MockExecutor(ExecutorBase):
    """Mock executor for testing."""
    
    def __init__(self, **kwargs):
        """Initialize the mock executor."""
        super().__init__(**kwargs)
        self.runnable_types = {"Step"}
        self.energy_per_execution = 0.1
        self.energy_level = 1.0
    
    def can_execute(self, step_name):
        """Check if the executor can execute the step."""
        return True
    
    async def execute(self, step_class, **kwargs):
        """Execute the step."""
        return await step_class.process(kwargs.get('data', {}))

@pytest.mark.asyncio
async def test_step_execution():
    """Test the execution of the TestStep."""
    # Create a mock executor
    executor = MockExecutor()
    
    # Create a step instance
    step = StepTestStep(executor=executor, name="TestStep")
    
    # Create test data
    test_data = {"input": "Test input data"}
    
    # Process the data
    result = await step.process(test_data)
    
    # Print the result
    print(f"Step execution result: {result}")
    
    # Verify the result
    assert result is not None
    assert "processed_by" in result
    assert result["processed_by"] == "StepTestStep"
    
    return result

if __name__ == "__main__":
    pytest.main(["-v", "--asyncio-mode=strict", __file__]) 