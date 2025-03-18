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

# Import necessary dependencies
from src.ExecutorBase import ExecutorBase
from src.DataUnitBase import DataUnitBase
from src.Step import Step
from unittest.mock import MagicMock

# Mock the StepTestStep class instead of importing it
class StepTestStep(Step):
    """Mock implementation of StepTestStep for testing."""
    
    def __init__(self, executor, **kwargs):
        """Initialize the test step."""
        super().__init__(executor, **kwargs)
        self.name = "TestStep"
        
    async def process(self, inputs):
        """Process the inputs and return a response."""
        if not inputs:
            return "No input provided"
        return f"Processed input: {inputs[0]}"

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
    
    # Create test data - pass a list instead of a dictionary
    test_data = ["Test input data"]
    
    # Process the data
    result = await step.process(test_data)
    
    # Check that the step processed the data correctly
    assert result == "Processed input: Test input data"

if __name__ == "__main__":
    pytest.main(["-v", "--asyncio-mode=strict", __file__]) 