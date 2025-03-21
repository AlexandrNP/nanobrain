#!/usr/bin/env python3
"""
TestResources - Templates and resources for testing NanoBrain components.

This module provides templates and utilities for creating and running tests
for NanoBrain workflows, steps, and other components.
"""

import os
import sys
import subprocess
from typing import Dict, Any, Optional
from pathlib import Path
import asyncio

# Utility function
def camel_case(s: str) -> str:
    """Convert a string to CamelCase."""
    # Replace non-alphanumeric characters with spaces
    import re
    s = re.sub(r'[^a-zA-Z0-9]', ' ', s)
    # Convert to CamelCase
    s = ''.join(word.capitalize() for word in s.split())
    return s


TEST_CODE_RESPONSE = '''```python
import asyncio
from src.Step import Step

class CustomStep(Step):
    """
    A custom step for processing data.
    
    Biological analogy: Neuron processing sensory information.
    Justification: Like how neurons transform sensory data into meaningful patterns,
    this step transforms input data into a more useful representation.
    """
    
    def __init__(self, name="CustomStep", **kwargs):
        """Initialize the step."""
        super().__init__(name=name, **kwargs)
    
    async def process(self, data_dict):
        """Process the input data."""
        result = data_dict.copy()
        result['processed'] = True
        return result
```'''

# Test Templates
STEP_TEST_TEMPLATE = '''#!/usr/bin/env python3
"""
Unit tests for {step_class_name}
"""

import unittest
from {workflow_path_dots}.src.{step_class_name}.{step_class_name} import {step_class_name}


class Test{step_class_name}(unittest.TestCase):
    """Test cases for {step_class_name}"""

    def setUp(self):
        """Set up test fixtures"""
        self.step = {step_class_name}()

    def test_process(self):
        """Test the process method"""
        # Prepare test data
        test_data = {{}}
        
        # Call the process method
        result = self.step.process(test_data)
        
        # Assert expected results
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)


if __name__ == '__main__':
    unittest.main()
'''

WORKFLOW_TEST_TEMPLATE = '''import unittest
import asyncio
from unittest.mock import MagicMock

from src.ExecutorBase import ExecutorBase
from src.{workflow_class_name} import {workflow_class_name}

class Test{workflow_class_name}(unittest.TestCase):
    def setUp(self):
        self.executor = MagicMock(spec=ExecutorBase)
        self.workflow = {workflow_class_name}(executor=self.executor)
    
    def test_initialization(self):
        """Test that the workflow initializes correctly."""
        self.assertIsInstance(self.workflow, {workflow_class_name})
        
    # Add more tests here

if __name__ == '__main__':
    unittest.main()
'''

# Test Prompts
INITIAL_TEST_PROMPT = """
Please generate a unit test file for the {step_class_name} class with the following details:
- Class to test: {step_class_name}
- Base class: {base_class}
- Workflow path: {workflow_path}

The test file should include:
- Proper imports
- Test class setup
- Basic test methods to verify functionality
- Any necessary mocks

Return the complete Python test code that I can save as test_{step_class_name}.py.
"""

COMPREHENSIVE_TEST_PROMPT = """
Please generate a comprehensive unit test file for the {step_class_name} class with the following details:
- Class to test: {step_class_name}
- Class file location: {class_file_location}
- Workflow path: {workflow_path}

The test file should include:
- Proper imports
- Test class setup with appropriate mocks
- Comprehensive test methods to verify all functionality
- Edge case testing
- Any necessary helper methods

Return the complete Python test code that I can save as test_{step_class_name}.py.
"""

# Test Step Class
class TestStepStep:
    """
    Test a step.
    
    This step creates mock inputs and tests the existing logic of a step.
    """
    @staticmethod
    async def test_step(builder, step_name: str) -> Dict[str, Any]:
        """
        Test a step in the workflow.
        
        Args:
            builder: The NanoBrainBuilder instance
            step_name: Name of the step to test
        
        Returns:
            Dictionary with the result of the operation
        """
        # Get the current workflow
        workflow_path = builder.get_current_workflow()
        if not workflow_path:
            return {
                "success": False,
                "error": "No active workflow. Create or activate a workflow first."
            }
        
        # Format the step name in CamelCase
        step_class_name = f"Step{camel_case(step_name)}"
        
        # Check if the step exists
        step_dir = os.path.join(workflow_path, 'src', step_class_name)
        step_file = os.path.join(step_dir, f'{step_class_name}.py')
        
        if not os.path.exists(step_file):
            return {
                "success": False,
                "error": f"Step file not found: {step_file}"
            }
        
        try:
            # Create a test file if it doesn't exist
            test_dir = os.path.join(workflow_path, 'test')
            test_file = os.path.join(test_dir, f'test_{step_class_name}.py')
            
            file_writer_tool = None
            if hasattr(builder, 'agent') and hasattr(builder.agent, 'tools'):
                file_writer_tool = next((tool for tool in builder.agent.tools if tool.__class__.__name__ == 'StepFileWriter'), None)
            
            if file_writer_tool:
                # Create the test file if it doesn't exist
                if not os.path.exists(test_file):
                    test_content = f"""import unittest
import asyncio
from unittest.mock import MagicMock

from src.ExecutorBase import ExecutorBase
from src.{step_class_name}.{step_class_name} import {step_class_name}

class Test{step_class_name}(unittest.TestCase):
    def setUp(self):
        self.executor = MagicMock(spec=ExecutorBase)
        self.step = {step_class_name}(executor=self.executor)
    
    def test_initialization(self):
        \"\"\"Test that the step initializes correctly.\"\"\"
        self.assertIsInstance(self.step, {step_class_name})
    
    def test_process(self):
        \"\"\"Test that the process method works correctly.\"\"\"
        result = asyncio.run(self.step.process([]))
        # Add assertions here
        
    # Add more tests here

if __name__ == '__main__':
    unittest.main()
"""
                    await file_writer_tool.create_file(test_file, test_content)
            
            # Run the tests
            try:
                # Change to the workflow directory
                original_dir = os.getcwd()
                os.chdir(workflow_path)
                
                # Run the test
                test_command = f"python -m unittest test/test_{step_class_name}.py"
                test_result = subprocess.run(test_command, shell=True, capture_output=True, text=True)
                
                # Change back to the original directory
                os.chdir(original_dir)
                
                # Check the test result
                if test_result.returncode == 0:
                    return {
                        "success": True,
                        "message": f"Tests for {step_class_name} passed",
                        "output": test_result.stdout
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Tests for {step_class_name} failed",
                        "output": test_result.stderr
                    }
            except Exception as e:
                # Change back to the original directory
                os.chdir(original_dir)
                
                return {
                    "success": False,
                    "error": f"Failed to run tests: {e}"
                }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to test step: {e}"
            }

async def generate_test_file(agent_builder, step_class_name, base_class, workflow_path, step_dir=None):
    """
    Generate a test file for a step.
    
    Args:
        agent_builder: The agent builder instance
        step_class_name: Name of the step class
        base_class: Base class of the step
        workflow_path: Path to the workflow
        step_dir: Optional directory of the step
    
    Returns:
        The generated test code
    """
    try:
        # Use the initial test prompt
        test_prompt = INITIAL_TEST_PROMPT.format(
            step_class_name=step_class_name,
            base_class=base_class,
            workflow_path=workflow_path
        )
        
        # Process the prompt with the agent
        if hasattr(agent_builder, 'process'):
            initial_test_result = await agent_builder.process([test_prompt])
        else:
            # Fallback to safe_process if available
            if 'safe_process' in globals():
                initial_test_result = await safe_process(agent_builder, [test_prompt])
            else:
                # Direct call if no safe_process
                initial_test_result = await agent_builder.process([test_prompt])
        
        # Extract code from the response
        if initial_test_result:
            # Try to extract code blocks
            import re
            test_blocks = re.findall(r'```(?:python)?\s*(.*?)```', initial_test_result, re.DOTALL)
            initial_test = test_blocks[0] if test_blocks else initial_test_result
        else:
            # Fallback to a basic test file
            workflow_path_dots = workflow_path.replace('/', '.')
            initial_test = STEP_TEST_TEMPLATE.format(
                step_class_name=step_class_name,
                workflow_path_dots=workflow_path_dots
            )
            
        return initial_test
    except Exception as e:
        print(f"Warning: Could not generate tests: {e}")
        workflow_path_dots = workflow_path.replace('/', '.')
        return STEP_TEST_TEMPLATE.format(
            step_class_name=step_class_name,
            workflow_path_dots=workflow_path_dots
        )

async def generate_comprehensive_test_file(agent_builder, step_class_name, workflow_path, step_dir):
    """
    Generate a comprehensive test file for a step.
    
    Args:
        agent_builder: The agent builder instance
        step_class_name: Name of the step class
        workflow_path: Path to the workflow
        step_dir: Directory of the step
    
    Returns:
        The generated test code
    """
    try:
        # Use the comprehensive test prompt
        class_file_location = os.path.join(step_dir, f'{step_class_name}.py')
        test_prompt = COMPREHENSIVE_TEST_PROMPT.format(
            step_class_name=step_class_name,
            class_file_location=class_file_location,
            workflow_path=workflow_path
        )
        
        # Process the prompt with the agent
        if hasattr(agent_builder, 'process'):
            test_result = await agent_builder.process([test_prompt])
        else:
            # Fallback to safe_process if available
            if 'safe_process' in globals():
                test_result = await safe_process(agent_builder, [test_prompt])
            else:
                # Direct call if no safe_process
                test_result = await agent_builder.process([test_prompt])
        
        # Extract code from the response
        if test_result:
            # Try to extract code blocks
            import re
            test_blocks = re.findall(r'```(?:python)?\s*(.*?)```', test_result, re.DOTALL)
            test_content = test_blocks[0] if test_blocks else test_result
            return test_content
        else:
            return None
    except Exception as e:
        print(f"Warning: Could not generate comprehensive tests: {e}")
        return None 