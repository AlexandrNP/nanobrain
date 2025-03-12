"""
Workflow Steps for NanoBrainBuilder

This module contains the implementation of the workflow steps that can be performed
by the NanoBrainBuilder.
"""

import os
import re
import shutil
import asyncio
import subprocess
import importlib.util
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple

from src.Step import Step
from src.ExecutorBase import ExecutorBase
from src.ConfigManager import ConfigManager
from builder.AgentWorkflowBuilder import AgentWorkflowBuilder
from src.DataStorageCommandLine import DataStorageCommandLine
from src.LinkDirect import LinkDirect
from src.TriggerDataChanged import TriggerDataChanged
from src.DataUnitBase import DataUnitBase


def camel_case(s: str) -> str:
    """Convert a string to CamelCase."""
    # Replace non-alphanumeric characters with spaces
    s = re.sub(r'[^a-zA-Z0-9]', ' ', s)
    # Convert to CamelCase
    s = ''.join(word.capitalize() for word in s.split())
    return s


class CreateWorkflow:
    """
    Create a new workflow.
    
    This step creates a new workflow using the same sequence of steps as the
    NanoBrainBuilder. It also initializes a git repository for the workflow.
    """
    @staticmethod
    async def execute(builder, workflow_name: str, base_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute the create workflow step.
        
        Args:
            builder: The NanoBrainBuilder instance
            workflow_name: Name of the workflow to create
            base_dir: Base directory for the workflow (optional)
        
        Returns:
            Dictionary with the result of the operation
        """
        # Get the workflows directory from config
        workflows_dir = builder.config.get('defaults', {}).get('workflows_dir', 'workflows')
        
        # Create the workflows directory if it doesn't exist
        if not os.path.exists(workflows_dir):
            os.makedirs(workflows_dir, exist_ok=True)
        
        # Create the workflow directory
        if base_dir is None:
            base_dir = workflows_dir
        workflow_path = os.path.join(base_dir, workflow_name)
        
        # Check if the directory already exists
        if os.path.exists(workflow_path):
            return {
                "success": False,
                "error": f"Workflow directory already exists: {workflow_path}"
            }
        
        try:
            # Create the directory structure
            os.makedirs(workflow_path, exist_ok=True)
            os.makedirs(os.path.join(workflow_path, 'src'), exist_ok=True)
            os.makedirs(os.path.join(workflow_path, 'config'), exist_ok=True)
            os.makedirs(os.path.join(workflow_path, 'test'), exist_ok=True)
            
            # Initialize git repository
            git_init_tool = next((tool for tool in builder.agent.tools if tool.__class__.__name__ == 'StepGitInit'), None)
            
            if git_init_tool:
                # Get author name from config
                author_name = builder.config.get('defaults', {}).get('author_name', 'NanoBrain Developer')
                git_result = await git_init_tool.process([workflow_path, workflow_name, author_name])
                
                if not git_result.get('success', False):
                    return {
                        "success": False,
                        "error": f"Failed to initialize git repository: {git_result.get('error', 'Unknown error')}"
                    }
            
            # Create basic files
            file_writer_tool = next((tool for tool in builder.agent.tools if tool.__class__.__name__ == 'StepFileWriter'), None)
            
            if file_writer_tool:
                # Create README.md
                readme_content = f"""# {workflow_name}

A NanoBrain workflow created with the NanoBrain builder tool.

## Author
{builder.config.get('defaults', {}).get('author_name', 'NanoBrain Developer')}
"""
                await file_writer_tool.create_file(os.path.join(workflow_path, 'README.md'), readme_content)
                
                # Format workflow name in CamelCase
                workflow_class_name = camel_case(workflow_name)
                
                # Create main workflow file
                workflow_content = f"""from typing import List, Dict, Any, Optional
from src.Workflow import Workflow
from src.ExecutorBase import ExecutorBase
from src.Step import Step

class {workflow_class_name}(Workflow):
    \"\"\"
    {workflow_name} workflow.
    
    Biological analogy: Coordinated neural circuit.
    Justification: Like how coordinated neural circuits work together to
    accomplish complex tasks, this workflow coordinates multiple steps
    to achieve its objectives.
    \"\"\"
    def __init__(self, executor: Optional[ExecutorBase] = None, steps: List[Step] = None, **kwargs):
        # Create an executor if none is provided
        if executor is None:
            from src.ExecutorBase import ExecutorBase
            executor = ExecutorBase()
        
        # Initialize the Workflow base class
        super().__init__(executor, steps, **kwargs)
        
        # {workflow_class_name}-specific attributes
        # Add your attributes here
"""
                workflow_file_path = os.path.join(workflow_path, 'src', f'{workflow_class_name}.py')
                await file_writer_tool.create_file(workflow_file_path, workflow_content)
                
                # Create a default configuration file
                config_content = f"""defaults:
  # Add your default configuration parameters here

metadata:
  description: "{workflow_name} workflow"
  biological_analogy: "Coordinated neural circuit"
  justification: >
    Like how coordinated neural circuits work together to accomplish complex tasks,
    this workflow coordinates multiple steps to achieve its objectives.
  objectives:
    # Add your workflow objectives here
  author: "{builder.config.get('defaults', {}).get('author_name', 'NanoBrain Developer')}"

validation:
  required:
    - executor  # ExecutorBase instance required
  optional:
    # Add your optional parameters here
  constraints:
    # Add your parameter constraints here

examples:
  basic:
    description: "Basic usage example"
    config:
      # Add example configuration here
"""
                await file_writer_tool.create_file(os.path.join(workflow_path, 'config', f'{workflow_class_name}.yml'), config_content)
                
                # Create a test file
                test_content = f"""import unittest
import asyncio
from unittest.mock import MagicMock

from src.ExecutorBase import ExecutorBase
from src.{workflow_class_name} import {workflow_class_name}

class Test{workflow_class_name}(unittest.TestCase):
    def setUp(self):
        self.executor = MagicMock(spec=ExecutorBase)
        self.workflow = {workflow_class_name}(executor=self.executor)
    
    def test_initialization(self):
        \"\"\"Test that the workflow initializes correctly.\"\"\"
        self.assertIsInstance(self.workflow, {workflow_class_name})
        
    # Add more tests here

if __name__ == '__main__':
    unittest.main()
"""
                await file_writer_tool.create_file(os.path.join(workflow_path, 'test', f'test_{workflow_class_name}.py'), test_content)
            
            # Add workflow to builder's workflow stack
            builder.push_workflow(workflow_path)
            
            # Update the agent's workflow context
            builder.agent.update_workflow_context(workflow_path)
            
            return {
                "success": True,
                "message": f"Created workflow {workflow_name} at {workflow_path}",
                "workflow_path": workflow_path
            }
            
        except Exception as e:
            # Clean up if an error occurred
            if os.path.exists(workflow_path):
                shutil.rmtree(workflow_path)
            
            return {
                "success": False,
                "error": f"Failed to create workflow: {e}"
            }


class CreateStep:
    """
    Create a new step.
    
    This step creates a new folder for the step with a Python file for the step
    class and a default configuration file. It starts an interactive code writing
    phase using AgentWorkflowBuilder and DataStorageCommandLine.
    """
    @staticmethod
    async def execute(builder, step_name: str, base_class: str = "Step", description: str = None) -> Dict[str, Any]:
        """
        Execute the create step step.
        
        Args:
            builder: The NanoBrainBuilder instance
            step_name: Name of the step to create
            base_class: Base class for the step (default: "Step")
            description: Description of the step (optional)
        
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
        
        # Ensure description is a string
        if description is not None:
            if hasattr(description, 'template'):
                description = description.template
            description = str(description)
        else:
            description = f"A custom step for {step_name}"
        
        # Format the step name in CamelCase
        step_class_name = f"Step{camel_case(step_name)}"
        
        # Create the step directory
        step_dir = os.path.join(workflow_path, step_class_name)
        
        # Check if the directory already exists
        if os.path.exists(step_dir):
            return {
                "success": False,
                "error": f"Step directory already exists: {step_dir}"
            }
        
        try:
            # Create the directory
            os.makedirs(step_dir, exist_ok=True)
            os.makedirs(os.path.join(step_dir, 'config'), exist_ok=True)
           
            # Create instances using ConfigManager
            config_manager = ConfigManager()
            
            # Create AgentWorkflowBuilder instance - let ConfigManager handle all configuration
            agent_builder = config_manager.create_instance("AgentWorkflowBuilder", 
                executor=builder.executor,
                # Only pass minimal context-specific overrides
                prompt_variables={
                    "role_description": f"create code for {step_class_name}",
                    "specific_instructions": f"Create a step class named {step_class_name}"
                }
            )
            
            # Create DataStorageCommandLine instance - let ConfigManager handle all configuration
            command_line = config_manager.create_instance("DataStorageCommandLine", 
                executor=builder.executor,
                # Only pass minimal context-specific overrides
                prompt=f"{step_class_name}> ",
                welcome_message=f"Starting interactive code writing phase for {step_class_name}.",
                goodbye_message="Step creation completed.",
                exit_command="finish"
            )
            
            # Create DataUnitBase instances for input and output
            input_data = DataUnitBase()
            output_data = DataUnitBase()
            
            # Create Step instances for source and sink
            source_step = Step(executor=builder.executor, output=output_data)
            sink_step = Step(executor=builder.executor)
            sink_step.register_input_source("link_id", output_data)
            
            # Create LinkDirect instance - let ConfigManager handle all configuration
            link = config_manager.create_instance("LinkDirect",
                source_step=source_step,
                sink_step=sink_step
            )
            
            # Create TriggerDataChanged instance - let ConfigManager handle all configuration
            trigger = config_manager.create_instance("TriggerDataChanged", 
                runnable=link
            )
            
            # Update the link with the trigger
            link.trigger = trigger
            
            # Start monitoring for input
            await link.start_monitoring()
            
            # Display initial menu
            menu = """
Available commands:
1. link <source_step> <target_step> - Link this step to another step
2. finish - End step creation and save
3. help - Show this menu
"""
            print(menu)
            
            # Start the command line monitoring
            await command_line.start_monitoring()
            
            # After monitoring stops, save all files and update configuration
            file_writer_tool = next((tool for tool in builder.agent.tools if tool.__class__.__name__ == 'StepFileWriter'), None)
            
            if file_writer_tool:
                # Create the step class file
                step_content = agent_builder.get_generated_code()
                await file_writer_tool.create_file(os.path.join(step_dir, f'{step_class_name}.py'), step_content)
                
                # Create __init__.py
                init_content = f"""from .{step_class_name} import {step_class_name}

__all__ = ['{step_class_name}']
"""
                await file_writer_tool.create_file(os.path.join(step_dir, '__init__.py'), init_content)
                
                # Create configuration file
                config_content = agent_builder.get_generated_config()
                await file_writer_tool.create_file(os.path.join(step_dir, 'config', f'{step_class_name}.yml'), config_content)
                
                # Create test file
                test_content = agent_builder.get_generated_tests()
                await file_writer_tool.create_file(os.path.join(workflow_path, 'test', f'test_{step_class_name}.py'), test_content)
            
            # Return success
            return {
                "success": True,
                "message": f"Created step {step_class_name} at {step_dir}",
                "step_dir": step_dir,
                "step_class_name": step_class_name
            }
            
        except Exception as e:
            # Clean up if an error occurred
            if os.path.exists(step_dir):
                shutil.rmtree(step_dir)
            
            return {
                "success": False,
                "error": f"Failed to create step: {e}"
            }


class TestStepStep:
    """
    Test a step.
    
    This step creates mock inputs and tests the existing logic of a step.
    """
    @staticmethod
    async def execute(builder, step_name: str) -> Dict[str, Any]:
        """
        Execute the test step step.
        
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


class SaveStepStep:
    """
    Save a step.
    
    This step finalizes a step by ensuring it is properly tested and
    documented.
    """
    @staticmethod
    async def execute(builder, step_name: str) -> Dict[str, Any]:
        """
        Execute the save step step.
        
        Args:
            builder: The NanoBrainBuilder instance
            step_name: Name of the step to save
        
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
            # Run the tests to make sure they pass
            test_result = await TestStepStep.execute(builder, step_name)
            
            if not test_result.get('success', False):
                return {
                    "success": False,
                    "error": f"Tests failed for step {step_class_name}: {test_result.get('error', 'Unknown error')}"
                }
            
            # Check if the configuration file exists in the workflow config directory
            config_dir = os.path.join(workflow_path, 'config')
            config_file = os.path.join(config_dir, f'{step_class_name}.yml')
            
            file_writer_tool = next((tool for tool in builder.agent.tools if tool.__class__.__name__ == 'StepFileWriter'), None)
            
            if file_writer_tool and not os.path.exists(config_file):
                # Copy the configuration file from the step directory to the workflow config directory
                step_config_file = os.path.join(step_dir, 'config', f'{step_class_name}.yml')
                
                if os.path.exists(step_config_file):
                    with open(step_config_file, 'r') as f:
                        config_content = f.read()
                    
                    await file_writer_tool.create_file(config_file, config_content)
            
            return {
                "success": True,
                "message": f"Saved step {step_class_name}"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to save step: {e}"
            }


class LinkStepsStep:
    """
    Link steps together.
    
    This step creates a link between two steps in a workflow.
    """
    @staticmethod
    async def execute(builder, source_step: str, target_step: str, link_type: str = "LinkDirect") -> Dict[str, Any]:
        """
        Execute the link steps step.
        
        Args:
            builder: The NanoBrainBuilder instance
            source_step: Name of the source step
            target_step: Name of the target step
            link_type: Type of link to create (default: "LinkDirect")
        
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
        
        # Format the step names in CamelCase
        source_class_name = f"Step{camel_case(source_step)}"
        target_class_name = f"Step{camel_case(target_step)}"
        
        # Check if the steps exist
        source_dir = os.path.join(workflow_path, 'src', source_class_name)
        target_dir = os.path.join(workflow_path, 'src', target_class_name)
        
        if not os.path.exists(source_dir):
            return {
                "success": False,
                "error": f"Source step not found: {source_dir}"
            }
        
        if not os.path.exists(target_dir):
            return {
                "success": False,
                "error": f"Target step not found: {target_dir}"
            }
        
        try:
            # Create a link file if it doesn't exist
            link_name = f"{source_class_name}To{target_class_name}Link"
            link_file = os.path.join(workflow_path, 'src', f'{link_name}.py')
            
            file_writer_tool = next((tool for tool in builder.agent.tools if tool.__class__.__name__ == 'StepFileWriter'), None)
            
            if file_writer_tool:
                # Create the link file if it doesn't exist
                if not os.path.exists(link_file):
                    link_content = f"""from typing import Any
from src.{link_type} import {link_type}
from src.{source_class_name}.{source_class_name} import {source_class_name}
from src.{target_class_name}.{target_class_name} import {target_class_name}

class {link_name}({link_type}):
    \"\"\"
    Link from {source_class_name} to {target_class_name}.
    
    Biological analogy: Synaptic connection.
    Justification: Like how synaptic connections transmit signals between
    neurons, this link transmits data from {source_class_name} to {target_class_name}.
    \"\"\"
    def __init__(self, source_step: {source_class_name}, target_step: {target_class_name}, **kwargs):
        # Initialize with data units from the steps
        super().__init__(
            input_data=source_step.output_data,
            output_data=target_step.input_data,
            **kwargs
        )
        
        # Store references to the steps
        self.source_step = source_step
        self.target_step = target_step
    
    async def transfer(self) -> Any:
        \"\"\"
        Transfer data from the source step to the target step.
        
        Returns:
            The transferred data
        \"\"\"
        # Use the base class transfer method
        return await super().transfer()
"""
                    await file_writer_tool.create_file(link_file, link_content)
                
                # Create a configuration file for the link
                config_file = os.path.join(workflow_path, 'config', f'{link_name}.yml')
                
                if not os.path.exists(config_file):
                    config_content = f"""defaults:
  # Add your default configuration parameters here

metadata:
  description: "Link from {source_class_name} to {target_class_name}"
  biological_analogy: "Synaptic connection"
  justification: >
    Like how synaptic connections transmit signals between neurons,
    this link transmits data from {source_class_name} to {target_class_name}.
  objectives:
    - Transfer data from {source_class_name} to {target_class_name}
    - Ensure reliable data transmission

validation:
  required:
    - input_data  # DataUnitBase instance required
    - output_data  # DataUnitBase instance required
  optional:
    # Add your optional parameters here
  constraints:
    # Add your parameter constraints here

examples:
  basic:
    description: "Basic usage example"
    config:
      # Add example configuration here
"""
                    await file_writer_tool.create_file(config_file, config_content)
            
            # Update the workflow file to include the connection
            workflow_class_name = os.path.basename(workflow_path)
            workflow_file = os.path.join(workflow_path, 'src', f'{workflow_class_name}.py')
            
            if os.path.exists(workflow_file):
                with open(workflow_file, 'r') as f:
                    workflow_content = f.read()
                
                # Check if the connection is already defined
                connection_code = f"self.connect('{source_class_name.lower()}', '{target_class_name.lower()}')"
                
                if connection_code not in workflow_content:
                    # Find the end of the __init__ method
                    init_end = workflow_content.find('super().__init__(executor, steps, **kwargs)')
                    
                    if init_end != -1:
                        # Find the next blank line after super().__init__
                        next_blank_line = workflow_content.find('\n\n', init_end)
                        
                        if next_blank_line != -1:
                            # Insert the connection code
                            new_workflow_content = (
                                workflow_content[:next_blank_line] +
                                f"\n        # Connect steps\n        {connection_code}\n" +
                                workflow_content[next_blank_line:]
                            )
                            
                            # Write the updated workflow file
                            await file_writer_tool.create_file(workflow_file, new_workflow_content)
            
            # Update the agent's workflow context
            builder.agent.update_workflow_context(workflow_path)
            
            return {
                "success": True,
                "message": f"Created link from {source_class_name} to {target_class_name}",
                "link_file": link_file
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to link steps: {e}"
            }


class SaveWorkflowStep:
    """
    Save a workflow.
    
    This step finalizes a workflow by ensuring all steps are properly linked
    and tested.
    """
    @staticmethod
    async def execute(builder) -> Dict[str, Any]:
        """
        Execute the save workflow step.
        
        Args:
            builder: The NanoBrainBuilder instance
        
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
        
        try:
            # Run tests for the workflow
            try:
                # Change to the workflow directory
                original_dir = os.getcwd()
                os.chdir(workflow_path)
                
                # Run the tests
                test_command = "python -m unittest discover -s test"
                test_result = subprocess.run(test_command, shell=True, capture_output=True, text=True)
                
                # Change back to the original directory
                os.chdir(original_dir)
                
                # Check the test result
                if test_result.returncode != 0:
                    return {
                        "success": False,
                        "error": f"Tests failed for workflow: {test_result.stderr}"
                    }
            except Exception as e:
                # Change back to the original directory
                os.chdir(original_dir)
                
                return {
                    "success": False,
                    "error": f"Failed to run tests: {e}"
                }
            
            # Pop the workflow from the stack
            builder.pop_workflow()
            
            # Go back to the parent directory
            workflow_dir = os.path.dirname(workflow_path)
            
            # Update the agent's workflow context
            if builder.get_current_workflow():
                builder.agent.update_workflow_context(builder.get_current_workflow())
            else:
                builder.agent.workflow_context = {}
            
            return {
                "success": True,
                "message": f"Saved workflow at {workflow_path}"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to save workflow: {e}"
            } 