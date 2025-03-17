#!/usr/bin/env python3
"""
WorkflowSteps - Steps for creating and managing workflows in NanoBrain.

This module provides step implementations for creating and managing
workflows and steps in the NanoBrain framework, implementing the
interactive workflow builder functionality.
"""

import os
import re
import sys
import json
import yaml
import types
import inspect
import asyncio
import traceback
import shutil
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path

from src.Step import Step
from src.Workflow import Workflow
from src.ConfigManager import ConfigManager
from builder.AgentWorkflowBuilder import AgentWorkflowBuilder
from src.DataStorageCommandLine import DataStorageCommandLine
from src.LinkDirect import LinkDirect
from src.TriggerDataUpdated import TriggerDataUpdated
from src.DataUnitBase import DataUnitBase
from src.DataUnitString import DataUnitString

class TestExecutor:
    """
    A simplified executor for testing purposes.
    
    This executor can be used as a replacement for ExecutorFunc during testing
    to avoid the complexities of the actual executor.
    """
    def __init__(self, **kwargs):
        """Initialize the test executor."""
        self.kwargs = kwargs
    
    async def execute(self, runnable):
        """
        Execute the runnable.
        
        This method handles various input formats and returns a simple response.
        
        Args:
            runnable: The input to process
            
        Returns:
            A simple response for testing
        """
        if isinstance(runnable, list):
            # Extract user message from list of messages
            user_message = ""
            for msg in runnable:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    user_message = msg.get("content", "")
                    break
            
            if user_message:
                if "code" in user_message.lower() or "step" in user_message.lower():
                    # Return a simple template for step code
                    return '''```python
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
                else:
                    # Return a simple text response
                    return f"Processed: {user_message}"
        
        # Default response
        return "Test executor response"


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
    async def create_workflow(builder, workflow_name: str, base_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a new workflow.
        
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
            git_init_tool = None
            if hasattr(builder, 'agent') and hasattr(builder.agent, 'tools'):
                git_init_tool = next((tool for tool in builder.agent.tools if tool.__class__.__name__ == 'StepGitInit'), None)
            if not git_init_tool:
                return {
                    "success": False,
                    "error": "StepGitInit tool not found. Cannot initialize git repository."
                }
            
            # Get author name from config
            author_name = builder.config.get('defaults', {}).get('author_name', 'NanoBrain Developer')
            git_result = await git_init_tool.process([workflow_path, workflow_name, author_name])
            
            if not git_result.get('success', False):
                return {
                    "success": False,
                    "error": f"Failed to initialize git repository: {git_result.get('error', 'Unknown error')}"
                }
            
            # Create basic files
            file_writer_tool = None
            if hasattr(builder, 'agent') and hasattr(builder.agent, 'tools'):
                file_writer_tool = next((tool for tool in builder.agent.tools if tool.__class__.__name__ == 'StepFileWriter'), None)
            if not file_writer_tool:
                return {
                    "success": False,
                    "error": "StepFileWriter tool not found. Cannot create files."
                }
                
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
    async def execute(builder, step_name: str, base_class: str = "Step", description: str = None, 
                     agent_builder=None, command_line=None) -> Dict[str, Any]:
        """
        Execute the create step step.
        
        Args:
            builder: The NanoBrainBuilder instance
            step_name: Name of the step to create
            base_class: Base class for the step (default: "Step")
            description: Description of the step (optional)
            agent_builder: Optional pre-configured AgentWorkflowBuilder instance (for testing)
            command_line: Optional pre-configured DataStorageBase instance (for testing)
        
        Returns:
            Dictionary with the result of the operation
        """
        # Create a variable to track if we need to clean up the directory on error
        step_dir = None
        need_cleanup = False
        
        try:
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
            step_dir = os.path.join(workflow_path, 'src', step_class_name)
            
            # Check if the directory already exists
            if os.path.exists(step_dir):
                return {
                    "success": False,
                    "error": f"Step directory already exists: {step_dir}"
                }
            
            # Create the directory structure
            os.makedirs(step_dir, exist_ok=True)
            need_cleanup = True
            os.makedirs(os.path.join(step_dir, 'config'), exist_ok=True)
           
            # File writer to create files
            file_writer_tool = None
            if hasattr(builder, 'agent') and hasattr(builder.agent, 'tools'):
                try:
                    file_writer_tool = next((tool for tool in builder.agent.tools if tool.__class__.__name__ == 'StepFileWriter'), None)
                except Exception as e:
                    return {
                        "success": False,
                        "error": f"Error finding StepFileWriter tool: {str(e)}"
                    }
            if not file_writer_tool:
                return {
                    "success": False,
                    "error": "StepFileWriter tool not found. Cannot create files."
                }
            
            # Create instances using ConfigManager or use provided instances
            if agent_builder is None or command_line is None:
                # Use the step directory as the base path
                config_manager = ConfigManager(base_path=step_dir)
                
                # Determine if we should use TestExecutor or the real executor
                use_test_executor = False
                try:
                    # Try importing ExecutorFunc
                    from src.ExecutorFunc import ExecutorFunc
                    # Create a test to see if it works
                    test_executor = ExecutorFunc()
                    # If we get here, it should work
                except (ImportError, TypeError, AttributeError) as e:
                    # If there's an issue, use TestExecutor instead
                    print(f"Warning: Error with ExecutorFunc: {e}. Using TestExecutor for compatibility.")
                    use_test_executor = True
                
                # Create executor instance
                if use_test_executor:
                    executor = TestExecutor(_debug_mode=builder._debug_mode)
                else:
                    # Use the real executor from the builder
                    executor = builder.executor
                
                # Create AgentWorkflowBuilder instance - let ConfigManager handle all configuration
                if agent_builder is None:
                    agent_builder = config_manager.create_instance("AgentWorkflowBuilder", 
                        executor=executor,
                        # Only pass minimal context-specific overrides
                        prompt_variables={
                            "role_description": f"create code for {step_class_name}",
                            "specific_instructions": f"Create a step class named {step_class_name} that inherits from {base_class} with description: {description}"
                        },
                        use_code_writer=True,  # Ensure code writer is used
                        _debug_mode=builder._debug_mode  # Pass debug mode
                    )
                
                # Set the current step directory in the agent builder
                agent_builder.current_step_dir = step_dir
                
                # Create DataStorageCommandLine instance - let ConfigManager handle all configuration
                if command_line is None:
                    command_line = config_manager.create_instance("DataStorageCommandLine", 
                        executor=executor,
                        # Only pass minimal context-specific overrides
                        prompt=f"{step_class_name}> ",
                        welcome_message=f"Starting interactive code writing phase for {step_class_name}.",
                        goodbye_message="Step creation completed.",
                        exit_command="finish",
                        _debug_mode=builder._debug_mode  # Pass debug mode
                    )
            else:
                # Set the current step directory in the agent builder if it's provided
                agent_builder.current_step_dir = step_dir
            
            # Create initial template files to show the user what's being created
            print(f"\n📁 Creating initial template files for {step_class_name}...")
            
            # Generate initial template code for the step
            initial_code = await agent_builder.generate_step_template(step_class_name, base_class, description)
            await file_writer_tool.create_file(os.path.join(step_dir, f'{step_class_name}.py'), initial_code)
            
            # Generate initial template config file
            initial_config = agent_builder.get_generated_config()
            await file_writer_tool.create_file(os.path.join(step_dir, 'config', f'{step_class_name}.yml'), initial_config)
            
            # Generate initial template test file
            initial_test = agent_builder.get_generated_tests()
            await file_writer_tool.create_file(os.path.join(workflow_path, 'test', f'test_{step_class_name}.py'), initial_test)
            
            # Create __init__.py file
            init_content = f"""from .{step_class_name} import {step_class_name}

__all__ = ['{step_class_name}']
"""
            await file_writer_tool.create_file(os.path.join(step_dir, '__init__.py'), init_content)
            
            print(f"✅ Initial template files created.")
            print(f"📄 Step class: {os.path.join(step_dir, f'{step_class_name}.py')}")
            print(f"📄 Config file: {os.path.join(step_dir, 'config', f'{step_class_name}.yml')}")
            print(f"📄 Test file: {os.path.join(workflow_path, 'test', f'test_{step_class_name}.py')}")
            print(f"📄 Init file: {os.path.join(step_dir, '__init__.py')}")
            
            # Create DataUnitString for output from command line
            try:
                from src.DataUnitString import DataUnitString
                cmd_output = DataUnitString(name="CommandOutput")
                command_line.output = cmd_output
                
                # Setup direct reference from command line to agent (primary path)
                command_line.agent_builder = agent_builder
                
                # Setup enhanced process method (backup path)
                def enhanced_process(self, data_dict):
                    # Process the input
                    result = DataStorageCommandLine.process(self, data_dict)
                    
                    # Extract the input from data_dict
                    user_input = None
                    if isinstance(data_dict, list) and len(data_dict) > 0:
                        user_input = data_dict[0]
                    elif isinstance(data_dict, dict) and 'query' in data_dict:
                        user_input = data_dict['query']
                    elif isinstance(data_dict, str):
                        user_input = data_dict
                    
                    # Directly call agent with result if available
                    if hasattr(self, 'agent_builder') and self.agent_builder and self.output:
                        # Get the actual output value
                        output_value = self.output.get()
                        
                        if output_value:
                            print("Command Line directly calling Agent Builder with result")
                            # Create a task and don't wait for it
                            loop = asyncio.get_event_loop()
                            if loop.is_running():
                                # Create a task without awaiting it
                                asyncio.create_task(self._call_agent_builder(output_value))
                            else:
                                # In test environments where there might not be a running loop
                                asyncio.ensure_future(self._call_agent_builder(output_value))
                                
                            # Update the files after processing input to show progress
                            asyncio.create_task(self._update_preview_files(user_input))
                    
                    return result
                
                # Add a helper method to call the agent builder
                async def _call_agent_builder(self, output_value):
                    """Helper method to call the agent builder with the output value."""
                    try:
                        await self.agent_builder.process(output_value)
                    except Exception as e:
                        print(f"Error calling agent builder: {e}")
                        
                # Add a helper method to update the preview files
                async def _update_preview_files(self, user_input):
                    """Update the preview files based on the agent builder's generated code."""
                    try:
                        if hasattr(self, 'agent_builder') and self.agent_builder:
                            # Get variables needed from the parent scope
                            step_dir = step_dir  # This captures the variable from the outer scope
                            step_class_name = step_class_name  # This captures the variable from the outer scope
                            workflow_path = workflow_path  # This captures the variable from the outer scope
                            
                            # Wait a short time for the agent builder to process the input
                            await asyncio.sleep(0.5)
                            
                            # Get the file writer tool
                            file_writer_tool = None
                            if hasattr(self.agent_builder, 'executor') and hasattr(self.agent_builder.executor, 'tools'):
                                file_writer_tool = next((tool for tool in self.agent_builder.executor.tools if tool.__class__.__name__ == 'StepFileWriter'), None)
                            
                            if not file_writer_tool and hasattr(builder, 'agent') and hasattr(builder.agent, 'tools'):
                                file_writer_tool = next((tool for tool in builder.agent.tools if tool.__class__.__name__ == 'StepFileWriter'), None)
                            
                            if file_writer_tool:
                                # Get the generated code from the agent builder
                                step_code = self.agent_builder.get_generated_code()
                                if step_code:
                                    # Validate the code before updating files
                                    validation_result = await validate_generated_code(step_code, step_class_name)
                                    if validation_result.get("valid", False):
                                        print(f"✅ Code quality check passed: {validation_result.get('message', '')}")
                                    else:
                                        print(f"⚠️ Code quality warning: {validation_result.get('message', 'Unknown issue')}")
                                        
                                    # Update the step class file
                                    preview_path = os.path.join(step_dir, f'{step_class_name}.py')
                                    await file_writer_tool.create_file(preview_path, step_code)
                                    print(f"📄 Updated step file: {preview_path}")
                                    print(f"   - {len(step_code.splitlines())} lines of code")
                                    print(f"   - Implements class {step_class_name}")
                                
                                # Get the generated config from the agent builder
                                config = self.agent_builder.get_generated_config()
                                if config:
                                    # Update the config file
                                    preview_path = os.path.join(step_dir, 'config', f'{step_class_name}.yml')
                                    await file_writer_tool.create_file(preview_path, config)
                                    print(f"📄 Updated config file: {preview_path}")
                                
                                # Get the generated tests from the agent builder
                                tests = self.agent_builder.get_generated_tests()
                                if tests:
                                    # Update the test file
                                    preview_path = os.path.join(workflow_path, 'test', f'test_{step_class_name}.py')
                                    await file_writer_tool.create_file(preview_path, tests)
                                    print(f"📄 Updated test file: {preview_path}")
                                    
                                # Show a summary of what was generated
                                print("\n📊 Code Generation Summary:")
                                print(f"   - Step Implementation: {'✅ Generated' if step_code else '❌ Not Generated'}")
                                print(f"   - Configuration: {'✅ Generated' if config else '❌ Not Generated'}")
                                print(f"   - Test Cases: {'✅ Generated' if tests else '❌ Not Generated'}")
                    except Exception as e:
                        print(f"Error updating preview files: {e}")
                        traceback.print_exc()
                
                # Add a helper method to show status
                async def _show_status(self):
                    """Show the current status of the step creation process."""
                    try:
                        if hasattr(self, 'agent_builder') and self.agent_builder:
                            # Get variables needed from the parent scope
                            step_dir = step_dir  # This captures the variable from the outer scope
                            step_class_name = step_class_name  # This captures the variable from the outer scope
                            base_class = base_class  # This captures the variable from the outer scope
                            description = description  # This captures the variable from the outer scope
                            
                            # Get the generated code, config, and tests
                            step_code = self.agent_builder.get_generated_code()
                            config = self.agent_builder.get_generated_config()
                            tests = self.agent_builder.get_generated_tests()
                            
                            print("\n📊 Current Step Status:")
                            print(f"   Step Name: {step_class_name}")
                            print(f"   Base Class: {base_class}")
                            print(f"   Description: {description}")
                            print(f"   Directory: {step_dir}")
                            
                            print("\n📊 Generated Content:")
                            print(f"   - Step Implementation: {'✅ Generated' if step_code else '❌ Not Generated'}")
                            if step_code:
                                print(f"     * {len(step_code.splitlines())} lines of code")
                                # Count methods
                                method_count = len(re.findall(r'def\s+\w+\s*\(', step_code))
                                print(f"     * {method_count} methods defined")
                            
                            print(f"   - Configuration: {'✅ Generated' if config else '❌ Not Generated'}")
                            if config:
                                print(f"     * {len(config.splitlines())} lines of configuration")
                            
                            print(f"   - Test Cases: {'✅ Generated' if tests else '❌ Not Generated'}")
                            if tests:
                                print(f"     * {len(tests.splitlines())} lines of test code")
                                # Count test methods
                                test_method_count = len(re.findall(r'def\s+test_\w+\s*\(', tests))
                                print(f"     * {test_method_count} test methods defined")
                            
                            # Validate the code
                            if step_code:
                                validation_result = await validate_generated_code(step_code, step_class_name)
                                print(f"\n🧪 Code Quality: {'✅ Good' if validation_result.get('valid', False) else '⚠️ Needs Improvement'}")
                                print(f"   {validation_result.get('message', '')}")
                                
                            # Provide suggestions
                            print("\n💡 Suggested Next Steps:")
                            if not step_code or not method_count:
                                print("   - Describe the functionality you want this step to implement")
                            elif method_count < 2:
                                print("   - Add more methods to your step (processing, validation, etc.)")
                            elif not tests or test_method_count < 2:
                                print("   - Add more test cases to ensure your step works correctly")
                            else:
                                print("   - The step looks good! Type 'finish' to complete")
                    except Exception as e:
                        print(f"Error showing status: {e}")
                        traceback.print_exc()
                
                # Apply the helper methods
                command_line.process = enhanced_process.__get__(command_line, command_line.__class__)
                command_line._call_agent_builder = _call_agent_builder.__get__(command_line, command_line.__class__)
                command_line._update_preview_files = _update_preview_files.__get__(command_line, command_line.__class__)
                command_line._show_status = _show_status.__get__(command_line, command_line.__class__)
                
                # Add special command handling to the start_monitoring method
                original_start_monitoring = command_line.start_monitoring
                
                async def enhanced_start_monitoring(self):
                    """Enhanced start_monitoring method with additional command handling."""
                    if self._monitoring:
                        return  # Already monitoring
                        
                    self._monitoring = True
                    
                    # Show welcome message
                    print(self.welcome_message)
                    
                    # Show instructions for the user
                    print("\n💡 You are now in an interactive step creation session.")
                    print("💡 Describe what you want this step to do, and I'll create the code for you.")
                    print("💡 You can use the following commands:")
                    print("   - Type 'help' to see available commands")
                    print("   - Type 'status' to see the current status of your step")
                    print("   - Type 'finish' when you're done creating the step")
                    print("   - Type 'link <source_step> <target_step>' to link steps")
                    print("   - Any other input will be used to enhance the step's code\n")
                    
                    # Show data flow info if in debug mode
                    if self._debug_mode:
                        print("\nData Flow Information:")
                        print("1. User input -> DataStorageCommandLine.process")
                        print("2. DataStorageCommandLine.process -> _force_output_change")
                        print("3. _force_output_change -> output.set")
                        print("4. TriggerDataUpdated detects output change")
                        print("5. TriggerDataUpdated -> LinkDirect.transfer")
                        print("6. LinkDirect.transfer -> AgentWorkflowBuilder.process")
                        
                        # Show connection information
                        if hasattr(self, 'agent_builder') and self.agent_builder:
                            print("\nDirect Connection: CommandLine -> AgentBuilder (backup path)")
                        else:
                            print("\nNo direct connection. Using Link/Trigger mechanism only.")
                            
                        # Show output information
                        if hasattr(self, 'output') and self.output:
                            print(f"Output Data Unit: {self.output.__class__.__name__}")
                        else:
                            print("No output data unit configured.")
                    
                    try:
                        # Loop to get user input
                        while self._monitoring:
                            # Show prompt and get input
                            sys.stdout.write(self.prompt)
                            sys.stdout.flush()
                            
                            # Get user input
                            try:
                                user_input = await asyncio.get_event_loop().run_in_executor(None, input)
                            except EOFError:
                                # Handle Ctrl+D
                                print("\nEOF detected. Exiting.")
                                break
                            except KeyboardInterrupt:
                                # Handle Ctrl+C
                                print("\nInterrupt detected. Exiting.")
                                break
                            
                            # Check for exit command
                            if user_input.strip().lower() == self.exit_command:
                                print(self.goodbye_message)
                                break
                            
                            try:
                                # Handle special commands directly here to avoid async issues
                                if user_input.strip().lower() == "finish":
                                    # Handle finish command
                                    print("✅ Finishing step creation...")
                                    
                                    # Stop monitoring to exit the loop
                                    self._monitoring = False
                                    response = "Step creation completed."
                                    self._add_to_history(user_input, response)
                                    self._force_output_change(response)
                                    break
                                elif user_input.strip().lower() == "help":
                                    # Handle help command
                                    help_text = """
Available commands:
1. link <source_step> <target_step> [link_type] - Link this step to another step
2. status - Show the current status of your step
3. finish - End step creation and save
4. help - Show this menu

Other inputs will be used to enhance the step's code. Examples:
- "Add a method to process JSON data"
- "Implement error handling for network requests"
- "The step should validate input parameters"
"""
                                    print(help_text)
                                    
                                    self._add_to_history(user_input, help_text)
                                    self._force_output_change(help_text)
                                    continue
                                elif user_input.strip().lower() == "status":
                                    # Handle status command
                                    await self._show_status()
                                    continue
                                elif user_input.strip().lower().startswith("link "):
                                    # Handle link command: link <source_step> <target_step>
                                    parts = user_input.strip().split()
                                    if len(parts) >= 3:
                                        source_step = parts[1]
                                        target_step = parts[2]
                                        link_type = parts[3] if len(parts) > 3 else "LinkDirect"
                                        
                                        print(f"🔗 Linking steps: {source_step} -> {target_step} using {link_type}...")
                                        
                                        response = f"Linking steps: {source_step} -> {target_step} using {link_type}"
                                        self._add_to_history(user_input, response)
                                        self._force_output_change(response)
                                        continue
                                
                                # For all other inputs, use the process method
                                print("⚙️ Processing your input and updating the step code...")
                                result = await self.process(user_input)
                                
                                if result:
                                    print(f"\n✅ Code updated based on your input. Keep adding more details or type 'finish' when done.")
                                else:
                                    print(f"\n⚠️ No changes were made. Please provide more specific instructions.")
                                
                                if self._debug_mode:
                                    print(f"Process result: {result}")
                                    
                                    # Check if output was updated
                                    if hasattr(self, 'output') and self.output:
                                        print(f"Current output value: {self.output.get()}")
                                
                            except asyncio.CancelledError:
                                # Handle cancellation
                                break
                            except Exception as e:
                                # Handle other exceptions
                                print(f"❌ Error processing input: {e}")
                                traceback.print_exc()
                    
                    except asyncio.CancelledError:
                        # Task was cancelled
                        pass
                    finally:
                        # Clean up
                        self._monitoring = False
                        print("\n✅ Step creation session ended. Files will be saved.")

                # Apply the enhanced start_monitoring method
                command_line.start_monitoring = enhanced_start_monitoring.__get__(command_line, command_line.__class__)
                
                # Set up direct link using LinkDirect
                from src.LinkDirect import LinkDirect
                from src.TriggerDataUpdated import TriggerDataUpdated
                
                # Create the link first with auto_setup_trigger=False
                link = LinkDirect(
                    source_step=command_line,
                    sink_step=agent_builder,
                    link_id="command_line_to_agent",
                    debug_mode=True,
                    auto_setup_trigger=False  # Don't auto setup the trigger
                )
                
                # Debug print to identify the object
                print(f"Link object: {link}, type: {type(link)}")
                print(f"Link has _debug_mode: {hasattr(link, '_debug_mode')}")
                print(f"Link has debug_mode: {hasattr(link, 'debug_mode')}")
                print(f"Link has _monitoring: {hasattr(link, '_monitoring')}")
                
                # Debug print to identify command_line and agent_builder
                print(f"Command line object: {command_line}, type: {type(command_line)}")
                print(f"Command line has _monitoring: {hasattr(command_line, '_monitoring')}")
                print(f"Agent builder object: {agent_builder}, type: {type(agent_builder)}")
                print(f"Agent builder has _monitoring: {hasattr(agent_builder, '_monitoring')}")
                
                # Explicitly create and set up the trigger - source_step is first parameter, runnable is second
                trigger = TriggerDataUpdated(source_step=command_line, runnable=link)
                
                # Debug print to identify the trigger object
                print(f"Trigger object: {trigger}, type: {type(trigger)}")
                print(f"Trigger has _debug_mode: {hasattr(trigger, '_debug_mode')}")
                print(f"Trigger has debug_mode: {hasattr(trigger, 'debug_mode')}")
                print(f"Trigger has _monitoring: {hasattr(trigger, '_monitoring')}")
                
                link.trigger = trigger
                
                # Start monitoring the trigger directly instead of through the link
                # The TriggerDataUpdated.start_monitoring() is not async, so no await needed
                trigger.start_monitoring()
                
                # Use link's debug mode instead of self
                if link._debug_mode:
                    print(f"LinkDirect: Started monitoring {command_line.name} -> {agent_builder.name}\n")
            except Exception as e:
                # Log the error but don't re-raise, as we want to continue with the rest of the setup
                print(f"Error setting up command line and agent builder: {e}")
                traceback.print_exc()
            
            # Start the command line monitoring
            await command_line.start_monitoring()
            
            # After monitoring stops, save all files and update configuration
            print("\n📝 Finalizing step creation...")
            
            # Update the files with the final generated code
            if file_writer_tool:
                # Create the step class file
                step_content = agent_builder.get_generated_code()
                if step_content:
                    await file_writer_tool.create_file(os.path.join(step_dir, f'{step_class_name}.py'), step_content)
                    print(f"✅ Saved final step class: {os.path.join(step_dir, f'{step_class_name}.py')}")
                
                # Create configuration file
                config_content = agent_builder.get_generated_config()
                if config_content:
                    await file_writer_tool.create_file(os.path.join(step_dir, 'config', f'{step_class_name}.yml'), config_content)
                    print(f"✅ Saved final config: {os.path.join(step_dir, 'config', f'{step_class_name}.yml')}")
                
                # Create test file
                test_content = agent_builder.get_generated_tests()
                if test_content:
                    await file_writer_tool.create_file(os.path.join(workflow_path, 'test', f'test_{step_class_name}.py'), test_content)
                    print(f"✅ Saved final test file: {os.path.join(workflow_path, 'test', f'test_{step_class_name}.py')}")
                    
                # Copy the configuration file to the workflow config directory
                try:
                    workflow_config_dir = os.path.join(workflow_path, 'config')
                    if not os.path.exists(workflow_config_dir):
                        os.makedirs(workflow_config_dir, exist_ok=True)
                        
                    workflow_config_file = os.path.join(workflow_config_dir, f'{step_class_name}.yml')
                    await file_writer_tool.create_file(workflow_config_file, config_content)
                    print(f"✅ Copied config to workflow: {workflow_config_file}")
                except Exception as e:
                    print(f"⚠️ Warning: Could not copy config to workflow: {e}")
            
            # Print a summary of the created files
            print("\n🎉 Step creation completed! Summary:")
            print(f"📁 Step directory: {step_dir}")
            print(f"📄 Step class: {os.path.join(step_dir, f'{step_class_name}.py')}")
            print(f"📄 Config file: {os.path.join(step_dir, 'config', f'{step_class_name}.yml')}")
            print(f"📄 Test file: {os.path.join(workflow_path, 'test', f'test_{step_class_name}.py')}")
            print(f"📄 Workflow config: {os.path.join(workflow_path, 'config', f'{step_class_name}.yml')}")
            
            # Return success
            return {
                "success": True,
                "message": f"Created step {step_class_name} at {step_dir}",
                "step_dir": step_dir,
                "step_class_name": step_class_name
            }
            
        except Exception as e:
            # Clean up if an error occurred
            if need_cleanup and step_dir and os.path.exists(step_dir):
                shutil.rmtree(step_dir)
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": f"Failed to create step: {str(e)}"
            }


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


class SaveStepStep:
    """
    Save a step.
    
    This step finalizes a step by ensuring it is properly tested and
    documented.
    """
    @staticmethod
    async def save_step(builder, step_name: str) -> Dict[str, Any]:
        """
        Save a step in the workflow.
        
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
            test_result = await TestStepStep.test_step(builder, step_name)
            
            if not test_result.get('success', False):
                return {
                    "success": False,
                    "error": f"Tests failed for step {step_class_name}: {test_result.get('error', 'Unknown error')}"
                }
            
            # Check if the configuration file exists in the workflow config directory
            config_dir = os.path.join(workflow_path, 'config')
            config_file = os.path.join(config_dir, f'{step_class_name}.yml')
            
            file_writer_tool = None
            if hasattr(builder, 'agent') and hasattr(builder.agent, 'tools'):
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
    async def link_steps(builder, source_step: str, target_step: str, link_type: str = "LinkDirect") -> Dict[str, Any]:
        """
        Link two steps together in the workflow.
        
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
            
            file_writer_tool = None
            if hasattr(builder, 'agent') and hasattr(builder.agent, 'tools'):
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
    async def save_workflow(builder) -> Dict[str, Any]:
        """
        Save the current workflow.
        
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


async def validate_generated_code(code: str, class_name: str) -> Dict[str, Any]:
    """
    Validate generated code for quality and completeness.
    
    Args:
        code: The code to validate
        class_name: Expected class name
        
    Returns:
        Dictionary with validation results
    """
    result = {
        "valid": True,
        "message": "Code looks good!",
        "issues": []
    }
    
    # Check if the code contains the expected class
    if f"class {class_name}" not in code:
        result["valid"] = False
        result["issues"].append(f"Expected class '{class_name}' not found")
    
    # Check if the code has necessary imports
    if "import" not in code and "from" not in code:
        result["valid"] = False
        result["issues"].append("No imports found")
    
    # Check if the code has a process method
    if "async def process" not in code and "def process" not in code:
        result["valid"] = False
        result["issues"].append("No process method found")
    
    # Check if the code has a constructor
    if "__init__" not in code:
        result["valid"] = False
        result["issues"].append("No constructor found")
    
    # Check if the code has docstrings
    if '"""' not in code and "'''" not in code:
        result["valid"] = False
        result["issues"].append("No docstrings found")
    
    # Check code length
    lines = code.splitlines()
    if len(lines) < 20:
        result["valid"] = False
        result["issues"].append(f"Code seems too short ({len(lines)} lines)")
    
    # Check if the code has proper indentation
    if not any(line.startswith("    ") for line in lines):
        result["valid"] = False
        result["issues"].append("No proper indentation found")
    
    # Update message based on issues
    if result["issues"]:
        result["message"] = ", ".join(result["issues"])
    
    return result 