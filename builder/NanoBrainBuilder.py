#!/usr/bin/env python3
"""
NanoBrainBuilder - Core builder for NanoBrain workflows.
"""

from typing import List, Dict, Any, Optional
import os
import asyncio
from pathlib import Path
import argparse
import importlib.util
import yaml
import traceback

from src.Workflow import Workflow
from src.ExecutorBase import ExecutorBase
from src.Step import Step
from src.DataStorageBase import DataStorageBase
from src.DataUnitBase import DataUnitBase
from src.TriggerBase import TriggerBase
from src.enums import ComponentState
from src.Agent import Agent
from tools_common import StepFileWriter, StepPlanner, StepCoder, StepGitInit, StepContextSearch, StepWebSearch

# Import the AgentWorkflowBuilder (will be created next)
from builder.AgentWorkflowBuilder import AgentWorkflowBuilder

# Import the ConfigManager
from src.ConfigManager import ConfigManager
from src.ExecutorFunc import ExecutorFunc
from src.GlobalConfig import GlobalConfig


class CommandLineTrigger(TriggerBase):
    """
    Trigger that activates when command line input is received.
    
    Biological analogy: Sensory neuron responding to external stimuli.
    Justification: Like how sensory neurons detect specific environmental
    stimuli and convert them to neural signals, this trigger detects
    command line input and activates the workflow.
    """
    def __init__(self, runnable: Any, **kwargs):
        super().__init__(runnable, **kwargs)
        self._monitoring = False
        
    async def monitor(self):
        """Start monitoring for command line input."""
        self._monitoring = True
        
        while self._monitoring:
            try:
                # Wait for user input
                user_input = await self._get_user_input()
                
                # Check if the condition is met
                if self.check_condition(input=user_input):
                    # Activate the runnable
                    if hasattr(self.runnable, 'process'):
                        await self.runnable.process([user_input])
            except asyncio.CancelledError:
                self._monitoring = False
                break
            except Exception as e:
                print(f"Error in command line trigger: {e}")
                self._monitoring = False
                break
    
    async def _get_user_input(self) -> str:
        """Get input from the command line asynchronously."""
        # Create a future to hold the result
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        
        # Run the blocking input call in a separate thread
        def _get_input():
            try:
                result = input("nb> ")
                loop.call_soon_threadsafe(future.set_result, result)
            except Exception as e:
                loop.call_soon_threadsafe(future.set_exception, e)
        
        # Run the input function in a thread
        await loop.run_in_executor(None, _get_input)
        
        # Wait for the result
        return await future
    
    def check_condition(self, **kwargs) -> bool:
        """Check if the condition for triggering is met."""
        # Command line trigger activates on any non-empty input
        input_text = kwargs.get('input', '')
        return bool(input_text.strip())
    
    async def stop_monitoring(self):
        """Stop monitoring for command line input."""
        self._monitoring = False


class CommandLineStorage(DataStorageBase):
    """
    Data storage for command line input and output.
    
    Biological analogy: Sensory processing and motor output areas.
    Justification: Like how sensory areas process input and motor areas
    produce output, this class processes command line input and produces
    command line output.
    """
    def __init__(self, executor: ExecutorBase, **kwargs):
        # Create input and output data units
        input_unit = kwargs.pop('input_unit', None) or DataUnitBase()
        output_unit = kwargs.pop('output_unit', None) or DataUnitBase()
        trigger = kwargs.pop('trigger', None) or CommandLineTrigger(None)
        
        super().__init__(executor, input_unit, output_unit, trigger, **kwargs)
        
        # Set this storage as the runnable for the trigger
        self.trigger.runnable = self
    
    async def _process_query(self, query: Any) -> Any:
        """Process the command line input."""
        # Just print the response to the command line
        return query
    
    def display_response(self, response: Any):
        """Display the response to the command line."""
        if response:
            print(f"{response}")


class NanoBrainBuilder:
    """
    NanoBrain Builder for constructing workflows and steps.
    
    Biological analogy: Brain's prefrontal cortex executive function.
    Justification: Like how the prefrontal cortex handles planning and
    decision-making, the NanoBrainBuilder plans and constructs workflows.
    """
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the NanoBrainBuilder.
        
        Args:
            config_path: Path to the configuration file (optional)
        """
        # Initialize the ConfigManager first
        self.config_manager = ConfigManager()
        
        # Initialize the executor using ConfigManager
        self.executor = self.config_manager.create_instance("ExecutorFunc")
        
        # Get configuration from ConfigManager - NO MANUAL CONFIGURATION
        self.config = self.config_manager.get_config("NanoBrainBuilder")
        
        # Set up internal state - use values from config
        self.workflows_dir = self.config.get('workflows_dir', 'workflows')
        self.current_workflow = None
        self._workflow_stack = []
        
        # Create workflows directory if it doesn't exist
        if not os.path.exists(self.workflows_dir):
            os.makedirs(self.workflows_dir)
        
        # Initialize the agent using ConfigManager
        self._init_agent()
        
        # Initialize the tools using ConfigManager
        self._init_tools()
    
    def _init_agent(self):
        """Initialize the agent from configuration."""
        # Simply use the config_manager to create the Agent instance
        # This will properly load all config from Agent.yml without manual overrides
        # Make sure to explicitly pass the executor which is required
        self.agent = self.config_manager.create_instance("Agent", executor=self.executor)
    
    def _init_tools(self):
        """Initialize the tools for the agent from configuration."""
        import traceback
        
        # Import tool classes dynamically as needed
        from tools_common import (
            StepFileWriter, StepPlanner, StepCoder, 
            StepGitInit, StepContextSearch, StepWebSearch
        )
        
        # Map tool names to classes
        tool_classes = {
            'StepFileWriter': StepFileWriter,
            'StepPlanner': StepPlanner,
            'StepCoder': StepCoder,
            'StepGitInit': StepGitInit,
            'StepContextSearch': StepContextSearch,
            'StepWebSearch': StepWebSearch
        }
        
        # Let ConfigManager get the list of tools from config files
        tools_config = self.config_manager.get_config("NanoBrainBuilder")
        tools_list = tools_config.get('tools', [])
        
        # Create and add tools
        for tool_name in tools_list:
            if tool_name in tool_classes:
                try:
                    # Create tool instance using ConfigManager
                    tool_instance = self.config_manager.create_instance(tool_name, executor=self.executor)
                    
                    # If ConfigManager couldn't create it (no config), fall back to direct instantiation
                    if tool_instance is None and tool_name in tool_classes:
                        tool_instance = tool_classes[tool_name](self.executor)
                        
                    # Add tool to agent
                    self.agent.add_tool(tool_instance)
                    print(f"Successfully added tool: {tool_name}")
                except Exception as e:
                    print(f"Error creating tool {tool_name}: {type(e).__name__} - {str(e)}")
                    print("\nStack trace:")
                    traceback.print_exc()
            else:
                print(f"Warning: Tool '{tool_name}' not found in available tools.")
    
    def get_current_workflow(self) -> Optional[str]:
        """
        Get the current workflow path.
        
        Returns:
            The current workflow path or None if no workflow is active
        """
        if not self._workflow_stack:
            return None
        return self._workflow_stack[-1]

    def get_current_workflow_object(self) -> Optional[Workflow]:
        """
        Get the current workflow object.
        
        Returns:
            The current workflow object or None if no workflow is active
        """
        workflow_path = self.get_current_workflow()
        if not workflow_path:
            return None
            
        try:
            # Get the workflow class name from the path
            workflow_name = os.path.basename(workflow_path)
            workflow_class_name = ''.join(word.capitalize() for word in workflow_name.split('_'))
            
            # Import the workflow module
            spec = importlib.util.spec_from_file_location(
                workflow_class_name,
                os.path.join(workflow_path, 'src', f'{workflow_class_name}.py')
            )
            if spec is None or spec.loader is None:
                return None
                
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Get the workflow class
            workflow_class = getattr(module, workflow_class_name)
            
            # Create and return the workflow instance
            return workflow_class(executor=self.executor)
        except Exception as e:
            print(f"Error getting workflow object: {e}")
            return None
    
    def push_workflow(self, workflow_path: str):
        """
        Push a workflow onto the stack.
        
        Args:
            workflow_path: Path to the workflow directory
        """
        self._workflow_stack.append(workflow_path)
    
    def pop_workflow(self) -> Optional[str]:
        """
        Pop a workflow from the stack.
        
        Returns:
            The popped workflow path or None if the stack is empty
        """
        if not self._workflow_stack:
            return None
        
        return self._workflow_stack.pop()
    
    def create_workflow(self, workflow_name: str) -> Dict[str, Any]:
        """
        Create a new workflow.
        
        Args:
            workflow_name: Name of the workflow to create
            
        Returns:
            Dictionary with the result of the operation
        """
        # Check if workflow already exists
        if workflow_name in self.get_workflows():
            return {
                "success": False,
                "error": f"Workflow '{workflow_name}' already exists."
            }
            
        # Create workflow directory
        workflow_path = os.path.join(self.workflows_dir, workflow_name)
        try:
            os.makedirs(workflow_path, exist_ok=True)
            os.makedirs(os.path.join(workflow_path, 'config'), exist_ok=True)
            
            # Create initial workflow.yml
            workflow_config = {
                "name": workflow_name,
                "description": f"Workflow {workflow_name}",
                "steps": []
            }
            
            with open(os.path.join(workflow_path, 'config', 'workflow.yml'), 'w') as f:
                yaml.dump(workflow_config, f, default_flow_style=False)
                
            # Set as current workflow
            self.current_workflow = workflow_name
                
            return {
                "success": True,
                "message": f"Workflow '{workflow_name}' created successfully."
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to create workflow: {str(e)}"
            }
    
    async def create_step(self, step_name: str, base_class: str = "Step", description: str = None) -> Dict[str, Any]:
        """
        Create a new step.
        
        Args:
            step_name: Name of the step to create
            base_class: Base class for the step (default: "Step")
            description: Description of the step (optional)
        
        Returns:
            Dictionary with the result of the operation
        """
        from builder.WorkflowSteps import CreateStep
        
        return await CreateStep.execute(self, step_name, base_class, description)
    
    async def test_step(self, step_name: str) -> Dict[str, Any]:
        """
        Test a step.
        
        Args:
            step_name: Name of the step to test
        
        Returns:
            Dictionary with the result of the operation
        """
        from builder.WorkflowSteps import TestStepStep
        
        return await TestStepStep.execute(self, step_name)
    
    async def save_step(self, step_name: str) -> Dict[str, Any]:
        """
        Save a step.
        
        Args:
            step_name: Name of the step to save
        
        Returns:
            Dictionary with the result of the operation
        """
        from builder.WorkflowSteps import SaveStepStep
        
        return await SaveStepStep.execute(self, step_name)
    
    async def link_steps(self, source_step: str, target_step: str, link_type: str = "LinkDirect") -> Dict[str, Any]:
        """
        Link steps together.
        
        Args:
            source_step: Name of the source step
            target_step: Name of the target step
            link_type: Type of link to create (default: "LinkDirect")
        
        Returns:
            Dictionary with the result of the operation
        """
        from builder.WorkflowSteps import LinkStepsStep
        
        return await LinkStepsStep.execute(self, source_step, target_step, link_type)
    
    async def save_workflow(self) -> Dict[str, Any]:
        """
        Save a workflow.
        
        Returns:
            Dictionary with the result of the operation
        """
        from builder.WorkflowSteps import SaveWorkflowStep
        
        return await SaveWorkflowStep.execute(self)
    
    async def process_command(self, command: str, args: List[str]) -> Dict[str, Any]:
        """
        Process a command.
        
        Args:
            command: The command to process
            args: List of arguments for the command
        
        Returns:
            Dictionary with the result of the operation
        """
        if command == "create-workflow":
            if len(args) < 1:
                return {
                    "success": False,
                    "error": "Missing workflow name"
                }
            
            workflow_name = args[0]
            username = args[1] if len(args) > 1 else None
            
            return await self.create_workflow(workflow_name)
        
        elif command == "create-step":
            if len(args) < 1:
                return {
                    "success": False,
                    "error": "Missing step name"
                }
            
            step_name = args[0]
            base_class = args[1] if len(args) > 1 else "Step"
            description = args[2] if len(args) > 2 else None
            
            return await self.create_step(step_name, base_class, description)
        
        elif command == "test-step":
            if len(args) < 1:
                return {
                    "success": False,
                    "error": "Missing step name"
                }
            
            step_name = args[0]
            
            return await self.test_step(step_name)
        
        elif command == "save-step":
            if len(args) < 1:
                return {
                    "success": False,
                    "error": "Missing step name"
                }
            
            step_name = args[0]
            
            return await self.save_step(step_name)
        
        elif command == "link-steps":
            if len(args) < 2:
                return {
                    "success": False,
                    "error": "Missing source or target step name"
                }
            
            source_step = args[0]
            target_step = args[1]
            link_type = args[2] if len(args) > 2 else "LinkDirect"
            
            return await self.link_steps(source_step, target_step, link_type)
        
        elif command == "save-workflow":
            return await self.save_workflow()
        
        else:
            return {
                "success": False,
                "error": f"Unknown command: {command}"
            }
    
    @staticmethod
    def get_parser() -> argparse.ArgumentParser:
        """
        Get the argument parser for the builder.
        
        Returns:
            ArgumentParser instance
        """
        parser = argparse.ArgumentParser(description="NanoBrain builder for constructing workflows")
        
        subparsers = parser.add_subparsers(dest="command", help="Command to execute")
        
        # Create workflow command
        create_workflow_parser = subparsers.add_parser("create-workflow", help="Create a new workflow")
        create_workflow_parser.add_argument("name", help="Name of the workflow to create")
        create_workflow_parser.add_argument("--username", help="Username for the git repository")
        
        # Create step command
        create_step_parser = subparsers.add_parser("create-step", help="Create a new step")
        create_step_parser.add_argument("name", help="Name of the step to create")
        create_step_parser.add_argument("--base-class", default="Step", help="Base class for the step")
        create_step_parser.add_argument("--description", help="Description of the step")
        
        # Test step command
        test_step_parser = subparsers.add_parser("test-step", help="Test a step")
        test_step_parser.add_argument("name", help="Name of the step to test")
        
        # Save step command
        save_step_parser = subparsers.add_parser("save-step", help="Save a step")
        save_step_parser.add_argument("name", help="Name of the step to save")
        
        # Link steps command
        link_steps_parser = subparsers.add_parser("link-steps", help="Link steps together")
        link_steps_parser.add_argument("source", help="Name of the source step")
        link_steps_parser.add_argument("target", help="Name of the target step")
        link_steps_parser.add_argument("--link-type", default="LinkDirect", help="Type of link to create")
        
        # Save workflow command
        subparsers.add_parser("save-workflow", help="Save a workflow")
        
        return parser
    
    @classmethod
    async def main(cls):
        """Main entry point for the builder."""
        parser = cls.get_parser()
        args = parser.parse_args()
        
        if not args.command:
            parser.print_help()
            return
        
        # Create the builder
        builder = cls()
        
        # Process the command
        if args.command == "create-workflow":
            result = await builder.create_workflow(args.name)
        elif args.command == "create-step":
            result = await builder.create_step(args.name, args.base_class, args.description)
        elif args.command == "test-step":
            result = await builder.test_step(args.name)
        elif args.command == "save-step":
            result = await builder.save_step(args.name)
        elif args.command == "link-steps":
            result = await builder.link_steps(args.source, args.target, args.link_type)
        elif args.command == "save-workflow":
            result = await builder.save_workflow()
        else:
            print(f"Unknown command: {args.command}")
            parser.print_help()
            return
        
        # Print the result
        if result.get("success", False):
            print(result.get("message", "Command executed successfully"))
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")

    def list_workflows(self) -> List[str]:
        """
        List all available workflows.
        
        Returns:
            List of workflow names
        """
        if not os.path.exists(self.workflows_dir):
            return []
            
        return [d for d in os.listdir(self.workflows_dir) 
                if os.path.isdir(os.path.join(self.workflows_dir, d))]

    def get_workflows(self) -> List[str]:
        """
        Get all available workflows.
        
        This is an alias for list_workflows() for API consistency.
        
        Returns:
            List of workflow names
        """
        return self.list_workflows()
                
    def get_workflow_path(self, workflow_name: str) -> Optional[str]:
        """
        Get the full path to a workflow.
        
        Args:
            workflow_name: Name of the workflow
            
        Returns:
            Full path to the workflow directory or None if not found
        """
        workflow_path = os.path.join(self.workflows_dir, workflow_name)
        return workflow_path if os.path.exists(workflow_path) else None


if __name__ == "__main__":
    # Run the main function
    asyncio.run(NanoBrainBuilder.main()) 