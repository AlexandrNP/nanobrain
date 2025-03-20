#!/usr/bin/env python3
"""
NanoBrainBuilder - Core builder for NanoBrain workflows.

This module provides the main CLI interface for the NanoBrain framework,
handling command parsing and delegating to appropriate components.

Biological analogy: Brain's prefrontal cortex executive function.
Justification: Like how the prefrontal cortex handles planning and
decision-making through integrating information from various brain regions,
the NanoBrainBuilder orchestrates workflow creation by coordinating
between different specialized components.
"""

from typing import List, Dict, Any, Optional, Set, Callable, Tuple, Union
import os
import asyncio
from pathlib import Path
import argparse
import importlib.util
import traceback
import importlib
import sys
import json

from src.Workflow import Workflow
from src.ExecutorBase import ExecutorBase
from src.Step import Step
from src.DataStorageBase import DataStorageBase
from src.DataUnitBase import DataUnitBase
from src.TriggerBase import TriggerBase
from src.enums import ComponentState
from src.Agent import Agent
from src.DataStorageCommandLine import DataStorageCommandLine
from src.LinkDirect import LinkDirect
from src.TriggerDataUpdated import TriggerDataUpdated
from src.DataUnitString import DataUnitString

# Import the AgentWorkflowBuilder
from builder.AgentWorkflowBuilder import AgentWorkflowBuilder

# Import the ConfigManager
from src.ConfigManager import ConfigManager
from src.ExecutorFunc import ExecutorFunc
from src.GlobalConfig import GlobalConfig

# Import WorkflowSteps for handling predefined commands
from builder.WorkflowSteps import (
    CreateWorkflow, CreateStep, TestStepStep,
    SaveStepStep, LinkStepsStep, SaveWorkflowStep
)

# Import the ensure_async decorator
from builder.utils import ensure_async

# Import nanobrain components
from src.regulations import ConnectionStrength, SystemRegulator, SystemModulator
from src.enums import AgentType


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




class NanoBrainBuilder:
    """
    NanoBrain Builder for constructing workflows and steps.
    
    This class serves as the main CLI interface for the NanoBrain framework,
    handling command parsing and delegating to appropriate components:
    1. For predefined commands: Invoke WorkflowSteps methods
    2. For step creation: Pass control to DataStorageCommandLine linked with AgentCodeWriter
    3. For non-predefined commands: Pass to AgentWorkflowBuilder for guidance
    
    Biological analogy: Brain's prefrontal cortex executive function.
    Justification: Like how the prefrontal cortex handles planning and
    decision-making, the NanoBrainBuilder plans and constructs workflows
    by orchestrating specialized components.
    """
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the NanoBrainBuilder.
        
        Args:
            config_path: Path to the configuration file (optional)
        """
        # Get the current directory as the base path
        base_path = os.path.dirname(os.path.abspath(__file__))
        
        # Initialize the ConfigManager first
        self.config_manager = ConfigManager(base_path=base_path)
        
        # Initialize the executor using ConfigManager
        self.executor = self.config_manager.create_instance("ExecutorFunc")
        
        # Get configuration from ConfigManager - NO MANUAL CONFIGURATION
        self.config = self.config_manager.get_config("NanoBrainBuilder")
        
        # Set up internal state - use values from config
        self.workflows_dir = self.config.get('workflows_dir', 'workflows')
        self.current_workflow = None
        self._workflow_stack = []
        
        # Set debug mode
        self._debug_mode = self.config.get('debug', False)
        
        # Create workflows directory if it doesn't exist
        if not os.path.exists(self.workflows_dir):
            os.makedirs(self.workflows_dir)
        
        # Initialize the agents
        self._init_agent()
        
        # Set predefined commands
        self._predefined_commands = {
            "create_workflow": self.create_workflow,
            "create_step": self.create_step,
            "test_step": self.test_step,
            "save_step": self.save_step,
            "link_steps": self.link_steps,
            "save_workflow": self.save_workflow,
            "list_workflows": lambda: {"success": True, "workflows": self.list_workflows()},
            "push_workflow": lambda workflow_name: {"success": True, "workflow": self.push_workflow(workflow_name)},
            "pop_workflow": lambda: {"success": True, "workflow": self.pop_workflow()},
            "get_current_workflow": lambda: {"success": True, "workflow": self.get_current_workflow()},
            "help": self._show_help,
        }
    
    def _init_agent(self):
        """
        Initialize the agents needed by the NanoBrainBuilder:
        1. AgentWorkflowBuilder - provides guidance on NanoBrain framework
        2. AgentCodeWriter - template accessible by DataStorageCommandLine for creating code
        
        All agent tools are initialized automatically via YAML configuration
        during the agent creation process. The AgentWorkflowBuilder class
        reads the tools.yml file and loads the specified tools.
        """
        # Create a DataStorageCommandLine instance using ConfigManager
        input_storage = self.config_manager.create_instance(
            "DataStorageCommandLine",
            executor=self.executor,
            _debug_mode=self._debug_mode
        )
        
        # Get the tools config path
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_dir = os.path.join(project_root, "builder", "config")
        tools_config_path = os.path.join(config_dir, "tools.yml")
        
        # Ensure the config directory exists
        os.makedirs(config_dir, exist_ok=True)
        
        if self._debug_mode:
            print(f"Using tools config path: {tools_config_path}")
            print(f"Tools config exists: {os.path.exists(tools_config_path)}")
        
        # Initialize AgentWorkflowBuilder for guidance
        # The AgentWorkflowBuilder will load tools from the YAML configuration automatically
        # in its __init__ method through the _load_tools() function
        self.agent = self.config_manager.create_instance(
            configuration_name = "AgentWorkflowBuilder.yml", 
            executor=self.executor,
            input_storage=input_storage,
            tools_config_path=tools_config_path,
            _debug_mode=self._debug_mode,
            use_code_writer=False  # This agent provides guidance only
        )
        
        
        # We don't directly initialize AgentCodeWriter here
        # It will be created by the CreateStep.execute method when needed
        
        # Create a tool_map that maps tool names to their classes
        self.tool_map = {}
        if hasattr(self.agent, 'tools'):
            for tool in self.agent.tools:
                tool_class = tool.__class__
                tool_name = tool_class.__name__
                self.tool_map[tool_name] = tool_class
        
        if self._debug_mode:
            print(f"Initialized AgentWorkflowBuilder for guidance")
            if hasattr(self.agent, 'tools'):
                print(f"Agent has {len(self.agent.tools)} tools loaded.")
                for i, tool in enumerate(self.agent.tools):
                    print(f"  {i+1}. {tool.__class__.__name__}")
            else:
                print("Agent has no tools loaded.")
    
    def _show_help(self) -> Dict[str, Any]:
        """
        Show help information.
        
        Returns:
            Dictionary with the available commands
        """
        commands = list(self._predefined_commands.keys())
        
        return {
            "success": True,
            "commands": commands,
            "message": "Available commands: " + ", ".join(commands)
        }
    
    def get_current_workflow(self) -> Optional[str]:
        """
        Get the path to the current workflow.
        
        Returns:
            Path to the current workflow or None if no workflow is active
        """
        return self.current_workflow
    
    def get_current_workflow_object(self) -> Optional[Workflow]:
        """
        Get the current workflow object.
        
        Returns:
            Current workflow object or None if no workflow is active
        """
        workflow_path = self.get_current_workflow()
        
        if not workflow_path:
            return None
        
        try:
            # Create a workflow-specific ConfigManager
            workflow_config_manager = ConfigManager(base_path=workflow_path)
            
            # Look for workflow configuration
            config = workflow_config_manager.get_config("workflow")
            
            if not config:
                return None
            
            # Get the entry point
            entry_point = config.get('entry_point')
            
            if not entry_point:
                return None
            
            # Check if the workflow configuration specifies a class
            if 'class' in config:
                # If a specific class is defined, use ConfigManager to create the instance
                workflow = workflow_config_manager.create_instance(configuration_name="workflow")
                return workflow
            
            # If no class is specified, use the traditional loading from entry_point
            module_path = os.path.join(workflow_path, entry_point)
            
            if not os.path.exists(module_path):
                return None
            
            # Import the module
            spec = importlib.util.spec_from_file_location("workflow_module", module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Get the workflow class
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                
                if isinstance(attr, type) and issubclass(attr, Workflow) and attr != Workflow:
                    # Create an instance of the workflow
                    workflow = attr()
                    return workflow
            
            return None
        except Exception as e:
            print(f"Error loading workflow: {e}")
            return None
    
    def push_workflow(self, workflow_path: str):
        """
        Push a workflow to the stack and make it the current workflow.
        
        Args:
            workflow_path: Path to the workflow
        """
        if self.current_workflow:
            self._workflow_stack.append(self.current_workflow)
            
        self.current_workflow = workflow_path
        return workflow_path
    
    def pop_workflow(self) -> Optional[str]:
        """
        Pop a workflow from the stack and make it the current workflow.
        
        Returns:
            Path to the current workflow or None if no workflow is active
        """
        if not self._workflow_stack:
            self.current_workflow = None
            return None
            
        self.current_workflow = self._workflow_stack.pop()
        return self.current_workflow
    
    @ensure_async
    async def create_workflow(self, workflow_name: str) -> Dict[str, Any]:
        """
        Create a new workflow.
        
        Args:
            workflow_name: Name of the workflow to create
        
        Returns:
            Dictionary with the result of the operation
        """
        # Delegate to the static create_workflow method in CreateWorkflow class
        return await CreateWorkflow.create_workflow(self, workflow_name)
    
    async def create_step(self, step_name: str, base_class: str = "Step", description: str = None) -> Dict[str, Any]:
        """
        Create a new step.
        
        This method delegates to the CreateStep class which:
        1. Sets up DataStorageCommandLine for step-specific CLI
        2. Connects it to an AgentCodeWriter via LinkDirect
        3. Handles the interactive session for creating the step
        
        Args:
            step_name: Name of the step to create
            base_class: Base class for the step (default: "Step")
            description: Description of the step (optional)
        
        Returns:
            Dictionary with the result of the operation
        """
        # Delegate to the static execute method of CreateStep
        return await CreateStep.execute(self, step_name, base_class, description)
    
    async def test_step(self, step_name: str) -> Dict[str, Any]:
        """
        Test a step.
        
        Args:
            step_name: Name of the step to test
        
        Returns:
            Dictionary with the result of the operation
        """
        # Delegate to the static test_step method of TestStepStep
        return await TestStepStep.test_step(self, step_name)
    
    async def save_step(self, step_name: str) -> Dict[str, Any]:
        """
        Save a step.
        
        Args:
            step_name: Name of the step to save
        
        Returns:
            Dictionary with the result of the operation
        """
        # Delegate to the static save_step method of SaveStepStep
        return await SaveStepStep.save_step(self, step_name)
    
    async def link_steps(self, source_step: str, target_step: str, link_type: str = "LinkDirect") -> Dict[str, Any]:
        """
        Link two steps.
        
        Args:
            source_step: Name of the source step
            target_step: Name of the target step
            link_type: Type of link (default: "LinkDirect")
        
        Returns:
            Dictionary with the result of the operation
        """
        # Delegate to the static link_steps method of LinkStepsStep
        return await LinkStepsStep.link_steps(self, source_step, target_step, link_type)
    
    async def save_workflow(self) -> Dict[str, Any]:
        """
        Save the current workflow.
        
        Returns:
            Dictionary with the result of the operation
        """
        # Delegate to the static save_workflow method of SaveWorkflowStep
        return await SaveWorkflowStep.save_workflow(self)
    
    async def process_command(self, command: str, args: List[str]) -> Dict[str, Any]:
        """
        Process a command.
        
        This is the main method for handling command line input:
        1. Check if the command is predefined and execute it if it is
        2. Otherwise, pass the command to the AgentWorkflowBuilder for guidance
        
        Args:
            command: Command to process
            args: Arguments for the command
        
        Returns:
            Dictionary with the result of the operation
        """
        # Normalize command
        command = command.lower().strip()
        
        # Check if the command is predefined
        if command in self._predefined_commands:
            try:
                # Get the command handler
                handler = self._predefined_commands[command]
                
                # Check if the handler is asynchronous
                if asyncio.iscoroutinefunction(handler):
                    # Call the handler with the arguments
                    if args:
                        result = await handler(*args)
                    else:
                        result = await handler()
                else:
                    # Call the handler with the arguments
                    if args:
                        result = handler(*args)
                    else:
                        result = handler()
                
                return result
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Error processing command '{command}': {str(e)}"
                }
        else:
            # For non-predefined commands, pass the input to the AgentWorkflowBuilder
            try:
                # Prepare the input for the agent
                agent_input = f"{command} {' '.join(args)}" if args else command
                
                # Process the input with the agent and get the response
                response = await self.agent.process([agent_input])
                
                # Return the result
                return {
                    "success": True,
                    "response": response
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Error processing input with AgentWorkflowBuilder: {str(e)}"
                }
    
    @staticmethod
    def get_parser() -> argparse.ArgumentParser:
        """
        Get the argument parser for command line arguments.
        
        Returns:
            Argument parser
        """
        parser = argparse.ArgumentParser(description="NanoBrain Builder for constructing workflows and steps.")
        
        # Command subparsers
        subparsers = parser.add_subparsers(dest="command", help="Command to execute")
        
        # Create workflow subparser
        create_workflow_parser = subparsers.add_parser("create_workflow", help="Create a new workflow")
        create_workflow_parser.add_argument("workflow_name", help="Name of the workflow to create")
        
        # Create step subparser
        create_step_parser = subparsers.add_parser("create_step", help="Create a new step")
        create_step_parser.add_argument("step_name", help="Name of the step to create")
        create_step_parser.add_argument("--base-class", dest="base_class", default="Step", help="Base class for the step")
        create_step_parser.add_argument("--description", dest="description", help="Description of the step")
        
        # Test step subparser
        test_step_parser = subparsers.add_parser("test_step", help="Test a step")
        test_step_parser.add_argument("step_name", help="Name of the step to test")
        
        # Save step subparser
        save_step_parser = subparsers.add_parser("save_step", help="Save a step")
        save_step_parser.add_argument("step_name", help="Name of the step to save")
        
        # Link steps subparser
        link_steps_parser = subparsers.add_parser("link_steps", help="Link two steps")
        link_steps_parser.add_argument("source_step", help="Name of the source step")
        link_steps_parser.add_argument("target_step", help="Name of the target step")
        link_steps_parser.add_argument("--link-type", dest="link_type", default="LinkDirect", help="Type of link")
        
        # Save workflow subparser
        subparsers.add_parser("save_workflow", help="Save the current workflow")
        
        # List workflows subparser
        subparsers.add_parser("list_workflows", help="List available workflows")
        
        # Push workflow subparser
        push_workflow_parser = subparsers.add_parser("push_workflow", help="Push a workflow to the stack")
        push_workflow_parser.add_argument("workflow_name", help="Name of the workflow to push")
        
        # Pop workflow subparser
        subparsers.add_parser("pop_workflow", help="Pop a workflow from the stack")
        
        # Get current workflow subparser
        subparsers.add_parser("get_current_workflow", help="Get the current workflow")
        
        # Help subparser
        subparsers.add_parser("help", help="Show help information")
        
        return parser
    
    @classmethod
    async def main(cls):
        """
        Main entry point for the NanoBrainBuilder.
        
        This sets up the NanoBrainBuilder and handles the command line interface.
        """
        try:
            # Create the builder
            builder = cls()
            
            # Get the argument parser
            parser = cls.get_parser()
            
            # Parse the arguments
            args = parser.parse_args()
            
            # Get the command
            command = args.command
            
            if command:
                # Convert args to dictionary
                args_dict = vars(args)
                
                # Remove the command
                del args_dict["command"]
                
                # Get the arguments
                command_args = list(args_dict.values())
                
                # Process the command
                result = await builder.process_command(command, command_args)
                
                # Print the result
                print(result)
            else:
                # Start the interactive session
                command_line_trigger = CommandLineTrigger(builder)
                
                # Start monitoring for command line input
                await command_line_trigger.monitor()
        except KeyboardInterrupt:
            print("\nProcess interrupted by user.")
        except Exception as e:
            print(f"Error in main process: {e}")
            traceback.print_exc()
    
    def list_workflows(self) -> List[str]:
        """
        List available workflows.
        
        Returns:
            List of available workflows
        """
        # Get workflows directory
        workflows_dir = self.config.get('workflows_dir', 'workflows')
        
        # Check if the directory exists
        if not os.path.exists(workflows_dir):
            return []
            
        # Get the workflows
        workflows = []
        
        for item in os.listdir(workflows_dir):
            # Check if it's a directory
            if os.path.isdir(os.path.join(workflows_dir, item)):
                workflows.append(item)
                
        return workflows
    
    def get_workflows(self) -> List[str]:
        """
        Get available workflows.
        
        Returns:
            List of available workflows
        """
        return self.list_workflows()
    
    def get_workflow_path(self, workflow_name: str) -> Optional[str]:
        """
        Get the path to a workflow.
        
        Args:
            workflow_name: Name of the workflow
        
        Returns:
            Path to the workflow or None if not found
        """
        # Get workflows directory
        workflows_dir = self.config.get('workflows_dir', 'workflows')
        
        # Get the workflow path
        workflow_path = os.path.join(workflows_dir, workflow_name)
        
        # Check if the directory exists
        if not os.path.exists(workflow_path):
            return None
            
        return workflow_path

# Entry point for the command line interface
if __name__ == "__main__":
    # Run the main function
    asyncio.run(NanoBrainBuilder.main()) 