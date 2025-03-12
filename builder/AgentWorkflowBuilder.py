"""
AgentWorkflowBuilder Module

This module provides a specialized builder for creating Agent-based workflows.
It extends the basic workflow building capabilities with agent-specific features.

Biological analogy: Neural circuit formation during development.
Justification: Like how neural circuits are formed through a guided process
during development, this builder helps construct agent-based workflows with
appropriate connections and configurations.
"""

from typing import List, Dict, Any, Optional
import os
import asyncio
from pathlib import Path

from src.Workflow import Workflow
from src.Agent import Agent
from src.Step import Step
from src.ExecutorBase import ExecutorBase
from src.enums import ComponentState
from src.DataUnitBase import DataUnitBase


class AgentWorkflowBuilder(Agent, DataUnitBase):
    """
    Builder for creating Agent-based workflows.
    
    Biological analogy: Neural circuit formation during development.
    Justification: Like how neural circuits are formed through a guided process
    during development, this builder helps construct agent-based workflows with
    appropriate connections and configurations.
    """
    
    def __init__(self, executor: Optional[ExecutorBase] = None, **kwargs):
        """Initialize the AgentWorkflowBuilder."""
        # Initialize Agent base class
        super().__init__(executor=executor or ExecutorBase(), **kwargs)
        
        # Initialize DataUnitBase attributes
        self.data = None
        self.persistence_level = 0.8
        
        # Builder-specific attributes
        self.workflows: List[Workflow] = []
        self.current_workflow: Optional[Workflow] = None
        self.generated_code: str = ""
        self.generated_config: str = ""
        self.generated_tests: str = ""
        self.current_context: Dict[str, Any] = {}
    
    def get(self) -> Any:
        """Get the current data (implements DataUnitBase)."""
        return self.data
    
    def set(self, data: Any) -> bool:
        """Set the data and process it (implements DataUnitBase)."""
        self.data = data
        asyncio.create_task(self._process_input(data))
        return True
    
    async def _process_input(self, input_data: str) -> None:
        """
        Process input from the command line interface.
        
        Args:
            input_data: The command or code input from the user
        """
        if not input_data:
            return
            
        # Parse the command
        parts = input_data.strip().split()
        command = parts[0].lower() if parts else ""
        
        if command == "link":
            if len(parts) >= 3:
                source_step = parts[1]
                target_step = parts[2]
                await self._handle_link_command(source_step, target_step)
        elif command == "help":
            self._display_help()
        else:
            # Treat as code input for the current step
            await self._handle_code_input(input_data)
    
    async def _handle_link_command(self, source_step: str, target_step: str) -> None:
        """
        Handle the link command to connect two steps.
        
        Args:
            source_step: Name of the source step
            target_step: Name of the target step
        """
        # Update the current context with link information
        self.current_context.setdefault("links", []).append({
            "source": source_step,
            "target": target_step
        })
        
        # Generate code for the link
        link_code = await self._generate_link_code(source_step, target_step)
        self.generated_code += f"\n{link_code}"
        
        print(f"Added link from {source_step} to {target_step}")
    
    async def _handle_code_input(self, code_input: str) -> None:
        """
        Handle code input from the user.
        
        Args:
            code_input: The code or instructions from the user
        """
        # Update the current context
        self.current_context.setdefault("code_inputs", []).append(code_input)
        
        # Generate or update code based on the input
        response = await self.process([
            f"Update the step implementation based on this input: {code_input}"
        ])
        
        if isinstance(response, str):
            self.generated_code = response
            print("Code updated successfully")
    
    def _display_help(self) -> None:
        """Display help information."""
        help_text = """
Available commands:
1. link <source_step> <target_step> - Link this step to another step
2. finish - End step creation and save
3. help - Show this menu

You can also input code or instructions directly to modify the step implementation.
"""
        print(help_text)
    
    async def _generate_link_code(self, source_step: str, target_step: str) -> str:
        """
        Generate code for linking two steps.
        
        Args:
            source_step: Name of the source step
            target_step: Name of the target step
            
        Returns:
            Generated code for the link
        """
        # Use the Agent's language model to generate appropriate link code
        response = await self.process([
            f"Generate code to link step {source_step} to {target_step}"
        ])
        
        return response if isinstance(response, str) else ""
    
    def get_generated_code(self) -> str:
        """Get the generated step implementation code."""
        return self.generated_code
    
    def get_generated_config(self) -> str:
        """Get the generated configuration YAML."""
        return self.generated_config
    
    def get_generated_tests(self) -> str:
        """Get the generated test code."""
        return self.generated_tests
    
    async def create_agent(self, name: str, model: str = "gpt-4", **kwargs) -> Agent:
        """
        Create a new agent with the specified parameters.
        
        Args:
            name: The name of the agent
            model: The model to use for the agent
            **kwargs: Additional parameters for the agent
            
        Returns:
            The created agent
        """
        agent = Agent(
            executor=self.executor,
            model_name=model,
            parameters={
                "temperature": 0.2
            },
            memories={
                "workflow_context": {}
            },
            **kwargs
        )
        self.agents.append(agent)
        self.current_agent = agent
        return agent
    
    async def create_workflow(self, name: str, **kwargs) -> Workflow:
        """
        Create a new workflow with the specified parameters.
        
        Args:
            name: The name of the workflow
            **kwargs: Additional parameters for the workflow
            
        Returns:
            The created workflow
        """
        #if not os.path.exists(f"workflows/{name}"):
        breakpoint()
        workflow = Workflow(name=name, **kwargs)
        self.workflows.append(workflow)
        self.current_workflow = workflow
        return workflow
    
    async def add_agent_to_workflow(self, agent: Agent, workflow: Optional[Workflow] = None) -> None:
        """
        Add an agent to a workflow.
        
        Args:
            agent: The agent to add
            workflow: The workflow to add the agent to. If None, uses the current workflow.
        """
        if workflow is None:
            if self.current_workflow is None:
                raise ValueError("No current workflow. Create a workflow first or specify one.")
            workflow = self.current_workflow
        
        # Add the agent as a step in the workflow
        workflow.add_step(agent)
    
    
    async def build_agent_workflow(self, name: str, agent_name: str, 
                                  model: str = "gpt-4", steps: List[Step] = None) -> Workflow:
        """
        Build a complete agent workflow with the specified parameters.
        
        Args:
            name: The name of the workflow
            agent_name: The name of the agent
            model: The model to use for the agent
            steps: Optional list of steps to add to the workflow
            
        Returns:
            The created workflow
        """
        # Create the workflow
        workflow = await self.create_workflow(name)
        
        # Add and connect steps if provided
        if steps:
            for step in steps:
                workflow.add_step(step)
        
        return workflow 