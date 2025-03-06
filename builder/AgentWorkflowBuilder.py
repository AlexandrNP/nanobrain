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


class AgentWorkflowBuilder:
    """
    Builder for creating Agent-based workflows.
    
    Biological analogy: Neural circuit formation during development.
    Justification: Like how neural circuits are formed through a guided process
    during development, this builder helps construct agent-based workflows with
    appropriate connections and configurations.
    """
    
    def __init__(self):
        """Initialize the AgentWorkflowBuilder."""
        self.agents: List[Agent] = []
        self.workflows: List[Workflow] = []
        self.current_agent: Optional[Agent] = None
        self.current_workflow: Optional[Workflow] = None
        self.executor = ExecutorBase()
    
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
    
    async def connect_agent_to_step(self, agent: Agent, step: Step, 
                                   workflow: Optional[Workflow] = None) -> None:
        """
        Connect an agent to a step in a workflow.
        
        Args:
            agent: The agent to connect
            step: The step to connect to
            workflow: The workflow containing the step. If None, uses the current workflow.
        """
        if workflow is None:
            if self.current_workflow is None:
                raise ValueError("No current workflow. Create a workflow first or specify one.")
            workflow = self.current_workflow
        
        # Ensure both the agent and step are in the workflow
        if agent not in workflow.steps:
            workflow.add_step(agent)
        if step not in workflow.steps:
            workflow.add_step(step)
        
        # Connect the agent to the step
        workflow.connect(agent, step)
    
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
        
        # Create the agent
        agent = await self.create_agent(agent_name, model)
        
        # Add the agent to the workflow
        await self.add_agent_to_workflow(agent, workflow)
        
        # Add and connect steps if provided
        if steps:
            for step in steps:
                workflow.add_step(step)
                await self.connect_agent_to_step(agent, step, workflow)
        
        return workflow 