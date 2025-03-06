from typing import List, Any, Optional, Dict, Union
import os
import asyncio
from pathlib import Path

from src.Step import Step
from src.ExecutorBase import ExecutorBase
from src.Agent import Agent


class StepPlanner(Step):
    """
    Tool for planning the implementation of a new step or workflow.
    
    Biological analogy: Prefrontal planning networks.
    Justification: Like how prefrontal networks plan complex actions
    by breaking them down into steps, this tool plans implementations
    by breaking them down into manageable components.
    """
    def __init__(self, executor: ExecutorBase, **kwargs):
        super().__init__(executor, **kwargs)
        
        # Tool-specific attributes
        self.model_name = kwargs.get('model_name', "gpt-4")  # Prefer GPT-4 for planning
        self.model_class = kwargs.get('model_class', None)
        self.temperature = kwargs.get('temperature', 0.3)
        self.detail_level = kwargs.get('detail_level', 'high')  # 'low', 'medium', 'high'
        
        # Create the planning agent
        self.planning_agent = Agent(
            executor=executor,
            model_name=self.model_name,
            model_class=self.model_class,
            prompt_template="""You are an expert system architect and planner. Your task is to create a detailed plan for implementing the requested feature or component. 

Analyze the requirements carefully and create a comprehensive plan that includes:

1. A high-level overview of the implementation
2. Required components and their relationships
3. Specific implementation steps in order
4. Potential challenges and mitigation strategies
5. Testing approach

Requirements: {input}

Context: {context}""",
            **kwargs
        )
    
    async def process(self, inputs: List[Any]) -> Any:
        """
        Generate a plan for implementing a step or workflow.
        
        Args:
            inputs: List containing:
                - requirements: Description of what needs to be implemented
                - component_type: Type of component to implement (step or workflow)
                - context: Additional context or constraints (optional)
        
        Returns:
            Dictionary with the generated plan
        """
        # Extract inputs
        if not inputs or len(inputs) < 2:
            return {
                "success": False,
                "error": "Missing required inputs: requirements and component_type are required"
            }
        
        requirements = inputs[0]
        component_type = inputs[1]
        context = inputs[2] if len(inputs) > 2 else ""
        
        # Validate component type
        if component_type not in ['step', 'workflow']:
            return {
                "success": False,
                "error": f"Invalid component type: {component_type}. Must be 'step' or 'workflow'"
            }
        
        # Format the requirements with component type and detail level
        formatted_requirements = f"Create a {self.detail_level}-detail implementation plan for a new {component_type} that meets the following requirements:\n\n{requirements}"
        
        try:
            # Generate plan using the planning agent
            plan = await self.planning_agent.process([formatted_requirements])
            
            # Return the plan
            return {
                "success": True,
                "plan": plan,
                "component_type": component_type,
                "detail_level": self.detail_level
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to generate plan: {e}"
            }
    
    async def plan_step(self, step_name: str, description: str, base_class: str = "Step", context: str = "") -> Dict[str, Any]:
        """
        Generate a plan for implementing a new step.
        
        Args:
            step_name: Name of the step to implement
            description: Description of the step's functionality
            base_class: Base class to inherit from (default: Step)
            context: Additional context or constraints
        
        Returns:
            Dictionary with the generated plan
        """
        # Format the requirements for a step
        requirements = f"""
Step Name: {step_name}
Base Class: {base_class}

Description:
{description}
"""
        
        # Generate the plan
        return await self.process([requirements, "step", context])
    
    async def plan_workflow(self, workflow_name: str, description: str, steps: List[str], context: str = "") -> Dict[str, Any]:
        """
        Generate a plan for implementing a new workflow.
        
        Args:
            workflow_name: Name of the workflow to implement
            description: Description of the workflow's functionality
            steps: List of steps that will be part of the workflow
            context: Additional context or constraints
        
        Returns:
            Dictionary with the generated plan
        """
        # Format the requirements for a workflow
        requirements = f"""
Workflow Name: {workflow_name}

Description:
{description}

Steps:
"""
        
        # Add steps
        for step in steps:
            requirements += f"- {step}\n"
        
        # Generate the plan
        return await self.process([requirements, "workflow", context])
    
    async def analyze_requirements(self, requirements: str) -> Dict[str, Any]:
        """
        Analyze requirements and extract key components and relationships.
        
        Args:
            requirements: Raw requirements text
        
        Returns:
            Dictionary with analysis results
        """
        # Format the requirements for analysis
        formatted_requirements = f"""
Analyze these requirements and extract:
1. The main components that need to be implemented
2. The relationships between these components
3. Any potential ambiguities or missing information that should be clarified
4. Suggested implementation approach

Requirements:
{requirements}
"""
        
        try:
            # Generate analysis using the planning agent
            analysis = await self.planning_agent.process([formatted_requirements])
            
            # Return the analysis
            return {
                "success": True,
                "analysis": analysis
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to analyze requirements: {e}"
            } 