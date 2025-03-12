"""
Mock builder module for testing.
"""

import os
import sys
from typing import List, Dict, Any, Optional
from unittest.mock import MagicMock

from test.mock_agent import Agent
from test.mock_executor import MockExecutorBase
from test.mock_tools import (
    StepFileWriter,
    StepPlanner,
    StepCoder,
    StepGitInit,
    StepContextSearch,
    StepWebSearch
)


class CreateWorkflow:
    """Mock CreateWorkflow for testing."""
    
    @staticmethod
    async def execute(builder, workflow_name: str, base_dir: Optional[str] = None) -> Dict[str, Any]:
        """Mock execute method."""
        return {
            "success": True,
            "message": f"Created workflow {workflow_name}",
            "workflow_path": f"/mock/path/{workflow_name}",
            "workflow": MagicMock()
        }


class CreateStep:
    """Mock CreateStep for testing."""
    @staticmethod
    async def execute(builder, step_name: str, base_class: str = "Step", description: str = None) -> Dict[str, Any]:
        """Mock execute method."""
        step_class_name = f"Step{step_name.capitalize()}"
        return {
            "success": True,
            "message": f"Created step {step_class_name}",
            "step_path": os.path.join(os.getcwd(), "test_workflow", "src", step_class_name),
            "step_class_name": step_class_name
        }


class TestStepStep:
    """Mock TestStepStep for testing."""
    @staticmethod
    async def execute(builder, step_name: str) -> Dict[str, Any]:
        """Mock execute method."""
        step_class_name = f"Step{step_name.capitalize()}"
        return {
            "success": True,
            "message": f"Tests for {step_class_name} passed",
            "output": ""
        }


class SaveStepStep:
    """Mock SaveStepStep for testing."""
    @staticmethod
    async def execute(builder, step_name: str) -> Dict[str, Any]:
        """Mock execute method."""
        step_class_name = f"Step{step_name.capitalize()}"
        return {
            "success": True,
            "message": f"Saved step {step_class_name}"
        }


class LinkStepsStep:
    """Mock LinkStepsStep for testing."""
    @staticmethod
    async def execute(builder, source_step: str, target_step: str, link_type: str = "LinkDirect") -> Dict[str, Any]:
        """Mock execute method."""
        source_class_name = f"Step{source_step.capitalize()}"
        target_class_name = f"Step{target_step.capitalize()}"
        return {
            "success": True,
            "message": f"Created link from {source_class_name} to {target_class_name}",
            "link_file": os.path.join(os.getcwd(), "test_workflow", "src", f"{source_class_name}To{target_class_name}Link.py")
        }


class SaveWorkflowStep:
    """Mock SaveWorkflowStep for testing."""
    @staticmethod
    async def execute(builder) -> Dict[str, Any]:
        """Mock execute method."""
        return {
            "success": True,
            "message": "Saved workflow at test_workflow"
        }


class NanoBrainBuilder:
    """Mock NanoBrainBuilder for testing."""
    
    def __init__(self, executor=None, **kwargs):
        """Initialize the mock NanoBrainBuilder."""
        self.executor = executor or MockExecutorBase()
        self._workflow_stack = []
        
        # Create an agent
        self.agent = Agent(
            llm_class="OpenAI",
            parameters={
                "model": "gpt-4",
                "temperature": 0.2
            },
            memories={
                "workflow_context": {}
            }
        )
        
        # Add tools to the agent
        self.agent.tools = [
            StepFileWriter(self.executor),
            StepPlanner(self.executor),
            StepCoder(self.executor),
            StepGitInit(self.executor),
            StepContextSearch(self.executor),
            StepWebSearch(self.executor)
        ]
    
    def get_current_workflow(self) -> Optional[str]:
        """Get the current workflow path."""
        if not self._workflow_stack:
            return None
        
        return self._workflow_stack[-1]
    
    def push_workflow(self, workflow_path: str):
        """Push a workflow onto the stack."""
        self._workflow_stack.append(workflow_path)
    
    def pop_workflow(self) -> Optional[str]:
        """Pop a workflow from the stack."""
        if not self._workflow_stack:
            return None
        
        return self._workflow_stack.pop()
    
    async def create_workflow(self, workflow_name: str, base_dir: Optional[str] = None) -> Dict[str, Any]:
        """Mock create_workflow method."""
        return await CreateWorkflow.execute(self, workflow_name, base_dir=base_dir)
    
    async def create_step(self, step_name: str, base_class: str = "Step", description: str = None) -> Dict[str, Any]:
        """Create a new step."""
        return await CreateStep.execute(self, step_name, base_class, description)
    
    async def test_step(self, step_name: str) -> Dict[str, Any]:
        """Test a step."""
        return await TestStepStep.execute(self, step_name)
    
    async def save_step(self, step_name: str) -> Dict[str, Any]:
        """Save a step."""
        return await SaveStepStep.execute(self, step_name)
    
    async def link_steps(self, source_step: str, target_step: str, link_type: str = "LinkDirect") -> Dict[str, Any]:
        """Link steps together."""
        return await LinkStepsStep.execute(self, source_step, target_step, link_type)
    
    async def save_workflow(self) -> Dict[str, Any]:
        """Save a workflow."""
        return await SaveWorkflowStep.execute(self)
    
    async def process_command(self, command: str, args: List[str]) -> Dict[str, Any]:
        """Process a command."""
        if command == "create-workflow":
            workflow_name = args[0]
            username = args[1] if len(args) > 1 else None
            return await self.create_workflow(workflow_name, username)
        
        elif command == "create-step":
            step_name = args[0]
            base_class = args[1] if len(args) > 1 else "Step"
            description = args[2] if len(args) > 2 else None
            return await self.create_step(step_name, base_class, description)
        
        elif command == "test-step":
            step_name = args[0]
            return await self.test_step(step_name)
        
        elif command == "save-step":
            step_name = args[0]
            return await self.save_step(step_name)
        
        elif command == "link-steps":
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