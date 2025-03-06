"""
Mock Agent class for testing purposes.
"""

from unittest.mock import MagicMock
from typing import List, Dict, Any, Optional


class Agent:
    """Mock Agent class for testing."""
    
    def __init__(self, llm_class=None, parameters=None, memories=None):
        """Initialize the mock Agent."""
        self.llm_class = llm_class
        self.parameters = parameters or {}
        self.memories = memories or {}
        self.tools = []
        self.workflow_context = {}
    
    def add_tool(self, tool):
        """Add a tool to the agent."""
        self.tools.append(tool)
    
    def remove_tool(self, tool_name):
        """Remove a tool from the agent."""
        self.tools = [tool for tool in self.tools if tool.__class__.__name__ != tool_name]
    
    def update_workflow_context(self, workflow_path):
        """Update the workflow context."""
        self.workflow_context = {"path": workflow_path}
    
    async def process(self, input_data):
        """Process input data."""
        return {"result": "Mock result"}
    
    def update_memories(self, memories):
        """Update the agent's memories."""
        self.memories.update(memories) 