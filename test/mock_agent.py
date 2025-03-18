"""
Mock Agent class for testing purposes.
"""

from unittest.mock import MagicMock
from typing import List, Dict, Any, Optional


class Agent:
    """Mock Agent class for testing."""
    
    def __init__(self, config=None, llm=None, debug_mode=False, llm_system_prompt=None, custom_tool_prompt=None, llm_class=None, parameters=None, memories=None):
        """Initialize the mock agent with support for both old and new initialization formats."""
        # New format
        self.config = config or {}
        self.llm = llm
        self.llm_system_prompt = llm_system_prompt
        self._messages = []
        self.tools = []
        self.debug_mode = debug_mode
        self.langchain_memory = None
        self.current_input = None
        self.custom_tool_prompt = custom_tool_prompt
        
        # Old format
        self.llm_class = llm_class
        self.parameters = parameters or {}
        self.memories = memories or {}
        self.workflow_context = {}
    
    def add_tool(self, tool):
        """Add a tool to the agent's tools."""
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
    
    def _load_prompt_template(self, prompt_name):
        """Mock implementation of _load_prompt_template."""
        return "This is a mock prompt template for testing."
    
    def _initialize_llm(self):
        """Mock implementation of _initialize_llm."""
        return None
    
    def get_context_history(self):
        """Get the agent's context history."""
        return {}
    
    def clear_memories(self):
        """Clear the agent's memories."""
        self.memories = {} 