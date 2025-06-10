"""
NanoBrain Library - Specialized Agents

Task-specific agents for common operations like code generation and file management.

Available base classes:
- SpecializedAgentBase: Base mixin for specialized functionality
- SimpleSpecializedAgent: Simple specialized agent base
- ConversationalSpecializedAgent: Conversational specialized agent base

Available agents:
- CodeWriterAgent: Simple code generation agent
- ConversationalCodeWriterAgent: Conversational code generation agent
- FileWriterAgent: Simple file operations agent
- ConversationalFileWriterAgent: Conversational file operations agent

Factory functions:
- create_specialized_agent: Create specialized agents of different types
"""

from .base import (
    SpecializedAgentBase,
    SimpleSpecializedAgent,
    ConversationalSpecializedAgent,
    create_specialized_agent
)
from .code_writer import (
    CodeWriterAgentMixin,
    CodeWriterAgent,
    ConversationalCodeWriterAgent
)
from .file_writer import (
    FileWriterAgentMixin,
    FileWriterAgent,
    ConversationalFileWriterAgent
)
from .parsl_agent import ParslAgent

__all__ = [
    # Base classes
    'SpecializedAgentBase',
    'SimpleSpecializedAgent',
    'ConversationalSpecializedAgent',
    'create_specialized_agent',
    
    # Code writer agents
    'CodeWriterAgentMixin',
    'CodeWriterAgent',
    'ConversationalCodeWriterAgent',
    
    # File writer agents
    'FileWriterAgentMixin',
    'FileWriterAgent',
    'ConversationalFileWriterAgent',
    
    'ParslAgent',
] 