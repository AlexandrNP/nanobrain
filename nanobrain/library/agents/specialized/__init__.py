"""
Nanobrain Library - Specialized Agents

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
- ParslAgent: Parsl workflow execution agent
- ProteinSynonymAgent: Protein synonym identification agent with ICTV standards
- VirusExtractionAgent: Virus species extraction agent for query classification
- QueryAnalysisAgent: Query analysis agent for biological context extraction

Factory functions:
- create_specialized_agent: Factory for creating specialized agents
"""

from .base import (
    SpecializedAgentBase, SimpleSpecializedAgent, ConversationalSpecializedAgent,
    create_specialized_agent
)
from .code_writer import CodeWriterAgent, ConversationalCodeWriterAgent
from .file_writer import FileWriterAgent, ConversationalFileWriterAgent
from .parsl_agent import ParslAgent
from .protein_synonym_agent import ProteinSynonymAgent
from .virus_extraction_agent import VirusExtractionAgent
from .query_analysis_agent import QueryAnalysisAgent

__all__ = [
    # Base classes
    'SpecializedAgentBase',
    'SimpleSpecializedAgent', 
    'ConversationalSpecializedAgent',
    
    # Concrete agents
    'CodeWriterAgent',
    'ConversationalCodeWriterAgent',
    'FileWriterAgent',
    'ConversationalFileWriterAgent', 
    'ParslAgent',
    'ProteinSynonymAgent',
    'VirusExtractionAgent',
    'QueryAnalysisAgent',
    
    # Factory functions
    'create_specialized_agent'
] 