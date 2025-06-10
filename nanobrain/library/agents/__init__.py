"""
NanoBrain Library - Agents

Collection of specialized and conversational agents built on the NanoBrain framework.

Available agent categories:
- specialized: Task-specific agents (code_writer, file_writer, etc.)
- conversational: Enhanced conversational agents with protocol support
- enhanced: Advanced agents with collaboration capabilities

Agent Types:
- Simple agents: Process input without conversation history
- Conversational agents: Maintain conversation context and history
- Enhanced agents: Include protocol support and collaboration features
"""

# Import specialized agents
from .specialized import *

# Import conversational agents
from .conversational import *

# Import enhanced agents (from enhanced directory)
try:
    from .enhanced import *
except ImportError:
    # Enhanced agents may not be fully implemented yet
    pass

__all__ = [
    # Specialized agent base classes
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
    
    # Conversational agents
    'EnhancedCollaborativeAgent',
    
    # Enhanced agents (if available)
    # 'CollaborativeAgent',  # Will be available when enhanced module is complete
] 