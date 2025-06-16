"""
NanoBrain Library

A comprehensive library of reusable components for the NanoBrain framework.

This library provides:
- Agents and agent configurations
- Workflows and workflow orchestration
- Infrastructure components (data units, steps, interfaces)
- Pre-built integrations and utilities

The library follows modular design principles and provides both
high-level orchestration tools and low-level building blocks.
"""

# Agent components
from .agents import (
    EnhancedCollaborativeAgent,
)

# Workflow components  
from .workflows import (
    ChatWorkflow
)

# Infrastructure components
from .infrastructure import (
    # Data components
    ConversationHistoryUnit,
    DataUnitMemory,
    
    # Step components
    ParallelConversationalAgentStep,
    ParallelAgentStep,
)

# Try to import web interface components
try:
    from .interfaces.web import (
        WebInterface,
        WebInterfaceConfig,
        ChatRequest,
        ChatOptions,
        ChatResponse,
        ChatMetadata
    )
    WEB_INTERFACE_AVAILABLE = True
except ImportError:
    WEB_INTERFACE_AVAILABLE = False
    # Define placeholder exports
    WebInterface = None
    WebInterfaceConfig = None
    ChatRequest = None
    ChatOptions = None
    ChatResponse = None
    ChatMetadata = None

__all__ = [
    # Agents
    'EnhancedCollaborativeAgent',
    
    # Workflows
    'ChatWorkflow',
    
    # Infrastructure - Data
    'ConversationHistoryUnit',
    'DataUnitMemory',
    
    # Infrastructure - Steps
    'ParallelConversationalAgentStep',
    'ParallelAgentStep',
    
    # Web Interface (if available)
    'WebInterface',
    'WebInterfaceConfig',
    'ChatRequest',
    'ChatOptions',
    'ChatResponse',
    'ChatMetadata',
    
    # Utility
    'WEB_INTERFACE_AVAILABLE'
]

# Version information
__version__ = "2.0.0"
__author__ = "NanoBrain Team"
__description__ = "Comprehensive library for NanoBrain framework components"

# Library metadata
LIBRARY_INFO = {
    "version": __version__,
    "components": {
        "agents": ["conversational", "specialized"],
        "infrastructure": ["data_units", "triggers", "links", "steps"],
        "workflows": ["chat_workflow", "parsl_chat_workflow"]
    },
    "description": __description__
}

def get_library_info():
    """Get information about the library components."""
    return LIBRARY_INFO

def list_available_components():
    """List all available components in the library."""
    components = []
    for category, items in LIBRARY_INFO["components"].items():
        for item in items:
            components.append(f"{category}.{item}")
    return components 