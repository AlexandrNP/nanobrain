"""
NanoBrain Library

A collection of reusable components, workflows, and infrastructure elements
built on top of the NanoBrain framework core.

This library provides:
- Specialized agents for common tasks
- Infrastructure components for advanced workflows
- Complete workflow implementations with proper step interconnections
- Templates and examples for building custom components

Organization:
- agents/: Specialized agent implementations
- infrastructure/: Custom data units, triggers, links, and steps
- workflows/: Complete workflow implementations with step hierarchies
"""

from .agents import *
from .infrastructure import *
from .workflows import *

__version__ = "1.0.0"
__author__ = "NanoBrain Framework"
__description__ = "Reusable components and workflows for the NanoBrain framework"

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