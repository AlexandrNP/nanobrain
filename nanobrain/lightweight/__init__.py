"""
NanoBrain Lightweight Wrapper Module
====================================

Provides lightweight, user-friendly interfaces for rapid prototyping
while maintaining the framework's configuration-driven backbone.

This module implements auto-discovery of framework components and
provides simplified interfaces that generate proper configurations
under the hood.
"""

from .discovery import ConfigDrivenDiscovery
from .workflow_builder import WorkflowBuilder

__all__ = [
    'ConfigDrivenDiscovery',
    'WorkflowBuilder',
]

# Global discovery instance (lazy-loaded)
_discovery = None

def get_discovery():
    """Get the global discovery instance."""
    global _discovery
    if _discovery is None:
        _discovery = ConfigDrivenDiscovery()
        _discovery.discover_components()
    return _discovery

def list_agents():
    """List all available agent classes."""
    return get_discovery().get_classes_by_category("agent")

def list_steps():
    """List all available step classes."""
    return get_discovery().get_classes_by_category("step")

def list_executors():
    """List all available executor classes."""
    return get_discovery().get_classes_by_category("executor")

def workflow(name, description=""):
    """Create a new workflow builder."""
    return WorkflowBuilder(name, description)
