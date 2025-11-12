"""
Distributed execution components for NanoBrain framework.

This module provides Parsl-based distributed execution capabilities
while maintaining compatibility with existing NanoBrain interfaces.
"""

from .workflow_execution import execute_workflow_distributed

__all__ = ['execute_workflow_distributed']
