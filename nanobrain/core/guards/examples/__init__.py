"""
NanoBrain ContextVar Guard Integration Examples

Demonstrates how to integrate ContextVar guards with existing Nanobrain components
using Option A: Pure Mixin Integration without modifying base classes.

✅ FRAMEWORK COMPLIANCE: Zero modifications to existing base classes
✅ PURE MIXIN: Add guard functionality through composition
✅ OPTION A: Recommended integration approach from implementation plan
"""

from .protected_step_example import ProtectedStepExample
from .protected_workflow_example import ProtectedWorkflowExample
from .protected_agent_example import ProtectedAgentExample

__all__ = [
    'ProtectedStepExample',
    'ProtectedWorkflowExample', 
    'ProtectedAgentExample'
] 