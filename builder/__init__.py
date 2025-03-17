"""
NanoBrain Builder

A package for building NanoBrain workflows.
"""

import os
import sys

# Add the parent directory to the path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from .NanoBrainBuilder import NanoBrainBuilder
    from .AgentWorkflowBuilder import AgentWorkflowBuilder
    from .AgentCodeWriter import AgentCodeWriter
    from .prompts import *
except ImportError as e:
    print(f"Warning: Could not import one or more modules: {e}")

__all__ = [
    "NanoBrainBuilder",
    "AgentWorkflowBuilder",
    "AgentCodeWriter",
    # Prompt constants
    "WORKFLOW_BUILDER_PROMPT",
    "WORKFLOW_BUILDER_FRAMEWORK_CONTEXT",
    "CODE_WRITER_PROMPT",
    "CODE_WRITER_STEP_CONTEXT",
    "CODE_WRITER_WORKFLOW_CONTEXT",
    "CODE_WRITER_LINK_CONTEXT",
    "CODE_WRITER_DATA_UNIT_CONTEXT",
    "CODE_WRITER_TRIGGER_CONTEXT"
] 