"""
NanoBrain Framework

A biologically-inspired framework for building adaptive, resilient systems.
"""

import os
import sys

# Add the current directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import key components for easier access
try:
    from .ConfigManager import ConfigManager
    from .DirectoryTracer import DirectoryTracer
    from .ExecutorBase import ExecutorBase
    from .Step import Step
    from .Workflow import Workflow
    from .Agent import Agent
except ImportError as e:
    print(f"Warning: Could not import a component: {e}")
