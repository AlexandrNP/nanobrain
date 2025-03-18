"""
NanoBrain Framework

A biologically-inspired framework for building adaptive, resilient systems.
"""

import os
import sys

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
