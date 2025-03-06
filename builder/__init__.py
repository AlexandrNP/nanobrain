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
except ImportError as e:
    print(f"Warning: Could not import NanoBrainBuilder: {e}")

__all__ = ["NanoBrainBuilder"] 