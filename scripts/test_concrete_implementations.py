#!/usr/bin/env python3
"""
Concrete Test Implementations
Simple concrete implementations of abstract base classes for testing purposes.
"""

import os
import sys
from typing import Dict, Any

# Add nanobrain to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from nanobrain.core.step import BaseStep
from nanobrain.core.tool import ToolBase


class TestTool(ToolBase):
    """Concrete tool implementation for testing."""
    
    async def execute(self, **parameters) -> Any:
        """Simple test implementation of execute method."""
        return {'result': 'test_executed', 'parameters': parameters}


class TestStep(BaseStep):
    """Concrete step implementation for testing."""
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simple test implementation of process method."""
        return {'processed': True, 'input': input_data}


# Make these available for import
__all__ = ['TestTool', 'TestStep'] 