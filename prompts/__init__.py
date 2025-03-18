"""
Prompts package for NanoBrain.

This package contains templates and utility functions for working with LLM prompts.
"""

import sys
import os

# Add the current directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import modules to make them available via the prompts package
try:
    # Import from relative paths
    from .templates import (
        create_chat_template, 
        BASE_ASSISTANT,
        TECHNICAL_EXPERT, 
        CREATIVE_ASSISTANT,
        CODE_WRITER_TEMPLATE
    )
    
    from .tool_calling_prompt import (
        create_tool_calling_prompt,
        parse_tool_call
    )
except ImportError as e:
    print(f"Warning: Error importing prompts modules: {e}")

# Define what should be imported when using "from prompts import *"
__all__ = [
    'create_chat_template',
    'BASE_ASSISTANT',
    'TECHNICAL_EXPERT',
    'CREATIVE_ASSISTANT',
    'CODE_WRITER_TEMPLATE',
    'create_tool_calling_prompt',
    'parse_tool_call'
]
