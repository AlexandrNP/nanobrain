#!/usr/bin/env python3
"""
StepFileWriter - Tool for writing files during step creation.

This tool is used by the AgentWorkflowBuilder and CreateStep classes to create
and update files during step and workflow creation.

Biological analogy: Motor neurons controlling fine motor movements.
Justification: Like how motor neurons precisely control muscle actions for
writing, this tool precisely writes code to files.
"""

import os
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional

class StepFileWriter:
    """
    Tool for writing files during step creation.
    
    This tool is designed to:
    1. Create new files with specified content
    2. Update existing files with new content
    3. Create directories if they don't exist
    
    Biological analogy: Motor neurons controlling fine motor movements.
    Justification: Like how motor neurons precisely control muscle actions for
    writing, this tool precisely writes code to files.
    """
    
    def __init__(self, executor=None, **kwargs):
        """
        Initialize the StepFileWriter tool.
        
        Args:
            executor: Executor instance (not used by this tool)
            **kwargs: Additional arguments
        """
        self._debug_mode = kwargs.get('_debug_mode', False)
    
    async def create_file(self, path: str, content: str) -> bool:
        """
        Create a file with the specified content.
        
        Args:
            path: Path to the file to create
            content: Content to write to the file
            
        Returns:
            True if the file was created successfully, False otherwise
        """
        try:
            # Ensure the directory exists
            directory = os.path.dirname(path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                if self._debug_mode:
                    print(f"Created directory: {directory}")
            
            # Write the content to the file
            with open(path, 'w') as f:
                f.write(content)
                
            if self._debug_mode:
                print(f"Created file: {path}")
                
            return True
        except Exception as e:
            if self._debug_mode:
                print(f"Error creating file {path}: {e}")
            return False
    
    async def update_file(self, path: str, content: str) -> bool:
        """
        Update an existing file with new content.
        
        Args:
            path: Path to the file to update
            content: New content for the file
            
        Returns:
            True if the file was updated successfully, False otherwise
        """
        return await self.create_file(path, content)
    
    async def read_file(self, path: str) -> Optional[str]:
        """
        Read the content of a file.
        
        Args:
            path: Path to the file to read
            
        Returns:
            Content of the file or None if the file doesn't exist
        """
        try:
            if not os.path.exists(path):
                return None
                
            with open(path, 'r') as f:
                content = f.read()
                
            return content
        except Exception as e:
            if self._debug_mode:
                print(f"Error reading file {path}: {e}")
            return None 