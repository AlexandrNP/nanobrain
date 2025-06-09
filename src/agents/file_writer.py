"""
File Writer Agent for NanoBrain Framework

Specialized agent for file operations based on natural language descriptions.
"""

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional
from core.agent import SimpleAgent, AgentConfig

logger = logging.getLogger(__name__)


class FileWriterAgent(SimpleAgent):
    """
    Specialized agent for file operations.
    
    This agent can create, write, and manage files based on natural
    language descriptions. It intelligently detects file operations
    and performs them directly when possible.
    """
    
    def __init__(self, config: Optional[AgentConfig] = None, **kwargs):
        # Use provided config or create minimal default
        if config is None:
            config = AgentConfig(
                name="file_writer",
                description="Specialized agent for file operations",
                model="gpt-3.5-turbo",  # Lighter model since this is more operational
                system_prompt="""You are a specialized file writing agent for the NanoBrain framework.

CRITICAL: When asked to create, write, or save files, you MUST actually perform the file operation, not just describe it.

Your capabilities:
1. Create and write files with any content
2. Parse file paths and content from natural language requests
3. Handle directory creation automatically
4. Provide clear feedback about file operations

When you receive a request to create/write/save a file:
1. Extract the file path from the request
2. Extract or generate the content as requested
3. Actually create the file using your file writing capabilities
4. Confirm the operation was successful

Examples of requests you should handle:
- "Create a Python file at path/to/file.py with a hello function"
- "Save this code to utils.py: [code content]"
- "Write a config file with database settings"

ALWAYS actually perform the file operation - never just provide instructions or example content.""",
                temperature=0.1  # Very low temperature for consistent file operations
            )
        
        super().__init__(config, **kwargs)
        
    async def process(self, input_text: str, **kwargs) -> str:
        """
        Process file operation request.
        
        Args:
            input_text: Natural language description of file operation
            **kwargs: Additional parameters like file_path, content, etc.
            
        Returns:
            Result of the file operation
        """
        # Check if this is a direct file operation request
        if self._is_direct_file_operation(input_text, **kwargs):
            return await self._handle_direct_file_operation(input_text, **kwargs)
        
        # Check if this is a file operation that can be parsed from text
        file_op_result = await self._try_parse_and_execute_file_operation(input_text)
        if file_op_result:
            return file_op_result
        
        # Otherwise, use LLM for complex operations
        return await super().process(input_text, **kwargs)
    
    def _is_direct_file_operation(self, input_text: str, **kwargs) -> bool:
        """
        Check if this is a direct file operation that doesn't need LLM.
        
        Args:
            input_text: Input text
            **kwargs: Additional parameters
            
        Returns:
            True if this is a direct file operation
        """
        # Check for explicit file operation parameters
        has_file_path = 'file_path' in kwargs or 'path' in kwargs
        has_content = 'content' in kwargs or 'data' in kwargs
        
        # Check for file operation keywords in input
        file_keywords = ['write', 'save', 'create', 'file', 'path']
        has_keywords = any(keyword in input_text.lower() for keyword in file_keywords)
        
        return has_file_path and (has_content or has_keywords)
    
    async def _try_parse_and_execute_file_operation(self, input_text: str) -> Optional[str]:
        """
        Try to parse and execute file operations from natural language.
        
        Args:
            input_text: Input text to parse
            
        Returns:
            Result of file operation if successful, None otherwise
        """
        try:
            # Look for file creation patterns
            create_patterns = [
                r"create.*?(?:file|script).*?(?:at|to|in)\s+([^\s]+)",
                r"save.*?(?:to|at|in)\s+([^\s]+)",
                r"write.*?(?:to|at|in)\s+([^\s]+)",
                r"(?:file|script).*?(?:called|named)\s+([^\s]+)",
            ]
            
            file_path = None
            for pattern in create_patterns:
                match = re.search(pattern, input_text, re.IGNORECASE)
                if match:
                    file_path = match.group(1).strip('"\'')
                    break
            
            if not file_path:
                return None
            
            # Look for content patterns
            content = None
            
            # Pattern 1: "with content: [content]"
            content_match = re.search(r"with content:\s*(.+)", input_text, re.IGNORECASE | re.DOTALL)
            if content_match:
                content = content_match.group(1).strip()
            
            # Pattern 2: Code blocks
            code_match = re.search(r"```(?:python|py)?\s*\n(.*?)\n```", input_text, re.DOTALL)
            if code_match:
                content = code_match.group(1).strip()
            
            # Pattern 3: Function/class descriptions
            if not content:
                if "function" in input_text.lower():
                    # Extract function name if mentioned
                    func_match = re.search(r"function.*?(?:called|named)\s+([a-zA-Z_][a-zA-Z0-9_]*)", input_text, re.IGNORECASE)
                    if func_match:
                        func_name = func_match.group(1)
                        content = f"def {func_name}():\n    \"\"\"Generated function.\"\"\"\n    pass"
                elif "class" in input_text.lower():
                    # Extract class name if mentioned
                    class_match = re.search(r"class.*?(?:called|named)\s+([a-zA-Z_][a-zA-Z0-9_]*)", input_text, re.IGNORECASE)
                    if class_match:
                        class_name = class_match.group(1)
                        content = f"class {class_name}:\n    \"\"\"Generated class.\"\"\"\n    pass"
            
            # If we have a file path, create the file
            if file_path:
                if not content:
                    content = f"# Generated file from request: {input_text[:100]}...\n"
                
                result = await self.write_file(file_path, content)
                return f"Successfully processed file operation: {result}"
            
            return None
            
        except Exception as e:
            logger.error(f"Error parsing file operation: {e}")
            return None
    
    async def _handle_direct_file_operation(self, input_text: str, **kwargs) -> str:
        """
        Handle direct file operations without LLM.
        
        Args:
            input_text: Input text
            **kwargs: Additional parameters
            
        Returns:
            Result of the file operation
        """
        try:
            # Extract file path
            file_path = kwargs.get('file_path') or kwargs.get('path')
            if not file_path:
                return "Error: No file path provided"
            
            # Extract content
            content = kwargs.get('content') or kwargs.get('data', '')
            
            # If content is not provided, try to extract from input
            if not content and 'content:' in input_text.lower():
                content = input_text.split('content:', 1)[1].strip()
            
            # Create file
            result = await self.write_file(file_path, content)
            
            logger.info(f"FileWriterAgent handled direct operation: {file_path}")
            return result
            
        except Exception as e:
            error_msg = f"Error in direct file operation: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    async def write_file(self, file_path: str, content: str, 
                        encoding: str = 'utf-8', create_dirs: bool = True) -> str:
        """
        Write content to a file.
        
        Args:
            file_path: Path to the file
            content: Content to write
            encoding: File encoding
            create_dirs: Whether to create directories if they don't exist
            
        Returns:
            Success message or error description
        """
        try:
            path = Path(file_path)
            
            # Create directories if needed
            if create_dirs and path.parent != path:
                path.parent.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Created directories for {file_path}")
            
            # Write file
            path.write_text(content, encoding=encoding)
            
            file_size = path.stat().st_size
            success_msg = f"Successfully wrote {file_size} bytes to {file_path}"
            logger.info(success_msg)
            return success_msg
            
        except PermissionError:
            error_msg = f"Permission denied: Cannot write to {file_path}"
            logger.error(error_msg)
            return error_msg
        except OSError as e:
            error_msg = f"OS error writing to {file_path}: {str(e)}"
            logger.error(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"Unexpected error writing to {file_path}: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    async def append_to_file(self, file_path: str, content: str, 
                           encoding: str = 'utf-8') -> str:
        """
        Append content to a file.
        
        Args:
            file_path: Path to the file
            content: Content to append
            encoding: File encoding
            
        Returns:
            Success message or error description
        """
        try:
            path = Path(file_path)
            
            # Append to file
            with open(path, 'a', encoding=encoding) as f:
                f.write(content)
            
            file_size = path.stat().st_size
            success_msg = f"Successfully appended content to {file_path} (total size: {file_size} bytes)"
            logger.info(success_msg)
            return success_msg
            
        except Exception as e:
            error_msg = f"Error appending to {file_path}: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    async def read_file(self, file_path: str, encoding: str = 'utf-8') -> str:
        """
        Read content from a file.
        
        Args:
            file_path: Path to the file
            encoding: File encoding
            
        Returns:
            File content or error description
        """
        try:
            path = Path(file_path)
            
            if not path.exists():
                return f"Error: File {file_path} does not exist"
            
            content = path.read_text(encoding=encoding)
            logger.info(f"Successfully read {len(content)} characters from {file_path}")
            return content
            
        except Exception as e:
            error_msg = f"Error reading {file_path}: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    async def delete_file(self, file_path: str) -> str:
        """
        Delete a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Success message or error description
        """
        try:
            path = Path(file_path)
            
            if not path.exists():
                return f"File {file_path} does not exist"
            
            path.unlink()
            success_msg = f"Successfully deleted {file_path}"
            logger.info(success_msg)
            return success_msg
            
        except Exception as e:
            error_msg = f"Error deleting {file_path}: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    async def create_directory(self, dir_path: str) -> str:
        """
        Create a directory.
        
        Args:
            dir_path: Path to the directory
            
        Returns:
            Success message or error description
        """
        try:
            path = Path(dir_path)
            path.mkdir(parents=True, exist_ok=True)
            
            success_msg = f"Successfully created directory {dir_path}"
            logger.info(success_msg)
            return success_msg
            
        except Exception as e:
            error_msg = f"Error creating directory {dir_path}: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    async def list_directory(self, dir_path: str) -> str:
        """
        List contents of a directory.
        
        Args:
            dir_path: Path to the directory
            
        Returns:
            Directory listing or error description
        """
        try:
            path = Path(dir_path)
            
            if not path.exists():
                return f"Directory {dir_path} does not exist"
            
            if not path.is_dir():
                return f"{dir_path} is not a directory"
            
            items = []
            for item in path.iterdir():
                item_type = "DIR" if item.is_dir() else "FILE"
                size = item.stat().st_size if item.is_file() else "-"
                items.append(f"{item_type:4} {size:>10} {item.name}")
            
            if not items:
                return f"Directory {dir_path} is empty"
            
            listing = f"Contents of {dir_path}:\n" + "\n".join(items)
            logger.info(f"Listed {len(items)} items in {dir_path}")
            return listing
            
        except Exception as e:
            error_msg = f"Error listing directory {dir_path}: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    async def file_exists(self, file_path: str) -> bool:
        """
        Check if a file exists.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if file exists, False otherwise
        """
        return Path(file_path).exists()
    
    async def get_file_info(self, file_path: str) -> str:
        """
        Get information about a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File information or error description
        """
        try:
            path = Path(file_path)
            
            if not path.exists():
                return f"File {file_path} does not exist"
            
            stat = path.stat()
            info = f"""File information for {file_path}:
- Size: {stat.st_size} bytes
- Type: {'Directory' if path.is_dir() else 'File'}
- Modified: {stat.st_mtime}
- Permissions: {oct(stat.st_mode)[-3:]}"""
            
            logger.info(f"Retrieved info for {file_path}")
            return info
            
        except Exception as e:
            error_msg = f"Error getting info for {file_path}: {str(e)}"
            logger.error(error_msg)
            return error_msg 