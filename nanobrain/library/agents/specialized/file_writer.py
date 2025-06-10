"""
File Writer Agent for NanoBrain Framework

Specialized agent for file operations based on natural language descriptions.
"""

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

# Updated imports for nanobrain package structure
from nanobrain.core.agent import AgentConfig
from .base import SpecializedAgentBase, SimpleSpecializedAgent, ConversationalSpecializedAgent

logger = logging.getLogger(__name__)


class FileWriterAgentMixin(SpecializedAgentBase):
    """
    Mixin class providing file operation capabilities for specialized agents.
    
    This agent can create, write, and manage files based on natural
    language descriptions. It intelligently detects file operations
    and performs them directly when possible.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # File operation patterns
        self.file_operation_patterns = {
            'create': [
                r"create.*?(?:file|script).*?(?:at|to|in)\s+([^\s]+)",
                r"(?:file|script).*?(?:called|named)\s+([^\s]+)",
            ],
            'save': [
                r"save.*?(?:to|at|in)\s+([^\s]+)",
                r"write.*?(?:to|at|in)\s+([^\s]+)",
            ],
            'content': [
                r"with content:\s*(.+)",
                r"```(?:python|py)?\s*\n(.*?)\n```",
            ]
        }
        
        # File operation keywords
        self.file_keywords = ['write', 'save', 'create', 'file', 'path', 'document']
    
    async def _initialize_specialized_features(self) -> None:
        """Initialize file writer specific features."""
        await super()._initialize_specialized_features()
        self.specialized_logger.info("FileWriter specialized features initialized")
    
    def _should_handle_specialized(self, input_text: str, **kwargs) -> bool:
        """
        Determine if this request should be handled by file operation logic.
        
        Args:
            input_text: Input text
            **kwargs: Additional parameters
            
        Returns:
            True if should be handled by file operation logic
        """
        # Check for explicit file operation parameters
        has_file_path = 'file_path' in kwargs or 'path' in kwargs
        has_content = 'content' in kwargs or 'data' in kwargs
        
        # Check for file operation keywords in input
        has_keywords = any(keyword in input_text.lower() for keyword in self.file_keywords)
        
        return has_file_path and (has_content or has_keywords) or self._can_parse_file_operation(input_text)
    
    def _can_parse_file_operation(self, input_text: str) -> bool:
        """Check if input text contains parseable file operations."""
        # Look for file creation patterns
        for pattern_list in self.file_operation_patterns['create'] + self.file_operation_patterns['save']:
            for pattern in pattern_list:
                if re.search(pattern, input_text, re.IGNORECASE):
                    return True
        return False
    
    async def _process_specialized_request(self, input_text: str, **kwargs) -> Optional[str]:
        """
        Process file operation requests directly without LLM.
        
        Args:
            input_text: Input text to process
            **kwargs: Additional parameters
            
        Returns:
            Result of file operation if handled, None otherwise
        """
        try:
            # Check if this is a direct file operation request
            if self._is_direct_file_operation(input_text, **kwargs):
                return await self._handle_direct_file_operation(input_text, **kwargs)
            
            # Try to parse and execute file operation from text
            result = await self._try_parse_and_execute_file_operation(input_text)
            if result:
                return result
            
            return None
            
        except Exception as e:
            self.specialized_logger.error(f"Error in specialized file processing: {e}")
            return f"Error processing file operation: {str(e)}"
    
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
        has_keywords = any(keyword in input_text.lower() for keyword in self.file_keywords)
        
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
            file_path = None
            for pattern in self.file_operation_patterns['create'] + self.file_operation_patterns['save']:
                match = re.search(pattern, input_text, re.IGNORECASE)
                if match:
                    file_path = match.group(1).strip('"\'')
                    break
            
            if not file_path:
                return None
            
            # Look for content patterns
            content = None
            for pattern in self.file_operation_patterns['content']:
                content_match = re.search(pattern, input_text, re.IGNORECASE | re.DOTALL)
                if content_match:
                    content = content_match.group(1).strip()
                    break
            
            # Generate content based on descriptions if not found
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
                self._track_specialized_operation("parse_and_execute", success=True)
                return f"Successfully processed file operation: {result}"
            
            return None
            
        except Exception as e:
            self._track_specialized_operation("parse_and_execute", success=False)
            self.specialized_logger.error(f"Error parsing file operation: {e}")
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
            self._track_specialized_operation("direct_operation", success=True)
            
            self.specialized_logger.info(f"FileWriterAgent handled direct operation: {file_path}")
            return result
            
        except Exception as e:
            self._track_specialized_operation("direct_operation", success=False)
            error_msg = f"Error in direct file operation: {str(e)}"
            self.specialized_logger.error(error_msg)
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
                self.specialized_logger.debug(f"Created directories for {file_path}")
            
            # Write file
            path.write_text(content, encoding=encoding)
            
            file_size = path.stat().st_size
            success_msg = f"Successfully wrote {file_size} bytes to {file_path}"
            self.specialized_logger.info(success_msg)
            self._track_specialized_operation("write_file", success=True)
            return success_msg
            
        except PermissionError:
            error_msg = f"Permission denied: Cannot write to {file_path}"
            self.specialized_logger.error(error_msg)
            self._track_specialized_operation("write_file", success=False)
            return error_msg
        except OSError as e:
            error_msg = f"OS error writing to {file_path}: {str(e)}"
            self.specialized_logger.error(error_msg)
            self._track_specialized_operation("write_file", success=False)
            return error_msg
        except Exception as e:
            error_msg = f"Unexpected error writing to {file_path}: {str(e)}"
            self.specialized_logger.error(error_msg)
            self._track_specialized_operation("write_file", success=False)
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
            self.specialized_logger.info(success_msg)
            self._track_specialized_operation("append_file", success=True)
            return success_msg
            
        except Exception as e:
            error_msg = f"Error appending to {file_path}: {str(e)}"
            self.specialized_logger.error(error_msg)
            self._track_specialized_operation("append_file", success=False)
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
            self.specialized_logger.info(f"Successfully read {len(content)} characters from {file_path}")
            self._track_specialized_operation("read_file", success=True)
            return content
            
        except Exception as e:
            error_msg = f"Error reading {file_path}: {str(e)}"
            self.specialized_logger.error(error_msg)
            self._track_specialized_operation("read_file", success=False)
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
            self.specialized_logger.info(success_msg)
            self._track_specialized_operation("delete_file", success=True)
            return success_msg
            
        except Exception as e:
            error_msg = f"Error deleting {file_path}: {str(e)}"
            self.specialized_logger.error(error_msg)
            self._track_specialized_operation("delete_file", success=False)
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
            self.specialized_logger.info(success_msg)
            self._track_specialized_operation("create_directory", success=True)
            return success_msg
            
        except Exception as e:
            error_msg = f"Error creating directory {dir_path}: {str(e)}"
            self.specialized_logger.error(error_msg)
            self._track_specialized_operation("create_directory", success=False)
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
            self.specialized_logger.info(f"Listed {len(items)} items in {dir_path}")
            self._track_specialized_operation("list_directory", success=True)
            return listing
            
        except Exception as e:
            error_msg = f"Error listing directory {dir_path}: {str(e)}"
            self.specialized_logger.error(error_msg)
            self._track_specialized_operation("list_directory", success=False)
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
            
            self.specialized_logger.info(f"Retrieved info for {file_path}")
            self._track_specialized_operation("get_file_info", success=True)
            return info
            
        except Exception as e:
            error_msg = f"Error getting info for {file_path}: {str(e)}"
            self.specialized_logger.error(error_msg)
            self._track_specialized_operation("get_file_info", success=False)
            return error_msg


class FileWriterAgent(FileWriterAgentMixin, SimpleSpecializedAgent):
    """
    Simple file writer agent that processes input without conversation history.
    
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
        
        super().__init__(config=config, **kwargs)


class ConversationalFileWriterAgent(FileWriterAgentMixin, ConversationalSpecializedAgent):
    """
    Conversational file writer agent that maintains conversation history.
    
    This agent can create, write, and manage files based on natural
    language descriptions while maintaining conversation context.
    """
    
    def __init__(self, config: Optional[AgentConfig] = None, **kwargs):
        # Use provided config or create minimal default
        if config is None:
            config = AgentConfig(
                name="conversational_file_writer",
                description="Conversational specialized agent for file operations",
                model="gpt-3.5-turbo",
                system_prompt="""You are a conversational file writing agent for the NanoBrain framework.

You maintain conversation context while handling file operations. When asked to create, write, or save files, you MUST actually perform the file operation.

Your capabilities:
1. Create and write files with any content
2. Parse file paths and content from natural language requests
3. Handle directory creation automatically
4. Maintain conversation context across file operations
5. Provide clear feedback about file operations

You can reference previous conversations and file operations to provide better assistance.""",
                temperature=0.3  # Slightly higher for conversational context
            )
        
        super().__init__(config=config, **kwargs) 