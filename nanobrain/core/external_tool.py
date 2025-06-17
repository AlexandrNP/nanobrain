"""
External Tool Base Class for NanoBrain Framework

This module provides the base class for integrating external tools into the NanoBrain
bioinformatics workflow system. It handles tool installation verification, command
execution, environment management, and result parsing.
"""

import asyncio
import os
import shutil
import subprocess
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from nanobrain.core.logging_system import get_logger


@dataclass
class ExternalToolConfig:
    """Configuration for external tools"""
    tool_name: str
    installation_path: Optional[str] = None
    executable_path: Optional[str] = None
    environment: Optional[Dict[str, str]] = None
    timeout_seconds: int = 300
    retry_attempts: int = 3
    verify_on_init: bool = True


@dataclass
class ToolResult:
    """Result from external tool execution"""
    returncode: int
    stdout: bytes
    stderr: bytes
    execution_time: float
    command: List[str]
    success: bool
    
    @property
    def stdout_text(self) -> str:
        """Get stdout as decoded text"""
        return self.stdout.decode('utf-8', errors='replace')
    
    @property
    def stderr_text(self) -> str:
        """Get stderr as decoded text"""
        return self.stderr.decode('utf-8', errors='replace')


class ExternalToolError(Exception):
    """Base exception for external tool errors"""
    pass


class ToolInstallationError(ExternalToolError):
    """Raised when tool installation verification fails"""
    pass


class ToolExecutionError(ExternalToolError):
    """Raised when tool execution fails"""
    pass


class ExternalTool(ABC):
    """
    Base class for external tool integration in NanoBrain framework.
    
    This class provides:
    - Tool installation and verification
    - Command execution with proper error handling
    - Environment management (conda, docker, etc.)
    - Result parsing and validation
    - Integration with NanoBrain logging and monitoring
    """
    
    def __init__(self, config: ExternalToolConfig):
        self.config = config
        self.tool_name = config.tool_name
        self.installation_path = config.installation_path
        self.executable_path = config.executable_path
        self.environment = config.environment or {}
        self.timeout_seconds = config.timeout_seconds
        self.retry_attempts = config.retry_attempts
        self.logger = get_logger(f"external_tool_{self.tool_name}")
        
        # Initialize tool if verification is enabled
        if config.verify_on_init:
            asyncio.create_task(self._initialize_tool())
    
    async def _initialize_tool(self) -> None:
        """Initialize tool with verification"""
        try:
            if not await self.verify_installation():
                self.logger.warning(f"Tool {self.tool_name} verification failed")
                if await self.install_if_missing():
                    self.logger.info(f"Tool {self.tool_name} installed successfully")
                else:
                    raise ToolInstallationError(f"Failed to install {self.tool_name}")
        except Exception as e:
            self.logger.error(f"Tool initialization failed: {e}")
            raise
    
    @abstractmethod
    async def verify_installation(self) -> bool:
        """Verify tool is properly installed and accessible"""
        pass
    
    @abstractmethod
    async def execute_command(self, command: List[str], **kwargs) -> ToolResult:
        """Execute tool command with proper error handling"""
        pass
    
    @abstractmethod
    async def parse_output(self, raw_output: str) -> Dict[str, Any]:
        """Parse tool-specific output format"""
        pass
    
    async def install_if_missing(self) -> bool:
        """Install tool if not found (default implementation returns False)"""
        self.logger.warning(f"Auto-installation not implemented for {self.tool_name}")
        return False
    
    async def get_version(self) -> str:
        """Get tool version information"""
        try:
            result = await self.execute_command(["--version"])
            if result.success:
                return result.stdout_text.strip()
            else:
                return "Unknown"
        except Exception:
            return "Unknown"
    
    async def _execute_with_retry(self, command: List[str], **kwargs) -> ToolResult:
        """Execute command with retry logic"""
        last_error = None
        
        for attempt in range(self.retry_attempts):
            try:
                self.logger.debug(f"Executing {self.tool_name} command (attempt {attempt + 1}): {' '.join(command)}")
                result = await self._execute_single_command(command, **kwargs)
                
                if result.success:
                    return result
                else:
                    last_error = ToolExecutionError(f"Command failed: {result.stderr_text}")
                    
            except Exception as e:
                last_error = e
                self.logger.warning(f"Attempt {attempt + 1} failed: {e}")
                
            if attempt < self.retry_attempts - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                await asyncio.sleep(wait_time)
        
        raise last_error or ToolExecutionError("All retry attempts failed")
    
    async def _execute_single_command(self, command: List[str], **kwargs) -> ToolResult:
        """Execute a single command with timeout and environment setup"""
        start_time = time.time()
        
        # Prepare environment
        env = os.environ.copy()
        env.update(self.environment)
        
        # Add executable path to PATH if specified
        if self.executable_path:
            env["PATH"] = f"{self.executable_path}:{env.get('PATH', '')}"
        
        # Prepare command with full path if needed
        full_command = self._prepare_command(command)
        
        try:
            process = await asyncio.create_subprocess_exec(
                *full_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                **kwargs
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.timeout_seconds
            )
            
            execution_time = time.time() - start_time
            
            return ToolResult(
                returncode=process.returncode,
                stdout=stdout,
                stderr=stderr,
                execution_time=execution_time,
                command=full_command,
                success=process.returncode == 0
            )
            
        except asyncio.TimeoutError:
            self.logger.error(f"Command timeout after {self.timeout_seconds} seconds")
            raise ToolExecutionError(f"Command timeout: {' '.join(command)}")
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Command execution failed: {e}")
            raise ToolExecutionError(f"Execution failed: {e}")
    
    def _prepare_command(self, command: List[str]) -> List[str]:
        """Prepare command with full executable path if needed"""
        if not command:
            raise ValueError("Empty command")
        
        executable = command[0]
        
        # If executable path is specified and command doesn't contain path separators
        if self.executable_path and not os.path.sep in executable:
            full_executable = os.path.join(self.executable_path, executable)
            if os.path.exists(full_executable):
                return [full_executable] + command[1:]
        
        # Check if executable exists in PATH
        if shutil.which(executable):
            return command
        
        # If installation path is specified, try there
        if self.installation_path:
            full_executable = os.path.join(self.installation_path, executable)
            if os.path.exists(full_executable):
                return [full_executable] + command[1:]
        
        # Return original command and let the system handle it
        return command
    
    async def create_temp_file(self, content: str, suffix: str = ".tmp") -> str:
        """Create temporary file with content"""
        with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as f:
            f.write(content)
            return f.name
    
    async def cleanup_temp_file(self, filepath: str) -> None:
        """Clean up temporary file"""
        try:
            os.unlink(filepath)
        except Exception as e:
            self.logger.warning(f"Failed to cleanup temp file {filepath}: {e}")
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.tool_name})"
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(tool_name='{self.tool_name}', installation_path='{self.installation_path}')" 