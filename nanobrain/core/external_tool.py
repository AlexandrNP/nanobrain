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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from nanobrain.core.logging_system import get_logger
from nanobrain.core.component_base import FromConfigBase
# Import new ConfigBase for constructor prohibition
from nanobrain.core.config.config_base import ConfigBase
from pydantic import Field


class ExternalToolConfig(ConfigBase):
    """
    Configuration for external tools - INHERITS constructor prohibition.
    
    ❌ FORBIDDEN: ExternalToolConfig(tool_name="test", ...)
    ✅ REQUIRED: ExternalToolConfig.from_config('path/to/config.yml')
    """
    tool_name: str
    installation_path: Optional[str] = None
    executable_path: Optional[str] = None
    environment: Optional[Dict[str, str]] = None
    timeout_seconds: int = 300
    retry_attempts: int = 3
    verify_on_init: bool = True
    
    # NEW: Auto-installation support
    conda_package: Optional[str] = None
    conda_channel: str = "conda-forge"
    pip_package: Optional[str] = None
    git_repository: Optional[str] = None
    create_isolated_environment: bool = False
    environment_name: Optional[str] = None
    local_installation_paths: List[str] = Field(default_factory=list)
    
    # NEW: Progressive scaling support
    progressive_scaling: Dict[int, Dict[str, Any]] = Field(default_factory=dict)
    initial_scale_level: int = 1
    
    # NEW: Enhanced error handling
    detailed_diagnostics: bool = True
    suggest_fixes: bool = True


@dataclass 
class InstallationStatus:
    """Status of tool installation detection"""
    found: bool = False
    installation_path: Optional[str] = None
    executable_path: Optional[str] = None
    version: Optional[str] = None
    installation_type: Optional[str] = None  # "local", "conda", "system"
    environment_name: Optional[str] = None
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


@dataclass
class DiagnosticReport:
    """Comprehensive diagnostic information"""
    tool_name: str
    installation_status: InstallationStatus
    connectivity_test: Optional[bool] = None
    permissions_check: Optional[bool] = None
    dependency_status: Dict[str, bool] = field(default_factory=dict)
    suggested_fixes: List[str] = field(default_factory=list)
    alternative_methods: List[str] = field(default_factory=list)


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


class ExternalTool(FromConfigBase, ABC):
    """
    Base class for external tool integration in NanoBrain framework.
    Enhanced with mandatory from_config pattern implementation.
    
    This class provides:
    - Tool installation and verification
    - Command execution with proper error handling
    - Environment management (conda, docker, etc.)
    - Result parsing and validation
    - Integration with NanoBrain logging and monitoring
    
    BREAKING CHANGE: Uses mandatory lazy async initialization pattern.
    All tools MUST call ensure_initialized() before use.
    """
    
    COMPONENT_TYPE = "external_tool"
    REQUIRED_CONFIG_FIELDS = ['tool_name']
    OPTIONAL_CONFIG_FIELDS = {
        'installation_path': None,
        'executable_path': None,
        'environment': None,
        'timeout_seconds': 300,
        'retry_attempts': 3,
        'verify_on_init': True
    }
    
    @classmethod
    def _get_config_class(cls):
        """UNIFIED PATTERN: Return ExternalToolConfig - ONLY method that differs from other components"""
        return ExternalToolConfig
    
    @classmethod
    def extract_component_config(cls, config: ExternalToolConfig) -> Dict[str, Any]:
        """Extract ExternalTool configuration"""
        return {
            'tool_name': config.tool_name,
            'installation_path': getattr(config, 'installation_path', None),
            'executable_path': getattr(config, 'executable_path', None),
            'environment': getattr(config, 'environment', {}),
            'timeout_seconds': getattr(config, 'timeout_seconds', 300),
            'retry_attempts': getattr(config, 'retry_attempts', 3),
            'verify_on_init': getattr(config, 'verify_on_init', True)
        }
    
    @classmethod  
    def resolve_dependencies(cls, component_config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Resolve ExternalTool dependencies"""
        return {
            'enable_logging': kwargs.get('enable_logging', True)
        }
    
    def _init_from_config(self, config: ExternalToolConfig, component_config: Dict[str, Any],
                         dependencies: Dict[str, Any]) -> None:
        """Initialize ExternalTool with resolved dependencies"""
        self.config = config
        self.tool_name = component_config['tool_name']
        self.installation_path = component_config['installation_path']
        self.executable_path = component_config['executable_path']
        self.environment = component_config['environment'] or {}
        self.timeout_seconds = component_config['timeout_seconds']
        self.retry_attempts = component_config['retry_attempts']
        self.logger = get_logger(f"external_tool_{self.tool_name}")
        
        # BREAKING CHANGE: Lazy initialization pattern
        self._initialization_task = None
        self._initialized = False
        self._initialization_lock = None  # Created when needed
        
        # NO ASYNC TASK CREATION IN __init__ - BREAKING CHANGE
        # Old problematic code removed: asyncio.create_task(self._initialize_tool())
    
    # ExternalTool inherits FromConfigBase.__init__ which prevents direct instantiation
    
    async def ensure_initialized(self) -> None:
        """
        MANDATORY: Ensure tool is initialized before use.
        
        This method MUST be called before any tool operation.
        Uses lazy initialization pattern to avoid async issues in __init__.
        
        BREAKING CHANGE: This method is now mandatory for all tool usage.
        """
        if self._initialized:
            return
            
        # Create lock when needed (avoids async in __init__)
        if self._initialization_lock is None:
            self._initialization_lock = asyncio.Lock()
            
        async with self._initialization_lock:
            if not self._initialized:
                if self._initialization_task is None:
                    self._initialization_task = self._initialize_tool()
                await self._initialization_task
                self._initialized = True
    
    async def _initialize_tool(self) -> None:
        """Initialize tool with verification"""
        try:
            if getattr(self.config, 'verify_on_init', True):
                if not await self.verify_installation():
                    self.logger.warning(f"Tool {self.tool_name} verification failed")
                    if await self.install_if_missing():
                        self.logger.info(f"Tool {self.tool_name} installed successfully")
                    else:
                        raise ToolInstallationError(f"Failed to install {self.tool_name}")
                else:
                    self.logger.info(f"Tool {self.tool_name} verification successful")
        except Exception as e:
            self.logger.error(f"Tool initialization failed: {e}")
            raise
    
    async def verify_installation(self) -> bool:
        """
        Verify tool is properly installed and accessible.
        
        This method provides a comprehensive verification workflow:
        1. Try to detect existing installation
        2. If not found, attempt auto-installation
        3. Re-verify after installation
        4. Return final status
        """
        # Try to detect existing installation first
        status = await self.detect_existing_installation()
        if status.found:
            self.logger.info(f"✅ {self.tool_name} found: {status.installation_type}")
            return True
        
        # If not found and auto-installation is configured, try installing
        if any([
            getattr(self.config, 'conda_package', None),
            getattr(self.config, 'pip_package', None),
            getattr(self.config, 'git_repository', None)
        ]):
            self.logger.info(f"🔄 {self.tool_name} not found, attempting installation...")
            installation_success = await self.auto_install()
            
            if installation_success:
                # Re-detect after installation
                status = await self.detect_existing_installation()
                if status.found:
                    self.logger.info(f"✅ {self.tool_name} installed and configured successfully")
                    return True
        
        # Installation failed or not configured
        self.logger.warning(f"❌ {self.tool_name} verification failed")
        return False
    
    async def detect_existing_installation(self) -> InstallationStatus:
        """
        Detect existing tool installation.
        
        Returns:
            InstallationStatus with detection results
        """
        status = InstallationStatus()
        status.issues = []
        status.suggestions = []
        
        try:
            # Check local installation paths first
            for local_path in getattr(self.config, 'local_installation_paths', []):
                if await self._check_local_installation(local_path, status):
                    return status
            
            # Check conda environments
            if await self._check_conda_environments(status):
                return status
            
            # Check system PATH
            if await self._check_system_path(status):
                return status
            
            # Check common locations and provide suggestions
            await self._check_common_locations(status)
            
        except Exception as e:
            status.issues.append(f"Error during installation detection: {e}")
            self.logger.error(f"Installation detection failed: {e}")
        
        return status
    
    async def _check_local_installation(self, local_path: str, status: InstallationStatus) -> bool:
        """Check if tool is installed in a specific local path."""
        try:
            if await self._check_tool_in_directory(local_path):
                status.found = True
                status.installation_path = local_path
                status.installation_type = "local"
                status.version = await self._get_version_from_path(local_path)
                return True
        except Exception as e:
            status.issues.append(f"Error checking local path {local_path}: {e}")
        return False
    
    async def _check_conda_environments(self, status: InstallationStatus) -> bool:
        """Check if tool is available in conda environments."""
        try:
            # Check if conda is available
            conda_result = await self._execute_single_command(["conda", "env", "list", "--json"])
            if not conda_result.success:
                return False
            
            import json
            env_data = json.loads(conda_result.stdout_text)
            environments = env_data.get('envs', [])
            
            if environments:
                # Look for tool-specific environments
                env_name = getattr(self.config, 'environment_name', None)
                if env_name:
                    for env_path in environments:
                        if Path(env_path).name == env_name:
                            if await self._check_tool_in_environment(env_path, env_name):
                                status.found = True
                                status.installation_path = env_path
                                status.installation_type = "conda"
                                status.environment_name = env_name
                                return True
                            
        except Exception as e:
            status.issues.append(f"Error checking conda environments: {e}")
            
        return False
    
    async def _check_system_path(self, status: InstallationStatus) -> bool:
        """Check if tool is available in system PATH."""
        try:
            executable = await self._find_executable_in_path()
            if executable:
                status.found = True
                status.executable_path = executable
                status.installation_type = "system"
                status.version = await self._get_version_from_path(os.path.dirname(executable))
                return True
        except Exception as e:
            status.issues.append(f"Error checking system PATH: {e}")
            
        return False
    
    async def _check_common_locations(self, status: InstallationStatus) -> None:
        """Check common installation locations."""
        common_paths = [
            "/usr/local/bin",
            "/opt/homebrew/bin", 
            "/usr/bin",
            f"{os.path.expanduser('~')}/bin",
            f"{os.path.expanduser('~')}/anaconda3/bin",
            f"{os.path.expanduser('~')}/miniconda3/bin"
        ]
        
        for path in common_paths:
            try:
                if await self._check_tool_in_directory(path):
                    status.suggestions.append(f"Tool might be available in: {path}")
            except Exception:
                continue
    
    # Abstract methods for tool-specific behavior
    @abstractmethod  
    async def _find_executable_in_path(self) -> Optional[str]:
        """Find tool executable in PATH - must be implemented by subclasses."""
        pass
    
    @abstractmethod
    async def _check_tool_in_environment(self, env_path: str, env_name: str) -> bool:
        """Check if tool is available in conda environment - must be implemented by subclasses."""
        pass
    
    @abstractmethod
    async def _check_tool_in_directory(self, directory: str) -> bool:
        """Check if tool is available in directory - must be implemented by subclasses."""
        pass
    
    async def _get_version_from_path(self, executable_path: str) -> Optional[str]:
        """Get tool version - generalizable with tool-specific override."""
        try:
            result = await self.execute_command(["--version"])
            if result.success:
                return result.stdout_text.strip()
            return None
        except Exception:
            return None
    
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
    
    async def auto_install(self) -> bool:
        """
        Attempt automatic installation using configured methods.
        
        Returns:
            True if installation successful, False otherwise
        """
        self.logger.info(f"🔧 Attempting auto-installation of {self.tool_name}")
        
        # Try conda installation first
        conda_package = getattr(self.config, 'conda_package', None)
        if conda_package:
            if await self._install_via_conda():
                return True
        
        # Try pip installation
        pip_package = getattr(self.config, 'pip_package', None)
        if pip_package:
            if await self._install_via_pip():
                return True
        
        # Try git installation
        git_repository = getattr(self.config, 'git_repository', None)
        if git_repository:
            if await self._install_from_git():
                return True
        
        self.logger.error(f"❌ All auto-installation methods failed for {self.tool_name}")
        return False
    
    async def _install_via_conda(self) -> bool:
        """Install tool via conda/mamba."""
        try:
            conda_package = getattr(self.config, 'conda_package', None)
            conda_channel = getattr(self.config, 'conda_channel', 'conda-forge')
            environment_name = getattr(self.config, 'environment_name', None)
            create_isolated = getattr(self.config, 'create_isolated_environment', False)
            
            if not conda_package:
                return False
            
            self.logger.info(f"📦 Installing {conda_package} via conda from {conda_channel}")
            
            # Create isolated environment if requested
            if create_isolated and environment_name:
                if not await self._create_conda_environment():
                    return False
                
                # Install in the specific environment
                cmd = ["conda", "install", "-n", environment_name, "-c", conda_channel, conda_package, "-y"]
            else:
                # Install in current environment
                cmd = ["conda", "install", "-c", conda_channel, conda_package, "-y"]
            
            result = await self._execute_single_command(cmd)
            if result.success:
                self.logger.info(f"✅ Successfully installed {conda_package} via conda")
                return True
            else:
                self.logger.error(f"❌ Conda installation failed: {result.stderr_text}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error during conda installation: {e}")
            return False
    
    async def _install_via_pip(self) -> bool:
        """Install tool via pip."""
        try:
            pip_package = getattr(self.config, 'pip_package', None)
            if not pip_package:
                return False
            
            self.logger.info(f"🐍 Installing {pip_package} via pip")
            
            cmd = ["pip", "install", pip_package]
            result = await self._execute_single_command(cmd)
            
            if result.success:
                self.logger.info(f"✅ Successfully installed {pip_package} via pip")
                return True
            else:
                self.logger.error(f"❌ Pip installation failed: {result.stderr_text}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error during pip installation: {e}")
            return False
    
    async def _install_from_git(self) -> bool:
        """Install tool from git repository."""
        try:
            git_repository = getattr(self.config, 'git_repository', None)
            environment_name = getattr(self.config, 'environment_name', None)
            
            if not git_repository:
                return False
            
            self.logger.info(f"🌐 Cloning {git_repository}")
            
            # Create temporary directory for git clone
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                clone_path = os.path.join(temp_dir, 'source')
                
                # Clone repository
                clone_cmd = ["git", "clone", git_repository, clone_path]
                result = await self._execute_single_command(clone_cmd)
                
                if not result.success:
                    self.logger.error(f"❌ Git clone failed: {result.stderr_text}")
                    return False
                
                # Build tool in environment if specified
                if environment_name:
                    return await self._build_tool_in_environment(clone_path)
                else:
                    # Try standard build process
                    return await self._build_tool_standard(clone_path)
                    
        except Exception as e:
            self.logger.error(f"❌ Error during git installation: {e}")
            return False
    
    async def _create_conda_environment(self) -> bool:
        """Create conda environment for tool."""
        try:
            environment_name = getattr(self.config, 'environment_name', None)
            if not environment_name:
                return False
            
            self.logger.info(f"🔧 Creating conda environment: {environment_name}")
            
            cmd = ["conda", "create", "-n", environment_name, "python", "-y"]
            result = await self._execute_single_command(cmd)
            
            if result.success:
                self.logger.info(f"✅ Created conda environment: {environment_name}")
                return True
            else:
                self.logger.error(f"❌ Failed to create environment: {result.stderr_text}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error creating conda environment: {e}")
            return False
    
    async def _build_tool_standard(self, source_dir: str) -> bool:
        """Standard build process for tools."""
        try:
            # Look for common build files
            source_path = Path(source_dir)
            
            if (source_path / "setup.py").exists():
                # Python package
                cmd = ["pip", "install", "-e", source_dir]
            elif (source_path / "Makefile").exists():
                # Makefile-based build
                cmd = ["make", "-C", source_dir, "install"]
            elif (source_path / "CMakeLists.txt").exists():
                # CMake build
                build_dir = source_path / "build"
                build_dir.mkdir(exist_ok=True)
                cmake_cmd = ["cmake", "-B", str(build_dir), "-S", source_dir]
                await self._execute_single_command(cmake_cmd)
                cmd = ["cmake", "--build", str(build_dir), "--target", "install"]
            else:
                self.logger.warning(f"⚠️ No recognized build system found in {source_dir}")
                return False
            
            result = await self._execute_single_command(cmd)
            return result.success
            
        except Exception as e:
            self.logger.error(f"❌ Error during standard build: {e}")
            return False
    
    @abstractmethod
    async def _build_tool_in_environment(self, source_dir: str) -> bool:
        """Build tool in specific environment - must be implemented by subclasses."""
        pass
    
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
        """
        Execute command with retry logic.
        
        BREAKING CHANGE: Now enforces mandatory initialization before execution.
        """
        # MANDATORY: Ensure tool is initialized before any execution
        await self.ensure_initialized()
        
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
        
        # Extract timeout from kwargs - it's handled by asyncio.wait_for, not subprocess
        subprocess_kwargs = {k: v for k, v in kwargs.items() if k != 'timeout'}
        
        try:
            process = await asyncio.create_subprocess_exec(
                *full_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                **subprocess_kwargs
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