"""
Enhanced Base Bioinformatics Tool for NanoBrain Framework

This module provides an enhanced base class for bioinformatics tools that extends
the core ExternalTool with:
- Auto-detection of existing installations
- Automated conda environment management 
- Installation from conda/bioconda and git repositories
- Detailed error diagnostics and troubleshooting
- Progressive scaling support for real data testing
"""

import asyncio
import os
import shutil
import subprocess
import tempfile
import time
import yaml
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from nanobrain.core.external_tool import ExternalTool, ExternalToolConfig, ToolResult
from nanobrain.core.logging_system import get_logger


@dataclass
class BioinformaticsToolConfig(ExternalToolConfig):
    """Enhanced configuration for bioinformatics tools"""
    # Installation management
    conda_package: Optional[str] = None
    conda_channel: str = "bioconda"
    git_repository: Optional[str] = None
    environment_name: Optional[str] = None
    local_installation_paths: List[str] = field(default_factory=list)
    
    # Progressive scaling
    progressive_scaling: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    initial_scale_level: int = 1
    
    # Error handling
    detailed_diagnostics: bool = True
    suggest_fixes: bool = True
    auto_retry: bool = True
    max_retries: int = 2
    
    # Real data configuration
    use_real_data: bool = True
    mock_fallback: bool = False


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


class BioinformaticsToolError(Exception):
    """Base exception for bioinformatics tool errors"""
    pass


class ToolInstallationError(BioinformaticsToolError):
    """Raised when tool installation fails"""
    pass


class ToolExecutionError(BioinformaticsToolError):
    """Raised when tool execution fails"""
    pass


class BioinformaticsExternalTool(ExternalTool):
    """
    Enhanced base class for bioinformatics external tools.
    
    Provides comprehensive tool management including:
    - Auto-detection of existing installations
    - Automated conda environment creation and management
    - Installation from conda/bioconda and git repositories  
    - Detailed error diagnostics with troubleshooting suggestions
    - Progressive scaling support for real data testing
    - Retry logic with exponential backoff
    """
    
    def __init__(self, config: BioinformaticsToolConfig):
        super().__init__(config)
        self.bio_config = config
        self.logger = get_logger(f"bio_tool_{self.tool_name}")
        
        # Installation state
        self.installation_status = None
        self.environment_name = config.environment_name
        
        # Progressive scaling state
        self.current_scale_level = config.initial_scale_level
        self.scale_config = config.progressive_scaling
        
    async def initialize_tool(self) -> InstallationStatus:
        """
        Initialize tool with comprehensive detection and installation
        
        Returns:
            InstallationStatus: Complete status of tool availability
        """
        self.logger.info(f"Initializing bioinformatics tool: {self.tool_name}")
        
        try:
            # Step 1: Detect existing installation
            self.installation_status = await self.detect_existing_installation()
            
            if self.installation_status.found:
                self.logger.info(f"✅ {self.tool_name} found at {self.installation_status.installation_path}")
                return self.installation_status
            
            # Step 2: Attempt automated installation
            self.logger.info(f"🔄 {self.tool_name} not found, attempting installation...")
            installation_success = await self.install_if_missing()
            
            if installation_success:
                # Re-detect after installation
                self.installation_status = await self.detect_existing_installation()
                if self.installation_status.found:
                    self.logger.info(f"✅ {self.tool_name} installed successfully")
                    return self.installation_status
            
            # Step 3: Generate comprehensive diagnostics
            diagnostic_report = await self.generate_installation_diagnostics()
            self.logger.error(f"❌ {self.tool_name} installation failed")
            self.logger.error(f"Diagnostic report: {diagnostic_report.suggested_fixes}")
            
            raise ToolInstallationError(f"Failed to install {self.tool_name}. See diagnostic report for details.")
            
        except Exception as e:
            self.logger.error(f"Tool initialization failed: {e}")
            raise
    
    async def detect_existing_installation(self) -> InstallationStatus:
        """
        Detect existing tool installations with comprehensive search
        
        Search priority:
        1. Local installation paths (e.g., /Applications/BV-BRC.app/)
        2. Existing conda environments (nanobrain-*)
        3. System PATH
        4. Common installation locations
        """
        status = InstallationStatus()
        
        # Check local installation paths first (for tools like BV-BRC)
        for local_path in self.bio_config.local_installation_paths:
            if await self._check_local_installation(local_path, status):
                return status
        
        # Check existing conda environments
        if await self._check_conda_environments(status):
            return status
            
        # Check system PATH
        if await self._check_system_path(status):
            return status
            
        # Check common installation locations
        await self._check_common_locations(status)
        
        return status
    
    async def _check_local_installation(self, local_path: str, status: InstallationStatus) -> bool:
        """Check local installation path (e.g., BV-BRC app)"""
        try:
            path = Path(local_path)
            if path.exists():
                # For BV-BRC specifically, check for executable scripts
                if "BV-BRC" in str(path):
                    executable_path = path
                    test_executable = executable_path / "p3-all-genomes"
                    
                    if test_executable.exists() and os.access(test_executable, os.X_OK):
                        status.found = True
                        status.installation_path = str(path.parent)
                        status.executable_path = str(executable_path)
                        status.installation_type = "local"
                        status.version = await self._get_version_from_path(str(executable_path))
                        self.logger.info(f"Found local installation: {executable_path}")
                        return True
                    else:
                        status.issues.append(f"Local path exists but executables not found: {executable_path}")
                else:
                    # Generic local installation check
                    if self._is_valid_tool_installation(path):
                        status.found = True
                        status.installation_path = str(path)
                        status.installation_type = "local"
                        return True
                        
        except Exception as e:
            status.issues.append(f"Error checking local path {local_path}: {e}")
            
        return False
    
    async def _check_conda_environments(self, status: InstallationStatus) -> bool:
        """Check existing conda environments for the tool"""
        try:
            # Check if conda is available
            conda_result = await asyncio.create_subprocess_exec(
                "conda", "env", "list", "--json",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await conda_result.communicate()
            
            if conda_result.returncode == 0:
                import json
                env_data = json.loads(stdout.decode())
                environments = env_data.get("envs", [])
                
                # Look for nanobrain environments
                for env_path in environments:
                    env_name = Path(env_path).name
                    if env_name.startswith("nanobrain-") and self.tool_name in env_name:
                        # Check if tool is available in this environment
                        if await self._check_tool_in_environment(env_path, env_name):
                            status.found = True
                            status.installation_path = env_path
                            status.installation_type = "conda"
                            status.environment_name = env_name
                            self.environment_name = env_name
                            return True
                            
        except Exception as e:
            status.issues.append(f"Error checking conda environments: {e}")
            
        return False
    
    async def _check_system_path(self, status: InstallationStatus) -> bool:
        """Check if tool is available in system PATH"""
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
        """Check common installation locations"""
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
    
    async def install_if_missing(self) -> bool:
        """
        Attempt automated installation with multiple strategies
        
        Installation priority:
        1. Conda/bioconda package
        2. Git repository clone and build
        3. Provide manual installation instructions
        """
        if not self.bio_config.conda_package and not self.bio_config.git_repository:
            self.logger.warning(f"No installation methods configured for {self.tool_name}")
            return False
        
        # Strategy 1: Conda installation
        if self.bio_config.conda_package:
            if await self._install_via_conda():
                return True
        
        # Strategy 2: Git repository installation
        if self.bio_config.git_repository:
            if await self._install_from_git():
                return True
        
        # Strategy 3: Manual installation guidance
        await self._provide_manual_installation_guidance()
        return False
    
    async def _install_via_conda(self) -> bool:
        """Install tool via conda/bioconda"""
        try:
            self.logger.info(f"🔄 Installing {self.tool_name} via conda...")
            
            # Create environment name if not specified
            if not self.environment_name:
                workflow_name = getattr(self.bio_config, 'workflow_name', 'viral_protein')
                self.environment_name = f"nanobrain-{workflow_name}-{self.tool_name}"
            
            # Create conda environment
            if not await self._create_conda_environment():
                return False
            
            # Install package in environment
            install_cmd = [
                "conda", "install", "-n", self.environment_name,
                "-c", self.bio_config.conda_channel,
                self.bio_config.conda_package, "-y"
            ]
            
            process = await asyncio.create_subprocess_exec(
                *install_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                self.logger.info(f"✅ Successfully installed {self.tool_name} via conda")
                return True
            else:
                self.logger.error(f"❌ Conda installation failed: {stderr.decode()}")
                return False
                
        except Exception as e:
            self.logger.error(f"Conda installation error: {e}")
            return False
    
    async def _install_from_git(self) -> bool:
        """Install tool from git repository"""
        try:
            self.logger.info(f"🔄 Installing {self.tool_name} from git repository...")
            
            # Create environment if needed
            if not self.environment_name:
                workflow_name = getattr(self.bio_config, 'workflow_name', 'viral_protein')
                self.environment_name = f"nanobrain-{workflow_name}-{self.tool_name}"
                
            if not await self._create_conda_environment():
                return False
            
            # Clone and build
            temp_dir = tempfile.mkdtemp()
            try:
                # Clone repository
                clone_cmd = ["git", "clone", self.bio_config.git_repository, temp_dir]
                clone_process = await asyncio.create_subprocess_exec(
                    *clone_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                await clone_process.communicate()
                
                if clone_process.returncode != 0:
                    self.logger.error("Git clone failed")
                    return False
                
                # Build in conda environment
                build_success = await self._build_tool_in_environment(temp_dir)
                return build_success
                
            finally:
                # Cleanup
                shutil.rmtree(temp_dir, ignore_errors=True)
                
        except Exception as e:
            self.logger.error(f"Git installation error: {e}")
            return False
    
    async def _create_conda_environment(self) -> bool:
        """Create conda environment for tool"""
        try:
            # Check if environment already exists
            list_cmd = ["conda", "env", "list", "--json"]
            list_process = await asyncio.create_subprocess_exec(
                *list_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await list_process.communicate()
            
            if list_process.returncode == 0:
                import json
                env_data = json.loads(stdout.decode())
                existing_envs = [Path(env).name for env in env_data.get("envs", [])]
                
                if self.environment_name in existing_envs:
                    self.logger.info(f"Environment {self.environment_name} already exists")
                    return True
            
            # Create new environment
            create_cmd = ["conda", "create", "-n", self.environment_name, "python=3.11", "-y"]
            create_process = await asyncio.create_subprocess_exec(
                *create_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await create_process.communicate()
            
            if create_process.returncode == 0:
                self.logger.info(f"✅ Created conda environment: {self.environment_name}")
                return True
            else:
                self.logger.error(f"❌ Failed to create environment: {stderr.decode()}")
                return False
                
        except Exception as e:
            self.logger.error(f"Environment creation error: {e}")
            return False
    
    async def generate_installation_diagnostics(self) -> DiagnosticReport:
        """Generate comprehensive diagnostic report"""
        report = DiagnosticReport(
            tool_name=self.tool_name,
            installation_status=self.installation_status or InstallationStatus()
        )
        
        # Check connectivity (if applicable)
        if hasattr(self, '_test_connectivity'):
            report.connectivity_test = await self._test_connectivity()
        
        # Check permissions
        report.permissions_check = await self._check_permissions()
        
        # Check dependencies
        report.dependency_status = await self._check_dependencies()
        
        # Generate specific suggestions based on tool type
        report.suggested_fixes = await self._generate_specific_suggestions()
        
        # Provide alternative installation methods
        report.alternative_methods = await self._get_alternative_methods()
        
        return report
    
    async def execute_with_progressive_scaling(self, scale_level: int = None) -> Any:
        """
        Execute tool with progressive scaling support
        
        Args:
            scale_level: Scale level (1-4), uses current if not specified
        """
        if scale_level is not None:
            self.current_scale_level = scale_level
        
        scale_config = self.scale_config.get(self.current_scale_level, {})
        self.logger.info(f"Executing {self.tool_name} at scale level {self.current_scale_level}: {scale_config}")
        
        # Let subclasses implement specific scaling logic
        return await self._execute_at_scale(scale_config)
    
    async def execute_with_retry(self, command: List[str], **kwargs) -> ToolResult:
        """
        Execute command with automatic retry and exponential backoff
        
        Args:
            command: Command to execute
            **kwargs: Additional arguments for execution
            
        Returns:
            ToolResult: Result of successful execution
            
        Raises:
            ToolExecutionError: If all retry attempts fail
        """
        last_error = None
        
        for attempt in range(self.bio_config.max_retries):
            try:
                self.logger.debug(f"Executing {self.tool_name} (attempt {attempt + 1}/{self.bio_config.max_retries})")
                
                result = await self.execute_command(command, **kwargs)
                
                if result.success:
                    if attempt > 0:
                        self.logger.info(f"✅ {self.tool_name} succeeded after {attempt + 1} attempts")
                    return result
                else:
                    last_error = ToolExecutionError(f"Command failed: {result.stderr_text}")
                    
            except Exception as e:
                last_error = e
                self.logger.warning(f"Attempt {attempt + 1} failed: {e}")
                
            if attempt < self.bio_config.max_retries - 1:
                # Exponential backoff: 1s, 2s, 4s
                wait_time = 2 ** attempt
                self.logger.debug(f"Waiting {wait_time}s before retry...")
                await asyncio.sleep(wait_time)
        
        # All attempts failed
        self.logger.error(f"❌ {self.tool_name} failed after {self.bio_config.max_retries} attempts")
        
        if last_error:
            raise last_error
        else:
            raise ToolExecutionError(f"{self.tool_name} execution failed after all retry attempts")
    
    # Abstract methods for subclasses
    
    @abstractmethod
    async def _execute_at_scale(self, scale_config: Dict[str, Any]) -> Any:
        """Execute tool with specific scale configuration"""
        pass
    
    @abstractmethod
    async def _find_executable_in_path(self) -> Optional[str]:
        """Find tool executable in system PATH"""
        pass
    
    @abstractmethod
    async def _check_tool_in_environment(self, env_path: str, env_name: str) -> bool:
        """Check if tool is available in conda environment"""
        pass
    
    @abstractmethod
    async def _check_tool_in_directory(self, directory: str) -> bool:
        """Check if tool is available in specific directory"""
        pass
    
    @abstractmethod
    async def _build_tool_in_environment(self, source_dir: str) -> bool:
        """Build tool from source in conda environment"""
        pass
    
    @abstractmethod
    async def _generate_specific_suggestions(self) -> List[str]:
        """Generate tool-specific installation suggestions"""
        pass
    
    @abstractmethod
    async def _get_alternative_methods(self) -> List[str]:
        """Get alternative installation methods"""
        pass
    
    # Helper methods
    
    def _is_valid_tool_installation(self, path: Path) -> bool:
        """Check if path contains valid tool installation"""
        # Default implementation - override in subclasses
        return path.exists() and path.is_dir()
    
    async def _get_version_from_path(self, executable_path: str) -> Optional[str]:
        """Get tool version from executable path"""
        try:
            # Try common version flags
            for version_flag in ["--version", "-v", "version", "--help"]:
                try:
                    process = await asyncio.create_subprocess_exec(
                        f"{executable_path}/{self.tool_name}",
                        version_flag,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    stdout, stderr = await process.communicate()
                    
                    if process.returncode == 0:
                        output = stdout.decode() + stderr.decode()
                        # Extract version number (basic regex)
                        import re
                        version_match = re.search(r'(\d+\.\d+(?:\.\d+)?)', output)
                        if version_match:
                            return version_match.group(1)
                except:
                    continue
                    
        except Exception:
            pass
            
        return "Unknown"
    
    async def _check_permissions(self) -> bool:
        """Check if we have necessary permissions"""
        try:
            # Check write permissions in typical installation directories
            test_dirs = [
                "/tmp",
                os.path.expanduser("~"),
                "/usr/local" if os.access("/usr/local", os.W_OK) else None
            ]
            
            for test_dir in test_dirs:
                if test_dir and os.access(test_dir, os.W_OK):
                    return True
                    
            return False
        except Exception:
            return False
    
    async def _check_dependencies(self) -> Dict[str, bool]:
        """Check common dependencies"""
        dependencies = {}
        
        # Check conda
        try:
            conda_process = await asyncio.create_subprocess_exec(
                "conda", "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await conda_process.communicate()
            dependencies["conda"] = conda_process.returncode == 0
        except:
            dependencies["conda"] = False
        
        # Check git
        try:
            git_process = await asyncio.create_subprocess_exec(
                "git", "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await git_process.communicate()
            dependencies["git"] = git_process.returncode == 0
        except:
            dependencies["git"] = False
        
        return dependencies
    
    async def _provide_manual_installation_guidance(self) -> None:
        """Provide manual installation instructions"""
        self.logger.info(f"Manual installation required for {self.tool_name}")
        
        instructions = [
            f"Please install {self.tool_name} manually:",
            f"1. Visit the official {self.tool_name} website",
            f"2. Download the appropriate version for your system",
            f"3. Follow the installation instructions",
            f"4. Ensure the tool is accessible in PATH or update configuration"
        ]
        
        if self.bio_config.conda_package:
            instructions.extend([
                f"",
                f"Alternative: Install via conda:",
                f"conda create -n {self.environment_name} -c {self.bio_config.conda_channel} {self.bio_config.conda_package}"
            ])
        
        for instruction in instructions:
            self.logger.info(instruction) 