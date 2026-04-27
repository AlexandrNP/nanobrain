"""
Cleanup Manager for the Nanobrain cleanup system.

This module provides comprehensive temporary file cleanup functionality,
identifying and removing various types of build artifacts, cache files,
and job outputs while preserving essential source code and documentation.
"""

import os
import re
import shutil
from pathlib import Path
from typing import List, Set, Dict, Optional
from datetime import datetime
import logging
import fnmatch

from .models import TemporaryFile, CleanupResult


class CleanupManager:
    """
    Manages cleanup of temporary files and build artifacts.
    
    Identifies and removes various categories of temporary files while
    preserving essential source code, documentation, and configuration files.
    """
    
    def __init__(self, repo_root: str):
        """
        Initialize the CleanupManager.
        
        Args:
            repo_root: Path to the repository root directory
        """
        self.repo_root = Path(repo_root).resolve()
        self.logger = logging.getLogger(__name__)
        
        # Directories to exclude from scanning for performance
        self.exclude_dirs = {
            '.git',
            'pubmed_text',  # Large text corpus - CRITICAL: This is huge
            'node_modules',
            '.venv',
            'venv',
            'env',
            '.env',
            'data',  # Exclude entire data directory - it's massive
            'rag_demo/frontend/node_modules',
            'rag_demo/frontend/build',
            '.tox',
            '.nox',
            'results',  # Large results directory
            '.pytest_cache',
            '.hypothesis',
            '__pycache__'
        }
        
        # Define file patterns for different types of temporary files
        self.pbs_patterns = [
            '*.err',
            '*.out', 
            'STDIN.e*',
            'STDIN.o*',
            '*.sh.e[0-9]*',  # Shell script PBS error files
            '*.sh.o[0-9]*'   # Shell script PBS output files
        ]
        
        self.python_cache_patterns = [
            '__pycache__',
            '*.pyc',
            '*.pyo',
            '*.pyd',
            '.pytest_cache',
            '.hypothesis',
            '.coverage',
            'htmlcov',
            '.tox',
            '.nox'
        ]
        
        self.build_artifact_patterns = [
            'build',
            'dist',
            '*.egg-info',
            '*.egg',
            '.eggs',
            'wheels',
            'sdist'
        ]
        
        self.log_patterns = [
            'logs',
            'runinfo',
            'executed_workflows',
            'cmd_parsl.*',
            'parsl.*',
            '*.log',
            'routing_test_logs.txt',
            'test-results'
        ]
        
        # Files and directories to always preserve
        self.preserve_patterns = [
            '.git',
            '.gitignore',
            '.gitattributes',
            'pyproject.toml',
            'setup.py',
            'setup.cfg',
            'requirements*.txt',
            'README*',
            'LICENSE*',
            'CHANGELOG*',
            'MANIFEST.in',
            '*.md',
            '*.rst',
            '*.yml',
            '*.yaml',
            '*.json',
            '*.toml',
            'nanobrain',  # Core framework
            'demos',      # Demo implementations
            'config',     # Configuration files
            'docs',       # Documentation
            'tests',      # Test files (source, not cache)
            'scripts',    # Utility scripts
            'containers', # Container definitions
            'envs',       # Environment definitions
            'examples',   # Example code
            '.kiro'       # Kiro specs
        ]
    
    def _should_skip_directory(self, dir_path: Path) -> bool:
        """Check if directory should be skipped for performance."""
        try:
            rel_path = dir_path.relative_to(self.repo_root)
            rel_path_str = str(rel_path)
            
            # Skip if any part of the path is in exclude_dirs
            for part in rel_path.parts:
                if part in self.exclude_dirs:
                    return True
            
            # Skip if the full relative path matches any exclude pattern
            for exclude_pattern in self.exclude_dirs:
                if rel_path_str.startswith(exclude_pattern):
                    return True
                if fnmatch.fnmatch(rel_path_str, exclude_pattern):
                    return True
            
            return False
        except ValueError:
            return True
    
    def scan_temporary_files(self) -> List[TemporaryFile]:
        """
        Scan the repository for temporary files.
        
        Returns:
            List of TemporaryFile objects representing files to be cleaned
        """
        temporary_files = []
        
        self.logger.info(f"Scanning for temporary files in {self.repo_root}")
        
        # Scan for each type of temporary file
        temporary_files.extend(self._scan_pbs_outputs())
        temporary_files.extend(self._scan_python_cache())
        temporary_files.extend(self._scan_build_artifacts())
        temporary_files.extend(self._scan_log_files())
        
        self.logger.info(f"Found {len(temporary_files)} temporary files")
        return temporary_files
    
    def remove_pbs_outputs(self) -> CleanupResult:
        """
        Remove PBS job output files.
        
        Returns:
            CleanupResult with details of the cleanup operation
        """
        return self._remove_files_by_type('pbs_output')
    
    def remove_python_cache(self) -> CleanupResult:
        """
        Remove Python cache files and directories.
        
        Returns:
            CleanupResult with details of the cleanup operation
        """
        return self._remove_files_by_type('python_cache')
    
    def remove_build_artifacts(self) -> CleanupResult:
        """
        Remove build artifacts and distribution files.
        
        Returns:
            CleanupResult with details of the cleanup operation
        """
        return self._remove_files_by_type('build_artifact')
    
    def remove_log_files(self) -> CleanupResult:
        """
        Remove log files and execution directories.
        
        Returns:
            CleanupResult with details of the cleanup operation
        """
        return self._remove_files_by_type('log')
    
    def update_gitignore(self) -> bool:
        """
        Update .gitignore to prevent future temporary file commits.
        
        Returns:
            True if .gitignore was updated successfully, False otherwise
        """
        gitignore_path = self.repo_root / '.gitignore'
        
        # Standard gitignore entries for temporary files
        gitignore_entries = [
            '',
            '# Nanobrain Cleanup - Temporary Files',
            '# Added by cleanup system to prevent future commits',
            '',
            '# PBS job outputs',
            '*.err',
            '*.out',
            '*.e[0-9]*',
            '*.o[0-9]*',
            'STDIN.e*',
            'STDIN.o*',
            '*.sh.e*',
            '*.sh.o*',
            '',
            '# Python cache',
            '__pycache__/',
            '*.pyc',
            '*.pyo',
            '*.pyd',
            '.pytest_cache/',
            '.hypothesis/',
            '.coverage',
            'htmlcov/',
            '.tox/',
            '.nox/',
            '',
            '# Build artifacts',
            'build/',
            'dist/',
            '*.egg-info/',
            '*.egg',
            '.eggs/',
            'wheels/',
            'sdist/',
            '',
            '# Log files',
            'logs/',
            'runinfo/',
            'executed_workflows/',
            'cmd_parsl.*',
            'parsl.*',
            '*.log',
            'test-results/',
            ''
        ]
        
        try:
            # Read existing gitignore if it exists
            existing_content = ""
            if gitignore_path.exists():
                existing_content = gitignore_path.read_text()
            
            # Check if our entries are already present
            if "# Nanobrain Cleanup - Temporary Files" in existing_content:
                self.logger.info(".gitignore already contains cleanup entries")
                return True
            
            # Append our entries
            with open(gitignore_path, 'a') as f:
                f.write('\n'.join(gitignore_entries))
            
            self.logger.info(f"Updated .gitignore with temporary file patterns")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update .gitignore: {str(e)}")
            return False
    
    def _scan_pbs_outputs(self) -> List[TemporaryFile]:
        """Scan for PBS job output files."""
        files = []
        
        # Use basic patterns first
        basic_patterns = ['*.err', '*.out', 'STDIN.e*', 'STDIN.o*']
        for pattern in basic_patterns:
            files.extend(self._find_files_by_pattern(pattern, 'pbs_output'))
        
        # Then do a custom scan for PBS files with job IDs
        files.extend(self._find_pbs_job_files())
        
        return files
    
    def _find_pbs_job_files(self) -> List[TemporaryFile]:
        """Find PBS job files with specific patterns."""
        files = []
        
        # Walk the directory tree manually for better control
        for root, dirs, filenames in os.walk(self.repo_root):
            root_path = Path(root)
            
            # Skip excluded directories
            if self._should_skip_directory(root_path):
                dirs.clear()  # Don't recurse into subdirectories
                continue
            
            # Filter out excluded subdirectories from further traversal
            dirs[:] = [d for d in dirs if d not in self.exclude_dirs]
            
            # Check files in current directory for PBS patterns
            for filename in filenames:
                file_path = root_path / filename
                
                # Check for PBS job files with specific patterns
                if self._is_pbs_job_file(filename):
                    if not self._should_preserve(file_path):
                        files.append(self._create_temporary_file(file_path, 'pbs_output'))
        
        return files
    
    def _is_pbs_job_file(self, filename: str) -> bool:
        """Check if a filename matches PBS job file patterns."""
        # PBS files typically end with .e<digits> or .o<digits> where digits are job IDs
        # But we need to be careful not to match CSS files or other files with similar patterns
        
        # Pattern 1: filename.e<digits> or filename.o<digits> (but not CSS/JS files)
        if re.match(r'^[^.]+\.[eo]\d+$', filename):
            return True
        
        # Pattern 2: script.sh.e<digits> or script.sh.o<digits>
        if re.match(r'^.*\.sh\.[eo]\d+$', filename):
            return True
        
        # Pattern 3: STDIN files
        if filename.startswith('STDIN.') and ('.e' in filename or '.o' in filename):
            return True
        
        return False
    
    def _scan_python_cache(self) -> List[TemporaryFile]:
        """Scan for Python cache files and directories."""
        files = []
        for pattern in self.python_cache_patterns:
            files.extend(self._find_files_by_pattern(pattern, 'python_cache'))
        return files
    
    def _scan_build_artifacts(self) -> List[TemporaryFile]:
        """Scan for build artifacts."""
        files = []
        for pattern in self.build_artifact_patterns:
            files.extend(self._find_files_by_pattern(pattern, 'build_artifact'))
        return files
    
    def _scan_log_files(self) -> List[TemporaryFile]:
        """Scan for log files and execution directories."""
        files = []
        for pattern in self.log_patterns:
            files.extend(self._find_files_by_pattern(pattern, 'log'))
        return files
    
    def _find_files_by_pattern(self, pattern: str, file_type: str) -> List[TemporaryFile]:
        """
        Find files matching a specific pattern.
        
        Args:
            pattern: File pattern to match
            file_type: Type of temporary file
            
        Returns:
            List of TemporaryFile objects
        """
        files = []
        
        try:
            # Handle directory patterns vs file patterns
            if '*' in pattern or '?' in pattern:
                # Glob pattern - use optimized search
                files.extend(self._find_files_optimized(pattern, file_type))
            else:
                # Exact directory/file name - use optimized search
                files.extend(self._find_files_optimized(pattern, file_type))
                        
        except Exception as e:
            self.logger.warning(f"Error scanning pattern {pattern}: {str(e)}")
        
        return files
    
    def _find_files_optimized(self, pattern: str, file_type: str) -> List[TemporaryFile]:
        """
        Optimized file finding that excludes large directories.
        
        Args:
            pattern: File pattern to match
            file_type: Type of temporary file
            
        Returns:
            List of TemporaryFile objects
        """
        files = []
        
        # Walk the directory tree manually for better control
        for root, dirs, filenames in os.walk(self.repo_root):
            root_path = Path(root)
            
            # Skip excluded directories
            if self._should_skip_directory(root_path):
                dirs.clear()  # Don't recurse into subdirectories
                continue
            
            # Filter out excluded subdirectories from further traversal
            dirs[:] = [d for d in dirs if d not in self.exclude_dirs]
            
            # Check files in current directory
            for filename in filenames:
                file_path = root_path / filename
                
                # Check if file matches pattern
                if fnmatch.fnmatch(filename, pattern) or fnmatch.fnmatch(str(file_path.name), pattern):
                    if not self._should_preserve(file_path):
                        files.append(self._create_temporary_file(file_path, file_type))
            
            # Check directories if pattern might match directories
            # Skip directory checking only for simple file extension patterns like *.py, *.pyc
            # But allow patterns like *.egg-info which are directory patterns
            is_simple_file_extension = (pattern.startswith('*.') and 
                                      pattern.count('.') == 1 and 
                                      len(pattern.split('.')[-1]) <= 4 and
                                      pattern.split('.')[-1].isalpha())
            
            if not is_simple_file_extension:
                for dirname in dirs:
                    dir_path = root_path / dirname
                    if fnmatch.fnmatch(dirname, pattern):
                        if not self._should_preserve(dir_path):
                            files.append(self._create_temporary_file(dir_path, file_type))
        
        return files
    
    def _should_preserve(self, path: Path) -> bool:
        """
        Check if a file or directory should be preserved.
        
        Args:
            path: Path to check
            
        Returns:
            True if the file should be preserved, False otherwise
        """
        # Get relative path from repo root
        try:
            rel_path = path.relative_to(self.repo_root)
        except ValueError:
            # Path is not under repo root
            return True
        
        # First check: if it's clearly a temporary file/directory, don't preserve
        temp_names = ['__pycache__', '.pytest_cache', '.hypothesis', 'build', 'dist', 
                     'logs', 'runinfo', 'executed_workflows', '.coverage', 'htmlcov',
                     '.tox', '.nox', '.eggs']
        
        if path.name in temp_names:
            return False
        
        # Check for temporary file extensions
        temp_extensions = ['.pyc', '.pyo', '.pyd', '.err', '.out', '.log']
        if path.suffix in temp_extensions:
            return False
        
        # Check for PBS job files with numbers
        if self._is_pbs_job_file(path.name):
            return False
        
        # Check for parsl files
        if path.name.startswith('cmd_parsl.') or path.name.startswith('parsl.'):
            return False
        
        # Check for egg-info directories
        if path.name.endswith('.egg-info') or path.name.endswith('.egg'):
            return False
        
        # Check if it's in a temporary directory path
        temp_path_parts = ['__pycache__', '.pytest_cache', '.hypothesis', 'build', 'dist', 
                          'logs', 'runinfo', 'executed_workflows', '.coverage', 'htmlcov',
                          '.tox', '.nox', '.eggs']
        
        for part in rel_path.parts:
            if part in temp_path_parts:
                return False
        
        # Check against preserve patterns for directories and important files
        for preserve_pattern in self.preserve_patterns:
            # Check if any part of the path matches preserve pattern
            path_parts = rel_path.parts
            
            # For directory preservation, check if the first part matches
            if len(path_parts) > 0 and fnmatch.fnmatch(path_parts[0], preserve_pattern):
                # But still don't preserve temp directories even in preserved directories
                if path.name in temp_names:
                    return False
                # And don't preserve files in temp subdirectories
                for part in path_parts:
                    if part in temp_path_parts:
                        return False
                return True
            
            # Full path pattern match
            if fnmatch.fnmatch(str(rel_path), preserve_pattern):
                # But still don't preserve temp files/dirs
                if path.name in temp_names:
                    return False
                for part in path_parts:
                    if part in temp_path_parts:
                        return False
                return True
        
        # Final case: preserve source files in any directory (if not already excluded above)
        if path.suffix in ['.py', '.md', '.rst', '.yml', '.yaml', '.json', '.toml', '.txt']:
            # But not if they're clearly temporary by name
            if any(temp_part in path.name.lower() for temp_part in ['temp', 'tmp', 'cache', 'log']):
                return False
            # And not if they're in clearly temporary directories
            for part in rel_path.parts:
                if part in temp_path_parts:
                    return False
            return True
        
        return False
    
    def _create_temporary_file(self, path: Path, file_type: str) -> TemporaryFile:
        """
        Create a TemporaryFile object from a path.
        
        Args:
            path: Path to the file or directory
            file_type: Type of temporary file
            
        Returns:
            TemporaryFile object
        """
        try:
            if path.is_file():
                size = path.stat().st_size
                modified = datetime.fromtimestamp(path.stat().st_mtime)
            elif path.is_dir():
                # Calculate directory size
                size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
                modified = datetime.fromtimestamp(path.stat().st_mtime)
            else:
                size = 0
                modified = datetime.now()
                
            return TemporaryFile(
                path=str(path),
                file_type=file_type,
                size_bytes=size,
                last_modified=modified
            )
        except Exception as e:
            self.logger.warning(f"Error getting file info for {path}: {str(e)}")
            return TemporaryFile(
                path=str(path),
                file_type=file_type,
                size_bytes=0,
                last_modified=datetime.now()
            )
    
    def _remove_files_by_type(self, file_type: str) -> CleanupResult:
        """
        Remove all temporary files of a specific type.
        
        Args:
            file_type: Type of files to remove
            
        Returns:
            CleanupResult with operation details
        """
        # Scan for files of this type
        temp_files = [f for f in self.scan_temporary_files() if f.file_type == file_type]
        
        files_removed = []
        bytes_freed = 0
        errors = []
        
        for temp_file in temp_files:
            try:
                path = Path(temp_file.path)
                
                if not path.exists():
                    continue
                
                # Double-check preservation rules
                if self._should_preserve(path):
                    self.logger.info(f"Preserving {path} (matches preserve pattern)")
                    continue
                
                # Remove file or directory
                if path.is_file():
                    path.unlink()
                    files_removed.append(str(path))
                    bytes_freed += temp_file.size_bytes
                    self.logger.debug(f"Removed file: {path}")
                elif path.is_dir():
                    shutil.rmtree(path)
                    files_removed.append(str(path))
                    bytes_freed += temp_file.size_bytes
                    self.logger.debug(f"Removed directory: {path}")
                    
            except PermissionError as e:
                error_msg = f"Permission denied removing {temp_file.path}: {str(e)}"
                errors.append(error_msg)
                self.logger.warning(error_msg)
            except FileNotFoundError:
                # File was already removed, not an error
                pass
            except Exception as e:
                error_msg = f"Error removing {temp_file.path}: {str(e)}"
                errors.append(error_msg)
                self.logger.error(error_msg)
        
        success = len(errors) == 0
        
        self.logger.info(f"Cleanup of {file_type} files: {len(files_removed)} files removed, "
                        f"{bytes_freed} bytes freed, {len(errors)} errors")
        
        return CleanupResult(
            files_removed=files_removed,
            bytes_freed=bytes_freed,
            errors=errors,
            success=success
        )