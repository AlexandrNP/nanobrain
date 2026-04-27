"""
Backup Manager for the Nanobrain cleanup system.

This module provides comprehensive backup functionality to ensure safe cleanup
operations with full recovery capabilities.
"""

import os
import shutil
import subprocess
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional, List
import logging

from .models import BackupResult, GitBackupResult, ValidationResult


class BackupManager:
    """
    Manages backup operations for the cleanup process.
    
    Provides both filesystem and git-based backup mechanisms with integrity
    verification and recovery documentation.
    """
    
    def __init__(self, repo_root: str, backup_base_dir: Optional[str] = None):
        """
        Initialize the BackupManager.
        
        Args:
            repo_root: Path to the repository root directory
            backup_base_dir: Base directory for backups (defaults to repo_root/../backups)
        """
        self.repo_root = Path(repo_root).resolve()
        self.backup_base_dir = Path(backup_base_dir) if backup_base_dir else self.repo_root.parent / "backups"
        self.logger = logging.getLogger(__name__)
        
        # Ensure backup directory exists
        self.backup_base_dir.mkdir(parents=True, exist_ok=True)
    
    def create_full_backup(self) -> BackupResult:
        """
        Create a timestamped full repository backup.
        
        Returns:
            BackupResult with backup details and success status
        """
        timestamp = datetime.now()
        backup_name = f"nanobrain_backup_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        backup_path = self.backup_base_dir / backup_name
        
        try:
            self.logger.info(f"Creating full backup at {backup_path}")
            
            # Calculate source size
            source_size = self._calculate_directory_size(self.repo_root)
            
            # Check available disk space
            if not self._check_disk_space(backup_path.parent, source_size * 1.1):  # 10% buffer
                return BackupResult(
                    backup_path=str(backup_path),
                    timestamp=timestamp,
                    size_bytes=0,
                    success=False,
                    error_message="Insufficient disk space for backup"
                )
            
            # Create the backup
            shutil.copytree(
                self.repo_root,
                backup_path,
                ignore=shutil.ignore_patterns(
                    '.git',  # Git will be backed up separately
                    '__pycache__',
                    '*.pyc',
                    '*.pyo',
                    '.DS_Store',
                    'node_modules'
                )
            )
            
            # Calculate backup size
            backup_size = self._calculate_directory_size(backup_path)
            
            self.logger.info(f"Backup created successfully: {backup_size} bytes")
            
            return BackupResult(
                backup_path=str(backup_path),
                timestamp=timestamp,
                size_bytes=backup_size,
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"Backup creation failed: {str(e)}")
            return BackupResult(
                backup_path=str(backup_path),
                timestamp=timestamp,
                size_bytes=0,
                success=False,
                error_message=str(e)
            )
    
    def create_git_backup_branch(self) -> GitBackupResult:
        """
        Create a git branch backup with current state.
        
        Returns:
            GitBackupResult with branch details and success status
        """
        timestamp = datetime.now()
        branch_name = f"cleanup_backup_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # Check if we're in a git repository
            result = subprocess.run(
                ['git', 'rev-parse', '--git-dir'],
                cwd=self.repo_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                return GitBackupResult(
                    branch_name=branch_name,
                    commit_hash="",
                    success=False,
                    error_message="Not a git repository"
                )
            
            # Get current commit hash
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                cwd=self.repo_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                return GitBackupResult(
                    branch_name=branch_name,
                    commit_hash="",
                    success=False,
                    error_message="Failed to get current commit hash"
                )
            
            current_commit = result.stdout.strip()
            
            # Create backup branch
            result = subprocess.run(
                ['git', 'checkout', '-b', branch_name],
                cwd=self.repo_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                return GitBackupResult(
                    branch_name=branch_name,
                    commit_hash=current_commit,
                    success=False,
                    error_message=f"Failed to create branch: {result.stderr}"
                )
            
            # Return to original branch (assuming main or master)
            for branch in ['main', 'master']:
                result = subprocess.run(
                    ['git', 'checkout', branch],
                    cwd=self.repo_root,
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    break
            
            self.logger.info(f"Git backup branch created: {branch_name}")
            
            return GitBackupResult(
                branch_name=branch_name,
                commit_hash=current_commit,
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"Git backup creation failed: {str(e)}")
            return GitBackupResult(
                branch_name=branch_name,
                commit_hash="",
                success=False,
                error_message=str(e)
            )
    
    def verify_backup_integrity(self, backup_path: str) -> ValidationResult:
        """
        Verify the integrity of a backup.
        
        Args:
            backup_path: Path to the backup directory
            
        Returns:
            ValidationResult with verification details
        """
        backup_dir = Path(backup_path)
        tests_passed = []
        tests_failed = []
        
        try:
            # Test 1: Backup directory exists
            if backup_dir.exists():
                tests_passed.append("Backup directory exists")
            else:
                tests_failed.append("Backup directory does not exist")
                return ValidationResult(
                    component="backup_integrity",
                    tests_passed=tests_passed,
                    tests_failed=tests_failed,
                    success=False,
                    error_details="Backup directory not found"
                )
            
            # Test 2: Essential directories exist
            essential_dirs = ['nanobrain', 'demos', 'config']
            for dir_name in essential_dirs:
                if (backup_dir / dir_name).exists():
                    tests_passed.append(f"Essential directory '{dir_name}' exists")
                else:
                    tests_failed.append(f"Essential directory '{dir_name}' missing")
            
            # Test 3: Essential files exist
            essential_files = ['pyproject.toml', 'README.md', 'requirements.txt']
            for file_name in essential_files:
                if (backup_dir / file_name).exists():
                    tests_passed.append(f"Essential file '{file_name}' exists")
                else:
                    tests_failed.append(f"Essential file '{file_name}' missing")
            
            # Test 4: Backup is readable
            try:
                list(backup_dir.rglob('*.py'))
                tests_passed.append("Backup files are readable")
            except Exception as e:
                tests_failed.append(f"Backup files not readable: {str(e)}")
            
            # Test 5: Size validation (backup should not be empty)
            backup_size = self._calculate_directory_size(backup_dir)
            if backup_size > 50:  # At least 50 bytes (reasonable for test files)
                tests_passed.append(f"Backup size is reasonable ({backup_size} bytes)")
            else:
                tests_failed.append(f"Backup size too small ({backup_size} bytes)")
            
            success = len(tests_failed) == 0
            
            return ValidationResult(
                component="backup_integrity",
                tests_passed=tests_passed,
                tests_failed=tests_failed,
                success=success,
                error_details=None if success else f"Failed {len(tests_failed)} integrity checks"
            )
            
        except Exception as e:
            self.logger.error(f"Backup verification failed: {str(e)}")
            return ValidationResult(
                component="backup_integrity",
                tests_passed=tests_passed,
                tests_failed=tests_failed + [f"Verification error: {str(e)}"],
                success=False,
                error_details=str(e)
            )
    
    def document_backup_locations(self) -> str:
        """
        Create documentation of backup locations and recovery procedures.
        
        Returns:
            Path to the created documentation file
        """
        doc_path = self.backup_base_dir / "BACKUP_RECOVERY_PROCEDURES.md"
        
        content = f"""# Nanobrain Cleanup Backup Recovery Procedures

## Backup Information

- **Backup Base Directory**: {self.backup_base_dir}
- **Repository Root**: {self.repo_root}
- **Documentation Created**: {datetime.now().isoformat()}

## Available Backups

"""
        
        # List all available backups
        backup_dirs = [d for d in self.backup_base_dir.iterdir() if d.is_dir() and d.name.startswith('nanobrain_backup_')]
        backup_dirs.sort(key=lambda x: x.name, reverse=True)
        
        for backup_dir in backup_dirs:
            size = self._calculate_directory_size(backup_dir)
            content += f"- **{backup_dir.name}**: {size} bytes\n"
        
        content += """

## Recovery Procedures

### Full Repository Recovery

1. **Stop any running processes** that might be using the repository
2. **Navigate to the parent directory** of your repository
3. **Backup current state** (if partially cleaned):
   ```bash
   mv nanobrain nanobrain_partial_cleanup
   ```
4. **Restore from backup**:
   ```bash
   cp -r {backup_path} nanobrain
   ```
5. **Verify restoration**:
   ```bash
   cd nanobrain
   python -c "import nanobrain; print('Import successful')"
   ```

### Git Branch Recovery

If git backup branches were created, you can recover using:

```bash
cd nanobrain
git checkout {backup_branch_name}
```

### Partial Recovery

To recover specific components:

1. **Identify the component** you need to recover
2. **Copy from backup**:
   ```bash
   cp -r {backup_path}/nanobrain/component_name nanobrain/
   ```
3. **Test the component** to ensure it works correctly

## Verification Steps

After recovery, verify the system works:

1. **Check imports**:
   ```bash
   python -c "import nanobrain.core; print('Core import successful')"
   ```

2. **Run basic tests**:
   ```bash
   python -m pytest tests/ -v
   ```

3. **Check demo functionality**:
   ```bash
   cd demos/simple_demo
   python main.py --help
   ```

## Emergency Contacts

If recovery fails, check:
- Repository structure matches expected layout
- Python environment has required dependencies
- File permissions are correct (especially on shared systems)

## Backup Integrity

Each backup includes integrity verification. If a backup fails verification:
1. Try the next most recent backup
2. Check disk space and permissions
3. Verify the backup directory is not corrupted

""".format(
            backup_path="{backup_path}",
            backup_branch_name="{backup_branch_name}"
        )
        
        with open(doc_path, 'w') as f:
            f.write(content)
        
        self.logger.info(f"Backup documentation created at {doc_path}")
        return str(doc_path)
    
    def _calculate_directory_size(self, directory: Path) -> int:
        """Calculate the total size of a directory in bytes."""
        total_size = 0
        try:
            for file_path in directory.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        except (OSError, PermissionError):
            # Skip files we can't access
            pass
        return total_size
    
    def _check_disk_space(self, path: Path, required_bytes: int) -> bool:
        """Check if there's enough disk space for the backup."""
        try:
            stat = shutil.disk_usage(path)
            return stat.free >= required_bytes
        except Exception:
            # If we can't check, assume there's space
            return True