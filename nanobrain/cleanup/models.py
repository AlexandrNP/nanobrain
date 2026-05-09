"""
Data models for the Nanobrain cleanup system.

This module defines all the data structures used throughout the cleanup process,
including results, configurations, and validation models.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any


@dataclass
class BackupResult:
    """Result of a backup operation."""
    backup_path: str
    timestamp: datetime
    size_bytes: int
    success: bool
    error_message: Optional[str] = None


@dataclass
class GitBackupResult:
    """Result of a git branch backup operation."""
    branch_name: str
    commit_hash: str
    success: bool
    error_message: Optional[str] = None


@dataclass
class TemporaryFile:
    """Represents a temporary file identified for cleanup."""
    path: str
    file_type: str  # 'pbs_output', 'python_cache', 'build_artifact', 'log'
    size_bytes: int
    last_modified: datetime


@dataclass
class CleanupResult:
    """Result of a cleanup operation."""
    success: bool
    files_removed: List[str] = None
    bytes_freed: int = 0
    errors: List[str] = None
    message: str = ""
    files_processed: int = 0
    details: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Initialize lists if None."""
        if self.files_removed is None:
            self.files_removed = []
        if self.errors is None:
            self.errors = []


@dataclass
class DemoCategories:
    """Categorization of demo directories."""
    keep: List[str]
    archive: List[str]
    delete: List[str]


@dataclass
class ArchiveResult:
    """Result of demo archiving operation."""
    archived_demos: List[str]
    archive_path: str
    index_created: bool
    success: bool
    errors: List[str] = None
    
    def __post_init__(self):
        """Initialize errors list if None."""
        if self.errors is None:
            self.errors = []


@dataclass
class ValidationResult:
    """Result of a validation operation."""
    component: str
    tests_passed: List[str]
    tests_failed: List[str]
    success: bool
    error_details: Optional[str] = None


@dataclass
class CircularDependency:
    """Represents a circular import dependency."""
    modules: List[str]
    import_chain: List[str]


@dataclass
class MissingInitFile:
    """Represents a missing __init__.py file."""
    directory: str
    required_exports: List[str]


@dataclass
class ResolutionResult:
    """Result of import resolution operation."""
    resolved_dependencies: List[CircularDependency]
    created_init_files: List[MissingInitFile]
    fixed_imports: List[str]
    success: bool
    errors: List[str]


@dataclass
class HardcodedPath:
    """Represents a hardcoded path in configuration."""
    file_path: str
    line_number: int
    original_path: str
    suggested_replacement: str


@dataclass
class ConfigValidationResult:
    """Result of configuration validation."""
    file_path: str
    is_valid: bool
    schema_errors: List[str]
    warnings: List[str]


@dataclass
class PhaseResult:
    """Result of a cleanup phase execution."""
    phase_name: str
    success: bool
    duration_seconds: float
    operations_completed: List[str]
    errors: List[str]
    rollback_available: bool


@dataclass
class FixResult:
    """Result of a demo fix operation."""
    demo_name: str
    fixes_applied: List[str]
    files_modified: List[str]
    tests_consolidated: int
    paths_fixed: int
    success: bool
    errors: List[str] = None
    
    def __post_init__(self):
        """Initialize errors list if None."""
        if self.errors is None:
            self.errors = []


@dataclass
class ConsolidationResult:
    """Result of test file consolidation."""
    original_test_count: int
    consolidated_test_count: int
    files_removed: List[str]
    files_created: List[str]
    success: bool
    errors: List[str] = None
    
    def __post_init__(self):
        """Initialize errors list if None."""
        if self.errors is None:
            self.errors = []


@dataclass
class PathFixResult:
    """Result of hardcoded path fixing."""
    files_processed: int
    paths_fixed: int
    config_files_created: List[str]
    success: bool
    errors: List[str] = None
    
    def __post_init__(self):
        """Initialize errors list if None."""
        if self.errors is None:
            self.errors = []


@dataclass
class CleanupSummary:
    """Summary of the entire cleanup process."""
    start_time: datetime
    end_time: datetime
    phases_completed: List[PhaseResult]
    total_files_removed: int
    total_bytes_freed: int
    demos_processed: int
    imports_fixed: int
    configs_standardized: int
    success: bool
    rollback_branch: Optional[str] = None