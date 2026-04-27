"""
Nanobrain Cleanup System

A comprehensive cleanup system for the Nanobrain HPC AI framework repository.
This system provides automated cleanup, validation, and reorganization capabilities.

Main Components:
- CleanupOrchestrator: Main orchestrator for coordinating all cleanup phases
- BackupManager: Creates and manages repository backups
- StructureManager: Manages repository structure reorganization
- DemoFixer: Fixes and standardizes demo implementations
- ConfigManager: Manages configuration file standardization
- ImportResolver: Resolves import issues and circular dependencies
- DocManager: Consolidates and manages documentation
- DependencyManager: Manages project dependencies
- Validator: Comprehensive system validation

Usage:
    from nanobrain.cleanup import CleanupOrchestrator
    
    # Create orchestrator
    orchestrator = CleanupOrchestrator('/path/to/repo', dry_run=True)
    
    # Run cleanup
    summary = orchestrator.execute_cleanup()
    
    # Check results
    if summary.success:
        print("Cleanup completed successfully!")
    else:
        print("Cleanup failed:", summary.phases_completed)

Command Line Usage:
    python -m nanobrain.cleanup.orchestrator --dry-run
    python -m nanobrain.cleanup.orchestrator --phases backup temp_cleanup
    python -m nanobrain.cleanup.orchestrator --list-phases
"""

from .orchestrator import CleanupOrchestrator, CleanupPhase
from .backup_manager import BackupManager
from .structure_manager import StructureManager
from .demo_fixer import DemoFixer
from .config_manager import ConfigManager
from .import_resolver import ImportResolver
from .doc_manager import DocManager
from .dependency_manager import DependencyManager
from .validator import Validator
from .models import (
    CleanupResult, ValidationResult, PhaseResult, CleanupSummary,
    BackupResult, GitBackupResult, TemporaryFile, DemoCategories,
    ArchiveResult, CircularDependency, MissingInitFile, ResolutionResult,
    HardcodedPath, ConfigValidationResult, FixResult, ConsolidationResult,
    PathFixResult
)

__version__ = "0.1.0"

__all__ = [
    # Main orchestrator
    'CleanupOrchestrator',
    'CleanupPhase',
    
    # Manager classes
    'BackupManager',
    'StructureManager', 
    'DemoFixer',
    'ConfigManager',
    'ImportResolver',
    'DocManager',
    'DependencyManager',
    'Validator',
    
    # Data models
    'CleanupResult',
    'ValidationResult',
    'PhaseResult',
    'CleanupSummary',
    'BackupResult',
    'GitBackupResult',
    'TemporaryFile',
    'DemoCategories',
    'ArchiveResult',
    'CircularDependency',
    'MissingInitFile',
    'ResolutionResult',
    'HardcodedPath',
    'ConfigValidationResult',
    'FixResult',
    'ConsolidationResult',
    'PathFixResult',
]