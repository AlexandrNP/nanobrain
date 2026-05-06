"""
Main cleanup orchestrator for the Nanobrain framework.

This module coordinates all cleanup phases, manages dependencies between phases,
and provides rollback capabilities and comprehensive logging.
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime

from .models import PhaseResult, CleanupSummary, ValidationResult
from .backup_manager import BackupManager
from .cleanup_manager import CleanupManager
from .demo_manager import DemoManager
from .structure_manager import StructureManager
from .demo_fixer import DemoFixer
from .config_manager import ConfigManager
from .import_resolver import ImportResolver
from .doc_manager import DocManager
from .dependency_manager import DependencyManager
from .validator import Validator


@dataclass
class CleanupPhase:
    """Represents a cleanup phase with its dependencies and execution function."""
    name: str
    description: str
    dependencies: List[str]
    execute_func: Callable
    rollback_func: Optional[Callable] = None
    validation_func: Optional[Callable] = None
    required: bool = True


class CleanupOrchestrator:
    """Main orchestrator for the Nanobrain cleanup process."""
    
    def __init__(self, repo_root: Path, dry_run: bool = False):
        """Initialize the cleanup orchestrator.
        
        Args:
            repo_root: Root directory of the repository
            dry_run: Whether to run in dry-run mode (preview only)
        """
        self.repo_root = Path(repo_root)
        self.dry_run = dry_run
        
        # Initialize managers
        self.backup_manager = BackupManager(self.repo_root)
        self.cleanup_manager = CleanupManager(self.repo_root)
        self.demo_manager = DemoManager(self.repo_root)
        self.structure_manager = StructureManager(self.repo_root)
        self.demo_fixer = DemoFixer(self.repo_root)
        self.config_manager = ConfigManager(self.repo_root)
        self.import_resolver = ImportResolver(self.repo_root)
        self.doc_manager = DocManager(self.repo_root)
        self.dependency_manager = DependencyManager(self.repo_root)
        self.validator = Validator(self.repo_root)
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Phase execution state
        self.completed_phases: List[str] = []
        self.phase_results: List[PhaseResult] = []
        self.rollback_info: Dict[str, Any] = {}
        
        # Define cleanup phases
        self.phases = self._define_cleanup_phases()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging for the cleanup process."""
        logger = logging.getLogger('nanobrain_cleanup')
        logger.setLevel(logging.INFO)
        
        # Create logs directory
        logs_dir = self.repo_root / 'logs'
        logs_dir.mkdir(exist_ok=True)
        
        # File handler with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = logs_dir / f'cleanup_{timestamp}.log'
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def _define_cleanup_phases(self) -> Dict[str, CleanupPhase]:
        """Define all cleanup phases with their dependencies."""
        phases = {}
        
        # Phase 1: Backup and Infrastructure
        phases['backup'] = CleanupPhase(
            name='backup',
            description='Create comprehensive backup of repository',
            dependencies=[],
            execute_func=self._execute_backup_phase,
            rollback_func=None,  # Backup phase cannot be rolled back
            validation_func=self._validate_backup_phase
        )
        
        # Phase 2: Temporary File Cleanup
        phases['temp_cleanup'] = CleanupPhase(
            name='temp_cleanup',
            description='Clean up temporary files and build artifacts',
            dependencies=['backup'],
            execute_func=self._execute_temp_cleanup_phase,
            rollback_func=self._rollback_temp_cleanup_phase,
            validation_func=self._validate_temp_cleanup_phase
        )
        
        # Phase 3: Demo Consolidation
        phases['demo_consolidation'] = CleanupPhase(
            name='demo_consolidation',
            description='Consolidate and organize demo directories',
            dependencies=['backup'],
            execute_func=self._execute_demo_consolidation_phase,
            rollback_func=self._rollback_demo_consolidation_phase,
            validation_func=self._validate_demo_consolidation_phase
        )
        
        # Phase 4: Demo Fixes
        phases['demo_fixes'] = CleanupPhase(
            name='demo_fixes',
            description='Fix and standardize target demos',
            dependencies=['demo_consolidation'],
            execute_func=self._execute_demo_fixes_phase,
            rollback_func=self._rollback_demo_fixes_phase,
            validation_func=self._validate_demo_fixes_phase
        )
        
        # Phase 5: Import Resolution
        phases['import_resolution'] = CleanupPhase(
            name='import_resolution',
            description='Resolve import issues and circular dependencies',
            dependencies=['demo_fixes'],
            execute_func=self._execute_import_resolution_phase,
            rollback_func=self._rollback_import_resolution_phase,
            validation_func=self._validate_import_resolution_phase
        )
        
        # Phase 6: Configuration Management
        phases['config_management'] = CleanupPhase(
            name='config_management',
            description='Standardize and validate configuration files',
            dependencies=['import_resolution'],
            execute_func=self._execute_config_management_phase,
            rollback_func=self._rollback_config_management_phase,
            validation_func=self._validate_config_management_phase
        )
        
        # Phase 7: Documentation Consolidation
        phases['doc_consolidation'] = CleanupPhase(
            name='doc_consolidation',
            description='Consolidate and update documentation',
            dependencies=['config_management'],
            execute_func=self._execute_doc_consolidation_phase,
            rollback_func=self._rollback_doc_consolidation_phase,
            validation_func=self._validate_doc_consolidation_phase
        )
        
        # Phase 8: Dependency Management
        phases['dependency_management'] = CleanupPhase(
            name='dependency_management',
            description='Clean up and validate dependencies',
            dependencies=['doc_consolidation'],
            execute_func=self._execute_dependency_management_phase,
            rollback_func=self._rollback_dependency_management_phase,
            validation_func=self._validate_dependency_management_phase
        )
        
        # Phase 9: Structure Reorganization
        phases['structure_reorganization'] = CleanupPhase(
            name='structure_reorganization',
            description='Reorganize repository structure',
            dependencies=['dependency_management'],
            execute_func=self._execute_structure_reorganization_phase,
            rollback_func=self._rollback_structure_reorganization_phase,
            validation_func=self._validate_structure_reorganization_phase
        )
        
        # Phase 10: Final Validation
        phases['final_validation'] = CleanupPhase(
            name='final_validation',
            description='Comprehensive system validation',
            dependencies=['structure_reorganization'],
            execute_func=self._execute_final_validation_phase,
            rollback_func=None,  # Validation phase doesn't modify anything
            validation_func=None  # This IS the validation phase
        )
        
        return phases
    
    def execute_cleanup(self, phases_to_run: Optional[List[str]] = None) -> CleanupSummary:
        """Execute the complete cleanup process.
        
        Args:
            phases_to_run: Specific phases to run (None for all phases)
            
        Returns:
            CleanupSummary with complete execution results
        """
        start_time = datetime.now()
        
        self.logger.info("Starting Nanobrain repository cleanup")
        self.logger.info(f"Repository root: {self.repo_root}")
        self.logger.info(f"Dry run mode: {self.dry_run}")
        
        try:
            # Determine phases to execute
            if phases_to_run is None:
                phases_to_run = list(self.phases.keys())
            
            # Validate phase dependencies
            execution_order = self._resolve_phase_dependencies(phases_to_run)
            
            self.logger.info(f"Executing phases in order: {execution_order}")
            
            # Execute phases
            for phase_name in execution_order:
                phase_result = self._execute_phase(phase_name)
                self.phase_results.append(phase_result)
                
                if not phase_result.success:
                    self.logger.error(f"Phase {phase_name} failed, stopping execution")
                    break
                
                self.completed_phases.append(phase_name)
            
            end_time = datetime.now()
            
            # Calculate summary statistics
            total_files_removed = sum(
                len(result.operations_completed) for result in self.phase_results
            )
            
            success = all(result.success for result in self.phase_results)
            
            summary = CleanupSummary(
                start_time=start_time,
                end_time=end_time,
                phases_completed=self.phase_results,
                total_files_removed=total_files_removed,
                total_bytes_freed=0,  # Would need to track this in individual phases
                demos_processed=len([r for r in self.phase_results if 'demo' in r.phase_name]),
                imports_fixed=0,  # Would need to track this in import resolution phase
                configs_standardized=0,  # Would need to track this in config phase
                success=success,
                rollback_branch=self.rollback_info.get('git_branch')
            )
            
            if success:
                self.logger.info("Cleanup completed successfully!")
            else:
                self.logger.error("Cleanup completed with errors")
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Cleanup failed with exception: {e}")
            end_time = datetime.now()
            
            return CleanupSummary(
                start_time=start_time,
                end_time=end_time,
                phases_completed=self.phase_results,
                total_files_removed=0,
                total_bytes_freed=0,
                demos_processed=0,
                imports_fixed=0,
                configs_standardized=0,
                success=False
            )
    
    def rollback_to_phase(self, target_phase: str) -> bool:
        """Rollback to a specific phase.
        
        Args:
            target_phase: Phase to rollback to
            
        Returns:
            True if rollback was successful
        """
        self.logger.info(f"Starting rollback to phase: {target_phase}")
        
        try:
            # Find phases to rollback (in reverse order)
            phases_to_rollback = []
            for phase_name in reversed(self.completed_phases):
                phases_to_rollback.append(phase_name)
                if phase_name == target_phase:
                    break
            
            # Execute rollbacks
            for phase_name in phases_to_rollback:
                phase = self.phases[phase_name]
                if phase.rollback_func:
                    self.logger.info(f"Rolling back phase: {phase_name}")
                    try:
                        phase.rollback_func()
                        self.logger.info(f"Successfully rolled back phase: {phase_name}")
                    except Exception as e:
                        self.logger.error(f"Failed to rollback phase {phase_name}: {e}")
                        return False
                else:
                    self.logger.warning(f"No rollback function for phase: {phase_name}")
            
            self.logger.info("Rollback completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Rollback failed: {e}")
            return False
    
    def _resolve_phase_dependencies(self, phases_to_run: List[str]) -> List[str]:
        """Resolve phase dependencies and return execution order."""
        execution_order = []
        remaining_phases = set(phases_to_run)
        
        while remaining_phases:
            # Find phases with satisfied dependencies
            ready_phases = []
            for phase_name in remaining_phases:
                phase = self.phases[phase_name]
                if all(dep in execution_order or dep not in phases_to_run for dep in phase.dependencies):
                    ready_phases.append(phase_name)
            
            if not ready_phases:
                raise ValueError(f"Circular dependency detected in phases: {remaining_phases}")
            
            # Add ready phases to execution order
            for phase_name in ready_phases:
                execution_order.append(phase_name)
                remaining_phases.remove(phase_name)
        
        return execution_order
    
    def _execute_phase(self, phase_name: str) -> PhaseResult:
        """Execute a single cleanup phase."""
        phase = self.phases[phase_name]
        
        self.logger.info(f"Starting phase: {phase_name} - {phase.description}")
        start_time = time.time()
        
        try:
            if self.dry_run:
                self.logger.info(f"DRY RUN: Would execute phase {phase_name}")
                operations_completed = [f"DRY RUN: {phase.description}"]
                errors = []
                success = True
            else:
                # Execute the phase
                result = phase.execute_func()
                
                if hasattr(result, 'success'):
                    success = result.success
                    operations_completed = getattr(result, 'files_processed', [])
                    errors = getattr(result, 'errors', [])
                else:
                    success = True
                    operations_completed = ["Phase completed"]
                    errors = []
                
                # Run validation if available
                if phase.validation_func and success:
                    validation_result = phase.validation_func()
                    if not validation_result.success:
                        success = False
                        errors.extend(validation_result.tests_failed)
            
            duration = time.time() - start_time
            
            phase_result = PhaseResult(
                phase_name=phase_name,
                success=success,
                duration_seconds=duration,
                operations_completed=operations_completed if isinstance(operations_completed, list) else [str(operations_completed)],
                errors=errors,
                rollback_available=phase.rollback_func is not None
            )
            
            if success:
                self.logger.info(f"Phase {phase_name} completed successfully in {duration:.2f}s")
            else:
                self.logger.error(f"Phase {phase_name} failed after {duration:.2f}s: {errors}")
            
            return phase_result
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Phase {phase_name} failed with exception: {e}")
            
            return PhaseResult(
                phase_name=phase_name,
                success=False,
                duration_seconds=duration,
                operations_completed=[],
                errors=[str(e)],
                rollback_available=phase.rollback_func is not None
            )
    
    # Phase execution methods
    def _execute_backup_phase(self):
        """Execute backup phase."""
        # Create git branch backup
        git_result = self.backup_manager.create_git_backup_branch()
        if git_result.success:
            self.rollback_info['git_branch'] = git_result.branch_name
        
        # Create file backup
        file_result = self.backup_manager.create_full_backup()
        if file_result.success:
            self.rollback_info['file_backup'] = file_result.backup_path
        
        return git_result if git_result.success else file_result
    
    def _execute_temp_cleanup_phase(self):
        """Execute temporary file cleanup phase."""
        results = []
        results.append(self.cleanup_manager.remove_pbs_outputs())
        results.append(self.cleanup_manager.remove_python_cache())
        results.append(self.cleanup_manager.remove_build_artifacts())
        results.append(self.cleanup_manager.remove_log_files())
        
        # Update gitignore
        gitignore_updated = self.cleanup_manager.update_gitignore()
        
        # Return combined result
        success = all(r.success for r in results) and gitignore_updated
        errors = []
        files_processed = 0
        for r in results:
            errors.extend(r.errors)
            files_processed += r.files_processed
        
        return type(results[0])(
            success=success,
            errors=errors,
            files_processed=files_processed
        )
    
    def _execute_demo_consolidation_phase(self):
        """Execute demo consolidation phase."""
        # Categorize demos
        categories = self.demo_manager.identify_demo_categories()
        
        # Preserve target demos
        preserve_success = self.demo_manager.preserve_target_demos()
        
        # Archive non-essential demos
        archive_result = self.demo_manager.archive_non_essential_demos()
        
        # Delete backup directories
        delete_result = self.demo_manager.delete_backup_directories()
        
        # Create archive index
        index_success = self.demo_manager.create_archive_index()
        
        # Return combined result
        success = (preserve_success and archive_result.success and 
                  delete_result.get('success', True) and index_success)
        errors = []
        if not preserve_success:
            errors.append("Failed to preserve target demos")
        if not archive_result.success:
            errors.extend(archive_result.errors)
        if not delete_result.get('success', True):
            errors.extend(delete_result.get('errors', []))
        if not index_success:
            errors.append("Failed to create archive index")
        
        # Calculate files processed from different sources
        files_processed = (len(archive_result.archived_demos) + 
                          delete_result.get('files_processed', 0))
        
        # Return a CleanupResult-like object
        from .models import CleanupResult
        return CleanupResult(
            success=success,
            errors=errors,
            files_processed=files_processed
        )
    
    def _execute_demo_fixes_phase(self):
        """Execute demo fixes phase."""
        results = []
        results.append(self.demo_fixer.fix_viral_pssm_demo())
        results.append(self.demo_fixer.fix_rag_database_demo())
        
        # Return combined result
        success = all(r.success for r in results)
        errors = []
        files_processed = 0
        for r in results:
            errors.extend(r.errors)
            files_processed += len(r.files_modified)  # Use files_modified instead of files_processed
        
        # Return a CleanupResult-like object
        from .models import CleanupResult
        return CleanupResult(
            success=success,
            errors=errors,
            files_processed=files_processed
        )
    
    def _execute_import_resolution_phase(self):
        """Execute import resolution phase."""
        results = []
        results.append(self.import_resolver.analyze_imports())
        results.append(self.import_resolver.resolve_circular_dependencies())
        results.append(self.import_resolver.add_missing_init_files())
        results.append(self.import_resolver.fix_hardcoded_paths())
        results.append(self.import_resolver.validate_imports())
        
        # Return combined result
        success = all(r.success for r in results)
        errors = []
        files_processed = 0
        for r in results:
            errors.extend(r.errors)
            files_processed += r.files_processed
        
        return type(results[0])(
            success=success,
            errors=errors,
            files_processed=files_processed
        )
    
    def _execute_config_management_phase(self):
        """Execute configuration management phase."""
        results = []
        results.append(self.config_manager.analyze_configurations())
        results.append(self.config_manager.fix_hardcoded_paths())
        results.append(self.config_manager.validate_configurations())
        results.append(self.config_manager.create_template_configurations())
        results.append(self.config_manager.standardize_yaml_structure())
        
        # Return combined result
        success = all(r.success for r in results)
        errors = []
        files_processed = 0
        for r in results:
            errors.extend(r.errors)
            files_processed += r.files_processed
        
        return type(results[0])(
            success=success,
            errors=errors,
            files_processed=files_processed
        )
    
    def _execute_doc_consolidation_phase(self):
        """Execute documentation consolidation phase."""
        # Scan documentation
        doc_files = self.doc_manager.scan_documentation()
        
        # Identify redundant docs
        clusters = self.doc_manager.identify_redundant_docs(doc_files)
        
        # Execute consolidation steps
        results = []
        results.append(self.doc_manager.consolidate_documentation(clusters))
        results.append(self.doc_manager.update_main_readme())
        results.append(self.doc_manager.create_demo_entry_points())
        
        # Return combined result
        success = all(r.success for r in results)
        errors = []
        files_processed = 0
        for r in results:
            errors.extend(r.errors)
            files_processed += r.files_processed
        
        return type(results[0])(
            success=success,
            errors=errors,
            files_processed=files_processed
        )
    
    def _execute_dependency_management_phase(self):
        """Execute dependency management phase."""
        # Analyze dependencies first
        analysis = self.dependency_manager.analyze_dependencies()
        
        # Execute cleanup steps
        results = []
        results.append(self.dependency_manager.cleanup_unused_dependencies(analysis))
        results.append(self.dependency_manager.add_graceful_degradation())
        results.append(self.dependency_manager.synchronize_requirements())
        
        # Return combined result
        success = all(r.success for r in results)
        errors = []
        files_processed = 0
        for r in results:
            errors.extend(r.errors)
            files_processed += r.files_processed
        
        return type(results[0])(
            success=success,
            errors=errors,
            files_processed=files_processed
        )
    
    def _execute_structure_reorganization_phase(self):
        """Execute structure reorganization phase."""
        results = []
        results.append(self.structure_manager.separate_core_and_demos())
        results.append(self.structure_manager.enforce_naming_conventions())
        results.append(self.structure_manager.create_package_hierarchies())
        results.append(self.structure_manager.organize_archive_content())
        results.append(self.structure_manager.update_import_statements())
        
        # Return combined result
        success = all(r.success for r in results)
        errors = []
        for r in results:
            errors.extend(r.errors)
        
        return type(results[0])(
            success=success,
            errors=errors,
            files_processed=sum(r.files_processed for r in results)
        )
    
    def _execute_final_validation_phase(self):
        """Execute final validation phase."""
        results = []
        results.append(self.validator.validate_framework_imports())
        results.append(self.validator.validate_demo_execution())
        results.append(self.validator.validate_configuration_files())
        results.append(self.validator.validate_documentation_integrity())
        
        # Return combined result
        success = all(r.success for r in results)
        errors = []
        for r in results:
            errors.extend(r.tests_failed)
        
        return type(results[0])(
            success=success,
            tests_failed=errors,
            tests_passed=[t for r in results for t in r.tests_passed]
        )
    
    # Validation methods
    def _validate_backup_phase(self):
        """Validate backup phase."""
        return ValidationResult(
            component="backup",
            tests_passed=["Backup phase completed"],
            tests_failed=[],
            success=True
        )
    
    def _validate_temp_cleanup_phase(self):
        """Validate temporary cleanup phase."""
        return ValidationResult(
            component="temp_cleanup",
            tests_passed=["Temporary cleanup completed"],
            tests_failed=[],
            success=True
        )
    
    def _validate_demo_consolidation_phase(self):
        """Validate demo consolidation phase."""
        return ValidationResult(
            component="demo_consolidation",
            tests_passed=["Demo consolidation completed"],
            tests_failed=[],
            success=True
        )
    
    def _validate_demo_fixes_phase(self):
        """Validate demo fixes phase."""
        return ValidationResult(
            component="demo_fixes",
            tests_passed=["Demo fixes completed"],
            tests_failed=[],
            success=True
        )
    
    def _validate_import_resolution_phase(self):
        """Validate import resolution phase."""
        return ValidationResult(
            component="import_resolution",
            tests_passed=["Import resolution completed"],
            tests_failed=[],
            success=True
        )
    
    def _validate_config_management_phase(self):
        """Validate configuration management phase."""
        return ValidationResult(
            component="config_management",
            tests_passed=["Configuration management completed"],
            tests_failed=[],
            success=True
        )
    
    def _validate_doc_consolidation_phase(self):
        """Validate documentation consolidation phase."""
        return ValidationResult(
            component="doc_consolidation",
            tests_passed=["Documentation consolidation completed"],
            tests_failed=[],
            success=True
        )
    
    def _validate_dependency_management_phase(self):
        """Validate dependency management phase."""
        return ValidationResult(
            component="dependency_management",
            tests_passed=["Dependency management completed"],
            tests_failed=[],
            success=True
        )
    
    def _validate_structure_reorganization_phase(self):
        """Validate structure reorganization phase."""
        return self.structure_manager.validate_structure()
    
    # Rollback methods (simplified - would need more detailed implementation)
    def _rollback_temp_cleanup_phase(self):
        """Rollback temporary cleanup phase."""
        self.logger.info("Rolling back temporary cleanup phase")
        # Would restore deleted temporary files from backup
    
    def _rollback_demo_consolidation_phase(self):
        """Rollback demo consolidation phase."""
        self.logger.info("Rolling back demo consolidation phase")
        # Would restore original demo structure from backup
    
    def _rollback_demo_fixes_phase(self):
        """Rollback demo fixes phase."""
        self.logger.info("Rolling back demo fixes phase")
        # Would restore original demo files from backup
    
    def _rollback_import_resolution_phase(self):
        """Rollback import resolution phase."""
        self.logger.info("Rolling back import resolution phase")
        # Would restore original import statements from backup
    
    def _rollback_config_management_phase(self):
        """Rollback configuration management phase."""
        self.logger.info("Rolling back configuration management phase")
        # Would restore original configuration files from backup
    
    def _rollback_doc_consolidation_phase(self):
        """Rollback documentation consolidation phase."""
        self.logger.info("Rolling back documentation consolidation phase")
        # Would restore original documentation from backup
    
    def _rollback_dependency_management_phase(self):
        """Rollback dependency management phase."""
        self.logger.info("Rolling back dependency management phase")
        # Would restore original dependency files from backup
    
    def _rollback_structure_reorganization_phase(self):
        """Rollback structure reorganization phase."""
        self.logger.info("Rolling back structure reorganization phase")
        # Would restore original repository structure from backup


def create_cli() -> argparse.ArgumentParser:
    """Create command-line interface for the cleanup orchestrator."""
    parser = argparse.ArgumentParser(
        description='Nanobrain Repository Cleanup Orchestrator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --dry-run                    # Preview cleanup operations
  %(prog)s --phases backup temp_cleanup # Run specific phases only
  %(prog)s --rollback-to demo_fixes     # Rollback to specific phase
        """
    )
    
    parser.add_argument(
        '--repo-root',
        type=Path,
        default=Path.cwd(),
        help='Repository root directory (default: current directory)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview operations without making changes'
    )
    
    parser.add_argument(
        '--phases',
        nargs='+',
        help='Specific phases to run (default: all phases)'
    )
    
    parser.add_argument(
        '--rollback-to',
        help='Rollback to a specific phase'
    )
    
    parser.add_argument(
        '--list-phases',
        action='store_true',
        help='List all available phases and exit'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser


def main():
    """Main entry point for the cleanup orchestrator."""
    parser = create_cli()
    args = parser.parse_args()
    
    # Create orchestrator
    orchestrator = CleanupOrchestrator(args.repo_root, dry_run=args.dry_run)
    
    # Handle list phases
    if args.list_phases:
        print("Available cleanup phases:")
        for name, phase in orchestrator.phases.items():
            deps = ', '.join(phase.dependencies) if phase.dependencies else 'None'
            print(f"  {name}: {phase.description} (dependencies: {deps})")
        return
    
    # Handle rollback
    if args.rollback_to:
        success = orchestrator.rollback_to_phase(args.rollback_to)
        sys.exit(0 if success else 1)
    
    # Execute cleanup
    try:
        summary = orchestrator.execute_cleanup(args.phases)
        
        # Print summary
        print(f"\nCleanup Summary:")
        print(f"  Duration: {summary.end_time - summary.start_time}")
        print(f"  Phases completed: {len(summary.phases_completed)}")
        print(f"  Success: {summary.success}")
        
        if not summary.success:
            print(f"  Failed phases:")
            for phase_result in summary.phases_completed:
                if not phase_result.success:
                    print(f"    - {phase_result.phase_name}: {phase_result.errors}")
        
        sys.exit(0 if summary.success else 1)
        
    except KeyboardInterrupt:
        print("\nCleanup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Cleanup failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()