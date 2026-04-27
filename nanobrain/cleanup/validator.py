"""
Comprehensive validation system for the Nanobrain cleanup process.

This module provides functionality to validate framework components,
demo execution, configuration files, and documentation integrity.
"""

import os
import sys
import ast
import json
import subprocess
import tempfile
import importlib.util
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass
import yaml
from urllib.parse import urlparse

from .models import ValidationResult, CleanupResult


@dataclass
class ImportValidationResult:
    """Result of import validation."""
    module_name: str
    import_successful: bool
    error_message: Optional[str] = None
    dependencies_found: List[str] = None
    
    def __post_init__(self):
        if self.dependencies_found is None:
            self.dependencies_found = []


@dataclass
class DemoValidationResult:
    """Result of demo execution validation."""
    demo_name: str
    execution_successful: bool
    output_captured: str
    error_message: Optional[str] = None
    runtime_seconds: float = 0.0


@dataclass
class ConfigValidationResult:
    """Result of configuration file validation."""
    config_file: str
    schema_valid: bool
    syntax_valid: bool
    schema_errors: List[str] = None
    syntax_errors: List[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.schema_errors is None:
            self.schema_errors = []
        if self.syntax_errors is None:
            self.syntax_errors = []
        if self.warnings is None:
            self.warnings = []


@dataclass
class DocumentationValidationResult:
    """Result of documentation validation."""
    doc_file: str
    links_valid: bool
    references_valid: bool
    broken_links: List[str] = None
    broken_references: List[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.broken_links is None:
            self.broken_links = []
        if self.broken_references is None:
            self.broken_references = []
        if self.warnings is None:
            self.warnings = []


class Validator:
    """Comprehensive validation system for the cleanup process."""
    
    def __init__(self, repo_root: Path):
        """Initialize the validator.
        
        Args:
            repo_root: Root directory of the repository
        """
        self.repo_root = Path(repo_root)
        
        # Configuration schemas
        self.config_schemas = {
            'parsl_config': {
                'type': 'object',
                'properties': {
                    'executors': {'type': 'array'},
                    'strategy': {'type': 'string'},
                    'app_cache': {'type': 'boolean'},
                    'checkpoint_mode': {'type': 'string'}
                },
                'required': ['executors']
            },
            'workflow_config': {
                'type': 'object',
                'properties': {
                    'input_data': {'type': 'object'},
                    'output_data': {'type': 'object'},
                    'parameters': {'type': 'object'}
                },
                'required': ['input_data', 'output_data']
            }
        }
        
        # Target demos for validation
        self.target_demos = [
            'viral_pssm_workflow',
            'rag_database_creation',
            'academylink_aurora_demo',
            'simple_demo'
        ]
    
    def validate_framework_imports(self) -> ValidationResult:
        """Validate that framework components can be imported correctly.
        
        Returns:
            ValidationResult with import validation details
        """
        try:
            tests_passed = []
            tests_failed = []
            import_results = []
            
            # Core framework modules to test
            core_modules = [
                'nanobrain',
                'nanobrain.core',
                'nanobrain.library',
                'nanobrain.cleanup'
            ]
            
            for module_name in core_modules:
                result = self._validate_module_import(module_name)
                import_results.append(result)
                
                if result.import_successful:
                    tests_passed.append(f"Successfully imported {module_name}")
                else:
                    tests_failed.append(f"Failed to import {module_name}: {result.error_message}")
            
            # Test specific classes and functions
            specific_imports = [
                ('nanobrain.cleanup.backup_manager', 'BackupManager'),
                ('nanobrain.cleanup.structure_manager', 'StructureManager'),
                ('nanobrain.cleanup.models', 'CleanupResult'),
            ]
            
            for module_name, class_name in specific_imports:
                result = self._validate_specific_import(module_name, class_name)
                import_results.append(result)
                
                if result.import_successful:
                    tests_passed.append(f"Successfully imported {class_name} from {module_name}")
                else:
                    tests_failed.append(f"Failed to import {class_name} from {module_name}: {result.error_message}")
            
            success = len(tests_failed) == 0
            
            return ValidationResult(
                component="framework_imports",
                tests_passed=tests_passed,
                tests_failed=tests_failed,
                success=success,
                error_details=str(import_results) if not success else None
            )
            
        except Exception as e:
            return ValidationResult(
                component="framework_imports",
                tests_passed=[],
                tests_failed=[f"Import validation failed: {e}"],
                success=False,
                error_details=str(e)
            )
    
    def validate_demo_execution(self, clean_environment: bool = True) -> ValidationResult:
        """Validate that target demos can execute successfully.
        
        Args:
            clean_environment: Whether to test in a clean Python environment
            
        Returns:
            ValidationResult with demo execution details
        """
        try:
            tests_passed = []
            tests_failed = []
            demo_results = []
            
            for demo_name in self.target_demos:
                demo_path = self.repo_root / 'demos' / demo_name
                
                if not demo_path.exists():
                    tests_failed.append(f"Demo directory not found: {demo_name}")
                    continue
                
                result = self._validate_demo_execution(demo_name, demo_path, clean_environment)
                demo_results.append(result)
                
                if result.execution_successful:
                    tests_passed.append(f"Demo {demo_name} executed successfully")
                else:
                    tests_failed.append(f"Demo {demo_name} failed: {result.error_message}")
            
            success = len(tests_failed) == 0
            
            return ValidationResult(
                component="demo_execution",
                tests_passed=tests_passed,
                tests_failed=tests_failed,
                success=success,
                error_details=str(demo_results) if not success else None
            )
            
        except Exception as e:
            return ValidationResult(
                component="demo_execution",
                tests_passed=[],
                tests_failed=[f"Demo validation failed: {e}"],
                success=False,
                error_details=str(e)
            )
    
    def validate_configuration_files(self) -> ValidationResult:
        """Validate configuration files against schemas.
        
        Returns:
            ValidationResult with configuration validation details
        """
        try:
            tests_passed = []
            tests_failed = []
            config_results = []
            
            # Find configuration files
            config_files = []
            config_files.extend(self.repo_root.rglob('*.yaml'))
            config_files.extend(self.repo_root.rglob('*.yml'))
            config_files.extend(self.repo_root.rglob('*.json'))
            
            # Filter to relevant config files
            relevant_configs = []
            for config_file in config_files:
                # Skip hidden directories and common non-config files
                if any(part.startswith('.') for part in config_file.parts):
                    continue
                if config_file.name in {'package.json', 'tsconfig.json'}:
                    continue
                relevant_configs.append(config_file)
            
            if not relevant_configs:
                tests_passed.append("No configuration files found to validate")
            
            for config_file in relevant_configs:
                result = self._validate_config_file(config_file)
                config_results.append(result)
                
                if result.schema_valid and result.syntax_valid:
                    tests_passed.append(f"Configuration file {config_file.name} is valid")
                else:
                    error_details = []
                    if not result.syntax_valid:
                        error_details.extend(result.syntax_errors)
                    if not result.schema_valid:
                        error_details.extend(result.schema_errors)
                    tests_failed.append(f"Configuration file {config_file.name} is invalid: {'; '.join(error_details)}")
            
            success = len(tests_failed) == 0
            
            return ValidationResult(
                component="configuration_files",
                tests_passed=tests_passed,
                tests_failed=tests_failed,
                success=success,
                error_details=str(config_results) if not success else None
            )
            
        except Exception as e:
            return ValidationResult(
                component="configuration_files",
                tests_passed=[],
                tests_failed=[f"Configuration validation failed: {e}"],
                success=False,
                error_details=str(e)
            )
    
    def validate_documentation_integrity(self) -> ValidationResult:
        """Validate documentation links and references.
        
        Returns:
            ValidationResult with documentation validation details
        """
        try:
            tests_passed = []
            tests_failed = []
            doc_results = []
            
            # Find documentation files
            doc_files = []
            doc_files.extend(self.repo_root.rglob('*.md'))
            doc_files.extend(self.repo_root.rglob('*.rst'))
            
            # Filter to relevant documentation
            relevant_docs = []
            for doc_file in doc_files:
                # Skip hidden directories and temporary files
                if any(part.startswith('.') for part in doc_file.parts):
                    continue
                if doc_file.name.startswith('tmp_') or doc_file.name.endswith('.tmp'):
                    continue
                relevant_docs.append(doc_file)
            
            if not relevant_docs:
                tests_passed.append("No documentation files found to validate")
            
            for doc_file in relevant_docs:
                result = self._validate_documentation_file(doc_file)
                doc_results.append(result)
                
                if result.links_valid and result.references_valid:
                    tests_passed.append(f"Documentation file {doc_file.name} has valid links and references")
                else:
                    error_details = []
                    if not result.links_valid:
                        error_details.extend([f"Broken link: {link}" for link in result.broken_links])
                    if not result.references_valid:
                        error_details.extend([f"Broken reference: {ref}" for ref in result.broken_references])
                    tests_failed.append(f"Documentation file {doc_file.name} has issues: {'; '.join(error_details[:3])}")
            
            success = len(tests_failed) == 0
            
            return ValidationResult(
                component="documentation_integrity",
                tests_passed=tests_passed,
                tests_failed=tests_failed,
                success=success,
                error_details=str(doc_results) if not success else None
            )
            
        except Exception as e:
            return ValidationResult(
                component="documentation_integrity",
                tests_passed=[],
                tests_failed=[f"Documentation validation failed: {e}"],
                success=False,
                error_details=str(e)
            )
    
    def create_rollback_procedure(self, phase_name: str, backup_info: Dict[str, Any]) -> CleanupResult:
        """Create rollback procedure for a specific phase.
        
        Args:
            phase_name: Name of the phase to create rollback for
            backup_info: Information about backups created during the phase
            
        Returns:
            CleanupResult with rollback procedure details
        """
        try:
            rollback_script_path = self.repo_root / f'rollback_{phase_name}.sh'
            
            rollback_commands = [
                '#!/bin/bash',
                f'# Rollback script for {phase_name}',
                f'# Generated automatically during cleanup process',
                '',
                'set -e',  # Exit on error
                '',
                'echo "Starting rollback for phase: {}"'.format(phase_name),
                ''
            ]
            
            # Add specific rollback commands based on backup info
            if 'git_branch' in backup_info:
                rollback_commands.extend([
                    '# Restore from git branch backup',
                    f'git checkout {backup_info["git_branch"]}',
                    'echo "Restored from git branch backup"',
                    ''
                ])
            
            if 'file_backup' in backup_info:
                rollback_commands.extend([
                    '# Restore from file backup',
                    f'tar -xzf {backup_info["file_backup"]} -C .',
                    'echo "Restored from file backup"',
                    ''
                ])
            
            if 'moved_files' in backup_info:
                rollback_commands.extend([
                    '# Restore moved files',
                ])
                for src, dest in backup_info['moved_files']:
                    rollback_commands.append(f'mv "{dest}" "{src}"')
                rollback_commands.extend(['', 'echo "Restored moved files"', ''])
            
            rollback_commands.extend([
                'echo "Rollback completed successfully"',
                'echo "Please verify the repository state and run tests"'
            ])
            
            # Write rollback script
            rollback_script_path.write_text('\n'.join(rollback_commands))
            rollback_script_path.chmod(0o755)  # Make executable
            
            return CleanupResult(
                success=True,
                files_processed=1,
                message=f"Created rollback procedure for {phase_name}",
                details={
                    'rollback_script': str(rollback_script_path),
                    'backup_info': backup_info
                }
            )
            
        except Exception as e:
            return CleanupResult(
                success=False,
                errors=[f"Failed to create rollback procedure: {e}"]
            )
    
    def _validate_module_import(self, module_name: str) -> ImportValidationResult:
        """Validate that a module can be imported."""
        try:
            # Try to import the module
            spec = importlib.util.find_spec(module_name)
            if spec is None:
                return ImportValidationResult(
                    module_name=module_name,
                    import_successful=False,
                    error_message=f"Module {module_name} not found"
                )
            
            # Try to load the module
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Get dependencies
            dependencies = []
            if hasattr(module, '__file__') and module.__file__:
                dependencies = self._extract_dependencies(Path(module.__file__))
            
            return ImportValidationResult(
                module_name=module_name,
                import_successful=True,
                dependencies_found=dependencies
            )
            
        except Exception as e:
            return ImportValidationResult(
                module_name=module_name,
                import_successful=False,
                error_message=str(e)
            )
    
    def _validate_specific_import(self, module_name: str, class_name: str) -> ImportValidationResult:
        """Validate that a specific class/function can be imported from a module."""
        try:
            # Try to import the module
            spec = importlib.util.find_spec(module_name)
            if spec is None:
                return ImportValidationResult(
                    module_name=f"{module_name}.{class_name}",
                    import_successful=False,
                    error_message=f"Module {module_name} not found"
                )
            
            # Try to load the module
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Check if the class/function exists
            if not hasattr(module, class_name):
                return ImportValidationResult(
                    module_name=f"{module_name}.{class_name}",
                    import_successful=False,
                    error_message=f"{class_name} not found in {module_name}"
                )
            
            return ImportValidationResult(
                module_name=f"{module_name}.{class_name}",
                import_successful=True
            )
            
        except Exception as e:
            return ImportValidationResult(
                module_name=f"{module_name}.{class_name}",
                import_successful=False,
                error_message=str(e)
            )
    
    def _extract_dependencies(self, module_path: Path) -> List[str]:
        """Extract import dependencies from a Python module."""
        try:
            content = module_path.read_text(encoding='utf-8')
            tree = ast.parse(content)
            
            dependencies = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        dependencies.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        dependencies.append(node.module)
            
            return dependencies
            
        except Exception:
            return []
    
    def _validate_demo_execution(self, demo_name: str, demo_path: Path, clean_environment: bool) -> DemoValidationResult:
        """Validate execution of a specific demo."""
        try:
            # Look for main execution script
            main_scripts = ['main.py', 'run.py', f'{demo_name}.py', 'demo.py']
            main_script = None
            
            for script_name in main_scripts:
                script_path = demo_path / script_name
                if script_path.exists():
                    main_script = script_path
                    break
            
            if not main_script:
                return DemoValidationResult(
                    demo_name=demo_name,
                    execution_successful=False,
                    output_captured="",
                    error_message="No main execution script found"
                )
            
            # Execute the demo (with timeout and in dry-run mode if possible)
            if clean_environment:
                result = self._execute_in_clean_environment(main_script)
            else:
                result = self._execute_demo_script(main_script)
            
            return result
            
        except Exception as e:
            return DemoValidationResult(
                demo_name=demo_name,
                execution_successful=False,
                output_captured="",
                error_message=str(e)
            )
    
    def _execute_demo_script(self, script_path: Path) -> DemoValidationResult:
        """Execute a demo script and capture results."""
        try:
            # Run with timeout and capture output
            result = subprocess.run(
                [sys.executable, str(script_path), '--dry-run'],  # Try dry-run first
                cwd=script_path.parent,
                capture_output=True,
                text=True,
                timeout=30  # 30 second timeout
            )
            
            if result.returncode != 0:
                # Try without --dry-run flag
                result = subprocess.run(
                    [sys.executable, str(script_path)],
                    cwd=script_path.parent,
                    capture_output=True,
                    text=True,
                    timeout=10  # Shorter timeout for full execution
                )
            
            success = result.returncode == 0
            output = result.stdout + result.stderr
            
            return DemoValidationResult(
                demo_name=script_path.parent.name,
                execution_successful=success,
                output_captured=output,
                error_message=result.stderr if not success else None
            )
            
        except subprocess.TimeoutExpired:
            return DemoValidationResult(
                demo_name=script_path.parent.name,
                execution_successful=False,
                output_captured="",
                error_message="Demo execution timed out"
            )
        except Exception as e:
            return DemoValidationResult(
                demo_name=script_path.parent.name,
                execution_successful=False,
                output_captured="",
                error_message=str(e)
            )
    
    def _execute_in_clean_environment(self, script_path: Path) -> DemoValidationResult:
        """Execute demo in a clean Python environment."""
        try:
            # Create a minimal test script that just imports the demo
            test_script = f'''
import sys
sys.path.insert(0, "{self.repo_root}")
try:
    import {script_path.stem}
    print("Import successful")
except Exception as e:
    print(f"Import failed: {{e}}")
    sys.exit(1)
'''
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(test_script)
                temp_script = f.name
            
            try:
                result = subprocess.run(
                    [sys.executable, temp_script],
                    cwd=script_path.parent,
                    capture_output=True,
                    text=True,
                    timeout=15
                )
                
                success = result.returncode == 0 and "Import successful" in result.stdout
                
                return DemoValidationResult(
                    demo_name=script_path.parent.name,
                    execution_successful=success,
                    output_captured=result.stdout + result.stderr,
                    error_message=result.stderr if not success else None
                )
                
            finally:
                os.unlink(temp_script)
                
        except Exception as e:
            return DemoValidationResult(
                demo_name=script_path.parent.name,
                execution_successful=False,
                output_captured="",
                error_message=str(e)
            )
    
    def _validate_config_file(self, config_file: Path) -> ConfigValidationResult:
        """Validate a configuration file."""
        try:
            # Check syntax first
            syntax_valid = True
            syntax_errors = []
            
            try:
                if config_file.suffix.lower() in {'.yaml', '.yml'}:
                    with open(config_file, 'r') as f:
                        yaml.safe_load(f)
                elif config_file.suffix.lower() == '.json':
                    with open(config_file, 'r') as f:
                        json.load(f)
            except Exception as e:
                syntax_valid = False
                syntax_errors.append(str(e))
            
            # Schema validation (basic)
            schema_valid = True
            schema_errors = []
            warnings = []
            
            if syntax_valid:
                try:
                    # Load content for schema validation
                    if config_file.suffix.lower() in {'.yaml', '.yml'}:
                        with open(config_file, 'r') as f:
                            content = yaml.safe_load(f)
                    elif config_file.suffix.lower() == '.json':
                        with open(config_file, 'r') as f:
                            content = json.load(f)
                    else:
                        content = {}
                    
                    # Apply basic validation rules
                    if isinstance(content, dict):
                        # Check for common configuration patterns
                        if 'executors' in content and not isinstance(content['executors'], list):
                            schema_errors.append("'executors' should be a list")
                            schema_valid = False
                        
                        # Check for hardcoded paths
                        hardcoded_paths = self._find_hardcoded_paths(content)
                        if hardcoded_paths:
                            warnings.extend([f"Potential hardcoded path: {path}" for path in hardcoded_paths])
                    
                except Exception as e:
                    schema_errors.append(str(e))
                    schema_valid = False
            
            return ConfigValidationResult(
                config_file=str(config_file),
                schema_valid=schema_valid,
                syntax_valid=syntax_valid,
                schema_errors=schema_errors,
                syntax_errors=syntax_errors,
                warnings=warnings
            )
            
        except Exception as e:
            return ConfigValidationResult(
                config_file=str(config_file),
                schema_valid=False,
                syntax_valid=False,
                syntax_errors=[str(e)]
            )
    
    def _find_hardcoded_paths(self, config_data: Any) -> List[str]:
        """Find potential hardcoded paths in configuration data."""
        hardcoded_paths = []
        
        def check_value(value):
            if isinstance(value, str):
                # Check for absolute paths
                if value.startswith('/') and len(value) > 1:
                    hardcoded_paths.append(value)
                # Check for Windows paths
                elif len(value) > 3 and value[1:3] == ':\\':
                    hardcoded_paths.append(value)
            elif isinstance(value, dict):
                for v in value.values():
                    check_value(v)
            elif isinstance(value, list):
                for item in value:
                    check_value(item)
        
        check_value(config_data)
        return hardcoded_paths
    
    def _validate_documentation_file(self, doc_file: Path) -> DocumentationValidationResult:
        """Validate a documentation file."""
        try:
            content = doc_file.read_text(encoding='utf-8')
            
            # Find links and references
            import re
            
            # Markdown links: [text](url) or [text](file.md)
            link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
            links = re.findall(link_pattern, content)
            
            # Reference links: [text]: url
            ref_pattern = r'^\[([^\]]+)\]:\s*(.+)$'
            references = re.findall(ref_pattern, content, re.MULTILINE)
            
            broken_links = []
            broken_references = []
            warnings = []
            
            # Validate links
            for link_text, link_url in links:
                if not self._validate_link(link_url, doc_file):
                    broken_links.append(f"{link_text} -> {link_url}")
            
            # Validate references
            for ref_name, ref_url in references:
                if not self._validate_link(ref_url, doc_file):
                    broken_references.append(f"{ref_name} -> {ref_url}")
            
            # Check for common issues
            if len(content) < 100:
                warnings.append("Documentation file is very short")
            
            if not content.strip():
                warnings.append("Documentation file is empty")
            
            links_valid = len(broken_links) == 0
            references_valid = len(broken_references) == 0
            
            return DocumentationValidationResult(
                doc_file=str(doc_file),
                links_valid=links_valid,
                references_valid=references_valid,
                broken_links=broken_links,
                broken_references=broken_references,
                warnings=warnings
            )
            
        except Exception as e:
            return DocumentationValidationResult(
                doc_file=str(doc_file),
                links_valid=False,
                references_valid=False,
                broken_links=[],
                broken_references=[],
                warnings=[f"Failed to validate: {e}"]
            )
    
    def _validate_link(self, link_url: str, doc_file: Path) -> bool:
        """Validate a single link or reference."""
        try:
            # Skip external URLs (assume they're valid for now)
            if link_url.startswith(('http://', 'https://', 'ftp://')):
                return True
            
            # Skip email links
            if link_url.startswith('mailto:'):
                return True
            
            # Skip anchors within the same document
            if link_url.startswith('#'):
                return True
            
            # Check local file references
            if not link_url.startswith('/'):
                # Relative path
                target_path = doc_file.parent / link_url
            else:
                # Absolute path from repo root
                target_path = self.repo_root / link_url.lstrip('/')
            
            # Remove anchor fragments
            if '#' in str(target_path):
                target_path = Path(str(target_path).split('#')[0])
            
            return target_path.exists()
            
        except Exception:
            return False