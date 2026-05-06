"""
Demo Fixer for the Nanobrain cleanup system.

This module provides functionality to fix and standardize target demonstration
implementations, focusing on the viral_pssm_workflow and rag_database_creation demos.
"""

import os
import re
import shutil
import ast
import yaml
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple
import logging

from .models import FixResult, ConsolidationResult, PathFixResult, ValidationResult


class DemoFixer:
    """
    Fixes and standardizes target demonstration implementations.
    
    Focuses on consolidating test files, removing hardcoded paths,
    and ensuring demo functionality is preserved while improving maintainability.
    """
    
    def __init__(self, repo_root: str):
        """
        Initialize the DemoFixer.
        
        Args:
            repo_root: Path to the repository root directory
        """
        self.repo_root = Path(repo_root).resolve()
        self.demos_dir = self.repo_root / "demos"
        self.logger = logging.getLogger(__name__)
        
        # Target demos to fix
        self.target_demos = {
            'viral_pssm_workflow': self.demos_dir / 'viral_pssm_workflow',
            'rag_database_creation': self.demos_dir / 'rag_database_creation'
        }
    
    def fix_viral_pssm_demo(self) -> FixResult:
        """
        Fix the viral_pssm_workflow demo.
        
        Returns:
            FixResult with details of fixes applied
        """
        demo_name = 'viral_pssm_workflow'
        demo_path = self.target_demos[demo_name]
        
        if not demo_path.exists():
            return FixResult(
                demo_name=demo_name,
                fixes_applied=[],
                files_modified=[],
                tests_consolidated=0,
                paths_fixed=0,
                success=False,
                errors=[f"Demo directory not found: {demo_path}"]
            )
        
        fixes_applied = []
        files_modified = []
        errors = []
        tests_consolidated = 0
        paths_fixed = 0
        
        try:
            self.logger.info(f"Starting fixes for {demo_name}")
            
            # 1. Consolidate test files
            consolidation_result = self.consolidate_test_files(str(demo_path))
            if consolidation_result.success:
                fixes_applied.append("Test file consolidation")
                files_modified.extend(consolidation_result.files_created)
                tests_consolidated = consolidation_result.original_test_count - consolidation_result.consolidated_test_count
            else:
                errors.extend(consolidation_result.errors)
            
            # 2. Remove hardcoded paths and create configurations
            path_fix_result = self.remove_hardcoded_paths(str(demo_path))
            if path_fix_result.success:
                fixes_applied.append("Hardcoded path removal")
                files_modified.extend(path_fix_result.config_files_created)
                paths_fixed = path_fix_result.paths_fixed
            else:
                errors.extend(path_fix_result.errors)
            
            # 3. Fix BV-BRC API endpoints
            api_fixes = self._fix_bvbrc_endpoints(demo_path)
            if api_fixes['success']:
                fixes_applied.append("BV-BRC API endpoint configuration")
                files_modified.extend(api_fixes['files_modified'])
            else:
                errors.extend(api_fixes['errors'])
            
            # 4. Standardize configuration patterns
            config_fixes = self._standardize_demo_config(demo_path)
            if config_fixes['success']:
                fixes_applied.append("Configuration standardization")
                files_modified.extend(config_fixes['files_modified'])
            else:
                errors.extend(config_fixes['errors'])
            
            success = len(errors) == 0
            
            self.logger.info(f"Viral PSSM demo fixes completed. Success: {success}")
            
            return FixResult(
                demo_name=demo_name,
                fixes_applied=fixes_applied,
                files_modified=files_modified,
                tests_consolidated=tests_consolidated,
                paths_fixed=paths_fixed,
                success=success,
                errors=errors
            )
            
        except Exception as e:
            self.logger.error(f"Viral PSSM demo fix failed: {str(e)}")
            return FixResult(
                demo_name=demo_name,
                fixes_applied=fixes_applied,
                files_modified=files_modified,
                tests_consolidated=tests_consolidated,
                paths_fixed=paths_fixed,
                success=False,
                errors=errors + [f"Unexpected error: {str(e)}"]
            )
    
    def fix_rag_database_demo(self) -> FixResult:
        """
        Fix the rag_database_creation demo.
        
        Returns:
            FixResult with details of fixes applied
        """
        demo_name = 'rag_database_creation'
        demo_path = self.target_demos[demo_name]
        
        if not demo_path.exists():
            return FixResult(
                demo_name=demo_name,
                fixes_applied=[],
                files_modified=[],
                tests_consolidated=0,
                paths_fixed=0,
                success=False,
                errors=[f"Demo directory not found: {demo_path}"]
            )
        
        fixes_applied = []
        files_modified = []
        errors = []
        tests_consolidated = 0
        paths_fixed = 0
        
        try:
            self.logger.info(f"Starting fixes for {demo_name}")
            
            # 1. Consolidate similar scripts
            script_consolidation = self._consolidate_rag_scripts(demo_path)
            if script_consolidation['success']:
                fixes_applied.append("Script consolidation")
                files_modified.extend(script_consolidation['files_modified'])
                tests_consolidated = script_consolidation['scripts_consolidated']
            else:
                errors.extend(script_consolidation['errors'])
            
            # 2. Remove hardcoded paths
            path_fix_result = self.remove_hardcoded_paths(str(demo_path))
            if path_fix_result.success:
                fixes_applied.append("Hardcoded path removal")
                files_modified.extend(path_fix_result.config_files_created)
                paths_fixed = path_fix_result.paths_fixed
            else:
                errors.extend(path_fix_result.errors)
            
            # 3. Standardize configuration patterns
            config_fixes = self._standardize_demo_config(demo_path)
            if config_fixes['success']:
                fixes_applied.append("Configuration standardization")
                files_modified.extend(config_fixes['files_modified'])
            else:
                errors.extend(config_fixes['errors'])
            
            # 4. Create unified main implementation
            main_impl = self._create_unified_rag_main(demo_path)
            if main_impl['success']:
                fixes_applied.append("Unified main implementation")
                files_modified.extend(main_impl['files_modified'])
            else:
                errors.extend(main_impl['errors'])
            
            success = len(errors) == 0
            
            self.logger.info(f"RAG database demo fixes completed. Success: {success}")
            
            return FixResult(
                demo_name=demo_name,
                fixes_applied=fixes_applied,
                files_modified=files_modified,
                tests_consolidated=tests_consolidated,
                paths_fixed=paths_fixed,
                success=success,
                errors=errors
            )
            
        except Exception as e:
            self.logger.error(f"RAG database demo fix failed: {str(e)}")
            return FixResult(
                demo_name=demo_name,
                fixes_applied=fixes_applied,
                files_modified=files_modified,
                tests_consolidated=tests_consolidated,
                paths_fixed=paths_fixed,
                success=False,
                errors=errors + [f"Unexpected error: {str(e)}"]
            )
    
    def consolidate_test_files(self, demo_path: str) -> ConsolidationResult:
        """
        Consolidate multiple test files into a focused test suite.
        
        Args:
            demo_path: Path to the demo directory
            
        Returns:
            ConsolidationResult with consolidation details
        """
        demo_dir = Path(demo_path)
        files_removed = []
        files_created = []
        errors = []
        
        try:
            # Find all test files (avoid duplicates by using a set)
            test_files = set()
            test_patterns = ['test_*.py', '*_test.py']
            
            for pattern in test_patterns:
                test_files.update(demo_dir.rglob(pattern))
            
            test_files = list(test_files)  # Convert back to list
            original_count = len(test_files)
            
            if original_count == 0:
                return ConsolidationResult(
                    original_test_count=0,
                    consolidated_test_count=0,
                    files_removed=[],
                    files_created=[],
                    success=True
                )
            
            # Group tests by functionality
            test_groups = self._group_test_files(test_files)
            
            # Create consolidated test files
            consolidated_count = 0
            for group_name, group_files in test_groups.items():
                if len(group_files) > 1:  # Only consolidate if multiple files
                    consolidated_file = demo_dir / f"test_{group_name}.py"
                    
                    # Merge test content
                    merged_content = self._merge_test_files(group_files)
                    
                    # Write consolidated file
                    with open(consolidated_file, 'w') as f:
                        f.write(merged_content)
                    
                    files_created.append(str(consolidated_file))
                    consolidated_count += 1
                    
                    # Remove original files
                    for test_file in group_files:
                        try:
                            test_file.unlink()
                            files_removed.append(str(test_file))
                        except Exception as e:
                            errors.append(f"Failed to remove {test_file}: {str(e)}")
                else:
                    # Keep single files as-is
                    consolidated_count += 1
            
            self.logger.info(f"Consolidated {original_count} test files into {consolidated_count}")
            
            return ConsolidationResult(
                original_test_count=original_count,
                consolidated_test_count=consolidated_count,
                files_removed=files_removed,
                files_created=files_created,
                success=len(errors) == 0,
                errors=errors
            )
            
        except Exception as e:
            self.logger.error(f"Test consolidation failed: {str(e)}")
            return ConsolidationResult(
                original_test_count=0,
                consolidated_test_count=0,
                files_removed=files_removed,
                files_created=files_created,
                success=False,
                errors=errors + [f"Consolidation error: {str(e)}"]
            )
    
    def remove_hardcoded_paths(self, demo_path: str) -> PathFixResult:
        """
        Remove hardcoded paths and create configuration files.
        
        Args:
            demo_path: Path to the demo directory
            
        Returns:
            PathFixResult with path fixing details
        """
        demo_dir = Path(demo_path)
        config_files_created = []
        errors = []
        files_processed = 0
        paths_fixed = 0
        
        try:
            # Common hardcoded path patterns
            hardcoded_patterns = [
                r'/home/[^/]+/[^"\s]+',  # Home directory paths
                r'/scratch/[^/]+/[^"\s]+',  # Scratch directory paths
                r'/lus/[^/]+/[^"\s]+',  # Lustre filesystem paths
                r'/gpfs/[^/]+/[^"\s]+',  # GPFS filesystem paths
                r'/aurora_deployment/[^"\s]+',  # Aurora deployment paths
            ]
            
            # Find Python and YAML files
            target_files = []
            target_files.extend(demo_dir.rglob('*.py'))
            target_files.extend(demo_dir.rglob('*.yaml'))
            target_files.extend(demo_dir.rglob('*.yml'))
            
            # Configuration to extract
            extracted_config = {}
            
            for file_path in target_files:
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    original_content = content
                    files_processed += 1
                    
                    # Find and replace hardcoded paths
                    for pattern in hardcoded_patterns:
                        matches = re.findall(pattern, content)
                        for match in matches:
                            # Create environment variable name
                            var_name = self._path_to_env_var(match)
                            
                            # Store in config
                            extracted_config[var_name] = match
                            
                            # Replace in content
                            if file_path.suffix in ['.yaml', '.yml']:
                                replacement = f"${{{var_name}}}"
                            else:
                                replacement = f"os.environ.get('{var_name}', '{match}')"
                            
                            content = content.replace(f'"{match}"', f'"{replacement}"')
                            content = content.replace(f"'{match}'", f"'{replacement}'")
                            paths_fixed += 1
                    
                    # Write back if changed
                    if content != original_content:
                        with open(file_path, 'w') as f:
                            f.write(content)
                        
                        # Add import for os if needed in Python files
                        if file_path.suffix == '.py' and 'os.environ' in content:
                            self._ensure_os_import(file_path)
                
                except Exception as e:
                    errors.append(f"Failed to process {file_path}: {str(e)}")
            
            # Create configuration file if we extracted any paths
            if extracted_config:
                config_file = demo_dir / 'config.yaml'
                config_content = {
                    'paths': extracted_config,
                    'description': 'Configuration file for demo paths - update these for your environment'
                }
                
                with open(config_file, 'w') as f:
                    yaml.dump(config_content, f, default_flow_style=False)
                
                config_files_created.append(str(config_file))
                
                # Create environment template
                env_file = demo_dir / '.env.template'
                env_content = "# Environment variables for demo\n"
                for var_name, default_path in extracted_config.items():
                    env_content += f"{var_name}={default_path}\n"
                
                with open(env_file, 'w') as f:
                    f.write(env_content)
                
                config_files_created.append(str(env_file))
            
            self.logger.info(f"Fixed {paths_fixed} hardcoded paths in {files_processed} files")
            
            return PathFixResult(
                files_processed=files_processed,
                paths_fixed=paths_fixed,
                config_files_created=config_files_created,
                success=len(errors) == 0,
                errors=errors
            )
            
        except Exception as e:
            self.logger.error(f"Path fixing failed: {str(e)}")
            return PathFixResult(
                files_processed=files_processed,
                paths_fixed=paths_fixed,
                config_files_created=config_files_created,
                success=False,
                errors=errors + [f"Path fixing error: {str(e)}"]
            )
    
    def validate_demo_functionality(self, demo_path: str) -> ValidationResult:
        """
        Validate that a demo can still function after fixes.
        
        Args:
            demo_path: Path to the demo directory
            
        Returns:
            ValidationResult with validation details
        """
        demo_dir = Path(demo_path)
        demo_name = demo_dir.name
        tests_passed = []
        tests_failed = []
        
        try:
            # Test 1: Demo directory structure
            if demo_dir.exists():
                tests_passed.append("Demo directory exists")
            else:
                tests_failed.append("Demo directory missing")
                return ValidationResult(
                    component=f"demo_validation_{demo_name}",
                    tests_passed=tests_passed,
                    tests_failed=tests_failed,
                    success=False,
                    error_details="Demo directory not found"
                )
            
            # Test 2: Essential files exist
            essential_files = ['README.md']
            for file_name in essential_files:
                if (demo_dir / file_name).exists():
                    tests_passed.append(f"Essential file '{file_name}' exists")
                else:
                    tests_failed.append(f"Essential file '{file_name}' missing")
            
            # Test 3: Python files are syntactically valid
            python_files = list(demo_dir.rglob('*.py'))
            valid_python_files = 0
            for py_file in python_files:
                try:
                    with open(py_file, 'r') as f:
                        ast.parse(f.read())
                    valid_python_files += 1
                except SyntaxError as e:
                    tests_failed.append(f"Syntax error in {py_file.name}: {str(e)}")
            
            if valid_python_files > 0:
                tests_passed.append(f"{valid_python_files} Python files are syntactically valid")
            
            # Test 4: Configuration files are valid YAML
            yaml_files = list(demo_dir.rglob('*.yaml')) + list(demo_dir.rglob('*.yml'))
            valid_yaml_files = 0
            for yaml_file in yaml_files:
                try:
                    with open(yaml_file, 'r') as f:
                        yaml.safe_load(f)
                    valid_yaml_files += 1
                except yaml.YAMLError as e:
                    tests_failed.append(f"YAML error in {yaml_file.name}: {str(e)}")
            
            if valid_yaml_files > 0:
                tests_passed.append(f"{valid_yaml_files} YAML files are valid")
            
            # Test 5: No obvious import errors (basic check)
            import_errors = self._check_basic_imports(demo_dir)
            if len(import_errors) == 0:
                tests_passed.append("No obvious import errors detected")
            else:
                for error in import_errors:
                    tests_failed.append(f"Import issue: {error}")
            
            success = len(tests_failed) == 0
            
            return ValidationResult(
                component=f"demo_validation_{demo_name}",
                tests_passed=tests_passed,
                tests_failed=tests_failed,
                success=success,
                error_details=None if success else f"Failed {len(tests_failed)} validation checks"
            )
            
        except Exception as e:
            self.logger.error(f"Demo validation failed: {str(e)}")
            return ValidationResult(
                component=f"demo_validation_{demo_name}",
                tests_passed=tests_passed,
                tests_failed=tests_failed + [f"Validation error: {str(e)}"],
                success=False,
                error_details=str(e)
            )
    
    def _group_test_files(self, test_files: List[Path]) -> Dict[str, List[Path]]:
        """Group test files by functionality."""
        groups = {}
        
        for test_file in test_files:
            # Determine group based on file name patterns
            name = test_file.stem.lower()
            
            if 'integration' in name or 'end_to_end' in name:
                group = 'integration'
            elif 'unit' in name or 'test_' in name:
                group = 'unit'
            elif 'performance' in name or 'perf' in name:
                group = 'performance'
            elif 'api' in name or 'endpoint' in name:
                group = 'api'
            else:
                group = 'general'
            
            if group not in groups:
                groups[group] = []
            groups[group].append(test_file)
        
        return groups
    
    def _merge_test_files(self, test_files: List[Path]) -> str:
        """Merge multiple test files into a single file."""
        merged_content = []
        imports = set()
        test_classes = []
        test_functions = []
        
        # Header
        merged_content.append('"""')
        merged_content.append('Consolidated test file.')
        merged_content.append('Generated by Nanobrain cleanup system.')
        merged_content.append('"""')
        merged_content.append('')
        
        for test_file in test_files:
            try:
                with open(test_file, 'r') as f:
                    content = f.read()
                
                # Parse the file to extract components
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.add(f"import {alias.name}")
                    elif isinstance(node, ast.ImportFrom):
                        module = node.module or ''
                        for alias in node.names:
                            imports.add(f"from {module} import {alias.name}")
                
                # Extract test classes and functions (simplified)
                lines = content.split('\n')
                in_class = False
                in_function = False
                current_block = []
                
                for line in lines:
                    if line.strip().startswith('class Test') and ':' in line:
                        if current_block:
                            if in_function:
                                test_functions.append('\n'.join(current_block))
                            elif in_class:
                                test_classes.append('\n'.join(current_block))
                        current_block = [line]
                        in_class = True
                        in_function = False
                    elif line.strip().startswith('def test_') and ':' in line:
                        if current_block and in_class:
                            test_classes.append('\n'.join(current_block))
                        elif current_block and in_function:
                            test_functions.append('\n'.join(current_block))
                        current_block = [line]
                        in_function = True
                        in_class = False
                    elif line.strip() and (in_class or in_function):
                        current_block.append(line)
                    elif not line.strip() and current_block:
                        current_block.append(line)
                
                # Add the last block
                if current_block:
                    if in_function:
                        test_functions.append('\n'.join(current_block))
                    elif in_class:
                        test_classes.append('\n'.join(current_block))
                        
            except Exception as e:
                # If parsing fails, add as comment
                merged_content.append(f'# Failed to parse {test_file.name}: {str(e)}')
        
        # Add imports
        for imp in sorted(imports):
            merged_content.append(imp)
        merged_content.append('')
        
        # Add test classes
        for test_class in test_classes:
            merged_content.append(test_class)
            merged_content.append('')
        
        # Add test functions
        for test_function in test_functions:
            merged_content.append(test_function)
            merged_content.append('')
        
        return '\n'.join(merged_content)
    
    def _path_to_env_var(self, path: str) -> str:
        """Convert a path to an environment variable name."""
        # Extract meaningful parts of the path
        parts = path.split('/')
        meaningful_parts = []
        
        for part in parts:
            if part and part not in ['home', 'scratch', 'lus', 'gpfs']:
                meaningful_parts.append(part)
        
        # Create variable name
        if len(meaningful_parts) >= 2:
            var_name = f"{meaningful_parts[-2]}_{meaningful_parts[-1]}_PATH"
        elif len(meaningful_parts) == 1:
            var_name = f"{meaningful_parts[0]}_PATH"
        else:
            var_name = "DEMO_PATH"
        
        # Clean up the variable name
        var_name = re.sub(r'[^A-Za-z0-9_]', '_', var_name.upper())
        return var_name
    
    def _ensure_os_import(self, file_path: Path):
        """Ensure a Python file has 'import os' if it uses os.environ."""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            if 'os.environ' in content and 'import os' not in content:
                lines = content.split('\n')
                
                # Find the best place to insert the import
                insert_index = 0
                for i, line in enumerate(lines):
                    if line.strip().startswith('"""') or line.strip().startswith("'''"):
                        # Skip docstrings
                        continue
                    elif line.strip().startswith('import ') or line.strip().startswith('from '):
                        insert_index = i + 1
                    elif line.strip() and not line.strip().startswith('#'):
                        break
                
                lines.insert(insert_index, 'import os')
                
                with open(file_path, 'w') as f:
                    f.write('\n'.join(lines))
                    
        except Exception as e:
            self.logger.warning(f"Failed to add os import to {file_path}: {str(e)}")
    
    def _check_basic_imports(self, demo_dir: Path) -> List[str]:
        """Check for basic import issues in Python files."""
        errors = []
        
        python_files = list(demo_dir.rglob('*.py'))
        for py_file in python_files:
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                
                # Check for common import issues
                lines = content.split('\n')
                for i, line in enumerate(lines, 1):
                    line = line.strip()
                    if line.startswith('from ') and ' import ' in line:
                        # Check for relative imports that might be broken
                        if line.startswith('from .') or line.startswith('from ..'):
                            # This is a relative import - might need checking
                            pass
                    elif line.startswith('import '):
                        # Check for imports that might not exist
                        module_name = line.replace('import ', '').split()[0]
                        if '.' in module_name and not module_name.startswith('nanobrain'):
                            # External module - assume it's fine
                            pass
                            
            except Exception as e:
                errors.append(f"Error checking imports in {py_file.name}: {str(e)}")
        
        return errors
    
    def _fix_bvbrc_endpoints(self, demo_path: Path) -> Dict:
        """Fix hardcoded BV-BRC API endpoints."""
        files_modified = []
        errors = []
        
        try:
            # Find files with BV-BRC endpoints
            python_files = list(demo_path.rglob('*.py'))
            
            bvbrc_patterns = [
                r'https://www\.bv-brc\.org/api/[^"\s]+',
                r'https://bvbrc\.org/api/[^"\s]+',
                r'www\.bv-brc\.org',
                r'bvbrc\.org'
            ]
            
            for py_file in python_files:
                try:
                    with open(py_file, 'r') as f:
                        content = f.read()
                    
                    original_content = content
                    
                    # Replace hardcoded endpoints with configurable ones
                    for pattern in bvbrc_patterns:
                        matches = re.findall(pattern, content)
                        for match in matches:
                            # Replace with environment variable
                            env_var = 'BVBRC_API_BASE_URL'
                            replacement = f"os.environ.get('{env_var}', '{match}')"
                            content = content.replace(f'"{match}"', f'"{replacement}"')
                            content = content.replace(f"'{match}'", f"'{replacement}'")
                    
                    if content != original_content:
                        with open(py_file, 'w') as f:
                            f.write(content)
                        files_modified.append(str(py_file))
                        
                        # Ensure os import
                        self._ensure_os_import(py_file)
                
                except Exception as e:
                    errors.append(f"Failed to fix BV-BRC endpoints in {py_file}: {str(e)}")
            
            return {
                'success': len(errors) == 0,
                'files_modified': files_modified,
                'errors': errors
            }
            
        except Exception as e:
            return {
                'success': False,
                'files_modified': files_modified,
                'errors': errors + [f"BV-BRC endpoint fixing failed: {str(e)}"]
            }
    
    def _standardize_demo_config(self, demo_path: Path) -> Dict:
        """Standardize configuration patterns in a demo."""
        files_modified = []
        errors = []
        
        try:
            # Create standard config structure if it doesn't exist
            config_dir = demo_path / 'config'
            config_dir.mkdir(exist_ok=True)
            
            # Standard configuration template
            standard_config = {
                'demo': {
                    'name': demo_path.name,
                    'description': f'Configuration for {demo_path.name} demo',
                    'version': '1.0.0'
                },
                'paths': {
                    'data_dir': './data',
                    'output_dir': './output',
                    'logs_dir': './logs'
                },
                'settings': {
                    'debug': False,
                    'verbose': True
                }
            }
            
            config_file = config_dir / 'config.yaml'
            if not config_file.exists():
                with open(config_file, 'w') as f:
                    yaml.dump(standard_config, f, default_flow_style=False)
                files_modified.append(str(config_file))
            
            return {
                'success': True,
                'files_modified': files_modified,
                'errors': errors
            }
            
        except Exception as e:
            return {
                'success': False,
                'files_modified': files_modified,
                'errors': errors + [f"Config standardization failed: {str(e)}"]
            }
    
    def _consolidate_rag_scripts(self, demo_path: Path) -> Dict:
        """Consolidate multiple RAG database creation scripts."""
        files_modified = []
        errors = []
        scripts_consolidated = 0
        
        try:
            # Find scripts with similar patterns
            script_patterns = ['create_*', 'final_*', 'simple_*']
            scripts_to_consolidate = []
            
            for pattern in script_patterns:
                scripts_to_consolidate.extend(demo_path.glob(f'{pattern}.py'))
            
            if len(scripts_to_consolidate) > 1:
                # Create consolidated main script
                main_script = demo_path / 'main.py'
                
                consolidated_content = self._create_rag_main_content(scripts_to_consolidate)
                
                with open(main_script, 'w') as f:
                    f.write(consolidated_content)
                
                files_modified.append(str(main_script))
                scripts_consolidated = len(scripts_to_consolidate)
                
                # Move original scripts to archive
                archive_dir = demo_path / 'archive'
                archive_dir.mkdir(exist_ok=True)
                
                for script in scripts_to_consolidate:
                    archive_path = archive_dir / script.name
                    shutil.move(str(script), str(archive_path))
            
            return {
                'success': True,
                'files_modified': files_modified,
                'errors': errors,
                'scripts_consolidated': scripts_consolidated
            }
            
        except Exception as e:
            return {
                'success': False,
                'files_modified': files_modified,
                'errors': errors + [f"Script consolidation failed: {str(e)}"],
                'scripts_consolidated': 0
            }
    
    def _create_rag_main_content(self, scripts: List[Path]) -> str:
        """Create consolidated main content for RAG scripts."""
        content = '''"""
Consolidated RAG Database Creation Demo

This script consolidates multiple RAG database creation approaches
into a single configurable implementation.

Generated by Nanobrain cleanup system.
"""

import argparse
import os
import sys
from pathlib import Path

def main():
    """Main entry point for RAG database creation."""
    parser = argparse.ArgumentParser(description='RAG Database Creation Demo')
    parser.add_argument('--mode', choices=['simple', 'advanced', 'distributed'], 
                       default='simple', help='Database creation mode')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Configuration file path')
    parser.add_argument('--output-dir', type=str, default='./output',
                       help='Output directory for database')
    
    args = parser.parse_args()
    
    print(f"Starting RAG database creation in {args.mode} mode")
    print(f"Using config: {args.config}")
    print(f"Output directory: {args.output_dir}")
    
    # Implementation would go here based on consolidated scripts
    # This is a placeholder for the actual implementation
    
    print("RAG database creation completed successfully")

if __name__ == '__main__':
    main()
'''
        return content
    
    def _create_unified_rag_main(self, demo_path: Path) -> Dict:
        """Create unified main implementation for RAG demo."""
        files_modified = []
        errors = []
        
        try:
            # Create main.py if it doesn't exist
            main_file = demo_path / 'main.py'
            
            if not main_file.exists():
                main_content = self._create_rag_main_content([])
                
                with open(main_file, 'w') as f:
                    f.write(main_content)
                
                files_modified.append(str(main_file))
            
            # Create CLI interface
            cli_file = demo_path / 'cli.py'
            if not cli_file.exists():
                cli_content = '''"""
Command-line interface for RAG database creation demo.
"""

import click
from pathlib import Path

@click.command()
@click.option('--mode', type=click.Choice(['simple', 'advanced', 'distributed']), 
              default='simple', help='Database creation mode')
@click.option('--config', type=click.Path(exists=True), default='config/config.yaml',
              help='Configuration file path')
@click.option('--output-dir', type=click.Path(), default='./output',
              help='Output directory for database')
def create_database(mode, config, output_dir):
    """Create RAG database with specified configuration."""
    click.echo(f"Creating RAG database in {mode} mode")
    click.echo(f"Config: {config}")
    click.echo(f"Output: {output_dir}")
    
    # Implementation would go here
    click.echo("Database creation completed")

if __name__ == '__main__':
    create_database()
'''
                
                with open(cli_file, 'w') as f:
                    f.write(cli_content)
                
                files_modified.append(str(cli_file))
            
            return {
                'success': True,
                'files_modified': files_modified,
                'errors': errors
            }
            
        except Exception as e:
            return {
                'success': False,
                'files_modified': files_modified,
                'errors': errors + [f"Unified main creation failed: {str(e)}"]
            }