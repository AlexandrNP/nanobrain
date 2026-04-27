"""
Import resolution system for the Nanobrain framework cleanup.

This module provides functionality to detect and resolve import issues including:
- Circular import dependencies
- Missing __init__.py files
- Hardcoded import paths
"""

import ast
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass
import logging

from .models import CleanupResult


@dataclass
class CleanupError:
    """Represents an error during cleanup operations."""
    error_type: str
    message: str


@dataclass
class ImportIssue:
    """Represents an import-related issue found during analysis."""
    file_path: Path
    issue_type: str  # 'circular', 'missing_init', 'hardcoded_path'
    description: str
    line_number: Optional[int] = None
    suggested_fix: Optional[str] = None


@dataclass
class CircularDependency:
    """Represents a circular import dependency."""
    cycle: List[Path]
    import_chain: List[Tuple[Path, str, int]]  # (file, import_statement, line_number)


class ImportAnalyzer(ast.NodeVisitor):
    """AST visitor to analyze import statements in Python files."""
    
    def __init__(self, file_path: Path, project_root: Path):
        self.file_path = file_path
        self.project_root = project_root
        self.imports: List[Tuple[str, int]] = []  # (import_name, line_number)
        self.from_imports: List[Tuple[str, str, int]] = []  # (module, name, line_number)
        
    def visit_Import(self, node: ast.Import) -> None:
        """Visit import statements."""
        for alias in node.names:
            self.imports.append((alias.name, node.lineno))
        self.generic_visit(node)
        
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Visit from...import statements."""
        if node.module:
            for alias in node.names:
                name = alias.name if alias.name != '*' else '*'
                self.from_imports.append((node.module, name, node.lineno))
        self.generic_visit(node)


class ImportResolver:
    """
    Resolves import issues in the Nanobrain framework.
    
    This class analyzes Python files to detect and fix:
    - Circular import dependencies
    - Missing __init__.py files
    - Hardcoded import paths
    """
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.logger = logging.getLogger(__name__)
        self.issues: List[ImportIssue] = []
        self.dependency_graph: Dict[Path, Set[Path]] = {}
        
    def analyze_imports(self) -> CleanupResult:
        """
        Analyze all Python files for import issues.
        
        Returns:
            CleanupResult with analysis results and detected issues
        """
        try:
            self.logger.info("Starting import analysis")
            self.issues.clear()
            self.dependency_graph.clear()
            
            # Find all Python files
            python_files = self._find_python_files()
            self.logger.info(f"Found {len(python_files)} Python files to analyze")
            
            # Analyze each file
            for py_file in python_files:
                self._analyze_file(py_file)
                
            # Detect circular dependencies
            circular_deps = self._detect_circular_dependencies()
            for cycle in circular_deps:
                self._add_circular_dependency_issue(cycle)
                
            # Detect missing __init__.py files
            self._detect_missing_init_files()
            
            # Detect hardcoded paths
            self._detect_hardcoded_paths()
            
            self.logger.info(f"Import analysis complete. Found {len(self.issues)} issues")
            
            return CleanupResult(
                success=True,
                message=f"Import analysis complete. Found {len(self.issues)} issues",
                files_processed=len(python_files),
                details={
                    'issues_found': len(self.issues),
                    'circular_dependencies': len([i for i in self.issues if i.issue_type == 'circular']),
                    'missing_init_files': len([i for i in self.issues if i.issue_type == 'missing_init']),
                    'hardcoded_paths': len([i for i in self.issues if i.issue_type == 'hardcoded_path'])
                }
            )
            
        except Exception as e:
            error_msg = f"Import analysis failed: {str(e)}"
            self.logger.error(error_msg)
            return CleanupResult(
                success=False,
                message=error_msg,
                errors=[error_msg]
            )
    
    def resolve_circular_dependencies(self) -> CleanupResult:
        """
        Resolve circular import dependencies through refactoring.
        
        Returns:
            CleanupResult with resolution results
        """
        try:
            circular_issues = [i for i in self.issues if i.issue_type == 'circular']
            if not circular_issues:
                return CleanupResult(
                    success=True,
                    message="No circular dependencies to resolve"
                )
                
            self.logger.info(f"Resolving {len(circular_issues)} circular dependencies")
            resolved_count = 0
            
            for issue in circular_issues:
                if self._resolve_circular_dependency(issue):
                    resolved_count += 1
                    
            return CleanupResult(
                success=True,
                message=f"Resolved {resolved_count}/{len(circular_issues)} circular dependencies",
                files_processed=resolved_count,
                details={'resolved_count': resolved_count, 'total_issues': len(circular_issues)}
            )
            
        except Exception as e:
            error_msg = f"Circular dependency resolution failed: {str(e)}"
            self.logger.error(error_msg)
            return CleanupResult(
                success=False,
                message=error_msg,
                errors=[error_msg]
            )
    
    def add_missing_init_files(self) -> CleanupResult:
        """
        Add missing __init__.py files with appropriate exports.
        
        Returns:
            CleanupResult with creation results
        """
        try:
            missing_init_issues = [i for i in self.issues if i.issue_type == 'missing_init']
            if not missing_init_issues:
                return CleanupResult(
                    success=True,
                    message="No missing __init__.py files to create"
                )
                
            self.logger.info(f"Creating {len(missing_init_issues)} missing __init__.py files")
            created_count = 0
            
            for issue in missing_init_issues:
                if self._create_init_file(issue):
                    created_count += 1
                    
            return CleanupResult(
                success=True,
                message=f"Created {created_count}/{len(missing_init_issues)} __init__.py files",
                files_processed=created_count,
                details={'created_count': created_count, 'total_missing': len(missing_init_issues)}
            )
            
        except Exception as e:
            error_msg = f"__init__.py file creation failed: {str(e)}"
            self.logger.error(error_msg)
            return CleanupResult(
                success=False,
                message=error_msg,
                errors=[error_msg]
            )
    
    def fix_hardcoded_paths(self) -> CleanupResult:
        """
        Replace hardcoded import paths with configurable path resolution.
        
        Returns:
            CleanupResult with fix results
        """
        try:
            hardcoded_issues = [i for i in self.issues if i.issue_type == 'hardcoded_path']
            if not hardcoded_issues:
                return CleanupResult(
                    success=True,
                    message="No hardcoded paths to fix"
                )
                
            self.logger.info(f"Fixing {len(hardcoded_issues)} hardcoded import paths")
            fixed_count = 0
            
            for issue in hardcoded_issues:
                if self._fix_hardcoded_path(issue):
                    fixed_count += 1
                    
            return CleanupResult(
                success=True,
                message=f"Fixed {fixed_count}/{len(hardcoded_issues)} hardcoded paths",
                files_processed=fixed_count,
                details={'fixed_count': fixed_count, 'total_issues': len(hardcoded_issues)}
            )
            
        except Exception as e:
            error_msg = f"Hardcoded path fixing failed: {str(e)}"
            self.logger.error(error_msg)
            return CleanupResult(
                success=False,
                message=error_msg,
                errors=[error_msg]
            )
    
    def validate_imports(self) -> CleanupResult:
        """
        Validate that all imports work correctly after fixes.
        
        Returns:
            CleanupResult with validation results
        """
        try:
            self.logger.info("Validating import resolution")
            
            # Re-analyze to check for remaining issues
            analysis_result = self.analyze_imports()
            if not analysis_result.success:
                return analysis_result
                
            # Try importing key modules
            validation_errors = []
            key_modules = self._identify_key_modules()
            
            for module_path in key_modules:
                try:
                    self._test_import_module(module_path)
                except ImportError as e:
                    validation_errors.append(f"Failed to import {module_path}: {str(e)}")
                    
            if validation_errors:
                return CleanupResult(
                    success=False,
                    message=f"Import validation failed with {len(validation_errors)} errors",
                    errors=validation_errors
                )
                
            return CleanupResult(
                success=True,
                message=f"Import validation successful. {len(self.issues)} remaining issues",
                details={'remaining_issues': len(self.issues), 'validated_modules': len(key_modules)}
            )
            
        except Exception as e:
            error_msg = f"Import validation failed: {str(e)}"
            self.logger.error(error_msg)
            return CleanupResult(
                success=False,
                message=error_msg,
                errors=[error_msg]
            )
    
    def get_issues(self) -> List[ImportIssue]:
        """Get all detected import issues."""
        return self.issues.copy()
    
    def _find_python_files(self) -> List[Path]:
        """Find all Python files in the project."""
        python_files = []
        for root, dirs, files in os.walk(self.project_root):
            # Skip common non-source directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'build', 'dist']]
            
            for file in files:
                if file.endswith('.py'):
                    python_files.append(Path(root) / file)
                    
        return python_files
    
    def _analyze_file(self, file_path: Path) -> None:
        """Analyze a single Python file for imports."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            tree = ast.parse(content, filename=str(file_path))
            analyzer = ImportAnalyzer(file_path, self.project_root)
            analyzer.visit(tree)
            
            # Build dependency graph
            dependencies = set()
            for import_name, _ in analyzer.imports:
                dep_path = self._resolve_import_to_path(import_name, file_path)
                if dep_path:
                    dependencies.add(dep_path)
                    
            for module, _, _ in analyzer.from_imports:
                dep_path = self._resolve_import_to_path(module, file_path)
                if dep_path:
                    dependencies.add(dep_path)
                    
            self.dependency_graph[file_path] = dependencies
            
        except (SyntaxError, UnicodeDecodeError) as e:
            self.logger.warning(f"Could not parse {file_path}: {str(e)}")
    
    def _resolve_import_to_path(self, import_name: str, from_file: Path) -> Optional[Path]:
        """Resolve an import name to a file path."""
        if not import_name.startswith('nanobrain'):
            return None
            
        # Convert import name to file path
        parts = import_name.split('.')
        potential_path = self.project_root
        
        for part in parts:
            potential_path = potential_path / part
            
        # Check for module file
        if (potential_path.with_suffix('.py')).exists():
            return potential_path.with_suffix('.py')
            
        # Check for package
        if (potential_path / '__init__.py').exists():
            return potential_path / '__init__.py'
            
        return None
    
    def _detect_circular_dependencies(self) -> List[CircularDependency]:
        """Detect circular dependencies using DFS."""
        visited = set()
        rec_stack = set()
        cycles = []
        
        def dfs(node: Path, path: List[Path]) -> None:
            if node in rec_stack:
                # Found a cycle
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                cycles.append(CircularDependency(cycle=cycle, import_chain=[]))
                return
                
            if node in visited:
                return
                
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for dependency in self.dependency_graph.get(node, set()):
                dfs(dependency, path)
                
            path.pop()
            rec_stack.remove(node)
        
        for file_path in self.dependency_graph:
            if file_path not in visited:
                dfs(file_path, [])
                
        return cycles
    
    def _add_circular_dependency_issue(self, cycle: CircularDependency) -> None:
        """Add a circular dependency issue."""
        cycle_str = " -> ".join([str(p.relative_to(self.project_root)) for p in cycle.cycle])
        
        for file_path in cycle.cycle:
            issue = ImportIssue(
                file_path=file_path,
                issue_type='circular',
                description=f"Part of circular dependency: {cycle_str}",
                suggested_fix="Consider moving shared code to a separate module or using lazy imports"
            )
            self.issues.append(issue)
    
    def _detect_missing_init_files(self) -> None:
        """Detect directories that should have __init__.py files."""
        for root, dirs, files in os.walk(self.project_root / 'nanobrain'):
            # Skip non-package directories
            if any(skip in str(root) for skip in ['__pycache__', '.git', 'build', 'dist']):
                continue
                
            # Check if directory contains Python files but no __init__.py
            has_python_files = any(f.endswith('.py') for f in files)
            has_init = '__init__.py' in files
            
            if has_python_files and not has_init:
                init_path = Path(root) / '__init__.py'
                issue = ImportIssue(
                    file_path=init_path,
                    issue_type='missing_init',
                    description=f"Missing __init__.py in package directory {root}",
                    suggested_fix="Create __init__.py with appropriate exports"
                )
                self.issues.append(issue)
    
    def _detect_hardcoded_paths(self) -> None:
        """Detect hardcoded paths in import statements."""
        hardcoded_patterns = [
            '/lus/flare/projects/FoundEpidem',
            '/home/onarykov',
            'C:\\',
            '/Users/',
            '/opt/',
            'sys.path.append'
        ]
        
        for file_path in self.dependency_graph:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    
                for line_num, line in enumerate(lines, 1):
                    for pattern in hardcoded_patterns:
                        if pattern in line and ('import' in line or 'sys.path' in line):
                            issue = ImportIssue(
                                file_path=file_path,
                                issue_type='hardcoded_path',
                                description=f"Hardcoded path detected: {pattern}",
                                line_number=line_num,
                                suggested_fix="Replace with relative imports or configurable paths"
                            )
                            self.issues.append(issue)
                            
            except (UnicodeDecodeError, IOError) as e:
                self.logger.warning(f"Could not read {file_path}: {str(e)}")
    
    def _resolve_circular_dependency(self, issue: ImportIssue) -> bool:
        """Attempt to resolve a circular dependency."""
        # This is a complex operation that would require sophisticated refactoring
        # For now, we'll just log the issue and suggest manual resolution
        self.logger.warning(f"Circular dependency in {issue.file_path} requires manual resolution")
        return False
    
    def _create_init_file(self, issue: ImportIssue) -> bool:
        """Create a missing __init__.py file."""
        try:
            init_path = issue.file_path
            package_dir = init_path.parent
            
            # Find Python modules in the directory
            modules = []
            for file in package_dir.glob('*.py'):
                if file.name != '__init__.py':
                    module_name = file.stem
                    modules.append(module_name)
            
            # Create basic __init__.py content
            content = f'"""\n{package_dir.name} package.\n"""\n\n'
            
            if modules:
                content += "# Import main modules\n"
                for module in sorted(modules):
                    content += f"from .{module} import *\n"
                content += "\n"
                content += f"__all__ = {sorted(modules)}\n"
            
            with open(init_path, 'w', encoding='utf-8') as f:
                f.write(content)
                
            self.logger.info(f"Created {init_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create {issue.file_path}: {str(e)}")
            return False
    
    def _fix_hardcoded_path(self, issue: ImportIssue) -> bool:
        """Fix a hardcoded path in an import statement."""
        try:
            with open(issue.file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if issue.line_number and issue.line_number <= len(lines):
                line = lines[issue.line_number - 1]
                
                # Simple replacements for common hardcoded paths
                replacements = {
                    '/lus/flare/projects/FoundEpidem/onarykov/nanobrain': '.',
                    '# # sys.path.append(': '# # # sys.path.append(',
                }
                
                modified = False
                for old, new in replacements.items():
                    if old in line:
                        lines[issue.line_number - 1] = line.replace(old, new)
                        modified = True
                        break
                
                if modified:
                    with open(issue.file_path, 'w', encoding='utf-8') as f:
                        f.writelines(lines)
                    self.logger.info(f"Fixed hardcoded path in {issue.file_path}:{issue.line_number}")
                    return True
                    
        except Exception as e:
            self.logger.error(f"Failed to fix hardcoded path in {issue.file_path}: {str(e)}")
            
        return False
    
    def _identify_key_modules(self) -> List[str]:
        """Identify key modules that should be importable."""
        key_modules = [
            'nanobrain.core',
            'nanobrain.cleanup',
            'nanobrain.library'
        ]
        
        # Add modules that exist
        existing_modules = []
        for module in key_modules:
            module_path = self.project_root
            for part in module.split('.'):
                module_path = module_path / part
                
            if (module_path / '__init__.py').exists() or module_path.with_suffix('.py').exists():
                existing_modules.append(module)
                
        return existing_modules
    
    def _test_import_module(self, module_name: str) -> None:
        """Test importing a module."""
        # Add project root to Python path temporarily
        original_path = sys.path.copy()
        try:
            if str(self.project_root) not in sys.path:
                sys.path.insert(0, str(self.project_root))
            
            __import__(module_name)
            
        finally:
            sys.path[:] = original_path