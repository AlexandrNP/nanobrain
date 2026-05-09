"""
Dependency management cleanup and synchronization system.

This module provides functionality to analyze, clean up, and synchronize
dependencies between pyproject.toml and requirements.txt files.
"""

import re
import tomllib
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
from dataclasses import dataclass
from packaging.requirements import Requirement

from .models import CleanupResult, ValidationResult


@dataclass
class DependencyInfo:
    """Information about a dependency."""
    name: str
    version_spec: str
    source: str  # 'pyproject', 'requirements', 'both'
    category: str  # 'core', 'optional', 'dev'
    is_installed: bool = False
    is_used: bool = False


@dataclass
class DependencyAnalysis:
    """Analysis results for dependencies."""
    unused_dependencies: List[DependencyInfo]
    missing_dependencies: List[DependencyInfo]
    version_conflicts: List[Tuple[DependencyInfo, DependencyInfo]]
    optional_category_issues: List[str]


class DependencyManager:
    """Manages dependency analysis, cleanup, and synchronization."""
    
    def __init__(self, repo_root: Path):
        """Initialize the dependency manager.
        
        Args:
            repo_root: Root directory of the repository
        """
        self.repo_root = Path(repo_root)
        self.pyproject_path = self.repo_root / 'pyproject.toml'
        self.requirements_path = self.repo_root / 'requirements.txt'
        
        # Core framework modules that should always be considered used
        self.core_modules = {
            'nanobrain.core',
            'nanobrain.library', 
            'nanobrain.cleanup',
            'nanobrain.demos'
        }
        
        # Dependencies that are commonly imported under different names
        self.import_name_mapping = {
            'pyyaml': 'yaml',
            'pillow': 'PIL',
            'beautifulsoup4': 'bs4',
            'python-dateutil': 'dateutil',
            'python-dotenv': 'dotenv',
            'python-multipart': 'multipart',
            'python-jose': 'jose',
            'scikit-learn': 'sklearn',
            'sentence-transformers': 'sentence_transformers',
            'faiss-cpu': 'faiss',
            'faiss-gpu': 'faiss',
            'langchain-community': 'langchain_community',
            'openai': 'openai',
            'anthropic': 'anthropic'
        }
    
    def analyze_dependencies(self) -> DependencyAnalysis:
        """Analyze all dependencies for issues.
        
        Returns:
            DependencyAnalysis with detected issues
        """
        pyproject_deps = self._parse_pyproject_dependencies()
        requirements_deps = self._parse_requirements_dependencies()
        used_imports = self._scan_for_imports()
        
        # Find unused dependencies
        unused_deps = []
        for dep_name, dep_info in pyproject_deps.items():
            import_name = self.import_name_mapping.get(dep_name, dep_name.replace('-', '_'))
            if import_name not in used_imports and dep_name not in self.core_modules:
                unused_deps.append(dep_info)
        
        # Find missing dependencies (in requirements.txt but not pyproject.toml)
        missing_deps = []
        for dep_name, dep_info in requirements_deps.items():
            if dep_name not in pyproject_deps:
                missing_deps.append(dep_info)
        
        # Find version conflicts
        version_conflicts = []
        for dep_name in pyproject_deps:
            if dep_name in requirements_deps:
                pyproject_dep = pyproject_deps[dep_name]
                requirements_dep = requirements_deps[dep_name]
                if pyproject_dep.version_spec != requirements_dep.version_spec:
                    version_conflicts.append((pyproject_dep, requirements_dep))
        
        # Check optional dependency categorization
        optional_issues = self._validate_optional_categories(pyproject_deps)
        
        return DependencyAnalysis(
            unused_dependencies=unused_deps,
            missing_dependencies=missing_deps,
            version_conflicts=version_conflicts,
            optional_category_issues=optional_issues
        )
    
    def cleanup_unused_dependencies(self, analysis: DependencyAnalysis) -> CleanupResult:
        """Remove unused dependencies from pyproject.toml.
        
        Args:
            analysis: Dependency analysis results
            
        Returns:
            CleanupResult with cleanup details
        """
        if not analysis.unused_dependencies:
            return CleanupResult(
                success=True,
                message="No unused dependencies found"
            )
        
        try:
            # Load pyproject.toml
            with open(self.pyproject_path, 'rb') as f:
                pyproject_data = tomllib.load(f)
            
            removed_deps = []
            
            # Remove unused dependencies from core dependencies
            if 'project' in pyproject_data and 'dependencies' in pyproject_data['project']:
                original_deps = pyproject_data['project']['dependencies'][:]
                filtered_deps = []
                
                for dep_spec in original_deps:
                    dep_name = self._extract_package_name(dep_spec)
                    if not any(unused.name == dep_name for unused in analysis.unused_dependencies):
                        filtered_deps.append(dep_spec)
                    else:
                        removed_deps.append(dep_spec)
                
                pyproject_data['project']['dependencies'] = filtered_deps
            
            # Remove from optional dependencies
            if 'project' in pyproject_data and 'optional-dependencies' in pyproject_data['project']:
                for category, deps in pyproject_data['project']['optional-dependencies'].items():
                    if isinstance(deps, list):
                        original_deps = deps[:]
                        filtered_deps = []
                        
                        for dep_spec in original_deps:
                            dep_name = self._extract_package_name(dep_spec)
                            if not any(unused.name == dep_name for unused in analysis.unused_dependencies):
                                filtered_deps.append(dep_spec)
                            else:
                                removed_deps.append(dep_spec)
                        
                        pyproject_data['project']['optional-dependencies'][category] = filtered_deps
            
            # Write back to file (simple TOML writing)
            self._write_pyproject_toml(pyproject_data)
            
            return CleanupResult(
                success=True,
                files_removed=removed_deps,
                files_processed=1,
                message=f"Removed {len(removed_deps)} unused dependencies",
                details={'removed_dependencies': removed_deps}
            )
            
        except Exception as e:
            return CleanupResult(
                success=False,
                errors=[f"Failed to cleanup unused dependencies: {e}"]
            )
    
    def validate_optional_categories(self) -> ValidationResult:
        """Validate that optional dependencies are properly categorized.
        
        Returns:
            ValidationResult with validation details
        """
        try:
            pyproject_deps = self._parse_pyproject_dependencies()
            issues = self._validate_optional_categories(pyproject_deps)
            
            return ValidationResult(
                component="optional_dependencies",
                tests_passed=["Optional dependency structure exists"] if not issues else [],
                tests_failed=issues,
                success=len(issues) == 0,
                error_details=str(issues) if issues else None
            )
            
        except Exception as e:
            return ValidationResult(
                component="optional_dependencies",
                tests_passed=[],
                tests_failed=[f"Validation failed: {e}"],
                success=False,
                error_details=str(e)
            )
    
    def add_graceful_degradation(self) -> CleanupResult:
        """Add graceful degradation for missing optional dependencies.
        
        Returns:
            CleanupResult with implementation details
        """
        degradation_code = '''"""
Graceful degradation utilities for optional dependencies.

This module provides utilities to handle missing optional dependencies gracefully,
allowing the framework to function with reduced capabilities rather than failing.
"""

import importlib
import warnings
from typing import Any, Optional, Callable
from functools import wraps


class OptionalDependencyError(ImportError):
    """Raised when an optional dependency is not available."""
    pass


def optional_import(module_name: str, package_name: str = None) -> Any:
    """Import a module with graceful degradation.
    
    Args:
        module_name: Name of the module to import
        package_name: Name of the package (for better error messages)
        
    Returns:
        The imported module or None if not available
        
    Raises:
        OptionalDependencyError: If the module is required but not available
    """
    try:
        return importlib.import_module(module_name)
    except ImportError:
        package_name = package_name or module_name
        warnings.warn(
            f"Optional dependency '{package_name}' not found. "
            f"Some features may not be available. "
            f"Install with: pip install {package_name}",
            UserWarning
        )
        return None


def requires_optional_dependency(dependency_name: str, install_command: str = None):
    """Decorator to mark functions that require optional dependencies.
    
    Args:
        dependency_name: Name of the required dependency
        install_command: Command to install the dependency
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except (ImportError, ModuleNotFoundError) as e:
                install_cmd = install_command or f"pip install {dependency_name}"
                raise OptionalDependencyError(
                    f"Function '{func.__name__}' requires optional dependency '{dependency_name}'. "
                    f"Install with: {install_cmd}"
                ) from e
        return wrapper
    return decorator


# Optional dependency imports with graceful degradation
def get_llm_client():
    """Get LLM client with graceful degradation."""
    openai = optional_import('openai')
    anthropic = optional_import('anthropic')
    
    if not openai and not anthropic:
        raise OptionalDependencyError(
            "No LLM client available. Install with: pip install 'nanobrain[llm]'"
        )
    
    return {'openai': openai, 'anthropic': anthropic}


def get_vector_store():
    """Get vector store with graceful degradation."""
    faiss = optional_import('faiss', 'faiss-cpu')
    chromadb = optional_import('chromadb')
    
    if not faiss and not chromadb:
        raise OptionalDependencyError(
            "No vector store available. Install with: pip install 'nanobrain[rag]'"
        )
    
    return {'faiss': faiss, 'chromadb': chromadb}


def get_distributed_executor():
    """Get distributed executor with graceful degradation."""
    parsl = optional_import('parsl')
    
    if not parsl:
        raise OptionalDependencyError(
            "Distributed execution not available. Install with: pip install 'nanobrain[distributed]'"
        )
    
    return parsl


# Feature availability checks
def has_llm_support() -> bool:
    """Check if LLM support is available."""
    try:
        get_llm_client()
        return True
    except OptionalDependencyError:
        return False


def has_rag_support() -> bool:
    """Check if RAG support is available."""
    try:
        get_vector_store()
        return True
    except OptionalDependencyError:
        return False


def has_distributed_support() -> bool:
    """Check if distributed execution support is available."""
    try:
        get_distributed_executor()
        return True
    except OptionalDependencyError:
        return False


def check_feature_availability() -> dict:
    """Check availability of all optional features.
    
    Returns:
        Dictionary mapping feature names to availability status
    """
    return {
        'llm': has_llm_support(),
        'rag': has_rag_support(),
        'distributed': has_distributed_support(),
    }
'''
        
        try:
            # Create graceful degradation module
            degradation_path = self.repo_root / 'nanobrain' / 'core' / 'optional_deps.py'
            degradation_path.parent.mkdir(parents=True, exist_ok=True)
            degradation_path.write_text(degradation_code)
            
            return CleanupResult(
                success=True,
                files_processed=1,
                message="Added graceful degradation for optional dependencies",
                details={'created_file': str(degradation_path)}
            )
            
        except Exception as e:
            return CleanupResult(
                success=False,
                errors=[f"Failed to add graceful degradation: {e}"]
            )
    
    def synchronize_requirements(self) -> CleanupResult:
        """Synchronize requirements.txt with pyproject.toml.
        
        Returns:
            CleanupResult with synchronization details
        """
        try:
            pyproject_deps = self._parse_pyproject_dependencies()
            
            # Generate new requirements.txt content
            requirements_content = self._generate_requirements_content(pyproject_deps)
            
            # Write to requirements.txt
            with open(self.requirements_path, 'w') as f:
                f.write(requirements_content)
            
            return CleanupResult(
                success=True,
                files_processed=1,
                message="Synchronized requirements.txt with pyproject.toml",
                details={
                    'total_dependencies': len(pyproject_deps),
                    'requirements_file': str(self.requirements_path)
                }
            )
            
        except Exception as e:
            return CleanupResult(
                success=False,
                errors=[f"Failed to synchronize requirements: {e}"]
            )
    
    def validate_installation(self) -> ValidationResult:
        """Validate that the framework installs correctly with minimal dependencies.
        
        Returns:
            ValidationResult with installation validation details
        """
        try:
            # Test core imports
            core_imports = [
                'nanobrain',
                'nanobrain.core',
                'nanobrain.cleanup',
                'pydantic',
                'yaml',
                'jsonschema'
            ]
            
            failed_imports = []
            successful_imports = []
            
            for module_name in core_imports:
                try:
                    __import__(module_name)
                    successful_imports.append(module_name)
                except ImportError as e:
                    failed_imports.append(f"{module_name}: {e}")
            
            # Test optional imports with graceful degradation
            optional_tests = []
            try:
                from nanobrain.core.optional_deps import check_feature_availability
                feature_status = check_feature_availability()
                optional_tests.append(f"Feature availability check: {feature_status}")
            except ImportError:
                optional_tests.append("Graceful degradation not yet implemented")
            
            success = len(failed_imports) == 0
            
            return ValidationResult(
                component="framework_installation",
                tests_passed=successful_imports + optional_tests,
                tests_failed=failed_imports,
                success=success,
                error_details=str(failed_imports) if failed_imports else None
            )
            
        except Exception as e:
            return ValidationResult(
                component="framework_installation",
                tests_passed=[],
                tests_failed=[f"Installation validation failed: {e}"],
                success=False,
                error_details=str(e)
            )
    
    def _parse_pyproject_dependencies(self) -> Dict[str, DependencyInfo]:
        """Parse dependencies from pyproject.toml."""
        dependencies = {}
        
        if not self.pyproject_path.exists():
            return dependencies
        
        try:
            with open(self.pyproject_path, 'rb') as f:
                pyproject_data = tomllib.load(f)
            
            # Parse core dependencies first
            if 'project' in pyproject_data and 'dependencies' in pyproject_data['project']:
                for dep_spec in pyproject_data['project']['dependencies']:
                    dep_name = self._extract_package_name(dep_spec)
                    dependencies[dep_name] = DependencyInfo(
                        name=dep_name,
                        version_spec=dep_spec,
                        source='pyproject',
                        category='core'
                    )
            
            # Parse optional dependencies (may override core if same package)
            if 'project' in pyproject_data and 'optional-dependencies' in pyproject_data['project']:
                for category, deps in pyproject_data['project']['optional-dependencies'].items():
                    if isinstance(deps, list):
                        for dep_spec in deps:
                            dep_name = self._extract_package_name(dep_spec)
                            # If already exists as core, keep as core but note it's also optional
                            if dep_name in dependencies and dependencies[dep_name].category == 'core':
                                # Keep the core dependency but update version if different
                                continue
                            else:
                                dependencies[dep_name] = DependencyInfo(
                                    name=dep_name,
                                    version_spec=dep_spec,
                                    source='pyproject',
                                    category=category
                                )
            
        except Exception:
            # Return empty dict on error, let caller handle
            pass
        
        return dependencies
    
    def _parse_requirements_dependencies(self) -> Dict[str, DependencyInfo]:
        """Parse dependencies from requirements.txt."""
        dependencies = {}
        
        if not self.requirements_path.exists():
            return dependencies
        
        try:
            with open(self.requirements_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        dep_name = self._extract_package_name(line)
                        if dep_name:
                            dependencies[dep_name] = DependencyInfo(
                                name=dep_name,
                                version_spec=line,
                                source='requirements',
                                category='unknown'
                            )
        except Exception:
            # Return empty dict on error
            pass
        
        return dependencies
    
    def _extract_package_name(self, dep_spec: str) -> str:
        """Extract package name from dependency specification."""
        try:
            req = Requirement(dep_spec)
            return req.name
        except Exception:
            # Fallback to simple parsing
            # Split on all version operators
            name = re.split(r'[><=!~]', dep_spec)[0]
            return name.strip()
    
    def _scan_for_imports(self) -> Set[str]:
        """Scan Python files for import statements."""
        imports = set()
        
        # Scan nanobrain package
        nanobrain_dir = self.repo_root / 'nanobrain'
        if nanobrain_dir.exists():
            for py_file in nanobrain_dir.rglob('*.py'):
                try:
                    content = py_file.read_text(encoding='utf-8')
                    file_imports = self._extract_imports_from_content(content)
                    imports.update(file_imports)
                except Exception:
                    continue
        
        # Scan demos
        demos_dir = self.repo_root / 'demos'
        if demos_dir.exists():
            for py_file in demos_dir.rglob('*.py'):
                try:
                    content = py_file.read_text(encoding='utf-8')
                    file_imports = self._extract_imports_from_content(content)
                    imports.update(file_imports)
                except Exception:
                    continue
        
        return imports
    
    def _extract_imports_from_content(self, content: str) -> Set[str]:
        """Extract import statements from Python content."""
        imports = set()
        
        # Match import statements at the beginning of lines (not indented)
        import_patterns = [
            r'^import\s+([a-zA-Z_][a-zA-Z0-9_\.]*)',
            r'^from\s+([a-zA-Z_][a-zA-Z0-9_\.]*)\s+import',
        ]
        
        lines = content.split('\n')
        for i, line in enumerate(lines):
            original_line = line
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Check if this is a top-level import (not indented)
            if original_line.startswith(' ') or original_line.startswith('\t'):
                continue
                
            for pattern in import_patterns:
                match = re.match(pattern, line)
                if match:
                    module_name = match.group(1)
                    # Get top-level module name
                    top_level = module_name.split('.')[0]
                    imports.add(top_level)
        
        return imports
    
    def _validate_optional_categories(self, dependencies: Dict[str, DependencyInfo]) -> List[str]:
        """Validate optional dependency categorization."""
        issues = []
        
        # Expected categories and their typical dependencies
        expected_categories = {
            'llm': ['openai', 'anthropic', 'langchain'],
            'distributed': ['parsl', 'dill'],
            'rag': ['sentence-transformers', 'faiss-cpu', 'transformers', 'torch'],
            'dev': ['pytest', 'black', 'flake8', 'mypy']
        }
        
        # Check if categories exist and have appropriate dependencies
        optional_deps = {dep.name: dep for dep in dependencies.values() if dep.category != 'core'}
        
        for category, expected_deps in expected_categories.items():
            category_deps = [dep for dep in optional_deps.values() if dep.category == category]
            
            if not category_deps:
                issues.append(f"Missing optional category: {category}")
                continue
            
            # Check if expected dependencies are in the right category
            for expected_dep in expected_deps:
                if expected_dep in optional_deps:
                    actual_category = optional_deps[expected_dep].category
                    if actual_category != category:
                        issues.append(f"Dependency '{expected_dep}' in wrong category: {actual_category} (should be {category})")
        
        return issues
    
    def _generate_requirements_content(self, dependencies: Dict[str, DependencyInfo]) -> str:
        """Generate requirements.txt content from dependencies."""
        content = """# Nanobrain Framework Dependencies
# =====================================
# Auto-generated from pyproject.toml - DO NOT EDIT MANUALLY
# Use 'nanobrain-cleanup sync-deps' to regenerate this file

# Core Dependencies
"""
        
        # Add core dependencies
        core_deps = [dep for dep in dependencies.values() if dep.category == 'core']
        for dep in sorted(core_deps, key=lambda x: x.name):
            content += f"{dep.version_spec}\n"
        
        # Add optional dependencies by category
        categories = set(dep.category for dep in dependencies.values() if dep.category != 'core')
        
        for category in sorted(categories):
            if category == 'unknown':
                continue
                
            content += f"\n# Optional: {category.title()} Dependencies\n"
            content += f"# Install with: pip install 'nanobrain[{category}]'\n"
            
            category_deps = [dep for dep in dependencies.values() if dep.category == category]
            for dep in sorted(category_deps, key=lambda x: x.name):
                content += f"# {dep.version_spec}\n"
        
        content += """
# Installation Instructions:
# =========================
# Core only:        pip install nanobrain
# With LLM:         pip install 'nanobrain[llm]'
# With RAG:         pip install 'nanobrain[rag]'
# With distributed: pip install 'nanobrain[distributed]'
# Everything:       pip install 'nanobrain[all]'
"""
        
        return content
    
    def _write_pyproject_toml(self, data: Dict[str, Any]) -> None:
        """Write pyproject.toml data back to file.
        
        Simple TOML writer for basic pyproject.toml structure.
        """
        lines = []
        
        # Write build-system section if it exists
        if 'build-system' in data:
            lines.append('[build-system]')
            build_system = data['build-system']
            if 'requires' in build_system:
                requires_str = json.dumps(build_system['requires'])
                lines.append(f'requires = {requires_str}')
            if 'build-backend' in build_system:
                lines.append(f'build-backend = "{build_system["build-backend"]}"')
            lines.append('')
        
        # Write project section
        if 'project' in data:
            lines.append('[project]')
            project = data['project']
            
            # Basic project metadata
            for key in ['name', 'version', 'description']:
                if key in project:
                    lines.append(f'{key} = "{project[key]}"')
            
            # Dependencies
            if 'dependencies' in project:
                lines.append('dependencies = [')
                for dep in project['dependencies']:
                    lines.append(f'    "{dep}",')
                lines.append(']')
            
            lines.append('')
            
            # Optional dependencies
            if 'optional-dependencies' in project:
                lines.append('[project.optional-dependencies]')
                for category, deps in project['optional-dependencies'].items():
                    lines.append(f'{category} = [')
                    for dep in deps:
                        lines.append(f'    "{dep}",')
                    lines.append(']')
                lines.append('')
        
        # Write other sections as-is (simplified)
        for section_name, section_data in data.items():
            if section_name not in ['build-system', 'project']:
                if isinstance(section_data, dict):
                    lines.append(f'[{section_name}]')
                    # Simple key-value pairs
                    for key, value in section_data.items():
                        if isinstance(value, str):
                            lines.append(f'{key} = "{value}"')
                        elif isinstance(value, (int, float, bool)):
                            lines.append(f'{key} = {value}')
                    lines.append('')
        
        # Write to file
        with open(self.pyproject_path, 'w') as f:
            f.write('\n'.join(lines))