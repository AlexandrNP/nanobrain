"""
Repository structure reorganization and management system.

This module provides functionality to reorganize the repository structure,
enforce consistent naming conventions, and manage package hierarchies.
"""

import re
import ast
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass
import shutil

from .models import CleanupResult, ValidationResult


@dataclass
class StructureRule:
    """Represents a repository structure rule."""
    path_pattern: str
    required: bool
    description: str
    file_type: str  # 'directory', 'file', 'package'


@dataclass
class ImportUpdate:
    """Represents an import statement update."""
    file_path: Path
    old_import: str
    new_import: str
    line_number: int


@dataclass
class StructureAnalysis:
    """Analysis results for repository structure."""
    violations: List[str]
    missing_init_files: List[Path]
    incorrect_naming: List[Tuple[Path, str]]  # (path, suggested_name)
    import_updates_needed: List[ImportUpdate]
    archive_candidates: List[Path]


class StructureManager:
    """Manages repository structure reorganization and validation."""
    
    def __init__(self, repo_root: Path):
        """Initialize the structure manager.
        
        Args:
            repo_root: Root directory of the repository
        """
        self.repo_root = Path(repo_root)
        
        # Define the target repository structure
        self.target_structure = {
            # Core framework structure
            'nanobrain/': StructureRule('nanobrain/', True, 'Core framework package', 'package'),
            'nanobrain/__init__.py': StructureRule('nanobrain/__init__.py', True, 'Framework root init', 'file'),
            'nanobrain/core/': StructureRule('nanobrain/core/', True, 'Core framework components', 'package'),
            'nanobrain/core/__init__.py': StructureRule('nanobrain/core/__init__.py', True, 'Core package init', 'file'),
            'nanobrain/library/': StructureRule('nanobrain/library/', True, 'Reusable components', 'package'),
            'nanobrain/library/__init__.py': StructureRule('nanobrain/library/__init__.py', True, 'Library package init', 'file'),
            'nanobrain/cleanup/': StructureRule('nanobrain/cleanup/', True, 'Cleanup tools', 'package'),
            'nanobrain/cleanup/__init__.py': StructureRule('nanobrain/cleanup/__init__.py', True, 'Cleanup package init', 'file'),
            
            # Demo structure
            'demos/': StructureRule('demos/', True, 'Demo implementations', 'directory'),
            'demos/__init__.py': StructureRule('demos/__init__.py', True, 'Demos package init', 'file'),
            'demos/viral_pssm_workflow/': StructureRule('demos/viral_pssm_workflow/', True, 'Target demo 1', 'directory'),
            'demos/rag_database_creation/': StructureRule('demos/rag_database_creation/', True, 'Target demo 2', 'directory'),
            'demos/academylink_aurora_demo/': StructureRule('demos/academylink_aurora_demo/', True, 'Reference demo 1', 'directory'),
            'demos/simple_demo/': StructureRule('demos/simple_demo/', True, 'Reference demo 2', 'directory'),
            
            # Documentation structure
            'docs/': StructureRule('docs/', True, 'Documentation', 'directory'),
            'docs/archive/': StructureRule('docs/archive/', False, 'Archived documentation', 'directory'),
            
            # Configuration and metadata
            'pyproject.toml': StructureRule('pyproject.toml', True, 'Project configuration', 'file'),
            'requirements.txt': StructureRule('requirements.txt', True, 'Dependencies', 'file'),
            'README.md': StructureRule('README.md', True, 'Main documentation', 'file'),
            
            # Test structure
            'tests/': StructureRule('tests/', True, 'Test suite', 'directory'),
            'tests/unit/': StructureRule('tests/unit/', True, 'Unit tests', 'directory'),
            'tests/integration/': StructureRule('tests/integration/', True, 'Integration tests', 'directory'),
        }
        
        # Naming conventions
        self.naming_conventions = {
            'package': re.compile(r'^[a-z][a-z0-9_]*$'),  # lowercase with underscores
            'module': re.compile(r'^[a-z][a-z0-9_]*\.py$'),  # lowercase with underscores
            'class': re.compile(r'^[A-Z][a-zA-Z0-9]*$'),  # PascalCase
            'function': re.compile(r'^[a-z][a-z0-9_]*$'),  # lowercase with underscores
            'constant': re.compile(r'^[A-Z][A-Z0-9_]*$'),  # UPPERCASE with underscores
        }
        
        # Archive directory
        self.archive_dir = self.repo_root / 'docs' / 'archive'
    
    def analyze_structure(self) -> StructureAnalysis:
        """Analyze current repository structure against target.
        
        Returns:
            StructureAnalysis with detected issues
        """
        violations = []
        missing_init_files = []
        incorrect_naming = []
        import_updates_needed = []
        archive_candidates = []
        
        # Check required structure elements
        for path, rule in self.target_structure.items():
            full_path = self.repo_root / path
            
            if rule.required and not full_path.exists():
                violations.append(f"Missing required {rule.file_type}: {path}")
        
        # Find missing __init__.py files
        missing_init_files = self._find_missing_init_files()
        
        # Check naming conventions
        incorrect_naming = self._check_naming_conventions()
        
        # Find import statements that need updating
        import_updates_needed = self._find_import_updates()
        
        # Find archive candidates
        archive_candidates = self._find_archive_candidates()
        
        return StructureAnalysis(
            violations=violations,
            missing_init_files=missing_init_files,
            incorrect_naming=incorrect_naming,
            import_updates_needed=import_updates_needed,
            archive_candidates=archive_candidates
        )
    
    def separate_core_and_demos(self) -> CleanupResult:
        """Separate core framework from demo implementations.
        
        Returns:
            CleanupResult with separation details
        """
        try:
            moved_files = []
            created_dirs = []
            errors = []
            
            # Ensure core framework structure exists
            core_dirs = ['nanobrain/core', 'nanobrain/library', 'nanobrain/cleanup']
            for core_dir in core_dirs:
                dir_path = self.repo_root / core_dir
                if not dir_path.exists():
                    dir_path.mkdir(parents=True, exist_ok=True)
                    created_dirs.append(str(dir_path))
            
            # Ensure demos structure exists
            demos_dir = self.repo_root / 'demos'
            if not demos_dir.exists():
                demos_dir.mkdir(parents=True, exist_ok=True)
                created_dirs.append(str(demos_dir))
            
            # Move any misplaced core files
            misplaced_core_files = self._find_misplaced_core_files()
            for src_path, dest_path in misplaced_core_files:
                try:
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(src_path), str(dest_path))
                    moved_files.append(f"{src_path} -> {dest_path}")
                except Exception as e:
                    errors.append(f"Failed to move {src_path}: {e}")
            
            return CleanupResult(
                success=len(errors) == 0,
                files_processed=len(moved_files) + len(created_dirs),
                message="Separated core framework and demos",
                details={
                    'moved_files': moved_files,
                    'created_directories': created_dirs,
                    'errors': errors
                }
            )
            
        except Exception as e:
            return CleanupResult(
                success=False,
                errors=[f"Failed to separate core and demos: {e}"]
            )
    
    def enforce_naming_conventions(self) -> CleanupResult:
        """Enforce consistent directory and file naming conventions.
        
        Returns:
            CleanupResult with naming enforcement details
        """
        try:
            renamed_items = []
            errors = []
            
            analysis = self.analyze_structure()
            
            for path, suggested_name in analysis.incorrect_naming:
                try:
                    new_path = path.parent / suggested_name
                    if not new_path.exists():
                        path.rename(new_path)
                        renamed_items.append(f"{path.name} -> {suggested_name}")
                    else:
                        errors.append(f"Cannot rename {path.name}: {suggested_name} already exists")
                except Exception as e:
                    errors.append(f"Failed to rename {path.name}: {e}")
            
            return CleanupResult(
                success=len(errors) == 0,
                files_processed=len(renamed_items),
                message="Enforced naming conventions",
                details={
                    'renamed_items': renamed_items,
                    'errors': errors
                }
            )
            
        except Exception as e:
            return CleanupResult(
                success=False,
                errors=[f"Failed to enforce naming conventions: {e}"]
            )
    
    def create_package_hierarchies(self) -> CleanupResult:
        """Create proper package hierarchies with __init__.py files.
        
        Returns:
            CleanupResult with package creation details
        """
        try:
            created_files = []
            errors = []
            
            analysis = self.analyze_structure()
            
            for init_path in analysis.missing_init_files:
                try:
                    # Generate appropriate __init__.py content
                    init_content = self._generate_init_content(init_path)
                    init_path.write_text(init_content, encoding='utf-8')
                    created_files.append(str(init_path))
                except Exception as e:
                    errors.append(f"Failed to create {init_path}: {e}")
            
            return CleanupResult(
                success=len(errors) == 0,
                files_processed=len(created_files),
                message="Created package hierarchies",
                details={
                    'created_files': created_files,
                    'errors': errors
                }
            )
            
        except Exception as e:
            return CleanupResult(
                success=False,
                errors=[f"Failed to create package hierarchies: {e}"]
            )
    
    def organize_archive_content(self) -> CleanupResult:
        """Organize archived content into dedicated archive directory.
        
        Returns:
            CleanupResult with archive organization details
        """
        try:
            archived_files = []
            errors = []
            
            # Ensure archive directory exists
            self.archive_dir.mkdir(parents=True, exist_ok=True)
            
            analysis = self.analyze_structure()
            
            for candidate_path in analysis.archive_candidates:
                try:
                    # Determine archive location
                    relative_path = candidate_path.relative_to(self.repo_root)
                    archive_path = self.archive_dir / relative_path
                    
                    # Create parent directories in archive
                    archive_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Move to archive
                    shutil.move(str(candidate_path), str(archive_path))
                    archived_files.append(f"{relative_path} -> archive/{relative_path}")
                    
                except Exception as e:
                    errors.append(f"Failed to archive {candidate_path}: {e}")
            
            # Create archive index
            if archived_files:
                self._create_archive_index(archived_files)
            
            return CleanupResult(
                success=len(errors) == 0,
                files_processed=len(archived_files),
                message="Organized archive content",
                details={
                    'archived_files': archived_files,
                    'archive_directory': str(self.archive_dir),
                    'errors': errors
                }
            )
            
        except Exception as e:
            return CleanupResult(
                success=False,
                errors=[f"Failed to organize archive content: {e}"]
            )
    
    def update_import_statements(self) -> CleanupResult:
        """Update all import statements to reflect new structure.
        
        Returns:
            CleanupResult with import update details
        """
        try:
            updated_files = []
            total_updates = 0
            errors = []
            
            analysis = self.analyze_structure()
            
            # Group updates by file
            updates_by_file = {}
            for update in analysis.import_updates_needed:
                if update.file_path not in updates_by_file:
                    updates_by_file[update.file_path] = []
                updates_by_file[update.file_path].append(update)
            
            # Apply updates to each file
            for file_path, updates in updates_by_file.items():
                try:
                    # Read file content
                    content = file_path.read_text(encoding='utf-8')
                    lines = content.split('\n')
                    
                    # Apply updates (in reverse order to preserve line numbers)
                    updates.sort(key=lambda u: u.line_number, reverse=True)
                    for update in updates:
                        if 0 <= update.line_number < len(lines):
                            lines[update.line_number] = lines[update.line_number].replace(
                                update.old_import, update.new_import
                            )
                            total_updates += 1
                    
                    # Write updated content
                    file_path.write_text('\n'.join(lines), encoding='utf-8')
                    updated_files.append(str(file_path))
                    
                except Exception as e:
                    errors.append(f"Failed to update imports in {file_path}: {e}")
            
            return CleanupResult(
                success=len(errors) == 0,
                files_processed=len(updated_files),
                message="Updated import statements",
                details={
                    'updated_files': updated_files,
                    'total_updates': total_updates,
                    'errors': errors
                }
            )
            
        except Exception as e:
            return CleanupResult(
                success=False,
                errors=[f"Failed to update import statements: {e}"]
            )
    
    def validate_structure(self) -> ValidationResult:
        """Validate repository structure against target.
        
        Returns:
            ValidationResult with validation details
        """
        try:
            analysis = self.analyze_structure()
            
            tests_passed = []
            tests_failed = []
            
            # Check for violations
            if analysis.violations:
                tests_failed.extend(analysis.violations)
            else:
                tests_passed.append("All required structure elements present")
            
            # Check for missing init files
            if analysis.missing_init_files:
                tests_failed.append(f"Missing {len(analysis.missing_init_files)} __init__.py files")
            else:
                tests_passed.append("All packages have __init__.py files")
            
            # Check naming conventions
            if analysis.incorrect_naming:
                tests_failed.append(f"Found {len(analysis.incorrect_naming)} naming convention violations")
            else:
                tests_passed.append("All names follow conventions")
            
            # Check for outdated imports
            if analysis.import_updates_needed:
                tests_failed.append(f"Found {len(analysis.import_updates_needed)} outdated import statements")
            else:
                tests_passed.append("All import statements are up to date")
            
            success = len(tests_failed) == 0
            
            return ValidationResult(
                component="repository_structure",
                tests_passed=tests_passed,
                tests_failed=tests_failed,
                success=success,
                error_details=str(analysis) if not success else None
            )
            
        except Exception as e:
            return ValidationResult(
                component="repository_structure",
                tests_passed=[],
                tests_failed=[f"Structure validation failed: {e}"],
                success=False,
                error_details=str(e)
            )
    
    def _find_missing_init_files(self) -> List[Path]:
        """Find directories that should be packages but lack __init__.py files."""
        missing_init_files = []
        
        # Check nanobrain package structure recursively
        nanobrain_dir = self.repo_root / 'nanobrain'
        if nanobrain_dir.exists():
            for package_dir in nanobrain_dir.rglob('*'):
                if package_dir.is_dir() and not package_dir.name.startswith('.'):
                    # Skip __pycache__ and other special directories
                    if package_dir.name in {'__pycache__', '.git', '.pytest_cache'}:
                        continue
                        
                    init_file = package_dir / '__init__.py'
                    if not init_file.exists():
                        # Check if directory contains Python files
                        if any(package_dir.glob('*.py')):
                            missing_init_files.append(init_file)
        
        # Check demos structure
        demos_dir = self.repo_root / 'demos'
        if demos_dir.exists():
            for demo_dir in demos_dir.iterdir():
                if demo_dir.is_dir() and not demo_dir.name.startswith('.'):
                    init_file = demo_dir / '__init__.py'
                    if not init_file.exists():
                        # Check if directory contains Python files
                        if any(demo_dir.glob('*.py')):
                            missing_init_files.append(init_file)
        
        return missing_init_files
    
    def _check_naming_conventions(self) -> List[Tuple[Path, str]]:
        """Check naming conventions and suggest corrections."""
        incorrect_naming = []
        
        # Check package and module names
        for py_file in self.repo_root.rglob('*.py'):
            if py_file.name == '__init__.py':
                continue
                
            # Check module name
            if not self.naming_conventions['module'].match(py_file.name):
                suggested_name = self._suggest_module_name(py_file.name)
                incorrect_naming.append((py_file, suggested_name))
        
        # Check directory names (packages)
        for package_dir in self.repo_root.rglob('*'):
            if package_dir.is_dir() and not package_dir.name.startswith('.'):
                # Skip certain directories
                if package_dir.name in {'__pycache__', '.git', '.pytest_cache', 'node_modules'}:
                    continue
                    
                if not self.naming_conventions['package'].match(package_dir.name):
                    suggested_name = self._suggest_package_name(package_dir.name)
                    incorrect_naming.append((package_dir, suggested_name))
        
        return incorrect_naming
    
    def _find_import_updates(self) -> List[ImportUpdate]:
        """Find import statements that need updating."""
        import_updates = []
        
        # Common import patterns that might need updating
        old_patterns = [
            (r'from nanobrain import (.+)', r'from nanobrain.core import \1'),
            (r'import nanobrain\.(.+)', r'import nanobrain.core.\1'),
            # Skip the demos pattern as it creates invalid syntax
        ]
        
        for py_file in self.repo_root.rglob('*.py'):
            try:
                content = py_file.read_text(encoding='utf-8')
                lines = content.split('\n')
                
                for line_num, line in enumerate(lines):
                    line = line.strip()
                    if line.startswith(('import ', 'from ')):
                        for old_pattern, new_pattern in old_patterns:
                            match = re.match(old_pattern, line)
                            if match:
                                new_import = re.sub(old_pattern, new_pattern, line)
                                if new_import != line:
                                    import_updates.append(ImportUpdate(
                                        file_path=py_file,
                                        old_import=line,
                                        new_import=new_import,
                                        line_number=line_num
                                    ))
                                    
            except Exception:
                continue
        
        return import_updates
    
    def _find_archive_candidates(self) -> List[Path]:
        """Find files and directories that should be archived."""
        archive_candidates = []
        
        # Look for old documentation files
        for md_file in self.repo_root.rglob('*.md'):
            # Skip main documentation
            if md_file.name in {'README.md'} or 'docs/' in str(md_file):
                continue
                
            # Archive analysis and status files
            if any(keyword in md_file.name.lower() for keyword in 
                   ['analysis', 'status', 'summary', 'report', 'assessment', 'plan']):
                archive_candidates.append(md_file)
        
        # Look for backup directories
        for backup_dir in self.repo_root.rglob('*backup*'):
            if backup_dir.is_dir():
                archive_candidates.append(backup_dir)
        
        # Look for old demo directories (not in target demos)
        demos_dir = self.repo_root / 'demos'
        if demos_dir.exists():
            target_demos = {'viral_pssm_workflow', 'rag_database_creation', 
                          'academylink_aurora_demo', 'simple_demo'}
            
            for demo_dir in demos_dir.iterdir():
                if demo_dir.is_dir() and demo_dir.name not in target_demos:
                    archive_candidates.append(demo_dir)
        
        return archive_candidates
    
    def _find_misplaced_core_files(self) -> List[Tuple[Path, Path]]:
        """Find core framework files that are in the wrong location."""
        misplaced_files = []
        
        # This would contain logic to identify files that belong in core/
        # For now, return empty list as the structure is mostly correct
        
        return misplaced_files
    
    def _generate_init_content(self, init_path: Path) -> str:
        """Generate appropriate content for an __init__.py file."""
        package_dir = init_path.parent
        package_name = package_dir.name
        
        # Basic __init__.py content
        content = f'"""\n{package_name.replace("_", " ").title()} package.\n"""\n\n'
        
        # Add version if this is the main package
        if package_name == 'nanobrain':
            content += '__version__ = "0.1.0"\n\n'
        
        # Auto-import main classes/functions from modules in the package
        py_files = [f for f in package_dir.glob('*.py') if f.name != '__init__.py']
        
        if py_files:
            content += '# Auto-generated imports\n'
            for py_file in py_files:
                module_name = py_file.stem
                # Try to find main classes/functions to import
                try:
                    file_content = py_file.read_text(encoding='utf-8')
                    tree = ast.parse(file_content)
                    
                    classes = [node.name for node in ast.walk(tree) 
                             if isinstance(node, ast.ClassDef)]
                    functions = [node.name for node in ast.walk(tree) 
                               if isinstance(node, ast.FunctionDef) and not node.name.startswith('_')]
                    
                    if classes or functions:
                        imports = classes + functions[:3]  # Limit to avoid clutter
                        content += f'from .{module_name} import {", ".join(imports)}\n'
                        
                except Exception:
                    # Fallback to simple import
                    content += f'from . import {module_name}\n'
        
        content += '\n__all__ = [\n'
        # Add exported names
        content += '    # Add exported names here\n'
        content += ']\n'
        
        return content
    
    def _suggest_module_name(self, current_name: str) -> str:
        """Suggest a corrected module name."""
        # Remove extension
        name = current_name.replace('.py', '')
        
        # Convert to lowercase with underscores
        name = re.sub(r'([A-Z])', r'_\1', name).lower()
        name = re.sub(r'[^a-z0-9_]', '_', name)
        name = re.sub(r'_+', '_', name)
        name = name.strip('_')
        
        return f"{name}.py"
    
    def _suggest_package_name(self, current_name: str) -> str:
        """Suggest a corrected package name."""
        # Convert to lowercase with underscores
        name = re.sub(r'([A-Z])', r'_\1', current_name).lower()
        name = re.sub(r'[^a-z0-9_]', '_', name)
        name = re.sub(r'_+', '_', name)
        name = name.strip('_')
        
        return name
    
    def _create_archive_index(self, archived_files: List[str]) -> None:
        """Create an index of archived files."""
        index_content = """# Archive Index

This directory contains files and directories that have been archived during the repository cleanup process.

## Archived Items

"""
        
        for archived_file in sorted(archived_files):
            index_content += f"- {archived_file}\n"
        
        index_content += """
## Archive Date

This archive was created during the nanobrain repository cleanup process.

## Recovery

To recover any archived item, move it back to its original location in the repository root.
"""
        
        index_path = self.archive_dir / 'INDEX.md'
        index_path.write_text(index_content, encoding='utf-8')