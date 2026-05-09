"""
Demo Manager for the Nanobrain cleanup system.

This module provides demo directory consolidation functionality,
organizing demos into keep, archive, and delete categories while
preserving essential documentation and functionality.
"""

import shutil
from pathlib import Path
from typing import List, Dict
from datetime import datetime
import logging
import json

from .models import DemoCategories, ArchiveResult


class DemoManager:
    """
    Manages consolidation and organization of demonstration implementations.
    
    Categorizes demos into keep, archive, and delete groups while preserving
    essential documentation and creating comprehensive archive indexes.
    """
    
    def __init__(self, repo_root: str):
        """
        Initialize the DemoManager.
        
        Args:
            repo_root: Path to the repository root directory
        """
        self.repo_root = Path(repo_root).resolve()
        self.demos_dir = self.repo_root / "demos"
        self.archive_dir = self.repo_root / "archive" / "demos"
        self.logger = logging.getLogger(__name__)
        
        # Target demos that must be preserved and fixed
        self.target_demos = {
            'viral_pssm_workflow',
            'rag_database_creation'
        }
        
        # Reference demos that should be kept as examples
        self.reference_demos = {
            'academylink_aurora_demo',
            'simple_demo'
        }
        
        # All demos to keep (target + reference)
        self.keep_demos = self.target_demos | self.reference_demos
        
        # Patterns for backup and duplicate directories to delete
        self.delete_patterns = [
            '*_backup_*',
            'bck-*',
            'duplicate_*',
            '*_bck_*',
            '*backup*',
            '*_old',
            '*_copy'
        ]
    
    def identify_demo_categories(self) -> DemoCategories:
        """
        Categorize all demos into keep, archive, and delete groups.
        
        Returns:
            DemoCategories object with lists of demos for each category
        """
        if not self.demos_dir.exists():
            self.logger.warning(f"Demos directory not found: {self.demos_dir}")
            return DemoCategories(keep=[], archive=[], delete=[])
        
        keep_demos = []
        archive_demos = []
        delete_demos = []
        
        # Scan all directories in demos/
        for item in self.demos_dir.iterdir():
            if not item.is_dir():
                continue
                
            demo_name = item.name
            
            # Check if it should be deleted (backup/duplicate)
            if self._should_delete_demo(demo_name):
                delete_demos.append(demo_name)
                self.logger.info(f"Marked for deletion: {demo_name}")
            
            # Check if it should be kept
            elif demo_name in self.keep_demos:
                keep_demos.append(demo_name)
                self.logger.info(f"Marked to keep: {demo_name}")
            
            # Everything else goes to archive
            else:
                archive_demos.append(demo_name)
                self.logger.info(f"Marked for archive: {demo_name}")
        
        categories = DemoCategories(
            keep=sorted(keep_demos),
            archive=sorted(archive_demos),
            delete=sorted(delete_demos)
        )
        
        self.logger.info(f"Demo categorization complete: "
                        f"{len(keep_demos)} keep, "
                        f"{len(archive_demos)} archive, "
                        f"{len(delete_demos)} delete")
        
        return categories
    
    def preserve_target_demos(self) -> bool:
        """
        Ensure target demos are preserved and accessible.
        
        Returns:
            True if all target demos are preserved successfully
        """
        success = True
        
        for demo_name in self.target_demos:
            demo_path = self.demos_dir / demo_name
            
            if not demo_path.exists():
                self.logger.error(f"Target demo not found: {demo_name}")
                success = False
                continue
            
            # Verify demo has essential files
            if not self._validate_demo_structure(demo_path):
                self.logger.warning(f"Target demo has incomplete structure: {demo_name}")
                # Don't fail - just warn, as this will be fixed in later phases
            
            self.logger.info(f"Target demo preserved: {demo_name}")
        
        return success
    
    def archive_non_essential_demos(self) -> ArchiveResult:
        """
        Move non-essential demos to archive directory.
        
        Returns:
            ArchiveResult with details of the archiving operation
        """
        categories = self.identify_demo_categories()
        archived_demos = []
        errors = []
        
        # Create archive directory if it doesn't exist
        try:
            self.archive_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            error_msg = f"Failed to create archive directory: {str(e)}"
            self.logger.error(error_msg)
            return ArchiveResult(
                archived_demos=[],
                archive_path=str(self.archive_dir),
                index_created=False,
                success=False,
                errors=[error_msg]
            )
        
        # Archive each demo
        for demo_name in categories.archive:
            try:
                source_path = self.demos_dir / demo_name
                target_path = self.archive_dir / demo_name
                
                if not source_path.exists():
                    self.logger.warning(f"Demo to archive not found: {demo_name}")
                    continue
                
                # Move demo to archive
                if target_path.exists():
                    shutil.rmtree(target_path)
                
                shutil.move(str(source_path), str(target_path))
                archived_demos.append(demo_name)
                self.logger.info(f"Archived demo: {demo_name}")
                
            except Exception as e:
                error_msg = f"Failed to archive demo {demo_name}: {str(e)}"
                errors.append(error_msg)
                self.logger.error(error_msg)
        
        # Create archive index
        index_created = self._create_archive_index(archived_demos)
        
        success = len(errors) == 0
        
        return ArchiveResult(
            archived_demos=archived_demos,
            archive_path=str(self.archive_dir),
            index_created=index_created,
            success=success,
            errors=errors
        )
    
    def delete_backup_directories(self) -> Dict[str, any]:
        """
        Delete backup and duplicate demo directories.
        
        Returns:
            Dictionary with deletion results
        """
        categories = self.identify_demo_categories()
        deleted_demos = []
        errors = []
        bytes_freed = 0
        
        for demo_name in categories.delete:
            try:
                demo_path = self.demos_dir / demo_name
                
                if not demo_path.exists():
                    self.logger.warning(f"Demo to delete not found: {demo_name}")
                    continue
                
                # Calculate size before deletion
                demo_size = self._calculate_directory_size(demo_path)
                
                # Delete the directory
                shutil.rmtree(demo_path)
                deleted_demos.append(demo_name)
                bytes_freed += demo_size
                self.logger.info(f"Deleted backup demo: {demo_name}")
                
            except Exception as e:
                error_msg = f"Failed to delete demo {demo_name}: {str(e)}"
                errors.append(error_msg)
                self.logger.error(error_msg)
        
        success = len(errors) == 0
        
        self.logger.info(f"Backup deletion complete: {len(deleted_demos)} demos deleted, "
                        f"{bytes_freed} bytes freed, {len(errors)} errors")
        
        return {
            'deleted_demos': deleted_demos,
            'bytes_freed': bytes_freed,
            'errors': errors,
            'success': success
        }
    
    def create_archive_index(self) -> bool:
        """
        Create comprehensive index of archived demos.
        
        Returns:
            True if index was created successfully
        """
        if not self.archive_dir.exists():
            self.logger.warning("Archive directory does not exist")
            return False
        
        archived_demos = [d.name for d in self.archive_dir.iterdir() if d.is_dir()]
        return self._create_archive_index(archived_demos)
    
    def _should_delete_demo(self, demo_name: str) -> bool:
        """
        Check if a demo should be deleted based on naming patterns.
        
        Args:
            demo_name: Name of the demo directory
            
        Returns:
            True if the demo should be deleted
        """
        demo_lower = demo_name.lower()
        
        for pattern in self.delete_patterns:
            # Convert glob pattern to simple string matching
            if pattern.startswith('*') and pattern.endswith('*'):
                # Pattern like '*_backup_*'
                middle = pattern[1:-1]
                if middle in demo_lower:
                    return True
            elif pattern.startswith('*'):
                # Pattern like '*_old'
                suffix = pattern[1:]
                if demo_lower.endswith(suffix):
                    return True
            elif pattern.endswith('*'):
                # Pattern like 'bck-*'
                prefix = pattern[:-1]
                if demo_lower.startswith(prefix):
                    return True
            else:
                # Exact match
                if demo_lower == pattern.lower():
                    return True
        
        return False
    
    def _validate_demo_structure(self, demo_path: Path) -> bool:
        """
        Validate that a demo has essential structure.
        
        Args:
            demo_path: Path to the demo directory
            
        Returns:
            True if demo has valid structure
        """
        # Check for README or documentation
        has_readme = any(
            (demo_path / name).exists() 
            for name in ['README.md', 'README.rst', 'README.txt', 'README']
        )
        
        # Check for Python files or configuration
        has_code = False
        for pattern in ['*.py', '*.yml', '*.yaml', '*.json']:
            if list(demo_path.glob(pattern)):
                has_code = True
                break
        
        return has_readme or has_code
    
    def _calculate_directory_size(self, directory: Path) -> int:
        """
        Calculate total size of a directory in bytes.
        
        Args:
            directory: Path to the directory
            
        Returns:
            Total size in bytes
        """
        total_size = 0
        try:
            for file_path in directory.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        except Exception as e:
            self.logger.warning(f"Error calculating size for {directory}: {str(e)}")
        
        return total_size
    
    def _create_archive_index(self, archived_demos: List[str]) -> bool:
        """
        Create comprehensive index documentation for archived demos.
        
        Args:
            archived_demos: List of archived demo names
            
        Returns:
            True if index was created successfully
        """
        try:
            index_path = self.archive_dir / "INDEX.md"
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Collect information about each archived demo
            demo_info = {}
            for demo_name in archived_demos:
                demo_path = self.archive_dir / demo_name
                if demo_path.exists():
                    info = self._extract_demo_info(demo_path)
                    demo_info[demo_name] = info
            
            # Generate index content
            index_content = self._generate_index_content(demo_info, timestamp)
            
            # Write index file
            index_path.write_text(index_content)
            
            # Also create JSON index for programmatic access
            json_index_path = self.archive_dir / "index.json"
            json_data = {
                'timestamp': timestamp,
                'archived_demos': demo_info,
                'total_demos': len(archived_demos)
            }
            json_index_path.write_text(json.dumps(json_data, indent=2))
            
            self.logger.info(f"Archive index created: {index_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create archive index: {str(e)}")
            return False
    
    def _extract_demo_info(self, demo_path: Path) -> Dict[str, any]:
        """
        Extract information about a demo for the index.
        
        Args:
            demo_path: Path to the demo directory
            
        Returns:
            Dictionary with demo information
        """
        info = {
            'name': demo_path.name,
            'path': str(demo_path.relative_to(self.repo_root)),
            'size_bytes': self._calculate_directory_size(demo_path),
            'files_count': len(list(demo_path.rglob('*'))),
            'has_readme': False,
            'has_config': False,
            'has_tests': False,
            'description': None,
            'main_files': []
        }
        
        try:
            # Check for README and extract description
            for readme_name in ['README.md', 'README.rst', 'README.txt', 'README']:
                readme_path = demo_path / readme_name
                if readme_path.exists():
                    info['has_readme'] = True
                    try:
                        content = readme_path.read_text(encoding='utf-8', errors='ignore')
                        # Extract first paragraph as description
                        lines = content.strip().split('\n')
                        for line in lines[1:]:  # Skip title
                            line = line.strip()
                            if line and not line.startswith('#'):
                                info['description'] = line[:200] + ('...' if len(line) > 200 else '')
                                break
                    except Exception:
                        pass
                    break
            
            # Check for configuration files
            config_patterns = ['*.yml', '*.yaml', '*.json', '*.toml', '*config*']
            for pattern in config_patterns:
                if list(demo_path.glob(pattern)):
                    info['has_config'] = True
                    break
            
            # Check for test files
            test_patterns = ['test_*.py', '*_test.py', 'tests/*']
            for pattern in test_patterns:
                if list(demo_path.rglob(pattern)):
                    info['has_tests'] = True
                    break
            
            # Identify main files
            main_files = []
            for file_path in demo_path.iterdir():
                if file_path.is_file():
                    name = file_path.name
                    if (name.startswith('run_') or 
                        name.startswith('main_') or 
                        name == 'main.py' or
                        name.endswith('_demo.py') or
                        name.endswith('_workflow.py')):
                        main_files.append(name)
            
            info['main_files'] = sorted(main_files)
            
        except Exception as e:
            self.logger.warning(f"Error extracting info for {demo_path.name}: {str(e)}")
        
        return info
    
    def _generate_index_content(self, demo_info: Dict[str, Dict], timestamp: str) -> str:
        """
        Generate markdown content for the archive index.
        
        Args:
            demo_info: Dictionary of demo information
            timestamp: Creation timestamp
            
        Returns:
            Markdown content for the index
        """
        content = [
            "# Nanobrain Framework - Archived Demos Index",
            "",
            f"**Generated:** {timestamp}",
            f"**Total Archived Demos:** {len(demo_info)}",
            "",
            "This directory contains demonstration implementations that were archived during the",
            "Nanobrain framework cleanup process. These demos are preserved for reference but",
            "are not part of the core framework distribution.",
            "",
            "## Active Demos",
            "",
            "The following demos remain active in the main `demos/` directory:",
            "",
            "- **viral_pssm_workflow** - Target demo for viral protein analysis",
            "- **rag_database_creation** - Target demo for RAG database creation",
            "- **academylink_aurora_demo** - Reference implementation for Academy integration",
            "- **simple_demo** - Basic framework usage example",
            "",
            "## Archived Demos",
            "",
        ]
        
        if not demo_info:
            content.append("No demos were archived.")
            return '\n'.join(content)
        
        # Sort demos by name
        sorted_demos = sorted(demo_info.items())
        
        for demo_name, info in sorted_demos:
            content.extend([
                f"### {demo_name}",
                "",
                f"**Path:** `{info['path']}`",
                f"**Size:** {info['size_bytes']:,} bytes",
                f"**Files:** {info['files_count']} total",
                ""
            ])
            
            if info['description']:
                content.extend([
                    f"**Description:** {info['description']}",
                    ""
                ])
            
            # Add features
            features = []
            if info['has_readme']:
                features.append("📖 Documentation")
            if info['has_config']:
                features.append("⚙️ Configuration")
            if info['has_tests']:
                features.append("🧪 Tests")
            
            if features:
                content.extend([
                    f"**Features:** {' | '.join(features)}",
                    ""
                ])
            
            # Add main files
            if info['main_files']:
                content.extend([
                    "**Main Files:**",
                    ""
                ])
                for file_name in info['main_files']:
                    content.append(f"- `{file_name}`")
                content.append("")
            
            content.append("---")
            content.append("")
        
        # Add footer
        content.extend([
            "## Recovery Instructions",
            "",
            "To restore an archived demo to active status:",
            "",
            "1. Copy the demo directory from `archive/demos/` to `demos/`",
            "2. Update any hardcoded paths in configuration files",
            "3. Test the demo functionality",
            "4. Update the main demos documentation",
            "",
            "## Archive Maintenance",
            "",
            "This archive was created during the Nanobrain framework cleanup process.",
            "Archived demos are preserved for historical reference and potential future use.",
            "Regular review of archived content is recommended to identify demos that",
            "could be permanently removed or restored to active status.",
        ])
        
        return '\n'.join(content)