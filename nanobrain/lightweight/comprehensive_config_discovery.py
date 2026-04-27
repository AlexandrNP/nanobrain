"""
Comprehensive Config Discovery - EVERYWHERE
===========================================

Scans ALL config files EVERYWHERE in the framework.
No more pathetic partial scanning of just a few directories.

BRUTAL TRUTH: This should scan EVERY .yml/.yaml file in the entire framework
and extract ALL class references. No excuses, no shortcuts.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ComprehensiveConfigDiscovery:
    """
    Discovery system that scans ALL config files EVERYWHERE in the framework.
    
    BRUTAL TRUTH: This scans the ENTIRE framework tree for config files.
    No more missing configs because I only looked in a few directories.
    """
    
    def __init__(self, framework_root: Optional[str] = None):
        """Initialize comprehensive config discovery."""
        
        if framework_root is None:
            current_file = Path(__file__)
            self.framework_root = current_file.parent.parent
        else:
            self.framework_root = Path(framework_root)
        
        # Storage for discovered mappings
        self.class_to_configs: Dict[str, List[Dict[str, Any]]] = {}
        self.config_files: Dict[str, Dict[str, Any]] = {}
        
        # Statistics
        self.stats = {
            "total_yaml_files_found": 0,
            "config_files_processed": 0,
            "config_files_failed": 0,
            "class_mappings_found": 0,
            "unique_classes": 0,
            "directories_scanned": 0
        }
    
    def discover_all_configs(self) -> Dict[str, Any]:
        """
        Discover ALL config files EVERYWHERE in the framework.
        
        Returns:
            Dictionary with class mappings and config file information
        """
        print(f"🔍 Starting COMPREHENSIVE CONFIG discovery from {self.framework_root}")
        print("🚨 SCANNING EVERYWHERE - no directory will be missed!")
        
        # Clear previous results
        self.class_to_configs.clear()
        self.config_files.clear()
        self.stats = {k: 0 for k in self.stats.keys()}
        
        # Scan ENTIRE framework tree for config files
        self._scan_entire_framework()
        
        # Process discovered mappings
        self._process_class_mappings()
        
        print(f"✅ Comprehensive config discovery complete")
        self._print_comprehensive_stats()
        
        return {
            "class_to_configs": self.class_to_configs,
            "config_files": self.config_files
        }
    
    def _scan_entire_framework(self) -> None:
        """Scan the ENTIRE framework tree for ALL config files."""
        
        print(f"📂 Scanning ENTIRE framework tree: {self.framework_root}")
        
        # Find ALL .yml and .yaml files recursively
        yaml_patterns = ["**/*.yml", "**/*.yaml"]
        all_yaml_files = []
        
        for pattern in yaml_patterns:
            yaml_files = list(self.framework_root.rglob(pattern))
            all_yaml_files.extend(yaml_files)
        
        # Remove duplicates and filter out unwanted files
        unique_yaml_files = []
        seen_files = set()
        
        for yaml_file in all_yaml_files:
            if str(yaml_file) not in seen_files:
                # Skip certain directories that definitely don't contain config files
                skip_dirs = [
                    "__pycache__",
                    ".git",
                    ".pytest_cache",
                    "node_modules",
                    ".venv",
                    "venv"
                ]
                
                if not any(skip_dir in str(yaml_file) for skip_dir in skip_dirs):
                    unique_yaml_files.append(yaml_file)
                    seen_files.add(str(yaml_file))
        
        self.stats["total_yaml_files_found"] = len(unique_yaml_files)
        print(f"📄 Found {len(unique_yaml_files)} YAML files to scan")
        
        # Group files by directory for better reporting
        directories = set()
        for yaml_file in unique_yaml_files:
            directories.add(yaml_file.parent)
        
        self.stats["directories_scanned"] = len(directories)
        print(f"📁 Scanning {len(directories)} directories")
        
        # Process each YAML file
        for yaml_file in unique_yaml_files:
            self._process_config_file_comprehensive(yaml_file)
    
    def _process_config_file_comprehensive(self, config_file: Path) -> None:
        """Process a single config file to extract class mappings."""
        
        try:
            self.stats["config_files_processed"] += 1
            
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            if config_data is None:
                return
            
            # Extract ALL class references from this file
            class_references = self._extract_all_class_references(config_data)
            
            if class_references:
                # Determine priority based on file path
                priority = self._get_comprehensive_priority(config_file)
                
                # Process each class reference found
                for class_path in class_references:
                    self._register_class_config_mapping(class_path, config_file, config_data, priority)
                    
        except yaml.YAMLError as e:
            logger.debug(f"YAML error in {config_file}: {e}")
            self.stats["config_files_failed"] += 1
        except Exception as e:
            logger.debug(f"Error processing {config_file}: {e}")
            self.stats["config_files_failed"] += 1
    
    def _extract_all_class_references(self, config_data: Any) -> List[str]:
        """Extract ONLY VALID class references from config data."""

        class_references = []

        def is_valid_class_reference(value: str) -> bool:
            """Check if a string is a valid nanobrain class reference."""
            if not isinstance(value, str):
                return False

            # Must contain nanobrain
            if "nanobrain" not in value:
                return False

            # Must have proper module.Class format
            if not value.count(".") >= 2:  # At least nanobrain.module.Class
                return False

            # Must end with a capitalized class name
            class_name = value.split(".")[-1]
            if not class_name[0].isupper():
                return False

            # Must not contain invalid characters
            if any(char in value for char in ["=", "{", "}", "[", "]", " ", "\n", "\t"]):
                return False

            # Must not be a URL or config value
            if value.startswith("http") or "://" in value:
                return False

            return True

        def recursive_search(data, path=""):
            if isinstance(data, dict):
                for key, value in data.items():
                    current_path = f"{path}.{key}" if path else key

                    # ONLY accept 'class' field as primary class reference
                    if key == "class" and is_valid_class_reference(value):
                        class_references.append(value)

                    # Recursive search in nested structures
                    elif isinstance(value, (dict, list)):
                        recursive_search(value, current_path)

            elif isinstance(data, list):
                for i, item in enumerate(data):
                    current_path = f"{path}[{i}]" if path else f"[{i}]"
                    if isinstance(item, (dict, list)):
                        recursive_search(item, current_path)

        recursive_search(config_data)
        return list(set(class_references))  # Remove duplicates
    
    def _get_comprehensive_priority(self, config_file: Path) -> int:
        """Get priority based on file path with comprehensive rules."""
        
        file_str = str(config_file)
        relative_path = str(config_file.relative_to(self.framework_root))
        
        # Highest priority: core configs
        if "/config/core/" in file_str or relative_path.startswith("core/"):
            return 1
        
        # High priority: component configs
        elif "/config/components/" in file_str or "components/" in relative_path:
            return 2
        
        # Medium priority: main config directory
        elif "/config/" in file_str and "/library/" not in file_str:
            return 3
        
        # Lower priority: library configs
        elif "/library/config/" in file_str or "library/" in relative_path:
            return 4
        
        # Lowest priority: other configs (workflows, examples, etc.)
        else:
            return 5
    
    def _register_class_config_mapping(self, class_path: str, config_file: Path, config_data: Dict[str, Any], priority: int) -> None:
        """Register a class-to-config mapping."""
        
        # Extract and clean class name from class path
        class_name = self._extract_clean_class_name(class_path)
        
        # Create config info
        config_info = {
            "file_path": str(config_file),
            "relative_path": str(config_file.relative_to(self.framework_root)),
            "class_path": class_path,
            "priority": priority,
            "config_data": config_data,
            "name": config_data.get("name", config_file.stem),
            "description": config_data.get("description", ""),
            "version": config_data.get("version", "unknown")
        }
        
        # Store by file path
        self.config_files[str(config_file)] = config_info
        
        # Add to class mappings
        if class_name not in self.class_to_configs:
            self.class_to_configs[class_name] = []
        
        self.class_to_configs[class_name].append(config_info)
        self.stats["class_mappings_found"] += 1
        
        logger.debug(f"Found mapping: {class_name} -> {config_file}")

    def _extract_clean_class_name(self, class_path: str) -> str:
        """Extract clean class name from full class path, removing meaningless prefixes."""

        # Get the actual class name (last part)
        class_name = class_path.split(".")[-1]

        # Remove common meaningless prefixes
        meaningless_prefixes = [
            "Nanobrain",
            "NanoBrain",
            "Framework",
            "Default",
            "Base",
            "Simple",
            "Basic",
            "Generic"
        ]

        # Remove prefixes if they exist
        for prefix in meaningless_prefixes:
            if class_name.startswith(prefix) and len(class_name) > len(prefix):
                # Only remove if there's something meaningful after the prefix
                remaining = class_name[len(prefix):]
                if remaining and remaining[0].isupper():
                    class_name = remaining
                    break

        return class_name

    def _process_class_mappings(self) -> None:
        """Process and prioritize class mappings."""
        
        # Sort configs by priority for each class
        for class_name, configs in self.class_to_configs.items():
            # Sort by priority (lower number = higher priority)
            configs.sort(key=lambda x: x["priority"])
        
        self.stats["unique_classes"] = len(self.class_to_configs)
    
    def _print_comprehensive_stats(self) -> None:
        """Print comprehensive discovery statistics."""
        
        print(f"\n📊 COMPREHENSIVE CONFIG DISCOVERY STATISTICS:")
        print(f"  Directories scanned: {self.stats['directories_scanned']}")
        print(f"  Total YAML files found: {self.stats['total_yaml_files_found']}")
        print(f"  Config files processed: {self.stats['config_files_processed']}")
        print(f"  Config files failed: {self.stats['config_files_failed']}")
        print(f"  Class mappings found: {self.stats['class_mappings_found']}")
        print(f"  Unique classes: {self.stats['unique_classes']}")
        
        # Show classes with multiple configs
        multiple_configs = {
            class_name: len(configs) 
            for class_name, configs in self.class_to_configs.items() 
            if len(configs) > 1
        }
        
        if multiple_configs:
            print(f"\n📂 CLASSES WITH MULTIPLE CONFIGS:")
            for class_name, count in sorted(multiple_configs.items(), key=lambda x: x[1], reverse=True):
                print(f"  {class_name}: {count} configs")
                configs = self.class_to_configs[class_name]
                for i, config in enumerate(configs[:3]):  # Show first 3
                    priority_labels = ["🥇 CORE", "🥈 COMPONENTS", "🥉 BASE", "🏅 LIBRARY", "🔹 OTHER"]
                    priority_label = priority_labels[min(config["priority"] - 1, 4)]
                    print(f"    {i+1}. {priority_label}: {config['relative_path']}")
                if len(configs) > 3:
                    print(f"    ... and {len(configs) - 3} more")
        
        # Show directory distribution
        directory_counts = {}
        for config_info in self.config_files.values():
            dir_path = str(Path(config_info["relative_path"]).parent)
            directory_counts[dir_path] = directory_counts.get(dir_path, 0) + 1
        
        print(f"\n📁 CONFIG FILES BY DIRECTORY (top 10):")
        sorted_dirs = sorted(directory_counts.items(), key=lambda x: x[1], reverse=True)
        for dir_path, count in sorted_dirs[:10]:
            print(f"  {dir_path}: {count} files")
    
    def get_default_config_for_class(self, class_name: str) -> Optional[Dict[str, Any]]:
        """Get the default (highest priority) config for a class."""
        
        configs = self.class_to_configs.get(class_name, [])
        if configs:
            return configs[0]  # First is highest priority
        return None
    
    def get_all_configs_for_class(self, class_name: str) -> List[Dict[str, Any]]:
        """Get all available configs for a class."""
        
        return self.class_to_configs.get(class_name, [])
    
    def list_available_classes(self) -> List[str]:
        """List all classes that have config files."""
        
        return list(self.class_to_configs.keys())
