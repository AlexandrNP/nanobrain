"""
Config-Based Discovery System - CORRECT Implementation
======================================================

Uses ACTUAL config files to discover class-to-config mappings.
Reads the explicit 'class:' references in config files instead of guessing.

BRUTAL TRUTH: This is what I should have implemented from the start.
The config files contain the EXPLICIT mappings - use them!
"""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ConfigBasedDiscovery:
    """
    Discovery system that uses ACTUAL config files to find class-to-config mappings.
    
    BRUTAL TRUTH: This reads the explicit 'class:' fields in config files
    instead of trying to guess relationships. Much more reliable.
    """
    
    def __init__(self, framework_root: Optional[str] = None):
        """Initialize config-based discovery."""
        
        if framework_root is None:
            current_file = Path(__file__)
            self.framework_root = current_file.parent.parent
        else:
            self.framework_root = Path(framework_root)
        
        # Storage for discovered mappings
        self.class_to_configs: Dict[str, List[Dict[str, Any]]] = {}
        self.config_files: Dict[str, Dict[str, Any]] = {}
        
        # Config directories to scan (prioritize core)
        self.config_directories = [
            self.framework_root / "config" / "core",      # Highest priority
            self.framework_root / "config" / "components", # Medium priority  
            self.framework_root / "config",               # Base priority
            self.framework_root / "library" / "config",   # Lower priority
        ]
        
        # Statistics
        self.stats = {
            "config_files_scanned": 0,
            "config_files_failed": 0,
            "class_mappings_found": 0,
            "unique_classes": 0
        }
    
    def discover_from_configs(self) -> Dict[str, Any]:
        """
        Discover class-to-config mappings from actual config files.
        
        Returns:
            Dictionary with class mappings and config file information
        """
        print(f"🔍 Starting CONFIG-BASED discovery from {self.framework_root}")
        
        # Clear previous results
        self.class_to_configs.clear()
        self.config_files.clear()
        self.stats = {k: 0 for k in self.stats.keys()}
        
        # Scan config directories in priority order
        for config_dir in self.config_directories:
            if config_dir.exists():
                print(f"📂 Scanning config directory: {config_dir}")
                self._scan_config_directory(config_dir)
            else:
                print(f"⚠️  Config directory not found: {config_dir}")
        
        # Process discovered mappings
        self._process_class_mappings()
        
        print(f"✅ Config-based discovery complete")
        self._print_config_stats()
        
        return {
            "class_to_configs": self.class_to_configs,
            "config_files": self.config_files
        }
    
    def _scan_config_directory(self, config_dir: Path) -> None:
        """Scan config directory for YAML files."""
        
        # Find all YAML files
        yaml_files = list(config_dir.rglob("*.yml")) + list(config_dir.rglob("*.yaml"))
        
        print(f"   Found {len(yaml_files)} config files")
        
        for yaml_file in yaml_files:
            self._process_config_file(yaml_file, config_dir)
    
    def _process_config_file(self, config_file: Path, base_dir: Path) -> None:
        """Process a single config file to extract class mappings."""
        
        try:
            self.stats["config_files_scanned"] += 1
            
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            if config_data is None:
                return
            
            # Extract class reference
            class_path = self._extract_class_reference(config_data)
            
            if class_path:
                # Determine priority based on directory
                priority = self._get_config_priority(base_dir)
                
                # Store config file info
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
                
                # Extract class name from class path
                class_name = class_path.split(".")[-1]
                
                # Add to class mappings
                if class_name not in self.class_to_configs:
                    self.class_to_configs[class_name] = []
                
                self.class_to_configs[class_name].append(config_info)
                self.stats["class_mappings_found"] += 1
                
                logger.debug(f"Found mapping: {class_name} -> {config_file}")
                
        except yaml.YAMLError as e:
            logger.debug(f"YAML error in {config_file}: {e}")
            self.stats["config_files_failed"] += 1
        except Exception as e:
            logger.debug(f"Error processing {config_file}: {e}")
            self.stats["config_files_failed"] += 1
    
    def _extract_class_reference(self, config_data: Any) -> Optional[str]:
        """Extract class reference from config data."""
        
        if isinstance(config_data, dict):
            # Direct class field
            if "class" in config_data and isinstance(config_data["class"], str):
                class_path = config_data["class"]
                if "." in class_path and "nanobrain" in class_path:
                    return class_path
            
            # Recursive search in nested structures
            for key, value in config_data.items():
                if isinstance(value, (dict, list)):
                    result = self._extract_class_reference(value)
                    if result:
                        return result
        
        elif isinstance(config_data, list):
            for item in config_data:
                if isinstance(item, (dict, list)):
                    result = self._extract_class_reference(item)
                    if result:
                        return result
        
        return None
    
    def _get_config_priority(self, config_dir: Path) -> int:
        """Get priority for config based on directory."""
        
        dir_str = str(config_dir)
        
        if "config/core" in dir_str:
            return 1  # Highest priority
        elif "config/components" in dir_str:
            return 2  # Medium priority
        elif "config" in dir_str and "library" not in dir_str:
            return 3  # Base priority
        else:
            return 4  # Lowest priority
    
    def _process_class_mappings(self) -> None:
        """Process and prioritize class mappings."""
        
        # Sort configs by priority for each class
        for class_name, configs in self.class_to_configs.items():
            # Sort by priority (lower number = higher priority)
            configs.sort(key=lambda x: x["priority"])
        
        self.stats["unique_classes"] = len(self.class_to_configs)
    
    def _print_config_stats(self) -> None:
        """Print discovery statistics."""
        
        print(f"\n📊 CONFIG-BASED DISCOVERY STATISTICS:")
        print(f"  Config files scanned: {self.stats['config_files_scanned']}")
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
            for class_name, count in sorted(multiple_configs.items()):
                print(f"  {class_name}: {count} configs")
                configs = self.class_to_configs[class_name]
                for i, config in enumerate(configs):
                    priority_label = ["🥇 CORE", "🥈 COMPONENTS", "🥉 BASE", "🏅 LIBRARY"][config["priority"] - 1]
                    print(f"    {i+1}. {priority_label}: {config['relative_path']}")
    
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
    
    def get_config_file_info(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific config file."""
        
        return self.config_files.get(file_path)
