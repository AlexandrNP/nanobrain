"""
Auto-Discovery System for NanoBrain Framework - Week 1 Implementation
=====================================================================

Config-driven discovery system that scans existing configuration files
to automatically discover available framework components.

ITERATION 1: Basic config file scanning and class extraction
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, List, Set, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ConfigDrivenDiscovery:
    """
    Discover framework components by scanning existing configuration files.
    
    This approach avoids import hell by discovering classes from their usage
    in configuration files rather than trying to import all modules.
    """
    
    def __init__(self, framework_root: str = None):
        """
        Initialize discovery system.
        
        Args:
            framework_root: Root directory of nanobrain framework
        """
        if framework_root is None:
            # Auto-detect framework root
            current_file = Path(__file__)
            self.framework_root = current_file.parent.parent  # nanobrain/
        else:
            self.framework_root = Path(framework_root)
        
        self.discovered_classes: Dict[str, Dict[str, Any]] = {}
        self.config_directories = [
            self.framework_root / "config",
            self.framework_root / "library" / "config", 
            self.framework_root / "core" / "config",
        ]
        
        # Track discovery statistics
        self.stats = {
            "config_files_scanned": 0,
            "config_files_failed": 0,
            "classes_discovered": 0,
            "unique_modules": set()
        }
    
    def discover_components(self) -> Dict[str, Dict[str, Any]]:
        """
        Scan configuration files and discover component classes.
        
        Returns:
            Dictionary mapping class names to their metadata
        """
        logger.info(f"Starting component discovery from {self.framework_root}")
        
        for config_dir in self.config_directories:
            if config_dir.exists():
                logger.info(f"Scanning config directory: {config_dir}")
                self._scan_config_directory(config_dir)
            else:
                logger.warning(f"Config directory not found: {config_dir}")
        
        logger.info(f"Discovery complete. Found {len(self.discovered_classes)} classes")
        self._log_discovery_stats()
        
        return self.discovered_classes
    
    def _scan_config_directory(self, config_dir: Path) -> None:
        """Recursively scan a config directory for YAML files."""
        
        for config_file in config_dir.rglob("*.yml"):
            self._process_config_file(config_file)
        
        for config_file in config_dir.rglob("*.yaml"):
            self._process_config_file(config_file)
    
    def _process_config_file(self, config_file: Path) -> None:
        """Process a single configuration file."""
        
        try:
            self.stats["config_files_scanned"] += 1
            
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            if config_data is None:
                logger.debug(f"Empty config file: {config_file}")
                return
            
            # Extract class references from config
            class_refs = self._extract_class_references(config_data)
            
            for class_ref in class_refs:
                self._register_class_reference(class_ref, config_file)
            
            logger.debug(f"Processed {config_file}: found {len(class_refs)} class references")
            
        except yaml.YAMLError as e:
            logger.error(f"YAML parsing error in {config_file}: {e}")
            self.stats["config_files_failed"] += 1
        except Exception as e:
            logger.error(f"Error processing {config_file}: {e}")
            self.stats["config_files_failed"] += 1
    
    def _extract_class_references(self, config_data: Any, path: str = "") -> List[str]:
        """
        Recursively extract class references from configuration data.
        
        Looks for 'class' fields that contain Python class paths.
        """
        class_refs = []
        
        if isinstance(config_data, dict):
            for key, value in config_data.items():
                if key == "class" and isinstance(value, str) and "." in value:
                    # Found a class reference
                    class_refs.append(value)
                elif isinstance(value, (dict, list)):
                    # Recurse into nested structures
                    nested_refs = self._extract_class_references(value, f"{path}.{key}")
                    class_refs.extend(nested_refs)
        
        elif isinstance(config_data, list):
            for i, item in enumerate(config_data):
                if isinstance(item, (dict, list)):
                    nested_refs = self._extract_class_references(item, f"{path}[{i}]")
                    class_refs.extend(nested_refs)
        
        return class_refs
    
    def _register_class_reference(self, class_path: str, source_config: Path) -> None:
        """Register a discovered class reference."""
        
        try:
            # Parse class path: "nanobrain.core.agent.ConversationalAgent"
            if "." not in class_path:
                logger.debug(f"Skipping invalid class path: {class_path}")
                return
            
            module_path, class_name = class_path.rsplit(".", 1)
            
            # Skip if already registered
            if class_name in self.discovered_classes:
                # Add this config as another source
                existing = self.discovered_classes[class_name]
                if str(source_config) not in existing["source_configs"]:
                    existing["source_configs"].append(str(source_config))
                return
            
            # Register new class
            self.discovered_classes[class_name] = {
                "class_name": class_name,
                "class_path": class_path,
                "module_path": module_path,
                "category": self._categorize_class(module_path),
                "source_configs": [str(source_config)],
                "has_optional_deps": self._check_optional_dependencies(module_path)
            }
            
            self.stats["classes_discovered"] += 1
            self.stats["unique_modules"].add(module_path)
            
            logger.debug(f"Registered class: {class_name} from {source_config}")
            
        except ValueError as e:
            logger.warning(f"Invalid class path '{class_path}' in {source_config}: {e}")
    
    def _categorize_class(self, module_path: str) -> str:
        """Categorize class based on module path."""
        
        module_lower = module_path.lower()
        
        if "agent" in module_lower:
            return "agent"
        elif "step" in module_lower:
            return "step"
        elif "executor" in module_lower:
            return "executor"
        elif "workflow" in module_lower:
            return "workflow"
        elif "tool" in module_lower:
            return "tool"
        elif "data_unit" in module_lower:
            return "data_unit"
        elif "link" in module_lower:
            return "link"
        else:
            return "unknown"
    
    def _check_optional_dependencies(self, module_path: str) -> bool:
        """Check if module likely has optional dependencies."""
        
        optional_patterns = [
            "parsl", "bioinformatics", "hpc", "ml", "torch", 
            "tensorflow", "elasticsearch", "docker"
        ]
        
        module_lower = module_path.lower()
        return any(pattern in module_lower for pattern in optional_patterns)
    
    def _log_discovery_stats(self) -> None:
        """Log discovery statistics."""
        
        stats = self.stats
        logger.info(f"Discovery Statistics:")
        logger.info(f"  Config files scanned: {stats['config_files_scanned']}")
        logger.info(f"  Config files failed: {stats['config_files_failed']}")
        logger.info(f"  Classes discovered: {stats['classes_discovered']}")
        logger.info(f"  Unique modules: {len(stats['unique_modules'])}")
        
        # Log by category
        categories = {}
        for class_info in self.discovered_classes.values():
            category = class_info["category"]
            categories[category] = categories.get(category, 0) + 1
        
        logger.info("  Classes by category:")
        for category, count in sorted(categories.items()):
            logger.info(f"    {category}: {count}")
    
    def get_classes_by_category(self, category: str) -> List[str]:
        """Get all class names in a specific category."""
        
        return [
            class_name for class_name, class_info in self.discovered_classes.items()
            if class_info["category"] == category
        ]
    
    def get_class_info(self, class_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific class."""
        
        return self.discovered_classes.get(class_name)
    
    def list_all_classes(self) -> List[str]:
        """List all discovered class names."""
        
        return list(self.discovered_classes.keys())
