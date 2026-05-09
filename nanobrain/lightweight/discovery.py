"""
Config-Driven Discovery System - Iteration 1
============================================

Discovers framework components by scanning existing configuration files.
This avoids import hell by finding classes through their usage in configs.
"""

import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ConfigDrivenDiscovery:
    """
    Discover framework components from existing configuration files.
    
    BRUTAL TRUTH: This is a simple, focused implementation that does ONE THING:
    scan config files for class references. No fancy features, no over-engineering.
    """
    
    def __init__(self, framework_root: Optional[str] = None):
        """Initialize discovery system."""
        
        if framework_root is None:
            # Auto-detect framework root from this file's location
            current_file = Path(__file__)
            self.framework_root = current_file.parent.parent  # nanobrain/
        else:
            self.framework_root = Path(framework_root)
        
        self.discovered_classes: Dict[str, Dict[str, Any]] = {}
        
        # Config directories to scan
        self.config_directories = [
            self.framework_root / "config",
            self.framework_root / "library" / "config",
            self.framework_root / "core" / "config",
        ]
        
        # Statistics for debugging
        self.stats = {
            "config_files_scanned": 0,
            "config_files_failed": 0,
            "classes_discovered": 0,
        }
    
    def discover_components(self) -> Dict[str, Dict[str, Any]]:
        """
        Scan configuration files and discover component classes.
        
        Returns:
            Dictionary mapping class names to their metadata
        """
        logger.info(f"Starting discovery from {self.framework_root}")
        
        # Clear previous results
        self.discovered_classes.clear()
        self.stats = {"config_files_scanned": 0, "config_files_failed": 0, "classes_discovered": 0}
        
        # Scan each config directory
        for config_dir in self.config_directories:
            if config_dir.exists():
                self._scan_directory(config_dir)
            else:
                logger.debug(f"Config directory not found: {config_dir}")
        
        logger.info(f"Discovery complete: {len(self.discovered_classes)} classes found")
        return self.discovered_classes
    
    def _scan_directory(self, directory: Path) -> None:
        """Recursively scan directory for YAML config files."""
        
        # Find all YAML files
        yaml_files = list(directory.rglob("*.yml")) + list(directory.rglob("*.yaml"))
        
        for yaml_file in yaml_files:
            self._process_config_file(yaml_file)
    
    def _process_config_file(self, config_file: Path) -> None:
        """Process a single configuration file."""
        
        try:
            self.stats["config_files_scanned"] += 1
            
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            if config_data is None:
                return
            
            # Extract class references
            class_refs = self._extract_class_references(config_data)
            
            # Register each class reference
            for class_ref in class_refs:
                self._register_class(class_ref, config_file)
                
        except Exception as e:
            logger.debug(f"Failed to process {config_file}: {e}")
            self.stats["config_files_failed"] += 1
    
    def _extract_class_references(self, data: Any) -> List[str]:
        """
        Extract class references from config data.
        
        Looks for 'class' fields containing Python module paths.
        """
        class_refs = []
        
        if isinstance(data, dict):
            for key, value in data.items():
                if key == "class" and isinstance(value, str) and "." in value:
                    class_refs.append(value)
                elif isinstance(value, (dict, list)):
                    class_refs.extend(self._extract_class_references(value))
        
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, (dict, list)):
                    class_refs.extend(self._extract_class_references(item))
        
        return class_refs
    
    def _register_class(self, class_path: str, source_file: Path) -> None:
        """Register a discovered class."""
        
        try:
            # Parse "module.path.ClassName" -> ("module.path", "ClassName")
            module_path, class_name = class_path.rsplit(".", 1)
            
            # Skip if already registered
            if class_name in self.discovered_classes:
                return
            
            # Register the class
            self.discovered_classes[class_name] = {
                "class_name": class_name,
                "class_path": class_path,
                "module_path": module_path,
                "category": self._categorize_class(module_path),
                "source_file": str(source_file),
                "has_optional_deps": self._has_optional_deps(module_path)
            }
            
            self.stats["classes_discovered"] += 1
            
        except ValueError:
            # Invalid class path format
            logger.debug(f"Invalid class path: {class_path}")
    
    def _categorize_class(self, module_path: str) -> str:
        """Categorize class based on its module path."""
        
        path_lower = module_path.lower()
        
        if "agent" in path_lower:
            return "agent"
        elif "step" in path_lower:
            return "step"
        elif "executor" in path_lower:
            return "executor"
        elif "workflow" in path_lower:
            return "workflow"
        elif "tool" in path_lower:
            return "tool"
        else:
            return "unknown"
    
    def _has_optional_deps(self, module_path: str) -> bool:
        """Check if module likely has optional dependencies."""
        
        optional_indicators = ["parsl", "bioinformatics", "hpc", "elasticsearch"]
        path_lower = module_path.lower()
        return any(indicator in path_lower for indicator in optional_indicators)
    
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
