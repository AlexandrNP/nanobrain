"""
Minimal Config-Driven Discovery - No External Dependencies
==========================================================

A minimal version that uses only Python standard library.
This is for testing in environments without pydantic/yaml.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional


class MinimalConfigDiscovery:
    """
    Minimal discovery system using only standard library.
    
    BRUTAL TRUTH: This is a stripped-down version for testing
    in environments without external dependencies.
    """
    
    def __init__(self, framework_root: Optional[str] = None):
        """Initialize minimal discovery system."""
        
        if framework_root is None:
            current_file = Path(__file__)
            self.framework_root = current_file.parent.parent
        else:
            self.framework_root = Path(framework_root)
        
        self.discovered_classes: Dict[str, Dict[str, Any]] = {}
        self.stats = {"files_scanned": 0, "files_failed": 0, "classes_found": 0}
    
    def discover_from_files(self) -> Dict[str, Dict[str, Any]]:
        """
        Discover classes by scanning config files.
        
        Uses simple text parsing instead of YAML parsing.
        """
        
        config_dirs = [
            self.framework_root / "config",
            self.framework_root / "library" / "config",
        ]
        
        for config_dir in config_dirs:
            if config_dir.exists():
                self._scan_directory_simple(config_dir)
        
        return self.discovered_classes
    
    def _scan_directory_simple(self, directory: Path) -> None:
        """Scan directory using simple text parsing."""
        
        # Find YAML files
        yaml_files = list(directory.rglob("*.yml")) + list(directory.rglob("*.yaml"))
        
        for yaml_file in yaml_files:
            self._parse_file_simple(yaml_file)
    
    def _parse_file_simple(self, file_path: Path) -> None:
        """Parse file using simple text search for 'class:' lines."""
        
        try:
            self.stats["files_scanned"] += 1
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Simple regex-like search for class definitions
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                
                # Look for lines like: class: "nanobrain.module.ClassName"
                if line.startswith('class:'):
                    class_value = line.split(':', 1)[1].strip()
                    # Remove quotes
                    class_value = class_value.strip('"\'')
                    
                    if '.' in class_value:
                        self._register_class_simple(class_value, file_path)
                        
        except Exception:
            self.stats["files_failed"] += 1
    
    def _register_class_simple(self, class_path: str, source_file: Path) -> None:
        """Register a discovered class."""
        
        try:
            module_path, class_name = class_path.rsplit(".", 1)
            
            if class_name not in self.discovered_classes:
                self.discovered_classes[class_name] = {
                    "class_name": class_name,
                    "class_path": class_path,
                    "module_path": module_path,
                    "category": self._categorize_simple(module_path),
                    "source_file": str(source_file)
                }
                self.stats["classes_found"] += 1
                
        except ValueError:
            pass  # Invalid class path
    
    def _categorize_simple(self, module_path: str) -> str:
        """Simple categorization based on module path."""

        path_lower = module_path.lower()

        if "agent" in path_lower:
            return "agent"
        elif "step" in path_lower:
            return "step"
        elif "executor" in path_lower:
            return "executor"
        elif "data_unit" in path_lower:
            return "data_unit"
        elif "link" in path_lower:
            return "link"
        elif "workflow" in path_lower:
            return "workflow"
        elif "tool" in path_lower:
            return "tool"
        else:
            return "unknown"
    
    def get_classes_by_category(self, category: str) -> List[str]:
        """Get classes by category."""
        
        return [
            name for name, info in self.discovered_classes.items()
            if info["category"] == category
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get discovery statistics."""
        return self.stats.copy()


def test_minimal_discovery():
    """Test the minimal discovery system."""
    
    print("🔥 TESTING MINIMAL DISCOVERY (NO EXTERNAL DEPS)")
    print("=" * 50)
    
    discovery = MinimalConfigDiscovery()
    
    print(f"Framework root: {discovery.framework_root}")
    print(f"Framework root exists: {discovery.framework_root.exists()}")
    
    # Check config directories
    config_dirs = [
        discovery.framework_root / "config",
        discovery.framework_root / "library" / "config",
    ]
    
    print("\nConfig directories:")
    for config_dir in config_dirs:
        exists = config_dir.exists()
        print(f"  {config_dir}: {'✅' if exists else '❌'}")
        if exists:
            yaml_files = list(config_dir.rglob("*.yml")) + list(config_dir.rglob("*.yaml"))
            print(f"    YAML files: {len(yaml_files)}")
    
    # Run discovery
    print("\n🔍 Running discovery...")
    discovered = discovery.discover_from_files()
    
    # Show results
    stats = discovery.get_stats()
    print("\n📊 Results:")
    print(f"Files scanned: {stats['files_scanned']}")
    print(f"Files failed: {stats['files_failed']}")
    print(f"Classes found: {stats['classes_found']}")
    
    if discovered:
        print("\n📂 Discovered classes:")
        categories = ["agent", "step", "executor", "data_unit", "link", "workflow", "tool", "unknown"]
        for category in categories:
            classes = discovery.get_classes_by_category(category)
            if classes:
                print(f"  {category}: {classes}")
    else:
        print("\n❌ No classes discovered!")
    
    return len(discovered) > 0


if __name__ == "__main__":
    success = test_minimal_discovery()
    exit(0 if success else 1)
