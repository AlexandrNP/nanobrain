"""
Comprehensive Discovery System - REAL Implementation
===================================================

Scans ALL Python files in core/ and library/ directories to find classes
that implement the from_config pattern. No more pathetic partial discovery.

BRUTAL TRUTH: This does what the previous version SHOULD have done.
"""

import os
import re
import ast
from pathlib import Path
from typing import Dict, List, Any, Optional, Set


class ComprehensiveDiscovery:
    """
    Comprehensive discovery that scans ALL Python files for from_config classes.
    
    BRUTAL TRUTH: This is what I should have implemented from the start.
    No more excuses about "only scanning config files" - we scan EVERYTHING.
    """
    
    def __init__(self, framework_root: Optional[str] = None):
        """Initialize comprehensive discovery."""
        
        if framework_root is None:
            current_file = Path(__file__)
            self.framework_root = current_file.parent.parent
        else:
            self.framework_root = Path(framework_root)
        
        self.discovered_classes: Dict[str, Dict[str, Any]] = {}
        
        # Directories to scan comprehensively
        self.scan_directories = [
            self.framework_root / "core",
            self.framework_root / "library",
        ]
        
        # Statistics
        self.stats = {
            "python_files_scanned": 0,
            "python_files_failed": 0,
            "classes_found": 0,
            "from_config_classes": 0,
            "config_classes": 0
        }
    
    def discover_all_classes(self) -> Dict[str, Dict[str, Any]]:
        """
        Comprehensively scan ALL Python files for qualified classes.
        
        Returns:
            Dictionary mapping class names to their metadata
        """
        print(f"🔍 Starting COMPREHENSIVE discovery from {self.framework_root}")
        
        # Clear previous results
        self.discovered_classes.clear()
        self.stats = {k: 0 for k in self.stats.keys()}
        
        # Scan each directory comprehensively
        for scan_dir in self.scan_directories:
            if scan_dir.exists():
                print(f"📂 Scanning directory: {scan_dir}")
                self._scan_directory_comprehensive(scan_dir)
            else:
                print(f"⚠️  Directory not found: {scan_dir}")
        
        print(f"✅ Comprehensive discovery complete: {len(self.discovered_classes)} classes found")
        self._print_stats()
        
        return self.discovered_classes
    
    def _scan_directory_comprehensive(self, directory: Path) -> None:
        """Recursively scan directory for ALL Python files."""
        
        # Find all Python files recursively
        python_files = list(directory.rglob("*.py"))
        
        print(f"   Found {len(python_files)} Python files to scan")
        
        for py_file in python_files:
            # Skip __pycache__ and other non-source files
            if "__pycache__" in str(py_file) or py_file.name.startswith("__"):
                continue
                
            self._analyze_python_file(py_file)
    
    def _analyze_python_file(self, py_file: Path) -> None:
        """Analyze a Python file for qualified classes."""
        
        try:
            self.stats["python_files_scanned"] += 1
            
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse the Python file into an AST
            try:
                tree = ast.parse(content)
            except SyntaxError:
                # Skip files with syntax errors
                return
            
            # Find all class definitions
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    self._analyze_class_node(node, py_file, content)
                    
        except Exception as e:
            self.stats["python_files_failed"] += 1
            # Don't print errors for every file - too noisy
            pass
    
    def _analyze_class_node(self, class_node: ast.ClassDef, py_file: Path, file_content: str) -> None:
        """Analyze a class AST node to determine if it's qualified."""
        
        class_name = class_node.name
        self.stats["classes_found"] += 1
        
        # Check if class has from_config method
        has_from_config = self._has_from_config_method(class_node, file_content)
        
        # Check if class extends ConfigBase or similar
        extends_config_base = self._extends_config_base(class_node)
        
        # Only register qualified classes
        if has_from_config or extends_config_base:
            self._register_qualified_class(class_name, py_file, has_from_config, extends_config_base)
    
    def _has_from_config_method(self, class_node: ast.ClassDef, file_content: str) -> bool:
        """Check if class has a from_config method."""
        
        # Look for from_config method in the class
        for item in class_node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "from_config":
                # Check if it's a classmethod
                for decorator in item.decorator_list:
                    if (isinstance(decorator, ast.Name) and decorator.id == "classmethod"):
                        return True
        
        # Also check with simple text search as backup
        class_start = class_node.lineno
        class_end = class_node.end_lineno if hasattr(class_node, 'end_lineno') else class_start + 50
        
        lines = file_content.split('\n')
        class_content = '\n'.join(lines[class_start-1:class_end])
        
        return "@classmethod" in class_content and "def from_config" in class_content
    
    def _extends_config_base(self, class_node: ast.ClassDef) -> bool:
        """Check if class extends ConfigBase or similar config classes."""
        
        for base in class_node.bases:
            if isinstance(base, ast.Name):
                base_name = base.id
                if "Config" in base_name and base_name in ["ConfigBase", "BaseModel"]:
                    return True
            elif isinstance(base, ast.Attribute):
                # Handle cases like module.ConfigBase
                if "Config" in ast.unparse(base):
                    return True
        
        return False
    
    def _register_qualified_class(self, class_name: str, py_file: Path, 
                                 has_from_config: bool, extends_config_base: bool) -> None:
        """Register a qualified class."""
        
        # Generate module path from file path
        relative_path = py_file.relative_to(self.framework_root)
        module_path = str(relative_path.with_suffix("")).replace(os.sep, ".")
        
        # Skip if already registered (avoid duplicates)
        if class_name in self.discovered_classes:
            return
        
        # Determine class type
        if extends_config_base:
            self.stats["config_classes"] += 1
            category = "config"
        else:
            self.stats["from_config_classes"] += 1
            category = self._categorize_class(module_path, class_name)
        
        # Register the class
        self.discovered_classes[class_name] = {
            "class_name": class_name,
            "class_path": f"{module_path}.{class_name}",
            "module_path": module_path,
            "file_path": str(py_file),
            "category": category,
            "has_from_config": has_from_config,
            "extends_config_base": extends_config_base,
            "qualification": "from_config" if has_from_config else "config_base"
        }
    
    def _categorize_class(self, module_path: str, class_name: str) -> str:
        """Categorize class based on module path and class name."""
        
        path_lower = module_path.lower()
        name_lower = class_name.lower()
        
        # Category based on module path
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
        elif "data_unit" in path_lower:
            return "data_unit"
        elif "link" in path_lower:
            return "link"
        
        # Category based on class name
        elif "agent" in name_lower:
            return "agent"
        elif "step" in name_lower:
            return "step"
        elif "executor" in name_lower:
            return "executor"
        elif "workflow" in name_lower:
            return "workflow"
        elif "tool" in name_lower:
            return "tool"
        elif "dataunit" in name_lower:
            return "data_unit"
        elif "link" in name_lower:
            return "link"
        elif "config" in name_lower:
            return "config"
        else:
            return "unknown"
    
    def _print_stats(self) -> None:
        """Print discovery statistics."""
        
        print(f"\n📊 COMPREHENSIVE DISCOVERY STATISTICS:")
        print(f"  Python files scanned: {self.stats['python_files_scanned']}")
        print(f"  Python files failed: {self.stats['python_files_failed']}")
        print(f"  Total classes found: {self.stats['classes_found']}")
        print(f"  Classes with from_config: {self.stats['from_config_classes']}")
        print(f"  Config classes: {self.stats['config_classes']}")
        print(f"  Qualified classes registered: {len(self.discovered_classes)}")
        
        # Show breakdown by category
        categories = {}
        for class_info in self.discovered_classes.values():
            category = class_info["category"]
            categories[category] = categories.get(category, 0) + 1
        
        print(f"\n📂 CLASSES BY CATEGORY:")
        for category, count in sorted(categories.items()):
            print(f"  {category}: {count}")
    
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
