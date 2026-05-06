"""
Refined Discovery System - Iteration 2
======================================

Improves on comprehensive discovery by:
1. Separating component classes from config classes
2. Extracting actual schemas from config classes
3. Filtering out abstract/unusable classes
4. Better categorization logic

BRUTAL TRUTH: The previous iteration was quantity over quality.
This iteration focuses on USABLE, HIGH-QUALITY discovery results.
"""

import os
import ast
import importlib
import inspect
from pathlib import Path
from typing import Dict, List, Any, Optional, Set


class RefinedDiscovery:
    """
    Refined discovery that focuses on USABLE classes with proper schemas.
    
    BRUTAL TRUTH: This separates the wheat from the chaff.
    We want component classes that can be instantiated, not abstract bases.
    """
    
    def __init__(self, framework_root: Optional[str] = None):
        """Initialize refined discovery."""
        
        if framework_root is None:
            current_file = Path(__file__)
            self.framework_root = current_file.parent.parent
        else:
            self.framework_root = Path(framework_root)
        
        # Separate storage for different types of classes
        self.component_classes: Dict[str, Dict[str, Any]] = {}
        self.config_classes: Dict[str, Dict[str, Any]] = {}
        
        # Directories to scan
        self.scan_directories = [
            self.framework_root / "core",
            self.framework_root / "library",
        ]
        
        # Statistics
        self.stats = {
            "python_files_scanned": 0,
            "total_classes_found": 0,
            "component_classes": 0,
            "config_classes": 0,
            "abstract_classes_skipped": 0,
            "schemas_extracted": 0
        }
    
    def discover_refined_classes(self) -> Dict[str, Any]:
        """
        Discover and refine classes, separating components from configs.
        
        Returns:
            Dictionary with 'components' and 'configs' keys
        """
        print(f"🔍 Starting REFINED discovery from {self.framework_root}")
        
        # Clear previous results
        self.component_classes.clear()
        self.config_classes.clear()
        self.stats = {k: 0 for k in self.stats.keys()}
        
        # Scan directories
        for scan_dir in self.scan_directories:
            if scan_dir.exists():
                print(f"📂 Scanning directory: {scan_dir}")
                self._scan_directory_refined(scan_dir)
        
        # Extract schemas from config classes
        print(f"🔬 Extracting schemas from config classes...")
        self._extract_schemas()
        
        print(f"✅ Refined discovery complete")
        self._print_refined_stats()
        
        return {
            "components": self.component_classes,
            "configs": self.config_classes
        }
    
    def _scan_directory_refined(self, directory: Path) -> None:
        """Scan directory with refined analysis."""
        
        python_files = [f for f in directory.rglob("*.py") 
                       if "__pycache__" not in str(f) and not f.name.startswith("__")]
        
        print(f"   Analyzing {len(python_files)} Python files...")
        
        for py_file in python_files:
            self._analyze_file_refined(py_file)
    
    def _analyze_file_refined(self, py_file: Path) -> None:
        """Analyze file with refined logic."""
        
        try:
            self.stats["python_files_scanned"] += 1
            
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            try:
                tree = ast.parse(content)
            except SyntaxError:
                return
            
            # Analyze each class
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    self._analyze_class_refined(node, py_file, content)
                    
        except Exception:
            pass  # Skip problematic files
    
    def _analyze_class_refined(self, class_node: ast.ClassDef, py_file: Path, file_content: str) -> None:
        """Analyze class with refined logic to separate components from configs."""
        
        class_name = class_node.name
        self.stats["total_classes_found"] += 1
        
        # Skip abstract classes and base classes
        if self._is_abstract_or_base_class(class_node, class_name):
            self.stats["abstract_classes_skipped"] += 1
            return
        
        # Determine if this is a config class or component class
        is_config_class = self._is_config_class(class_node, class_name)
        has_from_config = self._has_from_config_method(class_node, file_content)
        
        # Generate module path with nanobrain prefix
        relative_path = py_file.relative_to(self.framework_root.parent)
        module_path = str(relative_path.with_suffix("")).replace(os.sep, ".")
        
        if is_config_class:
            self._register_config_class(class_name, module_path, py_file)
        elif has_from_config:
            self._register_component_class(class_name, module_path, py_file)
    
    def _is_abstract_or_base_class(self, class_node: ast.ClassDef, class_name: str) -> bool:
        """Check if class is abstract or a base class that shouldn't be instantiated."""
        
        # Skip classes with "Base" in the name
        if "Base" in class_name and class_name.endswith("Base"):
            return True
        
        # Skip classes with "Abstract" in the name
        if "Abstract" in class_name:
            return True
        
        # Check for ABC decorators or inheritance
        for decorator in class_node.decorator_list:
            if isinstance(decorator, ast.Name) and decorator.id in ["abstractmethod", "ABC"]:
                return True
        
        # Check for ABC inheritance
        for base in class_node.bases:
            if isinstance(base, ast.Name) and base.id == "ABC":
                return True
        
        return False
    
    def _is_config_class(self, class_node: ast.ClassDef, class_name: str) -> bool:
        """Check if this is a configuration class."""
        
        # Check class name
        if class_name.endswith("Config"):
            return True
        
        # Check inheritance
        for base in class_node.bases:
            if isinstance(base, ast.Name):
                base_name = base.id
                if base_name in ["ConfigBase", "BaseModel"]:
                    return True
            elif isinstance(base, ast.Attribute):
                base_str = ast.unparse(base)
                if "ConfigBase" in base_str or "BaseModel" in base_str:
                    return True
        
        return False
    
    def _has_from_config_method(self, class_node: ast.ClassDef, file_content: str) -> bool:
        """Check if class has from_config classmethod."""
        
        # AST check
        for item in class_node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "from_config":
                for decorator in item.decorator_list:
                    if isinstance(decorator, ast.Name) and decorator.id == "classmethod":
                        return True
        
        # Text search backup
        class_start = class_node.lineno
        class_end = getattr(class_node, 'end_lineno', class_start + 50)
        lines = file_content.split('\n')
        class_content = '\n'.join(lines[class_start-1:class_end])
        
        return "@classmethod" in class_content and "def from_config" in class_content
    
    def _register_config_class(self, class_name: str, module_path: str, py_file: Path) -> None:
        """Register a configuration class."""
        
        if class_name in self.config_classes:
            return
        
        self.config_classes[class_name] = {
            "class_name": class_name,
            "class_path": f"{module_path}.{class_name}",
            "module_path": module_path,
            "file_path": str(py_file),
            "schema": None,  # Will be extracted later
            "category": "config"
        }
        
        self.stats["config_classes"] += 1
    
    def _register_component_class(self, class_name: str, module_path: str, py_file: Path) -> None:
        """Register a component class."""
        
        if class_name in self.component_classes:
            return
        
        category = self._categorize_component(module_path, class_name)
        
        self.component_classes[class_name] = {
            "class_name": class_name,
            "class_path": f"{module_path}.{class_name}",
            "module_path": module_path,
            "file_path": str(py_file),
            "category": category,
            "config_class": self._find_config_class_for_component(class_name),
            "schema": None  # Will be extracted from config class
        }
        
        self.stats["component_classes"] += 1
    
    def _categorize_component(self, module_path: str, class_name: str) -> str:
        """Categorize component class."""
        
        path_lower = module_path.lower()
        name_lower = class_name.lower()
        
        # Module path based categorization
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
        
        # Class name based categorization
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
        else:
            return "unknown"
    
    def _find_config_class_for_component(self, component_name: str) -> Optional[str]:
        """Find the corresponding config class for a component."""
        
        # Standard naming pattern: ComponentName -> ComponentNameConfig
        expected_config_name = f"{component_name}Config"
        
        # Check if we've discovered this config class
        if expected_config_name in self.config_classes:
            return expected_config_name
        
        return None
    
    def _extract_schemas(self) -> None:
        """Extract schemas from config classes using real imports."""
        
        for config_name, config_info in self.config_classes.items():
            try:
                schema = self._extract_schema_from_config(config_info)
                if schema:
                    config_info["schema"] = schema
                    self.stats["schemas_extracted"] += 1
                    
                    # Also add schema to corresponding component
                    component_name = config_name.replace("Config", "")
                    if component_name in self.component_classes:
                        self.component_classes[component_name]["schema"] = schema
                        
            except Exception as e:
                # Schema extraction failed for this class
                pass
    
    def _extract_schema_from_config(self, config_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract schema from a config class by importing it."""
        
        try:
            # Import the module
            module = importlib.import_module(config_info["module_path"])
            
            # Get the class
            config_class = getattr(module, config_info["class_name"])
            
            # Check if it has get_schema method
            if hasattr(config_class, 'get_schema'):
                schema = config_class.get_schema()
                return self._simplify_pydantic_schema(schema)
            
            # Check if it has model_json_schema method (Pydantic V2)
            if hasattr(config_class, 'model_json_schema'):
                schema = config_class.model_json_schema()
                return self._simplify_pydantic_schema(schema)
            
            return None
            
        except Exception:
            return None
    
    def _simplify_pydantic_schema(self, pydantic_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Pydantic schema to simplified format."""
        
        properties = pydantic_schema.get("properties", {})
        required = pydantic_schema.get("required", [])
        
        simple_schema = {}
        
        for field_name, field_info in properties.items():
            simple_schema[field_name] = {
                "type": field_info.get("type", "any"),
                "required": field_name in required,
                "description": field_info.get("description", ""),
                "default": field_info.get("default")
            }
            
            # Add constraints
            if "minimum" in field_info:
                simple_schema[field_name]["min"] = field_info["minimum"]
            if "maximum" in field_info:
                simple_schema[field_name]["max"] = field_info["maximum"]
            if "enum" in field_info:
                simple_schema[field_name]["choices"] = field_info["enum"]
        
        return simple_schema

    def _print_refined_stats(self) -> None:
        """Print refined discovery statistics."""

        print(f"\n📊 REFINED DISCOVERY STATISTICS:")
        print(f"  Python files scanned: {self.stats['python_files_scanned']}")
        print(f"  Total classes found: {self.stats['total_classes_found']}")
        print(f"  Abstract classes skipped: {self.stats['abstract_classes_skipped']}")
        print(f"  Component classes: {self.stats['component_classes']}")
        print(f"  Config classes: {self.stats['config_classes']}")
        print(f"  Schemas extracted: {self.stats['schemas_extracted']}")

        # Component breakdown
        component_categories = {}
        for comp_info in self.component_classes.values():
            category = comp_info["category"]
            component_categories[category] = component_categories.get(category, 0) + 1

        print(f"\n📂 COMPONENT CLASSES BY CATEGORY:")
        for category, count in sorted(component_categories.items()):
            print(f"  {category}: {count}")

    def get_component_classes_by_category(self, category: str) -> List[str]:
        """Get component class names by category."""

        return [
            class_name for class_name, class_info in self.component_classes.items()
            if class_info["category"] == category
        ]

    def get_component_info(self, class_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a component class."""

        return self.component_classes.get(class_name)

    def get_config_info(self, class_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a config class."""

        return self.config_classes.get(class_name)

    def list_all_components(self) -> List[str]:
        """List all component class names."""

        return list(self.component_classes.keys())

    def list_all_configs(self) -> List[str]:
        """List all config class names."""

        return list(self.config_classes.keys())
