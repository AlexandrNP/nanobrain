"""
Lightweight Workflow Builder - Iteration 1
==========================================

Simple workflow builder that generates configuration using discovered components.
Uses only standard library - no external dependencies.
"""

from typing import Dict, List, Any, Optional

# Import discovery - handle both relative and absolute imports
try:
    from .discovery_minimal import MinimalConfigDiscovery
except ImportError:
    # Fallback for isolated testing
    import sys
    from pathlib import Path
    discovery_path = Path(__file__).parent / "discovery_minimal.py"
    import importlib.util
    spec = importlib.util.spec_from_file_location("discovery_minimal", discovery_path)
    discovery_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(discovery_module)
    MinimalConfigDiscovery = discovery_module.MinimalConfigDiscovery


class WorkflowBuilder:
    """
    Lightweight workflow builder that generates proper configuration.
    
    BRUTAL TRUTH: This is iteration 1 - basic functionality only.
    No fancy features, just core workflow building with discovered components.
    """
    
    def __init__(self, name: str, description: str = ""):
        """Initialize workflow builder."""
        
        self.name = name
        self.description = description
        
        # Initialize discovery system
        self.discovery = MinimalConfigDiscovery()
        self.discovered_classes = self.discovery.discover_from_files()
        
        # Workflow configuration structure
        self.workflow_config = {
            "name": name,
            "description": description,
            "version": "2.0",
            "steps": {},
            "links": {},
            "input_data_units": {},
            "output_data_units": {}
        }
        
        self._step_counter = 0
    
    def add_step(self, step_name: str, component_class: str, **kwargs) -> 'WorkflowBuilder':
        """
        Add a step to the workflow using a discovered component.
        
        Args:
            step_name: Name for this step
            component_class: Class name (e.g., "EnhancedCollaborativeAgent")
            **kwargs: Additional configuration parameters
        
        Returns:
            Self for method chaining
        """
        
        # Validate component exists
        if component_class not in self.discovered_classes:
            available = list(self.discovered_classes.keys())
            raise ValueError(f"Unknown component '{component_class}'. Available: {available}")
        
        # Get component info
        component_info = self.discovered_classes[component_class]
        
        # Build step configuration
        step_config = {
            "name": step_name,
            "class": component_info["class_path"],
            "description": kwargs.get("description", f"Step using {component_class}")
        }
        
        # Add user-provided parameters
        for param, value in kwargs.items():
            if param != "description":  # Already handled above
                step_config[param] = value
        
        # Add to workflow
        self.workflow_config["steps"][step_name] = step_config
        self._step_counter += 1
        
        return self
    
    def add_input(self, name: str, data_unit_type: str = "DataUnitMemory") -> 'WorkflowBuilder':
        """Add input data unit to workflow."""
        
        # Find data unit class
        data_unit_classes = self.discovery.get_classes_by_category("data_unit")
        
        if data_unit_type not in data_unit_classes:
            raise ValueError(f"Unknown data unit type '{data_unit_type}'. Available: {data_unit_classes}")
        
        data_unit_info = self.discovered_classes[data_unit_type]
        
        input_config = {
            "name": name,
            "class": data_unit_info["class_path"],
            "description": f"Input data unit: {name}"
        }
        
        self.workflow_config["input_data_units"][name] = input_config
        return self
    
    def add_output(self, name: str, data_unit_type: str = "DataUnitMemory") -> 'WorkflowBuilder':
        """Add output data unit to workflow."""
        
        # Find data unit class
        data_unit_classes = self.discovery.get_classes_by_category("data_unit")
        
        if data_unit_type not in data_unit_classes:
            raise ValueError(f"Unknown data unit type '{data_unit_type}'. Available: {data_unit_classes}")
        
        data_unit_info = self.discovered_classes[data_unit_type]
        
        output_config = {
            "name": name,
            "class": data_unit_info["class_path"],
            "description": f"Output data unit: {name}"
        }
        
        self.workflow_config["output_data_units"][name] = output_config
        return self
    
    def connect(self, source: str, target: str) -> 'WorkflowBuilder':
        """Connect two components with a link."""
        
        # Find link class
        link_classes = self.discovery.get_classes_by_category("link")
        
        if not link_classes:
            raise ValueError("No link classes discovered")
        
        # Use first available link class (DirectLink if available)
        link_class = "DirectLink" if "DirectLink" in link_classes else link_classes[0]
        link_info = self.discovered_classes[link_class]
        
        link_name = f"link_{len(self.workflow_config['links'])}"
        
        link_config = {
            "name": link_name,
            "class": link_info["class_path"],
            "source": source,
            "target": target
        }
        
        self.workflow_config["links"][link_name] = link_config
        return self
    
    def get_config(self) -> Dict[str, Any]:
        """Get the generated workflow configuration."""
        return self.workflow_config.copy()
    
    def save_config(self, file_path: str) -> str:
        """Save workflow configuration to file."""
        
        import json
        
        with open(file_path, 'w') as f:
            json.dump(self.workflow_config, f, indent=2)
        
        return file_path
    
    def list_available_components(self, category: Optional[str] = None) -> List[str]:
        """List available components by category."""
        
        if category:
            return self.discovery.get_classes_by_category(category)
        else:
            return list(self.discovered_classes.keys())
    
    def get_component_info(self, component_class: str) -> Optional[Dict[str, Any]]:
        """Get information about a component."""
        
        return self.discovered_classes.get(component_class)
