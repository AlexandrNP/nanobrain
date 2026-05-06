"""
Enhanced Workflow Builder - REAL Implementation
===============================================

Uses comprehensive config discovery to build workflows with:
- Real class-to-config mappings from actual config files
- Schema validation using real Pydantic schemas
- Intelligent defaults based on priority system
- Parameter validation and error handling

BRUTAL TRUTH: This is what a workflow builder should actually be.
No more toy examples - this handles real framework complexity.
"""

import json
import yaml
import importlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field


@dataclass
class WorkflowComponent:
    """Represents a component in the workflow with full configuration."""
    
    name: str
    class_name: str
    class_path: str
    config_data: Dict[str, Any]
    config_file: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    schema: Optional[Dict[str, Any]] = None
    validation_errors: List[str] = field(default_factory=list)


@dataclass
class WorkflowConnection:
    """Represents a connection between workflow components."""
    
    source: str
    target: str
    source_port: Optional[str] = None
    target_port: Optional[str] = None
    link_class: str = "DirectLink"
    link_config: Dict[str, Any] = field(default_factory=dict)


class EnhancedWorkflowBuilder:
    """
    Enhanced workflow builder using comprehensive config discovery.
    
    BRUTAL TRUTH: This actually uses the real framework data instead
    of pretending with hardcoded examples.
    """
    
    def __init__(self, workflow_name: str, description: str = ""):
        """Initialize enhanced workflow builder."""
        
        self.workflow_name = workflow_name
        self.description = description
        
        # Import and initialize discovery
        self.discovery = self._initialize_discovery()
        
        # Workflow components
        self.components: Dict[str, WorkflowComponent] = {}
        self.connections: List[WorkflowConnection] = []
        self.input_data_units: Dict[str, WorkflowComponent] = {}
        self.output_data_units: Dict[str, WorkflowComponent] = {}
        
        # Available classes from discovery
        self.available_classes = self.discovery.list_available_classes()
        
        print(f"🔧 Enhanced Workflow Builder initialized")
        print(f"   Workflow: {workflow_name}")
        print(f"   Available classes: {len(self.available_classes)}")
    
    def _initialize_discovery(self):
        """Initialize the comprehensive config discovery."""
        
        try:
            # Import discovery module
            current_file = Path(__file__)
            discovery_path = current_file.parent / "comprehensive_config_discovery.py"
            
            import importlib.util
            spec = importlib.util.spec_from_file_location("comprehensive_config_discovery", discovery_path)
            discovery_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(discovery_module)
            
            # Create and run discovery
            discovery = discovery_module.ComprehensiveConfigDiscovery()
            discovery.discover_all_configs()
            
            return discovery
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize config discovery: {e}")
    
    def list_available_components(self, category: Optional[str] = None) -> List[str]:
        """List available components, optionally filtered by category."""
        
        if category is None:
            return self.available_classes
        
        # Filter by category based on class names and paths
        filtered_classes = []
        for class_name in self.available_classes:
            configs = self.discovery.get_all_configs_for_class(class_name)
            if configs:
                class_path = configs[0]["class_path"].lower()
                
                if category.lower() in class_path or category.lower() in class_name.lower():
                    filtered_classes.append(class_name)
        
        return filtered_classes
    
    def add_component(self, component_name: str, class_name: str, 
                     config_choice: Optional[str] = None, **parameters) -> 'EnhancedWorkflowBuilder':
        """
        Add a component to the workflow with real config validation.
        
        Args:
            component_name: Unique name for this component instance
            class_name: Name of the class to instantiate
            config_choice: Specific config file to use (optional, uses default if None)
            **parameters: Component parameters to override defaults
        """
        
        if component_name in self.components:
            raise ValueError(f"Component '{component_name}' already exists")
        
        if class_name not in self.available_classes:
            available = ", ".join(self.available_classes[:10])
            raise ValueError(f"Class '{class_name}' not available. Available: {available}...")
        
        # Get config for this class
        if config_choice:
            # User specified a specific config
            all_configs = self.discovery.get_all_configs_for_class(class_name)
            config_info = None
            for config in all_configs:
                if config_choice in config["relative_path"]:
                    config_info = config
                    break
            
            if config_info is None:
                available_configs = [c["relative_path"] for c in all_configs]
                raise ValueError(f"Config '{config_choice}' not found for {class_name}. Available: {available_configs}")
        else:
            # Use default (highest priority) config
            config_info = self.discovery.get_default_config_for_class(class_name)
            
            if config_info is None:
                raise ValueError(f"No config found for class '{class_name}'")
        
        # Create component
        component = WorkflowComponent(
            name=component_name,
            class_name=class_name,
            class_path=config_info["class_path"],
            config_data=config_info["config_data"],
            config_file=config_info["relative_path"],
            parameters=parameters
        )
        
        # Extract schema if possible
        component.schema = self._extract_component_schema(config_info)
        
        # Validate parameters against schema
        if component.schema:
            component.validation_errors = self._validate_parameters(parameters, component.schema)
        
        self.components[component_name] = component
        
        print(f"✅ Added component: {component_name} ({class_name})")
        print(f"   Config: {config_info['relative_path']}")
        if component.validation_errors:
            print(f"   ⚠️  Validation warnings: {len(component.validation_errors)}")
        
        return self
    
    def _extract_component_schema(self, config_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract schema for component configuration."""
        
        try:
            # Try to import the class and get its config schema
            module_path = ".".join(config_info["class_path"].split(".")[:-1])
            class_name = config_info["class_path"].split(".")[-1]
            
            module = importlib.import_module(module_path)
            component_class = getattr(module, class_name)
            
            # Look for config class
            config_class_name = f"{class_name}Config"
            if hasattr(module, config_class_name):
                config_class = getattr(module, config_class_name)
                
                # Try to get schema
                if hasattr(config_class, 'model_json_schema'):
                    schema = config_class.model_json_schema()
                    return self._simplify_schema(schema)
                elif hasattr(config_class, 'get_schema'):
                    schema = config_class.get_schema()
                    return self._simplify_schema(schema)
            
            return None
            
        except Exception:
            return None
    
    def _simplify_schema(self, pydantic_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Simplify Pydantic schema for easier validation."""
        
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
        
        return simple_schema
    
    def _validate_parameters(self, parameters: Dict[str, Any], 
                           schema: Dict[str, Any]) -> List[str]:
        """Validate parameters against schema."""
        
        errors = []
        
        # Check required fields
        for field_name, field_info in schema.items():
            if field_info.get("required", False) and field_name not in parameters:
                errors.append(f"Required parameter '{field_name}' missing")
        
        # Check parameter types (basic validation)
        for param_name, param_value in parameters.items():
            if param_name in schema:
                expected_type = schema[param_name].get("type")
                if expected_type == "string" and not isinstance(param_value, str):
                    errors.append(f"Parameter '{param_name}' should be string, got {type(param_value).__name__}")
                elif expected_type == "number" and not isinstance(param_value, (int, float)):
                    errors.append(f"Parameter '{param_name}' should be number, got {type(param_value).__name__}")
                elif expected_type == "boolean" and not isinstance(param_value, bool):
                    errors.append(f"Parameter '{param_name}' should be boolean, got {type(param_value).__name__}")
        
        return errors

    def add_input(self, input_name: str, data_unit_class: str = "DataUnitMemory",
                  **parameters) -> 'EnhancedWorkflowBuilder':
        """Add an input data unit to the workflow."""

        component = self.add_component(f"input_{input_name}", data_unit_class, **parameters)
        self.input_data_units[input_name] = self.components[f"input_{input_name}"]

        print(f"📥 Added input: {input_name} ({data_unit_class})")
        return self

    def add_output(self, output_name: str, data_unit_class: str = "DataUnitMemory",
                   **parameters) -> 'EnhancedWorkflowBuilder':
        """Add an output data unit to the workflow."""

        component = self.add_component(f"output_{output_name}", data_unit_class, **parameters)
        self.output_data_units[output_name] = self.components[f"output_{output_name}"]

        print(f"📤 Added output: {output_name} ({data_unit_class})")
        return self

    def connect(self, source: str, target: str, link_class: str = "DirectLink",
                **link_parameters) -> 'EnhancedWorkflowBuilder':
        """Connect two components in the workflow."""

        # Validate components exist
        if source not in self.components and f"input_{source}" not in self.components:
            raise ValueError(f"Source component '{source}' not found")

        if target not in self.components and f"output_{target}" not in self.components:
            raise ValueError(f"Target component '{target}' not found")

        # Resolve actual component names
        actual_source = source if source in self.components else f"input_{source}"
        actual_target = target if target in self.components else f"output_{target}"

        # Create connection
        connection = WorkflowConnection(
            source=actual_source,
            target=actual_target,
            link_class=link_class,
            link_config=link_parameters
        )

        self.connections.append(connection)

        print(f"🔗 Connected: {actual_source} → {actual_target} ({link_class})")
        return self

    def get_workflow_config(self) -> Dict[str, Any]:
        """Generate complete workflow configuration."""

        # Build steps configuration
        steps = {}
        for comp_name, component in self.components.items():
            if not comp_name.startswith("input_") and not comp_name.startswith("output_"):
                steps[comp_name] = {
                    "class": component.class_path,
                    "config_file": component.config_file,
                    **component.parameters
                }

        # Build links configuration
        links = []
        for connection in self.connections:
            links.append({
                "source": connection.source,
                "target": connection.target,
                "class": connection.link_class,
                **connection.link_config
            })

        # Build data units configuration
        input_data_units = {}
        for input_name, component in self.input_data_units.items():
            input_data_units[input_name] = {
                "class": component.class_path,
                "config_file": component.config_file,
                **component.parameters
            }

        output_data_units = {}
        for output_name, component in self.output_data_units.items():
            output_data_units[output_name] = {
                "class": component.class_path,
                "config_file": component.config_file,
                **component.parameters
            }

        # Complete workflow configuration
        workflow_config = {
            "name": self.workflow_name,
            "description": self.description,
            "version": "1.0.0",
            "steps": steps,
            "links": links,
            "input_data_units": input_data_units,
            "output_data_units": output_data_units,
            "metadata": {
                "builder": "EnhancedWorkflowBuilder",
                "total_components": len(self.components),
                "total_connections": len(self.connections),
                "validation_errors": self._get_all_validation_errors()
            }
        }

        return workflow_config

    def _get_all_validation_errors(self) -> List[str]:
        """Get all validation errors from all components."""

        all_errors = []
        for component in self.components.values():
            all_errors.extend(component.validation_errors)

        return all_errors

    def save_workflow(self, file_path: str, format: str = "yaml") -> str:
        """Save workflow configuration to file in YAML or JSON format."""

        config = self.get_workflow_config()

        output_path = Path(file_path)

        # Ensure correct file extension
        if format.lower() == "yaml" and not str(output_path).endswith(('.yml', '.yaml')):
            output_path = output_path.with_suffix('.yml')
        elif format.lower() == "json" and not str(output_path).endswith('.json'):
            output_path = output_path.with_suffix('.json')

        with open(output_path, 'w') as f:
            if format.lower() == "yaml":
                yaml.dump(config, f, default_flow_style=False, indent=2, sort_keys=False)
            else:
                json.dump(config, f, indent=2)

        print(f"💾 Saved workflow to: {output_path} ({format.upper()} format)")
        return str(output_path)

    def get_workflow_yaml(self) -> str:
        """Get workflow configuration as YAML string."""

        config = self.get_workflow_config()
        return yaml.dump(config, default_flow_style=False, indent=2, sort_keys=False)

    def print_workflow_yaml(self):
        """Print workflow configuration in YAML format."""

        print(f"\n📄 WORKFLOW CONFIGURATION (YAML):")
        print("=" * 50)
        yaml_content = self.get_workflow_yaml()
        print(yaml_content)

    def validate_workflow(self) -> Dict[str, Any]:
        """Validate the complete workflow."""

        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "component_count": len(self.components),
            "connection_count": len(self.connections)
        }

        # Check for validation errors in components
        all_errors = self._get_all_validation_errors()
        if all_errors:
            validation_result["errors"].extend(all_errors)
            validation_result["valid"] = False

        # Check for disconnected components
        connected_components = set()
        for connection in self.connections:
            connected_components.add(connection.source)
            connected_components.add(connection.target)

        for comp_name in self.components:
            if comp_name not in connected_components:
                validation_result["warnings"].append(f"Component '{comp_name}' is not connected")

        # Check for missing inputs/outputs
        if not self.input_data_units:
            validation_result["warnings"].append("No input data units defined")

        if not self.output_data_units:
            validation_result["warnings"].append("No output data units defined")

        return validation_result

    def print_workflow_summary(self):
        """Print a summary of the workflow."""

        print(f"\n📋 WORKFLOW SUMMARY: {self.workflow_name}")
        print("=" * 50)

        print(f"📝 Description: {self.description}")
        print(f"🔧 Components: {len(self.components)}")
        print(f"🔗 Connections: {len(self.connections)}")
        print(f"📥 Inputs: {len(self.input_data_units)}")
        print(f"📤 Outputs: {len(self.output_data_units)}")

        # Show components
        if self.components:
            print(f"\n🔧 COMPONENTS:")
            for comp_name, component in self.components.items():
                status = "⚠️" if component.validation_errors else "✅"
                print(f"  {status} {comp_name}: {component.class_name}")
                print(f"     Config: {component.config_file}")
                if component.parameters:
                    print(f"     Parameters: {list(component.parameters.keys())}")

        # Show connections
        if self.connections:
            print(f"\n🔗 CONNECTIONS:")
            for connection in self.connections:
                print(f"  {connection.source} → {connection.target} ({connection.link_class})")

        # Show validation status
        validation = self.validate_workflow()
        if validation["valid"]:
            print(f"\n✅ Workflow is valid")
        else:
            print(f"\n❌ Workflow has {len(validation['errors'])} errors")
            for error in validation["errors"]:
                print(f"   - {error}")

        if validation["warnings"]:
            print(f"\n⚠️  {len(validation['warnings'])} warnings:")
            for warning in validation["warnings"]:
                print(f"   - {warning}")
