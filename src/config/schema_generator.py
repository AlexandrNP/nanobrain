"""
Schema Generation System for NanoBrain Framework

Generates JSON schemas for YAML configuration validation.
"""

import json
import logging
from typing import Any, Dict, Optional, Type, Union
from pathlib import Path
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class SchemaGenerator:
    """
    Generator for JSON schemas from Pydantic models.
    """
    
    def __init__(self):
        self._schemas: Dict[str, Dict[str, Any]] = {}
    
    def generate_schema(self, model_class: Type[BaseModel], 
                       title: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate JSON schema for a Pydantic model.
        
        Args:
            model_class: Pydantic model class
            title: Optional schema title
            
        Returns:
            JSON schema dictionary
        """
        schema = model_class.schema()
        
        if title:
            schema["title"] = title
        
        # Add custom properties for NanoBrain
        schema["$schema"] = "http://json-schema.org/draft-07/schema#"
        schema["nanobrain_version"] = "2.0.0"
        
        # Store schema
        schema_name = title or model_class.__name__
        self._schemas[schema_name] = schema
        
        logger.debug(f"Generated schema for {schema_name}")
        return schema
    
    def generate_workflow_schema(self) -> Dict[str, Any]:
        """
        Generate comprehensive schema for workflow configurations.
        
        Returns:
            Complete workflow schema
        """
        from ..core.executor import ExecutorConfig
        from ..core.data_unit import DataUnitConfig
        from ..core.trigger import TriggerConfig
        from ..core.link import LinkConfig
        from ..core.step import StepConfig
        from ..core.agent import AgentConfig
        from ..core.tool import ToolConfig
        from .yaml_config import WorkflowConfig
        
        # Generate individual schemas
        executor_schema = self.generate_schema(ExecutorConfig, "ExecutorConfig")
        data_unit_schema = self.generate_schema(DataUnitConfig, "DataUnitConfig")
        trigger_schema = self.generate_schema(TriggerConfig, "TriggerConfig")
        link_schema = self.generate_schema(LinkConfig, "LinkConfig")
        step_schema = self.generate_schema(StepConfig, "StepConfig")
        agent_schema = self.generate_schema(AgentConfig, "AgentConfig")
        tool_schema = self.generate_schema(ToolConfig, "ToolConfig")
        workflow_schema = self.generate_schema(WorkflowConfig, "WorkflowConfig")
        
        # Create comprehensive schema with references
        comprehensive_schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "NanoBrain Workflow Configuration",
            "description": "Complete schema for NanoBrain workflow configurations",
            "nanobrain_version": "2.0.0",
            "type": "object",
            "properties": workflow_schema["properties"],
            "required": workflow_schema.get("required", []),
            "definitions": {
                "ExecutorConfig": executor_schema,
                "DataUnitConfig": data_unit_schema,
                "TriggerConfig": trigger_schema,
                "LinkConfig": link_schema,
                "StepConfig": step_schema,
                "AgentConfig": agent_schema,
                "ToolConfig": tool_schema
            }
        }
        
        return comprehensive_schema
    
    def save_schema(self, schema: Dict[str, Any], file_path: Union[str, Path]) -> None:
        """
        Save schema to JSON file.
        
        Args:
            schema: Schema dictionary
            file_path: Path to save file
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(schema, f, indent=2, sort_keys=False)
        
        logger.info(f"Schema saved to {file_path}")
    
    def save_all_schemas(self, output_dir: Union[str, Path]) -> None:
        """
        Save all generated schemas to directory.
        
        Args:
            output_dir: Output directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for schema_name, schema in self._schemas.items():
            file_path = output_dir / f"{schema_name.lower()}.schema.json"
            self.save_schema(schema, file_path)
        
        # Save comprehensive workflow schema
        workflow_schema = self.generate_workflow_schema()
        self.save_schema(workflow_schema, output_dir / "workflow.schema.json")
        
        logger.info(f"All schemas saved to {output_dir}")
    
    def get_schema(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get a generated schema by name.
        
        Args:
            name: Schema name
            
        Returns:
            Schema dictionary or None
        """
        return self._schemas.get(name)
    
    def list_schemas(self) -> list[str]:
        """
        List all generated schema names.
        
        Returns:
            List of schema names
        """
        return list(self._schemas.keys())
    
    def generate_ui_schema(self, model_class: Type[BaseModel]) -> Dict[str, Any]:
        """
        Generate UI schema for form generation.
        
        Args:
            model_class: Pydantic model class
            
        Returns:
            UI schema dictionary
        """
        schema = model_class.schema()
        ui_schema = {}
        
        # Generate UI hints based on field types and metadata
        for field_name, field_info in schema.get("properties", {}).items():
            field_type = field_info.get("type")
            field_format = field_info.get("format")
            
            ui_field = {}
            
            # Set UI widget based on type
            if field_type == "string":
                if field_format == "password":
                    ui_field["ui:widget"] = "password"
                elif "description" in field_info and len(field_info["description"]) > 100:
                    ui_field["ui:widget"] = "textarea"
                else:
                    ui_field["ui:widget"] = "text"
            elif field_type == "boolean":
                ui_field["ui:widget"] = "checkbox"
            elif field_type == "number" or field_type == "integer":
                ui_field["ui:widget"] = "updown"
            elif field_type == "array":
                ui_field["ui:widget"] = "array"
            elif field_type == "object":
                ui_field["ui:widget"] = "object"
            
            # Add help text from description
            if "description" in field_info:
                ui_field["ui:help"] = field_info["description"]
            
            # Set field order
            ui_field["ui:order"] = list(schema.get("properties", {}).keys()).index(field_name)
            
            ui_schema[field_name] = ui_field
        
        return ui_schema
    
    def generate_example_data(self, model_class: Type[BaseModel]) -> Dict[str, Any]:
        """
        Generate example data for a model.
        
        Args:
            model_class: Pydantic model class
            
        Returns:
            Example data dictionary
        """
        try:
            # Create instance with default values
            instance = model_class()
            return instance.dict()
        except Exception as e:
            logger.warning(f"Could not generate example for {model_class.__name__}: {e}")
            return {}


def generate_schema(model_class: Type[BaseModel], 
                   title: Optional[str] = None,
                   save_to: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Convenience function to generate a schema.
    
    Args:
        model_class: Pydantic model class
        title: Optional schema title
        save_to: Optional file path to save schema
        
    Returns:
        JSON schema dictionary
    """
    generator = SchemaGenerator()
    schema = generator.generate_schema(model_class, title)
    
    if save_to:
        generator.save_schema(schema, save_to)
    
    return schema


def generate_all_schemas(output_dir: Union[str, Path] = "schemas") -> None:
    """
    Generate all NanoBrain schemas.
    
    Args:
        output_dir: Output directory for schemas
    """
    generator = SchemaGenerator()
    
    # Import all config classes
    from ..core.executor import ExecutorConfig
    from ..core.data_unit import DataUnitConfig
    from ..core.trigger import TriggerConfig
    from ..core.link import LinkConfig
    from ..core.step import StepConfig
    from ..core.agent import AgentConfig
    from ..core.tool import ToolConfig
    from .yaml_config import WorkflowConfig, YAMLConfig
    
    # Generate all schemas
    config_classes = [
        (ExecutorConfig, "Executor Configuration"),
        (DataUnitConfig, "Data Unit Configuration"),
        (TriggerConfig, "Trigger Configuration"),
        (LinkConfig, "Link Configuration"),
        (StepConfig, "Step Configuration"),
        (AgentConfig, "Agent Configuration"),
        (ToolConfig, "Tool Configuration"),
        (WorkflowConfig, "Workflow Configuration"),
        (YAMLConfig, "Base YAML Configuration")
    ]
    
    for config_class, title in config_classes:
        generator.generate_schema(config_class, title)
    
    # Save all schemas
    generator.save_all_schemas(output_dir)
    
    logger.info(f"Generated {len(config_classes)} schemas in {output_dir}")


def validate_config_against_schema(config_dict: Dict[str, Any], 
                                 schema: Dict[str, Any]) -> list[str]:
    """
    Validate configuration against JSON schema.
    
    Args:
        config_dict: Configuration dictionary
        schema: JSON schema
        
    Returns:
        List of validation errors
    """
    try:
        import jsonschema
        
        validator = jsonschema.Draft7Validator(schema)
        errors = []
        
        for error in validator.iter_errors(config_dict):
            error_path = " -> ".join(str(p) for p in error.path)
            error_msg = f"{error_path}: {error.message}" if error_path else error.message
            errors.append(error_msg)
        
        return errors
        
    except ImportError:
        logger.warning("jsonschema not available. Install with: pip install jsonschema")
        return []
    except Exception as e:
        logger.error(f"Schema validation error: {e}")
        return [str(e)] 