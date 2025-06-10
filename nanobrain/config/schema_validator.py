"""
Schema Validation System for NanoBrain Framework

Provides advanced validation capabilities for YAML configurations.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Union, Callable, Type
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict, ValidationError, field_validator
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)


class FieldType(Enum):
    """Supported field types for validation."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    LIST = "list"
    DICT = "dict"
    OBJECT = "object"


class ConstraintType(Enum):
    """Types of constraints that can be applied."""
    MIN = "min"
    MAX = "max"
    MIN_LENGTH = "min_length"
    MAX_LENGTH = "max_length"
    PATTERN = "pattern"
    ENUM = "enum"
    TYPE = "type"
    REQUIRED = "required"


class FieldConstraint(BaseModel):
    """Represents a constraint on a field."""
    model_config = ConfigDict(use_enum_values=True)
    
    constraint_type: ConstraintType
    value: Any
    message: Optional[str] = None


class FieldSchema(BaseModel):
    """Schema definition for a field."""
    model_config = ConfigDict(use_enum_values=True)
    
    name: str
    field_type: FieldType
    description: Optional[str] = None
    required: bool = False
    default: Any = None
    constraints: List[FieldConstraint] = Field(default_factory=list)


class ParameterSchema(BaseModel):
    """Schema for operation-specific parameters."""
    model_config = ConfigDict(use_enum_values=True)
    
    operation: str
    description: Optional[str] = None
    fields: List[FieldSchema] = Field(default_factory=list)


class ValidatorFunction(BaseModel):
    """Custom validator function definition."""
    model_config = ConfigDict(use_enum_values=True)
    
    name: str
    fields: List[str] = Field(default_factory=list)
    pre: bool = False
    code: str
    description: Optional[str] = None


class ConfigSchema(BaseModel):
    """Complete configuration schema."""
    model_config = ConfigDict(use_enum_values=True)
    
    name: str
    version: str = "1.0.0"
    description: Optional[str] = None
    
    # Field definitions
    fields: List[FieldSchema] = Field(default_factory=list)
    
    # Parameter schemas for different operations
    parameters: List[ParameterSchema] = Field(default_factory=list)
    
    # Custom validators
    validators: List[ValidatorFunction] = Field(default_factory=list)
    
    # Validation rules
    required_fields: List[str] = Field(default_factory=list)
    optional_fields: List[str] = Field(default_factory=list)


class SchemaValidator:
    """
    Advanced schema validator for YAML configurations.
    
    Provides comprehensive validation including field constraints,
    custom validators, and operation-specific parameter validation.
    """
    
    def __init__(self, schema: Optional[ConfigSchema] = None):
        """
        Initialize the schema validator.
        
        Args:
            schema: Configuration schema to use for validation
        """
        self.schema = schema
        self._compiled_validators: Dict[str, Callable] = {}
        
        if schema:
            self._compile_validators()
    
    def load_schema_from_yaml(self, schema_path: Union[str, Path]) -> None:
        """Load schema from YAML file."""
        schema_path = Path(schema_path)
        
        if not schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {schema_path}")
        
        with open(schema_path, 'r') as f:
            schema_data = yaml.safe_load(f)
        
        self.schema = ConfigSchema(**schema_data)
        self._compile_validators()
        
        logger.debug(f"Loaded schema from {schema_path}")
    
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a configuration against the schema.
        
        Args:
            config: Configuration to validate
            
        Returns:
            Validated and processed configuration
            
        Raises:
            ValidationError: If validation fails
        """
        if not self.schema:
            logger.warning("No schema loaded, skipping validation")
            return config
        
        validated_config = config.copy()
        errors = []
        
        # Validate required fields
        for field_name in self.schema.required_fields:
            if field_name not in validated_config:
                errors.append(f"Required field missing: {field_name}")
        
        # Validate field schemas
        for field_schema in self.schema.fields:
            field_name = field_schema.name
            
            if field_name in validated_config:
                try:
                    validated_config[field_name] = self._validate_field(
                        field_schema, validated_config[field_name]
                    )
                except ValueError as e:
                    errors.append(f"Field '{field_name}': {e}")
            elif field_schema.required:
                errors.append(f"Required field missing: {field_name}")
            elif field_schema.default is not None:
                validated_config[field_name] = field_schema.default
        
        # Run custom validators
        for validator_name, validator_func in self._compiled_validators.items():
            try:
                validated_config = validator_func(validated_config)
            except Exception as e:
                errors.append(f"Validator '{validator_name}' failed: {e}")
        
        if errors:
            raise ValidationError(f"Validation failed: {'; '.join(errors)}")
        
        return validated_config
    
    def validate_parameters(self, operation: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate operation-specific parameters.
        
        Args:
            operation: Operation name
            parameters: Parameters to validate
            
        Returns:
            Validated parameters
        """
        if not self.schema:
            return parameters
        
        # Find parameter schema for operation
        param_schema = None
        for schema in self.schema.parameters:
            if schema.operation == operation:
                param_schema = schema
                break
        
        if not param_schema:
            logger.warning(f"No parameter schema found for operation: {operation}")
            return parameters
        
        validated_params = parameters.copy()
        errors = []
        
        # Validate each field in the parameter schema
        for field_schema in param_schema.fields:
            field_name = field_schema.name
            
            if field_name in validated_params:
                try:
                    validated_params[field_name] = self._validate_field(
                        field_schema, validated_params[field_name]
                    )
                except ValueError as e:
                    errors.append(f"Parameter '{field_name}': {e}")
            elif field_schema.required:
                errors.append(f"Required parameter missing: {field_name}")
            elif field_schema.default is not None:
                validated_params[field_name] = field_schema.default
        
        if errors:
            raise ValidationError(f"Parameter validation failed: {'; '.join(errors)}")
        
        return validated_params
    
    def _validate_field(self, field_schema: FieldSchema, value: Any) -> Any:
        """Validate a single field against its schema."""
        # Type validation
        validated_value = self._validate_type(field_schema.field_type, value)
        
        # Constraint validation
        for constraint in field_schema.constraints:
            self._validate_constraint(constraint, validated_value, field_schema.name)
        
        return validated_value
    
    def _validate_type(self, field_type: FieldType, value: Any) -> Any:
        """Validate and convert field type."""
        if isinstance(field_type, str):
            field_type = FieldType(field_type)
        
        if field_type == FieldType.STRING:
            return str(value)
        elif field_type == FieldType.INTEGER:
            if isinstance(value, bool):
                raise ValueError("Boolean cannot be converted to integer")
            return int(value)
        elif field_type == FieldType.FLOAT:
            return float(value)
        elif field_type == FieldType.BOOLEAN:
            if isinstance(value, str):
                return value.lower() in ('true', '1', 'yes', 'on')
            return bool(value)
        elif field_type == FieldType.LIST:
            if not isinstance(value, list):
                raise ValueError("Value must be a list")
            return value
        elif field_type == FieldType.DICT:
            if not isinstance(value, dict):
                raise ValueError("Value must be a dictionary")
            return value
        elif field_type == FieldType.OBJECT:
            return value
        else:
            raise ValueError(f"Unknown field type: {field_type}")
    
    def _validate_constraint(self, constraint: FieldConstraint, value: Any, field_name: str) -> None:
        """Validate a constraint against a value."""
        constraint_type = constraint.constraint_type
        if isinstance(constraint_type, str):
            constraint_type = ConstraintType(constraint_type)
        
        if constraint_type == ConstraintType.MIN:
            if value < constraint.value:
                raise ValueError(f"Value {value} is less than minimum {constraint.value}")
        
        elif constraint_type == ConstraintType.MAX:
            if value > constraint.value:
                raise ValueError(f"Value {value} is greater than maximum {constraint.value}")
        
        elif constraint_type == ConstraintType.MIN_LENGTH:
            if len(value) < constraint.value:
                raise ValueError(f"Length {len(value)} is less than minimum {constraint.value}")
        
        elif constraint_type == ConstraintType.MAX_LENGTH:
            if len(value) > constraint.value:
                raise ValueError(f"Length {len(value)} is greater than maximum {constraint.value}")
        
        elif constraint_type == ConstraintType.PATTERN:
            if not re.match(constraint.value, str(value)):
                raise ValueError(f"Value does not match pattern: {constraint.value}")
        
        elif constraint_type == ConstraintType.ENUM:
            if value not in constraint.value:
                raise ValueError(f"Value {value} not in allowed values: {constraint.value}")
        
        elif constraint_type == ConstraintType.TYPE:
            expected_type = constraint.value
            if not isinstance(value, expected_type):
                raise ValueError(f"Expected type {expected_type}, got {type(value)}")
    
    def _compile_validators(self) -> None:
        """Compile custom validator functions."""
        if not self.schema:
            return
        
        for validator_def in self.schema.validators:
            try:
                # Create a safe execution environment
                exec_globals = {
                    '__builtins__': {
                        'ValueError': ValueError,
                        'TypeError': TypeError,
                        'len': len,
                        'str': str,
                        'int': int,
                        'float': float,
                        'bool': bool,
                        'list': list,
                        'dict': dict,
                        'isinstance': isinstance,
                        'hasattr': hasattr,
                        'getattr': getattr,
                    }
                }
                
                # Compile validator function
                func_code = f"""
def validator_func(values):
    {validator_def.code}
    return values
"""
                exec(func_code, exec_globals)
                self._compiled_validators[validator_def.name] = exec_globals['validator_func']
                
                logger.debug(f"Compiled validator: {validator_def.name}")
                
            except Exception as e:
                logger.error(f"Failed to compile validator {validator_def.name}: {e}")
    
    def generate_schema_template(self, component_type: str) -> Dict[str, Any]:
        """Generate a schema template for a component type."""
        template = {
            "name": f"{component_type.title()}Schema",
            "version": "1.0.0",
            "description": f"Schema for {component_type} configuration",
            "fields": [],
            "parameters": [],
            "validators": [],
            "required_fields": [],
            "optional_fields": []
        }
        
        # Add common fields based on component type
        if component_type == "agent":
            template["fields"].extend([
                {
                    "name": "model",
                    "field_type": "string",
                    "description": "Model name to use",
                    "required": True,
                    "constraints": [
                        {
                            "constraint_type": "pattern",
                            "value": r"^(gpt-|claude-|llama-)",
                            "message": "Model must be a supported LLM"
                        }
                    ]
                },
                {
                    "name": "temperature",
                    "field_type": "float",
                    "description": "Temperature for generation",
                    "required": False,
                    "default": 0.7,
                    "constraints": [
                        {"constraint_type": "min", "value": 0.0},
                        {"constraint_type": "max", "value": 1.0}
                    ]
                }
            ])
            template["required_fields"] = ["model"]
            template["optional_fields"] = ["temperature", "max_tokens", "system_prompt"]
        
        elif component_type == "step":
            template["fields"].extend([
                {
                    "name": "name",
                    "field_type": "string",
                    "description": "Step name",
                    "required": True
                },
                {
                    "name": "description",
                    "field_type": "string",
                    "description": "Step description",
                    "required": False
                }
            ])
            template["required_fields"] = ["name"]
            template["optional_fields"] = ["description", "debug_mode"]
        
        return template


def create_schema_from_yaml(yaml_path: Union[str, Path]) -> ConfigSchema:
    """Create a schema from YAML file."""
    yaml_path = Path(yaml_path)
    
    with open(yaml_path, 'r') as f:
        schema_data = yaml.safe_load(f)
    
    return ConfigSchema(**schema_data)


def validate_config_with_schema(config: Dict[str, Any], schema_path: Union[str, Path]) -> Dict[str, Any]:
    """Validate configuration with schema file."""
    validator = SchemaValidator()
    validator.load_schema_from_yaml(schema_path)
    return validator.validate_config(config) 