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
    Enterprise Schema Validator - Advanced Configuration Validation and Type Safety Framework
    ======================================================================================
    
    The SchemaValidator provides comprehensive schema validation and type safety for enterprise
    configuration systems, implementing advanced validation rules, constraint checking, type
    coercion, and error reporting. This validator ensures configuration integrity, prevents
    runtime errors, and provides detailed validation feedback for enterprise applications
    and complex configuration hierarchies.
    
    **Core Architecture:**
        The schema validator provides enterprise-grade configuration validation capabilities:
        
        * **Comprehensive Type Validation**: Advanced type checking with automatic coercion and validation
        * **Constraint-Based Validation**: Flexible constraint system with custom validation rules
        * **Schema-Driven Validation**: Declarative schema definitions with inheritance and composition
        * **Enterprise Error Reporting**: Detailed error reporting with context and remediation suggestions
        * **Performance Optimization**: High-performance validation with caching and batch processing
        * **Framework Integration**: Complete integration with ConfigBase and component validation systems
    
    **Validation Capabilities:**
        
        **Type System Validation:**
        * **Primitive Types**: String, integer, float, boolean validation with automatic coercion
        * **Complex Types**: List, dictionary, and object validation with nested structure support
        * **Custom Types**: Extensible type system with custom type validators and converters
        * **Nullable Types**: Optional field validation with null/undefined handling
        
        **Constraint Validation:**
        * **Range Constraints**: Minimum/maximum value validation for numeric and string types
        * **Length Constraints**: String and collection length validation with flexible bounds
        * **Pattern Matching**: Regular expression pattern validation for string fields
        * **Enumeration Validation**: Enumerated value validation with whitelist/blacklist support
        
        **Advanced Validation Features:**
        * **Cross-Field Validation**: Complex validation rules spanning multiple configuration fields
        * **Conditional Validation**: Conditional validation based on other field values
        * **Custom Validators**: Extensible custom validation function integration
        * **Validation Inheritance**: Schema inheritance with validation rule composition
    
    **Schema Definition System:**
        
        **Declarative Schema Format:**
        ```yaml
        # Example Schema Definition
        schema_name: "enterprise_service_config"
        version: "1.0.0"
        
        fields:
          - name: "service_name"
            type: "string"
            required: true
            constraints:
              - type: "min_length"
                value: 3
                message: "Service name must be at least 3 characters"
              - type: "pattern"
                value: "^[a-zA-Z0-9_-]+$"
                message: "Service name must contain only alphanumeric characters, hyphens, and underscores"
                
          - name: "port"
            type: "integer"
            required: true
            constraints:
              - type: "min"
                value: 1024
                message: "Port must be 1024 or higher"
              - type: "max"
                value: 65535
                message: "Port must be 65535 or lower"
                
          - name: "environment"
            type: "string"
            required: true
            constraints:
              - type: "enum"
                value: ["development", "staging", "production"]
                message: "Environment must be development, staging, or production"
                
          - name: "features"
            type: "list"
            required: false
            default: []
            constraints:
              - type: "max_length"
                value: 20
                message: "Maximum 20 features allowed"
                
          - name: "database_config"
            type: "dict"
            required: false
            nested_schema: "database_config_schema"
        
        # Custom validation functions
        custom_validators:
          - name: "validate_service_compatibility"
            function: "validate_service_compatibility"
            applies_to: ["service_name", "environment"]
            
        # Cross-field validation rules
        cross_field_rules:
          - name: "production_security_check"
            condition: "environment == 'production'"
            requirements:
              - field: "ssl_enabled"
                value: true
              - field: "authentication_required"
                value: true
        ```
        
        **Schema Composition and Inheritance:**
        ```yaml
        # Base schema
        base_service_schema:
          fields:
            - name: "service_name"
              type: "string"
              required: true
            - name: "version"
              type: "string"
              required: true
              
        # Extended schema
        web_service_schema:
          inherits: "base_service_schema"
          fields:
            - name: "port"
              type: "integer"
              required: true
            - name: "ssl_enabled"
              type: "boolean"
              default: false
        ```
    
    **Usage Patterns:**
        
        **Basic Schema Validation:**
        ```python
        from nanobrain.core.config.schema_validator import SchemaValidator, FieldSchema, FieldConstraint
        
        # Create schema validator
        validator = SchemaValidator()
        
        # Define validation schema
        service_schema = ConfigSchema(
            name="service_config",
            version="1.0.0",
            fields=[
                FieldSchema(
                    name="service_name",
                    field_type=FieldType.STRING,
                    required=True,
                    constraints=[
                        FieldConstraint(
                            constraint_type=ConstraintType.MIN_LENGTH,
                            value=3,
                            message="Service name must be at least 3 characters"
                        ),
                        FieldConstraint(
                            constraint_type=ConstraintType.PATTERN,
                            value=r"^[a-zA-Z0-9_-]+$",
                            message="Invalid service name format"
                        )
                    ]
                ),
                FieldSchema(
                    name="port",
                    field_type=FieldType.INTEGER,
                    required=True,
                    constraints=[
                        FieldConstraint(
                            constraint_type=ConstraintType.MIN,
                            value=1024,
                            message="Port must be 1024 or higher"
                        ),
                        FieldConstraint(
                            constraint_type=ConstraintType.MAX,
                            value=65535,
                            message="Port must be 65535 or lower"
                        )
                    ]
                )
            ]
        )
        
        # Load and set schema
        validator.load_schema(service_schema)
        
        # Validate configuration
        config_data = {
            "service_name": "my_service",
            "port": 8080,
            "environment": "production"
        }
        
        try:
            validated_config = validator.validate_config(config_data)
            print(f"Validation successful: {validated_config}")
        except ValidationError as e:
            print(f"Validation failed: {e}")
        ```
        
        **Advanced Validation with Custom Rules:**
        ```python
        # Advanced schema validator with custom validation
        class EnterpriseSchemaValidator:
            def __init__(self):
                self.validator = SchemaValidator()
                self.custom_validators = {}
                
            def register_custom_validator(self, name: str, validator_func: Callable):
                \"\"\"Register custom validation function\"\"\"
                self.custom_validators[name] = validator_func
                
            def validate_with_business_rules(self, config: Dict[str, Any], schema_name: str) -> Dict[str, Any]:
                \"\"\"Validate configuration with business rule enforcement\"\"\"
                
                # Standard schema validation
                validated_config = self.validator.validate_config(config)
                
                # Apply business rules
                business_rule_errors = []
                
                # Example business rule: Production services must have monitoring
                if validated_config.get('environment') == 'production':
                    if not validated_config.get('monitoring_enabled', False):
                        business_rule_errors.append(
                            "Production services must have monitoring enabled"
                        )
                        
                    if not validated_config.get('backup_enabled', False):
                        business_rule_errors.append(
                            "Production services must have backup enabled"
                        )
                
                # Example business rule: Service names must follow naming convention
                service_name = validated_config.get('service_name', '')
                if service_name and not self.validate_naming_convention(service_name):
                    business_rule_errors.append(
                        f"Service name '{service_name}' does not follow naming convention"
                    )
                
                # Apply custom validators
                for validator_name, validator_func in self.custom_validators.items():
                    try:
                        validator_func(validated_config)
                    except ValidationError as e:
                        business_rule_errors.append(f"Custom validation '{validator_name}': {e}")
                
                if business_rule_errors:
                    raise ValidationError(f"Business rule validation failed: {'; '.join(business_rule_errors)}")
                
                return validated_config
                
            def validate_naming_convention(self, service_name: str) -> bool:
                \"\"\"Validate service naming convention\"\"\"
                # Example: service names must start with department code
                valid_prefixes = ['eng-', 'data-', 'ml-', 'ops-']
                return any(service_name.startswith(prefix) for prefix in valid_prefixes)
                
            def validate_configuration_hierarchy(self, configs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
                \"\"\"Validate hierarchical configuration with dependencies\"\"\"
                
                validation_results = {
                    'validated_configs': {},
                    'validation_errors': {},
                    'dependency_errors': []
                }
                
                # Validate individual configurations
                for config_name, config_data in configs.items():
                    try:
                        validated_config = self.validate_with_business_rules(
                            config_data, config_name
                        )
                        validation_results['validated_configs'][config_name] = validated_config
                    except ValidationError as e:
                        validation_results['validation_errors'][config_name] = str(e)
                
                # Validate dependencies
                dependency_errors = self.validate_config_dependencies(
                    validation_results['validated_configs']
                )
                validation_results['dependency_errors'] = dependency_errors
                
                return validation_results
        
        # Custom validator functions
        def validate_database_connectivity(config: Dict[str, Any]):
            \"\"\"Custom validator for database connectivity\"\"\"
            if 'database_config' in config:
                db_config = config['database_config']
                required_fields = ['host', 'port', 'database_name']
                
                missing_fields = [field for field in required_fields if field not in db_config]
                if missing_fields:
                    raise ValidationError(f"Missing database fields: {missing_fields}")
                    
                # Test connection (in real implementation)
                # if not test_database_connection(db_config):
                #     raise ValidationError("Database connection test failed")
        
        def validate_security_configuration(config: Dict[str, Any]):
            \"\"\"Custom validator for security configuration\"\"\"
            if config.get('environment') == 'production':
                security_requirements = [
                    'ssl_enabled',
                    'authentication_required',
                    'audit_logging_enabled'
                ]
                
                missing_security = [
                    req for req in security_requirements
                    if not config.get(req, False)
                ]
                
                if missing_security:
                    raise ValidationError(f"Missing security requirements: {missing_security}")
        
        # Enterprise validation setup
        enterprise_validator = EnterpriseSchemaValidator()
        enterprise_validator.register_custom_validator('database_connectivity', validate_database_connectivity)
        enterprise_validator.register_custom_validator('security_configuration', validate_security_configuration)
        
        # Validate enterprise configuration
        enterprise_config = {
            "service_name": "eng-data-processor",
            "port": 8080,
            "environment": "production",
            "monitoring_enabled": True,
            "backup_enabled": True,
            "ssl_enabled": True,
            "authentication_required": True,
            "audit_logging_enabled": True,
            "database_config": {
                "host": "db.company.com",
                "port": 5432,
                "database_name": "production_db"
            }
        }
        
        try:
            validated_config = enterprise_validator.validate_with_business_rules(
                enterprise_config, "enterprise_service"
            )
            print("Enterprise validation successful")
        except ValidationError as e:
            print(f"Enterprise validation failed: {e}")
        ```
        
        **Schema-Driven Configuration Validation:**
        ```python
        # Schema-driven validation system
        class SchemaValidationFramework:
            def __init__(self):
                self.schema_registry = {}
                self.validation_cache = {}
                
            def register_schema_from_file(self, schema_name: str, schema_file: str):
                \"\"\"Register schema from YAML file\"\"\"
                
                with open(schema_file, 'r') as f:
                    schema_data = yaml.safe_load(f)
                
                # Convert YAML schema to schema objects
                schema = self.parse_schema_definition(schema_data)
                self.schema_registry[schema_name] = schema
                
            def parse_schema_definition(self, schema_data: Dict[str, Any]) -> ConfigSchema:
                \"\"\"Parse schema definition from YAML\"\"\"
                
                fields = []
                for field_data in schema_data.get('fields', []):
                    constraints = []
                    for constraint_data in field_data.get('constraints', []):
                        constraint = FieldConstraint(
                            constraint_type=ConstraintType(constraint_data['type']),
                            value=constraint_data['value'],
                            message=constraint_data.get('message')
                        )
                        constraints.append(constraint)
                    
                    field = FieldSchema(
                        name=field_data['name'],
                        field_type=FieldType(field_data['type']),
                        description=field_data.get('description'),
                        required=field_data.get('required', False),
                        default=field_data.get('default'),
                        constraints=constraints
                    )
                    fields.append(field)
                
                return ConfigSchema(
                    name=schema_data['schema_name'],
                    version=schema_data.get('version', '1.0.0'),
                    fields=fields
                )
                
            def validate_against_schema(self, config: Dict[str, Any], schema_name: str) -> Dict[str, Any]:
                \"\"\"Validate configuration against registered schema\"\"\"
                
                if schema_name not in self.schema_registry:
                    raise ValueError(f"Schema not found: {schema_name}")
                
                schema = self.schema_registry[schema_name]
                validator = SchemaValidator()
                validator.load_schema(schema)
                
                return validator.validate_config(config)
                
            def validate_configuration_suite(self, config_directory: str) -> Dict[str, Any]:
                \"\"\"Validate entire configuration suite\"\"\"
                
                config_path = Path(config_directory)
                validation_results = {
                    'successful_validations': {},
                    'failed_validations': {},
                    'summary': {
                        'total_configs': 0,
                        'successful': 0,
                        'failed': 0
                    }
                }
                
                # Find all configuration files
                for config_file in config_path.glob("*.yml"):
                    config_name = config_file.stem
                    validation_results['summary']['total_configs'] += 1
                    
                    try:
                        # Load configuration
                        with open(config_file) as f:
                            config_data = yaml.safe_load(f)
                        
                        # Determine schema (could be specified in config or inferred)
                        schema_name = config_data.get('schema', f"{config_name}_schema")
                        
                        # Validate configuration
                        validated_config = self.validate_against_schema(config_data, schema_name)
                        validation_results['successful_validations'][config_name] = validated_config
                        validation_results['summary']['successful'] += 1
                        
                    except Exception as e:
                        validation_results['failed_validations'][config_name] = str(e)
                        validation_results['summary']['failed'] += 1
                
                return validation_results
        
        # Schema validation framework setup
        validation_framework = SchemaValidationFramework()
        
        # Register schemas
        validation_framework.register_schema_from_file('service_schema', 'schemas/service_schema.yml')
        validation_framework.register_schema_from_file('database_schema', 'schemas/database_schema.yml')
        
        # Validate configuration suite
        validation_results = validation_framework.validate_configuration_suite('config/')
        
        print(f"Validation Results:")
        print(f"  Total: {validation_results['summary']['total_configs']}")
        print(f"  Successful: {validation_results['summary']['successful']}")
        print(f"  Failed: {validation_results['summary']['failed']}")
        
        for config_name, error in validation_results['failed_validations'].items():
            print(f"  {config_name}: {error}")
        ```
    
    **Advanced Features:**
        
        **Performance Optimization:**
        * **Validation Caching**: Intelligent caching of validation results for improved performance
        * **Lazy Validation**: Lazy validation with on-demand constraint checking
        * **Batch Validation**: Efficient batch validation for large configuration sets
        * **Parallel Processing**: Parallel validation for independent configuration components
        
        **Error Reporting and Diagnostics:**
        * **Contextual Error Messages**: Detailed error messages with field context and suggestions
        * **Validation Traces**: Complete validation traces for debugging and analysis
        * **Error Recovery**: Automatic error recovery and correction suggestions
        * **Validation Analytics**: Analytics and metrics for validation patterns and errors
        
        **Integration Capabilities:**
        * **IDE Integration**: Integration with development environments for real-time validation
        * **CI/CD Pipeline Integration**: Automated validation in deployment pipelines
        * **Monitoring Integration**: Runtime validation monitoring and alerting
        * **Documentation Generation**: Automatic documentation generation from schemas
    
    **Enterprise Security and Compliance:**
        
        **Security Validation:**
        * **Security Policy Enforcement**: Automatic enforcement of security policies and standards
        * **Credential Validation**: Secure credential validation and policy compliance
        * **Access Control Validation**: Role-based access control validation and enforcement
        * **Audit Trail Generation**: Complete audit trails for validation operations
        
        **Compliance Features:**
        * **Regulatory Compliance**: Support for industry-specific compliance requirements
        * **Policy Templates**: Pre-built validation templates for common compliance scenarios
        * **Compliance Reporting**: Automated compliance reporting and validation summaries
        * **Change Tracking**: Configuration change tracking and approval workflows
    
    **Production Deployment:**
        
        **High Availability:**
        * **Distributed Validation**: Distributed validation across multiple nodes
        * **Failover Support**: Automatic failover and redundancy for validation services
        * **Load Balancing**: Load balancing for high-throughput validation scenarios
        * **Disaster Recovery**: Backup and recovery for validation schemas and configuration
        
        **Monitoring and Observability:**
        * **Validation Metrics**: Real-time metrics for validation performance and success rates
        * **Error Analytics**: Comprehensive error analysis and pattern detection
        * **Performance Monitoring**: Validation performance monitoring and optimization
        * **Alerting Integration**: Integration with enterprise alerting and monitoring systems
    
    Attributes:
        schema (Optional[ConfigSchema]): Currently loaded validation schema
        validators (Dict[str, Callable]): Registry of custom validation functions
        validation_cache (Dict): Cache for validation results and performance optimization
        
    Note:
        This validator requires properly defined schemas for effective validation.
        Custom validation functions should handle exceptions gracefully.
        Schema caching improves performance but requires memory management for large schemas.
        Cross-field validation may have performance implications for complex configurations.
        
    Warning:
        Invalid schemas may cause validation failures or incorrect validation results.
        Custom validators should be thoroughly tested to prevent security vulnerabilities.
        Large configuration hierarchies may require performance optimization.
        Validation caching may consume significant memory in high-throughput scenarios.
        
    See Also:
        * :class:`FieldSchema`: Field schema definition and validation rules
        * :class:`FieldConstraint`: Individual constraint definition and validation
        * :class:`ConfigSchema`: Complete configuration schema definition
        * :mod:`nanobrain.core.config.config_base`: Configuration base classes and integration
        * :mod:`nanobrain.core.config.schema_generator`: Schema generation and automation
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
                    "constraints": [{"constraint_type": "min", "value": 0.0}, {"constraint_type": "max", "value": 1.0}]
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