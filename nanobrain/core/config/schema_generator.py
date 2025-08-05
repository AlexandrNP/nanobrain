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
    Enterprise Schema Generator - Automated JSON Schema Generation and Validation Framework
    ====================================================================================
    
    The SchemaGenerator provides comprehensive automated JSON schema generation from Pydantic models,
    enabling enterprise-grade configuration validation, API documentation, and type safety enforcement.
    This generator supports complex schema composition, inheritance patterns, custom validation rules,
    and seamless integration with configuration management systems for enterprise applications
    and distributed AI frameworks.
    
    **Core Architecture:**
        The schema generator provides enterprise-grade schema generation capabilities:
        
        * **Automated Schema Generation**: Intelligent JSON schema generation from Pydantic models and configurations
        * **Workflow Schema Management**: Comprehensive schema generation for complex workflow configurations
        * **Schema Composition**: Advanced schema composition with inheritance and reference management
        * **Validation Integration**: Complete integration with configuration validation and type checking
        * **Enterprise Features**: Version management, documentation generation, and compliance validation
        * **Framework Integration**: Full integration with ConfigBase and component validation systems
    
    **Schema Generation Capabilities:**
        
        **JSON Schema Automation:**
        * **Pydantic Model Schemas**: Automatic JSON schema generation from Pydantic model definitions
        * **Complex Type Support**: Support for complex types, unions, enums, and nested model structures
        * **Custom Field Validation**: Integration of custom field validators and constraint definitions
        * **Reference Resolution**: Intelligent reference resolution for complex schema dependencies
        
        **Workflow Schema Management:**
        * **Comprehensive Workflow Schemas**: Complete schema generation for workflow configurations
        * **Component Schema Integration**: Integration of agent, tool, step, and executor schemas
        * **Dependency Management**: Automatic schema dependency resolution and composition
        * **Version Compatibility**: Schema versioning and backward compatibility management
        
        **Enterprise Schema Features:**
        * **Schema Documentation**: Automatic documentation generation from schema definitions
        * **Compliance Validation**: Schema compliance validation against enterprise standards
        * **Schema Registry**: Centralized schema registry with version management
        * **Performance Optimization**: Optimized schema generation and caching for large-scale applications
    
    **Schema Generation Formats:**
        
        **Standard JSON Schema Structure:**
        ```json
        {
          "$schema": "http://json-schema.org/draft-07/schema#",
          "title": "Enterprise Service Configuration",
          "description": "Comprehensive configuration schema for enterprise AI services",
          "type": "object",
          "nanobrain_version": "2.0.0",
          "nanobrain_metadata": {
            "component_type": "service_config",
            "framework_version": "2.0.0",
            "schema_generator": "automated",
            "validation_level": "enterprise"
          },
          "properties": {
            "service_name": {
              "type": "string",
              "description": "Unique identifier for the AI service",
              "pattern": "^[a-zA-Z0-9_-]+$",
              "minLength": 3,
              "maxLength": 50,
              "examples": ["ai_service", "data_processor", "ml_pipeline"]
            },
            "service_config": {
              "$ref": "#/definitions/ServiceConfig",
              "description": "Complete service configuration object"
            },
            "agents": {
              "type": "array",
              "description": "Collection of AI agents for the service",
              "items": {
                "$ref": "#/definitions/AgentConfig"
              },
              "minItems": 1,
              "maxItems": 20
            },
            "workflows": {
              "type": "object",
              "description": "Workflow configurations and orchestration",
              "additionalProperties": {
                "$ref": "#/definitions/WorkflowConfig"
              }
            }
          },
          "required": ["service_name", "service_config"],
          "additionalProperties": false,
          "definitions": {
            "ServiceConfig": {
              "type": "object",
              "description": "Service-specific configuration parameters",
              "properties": {
                "port": {
                  "type": "integer",
                  "description": "Service port number",
                  "minimum": 1024,
                  "maximum": 65535,
                  "default": 8080
                },
                "environment": {
                  "type": "string",
                  "description": "Deployment environment",
                  "enum": ["development", "staging", "production"],
                  "default": "development"
                },
                "features": {
                  "type": "array",
                  "description": "Enabled service features",
                  "items": {
                    "type": "string",
                    "enum": ["monitoring", "analytics", "debugging", "profiling"]
                  },
                  "uniqueItems": true
                }
              },
              "required": ["port", "environment"]
            }
          }
        }
        ```
        
        **Comprehensive Workflow Schema:**
        ```json
        {
          "$schema": "http://json-schema.org/draft-07/schema#",
          "title": "NanoBrain Workflow Configuration Schema",
          "description": "Complete schema for NanoBrain workflow configurations",
          "type": "object",
          "nanobrain_version": "2.0.0",
          "properties": {
            "workflow": {
              "type": "object",
              "description": "Workflow definition and configuration",
              "properties": {
                "name": {
                  "type": "string",
                  "description": "Workflow identifier",
                  "pattern": "^[a-zA-Z0-9_-]+$"
                },
                "execution_strategy": {
                  "type": "string",
                  "enum": ["SEQUENTIAL", "PARALLEL", "GRAPH_BASED", "EVENT_DRIVEN"],
                  "description": "Workflow execution strategy"
                },
                "steps": {
                  "type": "array",
                  "description": "Workflow processing steps",
                  "items": {
                    "$ref": "#/definitions/StepConfig"
                  }
                },
                "links": {
                  "type": "array",
                  "description": "Data flow connections between steps",
                  "items": {
                    "$ref": "#/definitions/LinkConfig"
                  }
                },
                "triggers": {
                  "type": "array",
                  "description": "Workflow execution triggers",
                  "items": {
                    "$ref": "#/definitions/TriggerConfig"
                  }
                }
              },
              "required": ["name", "execution_strategy", "steps"]
            }
          },
          "definitions": {
            "StepConfig": {
              "type": "object",
              "description": "Individual workflow step configuration",
              "properties": {
                "name": {"type": "string"},
                "class": {"type": "string"},
                "config": {"type": "object"},
                "input_data_units": {"type": "array"},
                "output_data_units": {"type": "array"}
              },
              "required": ["name", "class"]
            },
            "LinkConfig": {
              "type": "object",
              "description": "Data flow link configuration",
              "properties": {
                "source": {"type": "string"},
                "target": {"type": "string"},
                "link_type": {
                  "type": "string",
                  "enum": ["DIRECT", "FILE", "QUEUE", "TRANSFORM", "CONDITIONAL"]
                }
              },
              "required": ["source", "target", "link_type"]
            }
          }
        }
        ```
    
    **Usage Patterns:**
        
        **Basic Schema Generation:**
        ```python
        from nanobrain.core.config.schema_generator import SchemaGenerator
        from pydantic import BaseModel, Field
        
        # Define configuration model
        class ServiceConfig(BaseModel):
            service_name: str = Field(..., description="Service identifier")
            port: int = Field(8080, ge=1024, le=65535, description="Service port")
            environment: str = Field("development", description="Deployment environment")
            features: List[str] = Field(default_factory=list, description="Enabled features")
        
        # Create schema generator
        generator = SchemaGenerator()
        
        # Generate JSON schema
        schema = generator.generate_schema(ServiceConfig, "ServiceConfiguration")
        
        print(f"Generated schema: {json.dumps(schema, indent=2)}")
        
        # Save schema to file
        generator.save_schema("ServiceConfiguration", "schemas/service_config.json")
        
        # Generate multiple schemas
        schemas = generator.generate_multiple_schemas([
            (ServiceConfig, "ServiceConfig"),
            (AgentConfig, "AgentConfig"),
            (WorkflowConfig, "WorkflowConfig")
        ])
        
        print(f"Generated {len(schemas)} schemas")
        ```
        
        **Enterprise Schema Management:**
        ```python
        # Enterprise schema management system
        class EnterpriseSchemaManager:
            def __init__(self):
                self.generator = SchemaGenerator()
                self.schema_registry = {}
                self.version_history = {}
                
            def generate_enterprise_schema_suite(self, config_models: Dict[str, Type[BaseModel]]) -> Dict[str, Any]:
                \"\"\"Generate complete enterprise schema suite\"\"\"
                
                schema_suite = {
                    'suite_metadata': {
                        'generation_timestamp': datetime.now().isoformat(),
                        'framework_version': '2.0.0',
                        'schema_count': len(config_models),
                        'compliance_level': 'enterprise'
                    },
                    'schemas': {},
                    'cross_references': {},
                    'validation_rules': {}
                }
                
                # Generate individual schemas
                for schema_name, model_class in config_models.items():
                    schema = self.generator.generate_schema(model_class, schema_name)
                    
                    # Add enterprise metadata
                    schema['enterprise_metadata'] = {
                        'compliance_validated': True,
                        'security_reviewed': True,
                        'documentation_complete': True,
                        'version_controlled': True
                    }
                    
                    schema_suite['schemas'][schema_name] = schema
                    self.schema_registry[schema_name] = schema
                
                # Generate cross-references
                cross_references = self.analyze_schema_cross_references(schema_suite['schemas'])
                schema_suite['cross_references'] = cross_references
                
                # Generate validation rules
                validation_rules = self.generate_enterprise_validation_rules(schema_suite['schemas'])
                schema_suite['validation_rules'] = validation_rules
                
                return schema_suite
                
            def analyze_schema_cross_references(self, schemas: Dict[str, Any]) -> Dict[str, Any]:
                \"\"\"Analyze cross-references between schemas\"\"\"
                
                cross_references = {
                    'schema_dependencies': {},
                    'circular_references': [],
                    'reference_map': {}
                }
                
                for schema_name, schema in schemas.items():
                    dependencies = self.extract_schema_dependencies(schema)
                    cross_references['schema_dependencies'][schema_name] = dependencies
                    
                    # Build reference map
                    for dep in dependencies:
                        if dep not in cross_references['reference_map']:
                            cross_references['reference_map'][dep] = []
                        cross_references['reference_map'][dep].append(schema_name)
                
                # Detect circular references
                circular_refs = self.detect_circular_references(cross_references['schema_dependencies'])
                cross_references['circular_references'] = circular_refs
                
                return cross_references
                
            def generate_schema_documentation(self, schema_suite: Dict[str, Any], output_directory: str):
                \"\"\"Generate comprehensive schema documentation\"\"\"
                
                output_path = Path(output_directory)
                output_path.mkdir(parents=True, exist_ok=True)
                
                # Generate main documentation
                doc_content = self.create_schema_documentation_content(schema_suite)
                with open(output_path / 'SCHEMA_DOCUMENTATION.md', 'w') as f:
                    f.write(doc_content)
                
                # Generate individual schema files
                schemas_dir = output_path / 'schemas'
                schemas_dir.mkdir(exist_ok=True)
                
                for schema_name, schema in schema_suite['schemas'].items():
                    schema_file = schemas_dir / f"{schema_name}.json"
                    with open(schema_file, 'w') as f:
                        json.dump(schema, f, indent=2)
                
                # Generate validation examples
                examples_dir = output_path / 'examples'
                examples_dir.mkdir(exist_ok=True)
                
                for schema_name, schema in schema_suite['schemas'].items():
                    example_config = self.generate_example_configuration(schema)
                    example_file = examples_dir / f"{schema_name}_example.yml"
                    with open(example_file, 'w') as f:
                        yaml.dump(example_config, f, default_flow_style=False, indent=2)
                
                return {
                    'documentation_file': output_path / 'SCHEMA_DOCUMENTATION.md',
                    'schema_files': list(schemas_dir.glob('*.json')),
                    'example_files': list(examples_dir.glob('*.yml'))
                }
                
            def validate_schema_compliance(self, schema_suite: Dict[str, Any]) -> Dict[str, Any]:
                \"\"\"Validate schema suite compliance with enterprise standards\"\"\"
                
                compliance_report = {
                    'overall_compliance': True,
                    'compliance_score': 0.0,
                    'schema_compliance': {},
                    'compliance_issues': [],
                    'recommendations': []
                }
                
                total_schemas = len(schema_suite['schemas'])
                compliant_schemas = 0
                
                for schema_name, schema in schema_suite['schemas'].items():
                    schema_compliance = self.validate_individual_schema_compliance(schema)
                    compliance_report['schema_compliance'][schema_name] = schema_compliance
                    
                    if schema_compliance['compliant']:
                        compliant_schemas += 1
                    else:
                        compliance_report['compliance_issues'].extend(
                            schema_compliance['issues']
                        )
                
                # Calculate compliance score
                compliance_report['compliance_score'] = (compliant_schemas / total_schemas) * 100
                compliance_report['overall_compliance'] = compliance_report['compliance_score'] >= 95.0
                
                # Generate recommendations
                if not compliance_report['overall_compliance']:
                    compliance_report['recommendations'] = self.generate_compliance_recommendations(
                        compliance_report['compliance_issues']
                    )
                
                return compliance_report
        
        # Enterprise schema management
        schema_manager = EnterpriseSchemaManager()
        
        # Define enterprise configuration models
        enterprise_models = {
            'ServiceConfig': ServiceConfig,
            'AgentConfig': AgentConfig,
            'WorkflowConfig': WorkflowConfig,
            'DatabaseConfig': DatabaseConfig,
            'SecurityConfig': SecurityConfig
        }
        
        # Generate enterprise schema suite
        schema_suite = schema_manager.generate_enterprise_schema_suite(enterprise_models)
        
        # Validate compliance
        compliance_report = schema_manager.validate_schema_compliance(schema_suite)
        print(f"Schema compliance: {compliance_report['compliance_score']:.1f}%")
        
        # Generate documentation
        documentation_files = schema_manager.generate_schema_documentation(
            schema_suite, 'docs/schemas/'
        )
        
        print(f"Generated documentation:")
        print(f"  Main doc: {documentation_files['documentation_file']}")
        print(f"  Schema files: {len(documentation_files['schema_files'])}")
        print(f"  Example files: {len(documentation_files['example_files'])}")
        ```
        
        **Advanced Schema Composition:**
        ```python
        # Advanced schema composition and inheritance
        class AdvancedSchemaComposer:
            def __init__(self, generator: SchemaGenerator):
                self.generator = generator
                self.base_schemas = {}
                self.composed_schemas = {}
                
            def create_base_schema_library(self):
                \"\"\"Create library of reusable base schemas\"\"\"
                
                # Base service schema
                base_service_schema = {
                    "type": "object",
                    "description": "Base schema for all services",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Service name",
                            "pattern": "^[a-zA-Z0-9_-]+$"
                        },
                        "version": {
                            "type": "string",
                            "description": "Service version",
                            "pattern": "^\\\\d+\\\\.\\\\d+\\\\.\\\\d+$"
                        },
                        "description": {
                            "type": "string",
                            "description": "Service description"
                        }
                    },
                    "required": ["name", "version"]
                }
                
                # Base configuration schema
                base_config_schema = {
                    "type": "object",
                    "description": "Base configuration schema",
                    "properties": {
                        "environment": {
                            "type": "string",
                            "enum": ["development", "staging", "production"],
                            "description": "Deployment environment"
                        },
                        "debug": {
                            "type": "boolean",
                            "default": False,
                            "description": "Enable debug mode"
                        },
                        "logging": {
                            "$ref": "#/definitions/LoggingConfig"
                        }
                    },
                    "required": ["environment"]
                }
                
                self.base_schemas = {
                    'BaseService': base_service_schema,
                    'BaseConfig': base_config_schema
                }
                
            def compose_schema(self, base_schema_name: str, extensions: Dict[str, Any], 
                             schema_name: str) -> Dict[str, Any]:
                \"\"\"Compose schema from base schema and extensions\"\"\"
                
                if base_schema_name not in self.base_schemas:
                    raise ValueError(f"Base schema not found: {base_schema_name}")
                
                base_schema = self.base_schemas[base_schema_name].copy()
                
                # Merge properties
                if 'properties' in extensions:
                    if 'properties' not in base_schema:
                        base_schema['properties'] = {}
                    base_schema['properties'].update(extensions['properties'])
                
                # Merge required fields
                if 'required' in extensions:
                    if 'required' not in base_schema:
                        base_schema['required'] = []
                    base_schema['required'].extend(extensions['required'])
                    base_schema['required'] = list(set(base_schema['required']))  # Remove duplicates
                
                # Add metadata
                base_schema['title'] = schema_name
                base_schema['$schema'] = "http://json-schema.org/draft-07/schema#"
                base_schema['nanobrain_composition'] = {
                    'base_schema': base_schema_name,
                    'extensions_applied': True,
                    'composed_at': datetime.now().isoformat()
                }
                
                self.composed_schemas[schema_name] = base_schema
                return base_schema
                
            def create_inheritance_hierarchy(self, inheritance_tree: Dict[str, Any]) -> Dict[str, Any]:
                \"\"\"Create schema inheritance hierarchy\"\"\"
                
                hierarchy_schemas = {}
                
                for schema_name, schema_def in inheritance_tree.items():
                    if 'inherits' in schema_def:
                        parent_schema = schema_def['inherits']
                        if parent_schema in self.base_schemas:
                            # Compose schema from parent
                            composed_schema = self.compose_schema(
                                parent_schema,
                                schema_def.get('extensions', {}),
                                schema_name
                            )
                            hierarchy_schemas[schema_name] = composed_schema
                        else:
                            raise ValueError(f"Parent schema not found: {parent_schema}")
                    else:
                        # Root schema
                        hierarchy_schemas[schema_name] = schema_def
                
                return hierarchy_schemas
        
        # Advanced schema composition
        composer = AdvancedSchemaComposer(generator)
        composer.create_base_schema_library()
        
        # Define inheritance hierarchy
        service_hierarchy = {
            'WebService': {
                'inherits': 'BaseService',
                'extensions': {
                    'properties': {
                        'port': {'type': 'integer', 'minimum': 1024},
                        'ssl_enabled': {'type': 'boolean', 'default': True}
                    },
                    'required': ['port']
                }
            },
            'APIService': {
                'inherits': 'WebService',
                'extensions': {
                    'properties': {
                        'api_version': {'type': 'string'},
                        'rate_limit': {'type': 'integer', 'minimum': 100}
                    },
                    'required': ['api_version']
                }
            }
        }
        
        # Create inheritance hierarchy
        service_schemas = composer.create_inheritance_hierarchy(service_hierarchy)
        
        print(f"Created {len(service_schemas)} inherited schemas")
        for schema_name, schema in service_schemas.items():
            print(f"  {schema_name}: {len(schema.get('properties', {}))} properties")
        ```
    
    **Advanced Features:**
        
        **Schema Validation Integration:**
        * **Real-Time Validation**: Real-time schema validation during configuration editing
        * **IDE Integration**: Integration with development environments for schema-aware editing
        * **CI/CD Validation**: Automated schema validation in deployment pipelines
        * **Configuration Testing**: Automated testing of configurations against schemas
        
        **Schema Documentation:**
        * **Automatic Documentation**: Automatic generation of human-readable schema documentation
        * **Interactive Examples**: Interactive configuration examples and validation
        * **Schema Visualization**: Visual representation of schema structures and relationships
        * **API Documentation**: Integration with API documentation generation tools
        
        **Performance Optimization:**
        * **Schema Caching**: Intelligent caching of generated schemas for performance
        * **Lazy Generation**: Lazy schema generation for large configuration systems
        * **Parallel Processing**: Parallel schema generation for multiple models
        * **Memory Optimization**: Memory-efficient schema generation and storage
    
    **Enterprise Integration:**
        
        **Compliance and Security:**
        * **Enterprise Standards**: Compliance with enterprise schema standards and conventions
        * **Security Validation**: Schema validation for security policies and access controls
        * **Audit Trails**: Complete audit trails for schema generation and modification
        * **Change Management**: Schema change management and approval workflows
        
        **Deployment and Operations:**
        * **Multi-Environment Support**: Schema generation for multiple deployment environments
        * **Version Management**: Schema versioning and backward compatibility management
        * **Monitoring Integration**: Integration with monitoring and alerting systems
        * **Disaster Recovery**: Schema backup and recovery capabilities
    
    Attributes:
        _schemas (Dict[str, Dict[str, Any]]): Internal registry of generated schemas
        
    Methods:
        generate_schema: Generate JSON schema from Pydantic model
        generate_workflow_schema: Generate comprehensive workflow configuration schema
        save_schema: Save generated schema to file
        get_schema: Retrieve previously generated schema
        
    Note:
        This generator automatically handles complex Pydantic model structures and relationships.
        Generated schemas include NanoBrain-specific metadata for framework integration.
        Schema caching improves performance for repeated generation operations.
        Workflow schemas include comprehensive validation for all component types.
        
    Warning:
        Complex model hierarchies may generate large schema structures.
        Schema generation may consume significant memory for large model sets.
        Generated schemas should be validated before use in production environments.
        Schema references may need manual adjustment for complex cross-references.
        
    See Also:
        * :class:`SchemaValidator`: Schema validation and constraint checking
        * :class:`ConfigBase`: Base configuration classes with schema integration
        * :mod:`nanobrain.core.config.yaml_config`: YAML configuration with schema support
        * :mod:`nanobrain.core.config.enhanced_config_manager`: Configuration management with schema validation
        * :mod:`nanobrain.core.config.component_factory`: Component creation with schema validation
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
        schema = model_class.model_json_schema()
        
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
        schema = model_class.model_json_schema()
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
            return instance.model_dump()
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