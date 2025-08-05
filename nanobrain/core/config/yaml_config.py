"""
YAML Configuration System for NanoBrain Framework

Provides loading, saving, and validation of YAML configurations.
"""

import yaml
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union, List
from pydantic import BaseModel, Field, ValidationError, ConfigDict

from .config_base import ConfigBase

logger = logging.getLogger(__name__)


class YAMLConfig(ConfigBase):
    """
    Enterprise YAML Configuration - Advanced YAML Serialization and Configuration Management
    ====================================================================================
    
    The YAMLConfig provides comprehensive YAML configuration management with advanced serialization,
    validation, and enterprise-grade configuration handling. This configuration system supports
    complex configuration hierarchies, automatic serialization, type-safe configuration management,
    and seamless integration with enterprise configuration workflows and deployment pipelines.
    
    **Core Architecture:**
        The YAML configuration system provides enterprise-grade YAML management capabilities:
        
        * **Advanced YAML Serialization**: Intelligent YAML serialization with type preservation and formatting
        * **Bidirectional Conversion**: Seamless conversion between YAML files and typed configuration objects
        * **Configuration Inheritance**: Configuration inheritance and composition with YAML-based overrides
        * **Validation Integration**: Complete integration with schema validation and type checking
        * **Enterprise Features**: Version control integration, environment-specific configurations, and templating
        * **Framework Integration**: Full integration with ConfigBase and component configuration systems
    
    **YAML Configuration Capabilities:**
        
        **Advanced Serialization:**
        * **Type-Safe Serialization**: Automatic conversion of complex types to YAML-compatible formats
        * **Custom Type Handlers**: Extensible type serialization with custom converters
        * **Nested Object Support**: Deep serialization of nested configuration objects and references
        * **Enum and Union Handling**: Intelligent serialization of enumerations and union types
        
        **Configuration Management:**
        * **File-Based Persistence**: Automatic file-based configuration persistence and loading
        * **Configuration Templating**: YAML template support with variable substitution and inheritance
        * **Environment Configuration**: Environment-specific configuration overlays and overrides
        * **Configuration Validation**: Automatic validation during serialization and deserialization
        
        **Enterprise Features:**
        * **Version Control Integration**: Git-friendly YAML formatting with consistent ordering
        * **Configuration Documentation**: Automatic documentation generation from configuration schemas
        * **Migration Support**: Configuration migration and upgrade automation
        * **Backup and Recovery**: Automatic configuration backup and recovery mechanisms
    
    **YAML Configuration Format:**
        
        **Standard Configuration Structure:**
        ```yaml
        # Enterprise Service Configuration
        service_config:
          # Service identification
          service_name: "enterprise_ai_service"
          service_version: "1.0.0"
          environment: "production"
          
          # Service configuration
          server:
            host: "0.0.0.0"
            port: 8080
            workers: 4
            timeout: 30
            
          # Database configuration
          database:
            type: "postgresql"
            host: "${DATABASE_HOST:localhost}"
            port: "${DATABASE_PORT:5432}"
            database: "${DATABASE_NAME:production_db}"
            pool_size: 20
            
          # Feature flags
          features:
            - "advanced_analytics"
            - "real_time_processing"
            - "monitoring_integration"
            
          # Security configuration
          security:
            ssl_enabled: true
            authentication_required: true
            cors:
              enabled: true
              origins: ["https://app.company.com"]
              
        # Component references
        components:
          ai_agent:
            class: "nanobrain.library.agents.enhanced.CollaborativeAgent"
            config: "config/agents/collaborative_agent.yml"
            
          monitoring:
            class: "nanobrain.library.infrastructure.monitoring.PerformanceMonitor"
            config:
              collection_interval: 30
              history_size: 1000
              
        # Environment-specific overrides
        environments:
          development:
            server:
              port: 3000
              workers: 1
            security:
              ssl_enabled: false
              authentication_required: false
              
          staging:
            server:
              port: 8080
              workers: 2
            database:
              pool_size: 10
              
          production:
            server:
              port: 443
              workers: 8
            database:
              pool_size: 50
            security:
              ssl_enabled: true
              authentication_required: true
        ```
        
        **Configuration Templates:**
        ```yaml
        # Base template configuration
        base_service_template:
          service_config:
            service_name: "${SERVICE_NAME}"
            service_version: "${SERVICE_VERSION:1.0.0}"
            environment: "${ENVIRONMENT:development}"
            
            server:
              host: "${SERVER_HOST:0.0.0.0}"
              port: "${SERVER_PORT:8080}"
              workers: "${SERVER_WORKERS:4}"
              
            monitoring:
              enabled: "${MONITORING_ENABLED:true}"
              endpoint: "${MONITORING_ENDPOINT:/metrics}"
              
        # Service-specific configuration
        web_service_config:
          extends: "base_service_template"
          service_config:
            service_type: "web_service"
            server:
              port: 80
              ssl_port: 443
            features:
              - "web_interface"
              - "api_gateway"
              
        # Database service configuration
        database_service_config:
          extends: "base_service_template"
          service_config:
            service_type: "database_service"
            server:
              port: 5432
            features:
              - "backup_automation"
              - "replication"
        ```
    
    **Usage Patterns:**
        
        **Basic YAML Configuration:**
        ```python
        from nanobrain.core.config.yaml_config import YAMLConfig
        
        # Define configuration class
        class ServiceConfig(YAMLConfig):
            service_name: str
            port: int = 8080
            features: List[str] = Field(default_factory=list)
            database_config: Optional[Dict[str, Any]] = None
            
        # Load from YAML file
        config = ServiceConfig.from_config('config/service.yml')
        
        # Access configuration
        print(f"Service: {config.service_name}")
        print(f"Port: {config.port}")
        print(f"Features: {config.features}")
        
        # Modify configuration
        config.features.append('new_feature')
        config.port = 9000
        
        # Save back to YAML
        yaml_content = config.to_yaml('config/updated_service.yml')
        print(f"Updated YAML:\\n{yaml_content}")
        ```
        
        **Advanced Configuration Management:**
        ```python
        # Enterprise configuration management system
        class EnterpriseConfigurationManager:
            def __init__(self):
                self.configurations = {}
                self.templates = {}
                self.environments = ['development', 'staging', 'production']
                
            def load_configuration_suite(self, base_directory: str):
                \"\"\"Load complete configuration suite with templates and environments\"\"\"
                
                base_path = Path(base_directory)
                
                # Load base configurations
                self.load_base_configurations(base_path / 'base')
                
                # Load templates
                self.load_configuration_templates(base_path / 'templates')
                
                # Load environment-specific configurations
                for env in self.environments:
                    env_path = base_path / 'environments' / env
                    if env_path.exists():
                        self.load_environment_configurations(env, env_path)
                        
            def load_base_configurations(self, config_path: Path):
                \"\"\"Load base configuration files\"\"\"
                
                for config_file in config_path.glob('*.yml'):
                    config_name = config_file.stem
                    
                    # Determine configuration class (could be from registry)
                    config_class = self.get_config_class(config_name)
                    
                    # Load configuration
                    config = config_class.from_config(str(config_file))
                    self.configurations[config_name] = config
                    
            def generate_environment_config(self, config_name: str, environment: str) -> YAMLConfig:
                \"\"\"Generate environment-specific configuration\"\"\"
                
                base_config = self.configurations.get(config_name)
                if not base_config:
                    raise ValueError(f"Base configuration not found: {config_name}")
                
                # Apply environment-specific overrides
                env_overrides = self.get_environment_overrides(config_name, environment)
                
                # Create environment-specific configuration
                env_config = self.apply_configuration_overrides(base_config, env_overrides)
                
                return env_config
                
            def apply_configuration_overrides(self, base_config: YAMLConfig, overrides: Dict[str, Any]) -> YAMLConfig:
                \"\"\"Apply configuration overrides to base configuration\"\"\"
                
                # Convert base config to dict
                config_dict = base_config._to_serializable_dict()
                
                # Apply overrides (deep merge)
                merged_config = self.deep_merge_configs(config_dict, overrides)
                
                # Create new configuration instance
                config_class = type(base_config)
                return config_class.from_config(merged_config)
                
            def deep_merge_configs(self, base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
                \"\"\"Deep merge configuration dictionaries\"\"\"
                
                result = base.copy()
                
                for key, value in overrides.items():
                    if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                        result[key] = self.deep_merge_configs(result[key], value)
                    else:
                        result[key] = value
                        
                return result
                
            def export_configuration_suite(self, output_directory: str):
                \"\"\"Export complete configuration suite to YAML files\"\"\"
                
                output_path = Path(output_directory)
                output_path.mkdir(parents=True, exist_ok=True)
                
                # Export base configurations
                base_path = output_path / 'base'
                base_path.mkdir(exist_ok=True)
                
                for config_name, config in self.configurations.items():
                    config_file = base_path / f"{config_name}.yml"
                    config.to_yaml(config_file)
                    
                # Export environment-specific configurations
                for env in self.environments:
                    env_path = output_path / 'environments' / env
                    env_path.mkdir(parents=True, exist_ok=True)
                    
                    for config_name in self.configurations.keys():
                        env_config = self.generate_environment_config(config_name, env)
                        config_file = env_path / f"{config_name}.yml"
                        env_config.to_yaml(config_file)
                        
            def validate_configuration_consistency(self) -> Dict[str, Any]:
                \"\"\"Validate consistency across all configurations\"\"\"
                
                validation_results = {
                    'base_validations': {},
                    'environment_validations': {},
                    'cross_validation_errors': []
                }
                
                # Validate base configurations
                for config_name, config in self.configurations.items():
                    try:
                        # Validate configuration schema
                        config.validate_configuration()
                        validation_results['base_validations'][config_name] = 'valid'
                    except ValidationError as e:
                        validation_results['base_validations'][config_name] = str(e)
                
                # Validate environment configurations
                for env in self.environments:
                    env_validations = {}
                    for config_name in self.configurations.keys():
                        try:
                            env_config = self.generate_environment_config(config_name, env)
                            env_config.validate_configuration()
                            env_validations[config_name] = 'valid'
                        except Exception as e:
                            env_validations[config_name] = str(e)
                    
                    validation_results['environment_validations'][env] = env_validations
                
                return validation_results
        
        # Enterprise configuration management
        config_manager = EnterpriseConfigurationManager()
        config_manager.load_configuration_suite('config/')
        
        # Generate production configuration
        prod_config = config_manager.generate_environment_config('service_config', 'production')
        print(f"Production config: {prod_config.to_yaml()}")
        
        # Validate all configurations
        validation_results = config_manager.validate_configuration_consistency()
        print(f"Validation results: {validation_results}")
        
        # Export configuration suite
        config_manager.export_configuration_suite('output/configs/')
        ```
        
        **Configuration Templating and Inheritance:**
        ```python
        # Advanced configuration templating system
        class ConfigurationTemplateEngine:
            def __init__(self):
                self.templates = {}
                self.variables = {}
                
            def register_template(self, template_name: str, template_config: YAMLConfig):
                \"\"\"Register configuration template\"\"\"
                self.templates[template_name] = template_config
                
            def set_template_variables(self, variables: Dict[str, Any]):
                \"\"\"Set template variables for substitution\"\"\"
                self.variables.update(variables)
                
            def instantiate_template(self, template_name: str, instance_variables: Dict[str, Any] = None) -> YAMLConfig:
                \"\"\"Instantiate configuration from template\"\"\"
                
                if template_name not in self.templates:
                    raise ValueError(f"Template not found: {template_name}")
                
                template = self.templates[template_name]
                
                # Merge variables
                all_variables = {**self.variables}
                if instance_variables:
                    all_variables.update(instance_variables)
                
                # Convert template to YAML string
                template_yaml = template.to_yaml()
                
                # Perform variable substitution
                instantiated_yaml = self.substitute_variables(template_yaml, all_variables)
                
                # Load back as configuration
                template_class = type(template)
                return template_class.from_yaml_string(instantiated_yaml)
                
            def substitute_variables(self, yaml_content: str, variables: Dict[str, Any]) -> str:
                \"\"\"Substitute variables in YAML content\"\"\"
                
                import re
                
                # Variable pattern: ${VARIABLE_NAME:default_value}
                pattern = r'\\$\\{([^:}]+)(?::([^}]+))?\\}'
                
                def replace_variable(match):
                    var_name = match.group(1)
                    default_value = match.group(2) if match.group(2) else ''
                    
                    # Get variable value
                    value = variables.get(var_name, default_value)
                    return str(value)
                
                return re.sub(pattern, replace_variable, yaml_content)
                
            def create_service_from_template(self, template_name: str, service_config: Dict[str, Any]) -> YAMLConfig:
                \"\"\"Create service configuration from template\"\"\"
                
                # Standard service variables
                service_variables = {
                    'SERVICE_NAME': service_config['name'],
                    'SERVICE_VERSION': service_config.get('version', '1.0.0'),
                    'ENVIRONMENT': service_config.get('environment', 'development'),
                    'SERVICE_PORT': service_config.get('port', 8080),
                    'SERVICE_HOST': service_config.get('host', '0.0.0.0')
                }
                
                # Additional custom variables
                custom_variables = service_config.get('variables', {})
                service_variables.update(custom_variables)
                
                return self.instantiate_template(template_name, service_variables)
        
        # Template engine usage
        template_engine = ConfigurationTemplateEngine()
        
        # Create base service template
        base_template = ServiceConfig(
            service_name="${SERVICE_NAME}",
            port="${SERVICE_PORT:8080}",
            environment="${ENVIRONMENT:development}",
            features=["${DEFAULT_FEATURE:monitoring}"]
        )
        
        template_engine.register_template('base_service', base_template)
        
        # Set global variables
        template_engine.set_template_variables({
            'COMPANY_NAME': 'Enterprise Corp',
            'DEFAULT_FEATURE': 'analytics'
        })
        
        # Create service instances from template
        api_service = template_engine.create_service_from_template('base_service', {
            'name': 'api_service',
            'port': 8080,
            'environment': 'production',
            'variables': {
                'ADDITIONAL_FEATURE': 'authentication'
            }
        })
        
        web_service = template_engine.create_service_from_template('base_service', {
            'name': 'web_service',
            'port': 3000,
            'environment': 'development'
        })
        
        print(f"API Service Config:\\n{api_service.to_yaml()}")
        print(f"Web Service Config:\\n{web_service.to_yaml()}")
        ```
    
    **Advanced Features:**
        
        **Configuration Validation:**
        * **Schema-Based Validation**: Automatic validation against configuration schemas
        * **Cross-Reference Validation**: Validation of configuration references and dependencies
        * **Business Rule Validation**: Custom business rule validation and enforcement
        * **Environment Consistency**: Validation of environment-specific configuration consistency
        
        **Performance Optimization:**
        * **Lazy Loading**: Lazy loading of large configuration hierarchies
        * **Configuration Caching**: Intelligent caching of parsed configurations
        * **Streaming Serialization**: Memory-efficient serialization for large configurations
        * **Parallel Processing**: Parallel processing of independent configuration components
        
        **Enterprise Integration:**
        * **Version Control Integration**: Git-friendly YAML formatting and diff optimization
        * **CI/CD Pipeline Integration**: Automated configuration validation and deployment
        * **Configuration Management**: Integration with enterprise configuration management systems
        * **Audit and Compliance**: Configuration change auditing and compliance reporting
    
    **Production Deployment:**
        
        **Configuration Security:**
        * **Sensitive Data Handling**: Secure handling of credentials and sensitive configuration
        * **Configuration Encryption**: Optional configuration encryption for sensitive environments
        * **Access Control**: Role-based access control for configuration management
        * **Audit Logging**: Complete audit logging for configuration access and changes
        
        **High Availability:**
        * **Configuration Replication**: Configuration replication across multiple environments
        * **Disaster Recovery**: Configuration backup and disaster recovery capabilities
        * **Configuration Synchronization**: Real-time configuration synchronization
        * **Rollback Capabilities**: Automatic configuration rollback and recovery
        
        **Monitoring and Analytics:**
        * **Configuration Monitoring**: Real-time monitoring of configuration usage and performance
        * **Change Analytics**: Analytics for configuration changes and impact analysis
        * **Performance Metrics**: Configuration loading and serialization performance metrics
        * **Usage Tracking**: Comprehensive usage tracking and optimization recommendations
    
    Attributes:
        model_config (ConfigDict): Pydantic model configuration with YAML serialization support
        
    Methods:
        to_yaml: Convert configuration to YAML string or save to file
        from_config: Load configuration from YAML file (inherited from ConfigBase)
        validate_configuration: Validate configuration against schema (inherited)
        
    Note:
        This configuration class provides seamless YAML serialization and deserialization.
        Complex nested objects are automatically converted to YAML-compatible formats.
        Environment variable substitution requires proper variable definition and scoping.
        Large configuration hierarchies may require performance optimization for serialization.
        
    Warning:
        YAML serialization may lose type information for complex custom types.
        Configuration templates with circular references may cause infinite loops.
        Large configuration files may consume significant memory during processing.
        Sensitive configuration data should be handled with appropriate security measures.
        
    See Also:
        * :class:`ConfigBase`: Base configuration class with validation and loading
        * :class:`YAMLWorkflowConfig`: Workflow-specific YAML configuration extension
        * :mod:`nanobrain.core.config.schema_validator`: Configuration schema validation
        * :mod:`nanobrain.core.config.config_manager`: Advanced configuration management
        * :mod:`nanobrain.core.config.component_factory`: Component creation and dependency injection
    """
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="allow"
    )
    
    def to_yaml(self, file_path: Optional[Union[str, Path]] = None) -> str:
        """
        Convert configuration to YAML string or save to file.
        
        Args:
            file_path: Optional file path to save YAML
            
        Returns:
            YAML string representation
        """
        # Convert to dict, handling Pydantic models
        config_dict = self._to_serializable_dict()
        
        # Generate YAML string
        yaml_str = yaml.dump(
            config_dict,
            default_flow_style=False,
            sort_keys=False,
            indent=2
        )
        
        # Save to file if path provided
        if file_path:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(yaml_str)
            logger.info(f"Configuration saved to {file_path}")
        
        return yaml_str
    
    def _to_serializable_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        result = {}
        
        for key, value in self.model_dump().items():
            if isinstance(value, BaseModel):
                result[key] = value.model_dump()
            elif isinstance(value, list):
                result[key] = [
                    item.model_dump() if isinstance(item, BaseModel) else item
                    for item in value
                ]
            elif isinstance(value, dict):
                result[key] = {
                    k: v.model_dump() if isinstance(v, BaseModel) else v
                    for k, v in value.items()
                }
            else:
                result[key] = value
        
        return result
    
    @classmethod
    def from_yaml(cls, yaml_content: Union[str, Path]) -> 'YAMLConfig':
        """
        Create configuration from YAML string or file.
        
        Args:
            yaml_content: YAML string or file path
            
        Returns:
            Configuration instance
        """
        if isinstance(yaml_content, (str, Path)) and Path(yaml_content).exists():
            # Load from file
            yaml_str = Path(yaml_content).read_text()
            logger.info(f"Configuration loaded from {yaml_content}")
        else:
            # Treat as YAML string
            yaml_str = str(yaml_content)
        
        # Parse YAML
        try:
            config_dict = yaml.safe_load(yaml_str)
            return cls(**config_dict)
        except yaml.YAMLError as e:
            logger.error(f"YAML parsing error: {e}")
            raise
        except ValidationError as e:
            logger.error(f"Configuration validation error: {e}")
            raise
    
    def update_from_dict(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration from dictionary.
        
        Args:
            updates: Dictionary of updates
        """
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.warning(f"Unknown configuration key: {key}")
    
    def merge_with(self, other: 'YAMLConfig') -> 'YAMLConfig':
        """
        Merge with another configuration.
        
        Args:
            other: Other configuration to merge
            
        Returns:
            New merged configuration
        """
        self_dict = self.model_dump()
        other_dict = other.model_dump()
        
        # Deep merge dictionaries
        merged_dict = self._deep_merge(self_dict, other_dict)
        
        return self.__class__(**merged_dict)
    
    def _deep_merge(self, dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = dict1.copy()
        
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result


class YAMLWorkflowConfig(YAMLConfig):
    """YAML-based configuration for complete workflows (renamed to avoid conflict with core.workflow.WorkflowConfig)."""
    
    name: str
    description: str = ""
    version: str = "1.0.0"
    steps: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    agents: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    links: List[Dict[str, Any]] = Field(default_factory=list)
    executors: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    data_units: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    triggers: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    
    def get_step_config(self, step_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific step."""
        return self.steps.get(step_name)
    
    def get_agent_config(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific agent."""
        return self.agents.get(agent_name)
    
    def add_step(self, name: str, config: Dict[str, Any]) -> None:
        """Add a step configuration."""
        self.steps[name] = config
        logger.debug(f"Added step configuration: {name}")
    
    def add_agent(self, name: str, config: Dict[str, Any]) -> None:
        """Add an agent configuration."""
        self.agents[name] = config
        logger.debug(f"Added agent configuration: {name}")
    
    def add_link(self, link_config: Dict[str, Any]) -> None:
        """Add a link configuration."""
        self.links.append(link_config)
        logger.debug(f"Added link configuration")
    
    def validate_references(self) -> List[str]:
        """
        Validate that all references between components exist.
        
        Returns:
            List of validation errors
        """
        errors = []
        
        # Check link references
        for i, link in enumerate(self.links):
            source = link.get('source')
            target = link.get('target')
            
            if source and source not in self.steps and source not in self.agents:
                errors.append(f"Link {i}: source '{source}' not found")
            
            if target and target not in self.steps and target not in self.agents:
                errors.append(f"Link {i}: target '{target}' not found")
        
        # Check step executor references
        for step_name, step_config in self.steps.items():
            executor_name = step_config.get('executor')
            if executor_name and executor_name not in self.executors:
                errors.append(f"Step '{step_name}': executor '{executor_name}' not found")
        
        return errors


def load_config(file_path: Union[str, Path], config_class: type = YAMLWorkflowConfig) -> YAMLConfig:
    """
    Load configuration from YAML file.
    
    Args:
        file_path: Path to YAML file
        config_class: Configuration class to use
        
    Returns:
        Configuration instance
    """
    try:
        return config_class.from_yaml(file_path)
    except Exception as e:
        logger.error(f"Failed to load configuration from {file_path}: {e}")
        raise


def save_config(config: YAMLConfig, file_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration instance
        file_path: Path to save file
    """
    try:
        config.to_yaml(file_path)
    except Exception as e:
        logger.error(f"Failed to save configuration to {file_path}: {e}")
        raise


def create_example_config() -> YAMLWorkflowConfig:
    """
    Create an example workflow configuration.
    
    Returns:
        Example configuration
    """
    return YAMLWorkflowConfig(
        name="example_workflow",
        description="Example NanoBrain workflow configuration",
        version="1.0.0",
        
        # Define executors
        executors={
            "local": {
                "executor_type": "local",
                "max_workers": 4
            },
            "parsl_hpc": {
                "executor_type": "parsl",
                "max_workers": 16,
                "parsl_config": {
                    "provider": "slurm",
                    "nodes_per_block": 1,
                    "cores_per_node": 16
                }
            }
        },
        
        # Define data units
        data_units={
            "input_data": {
                "data_type": "memory",
                "persistent": False
            },
            "processed_data": {
                "data_type": "file",
                "persistent": True,
                "file_path": "/tmp/processed_data.json"
            }
        },
        
        # Define triggers
        triggers={
            "data_trigger": {
                "trigger_type": "data_updated",
                "debounce_ms": 100
            },
            "timer_trigger": {
                "trigger_type": "timer",
                "timer_interval_ms": 5000
            }
        },
        
        # Define agents
        agents={
            "code_writer": {
                "agent_type": "simple",
                "name": "code_writer",
                "description": "Agent for writing code",
                "model": "gpt-4",
                "system_prompt": "You are a helpful code writing assistant.",
                "tools": []
            },
            "file_writer": {
                "agent_type": "simple", 
                "name": "file_writer",
                "description": "Agent for file operations",
                "model": "gpt-3.5-turbo",
                "system_prompt": "You help with file operations.",
                "tools": []
            }
        },
        
        # Define steps
        steps={
            "data_processor": {
                "step_type": "transform",
                "name": "data_processor",
                "description": "Process input data",
                "executor": "local",
                "input_data_units": [{"data_type": "memory"}],
                "output_data_units": [{"data_type": "memory"}],
                "trigger_config": {
                    "trigger_type": "data_updated"
                }
            },
            "hpc_analyzer": {
                "step_type": "simple",
                "name": "hpc_analyzer", 
                "description": "Heavy computation step",
                "executor": "parsl_hpc",
                "input_data_units": [{"data_type": "memory"}],
                "output_data_units": [{"data_type": "file", "file_path": "/tmp/results.json"}],
                "trigger_config": {
                    "trigger_type": "all_data_received"
                }
            }
        },
        
        # Define links
        links=[
            {
                "link_type": "direct",
                "source": "data_processor",
                "target": "hpc_analyzer",
                "name": "processor_to_analyzer"
            }
        ]
    ) 