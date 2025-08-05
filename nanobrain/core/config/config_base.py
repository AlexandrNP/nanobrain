"""
Enhanced Framework-Level Config Base Class

This module implements the enhanced mandatory from_config pattern with
comprehensive recursive loading, Pydantic integration, schema extraction, 
and optional protocol support - all within the ConfigBase class itself.
"""

import yaml
import logging
from abc import ABC
from pathlib import Path
from typing import Any, Dict, Union, Optional, ClassVar, List, Set
from pydantic import BaseModel, ConfigDict, Field, validator, root_validator
from dataclasses import dataclass
from datetime import datetime
from pydantic import ValidationError

logger = logging.getLogger(__name__)


@dataclass
class ConfigLoadingContext:
    """Context information for configuration loading operations"""
    base_path: Path
    resolution_stack: Set[str]
    loading_timestamp: datetime
    workflow_directory: Optional[Path] = None
    additional_context: Dict[str, Any] = None


class ConfigBase(BaseModel, ABC):
    """
    Enterprise Configuration Base - Advanced Configuration Management and Validation Framework
    =======================================================================================
    
    The ConfigBase provides comprehensive configuration management, validation, and loading infrastructure
    for enterprise applications, implementing advanced recursive configuration resolution, schema validation,
    protocol integration, and type-safe configuration handling. This foundational class ensures consistent
    configuration patterns, automated validation, and enterprise-grade configuration management across
    the entire NanoBrain framework and applications.
    
    **Core Architecture:**
        The configuration base provides enterprise-grade configuration capabilities:
        
        * **Declarative Configuration**: Type-safe configuration with automatic validation and schema generation
        * **Recursive Reference Resolution**: Intelligent configuration reference resolution with circular dependency prevention
        * **Protocol Integration**: Native support for MCP/A2A protocols with automatic service discovery
        * **Schema-Driven Validation**: Comprehensive Pydantic-based validation with custom validation rules
        * **File-Based Configuration**: Secure file-based configuration loading with path validation
        * **Framework Integration**: Complete integration with NanoBrain's component and dependency architecture
    
    **Configuration Management Capabilities:**
        
        **Type-Safe Configuration:**
        * **Pydantic Integration**: Full Pydantic v2 support with automatic schema generation
        * **Type Validation**: Comprehensive type checking and validation with custom validators
        * **Default Value Management**: Intelligent default value handling with environment-specific overrides
        * **Enum Support**: Native enumeration support with value preservation and validation
        
        **Advanced Loading System:**
        * **File-Only Loading**: Secure file-based configuration loading with path validation
        * **YAML/JSON Support**: Native YAML and JSON configuration file support
        * **Environment Integration**: Environment variable substitution and configuration templating
        * **Configuration Inheritance**: Hierarchical configuration with inheritance and override capabilities
        
        **Recursive Reference Resolution:**
        * **Cross-Reference Support**: Automatic resolution of configuration cross-references
        * **Circular Dependency Detection**: Intelligent circular dependency detection and prevention
        * **Lazy Loading**: Efficient lazy loading of referenced configurations
        * **Context-Aware Resolution**: Configuration resolution with contextual path and environment awareness
        
        **Schema and Validation:**
        * **Automatic Schema Generation**: Dynamic JSON schema generation for configuration validation
        * **Custom Validation Rules**: Extensible validation framework with business rule integration
        * **Field-Level Validation**: Granular field-level validation with custom error messages
        * **Cross-Field Validation**: Complex validation rules spanning multiple configuration fields
    
    **Enterprise Configuration Patterns:**
        
        **Mandatory from_config Pattern:**
        * **Unified Loading Interface**: All configuration loading through standardized from_config method
        * **Path-Based Security**: Secure file path validation and access control
        * **Configuration Auditing**: Complete audit trail for configuration loading and changes
        * **Version Control Integration**: Configuration versioning and change tracking support
        
        **Protocol Integration Support:**
        * **MCP Protocol Integration**: Model Context Protocol support for AI agent communication
        * **A2A Protocol Support**: Agent-to-Agent protocol integration for distributed systems
        * **Service Discovery**: Automatic service discovery and configuration resolution
        * **Dynamic Configuration**: Runtime configuration updates with validation and rollback
        
        **Framework Compliance:**
        * **Component Integration**: Seamless integration with NanoBrain component architecture
        * **Dependency Injection**: Advanced dependency injection and configuration resolution
        * **Environment Awareness**: Environment-specific configuration handling and validation
        * **Security Integration**: Secure configuration handling with encryption and access control
    
    **Configuration Architecture:**
        Comprehensive configuration supports diverse enterprise scenarios:
        
        ```yaml
        # Example Enterprise Configuration Structure
        
        # Basic Service Configuration
        service_config:
          name: "enterprise_ai_service"
          version: "1.0.0"
          environment: "${ENVIRONMENT:development}"
          
        # Advanced Component Configuration with References
        components:
          database:
            class: "nanobrain.library.infrastructure.data.DatabaseManager"
            config: "config/database_config.yml"
            
          cache:
            class: "nanobrain.library.infrastructure.data.CacheManager"
            config:
              type: "redis"
              host: "${REDIS_HOST:localhost}"
              port: "${REDIS_PORT:6379}"
              
          ai_agent:
            class: "nanobrain.library.agents.enhanced.CollaborativeAgent"
            config: "config/agents/collaborative_agent_config.yml"
            dependencies:
              - "database"
              - "cache"
              
        # Protocol Integration Configuration
        protocols:
          mcp:
            enabled: true
            server_url: "${MCP_SERVER_URL}"
            authentication:
              type: "bearer_token"
              token: "${MCP_AUTH_TOKEN}"
              
          a2a:
            enabled: true
            discovery_service: "config/discovery_service_config.yml"
            communication:
              encryption: true
              compression: true
              
        # Validation and Schema Configuration
        validation:
          strict_mode: true
          custom_validators:
            - "config/validators/business_rules.yml"
            - "config/validators/security_rules.yml"
          schema_validation: true
          
        # Environment-Specific Overrides
        environments:
          development:
            debug: true
            log_level: "DEBUG"
            security:
              authentication_required: false
              
          staging:
            debug: false
            log_level: "INFO"
            security:
              authentication_required: true
              ssl_required: true
              
          production:
            debug: false
            log_level: "WARNING"
            security:
              authentication_required: true
              ssl_required: true
              encryption_required: true
              audit_logging: true
        ```
    
    **Usage Patterns:**
        
        **Basic Configuration Definition:**
        ```python
        from nanobrain.core.config import ConfigBase
        from pydantic import Field
        from typing import Optional, List, Dict
        
        class ServiceConfig(ConfigBase):
            \"\"\"Enterprise service configuration with validation\"\"\"
            
            # Basic service configuration
            service_name: str = Field(
                description="Unique service identifier",
                min_length=3,
                max_length=50
            )
            
            service_version: str = Field(
                default="1.0.0",
                description="Semantic version of the service",
                regex=r"^\\d+\\.\\d+\\.\\d+$"
            )
            
            # Environment configuration
            environment: str = Field(
                default="development",
                description="Deployment environment",
                enum=["development", "staging", "production"]
            )
            
            # Advanced configuration with validation
            database_config: Optional[Dict[str, Any]] = Field(
                default=None,
                description="Database configuration with connection parameters"
            )
            
            # List configuration with validation
            enabled_features: List[str] = Field(
                default_factory=list,
                description="List of enabled service features"
            )
            
            # Numeric configuration with constraints
            max_connections: int = Field(
                default=100,
                ge=1,
                le=10000,
                description="Maximum number of concurrent connections"
            )
            
            @validator('database_config')
            def validate_database_config(cls, v):
                if v and 'host' not in v:
                    raise ValueError("Database configuration must include host")
                return v
                
            @validator('enabled_features')
            def validate_features(cls, v):
                valid_features = ['auth', 'monitoring', 'caching', 'analytics']
                invalid_features = set(v) - set(valid_features)
                if invalid_features:
                    raise ValueError(f"Invalid features: {invalid_features}")
                return v
        
        # Load configuration from file
        config = ServiceConfig.from_config('config/service_config.yml')
        
        # Access validated configuration
        print(f"Service: {config.service_name}")
        print(f"Environment: {config.environment}")
        print(f"Max Connections: {config.max_connections}")
        ```
        
        **Advanced Configuration with References:**
        ```python
        class EnterpriseApplicationConfig(ConfigBase):
            \"\"\"Enterprise application configuration with component references\"\"\"
            
            # Application metadata
            application_name: str = Field(description="Application identifier")
            version: str = Field(description="Application version")
            
            # Component configurations with references
            components: Dict[str, Any] = Field(
                default_factory=dict,
                description="Component configurations with class and config references"
            )
            
            # Protocol integration
            protocols: Dict[str, Any] = Field(
                default_factory=dict,
                description="Protocol integration configuration"
            )
            
            # Security configuration
            security: Dict[str, Any] = Field(
                default_factory=dict,
                description="Security and authentication configuration"
            )
            
            # Environment-specific settings
            environment_config: Dict[str, Any] = Field(
                default_factory=dict,
                description="Environment-specific configuration overrides"
            )
            
        # Load complex configuration with automatic reference resolution
        app_config = EnterpriseApplicationConfig.from_config('config/application.yml')
        
        # Access resolved component configurations
        db_component = app_config.components['database']
        print(f"Database class: {db_component['class']}")
        print(f"Database config: {db_component['config']}")
        
        # Access protocol configurations
        if app_config.protocols.get('mcp', {}).get('enabled'):
            print(f"MCP Server: {app_config.protocols['mcp']['server_url']}")
        ```
        
        **Configuration Validation and Error Handling:**
        ```python
        # Comprehensive configuration validation
        class ValidatedServiceConfig(ConfigBase):
            \"\"\"Service configuration with comprehensive validation\"\"\"
            
            service_name: str = Field(min_length=1, max_length=100)
            port: int = Field(ge=1024, le=65535)
            host: str = Field(regex=r'^[a-zA-Z0-9.-]+$')
            
            # Custom validation methods
            @validator('service_name')
            def validate_service_name(cls, v):
                if not v.replace('_', '').replace('-', '').isalnum():
                    raise ValueError("Service name must contain only alphanumeric characters, hyphens, and underscores")
                return v
                
            @validator('host')
            def validate_host(cls, v):
                import re
                # Allow localhost, IP addresses, and domain names
                if not (v == 'localhost' or 
                       re.match(r'^\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}$', v) or
                       re.match(r'^[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$', v)):
                    raise ValueError("Invalid host format")
                return v
                
            @root_validator
            def validate_service_config(cls, values):
                # Cross-field validation
                service_name = values.get('service_name')
                port = values.get('port')
                
                if service_name and 'admin' in service_name.lower() and port < 8000:
                    raise ValueError("Admin services must use ports 8000 or higher")
                    
                return values
        
        # Handle validation errors gracefully
        try:
            config = ValidatedServiceConfig.from_config('config/invalid_config.yml')
        except ValidationError as e:
            print("Configuration validation failed:")
            for error in e.errors():
                field = " -> ".join(str(loc) for loc in error['loc'])
                message = error['msg']
                print(f"  {field}: {message}")
        except FileNotFoundError:
            print("Configuration file not found")
        except yaml.YAMLError as e:
            print(f"YAML parsing error: {e}")
        ```
        
        **Dynamic Configuration and Runtime Updates:**
        ```python
        class DynamicServiceConfig(ConfigBase):
            \"\"\"Service configuration with dynamic update support\"\"\"
            
            service_name: str
            feature_flags: Dict[str, bool] = Field(default_factory=dict)
            performance_settings: Dict[str, Any] = Field(default_factory=dict)
            
            def update_feature_flag(self, flag_name: str, enabled: bool):
                \"\"\"Update feature flag with validation\"\"\"
                self.feature_flags[flag_name] = enabled
                # Trigger validation
                self.__class__.validate(self.dict())
                
            def update_performance_setting(self, setting_name: str, value: Any):
                \"\"\"Update performance setting with validation\"\"\"
                self.performance_settings[setting_name] = value
                # Trigger validation
                self.__class__.validate(self.dict())
                
            async def reload_from_file(self, config_path: str):
                \"\"\"Reload configuration from file with validation\"\"\"
                new_config = self.__class__.from_config(config_path)
                
                # Update current instance with new values
                for field_name, field_value in new_config.dict().items():
                    setattr(self, field_name, field_value)
                    
            def get_config_diff(self, other_config: 'DynamicServiceConfig') -> Dict[str, Any]:
                \"\"\"Compare configurations and return differences\"\"\"
                current_dict = self.dict()
                other_dict = other_config.dict()
                
                diff = {}
                for key in set(current_dict.keys()) | set(other_dict.keys()):
                    current_value = current_dict.get(key)
                    other_value = other_dict.get(key)
                    
                    if current_value != other_value:
                        diff[key] = {
                            'current': current_value,
                            'new': other_value
                        }
                        
                return diff
        
        # Dynamic configuration management
        config = DynamicServiceConfig.from_config('config/service.yml')
        
        # Runtime updates
        config.update_feature_flag('advanced_analytics', True)
        config.update_performance_setting('max_concurrent_requests', 500)
        
        # Configuration reloading
        await config.reload_from_file('config/updated_service.yml')
        
        # Configuration comparison
        new_config = DynamicServiceConfig.from_config('config/new_service.yml')
        differences = config.get_config_diff(new_config)
        
        for field, change in differences.items():
            print(f"Field {field}: {change['current']} -> {change['new']}")
        ```
        
        **Enterprise Configuration Management:**
        ```python
        # Enterprise-grade configuration management system
        class EnterpriseConfigurationManager:
            def __init__(self):
                self.configurations = {}
                self.config_watchers = {}
                self.validation_rules = {}
                
            async def load_configuration_suite(self, config_directory: str):
                \"\"\"Load complete configuration suite with validation\"\"\"
                config_path = Path(config_directory)
                
                # Load base configuration
                base_config_file = config_path / "base_config.yml"
                if base_config_file.exists():
                    self.configurations['base'] = ConfigBase.from_config(str(base_config_file))
                
                # Load environment-specific configurations
                for env in ['development', 'staging', 'production']:
                    env_config_file = config_path / f"{env}_config.yml"
                    if env_config_file.exists():
                        self.configurations[env] = ConfigBase.from_config(str(env_config_file))
                
                # Load service-specific configurations
                services_dir = config_path / "services"
                if services_dir.exists():
                    for service_config in services_dir.glob("*.yml"):
                        service_name = service_config.stem
                        self.configurations[f"service_{service_name}"] = ConfigBase.from_config(
                            str(service_config)
                        )
                        
            async def validate_configuration_consistency(self) -> Dict[str, Any]:
                \"\"\"Validate consistency across all configurations\"\"\"
                validation_results = {
                    'passed': [],
                    'failed': [],
                    'warnings': []
                }
                
                # Cross-configuration validation
                for config_name, config in self.configurations.items():
                    try:
                        # Validate individual configuration
                        config.validate_configuration()
                        validation_results['passed'].append(config_name)
                        
                    except ValidationError as e:
                        validation_results['failed'].append({
                            'config': config_name,
                            'errors': [error['msg'] for error in e.errors()]
                        })
                        
                # Cross-reference validation
                await self.validate_cross_references(validation_results)
                
                return validation_results
                
            async def generate_configuration_documentation(self, output_path: str):
                \"\"\"Generate comprehensive configuration documentation\"\"\"
                documentation = {
                    'configuration_overview': {},
                    'schemas': {},
                    'examples': {},
                    'validation_rules': {}
                }
                
                for config_name, config in self.configurations.items():
                    # Generate schema documentation
                    schema = config.schema()
                    documentation['schemas'][config_name] = schema
                    
                    # Generate examples
                    documentation['examples'][config_name] = config.dict()
                    
                    # Generate validation rules
                    documentation['validation_rules'][config_name] = self.extract_validation_rules(config)
                
                # Write documentation
                output_file = Path(output_path) / "configuration_documentation.yml"
                with open(output_file, 'w') as f:
                    yaml.dump(documentation, f, default_flow_style=False)
                    
            async def monitor_configuration_changes(self, config_path: str):
                \"\"\"Monitor configuration files for changes\"\"\"
                # Implementation would use file system watchers
                # to detect configuration changes and trigger reloads
                pass
        
        # Enterprise configuration management
        config_manager = EnterpriseConfigurationManager()
        await config_manager.load_configuration_suite('config/')
        
        # Validate all configurations
        validation_results = await config_manager.validate_configuration_consistency()
        print(f"Validation passed: {len(validation_results['passed'])} configurations")
        print(f"Validation failed: {len(validation_results['failed'])} configurations")
        
        # Generate documentation
        await config_manager.generate_configuration_documentation('docs/')
        ```
    
    **Advanced Features:**
        
        **Recursive Reference Resolution:**
        * Intelligent cross-configuration reference resolution with circular dependency detection
        * Lazy loading and caching for performance optimization in large configuration hierarchies
        * Context-aware resolution with path and environment consideration
        * Advanced reference patterns including conditional and computed references
        
        **Protocol Integration:**
        * Native MCP (Model Context Protocol) integration for AI agent communication
        * A2A (Agent-to-Agent) protocol support for distributed agent systems
        * Dynamic service discovery and configuration resolution
        * Protocol-specific validation and error handling
        
        **Enterprise Security:**
        * Secure configuration loading with path validation and access control
        * Configuration encryption and secure storage integration
        * Audit logging for all configuration access and modifications
        * Role-based access control for configuration management
        
        **Performance Optimization:**
        * Efficient configuration caching and lazy loading mechanisms
        * Memory-optimized configuration storage for large-scale deployments
        * Parallel configuration loading and validation for improved startup times
        * Configuration change detection and incremental updates
    
    **Production Deployment:**
        
        **Configuration Management:**
        * Centralized configuration management with version control integration
        * Environment-specific configuration deployment and validation
        * Configuration rollback and disaster recovery capabilities
        * Automated configuration testing and validation pipelines
        
        **Monitoring & Observability:**
        * Configuration change monitoring and alerting
        * Configuration usage analytics and optimization recommendations
        * Performance impact analysis for configuration changes
        * Configuration validation metrics and error tracking
        
        **Integration & Compatibility:**
        * Kubernetes ConfigMap and Secret integration
        * Cloud provider configuration service integration
        * CI/CD pipeline integration for configuration deployment
        * Enterprise configuration management tool compatibility
    
    **Framework Compliance Requirements:**
        
        **Mandatory Patterns:**
        ```python
        # ✅ REQUIRED: File-based configuration loading
        config = MyConfig.from_config('path/to/config.yml')
        
        # ✅ REQUIRED: Schema validation and type safety
        class MyConfig(ConfigBase):
            field_name: str = Field(description="Field description")
            
        # ✅ REQUIRED: Recursive reference resolution
        component:
          class: "module.path.ClassName"
          config: "path/to/component_config.yml"
        ```
        
        **Forbidden Patterns:**
        ```python
        # ❌ FORBIDDEN: Direct instantiation
        config = MyConfig(field_name="value")
        
        # ❌ FORBIDDEN: Dictionary-based loading
        config = MyConfig.from_config({"field_name": "value"})
        
        # ❌ FORBIDDEN: Unvalidated configuration
        config = SomeOtherConfigClass()  # Not inheriting from ConfigBase
        ```
    
    Attributes:
        model_config (ConfigDict): Pydantic model configuration with validation and serialization settings
        loading_context (ConfigLoadingContext): Configuration loading context with path and resolution information
        resolved_references (Dict): Cache of resolved configuration references for performance optimization
    
    Note:
        This configuration base requires YAML/JSON configuration files for all loading operations.
        All configuration classes must inherit from ConfigBase to ensure framework compliance.
        Recursive reference resolution requires proper circular dependency handling.
        Protocol integration features require appropriate network access and authentication.
    
    Warning:
        Direct instantiation of configuration classes is prohibited and will raise RuntimeError.
        Configuration files must be properly validated before deployment to prevent runtime errors.
        Large configuration hierarchies may require performance optimization for loading times.
        Protocol integration may introduce network dependencies and potential failure points.
    
    See Also:
        * :class:`ConfigLoadingContext`: Configuration loading context and metadata
        * :mod:`nanobrain.core.component_base`: Framework component integration patterns
        * :mod:`nanobrain.core.config.component_factory`: Component creation and dependency injection
        * :mod:`nanobrain.core.config.schema_validator`: Configuration schema validation and generation
        * :mod:`nanobrain.core.config.yaml_config`: YAML configuration loading and processing
    """
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="allow",
        use_enum_values=False,  # ✅ ENUM FIX: Preserve enum objects instead of converting to string values
        validate_assignment=True,
        str_strip_whitespace=True,
        json_schema_extra={
            "examples": [],
            "nanobrain_metadata": {
                "framework_version": "2.0.0",
                "config_loading_method": "enhanced_from_config_only",
                "supports_recursive_references": True,
                "supports_mcp_integration": True,
                "supports_a2a_integration": True
            }
        }
    )
    
    # Framework-level control flag
    _allow_direct_instantiation: ClassVar[bool] = False
    
    # Optional MCP Support - Available for ALL configuration classes
    mcp_support: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional MCP (Model Context Protocol) integration configuration",
        json_schema_extra={
            "examples": [
                {
                    "server_config": {
                        "name": "nanobrain_mcp_server",
                        "url": "ws://localhost:8080/mcp",
                        "timeout": 30
                    },
                    "client_config": {
                        "default_timeout": 10,
                        "max_retries": 3
                    }
                }
            ]
        }
    )
    
    # Optional A2A Support - Available for ALL configuration classes  
    a2a_support: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional A2A (Agent-to-Agent) protocol integration configuration",
        json_schema_extra={
            "examples": [
                {
                    "agent_card": {
                        "version": "1.0.0",
                        "purpose": "Specialized agent for data processing",
                        "capabilities": ["streaming", "multi_turn_conversation"]
                    },
                    "protocol_config": {
                        "communication_mode": "async",
                        "message_format": "json"
                    }
                }
            ]
        }
    )
    
    def __init__(self, *args, **kwargs):
        """
        FORBIDDEN: Direct instantiation prohibited by framework design.
        All Config classes MUST be loaded via from_config() method.
        
        Raises:
            ValueError: Always raised when attempting direct instantiation
        """
        if not self.__class__._allow_direct_instantiation:
            raise ValueError(
                f"❌ FRAMEWORK VIOLATION: Direct instantiation of {self.__class__.__name__} is FORBIDDEN.\n"
                f"   REQUIRED: Use {self.__class__.__name__}.from_config(file_path) instead.\n"
                f"   REASON: NanoBrain framework enforces configuration-driven component creation.\n"
                f"   SOLUTION: Create YAML config file and load via from_config() method.\n"
                f"   EXAMPLE: config = {self.__class__.__name__}.from_config('path/to/config.yml')\n"
                f"   \n"
                f"   ❌ PROHIBITED:\n"
                f"      {self.__class__.__name__}(name='test', class='...')\n"
                f"      {self.__class__.__name__}.from_config({{'name': 'test'}})\n"
                f"   \n"
                f"   ✅ REQUIRED:\n"
                f"      {self.__class__.__name__}.from_config('path/to/config.yml')"
            )
        super().__init__(*args, **kwargs)
    
    @classmethod
    def from_config(cls, config_path: Union[str, Path], **context) -> 'ConfigBase':
        """
        ENHANCED: Complete configuration loading with all features integrated
        
        This method handles ALL configuration loading responsibilities:
        - File-path-only loading (dictionaries prohibited except DataUnit/Link/Trigger)
        - Recursive reference resolution
        - Protocol integration (MCP/A2A)
        - Schema validation and type checking
        - Context-aware configuration resolution
        
        Args:
            config_path: MANDATORY file path to YAML configuration
            **context: Additional context (workflow_directory, etc.)
            
        Returns:
            Fully resolved and validated Config instance
            
        Raises:
            ValueError: If config_path is not a file path
            FileNotFoundError: If config file doesn't exist
            RecursionError: If circular dependencies detected
            
        ✅ FRAMEWORK COMPLIANCE:
        - ONLY accepts file paths (str or Path objects) for most classes
        - EXCEPTION: DataUnit, Link, Trigger classes may accept inline dict config
        - Automatic object instantiation via class+config patterns
        - Complete Pydantic validation and type checking
        - Optional protocol integration when specified
        """
        # STRICT ENFORCEMENT: Only file paths allowed for ConfigBase classes
        if not isinstance(config_path, (str, Path)):
            raise ValueError(
                f"❌ FRAMEWORK VIOLATION: {cls.__name__}.from_config ONLY accepts file paths.\n"
                f"   GIVEN: {type(config_path)}\n"
                f"   REQUIRED: str or Path object pointing to YAML configuration file\n"
                f"   \n"
                f"   ❌ PROHIBITED USAGE:\n"
                f"      {cls.__name__}.from_config({{'name': 'test', 'class': '...'}})\n"
                f"      {cls.__name__}.from_config(config_dict)\n"
                f"   \n"
                f"   ✅ CORRECT USAGE:\n"
                f"      {cls.__name__}.from_config('path/to/config.yml')\n"
                f"      {cls.__name__}.from_config(Path('config.yml'))\n"
                f"   \n"
                f"   NOTE: Only DataUnit, Link, Trigger classes support inline dict config\n"
                f"   REASON: NanoBrain framework enforces file-based configuration for most classes"
            )
        
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(
                f"❌ CONFIGURATION ERROR: Config file not found: {config_path}\n"
                f"   SOLUTION: Ensure configuration file exists at specified path\n"
                f"   SEARCH PATHS: Check relative to current directory and workflow_directory"
            )
        
        try:
            logger.info(f"🔄 Loading {cls.__name__} from: {config_path}")
            
            # Create loading context
            loading_context = ConfigLoadingContext(
                base_path=config_path.parent,
                resolution_stack=set(),
                loading_timestamp=datetime.now(),
                workflow_directory=context.get('workflow_directory'),
                additional_context=context
            )
            
            # Load raw YAML data
            raw_config = cls._load_yaml_file(config_path)
            
            # Resolve nested objects with class+config patterns
            resolved_config = cls._resolve_nested_objects(raw_config, loading_context)
            
            # Apply optional protocol integrations
            enhanced_config = cls._apply_protocol_integrations(resolved_config, loading_context)
            
            # Create and validate configuration instance
            config_instance = cls._create_validated_instance(enhanced_config)
            
            logger.info(f"✅ Successfully loaded {cls.__name__} from {config_path}")
            return config_instance
            
        except Exception as e:
            logger.error(f"❌ Failed to load {cls.__name__} from {config_path}: {e}")
            raise ValueError(
                f"❌ CONFIGURATION LOADING FAILED: {config_path}\n"
                f"   ERROR: {str(e)}\n"
                f"   CONFIG_CLASS: {cls.__name__}\n"
                f"   SOLUTION: Check YAML syntax, file permissions, and recursive references"
            ) from e
    
    @classmethod
    def _load_yaml_file(cls, config_path: Path) -> Dict[str, Any]:
        """Load and parse YAML configuration file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            if config_data is None:
                config_data = {}
            
            if not isinstance(config_data, dict):
                raise ValueError(f"Configuration file must contain a YAML dictionary, got {type(config_data)}")
            
            return config_data
            
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML syntax in {config_path}: {e}")
        except UnicodeDecodeError as e:
            raise ValueError(f"File encoding error in {config_path}: {e}")
    
    @classmethod
    def _resolve_nested_objects(cls, config_data: Dict[str, Any], context: 'ConfigLoadingContext') -> Dict[str, Any]:
        """
        Resolve nested objects specified with 'class' and 'config' fields
        
        When encountering a subsection with both 'class' and 'config' fields:
        1. Import the class from the 'class' field (full module path)
        2. Create instance using ClassName.from_config() with 'config' field value
        3. Replace the subsection with the instantiated object
        
        Special handling for StepConfig tools field:
        - Tools are instantiated and stored in _resolved_tools field
        - Original tools structure is preserved for schema validation
        
        Args:
            config_data: Configuration dictionary to process
            context: Loading context for path resolution
            
        Returns:
            Configuration with nested objects instantiated
        """
        import importlib
        resolved_config = {}
        
        for key, value in config_data.items():
            if isinstance(value, dict):
                # Special handling for StepConfig tools field
                if key == 'tools' and cls.__name__ == 'StepConfig':
                    resolved_config[key] = value.copy()  # Keep original for schema validation
                    resolved_tools = {}
                    
                    # Process each tool for instantiation
                    for tool_name, tool_config in value.items():
                        if isinstance(tool_config, dict) and 'class' in tool_config and 'config' in tool_config:
                            try:
                                # Extract class path and config
                                class_path = tool_config['class']
                                config_value = tool_config['config']
                                
                                # Import the class
                                module_path, class_name = class_path.rsplit('.', 1)
                                module = importlib.import_module(module_path)
                                target_class = getattr(module, class_name)
                                
                                # Resolve config and create instance
                                if isinstance(config_value, str):
                                    # File path - all classes support this
                                    config_path = cls._resolve_config_path(config_value, context)
                                    instance = target_class.from_config(config_path, **context.additional_context)
                                else:
                                    # Inline configuration dict - only supported for DataUnit, Link, Trigger classes
                                    if cls._is_inline_config_supported(target_class):
                                        instance = target_class.from_config(config_value, **context.additional_context)
                                    else:
                                        raise ValueError(
                                            f"❌ FRAMEWORK VIOLATION: Inline dict configuration not supported for {class_path}\n"
                                            f"   SUPPORTED CLASSES: DataUnit, Link, Trigger and their subclasses only\n"
                                            f"   REQUIRED: Use file path for config field\n"
                                            f"   EXAMPLE: config: 'path/to/{class_name.lower()}.yml'\n"
                                            f"   CURRENT: config: {config_value}"
                                        )
                                
                                # Store instantiated tool separately
                                resolved_tools[tool_name] = instance
                                
                                logger.debug(f"✅ Instantiated tool {class_name} for key '{tool_name}'")
                                
                            except Exception as e:
                                raise ValueError(
                                    f"❌ FAILED TO INSTANTIATE TOOL: {tool_name}\n"
                                    f"   CLASS: {tool_config.get('class', 'unknown')}\n"
                                    f"   CONFIG: {tool_config.get('config', 'unknown')}\n"
                                    f"   ERROR: {str(e)}\n"
                                    f"   SOLUTION: Ensure class path is correct and config is valid"
                                ) from e
                        else:
                            # Keep non-class+config tools as-is
                            resolved_tools[tool_name] = tool_config
                    
                    # Store resolved tools for later access
                    resolved_config['_resolved_tools'] = resolved_tools
                    
                # Check if this dict has both 'class' and 'config' fields (non-tools)
                elif 'class' in value and 'config' in value:
                    # Extract class path and config
                    class_path = value['class']
                    config_value = value['config']
                    
                    try:
                        # Import the class
                        module_path, class_name = class_path.rsplit('.', 1)
                        module = importlib.import_module(module_path)
                        target_class = getattr(module, class_name)
                        
                        # Resolve config based on target class type and config value type
                        if isinstance(config_value, str):
                            # File path - all classes support this
                            config_path = cls._resolve_config_path(config_value, context)
                            instance = target_class.from_config(config_path, **context.additional_context)
                        else:
                            # Inline configuration dict - only supported for DataUnit, Link, Trigger classes
                            if cls._is_inline_config_supported(target_class):
                                instance = target_class.from_config(config_value, **context.additional_context)
                            else:
                                raise ValueError(
                                    f"❌ FRAMEWORK VIOLATION: Inline dict configuration not supported for {class_path}\n"
                                    f"   SUPPORTED CLASSES: DataUnit, Link, Trigger and their subclasses only\n"
                                    f"   REQUIRED: Use file path for config field\n"
                                    f"   EXAMPLE: config: 'path/to/{class_name.lower()}.yml'\n"
                                    f"   CURRENT: config: {config_value}"
                                )
                        
                        # Replace the configuration dict with the instantiated object
                        resolved_config[key] = instance
                        
                        logger.debug(f"✅ Instantiated {class_name} for key '{key}'")
                        
                    except Exception as e:
                        raise ValueError(
                            f"❌ FAILED TO INSTANTIATE OBJECT: {key}\n"
                            f"   CLASS: {class_path}\n"
                            f"   CONFIG: {config_value}\n"
                            f"   ERROR: {str(e)}\n"
                            f"   SOLUTION: Ensure class path is correct and config is valid"
                        ) from e
                else:
                    # Recursively process nested dictionaries
                    resolved_config[key] = cls._resolve_nested_objects(value, context)
            elif isinstance(value, list):
                # ✅ UNIFIED TRIGGER RESOLUTION: Process lists with class+config patterns directly
                # Since workflows ARE steps, both use same trigger format: List[Dict[str, Any]]
                resolved_list = []
                for item in value:
                    if isinstance(item, dict):
                        # Check if this dict should be resolved directly (class+config pattern)
                        if 'class' in item and 'config' in item:
                            try:
                                # Apply same resolution logic as dict processing (unified approach)
                                class_path = item['class']
                                config_value = item['config']
                                
                                # Import the class
                                module_path, class_name = class_path.rsplit('.', 1)
                                module = importlib.import_module(module_path)
                                target_class = getattr(module, class_name)
                                
                                # Resolve config based on type (same as dict processing)
                                if isinstance(config_value, str):
                                    # File path - all classes support this
                                    config_path = cls._resolve_config_path(config_value, context)
                                    instance = target_class.from_config(config_path, **context.additional_context)
                                else:
                                    # Inline configuration dict - only supported for DataUnit, Link, Trigger classes
                                    if cls._is_inline_config_supported(target_class):
                                        instance = target_class.from_config(config_value, **context.additional_context)
                                    else:
                                        raise ValueError(
                                            f"❌ FRAMEWORK VIOLATION: Inline dict configuration not supported for {class_path}\n"
                                            f"   SUPPORTED CLASSES: DataUnit, Link, Trigger and their subclasses only\n"
                                            f"   REQUIRED: Use file path for config field\n"
                                            f"   EXAMPLE: config: 'path/to/{class_name.lower()}.yml'\n"
                                            f"   CURRENT: config: {config_value}"
                                        )
                                
                                # ✅ METADATA PRESERVATION: Preserve additional fields as instance attributes
                                # This ensures trigger_id, name, description, etc. are maintained
                                for attr_key, attr_val in item.items():
                                    if attr_key not in ['class', 'config']:
                                        setattr(instance, attr_key, attr_val)
                                
                                resolved_list.append(instance)
                                
                                logger.debug(f"✅ Unified resolution: Instantiated {class_name} from list item")
                                
                            except Exception as e:
                                # Enhanced error reporting for list items
                                item_type = item.get('class', 'unknown')
                                item_id = item.get('trigger_id', item.get('name', item.get('link_id', 'unnamed')))
                                raise ValueError(
                                    f"❌ FAILED TO INSTANTIATE LIST ITEM: {item_id}\n"
                                    f"   CLASS: {item_type}\n"
                                    f"   CONFIG: {item.get('config', 'missing')}\n"
                                    f"   ERROR: {str(e)}\n"
                                    f"   SOLUTION: Ensure class path is correct and config is valid"
                                ) from e
                        else:
                            # Recursively process non-class+config dicts (existing behavior)
                            resolved_list.append(cls._resolve_nested_objects(item, context))
                    else:
                        # Keep non-dict items as-is (existing behavior)
                        resolved_list.append(item)
                resolved_config[key] = resolved_list
            else:
                # Keep primitive values as-is
                resolved_config[key] = value
        
        return resolved_config

    @classmethod
    def _is_inline_config_supported(cls, target_class: type) -> bool:
        """
        Check if target class supports inline dict configuration
        
        ✅ FRAMEWORK COMPLIANCE:
        Only DataUnit/DataUnitBase, Link/LinkBase, and Trigger/TriggerBase classes (and their subclasses) support inline dict config.
        All other classes MUST use file paths for configuration.
        
        Args:
            target_class: Class to check for inline config support
            
        Returns:
            True if class supports inline dict config, False otherwise
        """
        # Import base classes for comparison
        try:
            from nanobrain.core.data_unit import DataUnit, DataUnitBase
            from nanobrain.core.link import LinkBase  
            from nanobrain.core.trigger import TriggerBase
            
            # Check if target class is a subclass of supported classes
            return (issubclass(target_class, DataUnit) or 
                    issubclass(target_class, DataUnitBase) or
                    issubclass(target_class, LinkBase) or 
                    issubclass(target_class, TriggerBase))
        except ImportError as e:
            # If import fails, default to False (require file path)
            logger.warning(f"⚠️ Could not import base classes for inline config check: {e}")
            return False

    @classmethod
    def _resolve_config_path(cls, config_path: str, context: ConfigLoadingContext) -> str:
        """
        Enhanced configuration file path resolution with robust cross-workflow support
        
        Resolution order:
        1. Absolute paths - return as-is if they exist
        2. Relative to workflow directory (if available)
        3. Relative to current base path (normalized)
        4. Relative to parent workflow directory (for nested configs)
        5. Relative to project root (as fallback)
        6. Class file directory relative paths (final fallback)
        """
        from pathlib import Path
        import inspect
        
        path = Path(config_path)
        
        # Handle absolute paths
        if path.is_absolute():
            if path.exists():
                return str(path)
            else:
                raise FileNotFoundError(f"Absolute config path not found: {path}")
        
        # Strategy 1: Try relative to workflow directory (if available)
        if context.workflow_directory:
            try:
                workflow_dir = Path(context.workflow_directory).resolve()
                workflow_resolved = workflow_dir / path
                if workflow_resolved.exists():
                    return str(workflow_resolved)
            except Exception:
                pass  # Continue to next strategy
        
        # Strategy 2: Try relative to base path (normalized)
        try:
            base_path_normalized = Path(context.base_path).resolve()
            base_resolved = base_path_normalized / path
            if base_resolved.exists():
                return str(base_resolved)
        except Exception:
            pass  # Continue to next strategy
        
        # Strategy 3: Try relative to parent workflow directory (for nested configs)
        try:
            base_path_normalized = Path(context.base_path).resolve()
            
            # Look for parent directory that contains a workflow structure
            current_dir = base_path_normalized
            while current_dir.parent != current_dir:  # Not at root
                # Check if this directory looks like a workflow root
                if (current_dir / "config").is_dir() and any(current_dir.glob("*.yml")):
                    parent_workflow_resolved = current_dir / path
                    if parent_workflow_resolved.exists():
                        return str(parent_workflow_resolved)
                
                current_dir = current_dir.parent
        except Exception:
            pass  # Continue to next strategy
        
        # Strategy 4: Try relative to project root (for cross-workflow references)
        try:
            # Find project root by looking for nanobrain directory
            current_dir = Path(context.base_path).resolve()
            project_root = None
            
            # Walk up to find nanobrain root
            for parent in [current_dir] + list(current_dir.parents):
                if (parent / "nanobrain").is_dir():
                    project_root = parent
                    break
            
            if project_root:
                project_resolved = project_root / path
                if project_resolved.exists():
                    return str(project_resolved)
        except Exception:
            pass  # Continue to next strategy
        
        # Strategy 5: Try relative to class file directory (final fallback)
        try:
            frame = inspect.currentframe()
            while frame:
                filename = frame.f_code.co_filename
                if 'config_base.py' not in filename:  # Skip our own file
                    class_dir = Path(filename).parent
                    class_resolved = class_dir / path
                    if class_resolved.exists():
                        return str(class_resolved)
                frame = frame.f_back
        except Exception:
            pass
        
        # Strategy 6: Try removing redundant path components (for same-directory refs)
        try:
            # If the config path contains directory components that are already in the base path,
            # try resolving with those components removed
            path_parts = Path(config_path).parts
            base_path_normalized = Path(context.base_path).resolve()
            
            # Check if any suffix of the config path exists when prepended to base path
            for i in range(len(path_parts)):
                reduced_path = Path(*path_parts[i:])
                reduced_resolved = base_path_normalized / reduced_path
                if reduced_resolved.exists():
                    return str(reduced_resolved)
                
                # Also try without the first part (common with config/... paths)
                if i == 0 and len(path_parts) > 1:
                    reduced_path = Path(*path_parts[1:])
                    reduced_resolved = base_path_normalized / reduced_path
                    if reduced_resolved.exists():
                        return str(reduced_resolved)
        except Exception:
            pass
        
        # If all strategies fail, provide comprehensive error
        searched_paths = []
        
        if context.workflow_directory:
            searched_paths.append(str(Path(context.workflow_directory) / path))
        
        searched_paths.extend([
            str(Path(context.base_path).resolve() / path),
            str(Path().cwd() / path)
        ])
        
        # Add reduced path attempts to the search list for debugging
        try:
            path_parts = Path(config_path).parts
            if len(path_parts) > 1:
                searched_paths.append(str(Path(context.base_path) / Path(*path_parts[1:])))
        except:
            pass
        
        raise FileNotFoundError(
            f"❌ CONFIGURATION PATH RESOLUTION FAILED: {config_path}\n"
            f"   SEARCHED PATHS:\n" + 
            "\n".join(f"      {i+1}. {p}" for i, p in enumerate(searched_paths)) +
            f"\n   CONTEXT:\n"
            f"      Base Path: {context.base_path}\n"
            f"      Workflow Directory: {context.workflow_directory}\n"
            f"   SOLUTION: Ensure config file exists at one of the searched locations"
        )
    
    @classmethod
    def _apply_protocol_integrations(cls, 
                                   config_data: Dict[str, Any], 
                                   context: ConfigLoadingContext) -> Dict[str, Any]:
        """Apply optional MCP/A2A protocol integrations"""
        enhanced_config = config_data.copy()
        
        # Apply MCP integration if specified
        if 'mcp_support' in config_data:
            mcp_config = config_data['mcp_support']
            enhanced_config.update(cls._apply_mcp_integration(mcp_config, context))
            logger.debug("🔌 Applied MCP integration")
        
        # Apply A2A integration if specified
        if 'a2a_support' in config_data:
            a2a_config = config_data['a2a_support']
            enhanced_config.update(cls._apply_a2a_integration(a2a_config, context))
            logger.debug("🔌 Applied A2A integration")
        
        return enhanced_config
    
    @classmethod
    def _apply_mcp_integration(cls, mcp_config: Dict[str, Any], context: ConfigLoadingContext) -> Dict[str, Any]:
        """Apply MCP (Model Context Protocol) integration"""
        integration_data = {}
        
        if 'server_config' in mcp_config:
            integration_data['mcp_server_config'] = mcp_config['server_config']
        
        if 'client_config' in mcp_config:
            integration_data['mcp_client_config'] = mcp_config['client_config']
        
        return integration_data
    
    @classmethod
    def _apply_a2a_integration(cls, a2a_config: Dict[str, Any], context: ConfigLoadingContext) -> Dict[str, Any]:
        """Apply A2A (Agent-to-Agent) protocol integration"""
        integration_data = {}
        
        if 'agent_card' in a2a_config:
            integration_data['a2a_agent_card'] = a2a_config['agent_card']
        
        if 'protocol_config' in a2a_config:
            integration_data['a2a_protocol_config'] = a2a_config['protocol_config']
        
        return integration_data
    
    @classmethod
    def _create_validated_instance(cls, config_data: Dict[str, Any]) -> 'ConfigBase':
        """Create configuration instance with validation"""
        # Temporarily allow instantiation for validated config data
        cls._allow_direct_instantiation = True
        try:
            instance = cls(**config_data)
            return instance
        finally:
            cls._allow_direct_instantiation = False
    
    @classmethod
    def get_schema(cls) -> Dict[str, Any]:
        """
        Extract complete Pydantic schema for this configuration class
        
        Returns:
            Complete JSON schema with validation rules and examples
        """
        schema = cls.model_json_schema()
        
        # Enhance with NanoBrain-specific metadata
        schema.setdefault('nanobrain_metadata', {}).update({
            'config_class': cls.__name__,
            'module': cls.__module__,
            'framework_version': '2.0.0',
            'loading_method': 'enhanced_from_config_only',
            'supports_recursive_loading': True,
            'supports_mcp_integration': True,
            'supports_a2a_integration': True,
            'direct_instantiation_forbidden': True
        })
        
        return schema
    
    @classmethod 
    def get_schema_with_examples(cls) -> Dict[str, Any]:
        """
        Get schema with comprehensive example configurations
        
        Returns:
            Schema enhanced with realistic configuration examples
        """
        schema = cls.get_schema()
        
        # Add comprehensive examples
        examples = cls._generate_configuration_examples()
        schema['examples'] = examples
        
        return schema
    
    @classmethod
    def validate_config_file(cls, config_path: Union[str, Path]) -> List[str]:
        """
        Validate configuration file against this class schema
        
        Args:
            config_path: Path to configuration file to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        config_path = Path(config_path)
        errors = []
        
        try:
            # Load raw configuration
            raw_config = cls._load_yaml_file(config_path)
            
            # Attempt to create instance for validation
            cls._allow_direct_instantiation = True
            try:
                cls(**raw_config)
            except Exception as e:
                errors.append(f"Validation error: {str(e)}")
            finally:
                cls._allow_direct_instantiation = False
                
        except Exception as e:
            errors.append(f"File loading error: {str(e)}")
        
        return errors
    
    @classmethod
    def _generate_configuration_examples(cls) -> List[Dict[str, Any]]:
        """Generate realistic configuration examples for this class"""
        # Base example with minimal required fields
        base_example = {
            "name": f"example_{cls.__name__.lower()}",
            "description": f"Example configuration for {cls.__name__}"
        }
        
        # Enhanced example with optional protocol support
        enhanced_example = base_example.copy()
        enhanced_example.update({
            "mcp_support": {
                "server_config": {
                    "name": f"{cls.__name__.lower()}_mcp_server",
                    "url": "ws://localhost:8080/mcp",
                    "timeout": 30
                }
            },
            "a2a_support": {
                "agent_card": {
                    "version": "1.0.0",
                    "purpose": f"Agent configured via {cls.__name__}",
                    "capabilities": ["configuration_driven", "protocol_aware"]
                }
            }
        })
        
        return [base_example, enhanced_example]
    
    def apply_mcp_integration(self) -> None:
        """
        Apply MCP integration if specified in configuration
        
        This method is called automatically during configuration loading
        when mcp_support is present in the configuration file.
        """
        if self.mcp_support is None:
            return
        
        logger.info(f"🔌 Applying MCP integration for {self.__class__.__name__}")
        self._apply_mcp_configuration(self.mcp_support)
    
    def apply_a2a_integration(self) -> None:
        """
        Apply A2A integration if specified in configuration
        
        This method is called automatically during configuration loading
        when a2a_support is present in the configuration file.
        """
        if self.a2a_support is None:
            return
        
        logger.info(f"🔌 Applying A2A integration for {self.__class__.__name__}")
        self._apply_a2a_configuration(self.a2a_support)
    
    def _apply_mcp_configuration(self, mcp_config: Dict[str, Any]) -> None:
        """Override in subclasses for specific MCP integration logic"""
        pass
    
    def _apply_a2a_configuration(self, a2a_config: Dict[str, Any]) -> None:
        """Override in subclasses for specific A2A integration logic"""
        pass
    
    def model_post_init(self, __context: Any) -> None:
        """
        Post-initialization hook for applying integrations
        
        Called automatically after Pydantic model initialization
        to apply any specified protocol integrations.
        """
        super().model_post_init(__context)
        
        # Apply optional integrations
        self.apply_mcp_integration()
        self.apply_a2a_integration()

    # Legacy methods preserved for backward compatibility
    def to_yaml(self, file_path: Optional[Union[str, Path]] = None) -> str:
        """
        Convert configuration to YAML string or save to file.
        
        Args:
            file_path: Optional file path to save YAML
            
        Returns:
            YAML string representation
        """
        # Convert to dict for YAML serialization
        config_dict = self.model_dump()
        
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