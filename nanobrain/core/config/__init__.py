"""
Configuration system for NanoBrain Framework.

Provides YAML-based configuration management, simplified component factory,
and schema validation capabilities with focus on from_config pattern.
"""

from .yaml_config import YAMLConfig, WorkflowConfig
from .component_factory import (
    create_component,
    get_component_class,
    ComponentRegistry,
    load_config_file,
    validate_component_config
)
from .schema_validator import (
    SchemaValidator, ConfigSchema, FieldSchema, ParameterSchema,
    FieldType, ConstraintType, FieldConstraint, ValidatorFunction,
    create_schema_from_yaml, validate_config_with_schema
)
try:
    from .config_manager import (
        ConfigManager, ProviderConfig,
        get_config_manager, get_api_key, get_provider_config,
        get_default_model, initialize_config, get_logging_mode,
        should_log_to_console, should_log_to_file, get_logging_config
    )
except ImportError:
    # Config manager not available
    ConfigManager = None
    ProviderConfig = None
    get_config_manager = None
    get_api_key = None
    get_provider_config = None
    get_default_model = None
    initialize_config = None
    get_logging_mode = None
    should_log_to_console = None
    should_log_to_file = None
    get_logging_config = None

# Factory function - creates from YAML configuration (NO HARDCODING)
def create_from_config_file(config_path: str, **kwargs) -> object:
    """Create component from pure YAML configuration file"""
    config = load_config_file(config_path)
    validate_component_config(config)
    
    class_path = config.get('class')
    if not class_path:
        raise ValueError(f"Configuration file must specify 'class' field: {config_path}")
    
    return create_component(class_path, config, **kwargs)


# All exports
__all__ = [
    # Core factory functions
    'create_component',
    'get_component_class', 
    'create_from_config_file',
    'load_config_file',
    'validate_component_config',
    
    # Configuration classes
    'YAMLConfig',
    'WorkflowConfig',
    'ComponentRegistry',
    
    # Schema validation
    'SchemaValidator',
    'ConfigSchema',
    'FieldSchema', 
    'ParameterSchema',
    'FieldType',
    'ConstraintType',
    'FieldConstraint',
    'ValidatorFunction',
    'create_schema_from_yaml',
    'validate_config_with_schema',
    
    # Config manager (if available)
    'ConfigManager',
    'ProviderConfig',
    'get_config_manager',
    'get_api_key',
    'get_provider_config',
    'get_default_model',
    'initialize_config',
    'get_logging_mode',
    'should_log_to_console',
    'should_log_to_file',
    'get_logging_config',
] 