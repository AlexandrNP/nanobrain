"""
Configuration system for NanoBrain Framework.

Provides YAML-based configuration management, component factory,
advanced schema validation capabilities, and global configuration management.
"""

from .yaml_config import YAMLConfig, WorkflowConfig
from .component_factory import (
    ComponentFactory, ComponentType,
    get_factory, set_factory,
    create_component_from_yaml, create_workflow_from_yaml
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
    # Fallback for when running from project root with src in path
    try:
        from config_manager import (
            ConfigManager, ProviderConfig,
            get_config_manager, get_api_key, get_provider_config,
            get_default_model, initialize_config, get_logging_mode,
            should_log_to_console, should_log_to_file, get_logging_config
        )
    except ImportError:
        # If config_manager is not available, provide dummy functions
        def get_config_manager():
            return None
        def get_api_key(provider):
            return None
        def get_provider_config(provider):
            return None
        def get_default_model(provider, model_type='chat'):
            return None
        def initialize_config(config_path=None):
            pass
        def get_logging_mode():
            return 'both'
        def should_log_to_console():
            return True
        def should_log_to_file():
            return True
        def get_logging_config():
            return {}
        
        class ConfigManager:
            pass
        class ProviderConfig:
            pass

__all__ = [
    # YAML Configuration
    'YAMLConfig',
    'WorkflowConfig',
    
    # Component Factory
    'ComponentFactory',
    'ComponentType',
    'get_factory',
    'set_factory',
    'create_component_from_yaml',
    'create_workflow_from_yaml',
    
    # Schema Validation
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
    
    # Global Configuration Management
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