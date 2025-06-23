"""
Configuration system for NanoBrain Framework.

Provides YAML-based configuration management, simplified component factory,
and schema validation capabilities with focus on from_config pattern.
"""

from .yaml_config import YAMLConfig, WorkflowConfig
from .component_factory import (
    ComponentFactory,
    import_and_create_from_config
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
    
    # Simplified Component Factory
    'ComponentFactory',
    'import_and_create_from_config',
    
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

# Migration note: Legacy functions removed
# OLD: get_factory(), create_component_from_yaml(), create_workflow_from_yaml()
# NEW: Use ComponentFactory().create_component_from_config() or import_and_create_from_config()
# BEST: Use direct Class.from_config() pattern whenever possible 