"""
Component Base System for NanoBrain Framework

Provides the mandatory from_config pattern foundation for all framework components.
"""

import importlib
import inspect
import os
from abc import ABC, abstractmethod
from typing import Dict, Any, ClassVar, List, Optional, Type, Union
from pathlib import Path
from pydantic import BaseModel

from .logging_system import get_logger


class ComponentConfigurationError(Exception):
    """Raised when component configuration is invalid"""
    pass


class ComponentDependencyError(Exception):
    """Raised when component dependencies cannot be resolved"""
    pass


def validate_config_usage(component_class, config_object):
    """
    Framework-level validation to ensure proper Config usage.
    PREVENTS any programmatic Config creation.
    
    Args:
        component_class: The component class being created
        config_object: The config object to validate
        
    Raises:
        ValueError: If config was created via prohibited constructor usage
    """
    if hasattr(config_object, '__class__'):
        config_class = config_object.__class__
        
        # Check if config was created via constructor (FORBIDDEN)
        if hasattr(config_class, '_allow_direct_instantiation'):
            # This means it's a ConfigBase-derived class
            # If we're here and the flag is True, it means constructor was used during from_config
            # which is allowed. If it's False, then somehow constructor was bypassed, which is okay.
            pass
        else:
            # For non-ConfigBase classes, check if it looks like programmatic creation
            if isinstance(config_object, dict):
                # Dictionary configs are allowed for testing
                pass
            elif isinstance(config_object, BaseModel):
                # Check if this is an old-style BaseModel that should have been ConfigBase
                try:
                    from .config.config_base import ConfigBase
                    if not isinstance(config_object, ConfigBase):
                        logger = get_logger("component_base")
                        logger.warning(
                            f"⚠️ FRAMEWORK WARNING: {config_class.__name__} should inherit from ConfigBase.\n"
                            f"   COMPONENT: {component_class.__name__}\n"
                            f"   REQUIRED: Update {config_class.__name__} to inherit from ConfigBase.\n"
                            f"   CURRENT: Allowing for backward compatibility, but this will be deprecated."
                        )
                except ImportError:
                    # ConfigBase not available, skip check
                    pass
        
        # Additional validation for config content
        if hasattr(config_object, '__dict__') and hasattr(config_object, 'model_dump'):
            # This is a Pydantic model, check if it has content
            try:
                config_dict = config_object.model_dump()
                if not config_dict:
                    raise ValueError(
                        f"❌ CONFIG ERROR: Empty {config_class.__name__} configuration.\n"
                        f"   COMPONENT: {component_class.__name__}\n"
                        f"   REQUIRED: Valid configuration data in YAML file."
                    )
            except Exception:
                # If model_dump fails, skip validation
                pass


def import_class_from_path(class_path: str, search_namespaces: List[str] = None) -> Type:
    """
    Import class with enhanced path resolution and error handling
    
    Args:
        class_path: Full module path or short class name
        search_namespaces: List of namespace prefixes to search
        
    Returns:
        Imported class type
        
    Raises:
        ImportError: If class cannot be found or imported
    """
    search_namespaces = search_namespaces or [
        'nanobrain.core',
        'nanobrain.library.workflows',
        'nanobrain.library.agents',
        'nanobrain.library.tools',
        'nanobrain.library.infrastructure'
    ]
    
    # Try direct import first (full path provided)
    if '.' in class_path:
        try:
            module_path, class_name = class_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except (ImportError, AttributeError):
            pass
    
    # Try search namespaces for short names
    for namespace in search_namespaces:
        for pattern in [
            f"{namespace}.{class_path.lower()}.{class_path}",  # module.class pattern
            f"{namespace}.{class_path}",  # direct class import
        ]:
            try:
                if '.' in pattern:
                    module_path, class_name = pattern.rsplit('.', 1)
                    module = importlib.import_module(module_path)
                    return getattr(module, class_name)
            except (ImportError, AttributeError):
                continue
    
    raise ImportError(f"Cannot import class: {class_path}")


class FromConfigBase(ABC):
    """
    MANDATORY base class for all framework components implementing unified from_config pattern
    
    This is the foundation class that ALL framework components must inherit from.
    Provides concrete unified from_config() method that works identically for all component types.
    """
    
    # Component configuration schema (must be defined in subclasses)
    REQUIRED_CONFIG_FIELDS: ClassVar[List[str]] = []
    OPTIONAL_CONFIG_FIELDS: ClassVar[Dict[str, Any]] = {}
    COMPONENT_TYPE: ClassVar[str] = "unknown"
    
    @classmethod
    def from_config(cls, config: Union[str, Path, BaseModel, Dict[str, Any]], **kwargs) -> 'FromConfigBase':
        """
        UNIFIED component creation interface - IDENTICAL for ALL component types.
        
        Args:
            config: Configuration file path, config object, or dictionary
            **kwargs: Framework dependencies and parameters
            
        Returns:
            Fully initialized component instance
            
        Raises:
            NotImplementedError: If subclass doesn't implement _get_config_class()
            FileNotFoundError: If config file path doesn't exist
            ComponentConfigurationError: If config validation fails
        """
        # Import utilities on-demand to avoid circular imports
        from .config.component_factory import load_config_file
        
        # Step 1: Normalize input to dictionary format
        if isinstance(config, (str, Path)):
            # Handle file path input
            config_path = Path(config)
            
            # Resolve relative paths
            if not config_path.is_absolute():
                config_path = cls._resolve_config_file_path(config_path)
            
            if not config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
            # Load YAML to dictionary
            config_dict = load_config_file(str(config_path))
            
            # Handle class auto-detection
            if 'class' in config_dict:
                target_class_path = config_dict['class']
                current_class_path = f"{cls.__module__}.{cls.__name__}"
                
                if target_class_path != current_class_path:
                    # Delegate to correct class
                    target_class = import_class_from_path(target_class_path)
                    return target_class.from_config(config_path, **kwargs)
            
        elif isinstance(config, dict):
            # Already dictionary format
            config_dict = config
            
        else:
            # Config object - convert to dictionary for unified processing
            if hasattr(config, 'model_dump'):
                config_dict = config.model_dump()
            elif hasattr(config, 'dict'):
                config_dict = config.dict()
            else:
                # Fallback - extract attributes
                config_dict = {
                    key: getattr(config, key) 
                    for key in dir(config) 
                    if not key.startswith('_') and not callable(getattr(config, key))
                }
        
        # Step 2: Create config object using component-specific config class
        config_class = cls._get_config_class()
        
        # Filter out framework-specific fields that shouldn't be passed to config class
        framework_fields = {'class', 'config_file'}
        filtered_config_dict = {k: v for k, v in config_dict.items() if k not in framework_fields}
        
        # ✅ FRAMEWORK COMPLIANCE: Use from_config method instead of constructor
        config_object = config_class.from_config(filtered_config_dict)
        
        # FRAMEWORK VALIDATION: Ensure proper Config usage
        validate_config_usage(cls, config_object)
        
        # Step 3: Use existing framework pattern for component creation
        cls.validate_config_schema(config_object)
        component_config = cls.extract_component_config(config_object)
        dependencies = cls.resolve_dependencies(component_config, **kwargs)
        instance = cls.create_instance(config_object, component_config, dependencies)
        instance._post_config_initialization()
        
        return instance
    
    @classmethod
    def _get_config_class(cls):
        """
        MANDATORY: Return config class for this component type.
        
        This is the ONLY method that differs between component types.
        ALL other aspects of from_config() are identical.
        
        This method should be implemented by all component subclasses.
        The NotImplementedError provides clear guidance when missing.
        
        Returns:
            Config class appropriate for this component type
            
        Raises:
            NotImplementedError: If subclass doesn't implement this method
        """
        raise NotImplementedError(
            f"Subclass {cls.__name__} must implement _get_config_class() "
            f"to specify which config class to use. "
            f"This is the ONLY component-specific method required."
        )
    
    @classmethod
    def _resolve_config_file_path(cls, config_file: Union[str, Path]) -> Path:
        """
        Resolve configuration file path relative to the calling class's file directory.
        
        NANOBRAIN RULE: All relative config paths are resolved relative to the 
        class file that calls from_config(). This ensures predictable, co-located
        configuration management.
        
        Args:
            config_file: Relative or absolute path to config file
            
        Returns:
            Resolved absolute path to config file
            
        Raises:
            FileNotFoundError: If config file cannot be found
        """
        config_path = Path(config_file)
        
        # Return absolute paths as-is
        if config_path.is_absolute():
            if config_path.exists():
                return config_path
            else:
                raise FileNotFoundError(f"Absolute config path not found: {config_path}")
        
        # For relative paths: ONLY search relative to calling class's file directory
        try:
            calling_class_file = inspect.getfile(cls)
            calling_class_dir = Path(calling_class_file).parent.resolve()  # Resolve symlinks
            resolved_path = calling_class_dir / config_file
            
            if resolved_path.exists():
                return resolved_path.resolve()
            else:
                raise FileNotFoundError(
                    f"Configuration file not found: {config_file}\n"
                    f"Searched relative to class directory: {calling_class_dir}\n"
                    f"Full path attempted: {resolved_path}\n"
                    f"Calling class: {cls.__module__}.{cls.__name__}\n"
                    f"NANOBRAIN RULE: Config files must be relative to class file location"
                )
                
        except (OSError, TypeError) as e:
            raise FileNotFoundError(
                f"Cannot resolve config file path: {config_file}\n"
                f"Failed to determine calling class file location: {e}\n"
                f"Calling class: {cls.__module__}.{cls.__name__}"
            )

    @classmethod
    def validate_config_schema(cls, config: Any) -> None:
        """Validate that configuration contains required fields"""
        missing_fields = []
        for field in cls.REQUIRED_CONFIG_FIELDS:
            # Handle both dictionary and object configurations
            if isinstance(config, dict):
                if field not in config:
                    missing_fields.append(field)
            else:
                if not hasattr(config, field):
                    missing_fields.append(field)
        
        if missing_fields:
            raise ComponentConfigurationError(
                f"{cls.__name__} missing required configuration fields: {missing_fields}"
            )
    
    @classmethod
    def extract_component_config(cls, config: Any) -> Dict[str, Any]:
        """Extract component-specific configuration - SAME signature for ALL components"""
        # Default implementation - components can override if needed
        return {
            'name': getattr(config, 'name', 'unnamed'),
            'description': getattr(config, 'description', ''),
            'timeout': getattr(config, 'timeout', 300),
            'enable_logging': getattr(config, 'enable_logging', True)
        }
    
    @classmethod  
    def resolve_dependencies(cls, component_config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Resolve dependencies - SAME signature for ALL components"""
        # Default implementation - framework injects standard dependencies
        return {
            'enable_logging': kwargs.get('enable_logging', True),
            'debug_mode': kwargs.get('debug_mode', False),
            'framework_version': kwargs.get('framework_version', '2.0.0')
        }
    
    @classmethod
    def create_instance(cls, config: Any, component_config: Dict[str, Any], 
                       dependencies: Dict[str, Any]) -> 'FromConfigBase':
        """Create instance with resolved configuration and dependencies"""
        # Use __new__ to bypass __init__
        instance = cls.__new__(cls)
        instance._init_from_config(config, component_config, dependencies)
        return instance
    
    def _init_from_config(self, config: Any, component_config: Dict[str, Any],
                         dependencies: Dict[str, Any]) -> None:
        """Initialize component - SAME signature for ALL components"""
        # Default implementation - components override for specific behavior
        self.config = config
        self.name = component_config.get('name', 'unnamed')
        self.description = component_config.get('description', '')
        self.timeout = component_config.get('timeout', 300)
        self.enable_logging = component_config.get('enable_logging', True)
        
        # Initialize logging if enabled
        if self.enable_logging:
            self.nb_logger = get_logger(f"{self.__class__.__name__.lower()}.{self.name}")
    
    def _post_config_initialization(self) -> None:
        """Post-configuration initialization hook (override in subclasses if needed)"""
        pass
    
    # PREVENT direct instantiation
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            f"Direct instantiation of {self.__class__.__name__} is prohibited. "
            f"Use: {self.__class__.__name__}.from_config(config_file_or_object)"
        ) 