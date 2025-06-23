"""
Component Base System for NanoBrain Framework

Provides the mandatory from_config pattern foundation for all framework components.
"""

import importlib
from abc import ABC, abstractmethod
from typing import Dict, Any, ClassVar, List, Optional, Type
from pydantic import BaseModel

from .logging_system import get_logger


class ComponentConfigurationError(Exception):
    """Raised when component configuration is invalid"""
    pass


class ComponentDependencyError(Exception):
    """Raised when component dependencies cannot be resolved"""
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
    MANDATORY base class for all framework components implementing from_config pattern
    
    This is the foundation class that ALL framework components must inherit from
    and implement the from_config pattern.
    """
    
    # Component configuration schema (must be defined in subclasses)
    REQUIRED_CONFIG_FIELDS: ClassVar[List[str]] = []
    OPTIONAL_CONFIG_FIELDS: ClassVar[Dict[str, Any]] = {}
    COMPONENT_TYPE: ClassVar[str] = "unknown"
    
    @classmethod
    @abstractmethod
    def from_config(cls, config: Any, **kwargs) -> 'FromConfigBase':
        """
        MANDATORY class method for creating instances from configuration
        
        ALL framework components must implement this method.
        
        Args:
            config: Component configuration (StepConfig, etc.)
            **kwargs: Framework-provided dependencies
            
        Returns:
            Configured component instance
            
        Raises:
            ComponentConfigurationError: Invalid configuration
            ComponentDependencyError: Dependency resolution failed
        """
        pass
    
    @classmethod
    def validate_config_schema(cls, config: Any) -> None:
        """Validate that configuration contains required fields"""
        missing_fields = []
        for field in cls.REQUIRED_CONFIG_FIELDS:
            if not hasattr(config, field):
                missing_fields.append(field)
        
        if missing_fields:
            raise ComponentConfigurationError(
                f"{cls.__name__} missing required configuration fields: {missing_fields}"
            )
    
    @classmethod
    @abstractmethod
    def extract_component_config(cls, config: Any) -> Dict[str, Any]:
        """Extract component-specific configuration from framework config"""
        pass
    
    @classmethod  
    @abstractmethod
    def resolve_dependencies(cls, component_config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Resolve all component dependencies"""
        pass
    
    @classmethod
    def create_instance(cls, config: Any, component_config: Dict[str, Any], 
                       dependencies: Dict[str, Any]) -> 'FromConfigBase':
        """Create instance with resolved configuration and dependencies"""
        # Use __new__ to bypass __init__
        instance = cls.__new__(cls)
        instance._init_from_config(config, component_config, dependencies)
        return instance
    
    @abstractmethod
    def _init_from_config(self, config: Any, component_config: Dict[str, Any],
                         dependencies: Dict[str, Any]) -> None:
        """Initialize instance with resolved dependencies"""
        pass
    
    def _post_config_initialization(self) -> None:
        """Post-configuration initialization hook (override in subclasses if needed)"""
        pass
    
    # PREVENT direct instantiation
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            f"Direct instantiation of {self.__class__.__name__} is prohibited. "
            f"ALL framework components must use {self.__class__.__name__}.from_config() "
            f"as per mandatory framework requirements."
        ) 