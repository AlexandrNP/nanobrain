"""
Component Factory for NanoBrain Framework

Simplified factory using direct import path resolution with from_config pattern.
"""

import logging
import importlib
from typing import Any, Dict, Union, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


def import_and_create_from_config(class_path: str, config: Any, **kwargs) -> Any:
    """
    Universal component creation using from_config pattern
    
    NO HARDCODED VALUES OR MAPPINGS
    - Every component must have from_config class method
    - Configuration determines behavior entirely
    - No special cases or conditional logic
    """
    from ..logging_system import get_logger
    logger = get_logger("component_factory")
    
    # Basic validation
    if not class_path or '.' not in class_path:
        raise ValueError(f"Invalid class path: {class_path}. Must use full module.Class format")
    
    # Import the specified class
    try:
        module_path, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        component_class = getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Failed to import component '{class_path}': {e}")
    
    # Verify class has from_config method
    if not hasattr(component_class, 'from_config'):
        raise AttributeError(
            f"Component '{class_path}' does not implement required from_config method. "
            f"All NanoBrain components must implement from_config pattern."
        )
    
    # Create instance using from_config - NO PREPROCESSING
    try:
        instance = component_class.from_config(config, **kwargs)
        logger.debug(f"Created {class_path} via from_config")
        return instance
    except Exception as e:
        from ..component_base import ComponentConfigurationError
        raise ComponentConfigurationError(
            f"Failed to create '{class_path}' via from_config: {e}"
        )


# REMOVED: _convert_dict_to_config_object function - NO PREPROCESSING
# Components handle their own configuration conversion in from_config

def validate_component_config(config: Dict[str, Any]) -> None:
    """
    Validate component configuration has required fields
    NO HARDCODED VALIDATION - only structural checks
    """
    from pydantic import BaseModel
    
    if not isinstance(config, (dict, BaseModel)):
        raise ValueError("Configuration must be dict or BaseModel")
    
    # Only validate that class field exists for dict configs
    if isinstance(config, dict) and 'class' not in config:
        raise ValueError("Component configuration must specify 'class' field")


# REMOVED ComponentFactory class - use functions directly

def load_config_file(config_path: str) -> Dict[str, Any]:
    """Load configuration from file - NO MODIFICATIONS"""
    import yaml
    
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
        
    if not isinstance(config, dict):
        raise ValueError(f"Configuration file must contain a dictionary: {config_path}")
    
    # Return AS-IS - no modifications or hardcoded additions
    return config


# Component registry helpers (NO HARDCODING)
class ComponentRegistry:
    """
    Registry for component classes - NO HARDCODED MAPPINGS
    Components register themselves, no predefined mappings
    """
    _registry: Dict[str, type] = {}
    
    @classmethod
    def register(cls, component_class: type) -> None:
        """Components self-register - no hardcoded registration"""
        if hasattr(component_class, '__module__') and hasattr(component_class, '__name__'):
            full_path = f"{component_class.__module__}.{component_class.__name__}"
            cls._registry[full_path] = component_class
    
    @classmethod
    def get(cls, class_path: str) -> Optional[type]:
        """Get registered component - no fallbacks or defaults"""
        return cls._registry.get(class_path)
    
    @classmethod
    def clear(cls) -> None:
        """Clear registry - used for testing"""
        cls._registry.clear() 