"""
Component Factory for NanoBrain Framework

Simplified factory using direct import path resolution with from_config pattern.
"""

import logging
import importlib
from typing import Any, Dict, Union
from pathlib import Path

logger = logging.getLogger(__name__)


def import_and_create_from_config(class_path: str, config: Any, **kwargs) -> Any:
    """
    Import class and create instance using from_config pattern.
    
    Args:
        class_path: Full import path (e.g., 'module.submodule.ClassName')
        config: Component configuration (dict or config object)
        **kwargs: Additional dependencies for from_config
        
    Returns:
        Component instance created via from_config
        
    Raises:
        ValueError: If class_path is not a full import path or class doesn't implement from_config
        ImportError: If class cannot be imported
    """
    if '.' not in class_path:
        raise ValueError(
            f"Class path '{class_path}' must be a full import path. "
            f"Short class names are no longer supported. "
            f"Use format: 'module.submodule.ClassName'"
        )
    
    try:
        # Direct import - only supported method
        module_path, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        component_class = getattr(module, class_name)
        
        # Validate from_config implementation
        if not hasattr(component_class, 'from_config'):
            raise ValueError(
                f"Class '{class_path}' must implement from_config method. "
                f"All framework components must use the from_config pattern."
            )
        
        # Convert dict to appropriate config object if needed
        if isinstance(config, dict):
            config = _convert_dict_to_config_object(class_path, config)
        
        # Create instance via from_config
        logger.debug(f"Creating {class_path} via from_config pattern")
        instance = component_class.from_config(config, **kwargs)
        
        logger.info(f"Successfully created {class_path} via from_config")
        return instance
        
    except ImportError as e:
        raise ImportError(f"Cannot import module '{module_path}': {e}")
    except AttributeError as e:
        raise ImportError(f"Class '{class_name}' not found in module '{module_path}': {e}")
    except Exception as e:
        raise ValueError(f"Failed to create component '{class_path}': {e}")


def _convert_dict_to_config_object(class_path: str, config_dict: Dict[str, Any]) -> Any:
    """
    Convert dictionary configuration to appropriate config object based on component type.
    
    Args:
        class_path: Full import path to determine config type
        config_dict: Dictionary configuration
        
    Returns:
        Appropriate config object
    """
    # Determine config class based on component class path
    if 'executor' in class_path.lower():
        from nanobrain.core.executor import ExecutorConfig
        return ExecutorConfig(**config_dict)
    elif 'agent' in class_path.lower():
        from nanobrain.core.agent import AgentConfig
        return AgentConfig(**config_dict)
    elif 'step' in class_path.lower():
        from nanobrain.core.step import StepConfig
        return StepConfig(**config_dict)
    elif 'workflow' in class_path.lower():
        from nanobrain.core.workflow import WorkflowConfig
        return WorkflowConfig(**config_dict)
    elif 'data_unit' in class_path.lower():
        from nanobrain.core.data_unit import DataUnitConfig
        return DataUnitConfig(**config_dict)
    elif 'trigger' in class_path.lower():
        from nanobrain.core.trigger import TriggerConfig
        return TriggerConfig(**config_dict)
    elif 'link' in class_path.lower():
        from nanobrain.core.link import LinkConfig
        return LinkConfig(**config_dict)
    else:
        # If we can't determine the type, try to import the config class
        # by convention (ComponentName -> ComponentConfig)
        try:
            module_path, class_name = class_path.rsplit('.', 1)
            config_class_name = class_name + "Config"
            module = importlib.import_module(module_path)
            config_class = getattr(module, config_class_name)
            return config_class(**config_dict)
        except (ImportError, AttributeError):
            # Fall back to returning the dict - let the component handle it
            logger.warning(f"Could not determine config class for {class_path}, passing dict directly")
            return config_dict


class ComponentFactory:
    """
    Simplified factory for creating NanoBrain components using direct import paths.
    
    This factory enforces the modern from_config pattern and eliminates backward
    compatibility complexity. All components must use full import paths and 
    implement the from_config method.
    """
    
    def __init__(self):
        """Initialize the simplified component factory."""
        self.logger = logging.getLogger(__name__ + ".ComponentFactory")
        self.logger.debug("Simplified ComponentFactory initialized")
    
    def create_component_from_config(self, class_path: str, config: Any, **kwargs) -> Any:
        """
        Create component using direct import path and from_config pattern.
        
        Args:
            class_path: Full import path to component class
            config: Component configuration (dict or config object)
            **kwargs: Additional dependencies for from_config
            
        Returns:
            Component instance
        """
        return import_and_create_from_config(class_path, config, **kwargs)
    
    def create_from_yaml_file(self, yaml_path: Union[str, Path], class_path: str, **kwargs) -> Any:
        """
        Create component from YAML configuration file.
        
        Args:
            yaml_path: Path to YAML configuration file
            class_path: Full import path to component class
            **kwargs: Additional dependencies for from_config
            
        Returns:
            Component instance
        """
        import yaml
        
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Convert dict to appropriate config object based on component type
        return self.create_component_from_config(class_path, config_dict, **kwargs)


# Backwards compatibility note: Legacy global functions removed
# Use ComponentFactory().create_component_from_config() or import_and_create_from_config() directly
# For workflow creation, use direct: WorkflowClass.from_config(workflow_config) pattern 