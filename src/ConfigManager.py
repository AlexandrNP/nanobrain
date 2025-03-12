import os
import sys
import yaml
import importlib
import inspect
from typing import Dict, Any, Type, Optional

# Ensure the project root is in the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now we can import from interfaces
try:
    from src.interfaces import IConfigurable
except ImportError:
    # Fall back to relative import
    from .interfaces import IConfigurable


class ConfigManager(IConfigurable):
    """
    Handles configuration via YAML files and creates class instances.
    
    Biological analogy: Epigenetic mechanisms that control gene expression.
    Justification: Like how epigenetic modifications determine which genes are expressed
    without changing the DNA sequence, configuration parameters determine component
    behavior without changing the underlying code.
    """
    def __init__(self, **kwargs):
        self._config = {}
        self._adaptability = 0.5  # Ability to reconfigure (0.0-1.0)
        self._base_path = kwargs.get('base_path', '')
        self._class_cache = {}  # Cache for loaded classes
        
    def get_config(self, class_name: str) -> Dict:
        """
        Looks for <class_name>.yml in the appropriate directory and returns parameter dictionary.
        First checks in local 'config' directory relative to base_path, then falls back to 'default_configs'.
        Each parameter is supplemented with the property 'type' that references the class name.
        
        Biological analogy: Cellular response to environmental cues.
        Justification: Like how cells read their environment to determine appropriate
        protein expression, components read configuration files to determine behavior.
        """
        # First try local config directory if base_path is provided
        if self._base_path:
            local_config_dir = os.path.join(os.path.dirname(self._base_path), 'config')
            local_config_path = os.path.join(local_config_dir, f"{class_name}.yml")
            
            if os.path.exists(local_config_path):
                with open(local_config_path, 'r') as file:
                    self._config = yaml.safe_load(file)
                return self._config
        
        # Fall back to default_configs
        default_config_path = os.path.join('default_configs', f"{class_name}.yml")
        if os.path.exists(default_config_path):
            with open(default_config_path, 'r') as file:
                self._config = yaml.safe_load(file)
            return self._config
            
        return {}
            
    def update_config(self, updates: Dict, adaptability_threshold: float = 0.3) -> bool:
        """
        Updates configuration if adaptability threshold is met.
        
        Biological analogy: Cellular plasticity - ability to change in response to stimuli.
        Justification: Components with higher adaptability should be more responsive to
        configuration changes, similar to how more plastic neural circuits adapt more readily.
        """
        if self._adaptability >= adaptability_threshold:
            self._config.update(updates)
            return True
        return False
    
    def create_instance(self, class_name: str, **kwargs) -> Any:
        """
        Factory method that creates an instance of the specified class using configuration.
        First looks for a config file, then loads the class from a .py file with the same name,
        and finally creates an instance with the config parameters.
        
        Biological analogy: Protein synthesis from genetic instructions.
        Justification: Like how cells synthesize proteins based on DNA templates modified
        by epigenetic factors, this method creates objects based on class definitions
        modified by configuration parameters.
        
        Args:
            class_name: Name of the class to instantiate
            **kwargs: Additional parameters to override config values
            
        Returns:
            An instance of the specified class
            
        Raises:
            ImportError: If the class module cannot be found
            AttributeError: If the class cannot be found in the module
            TypeError: If the class cannot be instantiated with the given parameters
        """
        # Get configuration for the class
        config = self.get_config(class_name)
        
        # Load the class if not already cached
        cls = self._get_class(class_name)
        if not cls:
            raise ImportError(f"Could not load class {class_name}")
            
        # Merge config with kwargs, with kwargs taking precedence
        merged_config = {**config, **kwargs}
        
        # Special case for Agent class - remove prompt_template if it's not a string
        if class_name == "Agent" and "prompt_template" in merged_config:
            prompt_template = merged_config.get("prompt_template")
            if not isinstance(prompt_template, str) and prompt_template is not None:
                del merged_config["prompt_template"]
        
        # Handle executor if specified as string (class name)
        if "executor" in merged_config and isinstance(merged_config["executor"], str):
            executor_class_name = merged_config["executor"]
            # Create executor instance if it's not already a class instance
            if executor_class_name:
                try:
                    # Use ExecutorFunc as default if not specified
                    if executor_class_name.lower() == "none" or executor_class_name.lower() == "null":
                        executor_class_name = "ExecutorFunc"
                    
                    # Create executor instance using factory method (recursive call)
                    merged_config["executor"] = self.create_instance(executor_class_name)
                except Exception as e:
                    print(f"Warning: Could not create executor instance of type {executor_class_name}: {str(e)}")
                    # Fall back to ExecutorFunc
                    from src.ExecutorFunc import ExecutorFunc
                    merged_config["executor"] = ExecutorFunc()
        
        # Ensure executor is provided for Runnable classes if needed
        if "executor" not in merged_config:
            # Try to determine if this class requires an executor
            try:
                if cls.__name__ in ["Agent", "Step", "Workflow", "Runner", "Router"]:
                    # These classes typically need an executor
                    from src.ExecutorFunc import ExecutorFunc
                    merged_config["executor"] = ExecutorFunc()
            except Exception:
                # Ignore errors, just don't add an executor
                pass
                
        # Create instance with merged configuration
        try:
            # If the class accepts a config_manager parameter, pass self
            if 'config_manager' in inspect.signature(cls.__init__).parameters:
                merged_config['config_manager'] = self
            
            instance = cls(**merged_config)
            
            # If instance is configurable, update its config
            if isinstance(instance, IConfigurable):
                instance.update_config(merged_config)
                
            return instance
        except Exception as e:
            raise TypeError(f"Could not create instance of {class_name}: {str(e)}")
    
    def _get_class(self, class_name: str) -> Optional[Type]:
        """
        Get the class object for a given class name.
        First checks cache, then tries to import from src/, builder/, and tools_common/ directories.
        
        Args:
            class_name: Name of the class to find
            
        Returns:
            The class object or None if not found
        """
        # Check cache first
        if class_name in self._class_cache:
            return self._class_cache[class_name]
        
        # List of possible module paths to try
        module_paths = [
            f"src.{class_name}",
            f"builder.{class_name}",
            f"tools_common.{class_name}",
            class_name  # Try direct import
        ]
        
        # Try each module path
        for module_path in module_paths:
            try:
                # Try to import the module
                module = importlib.import_module(module_path)
                
                # Look for the class in the module
                if hasattr(module, class_name):
                    # Get the class
                    cls = getattr(module, class_name)
                    
                    # Verify it's a class
                    if inspect.isclass(cls):
                        # Cache the class
                        self._class_cache[class_name] = cls
                        return cls
            except ImportError:
                # Try the next path
                continue
        
        # If we get here, we couldn't find the class
        return None
        
    @property
    def adaptability(self) -> float:
        """Get the adaptability level of the component."""
        return self._adaptability
        
    @adaptability.setter
    def adaptability(self, value: float):
        """Set the adaptability level of the component."""
        self._adaptability = max(0.0, min(1.0, value))  # Clamp between 0 and 1 