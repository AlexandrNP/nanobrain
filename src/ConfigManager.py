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
        
        # Extract defaults from config
        defaults = config.get('defaults', {})
        
        # Update with provided kwargs
        params = {**defaults, **kwargs}
        
        # Get the class
        cls = self._get_class(class_name)
        if not cls:
            raise ImportError(f"Could not find class '{class_name}'")
        
        # Create and return the instance
        return cls(**params)
    
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