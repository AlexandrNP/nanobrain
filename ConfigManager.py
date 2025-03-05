import os
import yaml
from typing import Dict
from interfaces import IConfigurable


class ConfigManager(IConfigurable):
    """
    Handles configuration via YAML files.
    
    Biological analogy: Epigenetic mechanisms that control gene expression.
    Justification: Like how epigenetic modifications determine which genes are expressed
    without changing the DNA sequence, configuration parameters determine component
    behavior without changing the underlying code.
    """
    def __init__(self, **kwargs):
        self._config = {}
        self._adaptability = 0.5  # Ability to reconfigure (0.0-1.0)
        self._base_path = kwargs.get('base_path', '')
        
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
        
    @property
    def adaptability(self) -> float:
        """Get the adaptability level of the component."""
        return self._adaptability
        
    @adaptability.setter
    def adaptability(self, value: float):
        """Set the adaptability level of the component."""
        self._adaptability = max(0.0, min(1.0, value))  # Clamp between 0 and 1 