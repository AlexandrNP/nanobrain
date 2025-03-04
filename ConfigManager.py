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
        
    def get_config(self, class_dir: str) -> Dict:
        """
        Looks for <class_name>.yml in the specified directory and returns parameter dictionary.
        Each parameter is supplemented with the property 'type' that references the class name.
        
        Biological analogy: Cellular response to environmental cues.
        Justification: Like how cells read their environment to determine appropriate
        protein expression, components read configuration files to determine behavior.
        """
        class_name = self.__class__.__name__
        config_path = os.path.join(class_dir, f"{class_name}.yml")
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as file:
                self._config = yaml.safe_load(file)
            return self._config
        else:
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