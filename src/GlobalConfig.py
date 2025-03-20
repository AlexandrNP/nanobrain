"""
Global Configuration Manager for NanoBrain

This module provides a singleton class for managing global configuration settings
such as API keys, model defaults, and framework settings.

Biological analogy: Endocrine system.
Justification: Like how the endocrine system regulates global body functions through
hormones that affect multiple systems, the GlobalConfig manages framework-wide
settings that affect multiple components.
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from src.DirectoryTracer import DirectoryTracer
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GlobalConfig:
    """
    Singleton class for managing global configuration settings.
    
    Biological analogy: Endocrine system.
    Justification: Like how the endocrine system regulates global body functions through
    hormones that affect multiple systems, the GlobalConfig manages framework-wide
    settings that affect multiple components.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GlobalConfig, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._config = {}
        self._config_path = None
        self._initialized = True
        self._env_vars_loaded = False
        self.tracer = DirectoryTracer(self.__class__.__module__)
    
    def load_config(self, config_path: Optional[str] = None) -> bool:
        """
        Load configuration from a YAML file.
        
        Args:
            config_path: Path to the configuration file. If None, uses default locations.
            
        Returns:
            bool: True if configuration was loaded successfully, False otherwise.
        """
        # If no path is provided, try default locations
        if config_path is None:
            # Try to find the config file in standard locations
            possible_paths = [
                "config.yml",  # Current directory
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yml"),  # Project root
                os.path.expanduser("~/.nanobrain/config.yml"),  # User home directory
                "/etc/nanobrain/config.yml"  # System-wide
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    config_path = path
                    break
        
        if config_path is None or not os.path.exists(config_path):
            logger.warning(f"Configuration file not found. Using default configuration.")
            self._config = self._get_default_config()
            return False
        
        try:
            with open(config_path, 'r') as f:
                self._config = yaml.safe_load(f)
            self._config_path = config_path
            logger.info(f"Configuration loaded from {config_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            self._config = self._get_default_config()
            return False
    
    def save_config(self, config_path: Optional[str] = None) -> bool:
        """
        Save the current configuration to a YAML file.
        
        Args:
            config_path: Path to save the configuration file. If None, uses the path
                         from which the configuration was loaded.
                         
        Returns:
            bool: True if configuration was saved successfully, False otherwise.
        """
        if config_path is None:
            config_path = self._config_path
        
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yml")
        
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
            
            with open(config_path, 'w') as f:
                yaml.dump(self._config, f, default_flow_style=False)
            
            self._config_path = config_path
            logger.info(f"Configuration saved to {config_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False
    
    def load_from_env(self) -> None:
        """
        Load configuration values from environment variables.
        
        Environment variables take precedence over values in the config file.
        The naming convention is NANOBRAIN_SECTION_KEY, e.g., NANOBRAIN_API_KEYS_OPENAI.
        """
        if self._env_vars_loaded:
            return
        
        # Load API keys from environment variables
        for provider in ['openai', 'anthropic', 'google', 'mistral', 'huggingface']:
            env_var = f"NANOBRAIN_API_KEYS_{provider.upper()}"
            if env_var in os.environ:
                self.set(['api_keys', provider], os.environ[env_var])
        
        # Load other common settings
        if 'NANOBRAIN_MODELS_DEFAULT' in os.environ:
            self.set(['models', 'default'], os.environ['NANOBRAIN_MODELS_DEFAULT'])
        
        if 'NANOBRAIN_FRAMEWORK_LOG_LEVEL' in os.environ:
            self.set(['framework', 'log_level'], os.environ['NANOBRAIN_FRAMEWORK_LOG_LEVEL'])
        
        if 'NANOBRAIN_DEVELOPMENT_DEBUG' in os.environ:
            self.set(['development', 'debug'], os.environ['NANOBRAIN_DEVELOPMENT_DEBUG'].lower() == 'true')
        
        self._env_vars_loaded = True
    
    def get(self, path: Union[str, list], default: Any = None) -> Any:
        """
        Get a configuration value by path.
        
        Args:
            path: Path to the configuration value, either as a dot-separated string
                 or a list of keys.
            default: Default value to return if the path is not found.
            
        Returns:
            The configuration value, or the default if not found.
        """
        if isinstance(path, str):
            path = path.split('.')
        
        value = self._config
        try:
            for key in path:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, path: Union[str, list], value: Any) -> None:
        """
        Set a configuration value by path.
        
        Args:
            path: Path to the configuration value, either as a dot-separated string
                 or a list of keys.
            value: Value to set.
        """
        if isinstance(path, str):
            path = path.split('.')
        
        config = self._config
        for key in path[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[path[-1]] = value
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """
        Get an API key for a specific provider.
        
        Args:
            provider: The provider name (e.g., 'openai', 'anthropic').
            
        Returns:
            The API key, or None if not found.
        """
        return self.get(['api_keys', provider.lower()])
    
    def set_api_key(self, provider: str, key: str) -> None:
        """
        Set an API key for a specific provider.
        
        Args:
            provider: The provider name (e.g., 'openai', 'anthropic').
            key: The API key.
        """
        self.set(['api_keys', provider.lower()], key)
    
    def setup_environment(self) -> None:
        """
        Set up the environment based on the configuration.
        
        This includes setting environment variables for API keys and configuring logging.
        """
        # Set API keys as environment variables for libraries that expect them
        for provider, key in self.get('api_keys', {}).items():
            if key:  # Only set if the key is not empty
                if provider == 'openai':
                    os.environ['OPENAI_API_KEY'] = key
                elif provider == 'anthropic':
                    os.environ['ANTHROPIC_API_KEY'] = key
                elif provider == 'google':
                    os.environ['GOOGLE_API_KEY'] = key
                elif provider == 'mistral':
                    os.environ['MISTRAL_API_KEY'] = key
                elif provider == 'huggingface':
                    os.environ['HUGGINGFACE_API_TOKEN'] = key
        
        # Configure logging
        log_level = self.get('framework.log_level', 'INFO')
        numeric_level = getattr(logging, log_level.upper(), None)
        if isinstance(numeric_level, int):
            logging.getLogger().setLevel(numeric_level)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get the default configuration.
        
        Returns:
            The default configuration dictionary.
        """
        return {
            'api_keys': {
                'openai': '',
                'anthropic': '',
                'google': '',
                'mistral': '',
                'huggingface': ''
            },
            'models': {
                'default': 'gpt-3.5-turbo',
                'use_mock_in_testing': True,
                'default_parameters': {
                    'temperature': 0.7,
                    'max_tokens': 1000,
                    'top_p': 1.0
                }
            },
            'framework': {
                'log_level': 'INFO',
                'temp_dir': '.nanobrain_temp',
                'enable_telemetry': False
            },
            'development': {
                'debug': False,
                'verbose': False
            }
        }
    
    @property
    def config(self) -> Dict[str, Any]:
        """
        Get the entire configuration dictionary.
        
        Returns:
            The configuration dictionary.
        """
        return self._config
    
    @property
    def config_path(self) -> Optional[str]:
        """
        Get the path to the configuration file.
        
        Returns:
            The path to the configuration file, or None if not loaded from a file.
        """
        return self._config_path 