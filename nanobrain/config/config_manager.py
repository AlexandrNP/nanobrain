"""
NanoBrain Global Configuration Manager

This module handles loading and managing global configuration for the NanoBrain framework,
including API keys for commercial AI models and environment variable management.
"""

import os
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import re


@dataclass
class ProviderConfig:
    """Configuration for a specific AI provider."""
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: int = 60
    max_retries: int = 3
    retry_delay: float = 1.0
    additional_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.additional_params is None:
            self.additional_params = {}


class ConfigManager:
    """
    Global configuration manager for NanoBrain framework.
    
    Handles loading YAML configuration, environment variable substitution,
    and setting up API keys for commercial AI models.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration YAML file. If None, uses default location.
        """
        self.logger = logging.getLogger(__name__)
        self._config_path = config_path or self._get_default_config_path()
        self._config: Dict[str, Any] = {}
        self._providers: Dict[str, ProviderConfig] = {}
        self._loaded = False
        
    def _get_default_config_path(self) -> str:
        """Get the default configuration file path."""
        # Look for config in the project root config directory
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent  # Go up from src/config to project root
        return str(project_root / "config" / "global_config.yml")
    
    def load_config(self, force_reload: bool = False) -> None:
        """
        Load configuration from YAML file.
        
        Args:
            force_reload: Whether to force reload even if already loaded.
        """
        if self._loaded and not force_reload:
            return
            
        try:
            if not os.path.exists(self._config_path):
                self.logger.warning(f"Configuration file not found: {self._config_path}")
                self._config = self._get_default_config()
            else:
                with open(self._config_path, 'r', encoding='utf-8') as file:
                    raw_config = yaml.safe_load(file)
                    self._config = self._substitute_env_variables(raw_config)
                    
                if self._config.get('logging', {}).get('log_config_loading', True):
                    self.logger.info(f"Configuration loaded from: {self._config_path}")
                    
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            self._config = self._get_default_config()
            
        self._setup_providers()
        self._setup_environment_variables()
        self._loaded = True
        
        # Reconfigure logging system now that configuration is loaded
        # Import locally to avoid circular imports
        try:
            import importlib
            logging_module = importlib.import_module('core.logging_system')
            if hasattr(logging_module, 'reconfigure_global_logging'):
                logging_module.reconfigure_global_logging()
        except (ImportError, AttributeError):
            # If logging system not available or function not found, skip reconfiguration
            pass
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration when file is not available."""
        return {
            'framework': {
                'name': 'NanoBrain',
                'version': '1.0.0',
                'description': 'Event-driven AI workflow framework'
            },
            'api_keys': {},
            'default_models': {},
            'providers': {},
            'env_mappings': {},
            'security': {
                'validate_keys_on_startup': False,
                'log_key_validation': True,
                'require_valid_key': False,
                'mask_keys_in_logs': True
            },
            'logging': {
                'level': 'INFO',
                'log_config_loading': True,
                'log_env_loading': False,
                'log_missing_keys': True
            },
            'development': {
                'use_mock_clients': True,
                'validate_schema': True,
                'allow_env_override': True
            }
        }
    
    def _substitute_env_variables(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Substitute environment variables in configuration values.
        
        Supports ${VAR_NAME} syntax for environment variable substitution.
        """
        def substitute_value(value):
            if isinstance(value, str):
                # Find all ${VAR_NAME} patterns
                pattern = r'\$\{([^}]+)\}'
                matches = re.findall(pattern, value)
                
                for var_name in matches:
                    env_value = os.getenv(var_name, '')
                    value = value.replace(f'${{{var_name}}}', env_value)
                    
                return value
            elif isinstance(value, dict):
                return {k: substitute_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [substitute_value(item) for item in value]
            else:
                return value
                
        return substitute_value(config)
    
    def _setup_providers(self) -> None:
        """Setup provider configurations from loaded config."""
        api_keys = self._config.get('api_keys', {})
        provider_settings = self._config.get('providers', {})
        
        for provider_name, key_config in api_keys.items():
            settings = provider_settings.get(provider_name, {})
            
            # Extract API key and additional parameters
            if isinstance(key_config, dict):
                api_key = key_config.get('api_key', '')
                additional_params = {k: v for k, v in key_config.items() if k != 'api_key'}
            else:
                api_key = str(key_config) if key_config else ''
                additional_params = {}
            
            # Create provider config
            self._providers[provider_name] = ProviderConfig(
                api_key=api_key if api_key else None,
                base_url=additional_params.get('base_url', settings.get('base_url')),
                timeout=settings.get('timeout', 60),
                max_retries=settings.get('max_retries', 3),
                retry_delay=settings.get('retry_delay', 1.0),
                additional_params=additional_params
            )
    
    def _setup_environment_variables(self) -> None:
        """Setup environment variables based on configuration."""
        env_mappings = self._config.get('env_mappings', {})
        api_keys = self._config.get('api_keys', {})
        log_env_loading = self._config.get('logging', {}).get('log_env_loading', False)
        
        for provider_name, env_vars in env_mappings.items():
            provider_config = api_keys.get(provider_name, {})
            
            if not isinstance(provider_config, dict):
                continue
                
            for env_var in env_vars:
                # Map common environment variable names to config keys
                config_key = self._map_env_var_to_config_key(env_var)
                
                if config_key in provider_config:
                    value = provider_config[config_key]
                    if value and not os.getenv(env_var):
                        os.environ[env_var] = str(value)
                        if log_env_loading:
                            masked_value = self._mask_sensitive_value(str(value))
                            self.logger.debug(f"Set {env_var} = {masked_value}")
    
    def _map_env_var_to_config_key(self, env_var: str) -> str:
        """Map environment variable names to configuration keys."""
        mapping = {
            'OPENAI_API_KEY': 'api_key',
            'OPENAI_ORG_ID': 'organization',
            'OPENAI_BASE_URL': 'base_url',
            'ANTHROPIC_API_KEY': 'api_key',
            'ANTHROPIC_BASE_URL': 'base_url',
            'GOOGLE_AI_API_KEY': 'api_key',
            'GOOGLE_PROJECT_ID': 'project_id',
            'AZURE_OPENAI_API_KEY': 'api_key',
            'AZURE_OPENAI_ENDPOINT': 'endpoint',
            'AZURE_OPENAI_API_VERSION': 'api_version',
            'COHERE_API_KEY': 'api_key',
            'COHERE_BASE_URL': 'base_url',
            'HUGGINGFACE_API_KEY': 'api_key',
            'HUGGINGFACE_BASE_URL': 'base_url',
            'REPLICATE_API_TOKEN': 'api_key',
            'REPLICATE_BASE_URL': 'base_url',
            'TOGETHER_API_KEY': 'api_key',
            'TOGETHER_BASE_URL': 'base_url',
            'MISTRAL_API_KEY': 'api_key',
            'MISTRAL_BASE_URL': 'base_url',
        }
        return mapping.get(env_var, env_var.lower())
    
    def _mask_sensitive_value(self, value: str) -> str:
        """Mask sensitive values for logging."""
        if not self._config.get('security', {}).get('mask_keys_in_logs', True):
            return value
            
        if len(value) <= 8:
            return '*' * len(value)
        else:
            return f"{value[:4]}{'*' * (len(value) - 8)}{value[-4:]}"
    
    def get_provider_config(self, provider_name: str) -> Optional[ProviderConfig]:
        """
        Get configuration for a specific provider.
        
        Args:
            provider_name: Name of the AI provider (e.g., 'openai', 'anthropic').
            
        Returns:
            ProviderConfig object or None if provider not found.
        """
        if not self._loaded:
            self.load_config()
        return self._providers.get(provider_name)
    
    def get_api_key(self, provider_name: str) -> Optional[str]:
        """
        Get API key for a specific provider.
        
        Args:
            provider_name: Name of the AI provider.
            
        Returns:
            API key string or None if not found.
        """
        provider_config = self.get_provider_config(provider_name)
        return provider_config.api_key if provider_config else None
    
    def get_default_model(self, provider_name: str, model_type: str = 'chat') -> Optional[str]:
        """
        Get default model for a provider and model type.
        
        Args:
            provider_name: Name of the AI provider.
            model_type: Type of model (e.g., 'chat', 'text_generation', 'embeddings').
            
        Returns:
            Model name string or None if not found.
        """
        if not self._loaded:
            self.load_config()
            
        default_models = self._config.get('default_models', {})
        model_type_config = default_models.get(model_type, {})
        return model_type_config.get(provider_name)
    
    def get_available_providers(self) -> List[str]:
        """
        Get list of available providers with valid API keys.
        
        Returns:
            List of provider names.
        """
        if not self._loaded:
            self.load_config()
            
        return [
            name for name, config in self._providers.items()
            if config.api_key
        ]
    
    def validate_api_keys(self) -> Dict[str, bool]:
        """
        Validate API keys for all providers.
        
        Returns:
            Dictionary mapping provider names to validation status.
        """
        if not self._loaded:
            self.load_config()
            
        validation_results = {}
        
        for provider_name, config in self._providers.items():
            if config.api_key:
                # Basic validation - check if key is not empty and has reasonable length
                is_valid = len(config.api_key.strip()) >= 10
                validation_results[provider_name] = is_valid
                
                if self._config.get('security', {}).get('log_key_validation', True):
                    status = "valid" if is_valid else "invalid"
                    masked_key = self._mask_sensitive_value(config.api_key)
                    self.logger.info(f"API key for {provider_name}: {status} ({masked_key})")
            else:
                validation_results[provider_name] = False
                if self._config.get('logging', {}).get('log_missing_keys', True):
                    self.logger.warning(f"No API key found for {provider_name}")
        
        return validation_results
    
    def get_framework_info(self) -> Dict[str, str]:
        """
        Get framework information.
        
        Returns:
            Dictionary with framework name, version, and description.
        """
        if not self._loaded:
            self.load_config()
        return self._config.get('framework', {})
    
    def is_development_mode(self) -> bool:
        """
        Check if framework is running in development mode.
        
        Returns:
            True if development mode is enabled.
        """
        if not self._loaded:
            self.load_config()
        return self._config.get('development', {}).get('use_mock_clients', False)
    
    def reload_config(self) -> None:
        """Reload configuration from file."""
        self.load_config(force_reload=True)
    
    def get_config_dict(self) -> Dict[str, Any]:
        """
        Get the full configuration dictionary.
        
        Returns:
            Complete configuration dictionary.
        """
        if not self._loaded:
            self.load_config()
        return self._config.copy()
    
    def get_logging_config(self) -> Dict[str, Any]:
        """
        Get logging configuration settings.
        
        Returns:
            Dictionary containing logging configuration.
        """
        if not self._loaded:
            self.load_config()
        return self._config.get('logging', {})
    
    def get_logging_mode(self) -> str:
        """
        Get the logging mode setting.
        
        Returns:
            Logging mode: "console", "file", or "both" (default: "both")
        """
        logging_config = self.get_logging_config()
        return logging_config.get('mode', 'both')
    
    def should_log_to_console(self) -> bool:
        """
        Check if logging should go to console.
        
        Returns:
            True if console logging is enabled.
        """
        mode = self.get_logging_mode()
        return mode in ['console', 'both']
    
    def should_log_to_file(self) -> bool:
        """
        Check if logging should go to file.
        
        Returns:
            True if file logging is enabled.
        """
        mode = self.get_logging_mode()
        return mode in ['file', 'both']
    
    def get_log_file_config(self) -> Dict[str, Any]:
        """
        Get file logging configuration.
        
        Returns:
            Dictionary containing file logging settings.
        """
        logging_config = self.get_logging_config()
        return logging_config.get('file', {})
    
    def get_console_log_config(self) -> Dict[str, Any]:
        """
        Get console logging configuration.
        
        Returns:
            Dictionary containing console logging settings.
        """
        logging_config = self.get_logging_config()
        return logging_config.get('console', {})


# Global configuration manager instance
_global_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """
    Get the global configuration manager instance.
    
    Returns:
        Global ConfigManager instance.
    """
    global _global_config_manager
    if _global_config_manager is None:
        _global_config_manager = ConfigManager()
        _global_config_manager.load_config()
    return _global_config_manager


def get_api_key(provider_name: str) -> Optional[str]:
    """
    Convenience function to get API key for a provider.
    
    Args:
        provider_name: Name of the AI provider.
        
    Returns:
        API key string or None if not found.
    """
    return get_config_manager().get_api_key(provider_name)


def get_provider_config(provider_name: str) -> Optional[ProviderConfig]:
    """
    Convenience function to get provider configuration.
    
    Args:
        provider_name: Name of the AI provider.
        
    Returns:
        ProviderConfig object or None if not found.
    """
    return get_config_manager().get_provider_config(provider_name)


def get_default_model(provider_name: str, model_type: str = 'chat') -> Optional[str]:
    """
    Convenience function to get default model for a provider.
    
    Args:
        provider_name: Name of the AI provider.
        model_type: Type of model.
        
    Returns:
        Model name string or None if not found.
    """
    return get_config_manager().get_default_model(provider_name, model_type)


def initialize_config(config_path: Optional[str] = None) -> None:
    """
    Initialize the global configuration manager with a specific config file.
    
    Args:
        config_path: Path to the configuration YAML file.
    """
    global _global_config_manager
    _global_config_manager = ConfigManager(config_path)
    _global_config_manager.load_config()


def get_logging_mode() -> str:
    """
    Get the global logging mode setting.
    
    Returns:
        Logging mode: "console", "file", or "both"
    """
    config_manager = get_config_manager()
    return config_manager.get_logging_mode()


def should_log_to_console() -> bool:
    """
    Check if logging should go to console based on global configuration.
    
    Returns:
        True if console logging is enabled.
    """
    config_manager = get_config_manager()
    return config_manager.should_log_to_console()


def should_log_to_file() -> bool:
    """
    Check if logging should go to file based on global configuration.
    
    Returns:
        True if file logging is enabled.
    """
    config_manager = get_config_manager()
    return config_manager.should_log_to_file()


def get_logging_config() -> Dict[str, Any]:
    """
    Get the global logging configuration.
    
    Returns:
        Dictionary containing logging configuration.
    """
    config_manager = get_config_manager()
    return config_manager.get_logging_config() 