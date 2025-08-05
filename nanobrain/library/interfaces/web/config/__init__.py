#!/usr/bin/env python3
"""
Configuration Package for Universal NanoBrain Interface
Provides configuration management, loading, and validation utilities.

Author: NanoBrain Development Team
Date: January 2025
Version: 1.0.0
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional

# Configuration file paths
CONFIG_DIR = Path(__file__).parent
UNIVERSAL_CONFIG = CONFIG_DIR / "universal_server_config.yml"
MINIMAL_CONFIG = CONFIG_DIR / "minimal_server_config.yml"
DOCKER_CONFIG = CONFIG_DIR / "docker_server_config.yml"

# Component configuration files
SERVER_FACTORY_CONFIG = CONFIG_DIR / "server_factory_config.yml"
WORKFLOW_REGISTRY_CONFIG = CONFIG_DIR / "workflow_registry_config.yml"
REQUEST_ANALYZER_CONFIG = CONFIG_DIR / "request_analyzer_config.yml"
WORKFLOW_ROUTER_CONFIG = CONFIG_DIR / "workflow_router_config.yml"
RESPONSE_PROCESSOR_CONFIG = CONFIG_DIR / "response_processor_config.yml"
INTENT_CLASSIFIER_CONFIG = CONFIG_DIR / "intent_classifier_config.yml"
DOMAIN_CLASSIFIER_CONFIG = CONFIG_DIR / "domain_classifier_config.yml"

class ConfigurationManager:
    """Universal configuration management system"""
    
    @staticmethod
    def load_config(config_path: Path) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Validate configuration structure"""
        errors = []
        
        # Check required sections
        required_sections = ['components', 'endpoints']
        for section in required_sections:
            if section not in config:
                errors.append(f"Missing required section: {section}")
        
        # Validate components section
        if 'components' in config:
            required_components = ['workflow_registry', 'request_analyzer', 'workflow_router', 'response_processor']
            components = config['components']
            
            for component in required_components:
                if component not in components:
                    errors.append(f"Missing required component: {component}")
                elif 'class' not in components[component]:
                    errors.append(f"Component '{component}' missing 'class' field")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge configuration dictionaries"""
        merged = base_config.copy()
        
        for key, value in override_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = ConfigurationManager.merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged

# Configuration loading functions
def load_universal_config() -> Dict[str, Any]:
    """Load universal server configuration"""
    return ConfigurationManager.load_config(UNIVERSAL_CONFIG)

def load_minimal_config() -> Dict[str, Any]:
    """Load minimal server configuration"""
    return ConfigurationManager.load_config(MINIMAL_CONFIG)

def load_docker_config() -> Dict[str, Any]:
    """Load Docker server configuration"""
    return ConfigurationManager.load_config(DOCKER_CONFIG)

# Environment variable override support
def get_config_with_env_overrides(config: Dict[str, Any], prefix: str = "NANOBRAIN_") -> Dict[str, Any]:
    """Apply environment variable overrides to configuration"""
    config_copy = config.copy()
    
    for key, value in os.environ.items():
        if key.startswith(prefix):
            config_key = key[len(prefix):].lower()
            
            # Convert environment variable value to appropriate type
            if value.lower() in ['true', 'false']:
                env_value = value.lower() == 'true'
            elif value.isdigit():
                env_value = int(value)
            else:
                try:
                    env_value = float(value)
                except ValueError:
                    env_value = value
            
            config_copy[config_key] = env_value
    
    return config_copy

__all__ = [
    # Configuration paths
    "CONFIG_DIR", "UNIVERSAL_CONFIG", "MINIMAL_CONFIG", "DOCKER_CONFIG",
    "SERVER_FACTORY_CONFIG", "WORKFLOW_REGISTRY_CONFIG", "REQUEST_ANALYZER_CONFIG",
    "WORKFLOW_ROUTER_CONFIG", "RESPONSE_PROCESSOR_CONFIG", "INTENT_CLASSIFIER_CONFIG",
    "DOMAIN_CLASSIFIER_CONFIG",
    
    # Configuration management
    "ConfigurationManager",
    
    # Loading functions
    "load_universal_config", "load_minimal_config", "load_docker_config",
    "get_config_with_env_overrides"
] 