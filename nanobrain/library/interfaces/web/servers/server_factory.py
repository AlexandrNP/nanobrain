#!/usr/bin/env python3
"""
Universal Server Factory for NanoBrain Framework
Factory for assembling universal servers from configuration with modular components.

Author: NanoBrain Development Team
Date: January 2025
Version: 1.0.0
"""

import logging
import tempfile
import yaml
import os
from typing import Dict, Any, Optional, Type, Union, List
from pathlib import Path

from nanobrain.core.component_base import FromConfigBase
from nanobrain.core.config.config_base import ConfigBase
from nanobrain.library.interfaces.web.servers.base_server import BaseUniversalServer
from nanobrain.library.interfaces.web.servers.universal_server import UniversalNanoBrainServer
from pydantic import Field

# Factory logger
logger = logging.getLogger(__name__)


class ServerFactoryConfig(ConfigBase):
    """Configuration for server factory"""
    
    # Default server configurations
    default_server_type: str = Field(
        default='fastapi_universal',
        description="Default server type to create"
    )
    
    # Server type mappings
    server_type_mappings: Dict[str, str] = Field(
        default_factory=lambda: {
            'fastapi_universal': 'nanobrain.library.interfaces.web.servers.universal_server.UniversalNanoBrainServer',
            'fastapi': 'nanobrain.library.interfaces.web.servers.universal_server.UniversalNanoBrainServer',
            'universal': 'nanobrain.library.interfaces.web.servers.universal_server.UniversalNanoBrainServer'
        },
        description="Mapping of server types to implementation classes"
    )
    
    # Factory behavior configuration
    factory_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            'auto_validate_config': True,
            'strict_component_loading': True,
            'enable_server_caching': False
        },
        description="Factory behavior settings"
    )


class UniversalServerFactory(FromConfigBase):
    """
    Factory for assembling universal servers from configuration.
    Supports different server types and assembly strategies following NanoBrain patterns.
    """
    
    def __init__(self):
        """Initialize server factory - use from_config for creation"""
        super().__init__()
        self.config: Optional[ServerFactoryConfig] = None
        self.server_cache: Dict[str, BaseUniversalServer] = {}
    
    @classmethod
    def _get_config_class(cls):
        """Return the configuration class for this component"""
        return ServerFactoryConfig
    
    def _init_from_config(self, config, component_config, dependencies):
        """Initialize factory from configuration"""
        super()._init_from_config(config, component_config, dependencies)
        
        logger.info("🏭 Initializing Universal Server Factory")
        self.config = config
        
        # Initialize factory components
        self.setup_factory_configuration()
        
        logger.info("✅ Universal Server Factory initialized successfully")
    
    def setup_factory_configuration(self) -> None:
        """Setup factory configuration and validation"""
        if self.config.factory_config.get('auto_validate_config', True):
            self.validate_factory_configuration()
        
        logger.debug("✅ Factory configuration setup complete")
    
    def validate_factory_configuration(self) -> None:
        """Validate factory configuration"""
        # Validate server type mappings
        for server_type, class_path in self.config.server_type_mappings.items():
            try:
                self.validate_server_class_path(class_path)
            except Exception as e:
                logger.warning(f"⚠️ Server type '{server_type}' validation failed: {e}")
        
        logger.debug("✅ Factory configuration validation complete")
    
    def validate_server_class_path(self, class_path: str) -> None:
        """Validate that a server class path is importable"""
        try:
            module_path, class_name = class_path.rsplit('.', 1)
            module = __import__(module_path, fromlist=[class_name])
            server_class = getattr(module, class_name)
            
            if not issubclass(server_class, BaseUniversalServer):
                raise ValueError(f"Class '{class_path}' is not a subclass of BaseUniversalServer")
                
        except Exception as e:
            raise ValueError(f"Failed to validate server class '{class_path}': {e}")
    
    @classmethod
    def create_server(cls, server_type: str, config_path: Union[str, Path, Dict[str, Any]]) -> BaseUniversalServer:
        """
        Create and assemble server based on type and configuration.
        
        Args:
            server_type: Type of server to create (fastapi_universal, etc.)
            config_path: Path to configuration file or configuration dictionary
            
        Returns:
            Configured universal server instance
        """
        logger.info(f"🏭 Creating {server_type} server from configuration")
        
        try:
            # Determine server class
            server_class = cls.get_server_class(server_type)
            
            # ✅ FRAMEWORK COMPLIANCE: Pass file path directly to from_config for proper nested object resolution
            if isinstance(config_path, dict):
                # For dictionary configs, need to create temporary file since framework requires file paths
                # for proper nested object resolution
                import tempfile
                import yaml
                import os
                
                with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
                    yaml.dump(config_path, f, default_flow_style=False, sort_keys=False, indent=2)
                    temp_config_path = f.name
                
                try:
                    # Create server using file path for proper framework nested object resolution
                    server = server_class.from_config(temp_config_path)
                finally:
                    # Clean up temporary file
                    os.unlink(temp_config_path)
            else:
                # Create server using file path directly (framework-compliant)
                server = server_class.from_config(config_path)
            
            logger.info(f"✅ {server_type} server created successfully")
            return server
            
        except Exception as e:
            logger.error(f"❌ Failed to create {server_type} server: {e}")
            raise
    
    @classmethod
    def create_fastapi_server(cls, config_path: Union[str, Path, Dict[str, Any]]) -> UniversalNanoBrainServer:
        """
        Create FastAPI-based universal server.
        
        Args:
            config_path: Path to configuration file or configuration dictionary
            
        Returns:
            Configured FastAPI universal server
        """
        return cls.create_server('fastapi_universal', config_path)
    
    @classmethod
    def create_flask_server(cls, config_path: Union[str, Path, Dict[str, Any]]) -> BaseUniversalServer:
        """
        Create Flask-based universal server (if implemented).
        
        Args:
            config_path: Path to configuration file or configuration dictionary
            
        Returns:
            Configured Flask universal server
        """
        # Note: Flask implementation would go here if needed
        raise NotImplementedError("Flask server implementation not yet available")
    
    @classmethod
    def load_server_configuration(cls, config_path: Union[str, Path]) -> Dict[str, Any]:
        """Load server configuration from file"""
        try:
            config_path = Path(config_path)
            
            if not config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
            if config_path.suffix.lower() in ['.yml', '.yaml']:
                import yaml
                with open(config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                import json
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
            
            logger.debug(f"✅ Configuration loaded from {config_path}")
            return config_data
            
        except Exception as e:
            logger.error(f"❌ Failed to load configuration from {config_path}: {e}")
            raise
    
    @classmethod
    def get_server_class(cls, server_type: str) -> Type[BaseUniversalServer]:
        """Get server class for specified server type"""
        try:
            # Default mappings if factory not initialized
            default_mappings = {
                'fastapi_universal': 'nanobrain.library.interfaces.web.servers.universal_server.UniversalNanoBrainServer',
                'fastapi': 'nanobrain.library.interfaces.web.servers.universal_server.UniversalNanoBrainServer',
                'universal': 'nanobrain.library.interfaces.web.servers.universal_server.UniversalNanoBrainServer'
            }
            
            # Get class path from mappings
            class_path = default_mappings.get(server_type)
            if not class_path:
                raise ValueError(f"Unsupported server type: {server_type}")
            
            # Import and return class
            module_path, class_name = class_path.rsplit('.', 1)
            module = __import__(module_path, fromlist=[class_name])
            server_class = getattr(module, class_name)
            
            logger.debug(f"✅ Server class resolved: {class_path}")
            return server_class
            
        except Exception as e:
            logger.error(f"❌ Failed to get server class for type '{server_type}': {e}")
            raise
    
    def create_server_with_caching(self, server_type: str, config_path: Union[str, Path, Dict[str, Any]], 
                                  cache_key: Optional[str] = None) -> BaseUniversalServer:
        """
        Create server with optional caching for reuse.
        
        Args:
            server_type: Type of server to create
            config_path: Configuration path or dictionary
            cache_key: Optional cache key for server reuse
            
        Returns:
            Configured universal server (cached or new)
        """
        # Check if caching is enabled
        if not self.config.factory_config.get('enable_server_caching', False):
            return self.create_server(server_type, config_path)
        
        # Generate cache key if not provided
        if cache_key is None:
            cache_key = f"{server_type}_{hash(str(config_path))}"
        
        # Check cache first
        if cache_key in self.server_cache:
            logger.debug(f"✅ Returning cached server: {cache_key}")
            return self.server_cache[cache_key]
        
        # Create new server and cache it
        server = self.create_server(server_type, config_path)
        self.server_cache[cache_key] = server
        
        logger.debug(f"✅ Server cached: {cache_key}")
        return server
    
    def clear_server_cache(self) -> None:
        """Clear the server cache"""
        self.server_cache.clear()
        logger.debug("✅ Server cache cleared")
    
    def get_available_server_types(self) -> List[str]:
        """Get list of available server types"""
        if self.config:
            return list(self.config.server_type_mappings.keys())
        else:
            return ['fastapi_universal', 'fastapi', 'universal']
    
    def validate_server_configuration(self, config_data: Dict[str, Any]) -> bool:
        """
        Validate server configuration before creating server.
        
        Args:
            config_data: Configuration dictionary to validate
            
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        try:
            # Check required configuration sections
            required_sections = ['components', 'endpoints']
            for section in required_sections:
                if section not in config_data:
                    raise ValueError(f"Missing required configuration section: {section}")
            
            # Validate components configuration
            components = config_data.get('components', {})
            required_components = ['workflow_registry', 'request_analyzer', 'workflow_router', 'response_processor']
            
            for component in required_components:
                if component not in components:
                    raise ValueError(f"Missing required component configuration: {component}")
                
                component_config = components[component]
                if 'class' not in component_config:
                    raise ValueError(f"Component '{component}' missing required 'class' field")
            
            # Validate endpoints configuration
            endpoints = config_data.get('endpoints', {})
            required_endpoints = ['chat', 'capabilities', 'health']
            
            for endpoint in required_endpoints:
                if endpoint not in endpoints:
                    raise ValueError(f"Missing required endpoint configuration: {endpoint}")
            
            logger.debug("✅ Server configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Server configuration validation failed: {e}")
            raise
    
    @classmethod
    def create_server_from_template(cls, template_name: str, custom_config: Optional[Dict[str, Any]] = None) -> BaseUniversalServer:
        """
        Create server from predefined template configuration.
        
        Args:
            template_name: Name of template configuration to use
            custom_config: Optional custom configuration to merge with template
            
        Returns:
            Configured universal server
        """
        try:
            # Load template configuration
            template_config = cls.get_template_configuration(template_name)
            
            # Merge with custom configuration if provided
            if custom_config:
                template_config = cls.merge_configurations(template_config, custom_config)
            
            # ✅ FRAMEWORK COMPLIANCE: Write config to temporary file for from_config pattern
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
                yaml.dump(template_config, f, default_flow_style=False, sort_keys=False, indent=2)
                temp_config_path = f.name
            
            try:
                # Create server from temporary file path (framework-compliant)
                return cls.create_server('fastapi_universal', temp_config_path)
            finally:
                # Clean up temporary file
                os.unlink(temp_config_path)
            
        except Exception as e:
            logger.error(f"❌ Failed to create server from template '{template_name}': {e}")
            raise
    
    @classmethod
    def get_template_configuration(cls, template_name: str) -> Dict[str, Any]:
        """Get predefined template configuration"""
        templates = {
            'minimal': {
                'server_type': 'fastapi',
                'host': '0.0.0.0',
                'port': 5001,
                'debug': False,
                'components': {
                    'workflow_registry': {
                        'class': 'nanobrain.library.interfaces.web.routing.workflow_registry.WorkflowRegistry',
                        'config': 'nanobrain/library/interfaces/web/config/workflow_registry_config.yml'
                    },
                    'request_analyzer': {
                        'class': 'nanobrain.library.interfaces.web.analysis.request_analyzer.UniversalRequestAnalyzer',
                        'config': 'nanobrain/library/interfaces/web/config/request_analyzer_config.yml'
                    },
                    'workflow_router': {
                        'class': 'nanobrain.library.interfaces.web.routing.workflow_router.WorkflowRouter',
                        'config': 'nanobrain/library/interfaces/web/config/workflow_router_config.yml'
                    },
                    'response_processor': {
                        'class': 'nanobrain.library.interfaces.web.processing.response_processor.UniversalResponseProcessor',
                        'config': 'nanobrain/library/interfaces/web/config/response_processor_config.yml'
                    }
                },
                'endpoints': {
                    'chat': '/api/universal-chat',
                    'capabilities': '/api/workflows/capabilities',
                    'health': '/api/health'
                }
            },
            'development': {
                'server_type': 'fastapi',
                'host': '0.0.0.0',
                'port': 5001,
                'debug': True,
                'components': {
                    'workflow_registry': {
                        'class': 'nanobrain.library.interfaces.web.routing.workflow_registry.WorkflowRegistry',
                        'config': 'nanobrain/library/interfaces/web/config/workflow_registry_config.yml'
                    },
                    'request_analyzer': {
                        'class': 'nanobrain.library.interfaces.web.analysis.request_analyzer.UniversalRequestAnalyzer',
                        'config': 'nanobrain/library/interfaces/web/config/request_analyzer_config.yml'
                    },
                    'workflow_router': {
                        'class': 'nanobrain.library.interfaces.web.routing.workflow_router.WorkflowRouter',
                        'config': 'nanobrain/library/interfaces/web/config/workflow_router_config.yml'
                    },
                    'response_processor': {
                        'class': 'nanobrain.library.interfaces.web.processing.response_processor.UniversalResponseProcessor',
                        'config': 'nanobrain/library/interfaces/web/config/response_processor_config.yml'
                    }
                },
                'endpoints': {
                    'chat': '/api/universal-chat',
                    'capabilities': '/api/workflows/capabilities',
                    'health': '/api/health'
                }
            }
        }
        
        if template_name not in templates:
            raise ValueError(f"Unknown template: {template_name}. Available templates: {list(templates.keys())}")
        
        return templates[template_name]
    
    @classmethod
    def merge_configurations(cls, base_config: Dict[str, Any], custom_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge custom configuration into base configuration"""
        merged = base_config.copy()
        
        for key, value in custom_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = cls.merge_configurations(merged[key], value)
            else:
                merged[key] = value
        
        return merged 