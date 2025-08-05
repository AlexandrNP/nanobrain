#!/usr/bin/env python3
"""
Base Universal Server for NanoBrain Framework
Abstract foundation for universal server implementations with framework integration.

Author: NanoBrain Development Team
Date: January 2025
Version: 1.0.0
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from pydantic import Field, BaseModel
from pathlib import Path

from nanobrain.core.component_base import FromConfigBase
from nanobrain.core.config.config_base import ConfigBase
from nanobrain.library.interfaces.web.models.request_models import ChatRequest
from nanobrain.library.interfaces.web.models.response_models import ChatResponse
from nanobrain.library.interfaces.web.models.universal_models import (
    UniversalResponse, StandardizedResponse
)

# Server framework logger
logger = logging.getLogger(__name__)


class BaseServerConfig(ConfigBase):
    """Configuration for base universal server"""
    
    # Server basic configuration
    host: str = Field(
        default='0.0.0.0',
        description="Server host address"
    )
    port: int = Field(
        default=5001,
        description="Server port number"
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode"
    )
    server_type: str = Field(
        default='fastapi',
        description="Server type to use"
    )
    
    # Component configuration paths
    components: Dict[str, Any] = Field(
        default_factory=dict,
        description="Component configuration mapping"
    )
    
    # Endpoint configuration
    endpoints: Dict[str, str] = Field(
        default_factory=lambda: {
            'chat': '/api/universal-chat',
            'capabilities': '/api/workflows/capabilities',
            'health': '/api/health'
        },
        description="API endpoint configuration"
    )
    
    # Middleware configuration
    middleware: Dict[str, bool] = Field(
        default_factory=lambda: {
            'cors': True,
            'request_validation': True,
            'response_standardization': True
        },
        description="Middleware configuration"
    )
    
    # Framework integration settings
    framework_integration: Dict[str, Any] = Field(
        default_factory=lambda: {
            'auto_component_loading': True,
            'strict_validation': True,
            'error_handling': 'comprehensive'
        },
        description="Framework integration settings"
    )


class BaseUniversalServer(FromConfigBase, ABC):
    """
    Abstract base class for universal NanoBrain web servers.
    Provides core functionality and framework integration patterns.
    """
    
    def __init__(self):
        """Initialize base server - use from_config for creation"""
        super().__init__()
        # Instance variables moved to _init_from_config since framework uses __new__ and bypasses __init__
        
    @classmethod
    def _get_config_class(cls):
        """Return the configuration class for this component"""
        return BaseServerConfig
    
    def _init_from_config(self, config, component_config, dependencies):
        """Initialize server from configuration"""
        super()._init_from_config(config, component_config, dependencies)
        
        # ✅ FRAMEWORK COMPLIANCE: Initialize instance variables here since __init__ is bypassed
        self.config: Optional[BaseServerConfig] = None
        self.server_instance: Optional[Any] = None
        self.components: Dict[str, Any] = {}
        self.is_running: bool = False
        self.startup_time: Optional[datetime] = None
        self.request_count: int = 0
        self.error_count: int = 0  # ✅ FRAMEWORK FIX: Add missing error_count initialization
        
        logger.info("🔄 Initializing Universal Base Server")
        self.config = config
        
        logger.info("✅ Universal Base Server initialized successfully")
    
    def setup_framework_integration(self) -> None:
        """Setup integration with NanoBrain framework components"""
        logger.debug("🔧 Setting up framework integration")
        
        try:
            # Validate configuration compliance
            self.validate_framework_compliance()
            
            # Setup logging integration
            self.setup_logging_integration()
            
            # Initialize component loading strategy
            self.initialize_component_loading()
            
            logger.debug("✅ Framework integration setup complete")
            
        except Exception as e:
            logger.error(f"❌ Framework integration setup failed: {e}")
            raise
    
    def load_components_from_config(self) -> None:
        """Load universal interface components from configuration"""
        logger.debug("🔄 Loading components from configuration")
        
        try:
            components_config = self.config.components
            
            for component_name, component_config in components_config.items():
                logger.debug(f"📦 Loading component: {component_name}")
                
                # Use framework patterns to load component
                component = self.load_component_via_framework(component_name, component_config)
                self.components[component_name] = component
                
                # Set component as instance attribute for easy access
                setattr(self, component_name, component)
                
                logger.debug(f"✅ Component loaded: {component_name}")
            
            logger.info(f"✅ Loaded {len(self.components)} components successfully")
            
        except Exception as e:
            logger.error(f"❌ Component loading failed: {e}")
            raise
    
    def load_component_via_framework(self, component_name: str, component_config: Dict[str, Any]) -> Any:
        """Load a component using NanoBrain framework patterns"""
        try:
            # Extract component class and configuration
            component_class_path = component_config.get('class')
            component_cfg = component_config.get('config', {})
            
            if not component_class_path:
                raise ValueError(f"Component '{component_name}' missing required 'class' field")
            
            # Import component class dynamically
            module_path, class_name = component_class_path.rsplit('.', 1)
            module = __import__(module_path, fromlist=[class_name])
            component_class = getattr(module, class_name)
            
            # Create component using from_config pattern
            if hasattr(component_class, 'from_config'):
                component = component_class.from_config(component_cfg)
                logger.debug(f"✅ Component '{component_name}' created via from_config")
                return component
            else:
                raise ValueError(f"Component class '{component_class_path}' does not implement from_config pattern")
                
        except Exception as e:
            logger.error(f"❌ Failed to load component '{component_name}': {e}")
            raise
    
    @abstractmethod
    def configure_server_implementation(self) -> None:
        """Configure server-specific implementation (FastAPI, Flask, etc.)"""
        pass
    
    @abstractmethod
    def configure_endpoints(self) -> None:
        """Configure universal endpoints based on configuration"""
        pass
    
    @abstractmethod
    def initialize_middleware(self) -> None:
        """Initialize framework-compliant middleware"""
        pass
    
    async def process_universal_request(self, request: ChatRequest) -> ChatResponse:
        """
        Universal request processing pipeline.
        Routes to appropriate workflow based on request analysis.
        """
        try:
            self.request_count += 1
            start_time = datetime.now()
            
            logger.debug(f"🔄 Processing universal request: {request.request_id}")
            
            # Step 1: Analyze request for routing
            if not self.request_analyzer:
                raise RuntimeError("Request analyzer not configured")
            
            analysis = await self.request_analyzer.analyze_request(request)
            logger.debug(f"📊 Request analysis completed: {analysis.intent_classification.intent_type}")
            
            # Step 2: Route to appropriate workflow
            if not self.workflow_router:
                raise RuntimeError("Workflow router not configured")
            
            route = await self.workflow_router.route_request(request, analysis)
            logger.debug(f"🎯 Request routed to: {route.selected_workflow.workflow_id}")
            
            # Step 3: Execute workflow route
            universal_response = await self.workflow_router.execute_route(route)
            logger.debug(f"⚡ Workflow execution completed: {universal_response.success}")
            
            # Step 4: Process and standardize response
            if not self.response_processor:
                raise RuntimeError("Response processor not configured")
            
            standardized_response = await self.response_processor.standardize_response(universal_response)
            
            # Step 5: Convert to ChatResponse format
            chat_response = self.convert_to_chat_response(standardized_response, request)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"✅ Request processed successfully in {processing_time:.3f}s")
            
            return chat_response
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"❌ Request processing failed: {e}")
            
            # Return error response in ChatResponse format
            return self.create_error_response(request, str(e))
    
    def convert_to_chat_response(self, standardized_response: StandardizedResponse, 
                                original_request: ChatRequest) -> ChatResponse:
        """Convert standardized response to ChatResponse format"""
        from nanobrain.library.interfaces.web.models.response_models import ChatMetadata
        
        try:
            # ✅ FRAMEWORK COMPLIANCE: Create proper ChatMetadata with required processing_time_ms
            response_metadata = ChatMetadata(
                processing_time_ms=getattr(standardized_response, 'processing_time', 0.0) * 1000,  # ✅ SAFE ACCESS: Handle missing processing_time field
                workflow_metadata={
                    'response_id': standardized_response.response_id,
                    'response_format': standardized_response.response_format,
                    'data': standardized_response.data,
                    'frontend_hints': getattr(standardized_response, 'frontend_hints', {}),  # ✅ SAFE ACCESS: Handle missing field
                    'warnings': standardized_response.warnings
                }
            )
            
            return ChatResponse(
                response=standardized_response.message,
                request_id=original_request.request_id,
                conversation_id=original_request.options.conversation_id,
                success=standardized_response.success,
                metadata=response_metadata,  # ✅ FRAMEWORK COMPLIANCE: Use proper ChatMetadata object
                error_message=standardized_response.error_message
            )
        except Exception as e:
            logger.error(f"❌ Response conversion failed: {e}")
            return self.create_error_response(original_request, f"Response conversion failed: {e}")
    
    def create_error_response(self, request: ChatRequest, error_message: str) -> ChatResponse:
        """Create error response in ChatResponse format"""
        from nanobrain.library.interfaces.web.models.response_models import ChatMetadata
        
        # ✅ FRAMEWORK COMPLIANCE: Create proper ChatMetadata with required processing_time_ms
        error_metadata = ChatMetadata(
            processing_time_ms=0.0,  # ✅ REQUIRED FIELD: Set to 0 for error responses
            workflow_metadata={
                'error_type': 'processing_error',
                'timestamp': datetime.now().isoformat(),
                'server_type': self.config.server_type if self.config else 'unknown'
            }
        )
        
        return ChatResponse(
            response="I apologize, but I encountered an error processing your request. Please try again.",
            request_id=request.request_id,
            conversation_id=request.options.conversation_id if request.options else None,
            success=False,
            error_message=error_message,
            metadata=error_metadata  # ✅ FRAMEWORK COMPLIANCE: Use proper ChatMetadata object
        )
    
    def validate_framework_compliance(self) -> None:
        """Validate server configuration for framework compliance"""
        if not self.config:
            raise ValueError("Server configuration not loaded")
        
        # Validate required components are configured
        required_components = ['workflow_registry', 'request_analyzer', 'workflow_router', 'response_processor']
        for component in required_components:
            if component not in self.config.components:
                raise ValueError(f"Required component '{component}' not configured")
        
        logger.debug("✅ Framework compliance validation passed")
    
    def setup_logging_integration(self) -> None:
        """Setup logging integration with NanoBrain framework"""
        # Configure server-specific logging
        logging.getLogger(__name__).setLevel(
            logging.DEBUG if self.config.debug else logging.INFO
        )
    
    def initialize_component_loading(self) -> None:
        """Initialize component loading strategy"""
        if self.config.framework_integration.get('auto_component_loading', True):
            logger.debug("🔄 Auto component loading enabled")
    
    async def get_server_health(self) -> Dict[str, Any]:
        """Get comprehensive server health information"""
        try:
            workflow_status = "healthy"
            component_status = {}
            
            # Check component health
            for component_name, component in self.components.items():
                if hasattr(component, 'get_health_status'):
                    status = await component.get_health_status()
                    component_status[component_name] = status
                else:
                    component_status[component_name] = "healthy"
            
            # Check if any components are unhealthy
            if any(status != "healthy" for status in component_status.values()):
                workflow_status = "degraded"
            
            return {
                "status": "healthy" if self.is_running else "unhealthy",
                "server_type": self.config.server_type if self.config else "unknown",
                "uptime_seconds": (datetime.now() - self.startup_time).total_seconds() if self.startup_time else 0,
                "request_count": self.request_count,
                "error_count": self.error_count,
                "error_rate": self.error_count / max(self.request_count, 1),
                "components": component_status,
                "workflow_status": workflow_status,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    @abstractmethod
    async def start(self, host: Optional[str] = None, port: Optional[int] = None) -> None:
        """Start the server"""
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """Stop the server"""
        pass 

    @classmethod
    def resolve_dependencies(cls, component_config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Resolve dependencies including resolved components from config"""
        # Get default framework dependencies
        dependencies = super().resolve_dependencies(component_config, **kwargs)
        
        # ✅ FRAMEWORK COMPLIANCE: Extract resolved components from config if available
        # The framework's _resolve_nested_objects() has already instantiated components
        # We need to extract them and pass them as dependencies
        config = kwargs.get('_original_config')  # Original config object passed from from_config
        
        if config and hasattr(config, 'components'):
            # Extract resolved components from the config
            components = getattr(config, 'components', {})
            
            # The framework should have resolved class+config patterns to actual objects
            for component_name, component_instance in components.items():
                if hasattr(component_instance, '__class__'):  # Check if it's an instantiated object
                    dependencies[component_name] = component_instance
                    logger.debug(f"✅ Extracted resolved component: {component_name} -> {component_instance.__class__.__name__}")
        
        return dependencies 

    @classmethod
    def from_config(cls, config: Union[str, Path, BaseModel, Dict[str, Any]], **kwargs) -> 'BaseUniversalServer':
        """
        Enhanced server loading with automatic component instantiation
        
        ✅ FRAMEWORK COMPLIANCE:
        - Leverages ConfigBase._resolve_nested_objects() for automatic component instantiation
        - Components created via class+config patterns
        - No manual factory functions or redundant creation logic
        
        Args:
            config: Path to server configuration file or config object
            **kwargs: Additional context
            
        Returns:
            Fully initialized server instance with resolved components
        """
        
        # Use enhanced ServerConfig.from_config() method - automatically resolves class+config patterns  
        if isinstance(config, (str, Path)):
            server_config = cls._get_config_class().from_config(config, **kwargs)
        else:
            server_config = config
        
        # ConfigBase._resolve_nested_objects() has already instantiated all components
        # Extract resolved components from the configuration
        resolved_components = cls._extract_resolved_components(server_config)
        
        # Create server instance from resolved configuration
        server = cls._create_from_resolved_config(server_config, resolved_components, **kwargs)
        
        return server
    
    @classmethod
    def _extract_resolved_components(cls, server_config) -> Dict[str, Any]:
        """
        Extract instantiated components from resolved server configuration
        
        ConfigBase._resolve_nested_objects() has already instantiated all components
        specified with class+config patterns. This method extracts and validates them.
        
        Args:
            server_config: Resolved server configuration
            
        Returns:
            Dictionary containing instantiated components
        """
        resolved_components = {}
        
        # Extract resolved components from config
        components_config = getattr(server_config, 'components', {})
        for component_name, component_instance in components_config.items():
            if hasattr(component_instance, '__class__'):  # Check if it's an instantiated object
                resolved_components[component_name] = component_instance
                logger.debug(f"✅ Extracted {component_instance.__class__.__name__} for '{component_name}'")
        
        logger.info(f"✅ Extracted {len(resolved_components)} resolved components")
        return resolved_components
    
    @classmethod
    def _create_from_resolved_config(cls, server_config, resolved_components: Dict[str, Any], **kwargs) -> 'BaseUniversalServer':
        """
        Create server instance from resolved configuration and instantiated components
        
        Args:
            server_config: Resolved server configuration
            resolved_components: Dictionary of instantiated components
            **kwargs: Additional context
            
        Returns:
            Fully initialized server instance
        """
        # Create server instance using standard component creation pattern
        component_config = cls.extract_component_config(server_config)
        dependencies = cls.resolve_dependencies(component_config, **kwargs)
        
        # Add resolved components to dependencies
        dependencies.update(resolved_components)
        
        server = cls.create_instance(server_config, component_config, dependencies)
        
        # Store resolved components for server operation
        server._resolved_components = resolved_components
        
        logger.info(f"✅ Created server from resolved config with {len(resolved_components)} components")
        
        return server 