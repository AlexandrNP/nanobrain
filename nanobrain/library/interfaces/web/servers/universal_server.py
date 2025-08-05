#!/usr/bin/env python3
"""
Universal NanoBrain Server Implementation
Complete universal server supporting any NanoBrain workflow with natural language input.

Author: NanoBrain Development Team
Date: January 2025
Version: 1.0.0
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from nanobrain.library.interfaces.web.servers.base_server import BaseUniversalServer, BaseServerConfig
from nanobrain.library.interfaces.web.models.request_models import ChatRequest
from nanobrain.library.interfaces.web.models.response_models import ChatResponse, HealthResponse
from nanobrain.library.interfaces.web.models.workflow_models import WorkflowCapabilities
from pydantic import Field

# Server logger
logger = logging.getLogger(__name__)


class UniversalServerConfig(BaseServerConfig):
    """Configuration for universal NanoBrain server"""
    
    # FastAPI specific configuration
    fastapi_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            'title': 'Universal NanoBrain Server',
            'description': 'Universal interface for NanoBrain framework workflows',
            'version': '1.0.0',
            'docs_url': '/docs',
            'redoc_url': '/redoc'
        },
        description="FastAPI specific configuration"
    )
    
    # CORS configuration
    cors_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            'allow_origins': ['*'],
            'allow_methods': ['GET', 'POST'],
            'allow_headers': ['*'],
            'allow_credentials': True
        },
        description="CORS configuration"
    )
    
    # Server startup configuration
    startup_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            'auto_discover_workflows': True,
            'validate_components': True,
            'preload_workflows': False
        },
        description="Server startup configuration"
    )


class UniversalNanoBrainServer(BaseUniversalServer):
    """
    Complete universal server supporting any NanoBrain workflow.
    Uses modular assembly of framework components via configuration.
    """
    
    def __init__(self):
        """Initialize universal server - use from_config for creation"""
        super().__init__()
        # Instance variables moved to _init_from_config since framework uses __new__ and bypasses __init__
    
    @classmethod
    def _get_config_class(cls):
        """Return the configuration class for this component"""
        return UniversalServerConfig
    
    def _init_from_config(self, config, component_config, dependencies):
        """Initialize server from configuration"""
        super()._init_from_config(config, component_config, dependencies)
        
        # ✅ FRAMEWORK COMPLIANCE: Initialize instance variables here since __init__ is bypassed
        self.fastapi_app: Optional[FastAPI] = None
        self.uvicorn_server: Optional[Any] = None
        
        logger.info("🌐 Initializing Universal NanoBrain Server")
        
        # ✅ FRAMEWORK COMPLIANCE: Load components from dependencies (resolved by framework)
        if dependencies:
            # Load resolved components from dependencies
            self.workflow_registry = dependencies.get('workflow_registry')
            self.request_analyzer = dependencies.get('request_analyzer')
            self.workflow_router = dependencies.get('workflow_router')
            self.response_processor = dependencies.get('response_processor')
            
            loaded_components = []
            if self.workflow_registry:
                loaded_components.append('workflow_registry')
            if self.request_analyzer:
                loaded_components.append('request_analyzer')
            if self.workflow_router:
                loaded_components.append('workflow_router')
            if self.response_processor:
                loaded_components.append('response_processor')
                
            logger.info(f"✅ Loaded {len(loaded_components)} components: {loaded_components}")
        else:
            # Initialize components as None if no dependencies provided
            self.workflow_registry = None
            self.request_analyzer = None
            self.workflow_router = None
            self.response_processor = None
            logger.warning("⚠️ No dependencies provided - components will be None")
        
        logger.info("✅ Universal NanoBrain Server initialized successfully")
    
    def configure_server_implementation(self) -> None:
        """Configure FastAPI-specific implementation"""
        if not FASTAPI_AVAILABLE:
            raise RuntimeError("FastAPI not available. Install with: pip install fastapi uvicorn")
        
        logger.debug("🔧 Configuring FastAPI implementation")
        
        try:
            # Create FastAPI application
            self.fastapi_app = FastAPI(**self.config.fastapi_config)
            
            # Add exception handler for HTTPExceptions
            self.setup_exception_handlers()
            
            logger.debug("✅ FastAPI application configured")
            
        except Exception as e:
            logger.error(f"❌ FastAPI configuration failed: {e}")
            raise
    
    def configure_endpoints(self) -> None:
        """Configure universal endpoints based on configuration"""
        if not self.fastapi_app:
            raise RuntimeError("FastAPI application not initialized")
        
        logger.debug("🔧 Configuring universal endpoints")
        
        try:
            # Universal chat endpoint
            @self.fastapi_app.post(self.config.endpoints['chat'])
            async def universal_chat_endpoint(request: ChatRequest):
                """Universal chat endpoint supporting any NanoBrain workflow"""
                try:
                    response = await self.process_universal_request(request)
                    return response
                except Exception as e:
                    logger.error(f"❌ Chat endpoint error: {e}")
                    raise HTTPException(status_code=500, detail=str(e))
            
            # Workflow capabilities endpoint
            @self.fastapi_app.get(self.config.endpoints['capabilities'])
            async def get_workflow_capabilities():
                """Return available workflow capabilities for frontend adaptation"""
                try:
                    if not self.workflow_registry:
                        raise RuntimeError("Workflow registry not configured")
                    
                    capabilities = await self.workflow_registry.get_all_capabilities()
                    return {"capabilities": capabilities}
                except Exception as e:
                    logger.error(f"❌ Capabilities endpoint error: {e}")
                    raise HTTPException(status_code=500, detail=str(e))
            
            # Health check endpoint
            @self.fastapi_app.get(self.config.endpoints['health'])
            async def health_check():
                """Comprehensive health check for server and components"""
                try:
                    health_data = await self.get_server_health()
                    return health_data
                except Exception as e:
                    logger.error(f"❌ Health check error: {e}")
                    return {
                        "status": "unhealthy",
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    }
            
            # Request analysis endpoint (for debugging/transparency)
            @self.fastapi_app.post("/api/workflows/route-analysis")
            async def analyze_request_routing(request: ChatRequest):
                """Analyze request and return routing information"""
                try:
                    if not self.request_analyzer or not self.workflow_registry:
                        raise RuntimeError("Request analysis components not configured")
                    
                    analysis = await self.request_analyzer.analyze_request(request)
                    compatible_workflows = await self.workflow_registry.get_compatible_workflows(analysis)
                    
                    return {
                        "analysis": analysis.dict(),
                        "compatible_workflows": [w.dict() for w in compatible_workflows]
                    }
                except Exception as e:
                    logger.error(f"❌ Route analysis error: {e}")
                    raise HTTPException(status_code=500, detail=str(e))
            
            # Root endpoint
            @self.fastapi_app.get("/")
            async def root():
                """Root endpoint with server information"""
                return {
                    "message": "Universal NanoBrain Server",
                    "version": self.config.fastapi_config.get('version', '1.0.0'),
                    "status": "healthy" if self.is_running else "initializing",
                    "endpoints": self.config.endpoints,
                    "docs_url": self.config.fastapi_config.get('docs_url'),
                    "framework": "NanoBrain Universal Interface"
                }
            
            logger.info(f"✅ Configured {len(self.config.endpoints) + 2} endpoints")
            
        except Exception as e:
            logger.error(f"❌ Endpoint configuration failed: {e}")
            raise
    
    def initialize_middleware(self) -> None:
        """Initialize framework-compliant middleware"""
        if not self.fastapi_app:
            raise RuntimeError("FastAPI application not initialized")
        
        logger.debug("🔧 Initializing middleware")
        
        try:
            # CORS middleware
            if self.config.middleware.get('cors', True):
                self.fastapi_app.add_middleware(
                    CORSMiddleware,
                    **self.config.cors_config
                )
                logger.debug("✅ CORS middleware configured")
            
            # Request validation middleware (FastAPI handles this automatically)
            if self.config.middleware.get('request_validation', True):
                logger.debug("✅ Request validation middleware enabled (FastAPI)")
            
            # Response standardization middleware
            if self.config.middleware.get('response_standardization', True):
                @self.fastapi_app.middleware("http")
                async def response_standardization_middleware(request: Request, call_next):
                    """Middleware to standardize responses"""
                    try:
                        response = await call_next(request)
                        return response
                    except Exception as e:
                        logger.error(f"❌ Middleware error: {e}")
                        return JSONResponse(
                            status_code=500,
                            content={
                                "error": "Internal server error",
                                "message": str(e) if self.config.debug else "An error occurred",
                                "timestamp": datetime.now().isoformat()
                            }
                        )
                
                logger.debug("✅ Response standardization middleware configured")
            
            logger.info("✅ All middleware initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Middleware initialization failed: {e}")
            raise
    
    def setup_exception_handlers(self) -> None:
        """Setup global exception handlers"""
        if not self.fastapi_app:
            return
        
        @self.fastapi_app.exception_handler(HTTPException)
        async def http_exception_handler(request: Request, exc: HTTPException):
            """Handle HTTP exceptions"""
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "error": exc.detail,
                    "status_code": exc.status_code,
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        @self.fastapi_app.exception_handler(Exception)
        async def general_exception_handler(request: Request, exc: Exception):
            """Handle general exceptions"""
            logger.error(f"❌ Unhandled exception: {exc}")
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "message": str(exc) if self.config.debug else "An error occurred",
                    "timestamp": datetime.now().isoformat()
                }
            )
    
    async def startup_sequence(self) -> None:
        """Execute startup sequence for server initialization"""
        logger.info("🚀 Starting universal server startup sequence")
        
        try:
            # Step 1: Validate all components are loaded
            await self.validate_component_initialization()
            
            # Step 2: Auto-discover workflows if configured
            if self.config.startup_config.get('auto_discover_workflows', True):
                await self.auto_discover_workflows()
            
            # Step 3: Validate component health
            if self.config.startup_config.get('validate_components', True):
                await self.validate_component_health()
            
            # Step 4: Preload workflows if configured
            if self.config.startup_config.get('preload_workflows', False):
                await self.preload_workflows()
            
            self.is_running = True
            self.startup_time = datetime.now()
            
            logger.info("✅ Universal server startup sequence completed")
            
        except Exception as e:
            logger.error(f"❌ Startup sequence failed: {e}")
            raise
    
    async def validate_component_initialization(self) -> None:
        """Validate that all required components are properly initialized"""
        required_components = ['workflow_registry', 'request_analyzer', 'workflow_router', 'response_processor']
        
        for component_name in required_components:
            component = getattr(self, component_name, None)
            if not component:
                raise RuntimeError(f"Required component '{component_name}' not initialized")
        
        logger.debug("✅ All required components validated")
    
    async def auto_discover_workflows(self) -> None:
        """Auto-discover available workflows"""
        try:
            logger.info("🔍 Auto-discovering workflows...")
            discovery_result = await self.workflow_registry.discover_workflows()
            logger.info(f"✅ Discovered {len(discovery_result.discovered_workflows)} workflows")
        except Exception as e:
            logger.error(f"❌ Workflow discovery failed: {e}")
            raise
    
    async def validate_component_health(self) -> None:
        """Validate health of all components"""
        try:
            for component_name, component in self.components.items():
                if hasattr(component, 'validate_health'):
                    await component.validate_health()
            logger.debug("✅ All components health validated")
        except Exception as e:
            logger.error(f"❌ Component health validation failed: {e}")
            raise
    
    async def preload_workflows(self) -> None:
        """Preload workflows for faster response times"""
        try:
            logger.info("⚡ Preloading workflows...")
            if hasattr(self.workflow_registry, 'preload_workflows'):
                await self.workflow_registry.preload_workflows()
            logger.info("✅ Workflows preloaded successfully")
        except Exception as e:
            logger.warning(f"⚠️ Workflow preloading failed (non-critical): {e}")
    
    async def start(self, host: Optional[str] = None, port: Optional[int] = None) -> None:
        """Start the universal server"""
        if not self.fastapi_app:
            raise RuntimeError("Server not properly configured")
        
        # Use provided host/port or fall back to configuration
        server_host = host or self.config.host
        server_port = port or self.config.port
        
        logger.info(f"🚀 Starting Universal NanoBrain Server on {server_host}:{server_port}")
        
        try:
            # Execute startup sequence
            await self.startup_sequence()
            
            # Configure uvicorn server
            uvicorn_config = uvicorn.Config(
                app=self.fastapi_app,
                host=server_host,
                port=server_port,
                log_level="debug" if self.config.debug else "info",
                access_log=self.config.debug
            )
            
            self.uvicorn_server = uvicorn.Server(uvicorn_config)
            
            # Start server
            logger.info(f"✅ Universal NanoBrain Server running on http://{server_host}:{server_port}")
            logger.info(f"📚 API Documentation: http://{server_host}:{server_port}/docs")
            
            await self.uvicorn_server.serve()
            
        except Exception as e:
            logger.error(f"❌ Server startup failed: {e}")
            self.is_running = False
            raise
    
    async def stop(self) -> None:
        """Stop the universal server"""
        logger.info("🛑 Stopping Universal NanoBrain Server")
        
        try:
            if self.uvicorn_server:
                self.uvicorn_server.should_exit = True
                await self.uvicorn_server.shutdown()
            
            self.is_running = False
            logger.info("✅ Universal NanoBrain Server stopped")
            
        except Exception as e:
            logger.error(f"❌ Server shutdown error: {e}")
            raise 