"""
Web Interface Step - Pure HTTP Request/Response Handling within Workflow
=======================================================================

Provides pure HTTP interface functionality as a proper NanoBrain workflow step, 
enabling web interfaces to participate in event-driven workflow orchestration 
with complete framework compliance.

**SINGLE RESPONSIBILITY**: HTTP Interface Only
- HTTP request/response handling and conversion to/from DataUnits
- Basic session tracking (ID generation and management)
- WebSocket support for real-time communication
- CORS and middleware configuration
- Health and status endpoints

**DOES NOT INCLUDE**:
- Query classification (handled by QueryClassificationStep)
- Workflow routing (handled by WorkflowRoutingStep) 
- Multi-workflow support (composed via framework configuration)
- Complex business logic (delegated to downstream steps)

This component follows NanoBrain framework patterns:
- Inherits from BaseStep for framework compliance
- Uses from_config pattern for component creation
- Provides comprehensive configuration validation
- Supports event-driven workflow orchestration

Usage:
    from nanobrain.library.infrastructure.steps import WebInterfaceStep
    
    # Create via from_config (framework pattern)
    step = WebInterfaceStep.from_config('config/web_interface_step.yml')
    
    # Execute within workflow context
    await step.execute()
"""

import asyncio
import uuid
import time
import json
from typing import Dict, Any, Optional, List, Union, Set, Callable
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException, WebSocket, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.websockets import WebSocketDisconnect
import uvicorn
from pydantic import BaseModel, Field
import websockets

from nanobrain.core.step import BaseStep, StepConfig
from nanobrain.core.config.config_base import ConfigBase
from nanobrain.core.logging_system import get_logger
from nanobrain.core.component_base import ComponentConfigurationError, ComponentDependencyError
from pydantic import ConfigDict

# Import data units
from nanobrain.core.data_unit import DataUnitBase, DataUnitMemory
from nanobrain.library.infrastructure.data.session_context_data_unit import SessionContextDataUnit
from nanobrain.library.infrastructure.data.progress_update_data_unit import ProgressUpdateDataUnit

# Import triggers
from nanobrain.core.trigger import TriggerBase


class ServerConfig(BaseModel):
    """Server configuration schema."""
    host: str = Field(default="0.0.0.0", description="Server host address")
    port: int = Field(default=5001, description="Server port number")
    max_connections: int = Field(
        default=200, description="Maximum concurrent connections")
    enable_websocket: bool = Field(
        default=True, description="Enable WebSocket support")
    session_timeout: int = Field(
        default=3600, description="Session timeout in seconds")
    request_timeout: int = Field(
        default=900, description="Request processing timeout in seconds")
    enable_compression: bool = Field(
        default=True, description="Enable gzip compression")
    enable_metrics: bool = Field(
        default=True, description="Enable metrics collection")


class CORSConfig(BaseModel):
    """CORS configuration schema."""
    allow_origins: List[str] = Field(
        default=["http://localhost:3000", "http://127.0.0.1:3000"],
        description="Allowed origins for CORS"
    )
    allow_credentials: bool = Field(
        default=True, description="Allow credentials in CORS")
    allow_methods: List[str] = Field(
        default=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        description="Allowed HTTP methods"
    )
    allow_headers: List[str] = Field(
        default=["*"],
        description="Allowed headers"
    )


class WebInterfaceStepConfig(StepConfig):
    """
    Configuration schema for WebInterfaceStep

    MANDATORY FIELDS:
    - All configuration classes MUST inherit from ConfigBase
    - MUST include comprehensive field documentation
    - MUST use Pydantic V2 validation with ConfigDict
    """

    # WebInterfaceStep-specific configuration (inherits base fields from StepConfig)
    # Fields like name, description, auto_initialize, debug_mode, enable_logging
    # are inherited from StepConfig and don't need to be redefined

    # HTTP SERVER CONFIGURATION
    server_config: ServerConfig = Field(
        default_factory=ServerConfig,
        description="HTTP server configuration"
    )

    # CORS CONFIGURATION
    cors_config: CORSConfig = Field(
        default_factory=CORSConfig,
        description="CORS configuration for cross-origin requests"
    )

    # SESSION MANAGEMENT
    session_management: Dict[str, Any] = Field(
        default={
            "enable_sessions": True,
            "session_id_header": "X-Session-ID",
            "auto_create_sessions": True,
            "session_storage": "memory"
        },
        description="Session management configuration"
    )

    # HEALTH AND MONITORING
    health_config: Dict[str, Any] = Field(
        default={
            "enable_health_endpoint": True,
            "health_path": "/api/health",
            "enable_metrics_endpoint": True,
            "metrics_path": "/api/metrics",
            "enable_status_endpoint": True,
            "status_path": "/api/status"
        },
        description="Health and monitoring endpoints configuration"
    )

    # WEBSOCKET CONFIGURATION
    websocket_config: Dict[str, Any] = Field(
        default={
            "enable_websocket": True,
            "websocket_path": "/ws",
            "max_connections": 100,
            "heartbeat_interval": 30,
            "connection_timeout": 300
        },
        description="WebSocket configuration for real-time communication"
    )

    # STATIC FILES CONFIGURATION
    static_files_config: Dict[str, Any] = Field(
        default={
            "enable_static_files": False,
            "static_directory": "static",
            "mount_path": "/static",
            "cache_max_age": 3600
        },
        description="Static files configuration for serving frontend assets"
    )

    # MANDATORY PYDANTIC V2 CONFIGURATION
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="allow",
        use_enum_values=False,
        validate_assignment=True,
        str_strip_whitespace=True,
        json_schema_extra={
            "examples": [
                {
                    "name": "pure_web_interface",
                    "description": "Pure HTTP interface for framework integration",
                    "auto_initialize": True,
                    "debug_mode": False,
                    "enable_logging": True,
                    "server_config": {
                        "host": "0.0.0.0",
                        "port": 5001,
                        "max_connections": 200,
                        "enable_websocket": True,
                        "session_timeout": 3600,
                        "enable_compression": True,
                        "enable_metrics": True
                    },
                    "cors_config": {
                        "allow_origins": ["http://localhost:3000", "http://127.0.0.1:3000"],
                        "allow_credentials": True,
                        "allow_methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
                        "allow_headers": ["*"]
                    },
                    "session_management": {
                        "enable_sessions": True,
                        "session_id_header": "X-Session-ID",
                        "auto_create_sessions": True,
                        "session_storage": "memory"
                    },
                    "health_config": {
                        "enable_health_endpoint": True,
                        "health_path": "/api/health",
                        "enable_metrics_endpoint": True,
                        "metrics_path": "/api/metrics",
                        "enable_status_endpoint": True,
                        "status_path": "/api/status"
                    },
                    "websocket_config": {
                        "enable_websocket": True,
                        "websocket_path": "/ws",
                        "max_connections": 100,
                        "heartbeat_interval": 30,
                        "connection_timeout": 300
                    }
                }
            ],
            "nanobrain_metadata": {
                "framework_version": "2.0.0",
                "component_type": "web_interface_step",
                "config_loading_method": "from_config_only",
                "supports_recursive_references": True
            }
        }
    )


class WebInterfaceStep(BaseStep):
    """
    Web Interface Step - Pure HTTP Request/Response Handling within Workflow
    =======================================================================

    Provides pure HTTP interface functionality as a proper NanoBrain workflow step,
    enabling web interfaces to participate in event-driven workflow orchestration
    with complete framework compliance.

    **SINGLE RESPONSIBILITY**: HTTP Interface Only
    - HTTP request/response handling and conversion to/from DataUnits
    - Basic session tracking (ID generation and management)
    - WebSocket support for real-time communication
    - CORS and middleware configuration
    - Health and status endpoints

    **DOES NOT INCLUDE**:
    - Query classification (handled by QueryClassificationStep)
    - Workflow routing (handled by WorkflowRoutingStep) 
    - Multi-workflow support (composed via framework configuration)
    - Complex business logic (delegated to downstream steps)

    **Core Architecture:**
        This step provides a clean HTTP interface that integrates with NanoBrain's
        event-driven workflow orchestration by converting between HTTP protocols
        and framework DataUnits, enabling composable intelligent systems.

        * **HTTP Server**: FastAPI-based HTTP interface with proper middleware
        * **DataUnit Conversion**: HTTP ↔ DataUnit transformation for framework integration
        * **Session Management**: Basic session ID generation and tracking
        * **WebSocket Support**: Real-time communication channel for progress updates
        * **Health Monitoring**: Server status and basic resource monitoring

    **Configuration Architecture:**
        ```yaml
        # Basic web interface step configuration
        name: "pure_web_interface"
        description: "Pure HTTP interface for framework integration"
        auto_initialize: true
        enable_logging: true

        # Server configuration
        server_config:
          host: "0.0.0.0"
          port: 5001
          max_connections: 200
          enable_websocket: true
          session_timeout: 3600

        # CORS configuration
        cors_config:
          allow_origins:
            - "http://localhost:3000"
            - "http://127.0.0.1:3000"
          allow_credentials: true
          allow_methods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"]

        # Session management
        session_management:
          enable_sessions: true
          session_id_header: "X-Session-ID"
          auto_create_sessions: true

        # Health endpoints
        health_config:
          enable_health_endpoint: true
        ```

    **Usage Patterns:**
        ```python
        from nanobrain.library.infrastructure.steps import WebInterfaceStep

        # Create step from configuration
        web_step = WebInterfaceStep.from_config('config/web_interface_step.yml')

        # Execute within workflow context
        await web_step.execute()

        # Step automatically handles:
        # - HTTP request reception → RequestDataUnit
        # - ResponseDataUnit → HTTP response
        # - Session management
        # - WebSocket real-time updates
        ```

    Attributes:
        name (str): Step identifier for logging and debugging
        description (str): Human-readable step description
        logger (logging.Logger): Step-specific logger instance
        config (WebInterfaceStepConfig): Step configuration instance
        app (FastAPI): FastAPI application instance
        server_task (asyncio.Task): Background server task

    Note:
        This step follows the mandatory from_config pattern and cannot be
        instantiated directly. All configurations must be loaded from YAML files
        using the from_config method.

         See Also:
         * :class:`BaseStep`: Base framework step interface
         * :class:`WebInterfaceStepConfig`: Configuration schema
         * :class:`DataUnitMemory`: In-memory data storage for HTTP data
         * :class:`SessionContextDataUnit`: Session tracking data
    """

    # MANDATORY COMPONENT METADATA
    COMPONENT_TYPE: str = "web_interface_step"
    REQUIRED_CONFIG_FIELDS: List[str] = ['name', 'server_config']

    # Framework-compliant data units (simplified for pure interface)
    input_data_units = {
        # Responses from downstream processing
        'processed_responses': DataUnitMemory,
        'progress_updates': ProgressUpdateDataUnit    # Progress updates for streaming
    }

    output_data_units = {
        'http_requests': DataUnitMemory,              # Raw HTTP requests as DataUnits
        'session_context': SessionContextDataUnit    # Basic session tracking
    }

    # Pure interface triggers
    triggers = [
        TriggerBase,  # Base trigger class - specific triggers added in workflow
    ]

    # MANDATORY FRAMEWORK METHODS
    @classmethod
    def _get_config_class(cls) -> type:
        """
        Return the configuration class for this step.

        MANDATORY IMPLEMENTATION - This is the ONLY method that differs
        between step types in the unified framework pattern.

        Returns:
            Configuration class type for this step
        """
        return WebInterfaceStepConfig

    def _init_from_config(self, config: 'WebInterfaceStepConfig', component_config: Dict[str, Any],
                          dependencies: Dict[str, Any]) -> None:
        """
        Initialize step from validated configuration.

        MANDATORY IMPLEMENTATION - Step-specific initialization logic.

        Args:
            config: Validated configuration instance
            component_config: Component configuration dictionary
            dependencies: Component dependencies

        Raises:
            ComponentConfigurationError: If configuration is invalid
            ComponentDependencyError: If dependencies cannot be resolved
        """
        # Call parent initialization first
        super()._init_from_config(config, component_config, dependencies)

        # Store configuration
        self.config = config
        self.name = config.name
        self.description = config.description

        # Initialize logging (override parent logger with step-specific settings)
        self.logger = get_logger(
            f"web_interface.{self.name}",
            debug_mode=config.debug_mode
        )

        # Initialize HTTP server components
        self.app: Optional[FastAPI] = None
        self.server_task: Optional[asyncio.Task] = None
        self.websocket_connections: Set[WebSocket] = set()
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.request_counter: int = 0
        self.response_handlers: Dict[str, Callable] = {}

        # Performance metrics
        self.metrics = {
            'requests_total': 0,
            'responses_total': 0,
            'websocket_connections': 0,
            'active_sessions': 0,
            'server_start_time': None,
            'last_request_time': None
        }

        # Log component initialization
        self.logger.info(
            f"Initializing {self.__class__.__name__}",
            extra={
                "component_name": self.name,
                "component_type": self.COMPONENT_TYPE,
                "config_source": "from_config",
                "server_host": config.server_config.host,
                "server_port": config.server_config.port
            }
        )

        # Initialize FastAPI application
        self._initialize_fastapi_app()

        self.logger.info(
            f"{self.__class__.__name__} initialized successfully",
            extra={
                "component_name": self.name,
                "server_configured": self.app is not None,
                "websocket_enabled": config.websocket_config.get('enable_websocket', True)
            }
        )

    async def initialize(self) -> None:
        """
        Initialize the WebInterfaceStep and auto-start HTTP server if configured.

        This method extends the base initialization to automatically start the HTTP server
        when auto_initialize is True, making the WebInterfaceStep work properly in
        data-driven workflows where steps need to be ready to receive HTTP requests
        immediately upon initialization.
        """
        # Call parent initialization first
        await super().initialize()

        # Auto-start HTTP server if configured
        if self.config.auto_initialize:
            self.logger.info(
                "Auto-starting HTTP server due to auto_initialize=True")
            try:
                # Start the HTTP server directly during initialization
                await self._start_http_server()
                self.logger.info("HTTP server auto-started successfully")
            except Exception as e:
                self.logger.error(f"Failed to auto-start HTTP server: {e}")
                raise

    def _initialize_fastapi_app(self) -> None:
        """Initialize FastAPI application with middleware and routes."""

        # Create FastAPI app
        self.app = FastAPI(
            title="NanoBrain Web Interface",
            description="Pure HTTP interface for NanoBrain framework integration",
            version="2.0.0",
            docs_url="/docs" if self.config.debug_mode else None,
            redoc_url="/redoc" if self.config.debug_mode else None
        )

        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.cors_config.allow_origins,
            allow_credentials=self.config.cors_config.allow_credentials,
            allow_methods=self.config.cors_config.allow_methods,
            allow_headers=self.config.cors_config.allow_headers,
        )

        # Add compression middleware if enabled
        if self.config.server_config.enable_compression:
            self.app.add_middleware(GZipMiddleware, minimum_size=1000)

        # Setup routes
        self._setup_routes()

        self.logger.debug(
            "FastAPI application initialized",
            extra={
                "cors_origins": len(self.config.cors_config.allow_origins),
                "compression_enabled": self.config.server_config.enable_compression,
                "debug_mode": self.config.debug_mode
            }
        )

    def _setup_routes(self) -> None:
        """Setup HTTP routes and WebSocket endpoints."""

        # Health check endpoint
        if self.config.health_config.get('enable_health_endpoint', True):
            health_path = self.config.health_config.get(
                'health_path', '/api/health')

            @self.app.get(health_path)
            async def health_check():
                return await self._handle_health_check()

        # Metrics endpoint
        if self.config.health_config.get('enable_metrics_endpoint', True):
            metrics_path = self.config.health_config.get(
                'metrics_path', '/api/metrics')

            @self.app.get(metrics_path)
            async def metrics():
                return await self._handle_metrics()

        # Status endpoint
        if self.config.health_config.get('enable_status_endpoint', True):
            status_path = self.config.health_config.get(
                'status_path', '/api/status')

            @self.app.get(status_path)
            async def status():
                return await self._handle_status()

        # Main API endpoint for request processing
        @self.app.post("/api/v1/chat/")
        async def process_request(request: Request):
            self.logger.info("🔍 [ROUTE-DEBUG] /api/v1/chat/ route called")
            return await self._handle_http_request(request)

        # Legacy/Frontend compatible chat endpoint (same handler as v1)
        @self.app.post("/api/chat")
        async def chat_endpoint(request: Request):
            self.logger.info("🔍 [ROUTE-DEBUG] /api/chat route called")
            return await self._handle_http_request(request)

        # WebSocket endpoint
        if self.config.websocket_config.get('enable_websocket', True):
            websocket_path = self.config.websocket_config.get(
                'websocket_path', '/ws')

            @self.app.websocket(websocket_path)
            async def websocket_endpoint(websocket: WebSocket):
                await self._handle_websocket_connection(websocket)

        # Static files serving (for frontend)
        if self.config.static_files_config.get('enable_static_files', False):
            static_directory = self.config.static_files_config.get('static_directory', 'static')
            mount_path = self.config.static_files_config.get('mount_path', '/static')

            # Check if static directory exists
            static_path = Path(static_directory)
            if static_path.exists() and static_path.is_dir():
                self.app.mount(mount_path, StaticFiles(directory=static_directory, html=True), name="static")
                self.logger.info(f"🔍 [STATIC-FILES] Mounted {static_directory} at {mount_path}")

                # If mounting at root, add a catch-all route for SPA
                if mount_path == "/":
                    from fastapi.responses import FileResponse

                    @self.app.get("/{full_path:path}")
                    async def serve_spa(full_path: str):
                        # Get WebSocket path for exclusion
                        websocket_path = self.config.websocket_config.get('websocket_path', '/ws').lstrip('/')

                        # Serve index.html for all non-API routes (SPA routing)
                        # Exclude API routes, docs, redoc, static files, and WebSocket paths
                        if (not full_path.startswith("api/") and
                            not full_path.startswith("docs") and
                            not full_path.startswith("redoc") and
                            not full_path.startswith("static/") and
                            full_path != websocket_path):
                            index_path = Path(static_directory) / "index.html"
                            if index_path.exists():
                                return FileResponse(index_path)
                        # For API routes that don't exist, return 404
                        raise HTTPException(status_code=404, detail="Not found")
            else:
                self.logger.warning(f"⚠️ Static directory {static_directory} does not exist, skipping static files")

        # Debug: List all registered routes
        self.logger.info("🔍 [ROUTE-DEBUG] Registered FastAPI routes:")
        for route in self.app.routes:
            if hasattr(route, 'methods') and hasattr(route, 'path'):
                self.logger.info(f"🔍 [ROUTE-DEBUG] {route.methods} {route.path}")

        self.logger.debug(
            "HTTP routes and WebSocket endpoints configured",
            extra={
                "health_endpoint": self.config.health_config.get('enable_health_endpoint', True),
                "metrics_endpoint": self.config.health_config.get('enable_metrics_endpoint', True),
                "websocket_endpoint": self.config.websocket_config.get('enable_websocket', True)
            }
        )

    # HTTP REQUEST HANDLING
    async def _handle_http_request(self, request: Request) -> JSONResponse:
        """
        Handle HTTP request and convert to RequestDataUnit.

        Args:
            request: FastAPI request object

        Returns:
            JSON response
        """
        self.logger.info("🚀 _handle_http_request called - HTTP request received!")
        self.logger.info(f"🔍 [HTTP-DEBUG] Request method: {request.method}")
        self.logger.info(f"🔍 [HTTP-DEBUG] Request URL: {request.url}")
        self.logger.info(f"🔍 [HTTP-DEBUG] Request path: {request.url.path}")
        request_start_time = time.time()
        session_id = await self._get_or_create_session(request)
        self.logger.info(f"📋 Session ID: {session_id}")

        try:
            # Extract request data
            self.logger.info("📥 Extracting request body...")
            body = await request.body()
            self.logger.info(f"📦 Raw body extracted: {len(body) if body else 0} bytes")
            if body:
                try:
                    request_data = json.loads(body.decode('utf-8'))
                    self.logger.info(f"✅ Request data parsed: {request_data}")
                except json.JSONDecodeError:
                    request_data = {"raw_body": body.decode(
                        'utf-8', errors='ignore')}
                    self.logger.info(f"⚠️ JSON decode failed, using raw body: {request_data}")
            else:
                request_data = {}
                self.logger.info("📭 Empty request body")

            # Create RequestDataUnit
            request_data_unit = await self._create_request_data_unit(
                request=request,
                session_id=session_id,
                request_data=request_data,
                request_start_time=request_start_time
            )

            # Store request for processing (would be picked up by triggers)
            request_id = str(uuid.uuid4())
            self.pending_requests[request_id] = {
                'data_unit': request_data_unit,
                'start_time': request_start_time,
                'session_id': session_id
            }

            # ✅ CRITICAL FIX: Populate central http_requests data unit for routing
            self.logger.info("📤 Populating central http_requests data unit...")
            self.logger.info(f"🔍 [HTTP-DEBUG] Request ID: {request_id}")
            self.logger.info(f"🔍 [HTTP-DEBUG] Request data: {request_data}")
            await self._populate_central_http_requests_unit(
                request_id=request_id,
                session_id=session_id,
                request_data=request_data,
                request_start_time=request_start_time
            )
            self.logger.info("✅ Central http_requests data unit populated")
            self.logger.info("✅ Central http_requests data unit populated")

            # ✅ CRITICAL FIX: Update session context data unit
            self.logger.info("🔄 Updating session context data unit...")
            await self._update_session_context_unit(
                session_id=session_id,
                request_data=request_data
            )
            self.logger.info("✅ Session context data unit updated")

            # Update metrics
            self.metrics['requests_total'] += 1
            self.metrics['last_request_time'] = datetime.now(timezone.utc)

            # CRITICAL FIX: Wait for actual workflow processing via data flow
            # Monitor the processed_responses data unit for the response
            # Use configured request_timeout (900 seconds for BV-BRC operations)
            configured_timeout = getattr(self.config.server_config, 'request_timeout', 900)
            self.logger.info(f"⏳ Waiting for workflow response (timeout: {configured_timeout}s)...")
            response_data = await self._wait_for_workflow_response(
                request_id=request_id,
                session_id=session_id,
                timeout=float(configured_timeout)
            )
            self.logger.info(f"🎯 Workflow response received: {response_data}")

            # If no response received from workflow, return timeout error
            if response_data is None:
                response_data = {
                    "status": "timeout",
                    "request_id": request_id,
                    "session_id": session_id,
                    "message": "Request processing timeout - workflow did not respond in time",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }

            processing_time = time.time() - request_start_time

            self.logger.info(
                "HTTP request processed",
                extra={
                    "request_id": request_id,
                    "session_id": session_id,
                    "processing_time": processing_time,
                    "request_size": len(body) if body else 0
                }
            )

            return JSONResponse(content=response_data)

        except Exception as e:
            self.logger.error(
                f"Error processing HTTP request: {e}",
                extra={
                    "session_id": session_id,
                    "error_type": type(e).__name__,
                    "request_path": str(request.url)
                }
            )

            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": "Internal server error",
                    "error_code": "REQUEST_PROCESSING_ERROR"
                }
            )

    async def _wait_for_workflow_response(
        self,
        request_id: str,
        session_id: str,
        timeout: float = 30.0
    ) -> Optional[Dict[str, Any]]:
        """
        Wait for workflow to process request and return response via data flow.

        This implements the proper event-driven pattern:
        1. Request deposited in http_requests data unit
        2. Workflow processes through routing and execution
        3. Response arrives in processed_responses data unit
        4. We detect and return the response

        Args:
            request_id: Unique request identifier
            session_id: Session identifier
            timeout: Maximum wait time in seconds

        Returns:
            Response data from workflow or None if timeout
        """
        start_time = time.time()
        poll_interval = 0.05  # 50ms polling interval

        # Get the processed_responses data unit
        if 'processed_responses' not in self.step_input_data_units:
            self.logger.warning("processed_responses data unit not configured")
            return None

        responses_unit = self.step_input_data_units['processed_responses']

        # Poll for response until timeout
        while (time.time() - start_time) < timeout:
            try:
                # Check for response in the data unit
                responses_data = await responses_unit.get()

                if responses_data and isinstance(responses_data, dict):
                    # Check if this response matches our request
                    if responses_data.get('request_id') == request_id:
                        # Found our response!
                        self.logger.info(
                            f"Received workflow response for request {request_id}")

                        # Return the formatted response
                        return {
                            "status": "success",
                            "request_id": request_id,
                            "session_id": session_id,
                            "message": "Request processed successfully",
                            "response": responses_data.get('response', ''),
                            "workflow_type": responses_data.get('workflow_type', 'unknown'),
                            "processing_time": responses_data.get('processing_time', 0),
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        }

                    # Check if it's a batch response containing our request
                    batch_responses = responses_data.get('responses', [])
                    for response in batch_responses:
                        if response.get('request_id') == request_id:
                            self.logger.info(
                                f"Found response in batch for request {request_id}")
                            return {
                                "status": "success",
                                "request_id": request_id,
                                "session_id": session_id,
                                "message": "Request processed successfully",
                                "response": response.get('response', ''),
                                "workflow_type": response.get('workflow_type', 'unknown'),
                                "processing_time": response.get('processing_time', 0),
                                "timestamp": datetime.now(timezone.utc).isoformat()
                            }

            except Exception as e:
                self.logger.debug(f"Error checking for response: {e}")

            # Wait before next poll
            await asyncio.sleep(poll_interval)

        # Timeout reached
        self.logger.warning(
            f"Timeout waiting for response to request {request_id}")
        return None

    async def _handle_api_request(self, request: Request, path: str) -> JSONResponse:
        """
        Handle generic API requests.

        Args:
            request: FastAPI request object
            path: API path

        Returns:
            JSON response
        """
        session_id = await self._get_or_create_session(request)

        self.logger.debug(
            f"Generic API request to /{path}",
            extra={
                "session_id": session_id,
                "method": request.method,
                "path": path
            }
        )

        return JSONResponse(content={
            "status": "received",
            "path": path,
            "method": request.method,
            "session_id": session_id,
            "message": f"API endpoint /{path} received {request.method} request"
        })

    async def _create_request_data_unit(self, request: Request, session_id: str,
                                        request_data: Dict[str, Any],
                                        request_start_time: float) -> DataUnitMemory:
        """
        Create RequestDataUnit from HTTP request.

        Args:
            request: FastAPI request object
            session_id: Session identifier
            request_data: Parsed request data
            request_start_time: Request start timestamp

        Returns:
            DataUnitMemory instance with request data
        """
        # Create basic request data
        data_unit_data = {
            'request_id': str(uuid.uuid4()),
            'session_id': session_id,
            'timestamp': datetime.now(timezone.utc),
            'method': request.method,
            'url': str(request.url),
            'path': request.url.path,
            'query_params': dict(request.query_params),
            'headers': dict(request.headers),
            'body': request_data,
            'user_agent': request.headers.get('user-agent', ''),
            'client_ip': request.client.host if request.client else '',
            'content_type': request.headers.get('content-type', ''),
            'content_length': int(request.headers.get('content-length', 0)),
            'request_start_time': request_start_time
        }

        # Create and populate DataUnitMemory
        # Using inline config for nested data unit in step (allowed by framework)
        request_data_unit = DataUnitMemory.from_config({
            'class': 'nanobrain.core.data_unit.DataUnitMemory',
            'name': 'http_request',
            'description': 'HTTP request data',
            'cache_size': 100,
            'persistent': False
        })
        await request_data_unit.set(data_unit_data)

        return request_data_unit

    async def _populate_central_http_requests_unit(self, request_id: str, session_id: str,
                                                   request_data: Dict[str, Any],
                                                   request_start_time: float) -> None:
        """
        ✅ CRITICAL FIX: Populate the central http_requests data unit for routing.

        This method ensures that the central http_requests output data unit
        contains the standardized request data that RoutingStep expects.

        Args:
            request_id: Unique request identifier
            session_id: Session identifier
            request_data: Parsed request data from HTTP body
            request_start_time: Request start timestamp
        """
        try:
            self.logger.info("🔍 [HTTP-DEBUG] _populate_central_http_requests_unit called")
            self.logger.info(f"🔍 [HTTP-DEBUG] Available output data units: {list(self.step_output_data_units.keys())}")

            # Check if http_requests output data unit exists
            if 'http_requests' not in self.step_output_data_units:
                self.logger.warning(
                    "http_requests output data unit not configured - routing will not work")
                return

            http_requests_unit = self.step_output_data_units['http_requests']
            self.logger.info(f"🔍 [HTTP-DEBUG] http_requests_unit type: {type(http_requests_unit)}")

            # Extract user query from request data
            user_query = ""
            if isinstance(request_data, dict):
                user_query = request_data.get(
                    'query', request_data.get('message', ''))

            # Create standardized request format for routing
            routing_request = {
                'request_id': request_id,
                'session_id': session_id,
                'user_query': user_query,
                'request_type': request_data.get('request_type', 'chat') if isinstance(request_data, dict) else 'chat',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'metadata': {
                    'request_start_time': request_start_time,
                    'content_type': request_data.get('content_type', 'application/json') if isinstance(request_data, dict) else 'application/json',
                    'has_query': bool(user_query.strip()) if user_query else False,
                    'raw_request_data': request_data
                }
            }

            # Set data in central http_requests unit
            self.logger.info(f"🔍 [HTTP-DEBUG] Setting http_requests data unit with request {request_id}")
            self.logger.info(f"🔍 [HTTP-DEBUG] Routing request data: {routing_request}")
            await http_requests_unit.set(routing_request)
            self.logger.info("✅ http_requests data unit populated successfully")

            # Verify the data was set
            verification_data = await http_requests_unit.get()
            self.logger.info(f"🔍 [HTTP-DEBUG] Verification - data in http_requests unit: {verification_data}")

            self.logger.debug(
                f"Populated central http_requests data unit",
                extra={
                    "request_id": request_id,
                    "session_id": session_id,
                    "user_query_length": len(user_query) if user_query else 0,
                    "has_query": bool(user_query.strip()) if user_query else False
                }
            )

        except Exception as e:
            self.logger.error(
                f"Failed to populate central http_requests data unit: {e}")
            # Don't raise exception - this should not break HTTP request processing

    async def _update_session_context_unit(self, session_id: str, request_data: Dict[str, Any]) -> None:
        """
        ✅ CRITICAL FIX: Update the session context data unit.

        This method ensures that the session_context output data unit
        contains current session information for routing and processing.

        Args:
            session_id: Session identifier
            request_data: Parsed request data from HTTP body
        """
        try:
            # Check if session_context output data unit exists
            if 'session_context' not in self.step_output_data_units:
                self.logger.debug(
                    "session_context output data unit not configured")
                return

            session_unit = self.step_output_data_units['session_context']

            # Get current session data
            current_session = self.active_sessions.get(session_id, {})

            # Increment request count for this session
            if session_id in self.active_sessions:
                self.active_sessions[session_id]['request_count'] = self.active_sessions[session_id].get(
                    'request_count', 0) + 1

            # Create session context data
            session_data = {
                'session_id': session_id,
                'last_activity': datetime.now(timezone.utc).isoformat(),
                'request_count': current_session.get('request_count', 1),
                'user_context': {
                    'client_info': current_session.get('client_info', {}),
                    'session_created_at': current_session.get('created_at', datetime.now(timezone.utc)).isoformat() if hasattr(current_session.get('created_at', datetime.now(timezone.utc)), 'isoformat') else str(current_session.get('created_at', datetime.now(timezone.utc))),
                    'current_request_data': request_data if isinstance(request_data, dict) else {}
                },
                'conversation_history': [],  # Would be populated by conversation management
                'context_variables': {
                    'active_since': current_session.get('created_at', datetime.now(timezone.utc)).isoformat() if hasattr(current_session.get('created_at', datetime.now(timezone.utc)), 'isoformat') else str(current_session.get('created_at', datetime.now(timezone.utc))),
                    'total_requests': current_session.get('request_count', 1)
                }
            }

            # Set data in session context unit
            await session_unit.set(session_data)

            self.logger.debug(
                f"Updated session context data unit",
                extra={
                    "session_id": session_id,
                    "request_count": session_data['request_count']
                }
            )

        except Exception as e:
            self.logger.error(
                f"Failed to update session context data unit: {e}")
            # Don't raise exception - this should not break HTTP request processing

    # SESSION MANAGEMENT
    async def _get_or_create_session(self, request: Request) -> str:
        """
        Get existing session or create new one.

        Args:
            request: FastAPI request object

        Returns:
            Session identifier
        """
        if not self.config.session_management.get('enable_sessions', True):
            return f"no_session_{uuid.uuid4()}"

        session_header = self.config.session_management.get(
            'session_id_header', 'X-Session-ID')
        session_id = request.headers.get(session_header)

        if session_id and session_id in self.active_sessions:
            # Update last activity
            self.active_sessions[session_id]['last_activity'] = datetime.now(
                timezone.utc)
            return session_id

        # Create new session if auto-creation is enabled
        if self.config.session_management.get('auto_create_sessions', True):
            session_id = str(uuid.uuid4())

            self.active_sessions[session_id] = {
                'session_id': session_id,
                'created_at': datetime.now(timezone.utc),
                'last_activity': datetime.now(timezone.utc),
                'request_count': 0,
                'client_info': {
                    'user_agent': request.headers.get('user-agent', ''),
                    'client_ip': request.client.host if request.client else ''
                }
            }

            self.metrics['active_sessions'] = len(self.active_sessions)

            self.logger.debug(
                "New session created",
                extra={
                    "session_id": session_id,
                    "client_ip": request.client.host if request.client else 'unknown'
                }
            )

            return session_id

        return f"anonymous_{uuid.uuid4()}"

    # WEBSOCKET HANDLING
    async def _handle_websocket_connection(self, websocket: WebSocket) -> None:
        """
        Handle WebSocket connection for real-time updates.

        Args:
            websocket: WebSocket connection
        """
        await websocket.accept()
        self.websocket_connections.add(websocket)
        self.metrics['websocket_connections'] = len(self.websocket_connections)

        connection_id = str(uuid.uuid4())

        self.logger.info(
            "WebSocket connection established",
            extra={
                "connection_id": connection_id,
                "total_connections": len(self.websocket_connections)
            }
        )

        try:
            # Send initial connection message
            await websocket.send_json({
                "type": "connection_established",
                "connection_id": connection_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            # Keep connection alive and handle messages
            heartbeat_interval = self.config.websocket_config.get(
                'heartbeat_interval', 30)

            while True:
                try:
                    # Wait for messages with timeout for heartbeat
                    data = await asyncio.wait_for(
                        websocket.receive_json(),
                        timeout=heartbeat_interval
                    )

                    # Handle received message
                    await self._handle_websocket_message(websocket, data, connection_id)

                except asyncio.TimeoutError:
                    # Send heartbeat
                    await websocket.send_json({
                        "type": "heartbeat",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })

        except WebSocketDisconnect:
            self.logger.info(
                "WebSocket connection disconnected",
                extra={"connection_id": connection_id}
            )
        except Exception as e:
            self.logger.error(
                f"WebSocket error: {e}",
                extra={
                    "connection_id": connection_id,
                    "error_type": type(e).__name__
                }
            )
        finally:
            self.websocket_connections.discard(websocket)
            self.metrics['websocket_connections'] = len(
                self.websocket_connections)

    async def _handle_websocket_message(self, websocket: WebSocket, data: Dict[str, Any], connection_id: str) -> None:
        """
        Handle received WebSocket message.

        Args:
            websocket: WebSocket connection
            data: Received message data
            connection_id: Connection identifier
        """
        message_type = data.get('type', 'unknown')

        self.logger.debug(
            f"WebSocket message received: {message_type}",
            extra={
                "connection_id": connection_id,
                "message_type": message_type
            }
        )

        # Echo message back (simple implementation)
        await websocket.send_json({
            "type": "message_received",
            "original_type": message_type,
            "connection_id": connection_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

    # HEALTH AND STATUS ENDPOINTS
    async def _handle_health_check(self) -> Dict[str, Any]:
        """Handle health check request."""
        uptime = None
        if self.metrics['server_start_time']:
            uptime = (datetime.now(timezone.utc) -
                      self.metrics['server_start_time']).total_seconds()

        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "uptime_seconds": uptime,
            "component": self.name,
            "version": "2.0.0"
        }

    async def _handle_metrics(self) -> Dict[str, Any]:
        """Handle metrics request."""
        return {
            "metrics": self.metrics.copy(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "component": self.name
        }

    async def _handle_status(self) -> Dict[str, Any]:
        """Handle detailed status request."""
        return {
            "status": "operational",
            "component": self.name,
            "component_type": self.COMPONENT_TYPE,
            "server_config": {
                "host": self.config.server_config.host,
                "port": self.config.server_config.port,
                "websocket_enabled": self.config.websocket_config.get('enable_websocket', True)
            },
            "active_connections": {
                "http_sessions": len(self.active_sessions),
                "websocket_connections": len(self.websocket_connections)
            },
            "metrics": self.metrics.copy(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    # STEP PROCESSING (MANDATORY ABSTRACT METHOD)
    async def process(self, input_data: Dict[str, Any], **kwargs) -> Any:
        """
        Process input data and return result.

        MANDATORY IMPLEMENTATION for all steps.

        For WebInterfaceStep, this method handles the core HTTP interface
        logic by processing incoming HTTP requests and generating appropriate
        responses through the framework's data flow.

        Args:
            input_data: Dictionary of input data from registered data units
            **kwargs: Additional parameters

        Returns:
            Processing results with HTTP response data
        """
        try:
            self.logger.debug("Processing WebInterfaceStep data flow")

            # Process incoming HTTP requests from data units
            http_requests = input_data.get('http_requests', {})
            processed_responses = input_data.get('processed_responses', {})
            progress_updates = input_data.get('progress_updates', {})

            # Handle HTTP request processing
            results = {
                'http_responses': {},
                'session_context': {},
                'status': 'processed',
                'processing_time': 0.0
            }

            # In a full implementation, this would:
            # 1. Extract HTTP request data from input_data
            # 2. Route requests to appropriate handlers
            # 3. Generate HTTP responses from processed_responses
            # 4. Update session context
            # 5. Stream progress updates via WebSocket

            self.logger.debug("WebInterfaceStep processing completed")

            return results

        except Exception as e:
            self.logger.error(f"WebInterfaceStep processing failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'http_responses': {},
                'session_context': {}
            }

    async def _start_http_server(self) -> Dict[str, Any]:
        """
        Start the HTTP server (extracted from execute for initialization use).

        Returns:
            Server startup results
        """
        try:
            self.logger.info(
                f"Starting HTTP server on {self.config.server_config.host}:{self.config.server_config.port}")

            # Initialize pending requests storage
            self.pending_requests: Dict[str, Dict[str, Any]] = {}

            # Update server start time
            self.metrics['server_start_time'] = datetime.now(timezone.utc)

            # Start server in background task
            server_config = uvicorn.Config(
                app=self.app,
                host=self.config.server_config.host,
                port=self.config.server_config.port,
                log_level="info" if self.config.debug_mode else "warning",
                access_log=self.config.debug_mode
            )

            server = uvicorn.Server(server_config)
            self.server_task = asyncio.create_task(server.serve())

            self.logger.info(
                f"HTTP server started successfully",
                extra={
                    "host": self.config.server_config.host,
                    "port": self.config.server_config.port,
                    "websocket_enabled": self.config.websocket_config.get('enable_websocket', True)
                }
            )

            return {
                "status": "running",
                "server_host": self.config.server_config.host,
                "server_port": self.config.server_config.port,
                "websocket_enabled": self.config.websocket_config.get('enable_websocket', True),
                "start_time": self.metrics['server_start_time']
            }

        except Exception as e:
            self.logger.error(f"Failed to start HTTP server: {e}")
            raise ComponentDependencyError(f"HTTP server startup failed: {e}")

    # STEP EXECUTION
    async def execute(self) -> Dict[str, Any]:
        """
        Execute the web interface step (start HTTP server if not already started).

        MANDATORY IMPLEMENTATION for all steps.

        Returns:
            Step execution results
        """
        # If server is already running from auto_initialize, return status
        if hasattr(self, 'server_task') and self.server_task and not self.server_task.done():
            return {
                "status": "already_running",
                "server_host": self.config.server_config.host,
                "server_port": self.config.server_config.port,
                "websocket_enabled": self.config.websocket_config.get('enable_websocket', True),
                "start_time": self.metrics.get('server_start_time')
            }

        # Otherwise start the server
        return await self._start_http_server()

    # CLEANUP METHODS
    async def shutdown(self) -> None:
        """Shutdown the web interface step."""
        try:
            # Close WebSocket connections
            for websocket in self.websocket_connections.copy():
                try:
                    await websocket.close()
                except Exception:
                    pass

            # Cancel server task
            if self.server_task and not self.server_task.done():
                self.server_task.cancel()
                try:
                    await self.server_task
                except asyncio.CancelledError:
                    pass

            self.logger.info("Web interface step shutdown completed")

        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

    def cleanup(self) -> None:
        """
        Clean up step resources.

        RECOMMENDED IMPLEMENTATION - Override for step-specific cleanup.
        """
        self.logger.info(f"Cleaning up {self.__class__.__name__}")

        # Clear in-memory data
        self.active_sessions.clear()
        self.websocket_connections.clear()
        if hasattr(self, 'pending_requests'):
            self.pending_requests.clear()

        # Reset metrics
        self.metrics = {
            'requests_total': 0,
            'responses_total': 0,
            'websocket_connections': 0,
            'active_sessions': 0,
            'server_start_time': None,
            'last_request_time': None
        }
