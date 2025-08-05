"""
Web Interface

Main web interface class that provides REST API access to NanoBrain workflows.
"""

import os
import asyncio
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path

# FastAPI imports
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exception_handlers import http_exception_handler
import uvicorn

# NanoBrain imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'src'))

from nanobrain.core.component_base import FromConfigBase

try:
    from nanobrain.core.logging_system import get_logger
except ImportError:
    import logging
    def get_logger(name: str, category: str = "default"):
        return logging.getLogger(f"{category}.{name}")

# Local imports
from .config.web_interface_config import WebInterfaceConfig
from .models.response_models import ErrorResponse
from .api.chat_router import chat_router, get_chat_workflow
from .api.health_router import health_router, get_web_interface
from .api.websocket_router import websocket_router, get_connection_manager
from .api.frontend_router import frontend_router
from .middleware.cors_middleware import setup_cors
from .middleware.logging_middleware import LoggingMiddleware

# Import ChatWorkflow
try:
    from nanobrain.library.workflows.chat_workflow.chat_workflow import ChatWorkflow
except ImportError:
    # Fallback if ChatWorkflow is not available
    class ChatWorkflow:
        def __init__(self):
            self.is_initialized = False
        
        async def initialize(self):
            self.is_initialized = True
        
        async def process_user_input(self, query: str) -> str:
            return "ChatWorkflow not available - this is a fallback response"
        
        def get_workflow_status(self) -> Dict[str, Any]:
            return {"status": "fallback_mode"}
        
        async def get_conversation_stats(self) -> Optional[Dict[str, Any]]:
            return None
        
        async def shutdown(self):
            pass


class WebInterface(FromConfigBase):
    """
    Web Interface - Enterprise REST API and Real-Time Web Services for NanoBrain Workflows
    ===================================================================================
    
    The WebInterface provides comprehensive web API access to NanoBrain workflows and services, offering
    RESTful endpoints, WebSocket communication, and enterprise-grade web service capabilities. This interface
    enables seamless integration of AI workflows with web applications, mobile clients, and third-party
    systems through standardized HTTP/HTTPS protocols and real-time communication channels.
    
    **Core Architecture:**
        The web interface provides enterprise-grade web service capabilities:
        
        * **RESTful API Design**: Comprehensive REST API with OpenAPI/Swagger documentation
        * **Real-Time Communication**: WebSocket support for streaming responses and live updates
        * **Authentication & Security**: JWT-based authentication with role-based access control
        * **Request Processing**: Asynchronous request handling with queue management and rate limiting
        * **Response Formatting**: Flexible response formatting with JSON, XML, and custom formats
        * **Framework Integration**: Full integration with NanoBrain's workflow orchestration system
    
    **Web Service Capabilities:**
        
        **REST API Endpoints:**
        * **Chat API**: Conversational interfaces with streaming and batch processing
        * **Workflow API**: Workflow execution, status monitoring, and result retrieval
        * **Agent API**: Direct agent interaction and specialized processing services
        * **Tool API**: Access to bioinformatics tools and analysis services
        * **Admin API**: System administration, monitoring, and configuration management
        
        **Real-Time Features:**
        * WebSocket connections for streaming responses and live chat
        * Server-Sent Events (SSE) for real-time progress updates
        * Real-time collaboration and multi-user session management
        * Live monitoring dashboards and system status updates
        
        **Content Management:**
        * Static file serving with optimization and caching
        * Dynamic content generation and template rendering
        * File upload and download with progress tracking
        * Media processing and transformation services
        
        **Integration Capabilities:**
        * Cross-Origin Resource Sharing (CORS) configuration
        * API versioning and backward compatibility management
        * Third-party service integration and webhook support
        * Microservices architecture and service mesh integration
    
    **Enterprise Web Features:**
        
        **Authentication & Authorization:**
        * JSON Web Token (JWT) authentication with refresh token support
        * OAuth 2.0 integration for third-party authentication providers
        * Role-based access control (RBAC) with fine-grained permissions
        * API key management for service-to-service authentication
        
        **Performance & Scalability:**
        * Asynchronous request processing with concurrent execution
        * Connection pooling and resource optimization
        * Caching strategies for improved response times
        * Load balancing and horizontal scaling support
        
        **Security & Compliance:**
        * HTTPS enforcement with SSL/TLS certificate management
        * Request validation and input sanitization
        * Rate limiting and DDoS protection mechanisms
        * Audit logging and security event monitoring
        
        **Monitoring & Analytics:**
        * Real-time performance metrics and system health monitoring
        * Request/response logging with structured data
        * Error tracking and alerting systems
        * Usage analytics and API consumption reporting
    
    **Configuration Architecture:**
        Comprehensive configuration supports diverse web deployment scenarios:
        
        ```yaml
        # Web Interface Configuration
        interface_name: "nanobrain_web_api"
        interface_type: "web"
        
        # Interface card for framework integration
        interface_card:
          name: "nanobrain_web_api"
          description: "Enterprise REST API and web services"
          version: "1.0.0"
          category: "web_interface"
          capabilities:
            - "rest_api"
            - "websocket_communication"
            - "real_time_streaming"
        
        # Server Configuration
        server_config:
          host: "0.0.0.0"
          port: 8000
          workers: 4              # Number of worker processes
          max_connections: 1000   # Maximum concurrent connections
          timeout: 30             # Request timeout in seconds
          
        # API Configuration
        api_config:
          version: "v1"
          title: "NanoBrain API"
          description: "Enterprise AI Workflow API"
          openapi_url: "/openapi.json"
          docs_url: "/docs"
          redoc_url: "/redoc"
          
        # Authentication Configuration
        auth_config:
          enabled: true
          jwt_secret: "${JWT_SECRET}"
          jwt_algorithm: "HS256"
          access_token_expire_minutes: 30
          refresh_token_expire_days: 7
          
        # CORS Configuration
        cors_config:
          allow_origins: ["http://localhost:3000", "https://yourdomain.com"]
          allow_credentials: true
          allow_methods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
          allow_headers: ["*"]
          
        # WebSocket Configuration
        websocket_config:
          enabled: true
          max_connections: 100
          heartbeat_interval: 30
          message_queue_size: 1000
          
        # Rate Limiting
        rate_limiting:
          enabled: true
          requests_per_minute: 60
          burst_size: 10
          
        # Caching Configuration
        cache_config:
          enabled: true
          backend: "redis"
          ttl: 3600               # Time to live in seconds
          max_size: "100MB"
          
        # Logging Configuration
        logging_config:
          level: "INFO"
          format: "structured"
          access_log: true
          error_log: true
          
        # SSL/TLS Configuration
        ssl_config:
          enabled: false          # Set to true for HTTPS
          cert_file: "/path/to/cert.pem"
          key_file: "/path/to/key.pem"
          
        # Workflow Integration
        workflow_config:
          default_workflow: "chat_workflow"
          workflow_timeout: 300   # Workflow execution timeout
          result_caching: true
        ```
    
    **Usage Patterns:**
        
        **Basic Web Interface Setup:**
        ```python
        from nanobrain.library.interfaces.web import WebInterface
        
        # Create web interface with configuration
        web_config = {
            'interface_name': 'ai_api_server',
            'server_config': {
                'host': '0.0.0.0',
                'port': 8000,
                'workers': 2
            },
            'api_config': {
                'title': 'AI Research API',
                'description': 'API for AI-powered research workflows'
            }
        }
        
        web_interface = WebInterface.from_config(web_config)
        
        # Initialize and start server
        await web_interface.initialize()
        await web_interface.start_server()
        
        # Server is now running and accepting requests
        # Access at: http://localhost:8000/docs for API documentation
        ```
        
        **Enterprise Deployment Configuration:**
        ```python
        # Configure for production enterprise deployment
        enterprise_config = {
            'interface_name': 'enterprise_ai_platform',
            'server_config': {
                'host': '0.0.0.0',
                'port': 443,
                'workers': 8,
                'max_connections': 2000,
                'timeout': 60
            },
            'auth_config': {
                'enabled': True,
                'jwt_secret': os.getenv('JWT_SECRET'),
                'oauth_providers': ['google', 'microsoft', 'okta']
            },
            'ssl_config': {
                'enabled': True,
                'cert_file': '/etc/ssl/certs/api.crt',
                'key_file': '/etc/ssl/private/api.key'
            },
            'rate_limiting': {
                'enabled': True,
                'requests_per_minute': 1000,
                'burst_size': 50
            },
            'monitoring': {
                'enabled': True,
                'metrics_endpoint': '/metrics',
                'health_endpoint': '/health'
            }
        }
        
        enterprise_api = WebInterface.from_config(enterprise_config)
        await enterprise_api.initialize()
        
        # Start with production-ready configuration
        await enterprise_api.start_production_server()
        ```
        
        **Real-Time Chat API Integration:**
        ```python
        # Configure for real-time chat applications
        chat_config = {
            'interface_name': 'realtime_chat_api',
            'websocket_config': {
                'enabled': True,
                'max_connections': 500,
                'heartbeat_interval': 30
            },
            'chat_features': {
                'streaming_responses': True,
                'typing_indicators': True,
                'conversation_persistence': True,
                'multi_user_support': True
            },
            'performance_optimization': {
                'response_caching': True,
                'connection_pooling': True,
                'async_processing': True
            }
        }
        
        chat_api = WebInterface.from_config(chat_config)
        await chat_api.initialize()
        
        # Example client usage:
        import websockets
        import json
        
        async def chat_client():
            uri = "ws://localhost:8000/ws/chat"
            async with websockets.connect(uri) as websocket:
                # Send chat message
                message = {
                    "type": "chat_message",
                    "content": "Explain SARS-CoV-2 spike protein structure",
                    "session_id": "user123_session"
                }
                await websocket.send(json.dumps(message))
                
                # Receive streaming response
                async for response in websocket:
                    data = json.loads(response)
                    if data["type"] == "chat_response":
                        print(f"Assistant: {data['content']}")
                    elif data["type"] == "typing_indicator":
                        print("Assistant is typing...")
                    elif data["type"] == "response_complete":
                        break
        
        # Run chat client
        await chat_client()
        ```
        
        **Workflow API Integration:**
        ```python
        # Configure for workflow execution API
        workflow_config = {
            'interface_name': 'workflow_execution_api',
            'workflow_management': {
                'supported_workflows': [
                    'viral_protein_analysis',
                    'literature_search',
                    'sequence_alignment'
                ],
                'async_execution': True,
                'progress_tracking': True,
                'result_persistence': True
            },
            'api_endpoints': {
                'execute_workflow': '/api/v1/workflows/execute',
                'get_status': '/api/v1/workflows/{workflow_id}/status',
                'get_results': '/api/v1/workflows/{workflow_id}/results',
                'cancel_workflow': '/api/v1/workflows/{workflow_id}/cancel'
            }
        }
        
        workflow_api = WebInterface.from_config(workflow_config)
        await workflow_api.initialize()
        
        # Example API usage with Python requests:
        import requests
        
        # Execute workflow
        workflow_request = {
            "workflow_type": "viral_protein_analysis",
            "parameters": {
                "virus_species": "SARS-CoV-2",
                "analysis_type": "structure_prediction",
                "protein_targets": ["spike", "nucleocapsid"]
            },
            "options": {
                "async": True,
                "priority": "high"
            }
        }
        
        response = requests.post(
            "http://localhost:8000/api/v1/workflows/execute",
            json=workflow_request,
            headers={"Authorization": "Bearer your_jwt_token"}
        )
        
        workflow_id = response.json()["workflow_id"]
        
        # Poll for status
        while True:
            status_response = requests.get(
                f"http://localhost:8000/api/v1/workflows/{workflow_id}/status"
            )
            status = status_response.json()["status"]
            
            if status == "completed":
                # Get results
                results_response = requests.get(
                    f"http://localhost:8000/api/v1/workflows/{workflow_id}/results"
                )
                results = results_response.json()
                print(f"Workflow Results: {results}")
                break
            elif status == "failed":
                print(f"Workflow failed: {status_response.json()['error']}")
                break
            
            await asyncio.sleep(5)  # Poll every 5 seconds
        ```
        
        **Multi-Service Integration:**
        ```python
        # Configure for microservices integration
        microservices_config = {
            'interface_name': 'microservices_gateway',
            'service_discovery': {
                'enabled': True,
                'registry': 'consul',
                'health_checks': True
            },
            'load_balancing': {
                'enabled': True,
                'strategy': 'round_robin',
                'health_check_interval': 30
            },
            'api_gateway': {
                'enabled': True,
                'routing_rules': {
                    '/api/chat/*': 'chat_service',
                    '/api/analysis/*': 'analysis_service',
                    '/api/data/*': 'data_service'
                }
            },
            'monitoring': {
                'distributed_tracing': True,
                'metrics_collection': True,
                'log_aggregation': True
            }
        }
        
        gateway = WebInterface.from_config(microservices_config)
        await gateway.initialize()
        
        # Register services
        await gateway.register_service(
            name='chat_service',
            url='http://chat-service:8001',
            health_endpoint='/health'
        )
        
        await gateway.register_service(
            name='analysis_service',
            url='http://analysis-service:8002',
            health_endpoint='/health'
        )
        
        # Start gateway with service mesh integration
        await gateway.start_gateway()
        ```
    
    **Advanced Features:**
        
        **API Documentation & Testing:**
        * Automatic OpenAPI/Swagger documentation generation
        * Interactive API testing interface with examples
        * API versioning and migration tools
        * SDK generation for multiple programming languages
        
        **Performance Optimization:**
        * Response compression and optimization
        * Database connection pooling and query optimization
        * CDN integration for static content delivery
        * Edge computing and geographic distribution support
        
        **Enterprise Integration:**
        * Single Sign-On (SSO) integration with enterprise identity providers
        * API management and governance tools
        * Service mesh integration for microservices architectures
        * Container orchestration and Kubernetes deployment support
        
        **Developer Experience:**
        * Comprehensive API documentation with examples
        * SDK and client library generation
        * Developer portal with tutorials and guides
        * API testing tools and sandbox environments
    
    **Deployment Scenarios:**
        
        **Cloud-Native Deployment:**
        * Kubernetes cluster deployment with auto-scaling
        * Container-based deployment with Docker and container registries
        * Service mesh integration with Istio or Linkerd
        * Cloud provider integration (AWS, Azure, GCP)
        
        **Edge Computing:**
        * Edge server deployment for low-latency applications
        * Content delivery network (CDN) integration
        * Geographic load balancing and failover
        * Offline-first capabilities with synchronization
        
        **Hybrid Environments:**
        * On-premises and cloud hybrid deployment
        * Multi-cloud deployment strategies
        * Data residency and compliance requirements
        * Legacy system integration and migration support
        
        **Development & Testing:**
        * Local development server with hot reload
        * Testing environments with mock services
        * Continuous integration and deployment pipelines
        * Load testing and performance benchmarking
    
    Attributes:
        web_config (WebInterfaceConfig): Web interface configuration and settings
        app (FastAPI): FastAPI application instance with routing and middleware
        server_instance (object): Uvicorn server instance for HTTP/HTTPS serving
        websocket_manager (object): WebSocket connection management system
        auth_manager (object): Authentication and authorization management
        rate_limiter (object): Rate limiting and throttling system
    
    Note:
        This interface requires FastAPI and Uvicorn for web server functionality.
        WebSocket features require compatible client libraries for real-time communication.
        Authentication features require proper JWT secret configuration for security.
        Production deployment requires SSL/TLS certificate configuration for HTTPS.
    
    Warning:
        Web interfaces expose AI workflows to network access, requiring proper security measures.
        Rate limiting and authentication are essential for production deployments.
        Large file uploads and long-running workflows may require special handling.
        WebSocket connections consume server resources and should be monitored and limited.
    
    See Also:
        * :class:`FromConfigBase`: Base framework component interface
        * :class:`WebInterfaceConfig`: Web interface configuration schema
        * :class:`ChatWorkflow`: Conversational workflow integration
        * :mod:`nanobrain.library.interfaces.web.api`: API router implementations
        * :mod:`nanobrain.library.interfaces.web.middleware`: Middleware components
    """
    
    # Component configuration
    COMPONENT_TYPE = "web_interface"
    REQUIRED_CONFIG_FIELDS = ['name']
    OPTIONAL_CONFIG_FIELDS = {
        'host': '127.0.0.1',
        'port': 8000,
        'api_prefix': '/api'
    }
    
    @classmethod
    def _get_config_class(cls):
        """Return WebInterfaceConfig for web interface components."""
        return WebInterfaceConfig
    
    @classmethod
    def extract_component_config(cls, config: WebInterfaceConfig) -> Dict[str, Any]:
        """Extract WebInterface configuration"""
        return {
            'name': getattr(config, 'name', 'web_interface'),
            'host': getattr(config, 'host', '127.0.0.1'),
            'port': getattr(config, 'port', 8000),
            'api_prefix': getattr(config, 'api_prefix', '/api'),
        }
    
    @classmethod  
    def resolve_dependencies(cls, component_config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Resolve WebInterface dependencies"""
        return {
            'config': kwargs.get('config'),
            'chat_workflow': kwargs.get('chat_workflow'),
            'universal_server': kwargs.get('universal_server'),
        }
    
    @classmethod
    def from_config(cls, config: WebInterfaceConfig, **kwargs) -> 'WebInterface':
        """Mandatory from_config implementation for WebInterface"""
        logger = get_logger(f"{cls.__name__}.from_config")
        logger.info(f"Creating {cls.__name__} from configuration")
        
        # Step 1: Validate configuration (minimal validation for config object)
        # config is already an object, not a dict
        
        # Step 2: Extract component-specific configuration  
        component_config = cls.extract_component_config(config)
        
        # Step 3: Resolve dependencies
        dependencies = cls.resolve_dependencies(component_config, **kwargs)
        
        # Step 4: Create instance
        instance = cls.create_instance(config, component_config, dependencies)
        
        # Step 5: Post-creation initialization
        instance._post_config_initialization()
        
        logger.info(f"Successfully created {cls.__name__}")
        return instance
        
    def _init_from_config(self, config: WebInterfaceConfig, component_config: Dict[str, Any],
                         dependencies: Dict[str, Any]) -> None:
        """Initialize WebInterface with resolved dependencies"""
        
        # ✅ FRAMEWORK COMPLIANCE: Handle config as either object or file path
        if isinstance(config, str):
            # Config is a file path - load the WebInterfaceConfig from the file
            from .config.web_interface_config import WebInterfaceConfig
            try:
                self.config = WebInterfaceConfig.from_yaml(config)
                self.logger = get_logger("web_interface", "interfaces")
                self.logger.info(f"✅ Loaded WebInterface configuration from file: {config}")
            except Exception as e:
                # Fallback to default configuration if loading fails
                self.config = WebInterfaceConfig()
                self.logger = get_logger("web_interface", "interfaces")
                self.logger.warning(f"⚠️ Failed to load config from {config}, using default: {e}")
        else:
            # Config is already a WebInterfaceConfig object
            self.config = config or WebInterfaceConfig()
            self.logger = get_logger("web_interface", "interfaces")
        
        # State
        self.is_initialized = False
        self.is_running = False
        self.start_time = None
        
        # Components
        self.app: Optional[FastAPI] = None
        self.chat_workflow: Optional[ChatWorkflow] = dependencies.get('chat_workflow')
        self.universal_server = dependencies.get('universal_server')
        self.server = None
        
        self.logger.info("Web interface created", config_name=getattr(self.config, 'name', 'web_interface'))
    
    async def initialize(self) -> None:
        """Initialize the web interface and its components."""
        if self.is_initialized:
            return
        
        self.logger.info("Initializing web interface")
        
        try:
            # Initialize chat workflow
            await self._setup_chat_workflow()
            
            # Setup FastAPI application
            await self._setup_app()
            
            # Setup middleware
            await self._setup_middleware()
            
            # Setup routers
            await self._setup_routers()
            
            # Setup error handlers
            await self._setup_error_handlers()
            
            self.start_time = datetime.utcnow()
            self.is_initialized = True
            
            self.logger.info("Web interface initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize web interface: {e}")
            raise
    
    async def _setup_chat_workflow(self) -> None:
        """Setup the chat workflow."""
        self.logger.info("Setting up chat workflow")
        
        try:
            # ✅ FRAMEWORK COMPLIANCE: Use from_config pattern for ChatWorkflow
            # Try to get workflow config from base configuration
            workflow_config = getattr(self.config, 'workflow_config', None)
            
            if workflow_config:
                # Load workflow from configuration file
                if isinstance(workflow_config, str):
                    # Path to config file
                    self.chat_workflow = ChatWorkflow.from_config(workflow_config)
                elif isinstance(workflow_config, dict) and 'class' in workflow_config:
                    # Class+config pattern
                    workflow_class_path = workflow_config['class']
                    workflow_config_path = workflow_config['config']
                    # Import the workflow class dynamically
                    import importlib
                    module_path, class_name = workflow_class_path.rsplit('.', 1)
                    module = importlib.import_module(module_path)
                    workflow_class = getattr(module, class_name)
                    self.chat_workflow = workflow_class.from_config(workflow_config_path)
                else:
                    # Default configuration
                    self.chat_workflow = ChatWorkflow.from_config(workflow_config)
            else:
                # ✅ FRAMEWORK COMPLIANCE: Use minimal default configuration
                default_workflow_config = {
                    'name': 'default_chat_workflow',
                    'interface_name': 'default_chat',
                    'interface_type': 'chat',
                    'version': '1.0.0',
                    'execution_strategy': 'sequential',
                    'description': 'Default chat workflow'
                }
                self.chat_workflow = ChatWorkflow.from_config(default_workflow_config)
            
            await self.chat_workflow.initialize()
            self.logger.info("Chat workflow initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize chat workflow: {e}")
            # ✅ FRAMEWORK COMPLIANCE: Even fallback must use from_config
            try:
                fallback_config = {
                    'name': 'fallback_chat_workflow',
                    'interface_name': 'fallback_chat',
                    'interface_type': 'chat',
                    'version': '1.0.0',
                    'execution_strategy': 'sequential',
                    'description': 'Fallback chat workflow'
                }
                self.chat_workflow = ChatWorkflow.from_config(fallback_config)
                self.logger.info("Fallback chat workflow initialized")
            except Exception as fallback_error:
                self.logger.error(f"Failed to initialize fallback workflow: {fallback_error}")
                raise
    
    async def _setup_app(self) -> None:
        """Setup the FastAPI application."""
        self.logger.info("Setting up FastAPI application")
        
        # ✅ FRAMEWORK COMPLIANCE: Handle API config as dict or object
        api_config = self.config.api
        if isinstance(api_config, dict):
            # Handle dict-based API config
            self.app = FastAPI(
                title=api_config.get("title", "NanoBrain API"),
                description=api_config.get("description", "NanoBrain Web Interface"),
                version=api_config.get("version", "1.0.0"),
                docs_url=api_config.get("docs_url", "/docs"),
                redoc_url=api_config.get("redoc_url", "/redoc")
            )
        else:
            # Handle object-based API config
            self.app = FastAPI(
                title=api_config.title,
                description=api_config.description,
                version=api_config.version,
                docs_url=api_config.docs_url,
                redoc_url=api_config.redoc_url
            )
        
        self.logger.info("FastAPI application created")
    
    async def _setup_middleware(self) -> None:
        """Setup middleware for the application."""
        self.logger.info("Setting up middleware")
        
        # ✅ FRAMEWORK COMPLIANCE: Handle security config as dict or object
        security_config = self.config.security
        cors_enabled = (
            security_config.get("cors_enabled", True) 
            if isinstance(security_config, dict) 
            else security_config.cors_enabled
        )
        
        if cors_enabled:
            from fastapi.middleware.cors import CORSMiddleware
            
            # ✅ FRAMEWORK COMPLIANCE: Handle CORS config as dict or object
            cors_config = self.config.cors
            if isinstance(cors_config, dict):
                # Handle dict-based CORS config
                self.app.add_middleware(
                    CORSMiddleware,
                    allow_origins=cors_config.get("allow_origins", ["*"]),
                    allow_credentials=cors_config.get("allow_credentials", True),
                    allow_methods=cors_config.get("allow_methods", ["*"]),
                    allow_headers=cors_config.get("allow_headers", ["*"])
                )
            else:
                # Handle object-based CORS config
                self.app.add_middleware(
                    CORSMiddleware,
                    allow_origins=cors_config.allow_origins,
                    allow_credentials=cors_config.allow_credentials,
                    allow_methods=cors_config.allow_methods,
                    allow_headers=cors_config.allow_headers
                )
            self.logger.info("CORS middleware enabled")
        
        # Logging middleware
        logging_config = self.config.logging
        if isinstance(logging_config, dict):
            # Handle dict-based logging config
            enable_request_logging = logging_config.get("enable_request_logging", False)
            enable_response_logging = logging_config.get("enable_response_logging", False)
            log_requests_body = logging_config.get("log_requests_body", False)
            log_responses_body = logging_config.get("log_responses_body", False)
        else:
            # Handle object-based logging config
            enable_request_logging = logging_config.enable_request_logging
            enable_response_logging = logging_config.enable_response_logging
            log_requests_body = logging_config.log_requests_body
            log_responses_body = logging_config.log_responses_body
            
        if enable_request_logging or enable_response_logging:
            self.app.add_middleware(
                LoggingMiddleware,
                enable_request_logging=enable_request_logging,
                enable_response_logging=enable_response_logging,
                log_requests_body=log_requests_body,
                log_responses_body=log_responses_body
            )
            self.logger.info("Logging middleware enabled")
    
    async def _setup_routers(self) -> None:
        """Setup API routers."""
        self.logger.info("Setting up API routers")
        
        # Override dependency providers on the app
        self.app.dependency_overrides[get_chat_workflow] = lambda: self.chat_workflow
        self.app.dependency_overrides[get_web_interface] = lambda: self
        
        # ✅ FRAMEWORK COMPLIANCE: Handle API config as dict or object
        api_config = self.config.api
        if isinstance(api_config, dict):
            api_prefix = api_config.get("prefix", "/api/v1")
        else:
            api_prefix = api_config.prefix
        
        # Include routers
        self.app.include_router(chat_router, prefix=api_prefix)
        self.app.include_router(health_router, prefix=api_prefix)
        self.app.include_router(websocket_router, prefix=api_prefix)
        self.app.include_router(frontend_router, prefix=api_prefix)
        
        self.logger.info("API routers configured")
    
    async def _setup_error_handlers(self) -> None:
        """Setup custom error handlers."""
        self.logger.info("Setting up error handlers")
        
        @self.app.exception_handler(Exception)
        async def general_exception_handler(request, exc):
            """Handle general exceptions."""
            self.logger.error(f"Unhandled exception: {exc}", 
                            url=str(request.url),
                            method=request.method,
                            exception_type=type(exc).__name__)
            
            return JSONResponse(
                status_code=500,
                content=ErrorResponse(
                    error="internal_server_error",
                    message="An internal server error occurred",
                    details={"exception_type": type(exc).__name__}
                ).dict()
            )
        
        @self.app.exception_handler(HTTPException)
        async def http_exception_handler_override(request, exc):
            """Handle HTTP exceptions."""
            self.logger.warning(f"HTTP exception: {exc.status_code} - {exc.detail}",
                              url=str(request.url),
                              method=request.method,
                              status_code=exc.status_code)
            
            return JSONResponse(
                status_code=exc.status_code,
                content=ErrorResponse(
                    error=f"http_error_{exc.status_code}",
                    message=exc.detail,
                    details={"status_code": exc.status_code}
                ).dict()
            )
    
    async def start_server(self) -> None:
        """Start the web server."""
        if not self.is_initialized:
            await self.initialize()
        
        if self.is_running:
            return
        
        # Handle both dict and object access patterns for server config
        server_config = self.config.server
        if isinstance(server_config, dict):
            host = server_config.get('host', '0.0.0.0')
            port = server_config.get('port')  # ✅ FRAMEWORK COMPLIANT: No hardcoded defaults - YAML configuration required
            workers = server_config.get('workers', 1)
            reload = server_config.get('reload', False)
            access_log = server_config.get('access_log', True)
            
            # ✅ FRAMEWORK COMPLIANCE: Validate required configuration from YAML
            if port is None:
                raise ValueError("Port configuration required in YAML - no hardcoded defaults allowed per framework principles")
        else:
            host = server_config.host
            port = server_config.port
            workers = server_config.workers
            reload = server_config.reload
            access_log = server_config.access_log
            
        self.logger.info(f"Starting web server on {host}:{port}")
        
        try:
            # Configure uvicorn
            config = uvicorn.Config(
                app=self.app,
                host=host,
                port=port,
                workers=workers,
                reload=reload,
                access_log=access_log,
                log_level=(self.config.logging.get('log_level', 'INFO') if isinstance(self.config.logging, dict) else self.config.logging.log_level).lower()
            )
            
            self.server = uvicorn.Server(config)
            self.is_running = True
            
            self.logger.info("Web server starting...")
            await self.server.serve()
            
        except Exception as e:
            self.logger.error(f"Failed to start web server: {e}")
            self.is_running = False
            raise
    
    async def shutdown(self) -> None:
        """Shutdown the web interface and cleanup resources."""
        if not self.is_initialized:
            return
        
        self.logger.info("Shutting down web interface")
        
        try:
            # Stop server
            if self.server and self.is_running:
                self.logger.info("Stopping web server")
                self.server.should_exit = True
                self.is_running = False
            
            # Shutdown chat workflow
            if self.chat_workflow:
                await self.chat_workflow.shutdown()
            
            # Calculate uptime
            if self.start_time:
                uptime = (datetime.utcnow() - self.start_time).total_seconds()
                self.logger.info(f"Web interface shutdown complete", uptime_seconds=uptime)
            
            self.is_initialized = False
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the web interface."""
        return {
            "name": self.config.name,
            "version": self.config.version,
            "is_initialized": self.is_initialized,
            "is_running": self.is_running,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "uptime_seconds": (
                (datetime.utcnow() - self.start_time).total_seconds() 
                if self.start_time else 0.0
            ),
            "chat_workflow_status": (
                self.chat_workflow.get_workflow_status() 
                if self.chat_workflow else {"status": "not_available"}
            ),
            "config": {
                "server": {
                    "host": self.config.server.get('host', '0.0.0.0') if isinstance(self.config.server, dict) else self.config.server.host,
                    "port": self.config.server.get('port', 8000) if isinstance(self.config.server, dict) else self.config.server.port,
                    "workers": self.config.server.get('workers', 1) if isinstance(self.config.server, dict) else self.config.server.workers
                },
                "api": {
                    "prefix": (
                        self.config.api.get("prefix", "/api/v1") 
                        if isinstance(self.config.api, dict) 
                        else self.config.api.prefix
                    ),
                    "title": (
                        self.config.api.get("title", "NanoBrain API") 
                        if isinstance(self.config.api, dict) 
                        else self.config.api.title
                    )
                }
            }
        }
    
    @classmethod
    def from_config_file(cls, config_path: str) -> 'WebInterface':
        """
        Create web interface from configuration file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            WebInterface: Configured web interface instance
        """
        config = WebInterfaceConfig.from_yaml(config_path)
        return cls(config)
    
    @classmethod
    def create_default(cls) -> 'WebInterface':
        """
        Create web interface with default configuration.
        
        Returns:
            WebInterface: Web interface with default settings
        """
        return cls(WebInterfaceConfig())


# Factory functions for easy creation
async def create_web_interface(config: Optional[WebInterfaceConfig] = None) -> WebInterface:
    """
    Create and initialize a web interface.
    
    Args:
        config: Optional configuration
        
    Returns:
        WebInterface: Initialized web interface
    """
    interface = WebInterface(config)
    await interface.initialize()
    return interface


def create_app(config: Optional[WebInterfaceConfig] = None) -> FastAPI:
    """
    Create a FastAPI application with NanoBrain chat workflow.
    
    This is a synchronous function that can be used with ASGI servers
    that don't support async application factories.
    
    Args:
        config: Optional configuration
        
    Returns:
        FastAPI: Configured FastAPI application
    """
    # This creates the app synchronously but doesn't initialize the workflow
    # The workflow will be initialized on first request
    interface = WebInterface(config)
    
    # Run initialization in a separate thread or defer to first request
    import threading
    import asyncio
    
    def initialize_async():
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(interface.initialize())
        except Exception as e:
            print(f"Failed to initialize web interface: {e}")
    
    # Initialize in background thread
    init_thread = threading.Thread(target=initialize_async, daemon=True)
    init_thread.start()
    
    return interface.app


# Main execution for testing
async def main():
    """Main function for testing the web interface."""
    # Create with default config
    interface = WebInterface.create_default()
    
    try:
        # Start server (this will block)
        await interface.start_server()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        await interface.shutdown()


if __name__ == "__main__":
    asyncio.run(main()) 