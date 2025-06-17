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


class WebInterface:
    """
    Web interface for NanoBrain chat workflows.
    
    This class provides a REST API interface to NanoBrain chat workflows
    using FastAPI. It handles:
    - FastAPI application setup and configuration
    - Integration with ChatWorkflow
    - Middleware setup (CORS, logging)
    - API routing and error handling
    - Graceful startup and shutdown
    """
    
    def __init__(self, config: Optional[WebInterfaceConfig] = None):
        """
        Initialize the web interface.
        
        Args:
            config: Web interface configuration
        """
        self.config = config or WebInterfaceConfig()
        self.logger = get_logger("web_interface", "interfaces")
        
        # State
        self.is_initialized = False
        self.is_running = False
        self.start_time = None
        
        # Components
        self.app: Optional[FastAPI] = None
        self.chat_workflow: Optional[ChatWorkflow] = None
        self.server = None
        
        self.logger.info("Web interface created", config_name=self.config.name)
    
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
            self.chat_workflow = ChatWorkflow()
            await self.chat_workflow.initialize()
            self.logger.info("Chat workflow initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize chat workflow: {e}")
            # Continue with fallback workflow
            self.chat_workflow = ChatWorkflow()
    
    async def _setup_app(self) -> None:
        """Setup the FastAPI application."""
        self.logger.info("Setting up FastAPI application")
        
        self.app = FastAPI(
            title=self.config.api.title,
            description=self.config.api.description,
            version=self.config.api.version,
            docs_url=self.config.api.docs_url,
            redoc_url=self.config.api.redoc_url
        )
        
        self.logger.info("FastAPI application created")
    
    async def _setup_middleware(self) -> None:
        """Setup middleware for the application."""
        self.logger.info("Setting up middleware")
        
        # CORS middleware
        if self.config.security.cors_enabled:
            setup_cors(self.app, self.config.cors)
            self.logger.info("CORS middleware enabled")
        
        # Logging middleware
        if self.config.logging.enable_request_logging or self.config.logging.enable_response_logging:
            self.app.add_middleware(
                LoggingMiddleware,
                enable_request_logging=self.config.logging.enable_request_logging,
                enable_response_logging=self.config.logging.enable_response_logging,
                log_requests_body=self.config.logging.log_requests_body,
                log_responses_body=self.config.logging.log_responses_body
            )
            self.logger.info("Logging middleware enabled")
    
    async def _setup_routers(self) -> None:
        """Setup API routers."""
        self.logger.info("Setting up API routers")
        
        # Override dependency providers on the app
        self.app.dependency_overrides[get_chat_workflow] = lambda: self.chat_workflow
        self.app.dependency_overrides[get_web_interface] = lambda: self
        
        # Include routers
        self.app.include_router(chat_router, prefix=self.config.api.prefix)
        self.app.include_router(health_router, prefix=self.config.api.prefix)
        self.app.include_router(websocket_router, prefix=self.config.api.prefix)
        self.app.include_router(frontend_router, prefix=self.config.api.prefix)
        
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
        
        self.logger.info(f"Starting web server on {self.config.server.host}:{self.config.server.port}")
        
        try:
            # Configure uvicorn
            config = uvicorn.Config(
                app=self.app,
                host=self.config.server.host,
                port=self.config.server.port,
                workers=self.config.server.workers,
                reload=self.config.server.reload,
                access_log=self.config.server.access_log,
                log_level=self.config.logging.log_level.lower()
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
                    "host": self.config.server.host,
                    "port": self.config.server.port,
                    "workers": self.config.server.workers
                },
                "api": {
                    "prefix": self.config.api.prefix,
                    "title": self.config.api.title
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