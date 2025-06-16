"""
Logging Middleware

Request/response logging middleware that integrates with NanoBrain's logging system.
"""

import time
import json
from typing import Callable, Dict, Any, Optional
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

# Import NanoBrain logging system
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', 'src'))

try:
    from nanobrain.core.logging_system import get_logger, OperationType
except ImportError:
    # Fallback logging if NanoBrain logging is not available
    import logging
    def get_logger(name: str, category: str = "default"):
        return logging.getLogger(f"{category}.{name}")
    
    class OperationType:
        API_REQUEST = "api_request"
        API_RESPONSE = "api_response"


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Logging middleware for HTTP requests and responses.
    
    This middleware logs request and response information using the
    NanoBrain logging system, providing structured logging for the web interface.
    """
    
    def __init__(
        self,
        app: ASGIApp,
        enable_request_logging: bool = True,
        enable_response_logging: bool = True,
        log_requests_body: bool = False,
        log_responses_body: bool = False,
        logger_name: str = "web_interface",
        logger_category: str = "interfaces"
    ):
        """
        Initialize the logging middleware.
        
        Args:
            app: ASGI application
            enable_request_logging: Whether to log requests
            enable_response_logging: Whether to log responses
            log_requests_body: Whether to log request bodies
            log_responses_body: Whether to log response bodies
            logger_name: Name for the logger
            logger_category: Category for the logger
        """
        super().__init__(app)
        self.enable_request_logging = enable_request_logging
        self.enable_response_logging = enable_response_logging
        self.log_requests_body = log_requests_body
        self.log_responses_body = log_responses_body
        
        # Setup logger
        self.logger = get_logger(logger_name, logger_category)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process the request and response with logging.
        
        Args:
            request: The incoming request
            call_next: The next middleware or route handler
            
        Returns:
            Response: The response from the application
        """
        # Generate request ID for tracing
        request_id = self._generate_request_id(request)
        
        # Log request
        start_time = time.time()
        if self.enable_request_logging:
            await self._log_request(request, request_id)
        
        # Process request
        try:
            response = await call_next(request)
            
            # Log response
            if self.enable_response_logging:
                processing_time = (time.time() - start_time) * 1000
                await self._log_response(request, response, request_id, processing_time)
            
            return response
            
        except Exception as e:
            # Log error
            processing_time = (time.time() - start_time) * 1000
            self.logger.error(
                f"Request processing failed: {str(e)}",
                request_id=request_id,
                method=request.method,
                url=str(request.url),
                processing_time_ms=processing_time,
                error_type=type(e).__name__
            )
            raise
    
    async def _log_request(self, request: Request, request_id: str) -> None:
        """
        Log incoming request information.
        
        Args:
            request: The incoming request
            request_id: Unique request identifier
        """
        log_data = {
            "request_id": request_id,
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "headers": dict(request.headers),
            "client_host": request.client.host if request.client else None,
            "user_agent": request.headers.get("user-agent"),
        }
        
        # Add request body if enabled
        if self.log_requests_body and request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                if body:
                    # Try to parse as JSON
                    try:
                        log_data["body"] = json.loads(body.decode())
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        log_data["body"] = body.decode("utf-8", errors="replace")
            except Exception as e:
                log_data["body_error"] = str(e)
        
        self.logger.info(
            f"Incoming request: {request.method} {request.url.path}",
            operation_type=getattr(OperationType, 'API_REQUEST', 'api_request'),
            **log_data
        )
    
    async def _log_response(
        self,
        request: Request,
        response: Response,
        request_id: str,
        processing_time_ms: float
    ) -> None:
        """
        Log response information.
        
        Args:
            request: The original request
            response: The response being sent
            request_id: Unique request identifier
            processing_time_ms: Time taken to process the request
        """
        log_data = {
            "request_id": request_id,
            "method": request.method,
            "url": str(request.url),
            "status_code": response.status_code,
            "processing_time_ms": processing_time_ms,
            "response_headers": dict(response.headers),
        }
        
        # Add response body if enabled
        if self.log_responses_body and hasattr(response, 'body'):
            try:
                # This is a simplified approach - in production you might want
                # to be more careful about logging sensitive response data
                if hasattr(response, 'body') and response.body:
                    try:
                        log_data["response_body"] = json.loads(response.body.decode())
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        log_data["response_body"] = response.body.decode("utf-8", errors="replace")
            except Exception as e:
                log_data["response_body_error"] = str(e)
        
        # Determine log level based on status code
        if response.status_code >= 500:
            log_level = "error"
        elif response.status_code >= 400:
            log_level = "warning"
        else:
            log_level = "info"
        
        log_message = f"Response: {response.status_code} for {request.method} {request.url.path} ({processing_time_ms:.2f}ms)"
        
        getattr(self.logger, log_level)(
            log_message,
            operation_type=getattr(OperationType, 'API_RESPONSE', 'api_response'),
            **log_data
        )
    
    def _generate_request_id(self, request: Request) -> str:
        """
        Generate a unique request ID.
        
        Args:
            request: The incoming request
            
        Returns:
            str: Unique request identifier
        """
        # Use existing request ID from headers if available
        existing_id = request.headers.get("x-request-id")
        if existing_id:
            return existing_id
        
        # Generate new request ID
        import uuid
        return str(uuid.uuid4())[:8] 