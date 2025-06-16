"""
NanoBrain Library - Web Interfaces

Web-based interfaces for the NanoBrain framework.

This module provides:
- REST API interfaces for chat workflows
- FastAPI-based web server
- Request/response models
- Middleware and routing
- Configuration management

Main Components:
- WebInterface: Main web server class
- API routers for different endpoints
- Pydantic models for request/response validation
- Middleware for CORS, logging, and error handling
"""

from .web_interface import WebInterface
from .config.web_interface_config import WebInterfaceConfig
from .models.request_models import ChatRequest, ChatOptions
from .models.response_models import (
    ChatResponse, 
    ChatMetadata, 
    ErrorResponse, 
    HealthResponse, 
    StatusResponse
)

__all__ = [
    'WebInterface',
    'WebInterfaceConfig',
    'ChatRequest',
    'ChatOptions', 
    'ChatResponse',
    'ChatMetadata',
    'ErrorResponse',
    'HealthResponse',
    'StatusResponse'
] 