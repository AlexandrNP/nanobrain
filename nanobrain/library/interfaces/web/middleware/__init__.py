"""
Web Interface Middleware

Middleware components for the web interface.
"""

from .cors_middleware import setup_cors
from .logging_middleware import LoggingMiddleware

__all__ = [
    'setup_cors',
    'LoggingMiddleware'
] 