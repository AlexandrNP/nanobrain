"""
CORS Middleware

Cross-Origin Resource Sharing (CORS) middleware setup for the web interface.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ..config.web_interface_config import CORSConfig


def setup_cors(app: FastAPI, cors_config: CORSConfig) -> None:
    """
    Setup CORS middleware for the FastAPI application.
    
    Args:
        app: FastAPI application instance
        cors_config: CORS configuration settings
    """
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_config.allow_origins,
        allow_credentials=cors_config.allow_credentials,
        allow_methods=cors_config.allow_methods,
        allow_headers=cors_config.allow_headers,
    ) 