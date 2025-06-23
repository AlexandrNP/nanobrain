"""
Response Models

Pydantic models for API response serialization and validation.
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
from enum import Enum


class ResponseStatus(str, Enum):
    """Response status enumeration."""
    SUCCESS = "success"
    ERROR = "error"
    PARTIAL = "partial"


class ChatMetadata(BaseModel):
    """
    Metadata about the chat response.
    
    Contains information about processing time, token usage,
    model information, and other relevant metrics.
    """
    
    # Processing metrics
    processing_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Time taken to process the request in milliseconds"
    )
    
    # Token usage information
    token_count: Optional[int] = Field(
        default=None,
        ge=0,
        description="Number of tokens in the response"
    )
    
    prompt_tokens: Optional[int] = Field(
        default=None,
        ge=0,
        description="Number of tokens in the prompt"
    )
    
    total_tokens: Optional[int] = Field(
        default=None,
        ge=0,
        description="Total tokens used (prompt + response)"
    )
    
    # Model information
    model_used: Optional[str] = Field(
        default=None,
        description="Name of the model used for generation"
    )
    
    # RAG information
    rag_sources: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Sources used for RAG (if applicable)"
    )
    
    rag_enabled: bool = Field(
        default=False,
        description="Whether RAG was used for this response"
    )
    
    # Conversation information
    conversation_id: Optional[str] = Field(
        default=None,
        description="ID of the conversation this response belongs to"
    )
    
    message_count: Optional[int] = Field(
        default=None,
        ge=1,
        description="Number of messages in the conversation"
    )
    
    # Quality and confidence metrics
    confidence_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence score for the response (0.0 to 1.0)"
    )
    
    # Additional metadata
    custom_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Custom metadata from the workflow"
    )
    
    # Timestamps
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when the response was generated"
    )


class ChatResponse(BaseModel):
    """
    Main chat response model.
    
    Contains the agent's response in markdown format along with
    metadata about the processing and conversation.
    """
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "response": "Hello! I'm here to help you with any questions you might have. How can I assist you today?",
                "status": "success",
                "conversation_id": "conv_123e4567-e89b-12d3-a456-426614174000",
                "request_id": "req_123e4567-e89b-12d3-a456-426614174000",
                "metadata": {
                    "processing_time_ms": 1250.5,
                    "token_count": 25,
                    "prompt_tokens": 12,
                    "total_tokens": 37,
                    "model_used": "gpt-3.5-turbo",
                    "rag_enabled": False,
                    "conversation_id": "conv_123e4567-e89b-12d3-a456-426614174000",
                    "message_count": 1,
                    "timestamp": "2024-01-15T10:30:00Z"
                },
                "warnings": []
            }
        }
    )
    
    # Main response content
    response: str = Field(
        ...,
        description="The agent's response in markdown format"
    )
    
    # Response status
    status: ResponseStatus = Field(
        default=ResponseStatus.SUCCESS,
        description="Status of the response"
    )
    
    # Conversation management
    conversation_id: str = Field(
        ...,
        description="ID of the conversation"
    )
    
    # Request tracking
    request_id: Optional[str] = Field(
        default=None,
        description="ID of the original request"
    )
    
    # Metadata
    metadata: ChatMetadata = Field(
        ...,
        description="Metadata about the response processing"
    )
    
    # Error information (if applicable)
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if status is ERROR"
    )
    
    # Warnings
    warnings: List[str] = Field(
        default_factory=list,
        description="Any warnings generated during processing"
    )


class ErrorResponse(BaseModel):
    """
    Error response model for API errors.
    
    Provides structured error information for client handling.
    """
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": "ValidationError",
                "message": "The provided input failed validation",
                "details": {
                    "field": "query",
                    "issue": "Query cannot be empty"
                },
                "request_id": "req_123e4567-e89b-12d3-a456-426614174000",
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }
    )
    
    error: str = Field(
        ...,
        description="Error type or code"
    )
    
    message: str = Field(
        ...,
        description="Human-readable error message"
    )
    
    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional error details"
    )
    
    request_id: Optional[str] = Field(
        default=None,
        description="ID of the failed request"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when the error occurred"
    )


class HealthResponse(BaseModel):
    """
    Health check response model.
    
    Provides information about the API health status.
    """
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "timestamp": "2024-01-15T10:30:00Z",
                "components": {
                    "database": "healthy",
                    "external_api": "healthy",
                    "workflow": "healthy"
                }
            }
        }
    )
    
    status: str = Field(
        ...,
        description="Health status (healthy, unhealthy, degraded)"
    )
    
    version: str = Field(
        ...,
        description="API version"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of the health check"
    )
    
    components: Dict[str, str] = Field(
        default_factory=dict,
        description="Health status of individual components"
    )


class StatusResponse(BaseModel):
    """
    Status response model for API status information.
    
    Provides detailed information about the API and workflow status.
    """
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "api_status": "operational",
                "workflow_status": {
                    "chatbot_viral": "active",
                    "annotation_jobs": 5,
                    "queue_length": 2
                },
                "metrics": {
                    "requests_per_minute": 120,
                    "average_response_time": 1250.5,
                    "success_rate": 0.98
                },
                "uptime_seconds": 86400,
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }
    )
    
    api_status: str = Field(
        ...,
        description="Overall API status"
    )
    
    workflow_status: Dict[str, Any] = Field(
        default_factory=dict,
        description="Status of the underlying workflow"
    )
    
    metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Performance and usage metrics"
    )
    
    uptime_seconds: float = Field(
        ...,
        ge=0.0,
        description="API uptime in seconds"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of the status check"
    ) 