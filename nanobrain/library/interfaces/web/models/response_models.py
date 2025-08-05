"""
Response Models

Pydantic models for API response serialization and validation.
"""

from typing import Optional, Dict, Any, List, Union
from pydantic import BaseModel, Field, ConfigDict, field_validator
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
    
    # Model and processing information
    model_used: Optional[str] = Field(
        default=None,
        description="Name of the model used for generation"
    )
    
    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Temperature setting used"
    )
    
    # Context and conversation metadata
    conversation_id: Optional[str] = Field(
        default=None,
        description="Conversation ID if applicable"
    )
    
    # Additional metadata
    workflow_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata from the underlying workflow"
    )
    
    # Quality metrics
    confidence_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence score for the response (0.0 to 1.0)"
    )


class ChatResponse(BaseModel):
    """
    Standard chat response model.
    
    This model represents a complete response to a chat request,
    including the generated content and metadata.
    """
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "response": "I'd be happy to help you with that! Here's what I found...",
                "status": "success",
                "request_id": "req_12345",
                "conversation_id": "conv_67890",
                "metadata": {
                    "processing_time_ms": 1250.5,
                    "token_count": 150,
                    "model_used": "gpt-4",
                    "confidence_score": 0.95
                },
                "success": True
            }
        }
    )
    
    response: str = Field(
        ...,
        description="The generated response text"
    )
    
    status: ResponseStatus = Field(
        default=ResponseStatus.SUCCESS,
        description="Response status indicator"
    )
    
    # Request tracking
    request_id: Optional[str] = Field(
        default=None,
        description="ID of the original request"
    )
    
    conversation_id: Optional[str] = Field(
        default=None,
        description="Conversation ID for context tracking"
    )
    
    # Response metadata
    metadata: Optional[ChatMetadata] = Field(
        default=None,
        description="Metadata about the response generation"
    )
    
    # Success/failure information
    success: bool = Field(
        default=True,
        description="Whether the request was processed successfully"
    )
    
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if processing failed"
    )
    
    # Timestamp
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Response generation timestamp"
    )


class StreamingChatResponse(BaseModel):
    """
    Streaming chat response model for real-time communication.
    
    This model represents a streaming response chunk with session management
    and progress tracking capabilities.
    """
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "chunk": "Here's what I found about viral proteins...",
                "chunk_type": "content",
                "sequence_number": 5,
                "is_final": False,
                "session_id": "stream_123",
                "request_id": "req_456",
                "streaming_metadata": {
                    "progress_percentage": 45.5,
                    "estimated_remaining_ms": 2500,
                    "current_step": "protein_analysis",
                    "total_chunks": 12
                },
                "status": "success",
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }
    )
    
    # Content data
    chunk: str = Field(
        ...,
        description="Content chunk for this streaming response"
    )
    
    chunk_type: str = Field(
        default="content",
        description="Type of chunk (content, metadata, progress, error, final)"
    )
    
    # Streaming sequence management
    sequence_number: int = Field(
        ...,
        ge=0,
        description="Sequence number of this chunk in the stream"
    )
    
    is_final: bool = Field(
        default=False,
        description="Whether this is the final chunk in the stream"
    )
    
    # Session management
    session_id: str = Field(
        ...,
        description="Streaming session ID for connection management"
    )
    
    request_id: Optional[str] = Field(
        default=None,
        description="Original request ID"
    )
    
    # Streaming-specific metadata
    streaming_metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Metadata specific to streaming operations"
    )
    
    # Response status
    status: ResponseStatus = Field(
        default=ResponseStatus.SUCCESS,
        description="Status of this streaming chunk"
    )
    
    # Error handling for streaming
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if chunk processing failed"
    )
    
    # Content formatting hints for frontend
    content_format: Optional[str] = Field(
        default="text",
        description="Format hint for chunk content (text, json, html, markdown)"
    )
    
    # Progress tracking
    progress_info: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Progress information for long-running operations"
    )
    
    # Timestamp
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Chunk generation timestamp"
    )
    
    @field_validator('chunk_type')
    @classmethod
    def validate_chunk_type(cls, v):
        """Validate chunk type."""
        valid_types = ["content", "metadata", "progress", "error", "final", "start", "intermediate"]
        if v not in valid_types:
            raise ValueError(f"chunk_type must be one of: {valid_types}")
        return v
    
    @field_validator('content_format')
    @classmethod
    def validate_content_format(cls, v):
        """Validate content format."""
        valid_formats = ["text", "json", "html", "markdown", "xml", "csv", "binary"]
        if v not in valid_formats:
            raise ValueError(f"content_format must be one of: {valid_formats}")
        return v


class ErrorResponse(BaseModel):
    """
    Error response model for API failures.
    
    This model provides structured error information including
    error codes, messages, and debugging information.
    """
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": "Validation failed",
                "error_code": "VALIDATION_ERROR",
                "details": {
                    "field": "query",
                    "message": "Query cannot be empty"
                },
                "request_id": "req_12345",
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }
    )
    
    error: str = Field(
        ...,
        description="Human-readable error message"
    )
    
    error_code: Optional[str] = Field(
        default=None,
        description="Machine-readable error code"
    )
    
    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional error details and context"
    )
    
    request_id: Optional[str] = Field(
        default=None,
        description="ID of the failed request"
    )
    
    # For debugging (only in development)
    stack_trace: Optional[str] = Field(
        default=None,
        description="Stack trace for debugging (development only)"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Error occurrence timestamp"
    )


class HealthResponse(BaseModel):
    """
    Health check response model.
    
    Provides information about the API and system health status.
    """
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "timestamp": "2024-01-15T10:30:00Z",
                "checks": {
                    "database": "healthy",
                    "workflow_engine": "healthy",
                    "memory_usage": "normal"
                }
            }
        }
    )
    
    status: str = Field(
        ...,
        description="Overall health status"
    )
    
    version: str = Field(
        ...,
        description="API version"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Health check timestamp"
    )
    
    checks: Dict[str, str] = Field(
        default_factory=dict,
        description="Individual health check results"
    )


class WorkflowStatusResponse(BaseModel):
    """
    Workflow status response model.
    
    Provides detailed information about workflow execution status,
    performance metrics, and operational health.
    """
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "workflow_id": "chatbot_viral_integration",
                "status": "running",
                "execution_progress": {
                    "current_step": "virus_name_resolution",
                    "completed_steps": 2,
                    "total_steps": 5,
                    "progress_percentage": 40.0
                },
                "performance_metrics": {
                    "average_execution_time_ms": 2500.5,
                    "success_rate": 0.95,
                    "requests_processed": 150,
                    "errors_count": 3
                },
                "resource_usage": {
                    "memory_mb": 512.5,
                    "cpu_percentage": 25.3,
                    "active_connections": 8
                },
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }
    )
    
    # Workflow identification
    workflow_id: str = Field(
        ...,
        description="Unique identifier of the workflow"
    )
    
    status: str = Field(
        ...,
        description="Current workflow status (running, idle, error, stopped)"
    )
    
    # Execution progress information
    execution_progress: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Current execution progress and step information"
    )
    
    # Performance and health metrics
    performance_metrics: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Performance metrics and statistics"
    )
    
    # Resource utilization
    resource_usage: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Current resource usage information"
    )
    
    # Error information
    last_error: Optional[str] = Field(
        default=None,
        description="Last error message if applicable"
    )
    
    # Operational metadata
    uptime_seconds: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Workflow uptime in seconds"
    )
    
    # Configuration status
    configuration_status: Optional[str] = Field(
        default=None,
        description="Status of workflow configuration (valid, invalid, outdated)"
    )
    
    # Dependency status
    dependencies_status: Optional[Dict[str, str]] = Field(
        default_factory=dict,
        description="Status of workflow dependencies"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Status check timestamp"
    )
    
    @field_validator('status')
    @classmethod
    def validate_status(cls, v):
        """Validate workflow status."""
        valid_statuses = ["running", "idle", "error", "stopped", "starting", "stopping", "maintenance"]
        if v not in valid_statuses:
            raise ValueError(f"status must be one of: {valid_statuses}")
        return v
    
    @field_validator('configuration_status')
    @classmethod
    def validate_configuration_status(cls, v):
        """Validate configuration status."""
        if v is not None:
            valid_statuses = ["valid", "invalid", "outdated", "unknown"]
            if v not in valid_statuses:
                raise ValueError(f"configuration_status must be one of: {valid_statuses}")
        return v


class StatusResponse(BaseModel):
    """
    General system status response model.
    
    This model provides comprehensive status information about
    the entire system including workflow status and metrics.
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