"""
Request Models

Pydantic models for validating incoming API requests.
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, field_validator, ConfigDict
import uuid


class ChatOptions(BaseModel):
    """
    Chat options for customizing the conversation behavior.
    
    This model defines the optional parameters that can be passed
    with a chat request to customize the agent's behavior.
    """
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "temperature": 0.7,
                "max_tokens": 2000,
                "use_rag": False,
                "conversation_id": None,
                "enable_streaming": False,
                "model": None,
                "system_prompt": None,
                "metadata": {}
            }
        }
    )
    
    # Model parameters
    temperature: Optional[float] = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Controls randomness in the response (0.0 to 2.0)"
    )
    
    max_tokens: Optional[int] = Field(
        default=2000,
        gt=0,
        le=8192,
        description="Maximum number of tokens in the response"
    )
    
    # RAG and retrieval options
    use_rag: bool = Field(
        default=False,
        description="Whether to use retrieval-augmented generation"
    )
    
    # Conversation management
    conversation_id: Optional[str] = Field(
        default=None,
        description="Conversation ID for maintaining context"
    )
    
    # Streaming options
    enable_streaming: bool = Field(
        default=False,
        description="Whether to enable streaming responses"
    )
    
    # Model selection
    model: Optional[str] = Field(
        default=None,
        description="Specific model to use for the request"
    )
    
    # System configuration
    system_prompt: Optional[str] = Field(
        default=None,
        description="Custom system prompt for the conversation"
    )
    
    # Additional metadata
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for the request"
    )
    
    @field_validator('conversation_id')
    @classmethod
    def validate_conversation_id(cls, v):
        """Validate conversation ID format."""
        if v is not None:
            # Accept UUID format or reasonable string
            try:
                # Try to parse as UUID
                uuid.UUID(v)
            except ValueError:
                # If not UUID, check if it's a reasonable string
                if len(v) < 1 or len(v) > 100:
                    raise ValueError("Conversation ID must be between 1 and 100 characters")
        return v
    
    @field_validator('system_prompt')
    @classmethod
    def validate_system_prompt(cls, v):
        """Validate system prompt content."""
        if v is not None and len(v.strip()) == 0:
            return None
        return v


class ChatRequest(BaseModel):
    """
    Main chat request model.
    
    This model represents a complete chat request with the user's query
    and optional configuration parameters.
    """
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "Hello, how can you help me today?",
                "options": {
                    "temperature": 0.7,
                    "max_tokens": 2000,
                    "use_rag": False,
                    "conversation_id": None
                },
                "user_id": "user_123"
            }
        }
    )
    
    query: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="The user's question or message"
    )
    
    options: ChatOptions = Field(
        default_factory=ChatOptions,
        description="Optional configuration for the chat request"
    )
    
    # Request metadata
    request_id: Optional[str] = Field(
        default=None,
        description="Optional request ID for tracking"
    )
    
    user_id: Optional[str] = Field(
        default=None,
        description="Optional user ID for session management"
    )
    
    @field_validator('query')
    @classmethod
    def validate_query(cls, v):
        """Validate and clean the query."""
        # Remove excessive whitespace
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Query cannot be empty")
        return cleaned
    
    @field_validator('request_id', mode='before')
    @classmethod
    def set_request_id(cls, v):
        """Set request ID if not provided."""
        return v or str(uuid.uuid4())


class StreamingChatRequest(BaseModel):
    """
    Streaming chat request model for real-time conversation.
    
    This model extends the standard chat request with streaming-specific
    configuration and session management capabilities.
    """
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "Analyze this protein sequence with streaming updates",
                "options": {
                    "temperature": 0.7,
                    "max_tokens": 2000,
                    "enable_streaming": True,
                    "conversation_id": "conv_123"
                },
                "streaming_config": {
                    "stream_type": "realtime",
                    "chunk_size": 512,
                    "enable_progress": True,
                    "buffer_size": 1024,
                    "timeout": 300
                },
                "session_id": "session_456",
                "user_id": "user_789"
            }
        }
    )
    
    query: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="The user's question or message for streaming processing"
    )
    
    options: ChatOptions = Field(
        default_factory=lambda: ChatOptions(enable_streaming=True),
        description="Chat options with streaming enabled by default"
    )
    
    # Streaming-specific configuration
    streaming_config: Optional[Dict[str, Any]] = Field(
        default_factory=lambda: {
            "stream_type": "realtime",
            "chunk_size": 512,
            "enable_progress": True,
            "buffer_size": 1024,
            "timeout": 300
        },
        description="Configuration for streaming behavior"
    )
    
    # Session management for streaming
    session_id: Optional[str] = Field(
        default=None,
        description="Streaming session ID for connection management"
    )
    
    # Request metadata
    request_id: Optional[str] = Field(
        default=None,
        description="Request ID for tracking streaming operations"
    )
    
    user_id: Optional[str] = Field(
        default=None,
        description="User ID for session management and authentication"
    )
    
    # Connection preferences
    preferred_format: Optional[str] = Field(
        default="json",
        description="Preferred format for streaming chunks (json, text, binary)"
    )
    
    client_capabilities: Optional[List[str]] = Field(
        default_factory=lambda: ["websocket", "sse", "polling"],
        description="Client-supported streaming capabilities"
    )
    
    @field_validator('query')
    @classmethod
    def validate_query(cls, v):
        """Validate and clean the streaming query."""
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Streaming query cannot be empty")
        return cleaned
    
    @field_validator('options')
    @classmethod
    def ensure_streaming_enabled(cls, v):
        """Ensure streaming is enabled in options."""
        if not v.enable_streaming:
            v.enable_streaming = True
        return v
    
    @field_validator('streaming_config')
    @classmethod
    def validate_streaming_config(cls, v):
        """Validate streaming configuration parameters."""
        if v is None:
            return {
                "stream_type": "realtime",
                "chunk_size": 512,
                "enable_progress": True,
                "buffer_size": 1024,
                "timeout": 300
            }
        
        # Validate chunk_size
        if "chunk_size" in v and (v["chunk_size"] < 64 or v["chunk_size"] > 8192):
            raise ValueError("chunk_size must be between 64 and 8192 bytes")
        
        # Validate buffer_size
        if "buffer_size" in v and (v["buffer_size"] < 256 or v["buffer_size"] > 16384):
            raise ValueError("buffer_size must be between 256 and 16384 bytes")
        
        # Validate timeout
        if "timeout" in v and (v["timeout"] < 10 or v["timeout"] > 3600):
            raise ValueError("timeout must be between 10 and 3600 seconds")
        
        # Validate stream_type
        valid_stream_types = ["realtime", "batch", "progressive", "buffered"]
        if "stream_type" in v and v["stream_type"] not in valid_stream_types:
            raise ValueError(f"stream_type must be one of: {valid_stream_types}")
        
        return v
    
    @field_validator('session_id', mode='before')
    @classmethod
    def set_session_id(cls, v):
        """Set session ID if not provided."""
        return v or f"stream_{uuid.uuid4()}"
    
    @field_validator('request_id', mode='before')
    @classmethod
    def set_request_id(cls, v):
        """Set request ID if not provided."""
        return v or str(uuid.uuid4())
    
    @field_validator('preferred_format')
    @classmethod
    def validate_preferred_format(cls, v):
        """Validate preferred streaming format."""
        valid_formats = ["json", "text", "binary", "xml", "csv"]
        if v not in valid_formats:
            raise ValueError(f"preferred_format must be one of: {valid_formats}")
        return v
    
    @field_validator('client_capabilities')
    @classmethod
    def validate_client_capabilities(cls, v):
        """Validate client streaming capabilities."""
        valid_capabilities = ["websocket", "sse", "polling", "http2", "grpc"]
        
        if v is None:
            return ["websocket", "sse", "polling"]
        
        for capability in v:
            if capability not in valid_capabilities:
                raise ValueError(f"Invalid client capability: {capability}. Valid options: {valid_capabilities}")
        
        # Ensure at least one capability is specified
        if not v:
            return ["websocket", "sse", "polling"]
        
        return v 