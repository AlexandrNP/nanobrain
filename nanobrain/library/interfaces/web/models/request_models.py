"""
Request Models

Pydantic models for validating incoming API requests.
"""

from typing import Optional, Dict, Any
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
    use_rag: Optional[bool] = Field(
        default=False,
        description="Whether to use Retrieval-Augmented Generation"
    )
    
    rag_sources: Optional[int] = Field(
        default=5,
        gt=0,
        le=20,
        description="Number of RAG sources to retrieve"
    )
    
    # Conversation management
    conversation_id: Optional[str] = Field(
        default=None,
        description="Existing conversation ID to continue"
    )
    
    # Streaming options
    enable_streaming: Optional[bool] = Field(
        default=False,
        description="Whether to enable streaming responses"
    )
    
    # Model selection
    model: Optional[str] = Field(
        default=None,
        description="Specific model to use for this request"
    )
    
    # Custom system prompt
    system_prompt: Optional[str] = Field(
        default=None,
        max_length=2000,
        description="Custom system prompt for this conversation"
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