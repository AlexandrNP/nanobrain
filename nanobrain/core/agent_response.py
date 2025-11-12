"""
Agent Response and Metadata Dataclasses
=======================================

Specialized dataclasses for agent responses and metadata with easy dictionary conversion.
Provides type safety and structured data handling for agent processing results.
"""

from dataclasses import dataclass, asdict, field
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import time
import uuid


@dataclass
class AgentProcessingMetadata:
    """
    Comprehensive metadata for agent processing operations.

    Tracks all aspects of agent execution including performance,
    conversation context, tool usage, and error handling.
    """

    # Core processing information
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    processing_time_seconds: float = 0.0
    success: bool = True

    # Agent identification
    agent_name: str = ""
    agent_type: str = "agent"
    agent_version: str = "1.0.0"
    model: str = ""

    # Processing details
    # standard, specialized, tool_enhanced, fallback
    processing_method: str = "standard"
    input_length: int = 0
    response_length: int = 0

    # LLM usage tracking
    tokens_used: int = 0
    llm_calls: int = 0
    total_tokens_session: int = 0
    total_calls_session: int = 0
    estimated_cost: float = 0.0

    # Conversation context
    conversation_id: Optional[str] = None
    conversation_turn: int = 0
    history_length: int = 0
    context_optimized: bool = False
    original_history_length: int = 0

    # Tool integration metadata
    tools_available: List[str] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)
    tool_execution_time: float = 0.0
    tool_success_count: int = 0
    tool_failure_count: int = 0

    # Performance metrics
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    cache_hits: int = 0
    cache_misses: int = 0

    # Error handling
    error_occurred: bool = False
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    fallback_used: bool = False

    # Quality metrics
    confidence_score: Optional[float] = None
    response_quality_score: Optional[float] = None
    user_satisfaction_predicted: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary for easy serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentProcessingMetadata':
        """Create metadata from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def update_from_dict(self, data: Dict[str, Any]) -> None:
        """Update metadata fields from dictionary."""
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def add_error(self, error_type: str, error_message: str) -> None:
        """Add error information to metadata."""
        self.error_occurred = True
        self.error_type = error_type
        self.error_message = error_message
        self.success = False

    def add_llm_usage(self, tokens: int, calls: int = 1, cost: float = 0.0) -> None:
        """Add LLM usage information."""
        self.tokens_used += tokens
        self.llm_calls += calls
        self.total_tokens_session += tokens
        self.total_calls_session += calls
        self.estimated_cost += cost

    def add_tool_usage(self, tools_used: List[str], execution_time: float = 0.0,
                       success_count: int = 0, failure_count: int = 0) -> None:
        """Add tool usage information."""
        self.tools_used.extend(tools_used)
        self.tool_execution_time += execution_time
        self.tool_success_count += success_count
        self.tool_failure_count += failure_count

    def mark_fallback_used(self) -> None:
        """Mark that fallback processing was used."""
        self.fallback_used = True
        self.processing_method = "fallback"

    def calculate_efficiency_score(self) -> float:
        """Calculate processing efficiency score (0-1)."""
        if self.processing_time_seconds <= 0:
            return 1.0

        # Base score on processing time (faster = better)
        # 30s baseline
        time_score = max(0, 1 - (self.processing_time_seconds / 30))

        # Adjust for success/failure
        success_score = 1.0 if self.success else 0.5

        # Adjust for tool usage efficiency
        tool_score = 1.0
        if self.tools_used:
            total_tool_calls = self.tool_success_count + self.tool_failure_count
            tool_score = self.tool_success_count / \
                total_tool_calls if total_tool_calls > 0 else 0.5

        return (time_score * 0.4 + success_score * 0.4 + tool_score * 0.2)


@dataclass
class ConversationContext:
    """
    Structured conversation history with metadata and management capabilities.
    """

    conversation_id: str
    messages: List[Dict[str, Any]] = field(default_factory=list)
    total_messages: int = 0
    user_messages: int = 0
    assistant_messages: int = 0
    system_messages: int = 0
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    context_optimized: bool = False
    original_length: int = 0

    # Conversation metadata
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    conversation_topic: Optional[str] = None
    conversation_tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert conversation context to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationContext':
        """Create conversation context from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_message_list(cls, conversation_id: str, messages: List[Dict[str, Any]]) -> 'ConversationContext':
        """Create conversation context from message list."""
        context = cls(conversation_id=conversation_id, messages=messages)
        context.update_counts()
        return context

    def update_counts(self) -> None:
        """Update message counts and metadata."""
        self.total_messages = len(self.messages)
        self.user_messages = sum(
            1 for msg in self.messages if msg.get('role') == 'user')
        self.assistant_messages = sum(
            1 for msg in self.messages if msg.get('role') == 'assistant')
        self.system_messages = sum(
            1 for msg in self.messages if msg.get('role') == 'system')
        self.last_updated = time.time()

    def add_exchange(self, user_message: str, assistant_response: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a user-assistant exchange to the conversation."""
        timestamp = time.time()

        user_msg = {
            "role": "user",
            "content": user_message,
            "timestamp": timestamp
        }

        assistant_msg = {
            "role": "assistant",
            "content": assistant_response,
            "timestamp": timestamp
        }

        if metadata:
            assistant_msg["metadata"] = metadata

        self.messages.extend([user_msg, assistant_msg])
        self.update_counts()

    def get_recent_messages(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get the most recent messages."""
        return self.messages[-count:] if count > 0 else self.messages

    def trim_to_length(self, max_length: int) -> None:
        """Trim conversation to maximum length."""
        if len(self.messages) > max_length:
            self.original_length = len(self.messages)
            self.messages = self.messages[-max_length:]
            self.context_optimized = True
            self.update_counts()


@dataclass
class AgentResponse:
    """
    Comprehensive agent response with text, conversation context, and metadata.

    Designed for easy dictionary conversion and BaseStep integration.
    Supports multiple response types and comprehensive metadata tracking.
    """

    # Core response data
    response_text: str = ""
    conversation_context: ConversationContext = field(
        default_factory=lambda: ConversationContext(""))
    processing_metadata: AgentProcessingMetadata = field(
        default_factory=AgentProcessingMetadata)

    # Response classification
    # standard, error, fallback, tool_enhanced, specialized
    response_type: str = "standard"
    # informational, creative, analytical, etc.
    response_category: Optional[str] = None

    # Quality and confidence metrics
    confidence_score: Optional[float] = None
    quality_score: Optional[float] = None
    relevance_score: Optional[float] = None

    # Additional response data
    structured_data: Optional[Dict[str, Any]] = None
    tool_results: Optional[Dict[str, Any]] = None
    citations: List[str] = field(default_factory=list)
    suggested_followups: List[str] = field(default_factory=list)

    # Response metadata
    language: str = "en"
    content_warnings: List[str] = field(default_factory=list)
    response_length: int = field(init=False)

    def __post_init__(self):
        """Calculate derived fields after initialization."""
        self.response_length = len(self.response_text)

        # Update processing metadata with response info
        if self.processing_metadata:
            self.processing_metadata.response_length = self.response_length
            self.processing_metadata.confidence_score = self.confidence_score

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert agent response to dictionary for BaseStep integration.

        Returns dictionary with keys optimized for step data units:
        - response: Main response text
        - conversation_context: Conversation history
        - processing_metadata: Complete processing metadata
        - additional fields for extended functionality
        """
        return {
            # Core response fields (BaseStep compatible)
            'response': self.response_text,
            'conversation_context': self.conversation_context.to_dict(),
            'processing_metadata': self.processing_metadata.to_dict(),

            # Extended response fields
            'response_type': self.response_type,
            'response_category': self.response_category,
            'confidence_score': self.confidence_score,
            'quality_score': self.quality_score,
            'relevance_score': self.relevance_score,

            # Additional data
            'structured_data': self.structured_data,
            'tool_results': self.tool_results,
            'citations': self.citations,
            'suggested_followups': self.suggested_followups,

            # Metadata
            'language': self.language,
            'content_warnings': self.content_warnings,
            'response_length': self.response_length,
            'timestamp': self.processing_metadata.timestamp
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentResponse':
        """Create agent response from dictionary."""

        # Handle conversation context
        conversation_data = data.get('conversation_context', {})
        if isinstance(conversation_data, dict) and 'conversation_id' in conversation_data:
            conversation_context = ConversationContext.from_dict(
                conversation_data)
        else:
            conversation_context = ConversationContext("")

        # Handle processing metadata
        metadata_data = data.get('processing_metadata', {})
        processing_metadata = AgentProcessingMetadata.from_dict(
            metadata_data) if metadata_data else AgentProcessingMetadata()

        return cls(
            response_text=data.get('response', ''),
            conversation_context=conversation_context,
            processing_metadata=processing_metadata,
            response_type=data.get('response_type', 'standard'),
            response_category=data.get('response_category'),
            confidence_score=data.get('confidence_score'),
            quality_score=data.get('quality_score'),
            relevance_score=data.get('relevance_score'),
            structured_data=data.get('structured_data'),
            tool_results=data.get('tool_results'),
            citations=data.get('citations', []),
            suggested_followups=data.get('suggested_followups', []),
            language=data.get('language', 'en'),
            content_warnings=data.get('content_warnings', [])
        )

    @classmethod
    def create_success_response(cls, response_text: str, user_input: str,
                                conversation_id: str,
                                conversation_history: List[Dict[str, Any]] = None,
                                metadata: Optional[AgentProcessingMetadata] = None) -> 'AgentResponse':
        """Create a successful agent response with conversation context."""

        # Create or use provided metadata
        if metadata is None:
            metadata = AgentProcessingMetadata(success=True)

        # Create conversation context
        conversation_context = ConversationContext(
            conversation_id=conversation_id)
        if conversation_history:
            conversation_context.messages = conversation_history
            conversation_context.update_counts()

        # Add current exchange
        conversation_context.add_exchange(user_input, response_text, {
            'processing_metadata': metadata.to_dict()
        })

        return cls(
            response_text=response_text,
            conversation_context=conversation_context,
            processing_metadata=metadata,
            response_type="standard"
        )

    @classmethod
    def create_error_response(cls, error_message: str,
                              conversation_id: str = "",
                              conversation_history: List[Dict[str, Any]] = None,
                              error_type: str = "processing_error",
                              original_error: Optional[str] = None) -> 'AgentResponse':
        """Create an error response with appropriate metadata."""

        # Create error metadata
        metadata = AgentProcessingMetadata(success=False)
        metadata.add_error(error_type, original_error or error_message)

        # Create conversation context
        conversation_context = ConversationContext(
            conversation_id=conversation_id)
        if conversation_history:
            conversation_context.messages = conversation_history
            conversation_context.update_counts()

        return cls(
            response_text=f"I apologize, but I encountered an error: {error_message}",
            conversation_context=conversation_context,
            processing_metadata=metadata,
            response_type="error",
            content_warnings=["error_response"]
        )

    def is_success(self) -> bool:
        """Check if response represents successful processing."""
        return self.processing_metadata.success and self.response_type != "error"

    def is_error(self) -> bool:
        """Check if response represents an error."""
        return not self.processing_metadata.success or self.response_type == "error"

    def is_tool_enhanced(self) -> bool:
        """Check if response used tool integration."""
        return self.response_type == "tool_enhanced" or bool(self.tool_results)

    def get_response_summary(self) -> Dict[str, Any]:
        """Get summary of response for logging and monitoring."""
        return {
            'success': self.is_success(),
            'response_length': self.response_length,
            'response_type': self.response_type,
            'response_category': self.response_category,
            'processing_method': self.processing_metadata.processing_method,
            'tokens_used': self.processing_metadata.tokens_used,
            'processing_time': self.processing_metadata.processing_time_seconds,
            'conversation_id': self.conversation_context.conversation_id,
            'conversation_length': self.conversation_context.total_messages,
            'tools_used': len(self.processing_metadata.tools_used),
            'confidence_score': self.confidence_score,
            'efficiency_score': self.processing_metadata.calculate_efficiency_score()
        }
