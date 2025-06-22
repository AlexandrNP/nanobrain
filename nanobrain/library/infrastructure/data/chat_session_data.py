"""
Chat Session Data Units

Core data containers for chatbot-viral annotation integration following NanoBrain patterns.
These DataUnits manage chat sessions, message history, and interaction tracking.

Author: NanoBrain Development Team
Date: December 2024
Version: 4.1.0
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import uuid


class MessageRole(Enum):
    """Message role enumeration"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class MessageType(Enum):
    """Message type for tracking different kinds of interactions"""
    CHAT = "chat"
    ANNOTATION_REQUEST = "annotation_request"
    ANNOTATION_PROGRESS = "annotation_progress"
    ANNOTATION_RESULT = "annotation_result"
    CONVERSATIONAL = "conversational"
    ERROR = "error"
    PROGRESS = "progress"  # General progress updates
    INFO = "info"  # General information messages


@dataclass
class ChatMessage:
    """Individual chat message data unit with streaming support"""
    
    message_id: str
    role: MessageRole
    content: str
    timestamp: datetime
    message_type: MessageType = MessageType.CHAT
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_streaming: bool = False
    is_complete: bool = True
    error: Optional[str] = None
    
    def __post_init__(self):
        if not self.message_id:
            self.message_id = f"msg_{uuid.uuid4().hex[:8]}"
    
    def update_content(self, new_content: str, append: bool = False):
        """Update message content (for streaming responses)"""
        if append:
            self.content += new_content
        else:
            self.content = new_content
        self.timestamp = datetime.now()
    
    def mark_complete(self):
        """Mark message as complete (end of streaming)"""
        self.is_streaming = False
        self.is_complete = True
    
    def mark_error(self, error_message: str):
        """Mark message as having an error"""
        self.error = error_message
        self.is_streaming = False
        self.is_complete = True


@dataclass
class InteractionMetrics:
    """Detailed interaction tracking metrics"""
    
    total_messages: int = 0
    annotation_requests: int = 0
    conversational_queries: int = 0
    successful_annotations: int = 0
    failed_annotations: int = 0
    error_count: int = 0
    average_response_time: float = 0.0
    total_response_time: float = 0.0
    classification_accuracy: float = 0.0
    user_satisfaction_score: float = 0.0
    session_duration: float = 0.0
    
    def record_message(self, message_type: MessageType, response_time: float = 0.0):
        """Record a new message interaction"""
        self.total_messages += 1
        
        if message_type == MessageType.ANNOTATION_REQUEST:
            self.annotation_requests += 1
        elif message_type == MessageType.CONVERSATIONAL:
            self.conversational_queries += 1
        elif message_type == MessageType.ERROR:
            self.error_count += 1
        
        if response_time > 0:
            self.total_response_time += response_time
            self.average_response_time = self.total_response_time / max(1, self.total_messages - self.error_count)
    
    def record_annotation_result(self, success: bool):
        """Record annotation job completion"""
        if success:
            self.successful_annotations += 1
        else:
            self.failed_annotations += 1


@dataclass
class ChatSessionData:
    """Chat session data unit for maintaining conversation state"""
    
    session_id: str
    messages: List[ChatMessage] = field(default_factory=list)
    active_jobs: Dict[str, str] = field(default_factory=dict)  # job_id -> status
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    metrics: InteractionMetrics = field(default_factory=InteractionMetrics)
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    
    def __post_init__(self):
        if not self.expires_at:
            # Sessions expire after 24 hours of inactivity
            self.expires_at = self.last_activity + timedelta(hours=24)
    
    def add_message(self, role: MessageRole, content: str, 
                   message_type: MessageType = MessageType.CHAT,
                   metadata: Dict = None, is_streaming: bool = False) -> ChatMessage:
        """Add message to session"""
        message = ChatMessage(
            message_id=f"{self.session_id}_{len(self.messages)}",
            role=role,
            content=content,
            timestamp=datetime.now(),
            message_type=message_type,
            metadata=metadata or {},
            is_streaming=is_streaming,
            is_complete=not is_streaming
        )
        self.messages.append(message)
        self.last_activity = datetime.now()
        self.expires_at = self.last_activity + timedelta(hours=24)
        
        # Update metrics
        self.metrics.record_message(message_type)
        
        return message
    
    def get_last_message(self) -> Optional[ChatMessage]:
        """Get the last message in the session"""
        return self.messages[-1] if self.messages else None
    
    def update_last_message_content(self, new_content: str, append: bool = True):
        """Update the content of the last message (for streaming)"""
        if self.messages:
            self.messages[-1].update_content(new_content, append)
            self.last_activity = datetime.now()
    
    def mark_last_message_complete(self):
        """Mark the last message as complete"""
        if self.messages:
            self.messages[-1].mark_complete()
    
    def mark_last_message_error(self, error_message: str):
        """Mark the last message as having an error"""
        if self.messages:
            self.messages[-1].mark_error(error_message)
            self.metrics.record_message(MessageType.ERROR)
    
    def get_conversation_history(self, limit: int = 10) -> List[ChatMessage]:
        """Get recent conversation history"""
        return self.messages[-limit:] if self.messages else []
    
    def add_annotation_job(self, job_data):
        """Add annotation job to session"""
        self.active_jobs[job_data.job_id] = job_data.status
        self.last_activity = datetime.now()
    
    def update_job_status(self, job_id: str, status: str):
        """Update annotation job status"""
        if job_id in self.active_jobs:
            self.active_jobs[job_id] = status
            self.last_activity = datetime.now()
            
            # Update metrics for completed jobs
            if status == "completed":
                self.metrics.record_annotation_result(success=True)
            elif status == "failed":
                self.metrics.record_annotation_result(success=False)
    
    def remove_job(self, job_id: str):
        """Remove completed job"""
        if job_id in self.active_jobs:
            del self.active_jobs[job_id]
    
    def get_active_job_count(self) -> int:
        """Get number of active jobs"""
        active_statuses = ["pending", "running"]
        return len([job for job, status in self.active_jobs.items() if status in active_statuses])
    
    @property
    def is_expired(self) -> bool:
        """Check if session has expired"""
        return datetime.now() > self.expires_at
    
    def update_session_duration(self):
        """Update session duration metric"""
        self.metrics.session_duration = (self.last_activity - self.created_at).total_seconds()


@dataclass
class QueryClassificationData:
    """Query classification result data unit"""
    
    original_query: str
    intent: str  # 'annotation' | 'conversational' | 'unknown'
    confidence: float  # 0.0 - 1.0
    extracted_parameters: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    classification_time: datetime = field(default_factory=datetime.now)
    processing_time_ms: float = 0.0
    
    @property
    def is_annotation_request(self) -> bool:
        return self.intent == 'annotation' and self.confidence >= 0.6
    
    @property
    def is_conversational_request(self) -> bool:
        return self.intent == 'conversational' and self.confidence >= 0.5
    
    @property
    def is_unknown_request(self) -> bool:
        return self.intent == 'unknown' or self.confidence < 0.4


@dataclass
class AnnotationJobData:
    """Viral annotation job tracking data unit"""
    
    job_id: str
    session_id: str
    user_query: str
    extracted_parameters: Dict[str, Any] = field(default_factory=dict)
    status: str = "queued"  # 'queued', 'running', 'completed', 'failed'
    progress: int = 0  # 0-100
    message: str = ""
    result: Optional[Dict[str, Any]] = None
    backend_job_id: Optional[str] = None
    backend_url: str = ""
    priority: str = "normal"
    estimated_duration: float = 0.0
    actual_start_time: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    error_details: Optional[str] = None
    
    def __post_init__(self):
        if not self.expires_at:
            # Results expire after 1 week
            self.expires_at = self.created_at + timedelta(days=7)
    
    @property
    def is_expired(self) -> bool:
        return datetime.now() > self.expires_at
    
    @property
    def execution_time(self) -> Optional[float]:
        """Get execution time in seconds"""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    def start_execution(self):
        """Mark job as started"""
        self.status = 'running'
        self.started_at = datetime.now()
        self.progress = 0
    
    def update_progress(self, progress: int, message: str = ""):
        """Update job progress"""
        self.progress = max(0, min(100, progress))
        self.message = message
        if progress >= 100:
            self.complete_successfully()
    
    def complete_successfully(self, result: Dict[str, Any] = None):
        """Mark job as completed successfully"""
        self.status = 'completed'
        self.progress = 100
        self.completed_at = datetime.now()
        if result:
            self.result = result
    
    def fail_with_error(self, error_message: str, error_details: str = None):
        """Mark job as failed with error"""
        self.status = 'failed'
        self.message = error_message
        self.error_details = error_details
        self.completed_at = datetime.now()


@dataclass
class ConversationalResponseData:
    """Conversational AI response data unit"""
    
    query: str
    response: str
    response_type: str  # 'educational', 'factual', 'clarification'
    confidence: float
    topic_area: str = "general"  # 'structure', 'replication', 'diseases', etc.
    references: List[Dict[str, Any]] = field(default_factory=list)
    response_time: datetime = field(default_factory=datetime.now)
    processing_time_ms: float = 0.0
    
    def add_reference(self, title: str, authors: str, journal: str, year: int, pmid: str = None):
        """Add literature reference"""
        reference = {
            'title': title,
            'authors': authors, 
            'journal': journal,
            'year': year,
            'pmid': pmid,
            'url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None
        }
        self.references.append(reference)
    
    def format_with_references(self) -> str:
        """Format response with literature references"""
        formatted_response = self.response
        
        if self.references:
            formatted_response += "\n\n**📚 References:**\n"
            for i, ref in enumerate(self.references, 1):
                ref_line = f"{i}. {ref['authors']} ({ref['year']}). {ref['title']}. *{ref['journal']}*"
                if ref.get('pmid'):
                    ref_line += f" [PMID: {ref['pmid']}]({ref['url']})"
                formatted_response += f"\n{ref_line}"
        
        return formatted_response 