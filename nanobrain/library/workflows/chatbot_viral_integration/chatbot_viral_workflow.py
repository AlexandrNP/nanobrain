"""
Chatbot Viral Integration Workflow

Main workflow for integrating chatbot interface with viral annotation backend.
Provides intelligent query routing, job management, and conversational capabilities
with comprehensive progress reporting and YAML-configurable steps.

Author: NanoBrain Development Team
Date: December 2024
Version: 4.2.0
"""

from nanobrain.core.workflow import Workflow, WorkflowConfig
from nanobrain.core.step import StepConfig
from nanobrain.library.infrastructure.data.chat_session_data import (
    ChatSessionData, ChatMessage, MessageRole, MessageType,
    QueryClassificationData, AnnotationJobData, ConversationalResponseData
)

from .steps import (
    QueryClassificationStep,
    AnnotationJobStep,
    ConversationalResponseStep,
    ResponseFormattingStep
)

from typing import Dict, Any, Optional, AsyncGenerator, Callable
import uuid
import asyncio
import time
import yaml
from pathlib import Path
from datetime import datetime


class ChatbotViralWorkflow(Workflow):
    """
    Workflow for chatbot-viral annotation integration.
    
    Handles intelligent query classification, job routing, conversational responses,
    and real-time progress updates following NanoBrain patterns with comprehensive
    progress reporting and YAML-configurable steps.
    """
    
    def __init__(self, config_path: Optional[str] = None, session_id: Optional[str] = None, **kwargs):
        # Load YAML configuration
        if config_path is None:
            config_path = Path(__file__).parent / "ChatbotViralWorkflow.yml"
        
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
        
        # Create WorkflowConfig from YAML
        workflow_config = WorkflowConfig(**yaml_config)
        
        # Initialize with session ID for progress tracking
        super().__init__(workflow_config, session_id=session_id, **kwargs)
        
        # Store YAML config for step initialization
        self.yaml_config = yaml_config
        
        # Initialize session manager
        self.session_manager = InMemorySessionManager()
        
        # Progress tracking for streaming responses
        self.streaming_callbacks: Dict[str, Callable] = {}
        
        self.nb_logger.info("🤖 Chatbot Viral Integration Workflow initialized with progress reporting")
    
    async def process_user_message(self, user_message: str, session_id: Optional[str] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process user message with streaming response capability and progress tracking.
        
        Args:
            user_message: User's input message
            session_id: Optional session ID (creates new if None)
            
        Yields:
            Response chunks for real-time streaming with progress updates
        """
        try:
            # Get or create session
            if not session_id:
                session_id = f"session_{uuid.uuid4().hex[:8]}"
            
            session_data = await self.session_manager.get_or_create_session(session_id)
            
            # Add user message to session
            user_msg = session_data.add_message(
                role=MessageRole.USER,
                content=user_message,
                message_type=MessageType.CHAT
            )
            
            self.nb_logger.info(f"🔄 Processing message for session {session_id}: '{user_message[:100]}...'")
            
            # Start assistant response (streaming placeholder)
            assistant_msg = session_data.add_message(
                role=MessageRole.ASSISTANT,
                content="",
                message_type=MessageType.CHAT,
                is_streaming=True
            )
            
            # Set up progress callback for this session
            progress_updates = []
            
            def progress_callback(progress_data: Dict[str, Any]) -> None:
                progress_updates.append(progress_data)
            
            if self.progress_reporter:
                self.add_progress_callback(progress_callback)
            
            # Yield initial response acknowledgment with progress setup
            yield {
                'type': 'message_start',
                'session_id': session_id,
                'message_id': assistant_msg.message_id,
                'timestamp': datetime.now().isoformat(),
                'progress_enabled': self.progress_reporter is not None
            }
            
            # Execute workflow with input data
            input_data = {
                'user_query': user_message,
                'session_data': session_data,
                'session_id': session_id
            }
            
            # Process through workflow steps
            result = await self.process(input_data)
            
            # Stream any accumulated progress updates
            for progress_update in progress_updates:
                yield {
                    'type': 'progress_update',
                    'progress_data': progress_update,
                    'session_id': session_id
                }
            
            # Yield final result
            if result and 'formatted_response' in result:
                formatted_response = result['formatted_response']
                
                # Handle streaming responses
                if formatted_response.get('is_streaming') and 'streaming_chunks' in formatted_response:
                    for chunk in formatted_response['streaming_chunks']:
                        yield {
                            'type': 'content_chunk',
                            'content': chunk,
                            'session_id': session_id
                        }
                        await asyncio.sleep(0.1)  # Simulate streaming delay
                else:
                    yield {
                        'type': 'content_complete',
                        'content': formatted_response.get('content', ''),
                        'metadata': {
                            'message_type': formatted_response.get('message_type'),
                            'requires_markdown': formatted_response.get('requires_markdown', False),
                            'job_id': formatted_response.get('job_id'),
                            'status': formatted_response.get('status')
                        },
                        'session_id': session_id
                    }
            
            # Update assistant message with final content
            if result and 'formatted_response' in result:
                assistant_msg.content = result['formatted_response'].get('content', '')
                assistant_msg.is_streaming = False
            
            # Yield completion
            yield {
                'type': 'message_complete',
                'session_id': session_id,
                'message_id': assistant_msg.message_id,
                'timestamp': datetime.now().isoformat(),
                'success': True
            }
            
        except Exception as e:
            self.nb_logger.error(f"❌ Workflow processing failed: {e}")
            
            # Yield error response
            yield {
                'type': 'error',
                'error': str(e),
                'session_id': session_id if 'session_id' in locals() else None,
                'timestamp': datetime.now().isoformat()
            }
    
    async def process(self, input_data: Dict[str, Any], **kwargs) -> Any:
        """
        Execute the chatbot viral workflow with progress reporting.
        
        This method routes through the workflow steps based on YAML configuration
        and provides comprehensive progress tracking.
        """
        # Update progress: workflow started
        if self.progress_reporter:
            await self.progress_reporter.update_progress(
                'workflow_start', 0, 'running',
                message="Starting chatbot viral integration workflow"
            )
        
        # Execute the workflow using the parent class method
        result = await super().process(input_data, **kwargs)
        
        # Update progress: workflow completed
        if self.progress_reporter:
            await self.progress_reporter.update_progress(
                'workflow_complete', 100, 'completed',
                message="Chatbot viral integration workflow completed"
            )
        
        return result
    
    async def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session information including progress history."""
        session_data = await self.session_manager.get_session(session_id)
        if not session_data:
            return None
        
        session_info = {
            'session_id': session_id,
            'created_at': session_data.created_at,
            'last_activity': session_data.last_activity,
            'message_count': len(session_data.messages),
            'active_jobs': len(session_data.active_annotation_jobs)
        }
        
        # Add progress history if available
        if self.progress_reporter:
            session_info['progress_history'] = self.get_progress_history()
            session_info['current_progress'] = self.get_progress_summary()
        
        return session_info
    
    async def get_job_progress(self, job_id: str, session_id: str) -> Optional[Dict[str, Any]]:
        """Get progress information for a specific annotation job."""
        session_data = await self.session_manager.get_session(session_id)
        if not session_data:
            return None
        
        # Find job in session
        job_data = None
        for job in session_data.active_annotation_jobs:
            if job.job_id == job_id:
                job_data = job
                break
        
        if not job_data:
            return None
        
        # Get progress from workflow reporter
        progress_summary = self.get_progress_summary()
        
        return {
            'job_id': job_id,
            'status': job_data.status,
            'progress': progress_summary.get('overall_progress', 0) if progress_summary else 0,
            'estimated_time_remaining': progress_summary.get('estimated_time_remaining') if progress_summary else None,
            'last_updated': time.time()
        }
    
    async def cleanup_expired_sessions(self):
        """Clean up expired sessions and their progress data."""
        await self.session_manager.cleanup_expired_sessions()
        
        # Clean up any orphaned progress data
        if self.progress_reporter:
            # This could be extended to clean up old progress history
            pass
    
    def add_streaming_callback(self, session_id: str, callback: Callable) -> None:
        """Add streaming callback for a specific session."""
        self.streaming_callbacks[session_id] = callback
    
    def remove_streaming_callback(self, session_id: str) -> None:
        """Remove streaming callback for a session."""
        self.streaming_callbacks.pop(session_id, None)


class InMemorySessionManager:
    """
    In-memory session manager for chatbot interactions.
    
    Manages chat sessions with progress tracking integration.
    """
    
    def __init__(self):
        self.sessions: Dict[str, ChatSessionData] = {}
        self.session_progress: Dict[str, Dict[str, Any]] = {}
    
    async def get_or_create_session(self, session_id: str) -> ChatSessionData:
        """Get existing session or create new one."""
        if session_id not in self.sessions:
            self.sessions[session_id] = ChatSessionData(
                session_id=session_id,
                created_at=datetime.now(),
                last_activity=datetime.now()
            )
            
            # Initialize progress tracking for session
            self.session_progress[session_id] = {
                'created_at': time.time(),
                'workflow_runs': []
            }
        else:
            # Update last activity
            self.sessions[session_id].last_activity = datetime.now()
        
        return self.sessions[session_id]
    
    async def get_session(self, session_id: str) -> Optional[ChatSessionData]:
        """Get session by ID."""
        return self.sessions.get(session_id)
    
    async def update_session_progress(self, session_id: str, progress_data: Dict[str, Any]) -> None:
        """Update progress data for a session."""
        if session_id in self.session_progress:
            if 'workflow_runs' not in self.session_progress[session_id]:
                self.session_progress[session_id]['workflow_runs'] = []
            
            self.session_progress[session_id]['workflow_runs'].append({
                'timestamp': time.time(),
                'progress': progress_data
            })
    
    async def get_session_progress(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get progress data for a session."""
        return self.session_progress.get(session_id)
    
    async def cleanup_expired_sessions(self):
        """Clean up expired sessions (older than 24 hours)."""
        current_time = datetime.now()
        expired_sessions = []
        
        for session_id, session_data in self.sessions.items():
            if (current_time - session_data.last_activity).total_seconds() > 86400:  # 24 hours
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.sessions[session_id]
            self.session_progress.pop(session_id, None)


# Factory function for creating workflow instances
async def create_chatbot_viral_workflow(config_path: Optional[str] = None, 
                                      session_id: Optional[str] = None,
                                      **kwargs) -> ChatbotViralWorkflow:
    """
    Factory function to create and initialize ChatbotViralWorkflow.
    
    Args:
        config_path: Path to YAML configuration file
        session_id: Session ID for progress tracking
        **kwargs: Additional workflow parameters
        
    Returns:
        Initialized ChatbotViralWorkflow instance
    """
    workflow = ChatbotViralWorkflow(config_path=config_path, session_id=session_id, **kwargs)
    await workflow.initialize()
    return workflow 