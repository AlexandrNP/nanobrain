"""
Chatbot Viral Integration Workflow

Main workflow for integrating chatbot interface with viral annotation backend.
Provides intelligent query routing, job management, and conversational capabilities
with comprehensive progress reporting and YAML-configurable steps.

Uses proper data-driven architecture with DataUnit objects and triggers instead
of manual step execution.

Author: NanoBrain Development Team
Date: December 2024
Version: 4.5.0 - Updated for Phase 3 workflow-as-step integration
"""

from nanobrain.core.workflow import Workflow, WorkflowConfig
from nanobrain.core.step import StepConfig, Step
from nanobrain.core.data_unit import DataUnit, DataUnitConfig
from nanobrain.core.trigger import DataUnitChangeTrigger, TriggerBase
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
import importlib


class ChatbotViralWorkflow(Workflow):
    """
    Chatbot Viral Integration Workflow
    
    Framework-compliant implementation for intelligent chatbot responses.
    Routes user queries through viral analysis and conversational response paths.
    """
    
    @classmethod
    def _get_config_class(cls):
        """Return the configuration class for ChatbotViralWorkflow"""
        return WorkflowConfig
    
    def _init_from_config(self, config: WorkflowConfig, component_config: Dict[str, Any],
                         dependencies: Dict[str, Any]) -> None:
        """Initialize ChatbotViralWorkflow with framework compliance"""
        super()._init_from_config(config, component_config, dependencies)
        
        # Store config for step initialization
        self.yaml_config = config.__dict__ if hasattr(config, '__dict__') else {}
        
        # Initialize session manager
        self.session_manager = InMemorySessionManager()
        
        # Progress tracking for streaming responses
        self.streaming_callbacks: Dict[str, Callable] = {}
        
        # ✅ FRAMEWORK COMPLIANCE: Data units are now loaded from config files automatically
        # NO programmatic component creation - all data units are created from YAML configuration
        # Framework handles component creation based on ChatbotViralWorkflow.yml configuration
        
        # CRITICAL FIX: Setup workflow-level triggers for event-driven execution
        self._setup_workflow_triggers()
        
        self.nb_logger.info("🤖 Chatbot Viral Integration Workflow initialized with AlphavirusWorkflow as step")
    
    def _setup_workflow_triggers(self) -> None:
        """Setup workflow-level triggers for event-driven execution"""
        try:
            # ✅ FRAMEWORK COMPLIANCE: Framework handles trigger creation from config files
            # Triggers are now defined in config/Triggers/ directory and loaded automatically
            self.nb_logger.info("✅ Triggers will be loaded from configuration files automatically")
            
            # ✅ ARCHITECTURAL FIX: Remove old input_trigger code - now handled by framework
            # Triggers are created and bound via the from_config pattern, not manually
            
            self.nb_logger.info("🔥 Workflow-level triggers registered for event-driven execution")
            
            return True
        except Exception as e:
            self.nb_logger.error(f"❌ Failed to setup workflow triggers: {e}")
            # Don't raise - allow workflow to continue without triggers for debugging
            self.workflow_triggers = []
    
    async def _execute_workflow_on_trigger(self, trigger_event: Dict[str, Any]) -> None:
        """Execute workflow when triggered by input data unit change"""
        try:
            input_data = trigger_event.get('new_data', {})
            self.nb_logger.info(f"🚀 Workflow triggered by data unit change: {input_data}")
            
            # Execute workflow using data-driven architecture
            result = await self.process_with_data_driven_architecture(input_data)
            
            # Update final result data unit to complete the chain
            if result:
                await self.final_result_data_unit.set(result)
                self.nb_logger.info("✅ Workflow execution completed and result set")
            else:
                self.nb_logger.warning("⚠️ Workflow execution returned no result")
                
        except Exception as e:
            self.nb_logger.error(f"❌ Workflow execution failed: {e}")
            # Set error result
            error_result = {
                'success': False,
                'error': str(e),
                'formatted_response': {
                    'content': f'Error processing request: {e}',
                    'message_type': 'error',
                    'requires_markdown': False
                }
            }
            await self.final_result_data_unit.set(error_result)
    
    async def process_user_message(self, user_message: str, session_id: Optional[str] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process user message using proper data-driven architecture with streaming response capability.
        
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
            
            self.nb_logger.info(f"🔄 Processing message with data-driven architecture for session {session_id}: '{user_message[:100]}...'")
            
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
            
            # Execute workflow using data-driven architecture
            input_data = {
                'user_query': user_message,
                'session_data': session_data,
                'session_id': session_id
            }
            
            # Use proper data-driven execution instead of manual step execution
            result = await self.process_with_data_driven_architecture(input_data)
            
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
                        'session_id': session_id,
                        'message_type': formatted_response.get('message_type', 'chat'),
                        'requires_markdown': formatted_response.get('requires_markdown', False)
                    }
            
            # Update session with final response
            if result and 'formatted_response' in result:
                assistant_msg.content = result['formatted_response'].get('content', '')
                assistant_msg.is_streaming = False
                session_data.last_activity = datetime.now()
            
            # Yield completion
            yield {
                'type': 'message_complete',
                'session_id': session_id,
                'message_id': assistant_msg.message_id,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.nb_logger.error(f"❌ Error processing user message: {e}")
            
            # Yield error response
            yield {
                'type': 'error',
                'error': str(e),
                'session_id': session_id,
                'timestamp': datetime.now().isoformat()
            }
    
    async def process_with_data_driven_architecture(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute workflow using proper data-driven architecture with DataUnit objects and triggers.
        
        This method uses the NanoBrain data-driven pattern instead of manual step execution.
        """
        try:
            # Update progress: workflow started
            if self.progress_reporter:
                await self.progress_reporter.update_progress(
                    'workflow_start', 0, 'running',
                    message="Starting data-driven chatbot viral integration workflow"
                )
            
            # Initialize data units with input data
            await self.user_query_data_unit.set(input_data)
            
            # Start the data-driven execution by triggering the first step
            # The framework will handle the rest based on the configured links and triggers
            await self._trigger_workflow_execution(input_data)
            
            # Wait for workflow completion by monitoring final data unit
            final_result = await self._wait_for_workflow_completion()
            
            # Update progress: workflow completed
            if self.progress_reporter:
                await self.progress_reporter.update_progress(
                    'workflow_complete', 100, 'completed',
                    message="Data-driven chatbot viral integration workflow completed"
                )
            
            self.nb_logger.info("✅ Data-driven workflow execution completed successfully")
            return final_result
            
        except Exception as e:
            self.nb_logger.error(f"❌ Data-driven workflow execution failed: {e}")
            
            # Log full traceback for debugging
            import traceback
            full_traceback = traceback.format_exc()
            self.nb_logger.error(f"Workflow execution traceback: {full_traceback}")
            
            # Update progress: workflow failed
            if self.progress_reporter:
                await self.progress_reporter.update_progress(
                    'workflow_error', 0, 'failed',
                    message=f"Workflow failed: {str(e)}"
                )
            
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__,
                'traceback': full_traceback if hasattr(self.workflow_config, 'debug_mode') and self.workflow_config.debug_mode else None,
                'formatted_response': {
                    'content': f'I encountered an error processing your request: {str(e)}. Please check the logs for more details.',
                    'message_type': 'error',
                    'requires_markdown': True,
                    'error_details': {
                        'error_type': type(e).__name__,
                        'error_message': str(e)
                    }
                }
            }
    
    async def _trigger_workflow_execution(self, input_data: Dict[str, Any]) -> None:
        """
        Trigger the workflow execution using data-driven architecture.
        
        This method starts the data flow by updating the initial data unit,
        which should trigger the first step via its configured trigger.
        """
        # Update the user query data unit to trigger the first step
        await self.user_query_data_unit.set({
            'user_query': input_data.get('user_query', ''),
            'session_id': input_data.get('session_id'),
            'timestamp': datetime.now().isoformat()
        })
        
        self.nb_logger.info("🚀 Workflow execution triggered via data unit update")
    
    async def _wait_for_workflow_completion(self, timeout: int = 300) -> Dict[str, Any]:
        """
        Wait for workflow completion by monitoring the final data unit.
        
        Args:
            timeout: Maximum time to wait for completion
            
        Returns:
            Final workflow result
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Check if the final result data unit has been updated
            final_result = await self.final_result_data_unit.get()
            if final_result is not None:
                self.nb_logger.info("✅ Workflow completed - final result available")
                return final_result
            
            # Wait a bit before checking again
            await asyncio.sleep(0.5)
        
        # Timeout occurred
        self.nb_logger.error("⏱️ Workflow completion timeout")
        return {
            'success': False,
            'error': 'Workflow execution timeout',
            'formatted_response': {
                'content': 'I encountered a timeout while processing your request. Please try again.',
                'message_type': 'error',
                'requires_markdown': True
            }
        }
    
    async def process(self, input_data: Dict[str, Any], **kwargs) -> Any:
        """
        Legacy process method for backward compatibility.
        
        This method is kept for compatibility but now uses the data-driven architecture.
        """
        return await self.process_with_data_driven_architecture(input_data)
    
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

    async def _create_step_instance(self, step_id: str, step_config: StepConfig, config_dict: Dict[str, Any]) -> Step:
        """Create step instance using enhanced from_config patterns"""
        # Enhanced framework automatically handles class+config instantiation
        step_class = config_dict.get('class')
        config_path = config_dict.get('config')
        
        if not step_class or not config_path:
            raise ValueError(f"Step configuration must include 'class' and 'config' fields")
        
        # Import and instantiate using enhanced from_config
        module_path, class_name = step_class.rsplit('.', 1)
        module = importlib.import_module(module_path)
        step_cls = getattr(module, class_name)
        
        # Use enhanced from_config with context
        try:
            step_instance = step_cls.from_config(
                config_path,
                executor=self.executor,
                workflow_directory=self.workflow_directory
            )
            
            self.workflow_logger.info(f"✅ Created step instance: {step_id} ({step_class})")
            return step_instance
        except Exception as e:
            self.workflow_logger.error(f"❌ Failed to create step {step_id}: {e}")
            import traceback
            self.workflow_logger.error(f"Step creation traceback: {traceback.format_exc()}")
            raise


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
        """Get session by ID - returns the same object reference as get_or_create_session."""
        session = self.sessions.get(session_id)
        if session:
            # Update last activity timestamp when accessing session
            session.last_activity = datetime.now()
        return session
    
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
    # ✅ FRAMEWORK COMPLIANCE: Use direct from_config pattern
    if config_path is None:
        config_path = Path(__file__).parent / "ChatbotViralWorkflow.yml"
    
    # Add session_id to kwargs for access during initialization
    if session_id:
        kwargs['session_id'] = session_id
    
    # Create workflow using direct from_config pattern (NO programmatic WorkflowConfig creation)
    workflow = ChatbotViralWorkflow.from_config(str(config_path), **kwargs)
    await workflow.initialize()
    return workflow 