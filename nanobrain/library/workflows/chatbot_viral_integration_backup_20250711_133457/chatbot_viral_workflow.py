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
    Workflow for chatbot-viral annotation integration with AlphavirusWorkflow as step.
    
    Uses proper data-driven architecture with DataUnit objects and triggers
    instead of manual step execution. Handles intelligent query classification,
    workflow routing, conversational responses, and real-time progress updates.
    """
    
    REQUIRED_CONFIG_FIELDS = ['name']
    OPTIONAL_CONFIG_FIELDS = {
        'description': 'Chatbot viral integration workflow',
        'config_path': None,
        'session_id': None
    }
    
    @classmethod
    def extract_component_config(cls, config: WorkflowConfig) -> Dict[str, Any]:
        """Extract ChatbotViralWorkflow configuration"""
        base_config = super().extract_component_config(config)
        return {
            **base_config,
            'config_path': getattr(config, 'config_path', None),
            'session_id': getattr(config, 'session_id', None),
        }
    
    @classmethod  
    def resolve_dependencies(cls, component_config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Resolve ChatbotViralWorkflow dependencies"""
        from nanobrain.core.executor import LocalExecutor, ExecutorConfig
        
        # Create executor using from_config pattern
        executor = kwargs.get('executor')
        if not executor:
            executor_config = ExecutorConfig(executor_type="local", max_workers=3)
            executor = LocalExecutor.from_config(executor_config)
        
        return {'executor': executor}
    
    def _init_from_config(self, config: WorkflowConfig, component_config: Dict[str, Any],
                         dependencies: Dict[str, Any]) -> None:
        """Initialize ChatbotViralWorkflow with resolved dependencies"""
        super()._init_from_config(config, component_config, dependencies)
        
        # Store config for step initialization
        self.yaml_config = config.__dict__ if hasattr(config, '__dict__') else {}
        
        # Initialize session manager
        self.session_manager = InMemorySessionManager()
        
        # Progress tracking for streaming responses
        self.streaming_callbacks: Dict[str, Callable] = {}
        
        # UPDATED: Initialize data units with class field (mandatory from_config pattern)
        self.user_query_data_unit = DataUnit.from_config(DataUnitConfig(
            name="user_query",
            **{"class": "nanobrain.core.data_unit.DataUnitMemory"},
            persistent=False
        ))
        self.extraction_result_data_unit = DataUnit.from_config(DataUnitConfig(
            name="extracted_query_data", 
            **{"class": "nanobrain.core.data_unit.DataUnitMemory"},
            persistent=False
        ))
        self.resolution_result_data_unit = DataUnit.from_config(DataUnitConfig(
            name="resolution_output",
            **{"class": "nanobrain.core.data_unit.DataUnitMemory"}, 
            persistent=False
        ))
        self.analysis_result_data_unit = DataUnit.from_config(DataUnitConfig(
            name="analysis_results",
            **{"class": "nanobrain.core.data_unit.DataUnitFile"},
            persistent=True,
            file_path="results/viral_protein_analysis.json"
        ))
        self.final_result_data_unit = DataUnit.from_config(DataUnitConfig(
            name="formatted_response",
            **{"class": "nanobrain.core.data_unit.DataUnitMemory"},
            persistent=False
        ))
        
        # CRITICAL FIX: Register workflow data units with the workflow's data unit system
        # This ensures the framework can access them during step execution
        self.workflow_data_units = {
            'user_query': self.user_query_data_unit,
            'extracted_query_data': self.extraction_result_data_unit,
            'resolution_output': self.resolution_result_data_unit,
            'analysis_results': self.analysis_result_data_unit,
            'formatted_response': self.final_result_data_unit
        }
        
        self.nb_logger.info("🤖 Chatbot Viral Integration Workflow initialized with AlphavirusWorkflow as step")
    
    async def initialize(self) -> None:
        """Initialize the workflow and register data units with steps."""
        # Call parent initialization first
        await super().initialize()
        
        # CRITICAL FIX: After steps are created, register workflow data units with steps
        await self._register_data_units_with_steps()
        
        self.nb_logger.info("✅ Workflow initialization complete with data unit registration")
    
    async def _register_data_units_with_steps(self) -> None:
        """Register workflow data units with the appropriate steps via links."""
        try:
            # CRITICAL FIX: Use proper link registration instead of data unit reassignment
            # The workflow's role is to connect data units via links, not reassign them
            
            # Register links between workflow data units and step data units
            # This ensures proper data flow without breaking step encapsulation
            
            # Link workflow's user_query data unit to query_classification step's input
            if 'query_classification' in self.child_steps:
                query_step = self.child_steps['query_classification']
                if hasattr(query_step, 'input_data_units') and 'user_query' in query_step.input_data_units:
                    # Create a link from workflow's user_query to step's user_query input
                    from nanobrain.core.link import DirectLink, LinkConfig
                    link_config = LinkConfig(
                        name="workflow_user_query_to_step",
                        source=self.user_query_data_unit,
                        target=query_step.input_data_units['user_query'],
                        buffer_size=1
                    )
                    link = DirectLink.from_config(link_config)
                    self.links['workflow_user_query_to_step'] = link
                    self.nb_logger.info("✅ Registered link from workflow user_query to query_classification step")
            
            # Register link from query_classification output to workflow's extraction_result_data_unit
            if 'query_classification' in self.child_steps:
                query_step = self.child_steps['query_classification']
                if hasattr(query_step, 'output_data_unit') and query_step.output_data_unit:
                    # Create a link from step's output to workflow's data unit
                    from nanobrain.core.link import DirectLink, LinkConfig
                    link_config = LinkConfig(
                        name="step_output_to_workflow_extraction",
                        source=query_step.output_data_unit,
                        target=self.extraction_result_data_unit,
                        buffer_size=1
                    )
                    link = DirectLink.from_config(link_config)
                    self.links['step_output_to_workflow_extraction'] = link
                    self.nb_logger.info("✅ Registered link from query_classification output to workflow extraction_result")
            
            # Let the framework handle the rest of the data flow via the configured links in YAML
            # The YAML configuration already defines the proper step-to-step links
            
            self.nb_logger.info("✅ Workflow data unit links registered successfully")
            
        except Exception as e:
            self.nb_logger.error(f"❌ Failed to register data unit links: {e}")
            import traceback
            self.nb_logger.error(f"Data unit link registration traceback: {traceback.format_exc()}")
    
    async def process_user_message(self, user_message: str, session_id: Optional[str] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process user message using NanoBrain framework's event-driven execution.
        
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
            
            self.nb_logger.info(f"🔄 Processing message with NanoBrain event-driven execution for session {session_id}: '{user_message[:100]}...'")
            
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
            
            # FIXED: Use framework's event-driven execution instead of custom implementation
            # CRITICAL FIX: Set the user_query data unit BEFORE calling execute
            try:
                # Ensure data unit is initialized
                if not self.user_query_data_unit.is_initialized:
                    await self.user_query_data_unit.initialize()
                
                # Set the user query in the user_query data unit with the actual query text
                await self.user_query_data_unit.set(user_message)
                self.nb_logger.info(f"✅ Set user_query data unit with query: '{user_message[:50]}...'")
                
                # Verify the data was set correctly
                verification_data = await self.user_query_data_unit.get()
                if verification_data == user_message:
                    self.nb_logger.info("✅ User query data unit verification successful")
                else:
                    self.nb_logger.error(f"❌ User query data unit verification failed: expected '{user_message}', got '{verification_data}'")
                    
            except Exception as e:
                self.nb_logger.error(f"❌ Failed to set user_query data unit: {e}")
                import traceback
                self.nb_logger.error(f"User query data unit error traceback: {traceback.format_exc()}")
            
            # Use the framework's built-in execute method which handles event-driven execution
            # CRITICAL FIX: Don't pass input_data as keyword argument to avoid conflict
            result = await self.execute()
            
            # Stream any accumulated progress updates
            for progress_update in progress_updates:
                yield {
                    'type': 'progress_update',
                    'progress_data': progress_update,
                    'session_id': session_id
                }
            
            # Extract and yield the final result
            if result and isinstance(result, dict):
                # Handle successful completion
                final_response = result.get('formatted_response', {})
                if isinstance(final_response, dict):
                    content = final_response.get('content', 'Analysis completed successfully.')
                    message_type = final_response.get('message_type', 'response')
                    requires_markdown = final_response.get('requires_markdown', True)
                else:
                    content = str(final_response) if final_response else 'Analysis completed successfully.'
                    message_type = 'response'
                    requires_markdown = True
                
                # Update assistant message with final content
                assistant_msg.content = content
                assistant_msg.is_streaming = False
                assistant_msg.metadata = result.get('metadata', {})
                
                yield {
                    'type': 'content_complete',
                    'content': content,
                    'session_id': session_id,
                    'message_id': assistant_msg.message_id,
                    'metadata': {
                        'message_type': message_type,
                        'requires_markdown': requires_markdown,
                        'processing_successful': True,
                        'execution_time': result.get('execution_time', 0)
                    }
                }
            else:
                # Handle case where result is None or unexpected format
                error_content = "I encountered an issue processing your request. Please try again."
                assistant_msg.content = error_content
                assistant_msg.is_streaming = False
                
                yield {
                    'type': 'content_complete',
                    'content': error_content,
                    'session_id': session_id,
                    'message_id': assistant_msg.message_id,
                    'metadata': {
                        'message_type': 'error',
                        'requires_markdown': False,
                        'processing_successful': False
                    }
                }
            
            # Yield final completion marker
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
    
    async def process(self, input_data: Dict[str, Any], **kwargs) -> Any:
        """
        Process method for workflow execution using the NanoBrain framework's data-driven architecture.
        
        This method integrates with the framework's built-in workflow execution system.
        """
        # CRITICAL FIX: Extract user query from workflow's data unit if input_data is empty
        user_query = input_data.get('user_query', '')
        
        # If user_query is empty in input_data, try to get it from workflow's user_query data unit
        if not user_query and hasattr(self, 'user_query_data_unit'):
            try:
                user_query_data = await self.user_query_data_unit.get()
                if user_query_data:
                    user_query = user_query_data
                    self.nb_logger.info(f"✅ Retrieved user query from workflow data unit: '{user_query[:50]}...'")
            except Exception as e:
                self.nb_logger.warning(f"⚠️ Could not retrieve user query from workflow data unit: {e}")
        
        session_id = input_data.get('session_id', f"session_{uuid.uuid4().hex[:8]}")
        
        # Get or create session
        session_data = await self.session_manager.get_or_create_session(session_id)
        
        # Log the start of processing
        self.nb_logger.info(f"🚀 Starting workflow execution for query: '{user_query[:100]}...'")
        
        # CRITICAL FIX: Don't overwrite user_query data unit if it already contains the query
        if hasattr(self, 'user_query_data_unit'):
            try:
                current_query = await self.user_query_data_unit.get()
                if current_query and current_query.strip():
                    self.nb_logger.info(f"✅ User query data unit already contains: '{current_query[:50]}...' - using existing value")
                    user_query = current_query
                else:
                    # Set the user query in the data unit
                    await self.user_query_data_unit.set(user_query)
                    self.nb_logger.info(f"✅ Set user query data unit with: '{user_query[:50]}...'")
            except Exception as e:
                self.nb_logger.error(f"❌ Error handling user_query data unit: {e}")
        
        # Create workflow context with the actual user query
        workflow_context = {
            'user_query': user_query,  # Use the actual user query, not empty string
            'session_id': session_id,
            'session_data': session_data,
            'workflow_data_units': {
                'user_query': self.user_query_data_unit,
                'extracted_query_data': self.extraction_result_data_unit,
                'resolution_output': self.resolution_result_data_unit,
                'analysis_results': self.analysis_result_data_unit,
                'formatted_response': self.final_result_data_unit
            }
        }
        
        # Execute the workflow with the framework's event-driven execution
        try:
            result = await super().process(workflow_context, **kwargs)
            
            # Extract final response from result
            final_response = "Successfully processed query: " + user_query
            if result and isinstance(result, dict):
                if 'formatted_response' in result:
                    final_response = result['formatted_response'].get('content', final_response)
                elif 'content' in result:
                    final_response = result['content']
            
            return {
                'formatted_response': {
                    'content': final_response,
                    'message_type': 'response',
                    'requires_markdown': True,
                    'processing_successful': True
                },
                'session_id': session_id,
                'execution_time': time.time(),
                'metadata': {
                    'workflow_result': result,
                    'user_query': user_query
                }
            }
            
        except Exception as e:
            self.nb_logger.error(f"❌ Workflow execution failed: {e}")
            import traceback
            self.nb_logger.error(f"Workflow execution traceback: {traceback.format_exc()}")
            raise
    
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
        """Create a step instance from configuration using framework's create_step function."""
        # Import here to avoid circular imports
        from nanobrain.core.step import create_step
        
        # Determine step type/class
        step_class = config_dict.get('class', config_dict.get('step_type', 'nanobrain.core.step.Step'))
        
        # Update step config with ID and name
        step_config.name = step_id
        
        # Create step instance using framework's create_step function
        try:
            step = create_step(step_class, step_config, executor=self.executor)
            self.workflow_logger.info(f"✅ Created step instance: {step_id} ({step_class})")
            return step
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
    # Load YAML configuration
    if config_path is None:
        config_path = Path(__file__).parent / "ChatbotViralWorkflow.yml"
    
    with open(config_path, 'r') as f:
        yaml_config = yaml.safe_load(f)
    
    # Create WorkflowConfig from YAML
    workflow_config = WorkflowConfig(**yaml_config)
    
    # Add session_id to config for access during initialization
    if session_id:
        workflow_config.session_id = session_id
    
    # Create workflow using from_config pattern
    workflow = ChatbotViralWorkflow.from_config(workflow_config, **kwargs)
    await workflow.initialize()
    return workflow 