"""
Progress Update Data Unit - Real-time Progress Tracking
======================================================

Enables real-time progress tracking for long-running workflows with streaming 
updates to frontend clients for composable web interface architecture.

This component follows NanoBrain framework patterns:
- Inherits from DataUnitBase for framework compliance
- Uses from_config pattern for component creation
- Provides comprehensive data schema validation  
- Supports configuration-driven behavior

Usage:
    from nanobrain.library.infrastructure.data import ProgressUpdateDataUnit
    
    # Create via from_config (framework pattern)
    data_unit = ProgressUpdateDataUnit.from_config('config/progress_update_data_unit.yml')
    
    # Set progress data
    data_unit.set_data({
        'job_id': 'job_12345',
        'session_id': 'session_abc',
        'progress_percentage': 45.5,
        'current_step': 'protein_analysis',
        'status': 'processing'
    })
"""

from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
import uuid

from nanobrain.core.data_unit import DataUnitBase
from nanobrain.core.config.config_base import ConfigBase
from nanobrain.core.logging_system import get_logger
from nanobrain.core.component_base import ComponentConfigurationError
from pydantic import Field, ConfigDict


class ProgressUpdateDataUnitConfig(ConfigBase):
    """
    Configuration schema for ProgressUpdateDataUnit
    
    MANDATORY FIELDS:
    - All configuration classes MUST inherit from ConfigBase
    - MUST include comprehensive field documentation
    - MUST use Pydantic V2 validation with ConfigDict
    """
    
    # REQUIRED FIELDS
    name: str = Field(..., description="Data unit identifier for logging and monitoring")
    
    # OPTIONAL FIELDS
    description: str = Field(
        default="Real-time progress tracking for workflows",
        description="Human-readable data unit description"
    )
    enable_logging: bool = Field(
        default=True,
        description="Enable comprehensive logging and monitoring"
    )
    enable_validation: bool = Field(
        default=True,
        description="Enable data schema validation"
    )
    progress_config: Dict[str, Any] = Field(
        default={
            "update_frequency_seconds": 2.0,
            "enable_streaming": True,
            "enable_websocket": True,
            "max_concurrent_jobs": 100,
            "job_timeout_minutes": 30,
            "enable_job_persistence": True
        },
        description="Progress tracking configuration"
    )
    status_definitions: Dict[str, str] = Field(
        default={
            "initializing": "Job is being set up and prepared",
            "queued": "Job is waiting in the execution queue",
            "processing": "Job is actively being processed",
            "completed": "Job has completed successfully",
            "error": "Job encountered an error and failed",
            "cancelled": "Job was cancelled by user or system",
            "timeout": "Job exceeded maximum execution time"
        },
        description="Definitions for job status values"
    )
    streaming_config: Dict[str, Any] = Field(
        default={
            "enable_partial_results": True,
            "partial_result_frequency": 5.0,
            "enable_step_updates": True,
            "enable_error_streaming": True,
            "max_update_size_kb": 100
        },
        description="Streaming and real-time update configuration"
    )
    
    # MANDATORY PYDANTIC V2 CONFIGURATION
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="allow",
        use_enum_values=False,
        validate_assignment=True,
        str_strip_whitespace=True,
        json_schema_extra={
            "examples": [
                {
                    "name": "progress_update_data",
                    "description": "Real-time progress tracking for workflows",
                    "enable_logging": True,
                    "enable_validation": True,
                    "progress_config": {
                        "update_frequency_seconds": 2.0,
                        "enable_streaming": True,
                        "enable_websocket": True,
                        "max_concurrent_jobs": 100,
                        "job_timeout_minutes": 30,
                        "enable_job_persistence": True
                    },
                    "status_definitions": {
                        "initializing": "Job is being set up and prepared",
                        "queued": "Job is waiting in the execution queue",
                        "processing": "Job is actively being processed",
                        "completed": "Job has completed successfully",
                        "error": "Job encountered an error and failed",
                        "cancelled": "Job was cancelled by user or system",
                        "timeout": "Job exceeded maximum execution time"
                    },
                    "streaming_config": {
                        "enable_partial_results": True,
                        "partial_result_frequency": 5.0,
                        "enable_step_updates": True,
                        "enable_error_streaming": True,
                        "max_update_size_kb": 100
                    }
                }
            ],
            "nanobrain_metadata": {
                "framework_version": "2.0.0",
                "component_type": "progress_update_data_unit",
                "config_loading_method": "from_config_only",
                "supports_recursive_references": True
            }
        }
    )


class ProgressUpdateDataUnit(DataUnitBase):
    """
    Progress Update Data Unit - Real-time Progress Tracking
    ======================================================
    
    Enables real-time progress tracking for long-running workflows with streaming
    updates to frontend clients for composable web interface architecture.
    
    **Core Architecture:**
        This data unit enables real-time visibility into workflow execution by
        maintaining job progress state, streaming updates, and providing detailed
        execution context for both users and system monitoring.
        
        * **Job Management**: Job creation, tracking, and lifecycle management
        * **Progress Tracking**: Percentage completion and step-by-step progress
        * **Real-time Updates**: WebSocket and streaming update support
        * **Error Handling**: Comprehensive error tracking and reporting
    
    **Configuration Architecture:**
        ```yaml
        # Basic progress update data unit configuration
        name: "progress_update_data"
        description: "Real-time progress tracking for workflows"
        enable_logging: true
        enable_validation: true
        
        # Progress tracking configuration
        progress_config:
          update_frequency_seconds: 2.0
          enable_streaming: true
          enable_websocket: true
          max_concurrent_jobs: 100
          job_timeout_minutes: 30
          enable_job_persistence: true
          
        # Status definitions
        status_definitions:
          initializing: "Job is being set up and prepared"
          queued: "Job is waiting in the execution queue"
          processing: "Job is actively being processed"
          completed: "Job has completed successfully"
          error: "Job encountered an error and failed"
          cancelled: "Job was cancelled by user or system"
          timeout: "Job exceeded maximum execution time"
          
        # Streaming configuration
        streaming_config:
          enable_partial_results: true
          partial_result_frequency: 5.0
          enable_step_updates: true
          enable_error_streaming: true
          max_update_size_kb: 100
        ```
    
    **Usage Patterns:**
        ```python
        from nanobrain.library.infrastructure.data import ProgressUpdateDataUnit
        
        # Create data unit from configuration
        data_unit = ProgressUpdateDataUnit.from_config('config/progress_update_data_unit.yml')
        
        # Create new job
        job_data = data_unit.create_new_job(
            session_id="session123",
            workflow_name="viral_protein_analysis",
            total_steps=5
        )
        
        # Update progress
        data_unit.update_progress(
            progress_percentage=45.5,
            current_step="protein_analysis",
            step_description="Analyzing protein sequences"
        )
        
        # Add partial results
        data_unit.add_partial_result({
            "proteins_analyzed": 23,
            "total_proteins": 50
        })
        
        # Get progress summary
        progress = data_unit.get_progress_summary()
        ```
    
    Attributes:
        name (str): Data unit identifier for logging and debugging
        description (str): Human-readable data unit description
        logger (logging.Logger): Data unit-specific logger instance
        config (ProgressUpdateDataUnitConfig): Data unit configuration
        
    Note:
        This data unit follows the mandatory from_config pattern and cannot be
        instantiated directly. All configurations must be loaded from YAML files
        using the from_config method.
    
    See Also:
        * :class:`DataUnitBase`: Base framework data unit interface
        * :class:`ProgressUpdateDataUnitConfig`: Configuration schema
        * :class:`ProgressTrackingStep`: Component that uses this data
    """
    
    # MANDATORY COMPONENT METADATA
    COMPONENT_TYPE: str = "progress_update_data_unit"
    REQUIRED_CONFIG_FIELDS: List[str] = ['name']
    
    # DATA SCHEMA DEFINITION
    data_schema = {
        # Job identification
        'job_id': 'string',                   # Unique job identifier
        'session_id': 'string',               # Associated session identifier
        'workflow_name': 'string',            # Name of workflow being executed
        'workflow_id': 'string',              # Unique workflow instance identifier
        'user_id': 'string',                  # User who initiated the job
        
        # Job lifecycle
        'job_start_time': 'datetime',         # When job was created
        'job_end_time': 'datetime',           # When job completed/failed
        'last_update_time': 'datetime',       # Last progress update timestamp
        'elapsed_time': 'float',              # Current elapsed time in seconds
        'estimated_completion': 'datetime',   # Estimated completion time
        'estimated_remaining': 'float',       # Estimated remaining time in seconds
        
        # Progress tracking
        'progress_percentage': 'float',       # Overall progress (0.0-100.0)
        'current_step': 'string',             # Current step name
        'step_number': 'integer',             # Current step number (1-based)
        'total_steps': 'integer',             # Total number of steps
        'step_description': 'string',         # Human-readable step description
        'step_start_time': 'datetime',        # When current step started
        
        # Status and state
        'status': 'string',                   # Job status ('initializing', 'processing', 'completed', etc.)
        'substatus': 'string',                # Detailed substatus information
        'status_message': 'string',           # Human-readable status message
        'status_history': 'list',             # History of status changes
        
        # Results and outputs
        'partial_results': 'dict',            # Partial results available for streaming
        'final_results': 'dict',              # Final results when job completes
        'output_data': 'dict',                # Output data for downstream steps
        'result_metadata': 'dict',            # Metadata about results
        
        # Error handling
        'error_details': 'dict',              # Error information if job fails
        'error_stack_trace': 'string',        # Stack trace for debugging
        'warning_messages': 'list',           # Non-fatal warning messages
        'retry_count': 'integer',             # Number of retry attempts
        'max_retries': 'integer',             # Maximum allowed retries
        
        # Performance metrics
        'memory_usage_mb': 'float',           # Current memory usage
        'cpu_usage_percent': 'float',         # Current CPU usage
        'processing_rate': 'float',           # Items processed per second
        'throughput_metrics': 'dict',         # Detailed throughput metrics
        
        # Streaming and updates
        'supports_streaming': 'boolean',      # Whether job supports streaming updates
        'stream_endpoint': 'string',          # WebSocket/SSE endpoint for streaming
        'update_frequency': 'float',          # Update frequency in seconds
        'last_stream_update': 'datetime',     # Last streaming update timestamp
        
        # Quality and validation
        'quality_score': 'float',             # Quality assessment of progress (0.0-1.0)
        'confidence_level': 'float',          # Confidence in progress estimates (0.0-1.0)
        'validation_status': 'string',        # Validation status of intermediate results
        'accuracy_metrics': 'dict',           # Accuracy measurements
        
        # Resource management
        'resource_allocation': 'dict',        # Allocated computational resources
        'resource_usage': 'dict',             # Current resource utilization
        'priority_level': 'float',            # Job priority (0.0-1.0)
        'queue_position': 'integer',          # Position in execution queue
        
        # Cancellation and control
        'cancellation_requested': 'boolean',  # Whether cancellation was requested
        'cancellation_reason': 'string',      # Reason for cancellation
        'pause_requested': 'boolean',         # Whether pause was requested
        'user_control_enabled': 'boolean',    # Whether user can control execution
        
        # Context and metadata
        'execution_context': 'dict',          # Execution environment context
        'job_metadata': 'dict',               # Additional job metadata
        'tags': 'list',                       # Tags for job categorization
        'correlation_id': 'string'            # Correlation ID for tracking
    }
    
    # MANDATORY FRAMEWORK METHODS
    @classmethod
    def _get_config_class(cls) -> type:
        """
        Return the configuration class for this data unit.
        
        MANDATORY IMPLEMENTATION - This is the ONLY method that differs
        between data unit types in the unified framework pattern.
        
        Returns:
            Configuration class type for this data unit
        """
        return ProgressUpdateDataUnitConfig
    
    def _init_from_config(self, config: 'ProgressUpdateDataUnitConfig') -> None:
        """
        Initialize data unit from validated configuration.
        
        MANDATORY IMPLEMENTATION - Data unit-specific initialization logic.
        
        Args:
            config: Validated configuration instance
            
        Raises:
            ComponentConfigurationError: If configuration is invalid
        """
        # Store configuration
        self.config = config
        self.name = config.name
        self.description = config.description
        
        # Initialize logging
        self.logger = get_logger(
            f"{self.__class__.__module__}.{self.__class__.__name__}",
            enable_logging=config.enable_logging
        )
        
        # Store progress configuration
        self.progress_config = config.progress_config
        self.status_definitions = config.status_definitions
        self.streaming_config = config.streaming_config
        self.enable_validation = config.enable_validation
        
        # Initialize data storage
        self._data: Dict[str, Any] = {}
        
        # Log component initialization
        self.logger.info(
            f"Initializing {self.__class__.__name__}",
            extra={
                "component_name": self.name,
                "component_type": self.COMPONENT_TYPE,
                "config_source": "from_config",
                "max_concurrent_jobs": self.progress_config.get('max_concurrent_jobs', 100)
            }
        )
        
        self.logger.info(
            f"{self.__class__.__name__} initialized successfully",
            extra={
                "component_name": self.name,
                "data_schema_fields": len(self.data_schema)
            }
        )
    
    # JOB MANAGEMENT METHODS
    def create_new_job(self, session_id: str, workflow_name: str, 
                      user_id: Optional[str] = None, total_steps: int = 1,
                      supports_streaming: bool = True) -> Dict[str, Any]:
        """
        Create a new job for progress tracking.
        
        Args:
            session_id: Associated session identifier
            workflow_name: Name of workflow being executed
            user_id: User who initiated the job
            total_steps: Total number of steps in the workflow
            supports_streaming: Whether job supports streaming updates
            
        Returns:
            Dictionary with new job data
        """
        now = datetime.now(timezone.utc)
        job_id = str(uuid.uuid4())
        workflow_id = f"{workflow_name}_{job_id[:8]}"
        
        if not user_id:
            user_id = f"anonymous_{session_id[:8]}"
        
        job_data = {
            'job_id': job_id,
            'session_id': session_id,
            'workflow_name': workflow_name,
            'workflow_id': workflow_id,
            'user_id': user_id,
            'job_start_time': now,
            'job_end_time': None,
            'last_update_time': now,
            'elapsed_time': 0.0,
            'estimated_completion': None,
            'estimated_remaining': None,
            'progress_percentage': 0.0,
            'current_step': 'initializing',
            'step_number': 0,
            'total_steps': total_steps,
            'step_description': 'Initializing workflow execution',
            'step_start_time': now,
            'status': 'initializing',
            'substatus': '',
            'status_message': 'Job is being prepared for execution',
            'status_history': [{
                'status': 'initializing',
                'timestamp': now,
                'message': 'Job created and initialized'
            }],
            'partial_results': {},
            'final_results': {},
            'output_data': {},
            'result_metadata': {},
            'error_details': {},
            'error_stack_trace': '',
            'warning_messages': [],
            'retry_count': 0,
            'max_retries': 3,
            'memory_usage_mb': 0.0,
            'cpu_usage_percent': 0.0,
            'processing_rate': 0.0,
            'throughput_metrics': {},
            'supports_streaming': supports_streaming,
            'stream_endpoint': f"/progress/stream/{job_id}" if supports_streaming else '',
            'update_frequency': self.progress_config.get('update_frequency_seconds', 2.0),
            'last_stream_update': now,
            'quality_score': 0.0,
            'confidence_level': 1.0,
            'validation_status': 'pending',
            'accuracy_metrics': {},
            'resource_allocation': {},
            'resource_usage': {},
            'priority_level': 0.5,
            'queue_position': 0,
            'cancellation_requested': False,
            'cancellation_reason': '',
            'pause_requested': False,
            'user_control_enabled': True,
            'execution_context': {},
            'job_metadata': {},
            'tags': [workflow_name],
            'correlation_id': f"corr_{job_id[:12]}"
        }
        
        self.set_data(job_data)
        
        self.logger.info(
            "New job created for progress tracking",
            extra={
                "job_id": job_id,
                "session_id": session_id,
                "workflow_name": workflow_name,
                "total_steps": total_steps,
                "supports_streaming": supports_streaming
            }
        )
        
        return job_data
    
    def update_progress(self, progress_percentage: float, current_step: Optional[str] = None,
                       step_description: Optional[str] = None, status: Optional[str] = None) -> None:
        """
        Update job progress.
        
        Args:
            progress_percentage: Progress percentage (0.0-100.0)
            current_step: Current step name
            step_description: Step description
            status: Job status
        """
        if not self._data:
            raise ComponentConfigurationError("Job not initialized. Call create_new_job first.")
        
        now = datetime.now(timezone.utc)
        
        # Update progress
        self._data['progress_percentage'] = max(0.0, min(100.0, progress_percentage))
        self._data['last_update_time'] = now
        
        # Update elapsed time
        start_time = self._data.get('job_start_time')
        if start_time:
            elapsed = (now - start_time).total_seconds()
            self._data['elapsed_time'] = elapsed
            
            # Estimate completion time
            if progress_percentage > 0:
                estimated_total = elapsed * (100.0 / progress_percentage)
                estimated_remaining = estimated_total - elapsed
                self._data['estimated_remaining'] = max(0.0, estimated_remaining)
                self._data['estimated_completion'] = now + datetime.timedelta(seconds=estimated_remaining)
        
        # Update step information
        if current_step:
            old_step = self._data.get('current_step')
            self._data['current_step'] = current_step
            
            # If step changed, update step timing
            if old_step != current_step:
                self._data['step_start_time'] = now
                # Increment step number if it's a new step
                if old_step and old_step != 'initializing':
                    self._data['step_number'] = self._data.get('step_number', 0) + 1
        
        if step_description:
            self._data['step_description'] = step_description
        
        # Update status
        if status:
            old_status = self._data.get('status')
            self._data['status'] = status
            
            # Add to status history if status changed
            if old_status != status:
                status_history = self._data.get('status_history', [])
                status_history.append({
                    'status': status,
                    'timestamp': now,
                    'message': f"Status changed from {old_status} to {status}"
                })
                self._data['status_history'] = status_history
                
                # Update status message
                status_def = self.status_definitions.get(status, f"Job status: {status}")
                self._data['status_message'] = status_def
        
        # Update streaming timestamp if streaming is enabled
        if self._data.get('supports_streaming'):
            self._data['last_stream_update'] = now
        
        self.logger.debug(
            "Progress updated",
            extra={
                "job_id": self._data.get('job_id'),
                "progress_percentage": progress_percentage,
                "current_step": current_step,
                "status": status,
                "elapsed_time": self._data.get('elapsed_time', 0.0)
            }
        )
    
    def add_partial_result(self, result_data: Dict[str, Any], result_type: str = "progress") -> None:
        """
        Add partial results for streaming.
        
        Args:
            result_data: Partial result data
            result_type: Type of result ('progress', 'intermediate', 'debug')
        """
        if not self._data:
            return
        
        partial_results = self._data.get('partial_results', {})
        
        # Add timestamp to result
        result_entry = {
            'data': result_data,
            'type': result_type,
            'timestamp': datetime.now(timezone.utc)
        }
        
        # Store by type
        if result_type not in partial_results:
            partial_results[result_type] = []
        
        partial_results[result_type].append(result_entry)
        
        # Keep only recent results (last 50 per type)
        for rtype in partial_results:
            if len(partial_results[rtype]) > 50:
                partial_results[rtype] = partial_results[rtype][-50:]
        
        self._data['partial_results'] = partial_results
        
        self.logger.debug(
            "Partial result added",
            extra={
                "job_id": self._data.get('job_id'),
                "result_type": result_type,
                "data_keys": list(result_data.keys()) if isinstance(result_data, dict) else []
            }
        )
    
    def complete_job(self, final_results: Dict[str, Any], status: str = "completed") -> None:
        """
        Mark job as completed with final results.
        
        Args:
            final_results: Final job results
            status: Final status ('completed', 'error', 'cancelled')
        """
        if not self._data:
            return
        
        now = datetime.now(timezone.utc)
        
        # Update final status
        self._data['status'] = status
        self._data['job_end_time'] = now
        self._data['final_results'] = final_results
        self._data['progress_percentage'] = 100.0 if status == "completed" else self._data.get('progress_percentage', 0.0)
        
        # Calculate final elapsed time
        start_time = self._data.get('job_start_time')
        if start_time:
            final_elapsed = (now - start_time).total_seconds()
            self._data['elapsed_time'] = final_elapsed
        
        # Add to status history
        status_history = self._data.get('status_history', [])
        status_history.append({
            'status': status,
            'timestamp': now,
            'message': f"Job completed with status: {status}"
        })
        self._data['status_history'] = status_history
        
        # Update status message
        status_def = self.status_definitions.get(status, f"Job status: {status}")
        self._data['status_message'] = status_def
        
        self.logger.info(
            "Job completed",
            extra={
                "job_id": self._data.get('job_id'),
                "final_status": status,
                "elapsed_time": self._data.get('elapsed_time', 0.0),
                "progress_percentage": self._data.get('progress_percentage', 0.0)
            }
        )
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """
        Get current progress summary.
        
        Returns:
            Dictionary with progress summary information
        """
        if not self._data:
            return {}
        
        return {
            "job_id": self._data.get('job_id'),
            "session_id": self._data.get('session_id'),
            "workflow_name": self._data.get('workflow_name'),
            "status": self._data.get('status'),
            "progress_percentage": self._data.get('progress_percentage', 0.0),
            "current_step": self._data.get('current_step'),
            "step_number": self._data.get('step_number', 0),
            "total_steps": self._data.get('total_steps', 1),
            "step_description": self._data.get('step_description'),
            "elapsed_time": self._data.get('elapsed_time', 0.0),
            "estimated_remaining": self._data.get('estimated_remaining'),
            "estimated_completion": self._data.get('estimated_completion'),
            "supports_streaming": self._data.get('supports_streaming', False),
            "stream_endpoint": self._data.get('stream_endpoint', ''),
            "last_update_time": self._data.get('last_update_time')
        }
    
    def is_job_active(self) -> bool:
        """Check if job is still active (not completed, failed, or cancelled)."""
        if not self._data:
            return False
        
        status = self._data.get('status', '')
        active_statuses = ['initializing', 'queued', 'processing']
        return status in active_statuses
    
    def get_streaming_data(self) -> Dict[str, Any]:
        """
        Get data suitable for streaming updates.
        
        Returns:
            Dictionary with streaming-friendly data
        """
        summary = self.get_progress_summary()
        
        # Add latest partial results if available
        partial_results = self._data.get('partial_results', {})
        latest_partial = {}
        
        for result_type, results in partial_results.items():
            if results:
                latest_partial[result_type] = results[-1]  # Get most recent
        
        summary['latest_partial_results'] = latest_partial
        summary['has_new_data'] = len(latest_partial) > 0
        
        return summary 