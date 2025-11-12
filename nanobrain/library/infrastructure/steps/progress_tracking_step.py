"""
Progress Tracking Step - Pure Progress Monitoring and Data Management
====================================================================

Provides focused progress tracking functionality as a proper NanoBrain workflow step,
enabling progress monitoring within event-driven workflow orchestration with complete
framework compliance.

**SINGLE RESPONSIBILITY**: Progress Tracking Only
- Progress data collection and calculation
- Milestone detection and tracking
- Progress data unit updates
- Status and timeline management

**DOES NOT INCLUDE**:
- WebSocket communication (handled by WebInterfaceStep)
- Real-time notifications (handled by notification components)
- Complex business logic (delegated to workflow steps)

This component follows NanoBrain framework patterns:
- Inherits from BaseStep for framework compliance
- Uses from_config pattern for component creation
- Provides comprehensive configuration validation
- Supports event-driven workflow orchestration

Usage:
    from nanobrain.library.infrastructure.steps import ProgressTrackingStep
    
    # Create via from_config (framework pattern)
    step = ProgressTrackingStep.from_config('config/progress_tracking_step.yml')
    
    # Execute within workflow context
    await step.execute()
"""

import asyncio
import time
import uuid
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timezone
from pathlib import Path

from nanobrain.core.step import BaseStep, StepConfig
from nanobrain.core.config.config_base import ConfigBase
from nanobrain.core.component_base import ComponentConfigurationError, ComponentDependencyError
from nanobrain.core.logging_system import get_logger
from nanobrain.core.data_unit import DataUnitBase, DataUnitMemory
from nanobrain.core.trigger import TriggerBase
from nanobrain.library.infrastructure.data.progress_update_data_unit import ProgressUpdateDataUnit
from nanobrain.library.infrastructure.data.session_context_data_unit import SessionContextDataUnit

from pydantic import BaseModel, Field, ConfigDict


class ProgressTrackingStepConfig(StepConfig):
    """
    Configuration schema for ProgressTrackingStep
    
    Provides comprehensive progress tracking configuration including milestone
    tracking, timing calculations, and data management settings.
    """
    
    # Progress calculation settings
    milestone_percentage_points: List[float] = Field(
        default=[10.0, 25.0, 50.0, 75.0, 90.0, 100.0],
        description="Progress percentage points that trigger milestone events"
    )
    
    progress_update_interval_seconds: float = Field(
        default=1.0,
        description="Minimum interval between progress updates in seconds"
    )
    
    enable_detailed_tracking: bool = Field(
        default=True,
        description="Enable detailed progress step tracking and timing"
    )
    
    # Timing and performance settings
    max_expected_duration_seconds: float = Field(
        default=300.0,
        description="Maximum expected duration for progress tracking in seconds"
    )
    
    enable_eta_calculation: bool = Field(
        default=True,
        description="Enable estimated time of arrival calculations"
    )
    
    # Data management settings
    preserve_progress_history: bool = Field(
        default=True,
        description="Preserve progress history for analysis and debugging"
    )
    
    max_history_entries: int = Field(
        default=100,
        description="Maximum number of progress history entries to keep"
    )
    
    # Monitoring and validation
    enable_progress_validation: bool = Field(
        default=True,
        description="Enable progress value validation and consistency checks"
    )
    
    allow_progress_rollback: bool = Field(
        default=False,
        description="Allow progress values to decrease (rollback scenarios)"
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
                    "name": "progress_tracker",
                    "description": "Progress tracking for web workflow",
                    "milestone_percentage_points": [25.0, 50.0, 75.0, 100.0],
                    "progress_update_interval_seconds": 2.0,
                    "enable_detailed_tracking": True,
                    "max_expected_duration_seconds": 300.0,
                    "enable_eta_calculation": True
                }
            ],
            "nanobrain_metadata": {
                "framework_version": "2.0.0",
                "component_type": "progress_tracking_step",
                "config_loading_method": "from_config_only",
                "supports_recursive_references": True
            }
        }
    )


class ProgressTrackingStep(BaseStep):
    """
    Progress Tracking Step - Pure Progress Monitoring and Data Management
    ====================================================================
    
    Provides focused progress tracking functionality as a proper NanoBrain workflow step,
    enabling progress monitoring within event-driven workflow orchestration with complete
    framework compliance.
    
    **SINGLE RESPONSIBILITY**: Progress Tracking Only
    - Progress data collection and calculation
    - Milestone detection and tracking
    - Progress data unit updates
    - Status and timeline management
    
    **DOES NOT INCLUDE**:
    - WebSocket communication (handled by WebInterfaceStep)
    - Real-time notifications (handled by notification components)
    - Complex business logic (delegated to workflow steps)
    
    **Core Architecture:**
        This step provides clean progress tracking that integrates with NanoBrain's
        event-driven workflow orchestration by managing progress data and detecting
        milestones, enabling responsive progress monitoring systems.
        
        * **Progress Calculation**: Accurate progress percentage and ETA calculations
        * **Milestone Detection**: Configurable milestone detection and event triggering
        * **Data Management**: Progress data unit updates and history preservation
        * **Validation**: Progress value validation and consistency checking
    
    **Configuration Architecture:**
        ```yaml
        # Basic progress tracking step configuration
        name: "progress_tracker"
        description: "Progress tracking for workflow monitoring"
        auto_initialize: true
        enable_logging: true
        
        # Progress tracking settings
        milestone_percentage_points: [25.0, 50.0, 75.0, 100.0]
        progress_update_interval_seconds: 1.0
        enable_detailed_tracking: true
        max_expected_duration_seconds: 300.0
        enable_eta_calculation: true
        
        # Data management
        preserve_progress_history: true
        max_history_entries: 100
        enable_progress_validation: true
        ```
    
    **Usage Patterns:**
        ```python
        from nanobrain.library.infrastructure.steps import ProgressTrackingStep
        
        # Create step from configuration
        progress_step = ProgressTrackingStep.from_config('config/progress_tracking_step.yml')
        
        # Execute within workflow context
        await progress_step.execute()
        
        # Step automatically handles:
        # - Progress calculation and validation
        # - Milestone detection and triggering
        # - Progress data unit updates
        # - ETA calculations and history tracking
        ```
    
    Attributes:
        name (str): Step identifier for logging and debugging
        description (str): Human-readable step description
        logger (logging.Logger): Step-specific logger instance
        config (ProgressTrackingStepConfig): Step configuration instance
        current_progress (float): Current progress percentage (0-100)
        start_time (float): Progress tracking start timestamp
        progress_history (List[Dict]): Historical progress data entries
        
    Note:
        This step follows the mandatory from_config pattern and cannot be
        instantiated directly. All configurations must be loaded from YAML files
        using the from_config method.
    
    See Also:
        * :class:`BaseStep`: Base framework step interface
        * :class:`ProgressTrackingStepConfig`: Configuration schema
        * :class:`ProgressUpdateDataUnit`: Progress data storage
        * :class:`SessionContextDataUnit`: Session tracking integration
    """
    
    # MANDATORY COMPONENT METADATA
    COMPONENT_TYPE: str = "progress_tracking_step"
    REQUIRED_CONFIG_FIELDS: List[str] = ['name']
    
    # Define data unit interfaces
    input_data_units = {
        'routing_decisions': DataUnitMemory,  # Routing decisions to track
        'session_context': SessionContextDataUnit,  # Session context for tracking
    }
    
    output_data_units = {
        'progress_updates': ProgressUpdateDataUnit,  # Progress update results
        'session_context': SessionContextDataUnit,  # Updated session context
    }
    
    # Define triggers
    triggers = {
        'progress_milestone': TriggerBase,  # Triggered when milestone reached
        'progress_complete': TriggerBase,  # Triggered when progress completes
    }
    
    def __init__(self):
        """Initialize Progress Tracking Step - use from_config for creation"""
        super().__init__()
        # Prevent direct instantiation
        if not hasattr(self, '_from_config_called'):
            raise RuntimeError(
                "Direct instantiation of ProgressTrackingStep is prohibited. "
                "Use: ProgressTrackingStep.from_config(config_file_or_object)"
            )
    
    @classmethod
    def _get_config_class(cls):
        """Return the configuration class for this component"""
        return ProgressTrackingStepConfig
    
    def _init_from_config(self, config: ProgressTrackingStepConfig, component_config: Dict[str, Any], dependencies: Dict[str, Any]) -> None:
        """
        Initialize step from validated configuration.
        
        Args:
            config: Validated ProgressTrackingStepConfig instance
            component_config: Component-specific configuration data
            dependencies: Resolved component dependencies
        """
        # Call parent initialization
        super()._init_from_config(config, component_config, dependencies)
        
        # Store configuration
        self.config = config
        self.name = config.name
        self.description = config.description
        
        # Initialize logging
        self.logger = get_logger(
            f"{self.__class__.__module__}.{self.__class__.__name__}",
            debug_mode=config.debug_mode
        )
        
        # Progress tracking state
        self.current_progress = 0.0
        self.start_time = None
        self.last_update_time = 0.0
        self.progress_history = []
        self.milestone_index = 0
        self.is_complete = False
        
        # Configuration-driven settings
        self.milestone_points = sorted(config.milestone_percentage_points)
        self.update_interval = config.progress_update_interval_seconds
        self.enable_detailed_tracking = config.enable_detailed_tracking
        self.max_duration = config.max_expected_duration_seconds
        self.enable_eta = config.enable_eta_calculation
        self.preserve_history = config.preserve_progress_history
        self.max_history = config.max_history_entries
        self.enable_validation = config.enable_progress_validation
        self.allow_rollback = config.allow_progress_rollback
        
        self.logger.info(
            f"ProgressTrackingStep initialized successfully",
            extra={
                "component_name": self.name,
                "milestone_points": len(self.milestone_points),
                "update_interval": self.update_interval,
                "max_duration": self.max_duration
            }
        )
    
    async def process(self, input_data: Dict[str, Any], **kwargs) -> Any:
        """
        Process routing decisions and update progress tracking.
        
        For ProgressTrackingStep, this method handles progress calculation,
        milestone detection, and progress data unit updates based on routing
        decisions and workflow state.
        
        Args:
            input_data: Dictionary containing routing_decisions and session_context
            **kwargs: Additional parameters
            
        Returns:
            Progress tracking results with updated progress data
        """
        try:
            self.logger.debug("Processing ProgressTrackingStep data flow")
            
            # Extract input data
            routing_decisions = input_data.get('routing_decisions', {})
            session_context = input_data.get('session_context', {})
            
            # Initialize progress tracking if needed
            if self.start_time is None:
                self.start_time = time.time()
                self.logger.info("Starting progress tracking session")
            
            # Calculate current progress based on routing decisions
            current_progress = await self._calculate_progress(routing_decisions, session_context)
            
            # Update progress if changed significantly
            if await self._should_update_progress(current_progress):
                await self._update_progress(current_progress, routing_decisions)
            
            # Check for milestone events
            milestone_reached = await self._check_milestones(current_progress)
            
            # Create progress update data
            progress_data = await self._create_progress_data(
                current_progress, routing_decisions, session_context, milestone_reached
            )
            
            # Update session context with progress information
            updated_session = await self._update_session_context(session_context, progress_data)
            
            results = {
                'progress_updates': progress_data,
                'session_context': updated_session,
                'current_progress': current_progress,
                'milestone_reached': milestone_reached,
                'is_complete': self.is_complete,
                'processing_time': time.time() - self.start_time
            }
            
            self.logger.debug(
                f"ProgressTrackingStep processing completed",
                extra={
                    "current_progress": current_progress,
                    "milestone_reached": milestone_reached,
                    "is_complete": self.is_complete
                }
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"ProgressTrackingStep processing failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'progress_updates': {},
                'session_context': input_data.get('session_context', {}),
                'current_progress': self.current_progress
            }
    
    async def _calculate_progress(self, routing_decisions: Dict[str, Any], session_context: Dict[str, Any]) -> float:
        """
        Calculate current progress percentage based on routing decisions and workflow state.
        
        Args:
            routing_decisions: Current routing decision data
            session_context: Session context with workflow information
            
        Returns:
            Progress percentage (0.0 to 100.0)
        """
        try:
            # Extract workflow state information
            selected_workflow = routing_decisions.get('selected_workflow', '')
            routing_confidence = routing_decisions.get('routing_confidence', 0.0)
            
            # Base progress calculation
            progress = 0.0
            
            # Progress based on routing completion
            if selected_workflow:
                progress += 25.0  # 25% for successful routing
                
            if routing_confidence > 0.7:
                progress += 15.0  # Additional 15% for high confidence routing
            
            # Progress based on workflow execution state
            workflow_state = session_context.get('workflow_state', {})
            completed_steps = workflow_state.get('completed_steps', [])
            total_steps = workflow_state.get('total_steps', 4)  # Default workflow steps
            
            if completed_steps and total_steps > 0:
                step_progress = (len(completed_steps) / total_steps) * 60.0  # 60% for step completion
                progress += step_progress
            
            # Ensure progress is within bounds
            progress = max(0.0, min(100.0, progress))
            
            # Apply validation if enabled
            if self.enable_validation:
                progress = await self._validate_progress(progress)
            
            return progress
            
        except Exception as e:
            self.logger.warning(f"Progress calculation failed, using previous value: {e}")
            return self.current_progress
    
    async def _should_update_progress(self, new_progress: float) -> bool:
        """
        Determine if progress should be updated based on interval and change thresholds.
        
        Args:
            new_progress: New progress value to consider
            
        Returns:
            True if progress should be updated
        """
        current_time = time.time()
        
        # Check minimum update interval
        if current_time - self.last_update_time < self.update_interval:
            return False
        
        # Check if progress has changed significantly (>= 1%)
        progress_change = abs(new_progress - self.current_progress)
        if progress_change < 1.0:
            return False
        
        # Check rollback validation
        if not self.allow_rollback and new_progress < self.current_progress:
            self.logger.warning(f"Progress rollback detected: {self.current_progress} -> {new_progress}")
            return False
        
        return True
    
    async def _update_progress(self, progress: float, routing_decisions: Dict[str, Any]) -> None:
        """
        Update current progress and maintain progress history.
        
        Args:
            progress: New progress percentage
            routing_decisions: Current routing decisions for context
        """
        previous_progress = self.current_progress
        self.current_progress = progress
        self.last_update_time = time.time()
        
        # Update completion status
        if progress >= 100.0:
            self.is_complete = True
        
        # Add to progress history if enabled
        if self.preserve_history:
            history_entry = {
                'timestamp': self.last_update_time,
                'progress': progress,
                'previous_progress': previous_progress,
                'routing_workflow': routing_decisions.get('selected_workflow', ''),
                'routing_confidence': routing_decisions.get('routing_confidence', 0.0)
            }
            
            self.progress_history.append(history_entry)
            
            # Limit history size
            if len(self.progress_history) > self.max_history:
                self.progress_history = self.progress_history[-self.max_history:]
        
        self.logger.debug(
            f"Progress updated: {previous_progress:.1f}% -> {progress:.1f}%",
            extra={
                "previous_progress": previous_progress,
                "current_progress": progress,
                "is_complete": self.is_complete
            }
        )
    
    async def _check_milestones(self, current_progress: float) -> bool:
        """
        Check if any progress milestones have been reached.
        
        Args:
            current_progress: Current progress percentage
            
        Returns:
            True if a milestone was reached
        """
        milestone_reached = False
        
        # Check each milestone point
        while (self.milestone_index < len(self.milestone_points) and 
               current_progress >= self.milestone_points[self.milestone_index]):
            
            milestone = self.milestone_points[self.milestone_index]
            self.logger.info(f"Progress milestone reached: {milestone}%")
            milestone_reached = True
            self.milestone_index += 1
        
        return milestone_reached
    
    async def _validate_progress(self, progress: float) -> float:
        """
        Validate progress value for consistency and bounds.
        
        Args:
            progress: Progress value to validate
            
        Returns:
            Validated progress value
        """
        # Ensure bounds
        if progress < 0.0:
            self.logger.warning(f"Progress below 0%: {progress}, setting to 0%")
            progress = 0.0
        elif progress > 100.0:
            self.logger.warning(f"Progress above 100%: {progress}, setting to 100%")
            progress = 100.0
        
        # Check for unrealistic jumps
        if self.current_progress > 0:
            jump = progress - self.current_progress
            if jump > 50.0:  # More than 50% jump
                self.logger.warning(f"Large progress jump detected: {jump:.1f}%")
        
        return progress
    
    async def _calculate_eta(self, current_progress: float) -> Optional[float]:
        """
        Calculate estimated time of arrival based on current progress.
        
        Args:
            current_progress: Current progress percentage
            
        Returns:
            Estimated seconds remaining, or None if cannot calculate
        """
        if not self.enable_eta or self.start_time is None or current_progress <= 0:
            return None
        
        elapsed_time = time.time() - self.start_time
        
        if current_progress >= 100.0:
            return 0.0
        
        # Simple linear projection
        estimated_total_time = elapsed_time * (100.0 / current_progress)
        estimated_remaining = estimated_total_time - elapsed_time
        
        # Apply reasonable bounds
        estimated_remaining = max(0.0, min(estimated_remaining, self.max_duration))
        
        return estimated_remaining
    
    async def _create_progress_data(self, progress: float, routing_decisions: Dict[str, Any], 
                                   session_context: Dict[str, Any], milestone_reached: bool) -> DataUnitBase:
        """
        Create ProgressUpdateDataUnit with current progress information.
        
        Args:
            progress: Current progress percentage
            routing_decisions: Routing decision context
            session_context: Session context information
            milestone_reached: Whether a milestone was reached
            
        Returns:
            ProgressUpdateDataUnit with progress data
        """
        try:
            # Calculate ETA if enabled
            eta_seconds = await self._calculate_eta(progress) if self.enable_eta else None
            
            # Prepare progress data
            progress_data = {
                'job_id': session_context.get('session_id', str(uuid.uuid4())),
                'current_progress': progress,
                'previous_progress': self.current_progress if hasattr(self, 'current_progress') else 0.0,
                'milestone_reached': milestone_reached,
                'is_complete': self.is_complete,
                'current_step': routing_decisions.get('selected_workflow', 'processing'),
                'status': 'completed' if self.is_complete else 'in_progress',
                'eta_seconds': eta_seconds,
                'elapsed_time': time.time() - self.start_time if self.start_time else 0.0,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'routing_context': {
                    'selected_workflow': routing_decisions.get('selected_workflow', ''),
                    'routing_confidence': routing_decisions.get('routing_confidence', 0.0)
                }
            }
            
            # Add detailed tracking data if enabled
            if self.enable_detailed_tracking:
                progress_data.update({
                    'milestone_points': self.milestone_points,
                    'next_milestone': (self.milestone_points[self.milestone_index] 
                                     if self.milestone_index < len(self.milestone_points) else None),
                    'progress_history_count': len(self.progress_history),
                    'update_interval': self.update_interval
                })
            
            # Create data unit using DataUnitMemory for in-memory storage
            data_unit = DataUnitMemory.from_config({
                'class': 'nanobrain.core.data_unit.DataUnitMemory',
                'name': 'progress_update',
                'description': 'Progress update data',
                'cache_size': 100,
                'persistent': False
            })
            
            await data_unit.set(progress_data)
            return data_unit
            
        except Exception as e:
            self.logger.error(f"Failed to create progress data unit: {e}")
            raise ComponentConfigurationError(f"Progress data creation failed: {e}")
    
    async def _update_session_context(self, session_context: Dict[str, Any], 
                                     progress_data: DataUnitBase) -> DataUnitBase:
        """
        Update session context with progress information.
        
        Args:
            session_context: Current session context data
            progress_data: Progress data to integrate
            
        Returns:
            Updated SessionContextDataUnit
        """
        try:
            # Get progress data
            progress_info = await progress_data.get()
            
            # Update session context with progress information
            updated_context = session_context.copy()
            updated_context.update({
                'last_progress_update': time.time(),
                'current_progress': progress_info.get('current_progress', 0.0),
                'is_complete': progress_info.get('is_complete', False),
                'progress_tracking_active': True,
                'eta_seconds': progress_info.get('eta_seconds')
            })
            
            # Create updated session context data unit
            session_data_unit = DataUnitMemory.from_config({
                'class': 'nanobrain.core.data_unit.DataUnitMemory',
                'name': 'session_context',
                'description': 'Session context data',
                'cache_size': 100,
                'persistent': True
            })
            
            await session_data_unit.set(updated_context)
            return session_data_unit
            
        except Exception as e:
            self.logger.error(f"Failed to update session context: {e}")
            # Return original context as fallback
            fallback_unit = DataUnitMemory.from_config({
                'class': 'nanobrain.core.data_unit.DataUnitMemory',
                'name': 'session_context_fallback',
                'description': 'Fallback session context',
                'cache_size': 100,
                'persistent': False
            })
            await fallback_unit.set(session_context)
            return fallback_unit
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current progress tracking status and metrics.
        
        Returns:
            Dictionary containing step status information
        """
        eta_seconds = None
        if self.enable_eta and self.start_time:
            eta_seconds = asyncio.create_task(self._calculate_eta(self.current_progress))
        
        return {
            "name": self.name,
            "type": self.COMPONENT_TYPE,
            "status": "operational",
            "progress_state": {
                "current_progress": self.current_progress,
                "is_complete": self.is_complete,
                "milestone_index": self.milestone_index,
                "total_milestones": len(self.milestone_points),
                "start_time": self.start_time,
                "last_update_time": self.last_update_time,
                "history_entries": len(self.progress_history)
            },
            "configuration": {
                "milestone_points": self.milestone_points,
                "update_interval": self.update_interval,
                "enable_detailed_tracking": self.enable_detailed_tracking,
                "enable_eta": self.enable_eta,
                "preserve_history": self.preserve_history
            }
        }
    
    def cleanup(self) -> None:
        """Clean up progress tracking resources"""
        self.logger.info(f"Cleaning up ProgressTrackingStep")
        self.progress_history.clear()
        self.current_progress = 0.0
        self.start_time = None
        self.is_complete = False 