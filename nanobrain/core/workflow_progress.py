"""
Workflow Progress Tracking System
=================================

Extracted from workflow.py monolith to provide dedicated progress tracking
functionality with debug/production error handling modes.
"""

import asyncio
import logging
import time
import os
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class ErrorMode(Enum):
    """Error handling modes for the framework."""
    DEBUG = "debug"      # Raise exceptions immediately
    PRODUCTION = "production"  # Graceful degradation


class ProgressTrackingError(Exception):
    """Base exception for progress tracking errors."""
    pass


def get_error_mode() -> ErrorMode:
    """Get current error handling mode from environment or default to production."""
    mode = os.environ.get('NANOBRAIN_ERROR_MODE', 'production').lower()
    return ErrorMode.DEBUG if mode == 'debug' else ErrorMode.PRODUCTION


def handle_error(error: Exception, context: str, default_return=None):
    """
    Handle errors based on current mode.
    
    Args:
        error: The exception that occurred
        context: Description of where the error occurred
        default_return: Value to return in production mode
    
    Returns:
        default_return in production mode
        
    Raises:
        error in debug mode
    """
    error_mode = get_error_mode()
    
    if error_mode == ErrorMode.DEBUG:
        logger.error(f"DEBUG MODE: {context} - {error}", exc_info=True)
        raise error
    else:
        logger.warning(f"PRODUCTION MODE: {context} - {error} (graceful degradation)")
        return default_return


@dataclass
class ProgressStep:
    """Individual step progress information."""
    step_id: str
    name: str
    description: str
    status: str  # 'pending', 'running', 'completed', 'failed', 'skipped'
    progress_percentage: int = 0
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    elapsed_time: float = 0.0
    estimated_time: Optional[float] = None
    error_message: Optional[str] = None
    technical_details: Optional[Dict[str, Any]] = None
    checkpoint_data: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        try:
            return asdict(self)
        except Exception as e:
            return handle_error(e, "ProgressStep.to_dict", {})

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProgressStep':
        """Create from dictionary."""
        try:
            return cls(**data)
        except Exception as e:
            return handle_error(e, "ProgressStep.from_dict", cls(
                step_id="unknown", name="Unknown", description="", status="failed"
            ))


@dataclass
class WorkflowProgress:
    """Complete workflow progress information."""
    workflow_id: str
    workflow_name: str
    session_id: Optional[str] = None
    overall_progress: int = 0
    status: str = 'pending'  # 'pending', 'running', 'completed', 'failed', 'paused'
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    estimated_total_time: Optional[float] = None
    steps: List[ProgressStep] = field(default_factory=list)
    current_step_index: int = 0
    error_message: Optional[str] = None
    last_updated: float = field(default_factory=time.time)

    # Progress reporting configuration
    batch_interval: float = 3.0  # Batch progress every 3 seconds
    collapsed_by_default: bool = True
    show_technical_errors: bool = True
    preserve_session_history: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        try:
            data = asdict(self)
            data['steps'] = [step.to_dict() if isinstance(step, ProgressStep)
                             else step for step in self.steps]
            return data
        except Exception as e:
            return handle_error(e, "WorkflowProgress.to_dict", {
                "workflow_id": self.workflow_id,
                "workflow_name": self.workflow_name,
                "status": "error",
                "error_message": str(e)
            })

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkflowProgress':
        """Create from dictionary."""
        try:
            steps_data = data.pop('steps', [])
            progress = cls(**data)
            progress.steps = [ProgressStep.from_dict(step) if isinstance(
                step, dict) else step for step in steps_data]
            return progress
        except Exception as e:
            return handle_error(e, "WorkflowProgress.from_dict", cls(
                workflow_id="unknown", workflow_name="Unknown Workflow"
            ))

    def get_current_step(self) -> Optional[ProgressStep]:
        """Get currently executing step."""
        try:
            if 0 <= self.current_step_index < len(self.steps):
                return self.steps[self.current_step_index]
            return None
        except Exception as e:
            return handle_error(e, "WorkflowProgress.get_current_step", None)

    def update_step_progress(self, step_id: str, progress: int, status: str = None,
                             error: str = None, technical_details: Dict[str, Any] = None) -> None:
        """Update progress for a specific step."""
        try:
            for step in self.steps:
                if step.step_id == step_id:
                    step.progress_percentage = max(0, min(100, progress))
                    if status:
                        step.status = status
                    if error:
                        step.error_message = error
                    if technical_details:
                        step.technical_details = technical_details

                    # Update timing
                    current_time = time.time()
                    if status == 'running' and not step.start_time:
                        step.start_time = current_time
                    elif status in ['completed', 'failed'] and step.start_time:
                        step.end_time = current_time
                        step.elapsed_time = current_time - step.start_time

                    self.last_updated = current_time
                    break
        except Exception as e:
            handle_error(e, f"WorkflowProgress.update_step_progress for {step_id}")

    def calculate_overall_progress(self) -> int:
        """Calculate overall workflow progress."""
        try:
            if not self.steps:
                return 0

            total_progress = sum(step.progress_percentage for step in self.steps)
            return min(100, total_progress // len(self.steps))
        except Exception as e:
            return handle_error(e, "WorkflowProgress.calculate_overall_progress", 0)


class ProgressReporter:
    """Handles progress reporting for workflows."""

    def __init__(self, workflow_id: str, workflow_name: str, session_id: str = None):
        try:
            self.workflow_progress = WorkflowProgress(
                workflow_id=workflow_id,
                workflow_name=workflow_name,
                session_id=session_id
            )
            self.progress_callbacks: List[Callable] = []
            self.last_batch_time = 0.0
            self.progress_history: List[Dict[str, Any]] = []
            self.checkpoint_storage: Dict[str, Any] = {}
        except Exception as e:
            handle_error(e, f"ProgressReporter.__init__ for {workflow_id}")
            # Fallback initialization
            self.workflow_progress = WorkflowProgress(
                workflow_id=workflow_id or "unknown",
                workflow_name=workflow_name or "Unknown Workflow"
            )
            self.progress_callbacks = []
            self.last_batch_time = 0.0
            self.progress_history = []
            self.checkpoint_storage = {}

    def add_progress_callback(self, callback: Callable) -> None:
        """Add callback for progress updates."""
        try:
            if callable(callback):
                self.progress_callbacks.append(callback)
            else:
                raise ValueError(f"Callback must be callable, got {type(callback)}")
        except Exception as e:
            handle_error(e, "ProgressReporter.add_progress_callback")

    def initialize_steps(self, step_configs) -> None:
        """Initialize progress steps from configuration."""
        try:
            self.workflow_progress.steps = []

            # Handle both dict and list formats
            if isinstance(step_configs, dict):
                # New dict-based format: steps = {step_id: step_object}
                for step_id, step_obj in step_configs.items():
                    # Check if it's an actual step object or a config dict
                    if hasattr(step_obj, 'name') and hasattr(step_obj, 'description'):
                        # It's an instantiated step object
                        step = ProgressStep(
                            step_id=step_id,
                            name=getattr(step_obj, 'name',
                                         step_id.replace('_', ' ').title()),
                            description=getattr(step_obj, 'description', ''),
                            status='pending',
                            estimated_time=getattr(
                                step_obj, 'estimated_time', None)
                        )
                    else:
                        # It's a config dictionary
                        step = ProgressStep(
                            step_id=step_obj.get('step_id', step_id),
                            name=step_obj.get(
                                'name', step_id.replace('_', ' ').title()),
                            description=step_obj.get('description', ''),
                            status='pending',
                            estimated_time=step_obj.get('estimated_time')
                        )
                    self.workflow_progress.steps.append(step)
            else:
                # Legacy list-based format: steps = [step_config, ...]
                for i, step_config in enumerate(step_configs or []):
                    step = ProgressStep(
                        step_id=step_config.get('step_id', f'step_{i}'),
                        name=step_config.get('name', f'Step {i+1}'),
                        description=step_config.get('description', ''),
                        status='pending',
                        estimated_time=step_config.get('estimated_time')
                    )
                    self.workflow_progress.steps.append(step)
        except Exception as e:
            handle_error(e, "ProgressReporter.initialize_steps")

    async def update_progress(self, step_id: str, progress: int, status: str = None,
                              message: str = None, error: str = None,
                              technical_details: Dict[str, Any] = None,
                              force_emit: bool = False) -> None:
        """Update step progress with batched reporting."""
        try:
            # Update step progress
            self.workflow_progress.update_step_progress(
                step_id, progress, status, error, technical_details
            )

            # Update overall progress
            self.workflow_progress.overall_progress = self.workflow_progress.calculate_overall_progress()

            # Store checkpoint data
            if status in ['completed', 'failed'] or progress == 100:
                await self._save_checkpoint(step_id)

            # Emit progress updates (batched)
            current_time = time.time()
            should_emit = (
                force_emit or
                (current_time - self.last_batch_time) >= self.workflow_progress.batch_interval or
                status in ['completed', 'failed'] or
                progress == 100
            )

            if should_emit:
                await self._emit_progress_update()
                self.last_batch_time = current_time
        except Exception as e:
            handle_error(e, f"ProgressReporter.update_progress for {step_id}")

    async def _emit_progress_update(self) -> None:
        """Emit progress update to all callbacks."""
        try:
            progress_data = self.workflow_progress.to_dict()

            # Add to history if preserving session history
            if self.workflow_progress.preserve_session_history:
                self.progress_history.append({
                    'timestamp': time.time(),
                    'progress': progress_data.copy()
                })

            # Call all registered callbacks
            for callback in self.progress_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(progress_data)
                    else:
                        callback(progress_data)
                except Exception as callback_error:
                    logger.error(f"Progress callback failed: {callback_error}", exc_info=True)
        except Exception as e:
            handle_error(e, "ProgressReporter._emit_progress_update")

    async def _save_checkpoint(self, step_id: str) -> None:
        """Save checkpoint data for step recovery."""
        try:
            step = next(
                (s for s in self.workflow_progress.steps if s.step_id == step_id), None)
            if step and step.checkpoint_data:
                self.checkpoint_storage[step_id] = {
                    'timestamp': time.time(),
                    'step_data': step.to_dict(),
                    'checkpoint_data': step.checkpoint_data
                }
        except Exception as e:
            handle_error(e, f"ProgressReporter._save_checkpoint for {step_id}")

    async def restore_from_checkpoint(self, step_id: str) -> Optional[Dict[str, Any]]:
        """Restore checkpoint data for step recovery."""
        try:
            return self.checkpoint_storage.get(step_id)
        except Exception as e:
            return handle_error(e, f"ProgressReporter.restore_from_checkpoint for {step_id}", None)

    def get_progress_summary(self) -> Dict[str, Any]:
        """Get condensed progress summary for UI."""
        try:
            current_step = self.workflow_progress.get_current_step()

            return {
                'workflow_id': self.workflow_progress.workflow_id,
                'workflow_name': self.workflow_progress.workflow_name,
                'overall_progress': self.workflow_progress.overall_progress,
                'status': self.workflow_progress.status,
                'current_step': {
                    'name': current_step.name if current_step else None,
                    'progress': current_step.progress_percentage if current_step else 0,
                    'status': current_step.status if current_step else 'pending'
                } if current_step else None,
                'collapsed': self.workflow_progress.collapsed_by_default,
                'estimated_time_remaining': self._calculate_estimated_time_remaining(),
                'last_updated': self.workflow_progress.last_updated
            }
        except Exception as e:
            return handle_error(e, "ProgressReporter.get_progress_summary", {
                'workflow_id': self.workflow_progress.workflow_id,
                'workflow_name': self.workflow_progress.workflow_name,
                'status': 'error',
                'error_message': str(e)
            })

    def _calculate_estimated_time_remaining(self) -> Optional[float]:
        """Calculate estimated time remaining."""
        try:
            if not self.workflow_progress.steps:
                return None

            completed_steps = [
                s for s in self.workflow_progress.steps if s.status == 'completed']
            if not completed_steps:
                return None

            avg_time_per_step = sum(
                s.elapsed_time for s in completed_steps) / len(completed_steps)
            remaining_steps = len(
                [s for s in self.workflow_progress.steps if s.status == 'pending'])

            return avg_time_per_step * remaining_steps
        except Exception as e:
            return handle_error(e, "ProgressReporter._calculate_estimated_time_remaining", None)
