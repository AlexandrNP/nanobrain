"""
Workflow System for NanoBrain Framework

Provides graph-based workflow orchestration extending the Step system.
Workflows can contain multiple steps connected by links, with support for
hierarchical configuration, various execution strategies, and comprehensive
progress reporting with persistent checkpoints.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Set, Tuple, Callable
from pathlib import Path
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict
import yaml
import json
from datetime import datetime, timezone
from dataclasses import dataclass, asdict, field
from collections import defaultdict

from .step import Step, StepConfig
from .data_unit import DataUnitBase, DataUnitConfig
from .trigger import TriggerBase, TriggerConfig
from .link import LinkBase, DirectLink, ConditionalLink, TransformLink, LinkConfig, LinkType
from .executor import ExecutorBase, LocalExecutor, ExecutorConfig
from .logging_system import get_logger, OperationType

logger = logging.getLogger(__name__)


class ExecutionStrategy(Enum):
    """Workflow execution strategies."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    GRAPH_BASED = "graph_based"
    EVENT_DRIVEN = "event_driven"


class ErrorHandlingStrategy(Enum):
    """Error handling strategies for workflows."""
    CONTINUE = "continue"
    STOP = "stop"
    RETRY = "retry"
    ROLLBACK = "rollback"


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
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProgressStep':
        """Create from dictionary."""
        return cls(**data)


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
        data = asdict(self)
        data['steps'] = [step.to_dict() if isinstance(step, ProgressStep) else step for step in self.steps]
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkflowProgress':
        """Create from dictionary."""
        steps_data = data.pop('steps', [])
        progress = cls(**data)
        progress.steps = [ProgressStep.from_dict(step) if isinstance(step, dict) else step for step in steps_data]
        return progress
    
    def get_current_step(self) -> Optional[ProgressStep]:
        """Get currently executing step."""
        if 0 <= self.current_step_index < len(self.steps):
            return self.steps[self.current_step_index]
        return None
    
    def update_step_progress(self, step_id: str, progress: int, status: str = None, 
                           error: str = None, technical_details: Dict[str, Any] = None) -> None:
        """Update progress for a specific step."""
        for step in self.steps:
            if step.step_id == step_id:
                step.progress_percentage = progress
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
    
    def calculate_overall_progress(self) -> int:
        """Calculate overall workflow progress."""
        if not self.steps:
            return 0
        
        total_progress = sum(step.progress_percentage for step in self.steps)
        return min(100, total_progress // len(self.steps))


class ProgressReporter:
    """Handles progress reporting for workflows."""
    
    def __init__(self, workflow_id: str, workflow_name: str, session_id: str = None):
        self.workflow_progress = WorkflowProgress(
            workflow_id=workflow_id,
            workflow_name=workflow_name,
            session_id=session_id
        )
        self.progress_callbacks: List[Callable] = []
        self.last_batch_time = 0.0
        self.progress_history: List[Dict[str, Any]] = []
        self.checkpoint_storage: Dict[str, Any] = {}
        
    def add_progress_callback(self, callback: Callable) -> None:
        """Add callback for progress updates."""
        self.progress_callbacks.append(callback)
    
    def initialize_steps(self, step_configs: List[Dict[str, Any]]) -> None:
        """Initialize progress steps from configuration."""
        self.workflow_progress.steps = []
        
        for i, step_config in enumerate(step_configs):
            step = ProgressStep(
                step_id=step_config.get('step_id', f'step_{i}'),
                name=step_config.get('name', f'Step {i+1}'),
                description=step_config.get('description', ''),
                status='pending',
                estimated_time=step_config.get('estimated_time')
            )
            self.workflow_progress.steps.append(step)
    
    async def update_progress(self, step_id: str, progress: int, status: str = None,
                            message: str = None, error: str = None,
                            technical_details: Dict[str, Any] = None,
                            force_emit: bool = False) -> None:
        """Update step progress with batched reporting."""
        
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
    
    async def _emit_progress_update(self) -> None:
        """Emit progress update to all callbacks."""
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
            except Exception as e:
                logger.error(f"Progress callback failed: {e}")
    
    async def _save_checkpoint(self, step_id: str) -> None:
        """Save checkpoint data for step recovery."""
        step = next((s for s in self.workflow_progress.steps if s.step_id == step_id), None)
        if step and step.checkpoint_data:
            self.checkpoint_storage[step_id] = {
                'timestamp': time.time(),
                'step_data': step.to_dict(),
                'checkpoint_data': step.checkpoint_data
            }
    
    async def restore_from_checkpoint(self, step_id: str) -> Optional[Dict[str, Any]]:
        """Restore checkpoint data for step recovery."""
        return self.checkpoint_storage.get(step_id)
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get condensed progress summary for UI."""
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
    
    def _calculate_estimated_time_remaining(self) -> Optional[float]:
        """Calculate estimated time remaining."""
        if not self.workflow_progress.steps:
            return None
        
        completed_steps = [s for s in self.workflow_progress.steps if s.status == 'completed']
        if not completed_steps:
            return None
        
        avg_time_per_step = sum(s.elapsed_time for s in completed_steps) / len(completed_steps)
        remaining_steps = len([s for s in self.workflow_progress.steps if s.status == 'pending'])
        
        return avg_time_per_step * remaining_steps


class WorkflowConfig(StepConfig):
    """Configuration for workflows extending StepConfig."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # Workflow-specific configuration
    steps: List[Dict[str, Any]] = Field(default_factory=list)
    links: List[Dict[str, Any]] = Field(default_factory=list)
    execution_strategy: ExecutionStrategy = ExecutionStrategy.SEQUENTIAL
    error_handling: ErrorHandlingStrategy = ErrorHandlingStrategy.CONTINUE
    enable_monitoring: bool = True
    workflow_directory: Optional[str] = None
    
    # Execution configuration
    max_parallel_steps: int = 10
    step_timeout: float = 300.0  # 5 minutes
    retry_attempts: int = 3
    retry_delay: float = 1.0
    
    # Validation configuration
    validate_graph: bool = True
    allow_cycles: bool = False
    require_connected_graph: bool = True
    
    # Progress reporting configuration
    enable_progress_reporting: bool = True
    progress_batch_interval: float = 3.0
    progress_collapsed_by_default: bool = True
    progress_show_technical_errors: bool = True
    progress_preserve_session_history: bool = True


class WorkflowGraph:
    """
    Internal graph representation of workflow structure.
    
    Manages the graph of steps (nodes) and links (edges) within a workflow.
    Provides graph analysis capabilities including cycle detection and
    topological sorting for execution order determination.
    """
    
    def __init__(self):
        """Initialize empty workflow graph."""
        self.nodes: Dict[str, Step] = {}  # step_id -> Step instance
        self.edges: Dict[str, LinkBase] = {}  # link_id -> Link instance
        self.adjacency: Dict[str, Set[str]] = {}  # step_id -> set of connected step_ids
        self.reverse_adjacency: Dict[str, Set[str]] = {}  # step_id -> set of predecessor step_ids
        
        # Graph metadata
        self._is_valid = False
        self._execution_order: Optional[List[str]] = None
        self._strongly_connected_components: Optional[List[List[str]]] = None
        
        self.logger = get_logger("workflow.graph")
    
    def add_step(self, step_id: str, step: Step) -> None:
        """Add a step node to the graph."""
        if step_id in self.nodes:
            raise ValueError(f"Step {step_id} already exists in workflow graph")
        
        self.nodes[step_id] = step
        self.adjacency[step_id] = set()
        self.reverse_adjacency[step_id] = set()
        
        # Invalidate cached computations
        self._invalidate_cache()
        
        self.logger.debug(f"Added step to workflow graph: {step_id}")
    
    def add_link(self, link_id: str, link: LinkBase, source_id: str, target_id: str) -> None:
        """Add a link edge to the graph."""
        if link_id in self.edges:
            raise ValueError(f"Link {link_id} already exists in workflow graph")
        
        if source_id not in self.nodes:
            raise ValueError(f"Source step {source_id} not found in workflow graph")
        
        if target_id not in self.nodes:
            raise ValueError(f"Target step {target_id} not found in workflow graph")
        
        self.edges[link_id] = link
        self.adjacency[source_id].add(target_id)
        self.reverse_adjacency[target_id].add(source_id)
        
        # Invalidate cached computations
        self._invalidate_cache()
        
        self.logger.debug(f"Added link to workflow graph: {link_id} ({source_id} -> {target_id})")
    
    def remove_step(self, step_id: str) -> None:
        """Remove a step and all its connections from the graph."""
        if step_id not in self.nodes:
            raise ValueError(f"Step {step_id} not found in workflow graph")
        
        # Remove all edges involving this step
        edges_to_remove = []
        for link_id, link in self.edges.items():
            if (hasattr(link, 'source') and getattr(link.source, 'name', None) == step_id) or \
               (hasattr(link, 'target') and getattr(link.target, 'name', None) == step_id):
                edges_to_remove.append(link_id)
        
        for link_id in edges_to_remove:
            self.remove_link(link_id)
        
        # Remove from adjacency lists
        for connected_id in self.adjacency[step_id]:
            self.reverse_adjacency[connected_id].discard(step_id)
            
        for predecessor_id in self.reverse_adjacency[step_id]:
            self.adjacency[predecessor_id].discard(step_id)
        
        # Remove the step
        del self.nodes[step_id]
        del self.adjacency[step_id]
        del self.reverse_adjacency[step_id]
        
        self._invalidate_cache()
        self.logger.debug(f"Removed step from workflow graph: {step_id}")
    
    def remove_link(self, link_id: str) -> None:
        """Remove a link from the graph."""
        if link_id not in self.edges:
            raise ValueError(f"Link {link_id} not found in workflow graph")
        
        link = self.edges[link_id]
        
        # Find source and target step IDs
        source_id = getattr(link.source, 'name', None)
        target_id = getattr(link.target, 'name', None)
        
        if source_id and target_id:
            self.adjacency[source_id].discard(target_id)
            self.reverse_adjacency[target_id].discard(source_id)
        
        del self.edges[link_id]
        self._invalidate_cache()
        self.logger.debug(f"Removed link from workflow graph: {link_id}")
    
    def get_step(self, step_id: str) -> Optional[Step]:
        """Get a step by ID."""
        return self.nodes.get(step_id)
    
    def get_link(self, link_id: str) -> Optional[LinkBase]:
        """Get a link by ID."""
        return self.edges.get(link_id)
    
    def get_step_dependencies(self, step_id: str) -> Set[str]:
        """Get all steps that must execute before the given step."""
        if step_id not in self.nodes:
            raise ValueError(f"Step {step_id} not found in workflow graph")
        return self.reverse_adjacency[step_id].copy()
    
    def get_step_dependents(self, step_id: str) -> Set[str]:
        """Get all steps that depend on the given step."""
        if step_id not in self.nodes:
            raise ValueError(f"Step {step_id} not found in workflow graph")
        return self.adjacency[step_id].copy()
    
    def has_cycles(self) -> bool:
        """Check if the graph contains cycles using DFS."""
        color = {step_id: 0 for step_id in self.nodes}  # 0: white, 1: gray, 2: black
        
        def dfs(step_id: str) -> bool:
            if color[step_id] == 1:  # Back edge found - cycle detected
                return True
            if color[step_id] == 2:  # Already processed
                return False
            
            color[step_id] = 1  # Mark as being processed
            
            for neighbor in self.adjacency[step_id]:
                if dfs(neighbor):
                    return True
            
            color[step_id] = 2  # Mark as completely processed
            return False
        
        for step_id in self.nodes:
            if color[step_id] == 0:
                if dfs(step_id):
                    return True
        
        return False
    
    def get_execution_order(self) -> List[str]:
        """Get topological execution order using Kahn's algorithm."""
        if self._execution_order is not None:
            return self._execution_order.copy()
        
        # Kahn's algorithm for topological sorting
        in_degree = {step_id: len(self.reverse_adjacency[step_id]) for step_id in self.nodes}
        queue = [step_id for step_id, degree in in_degree.items() if degree == 0]
        execution_order = []
        
        while queue:
            current = queue.pop(0)
            execution_order.append(current)
            
            for neighbor in self.adjacency[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        if len(execution_order) != len(self.nodes):
            raise ValueError("Workflow graph contains cycles - cannot determine execution order")
        
        self._execution_order = execution_order
        return execution_order.copy()
    
    def get_parallel_execution_levels(self) -> List[List[str]]:
        """Get steps grouped by execution level for parallel execution."""
        execution_order = self.get_execution_order()
        levels = []
        processed = set()
        
        while processed != set(self.nodes.keys()):
            current_level = []
            
            for step_id in execution_order:
                if step_id in processed:
                    continue
                
                # Check if all dependencies are satisfied
                dependencies = self.get_step_dependencies(step_id)
                if dependencies.issubset(processed):
                    current_level.append(step_id)
            
            if not current_level:
                raise ValueError("Cannot determine parallel execution levels - possible circular dependency")
            
            levels.append(current_level)
            processed.update(current_level)
        
        return levels
    
    def validate_graph(self, allow_cycles: bool = False, require_connected: bool = True) -> Tuple[bool, List[str]]:
        """
        Validate the workflow graph structure.
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check for empty graph
        if not self.nodes:
            errors.append("Workflow graph is empty - no steps defined")
        
        # Check for cycles
        if not allow_cycles and self.has_cycles():
            errors.append("Workflow graph contains cycles")
        
        # Check for disconnected components if required
        if require_connected and len(self.nodes) > 1:
            if not self._is_weakly_connected():
                errors.append("Workflow graph is not connected - contains isolated components")
        
        # Check for orphaned steps (no inputs or outputs)
        orphaned_steps = []
        for step_id in self.nodes:
            has_input = len(self.reverse_adjacency[step_id]) > 0
            has_output = len(self.adjacency[step_id]) > 0
            
            if not has_input and not has_output and len(self.nodes) > 1:
                orphaned_steps.append(step_id)
        
        if orphaned_steps:
            errors.append(f"Orphaned steps found (no connections): {orphaned_steps}")
        
        # Validate that all links have valid source and target steps
        for link_id, link in self.edges.items():
            source_id = getattr(link.source, 'name', None)
            target_id = getattr(link.target, 'name', None)
            
            if source_id not in self.nodes:
                errors.append(f"Link {link_id} has invalid source step: {source_id}")
            
            if target_id not in self.nodes:
                errors.append(f"Link {link_id} has invalid target step: {target_id}")
        
        is_valid = len(errors) == 0
        self._is_valid = is_valid
        
        return is_valid, errors
    
    def _is_weakly_connected(self) -> bool:
        """Check if the graph is weakly connected (ignoring edge direction)."""
        if not self.nodes:
            return True
        
        visited = set()
        start_node = next(iter(self.nodes))
        stack = [start_node]
        
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            
            visited.add(current)
            
            # Add both successors and predecessors (treat as undirected)
            stack.extend(self.adjacency[current] - visited)
            stack.extend(self.reverse_adjacency[current] - visited)
        
        return len(visited) == len(self.nodes)
    
    def _invalidate_cache(self) -> None:
        """Invalidate cached computations when graph structure changes."""
        self._execution_order = None
        self._strongly_connected_components = None
        self._is_valid = False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        return {
            "num_steps": len(self.nodes),
            "num_links": len(self.edges),
            "has_cycles": self.has_cycles(),
            "is_connected": self._is_weakly_connected(),
            "max_depth": self._calculate_max_depth(),
            "avg_branching_factor": self._calculate_avg_branching_factor()
        }
    
    def _calculate_max_depth(self) -> int:
        """Calculate maximum depth of the graph."""
        if not self.nodes:
            return 0
        
        try:
            levels = self.get_parallel_execution_levels()
            return len(levels)
        except ValueError:
            # Graph has cycles, return -1
            return -1
    
    def _calculate_avg_branching_factor(self) -> float:
        """Calculate average branching factor."""
        if not self.nodes:
            return 0.0
        
        total_edges = sum(len(neighbors) for neighbors in self.adjacency.values())
        return total_edges / len(self.nodes)


class ConfigLoader:
    """
    Recursive configuration loader for workflows.
    
    Handles loading workflow configurations with support for:
    - Recursive loading of step configurations
    - Path resolution for nested configurations
    - Caching of loaded configurations
    - YAML schema validation
    """
    
    def __init__(self, base_path: str = "."):
        """Initialize configuration loader."""
        self.base_path = Path(base_path)
        self.loaded_configs: Dict[str, Dict] = {}  # Cache for loaded configurations
        self.logger = get_logger("workflow.config_loader")
    
    def load_workflow_config(self, config_path: str) -> WorkflowConfig:
        """Load workflow configuration from YAML file."""
        config_file = self.resolve_config_path(config_path)
        
        if str(config_file) in self.loaded_configs:
            config_data = self.loaded_configs[str(config_file)]
        else:
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            self.loaded_configs[str(config_file)] = config_data
        
        # Determine workflow directory for step loading
        workflow_dir = config_file.parent
        if 'workflow_directory' not in config_data:
            config_data['workflow_directory'] = str(workflow_dir)
        
        self.logger.info(f"Loaded workflow configuration: {config_file}")
        return WorkflowConfig(**config_data)
    
    def load_step_config(self, config_path: str, workflow_dir: Optional[str] = None) -> StepConfig:
        """Load step configuration from YAML file."""
        config_file = self.resolve_config_path(config_path, workflow_dir)
        
        if str(config_file) in self.loaded_configs:
            config_data = self.loaded_configs[str(config_file)]
        else:
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            self.loaded_configs[str(config_file)] = config_data
        
        self.logger.debug(f"Loaded step configuration: {config_file}")
        return StepConfig(**config_data)
    
    def resolve_config_path(self, config_file: str, workflow_dir: Optional[str] = None) -> Path:
        """Resolve configuration file path with proper search order."""
        # If absolute path, use as-is
        if Path(config_file).is_absolute():
            config_path = Path(config_file)
            if config_path.exists():
                return config_path
            else:
                raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        # Search paths in order of preference
        search_paths = []
        
        # 1. Workflow-specific directory
        if workflow_dir:
            search_paths.append(Path(workflow_dir) / config_file)
        
        # 2. Relative to base path
        search_paths.append(self.base_path / config_file)
        
        # 3. Current working directory
        search_paths.append(Path(config_file))
        
        # 4. Common configuration directories
        common_dirs = [
            "config", "configs", "workflow_configs", 
            "nanobrain/config", "library/config"
        ]
        for common_dir in common_dirs:
            search_paths.append(Path(common_dir) / config_file)
        
        # Try each search path
        for search_path in search_paths:
            if search_path.exists():
                return search_path
        
        # If not found, raise error with search paths
        search_paths_str = "\n".join(f"  - {path}" for path in search_paths)
        raise FileNotFoundError(
            f"Configuration file not found: {config_file}\n"
            f"Searched in:\n{search_paths_str}"
        )
    
    def clear_cache(self) -> None:
        """Clear the configuration cache."""
        self.loaded_configs.clear()
        self.logger.debug("Configuration cache cleared")


class Workflow(Step):
    """
    Base workflow class extending Step.
    
    Biological analogy: Neural circuit complex containing multiple interconnected circuits.
    Justification: Like how complex neural circuits are composed of simpler circuits 
    working together in coordination, workflows are composed of steps working together
    through defined connections and data flow patterns.
    
    A workflow can contain:
    - Multiple child steps (including other workflows)
    - Links connecting steps for data flow
    - Execution strategies for orchestrating step execution
    - Error handling and retry mechanisms
    - Performance monitoring and metrics collection
    """
    
    def __init__(self, config: WorkflowConfig, executor: Optional[ExecutorBase] = None, **kwargs):
        """Initialize workflow with configuration."""
        super().__init__(config, executor, **kwargs)
        
        # Workflow-specific configuration
        self.workflow_config = config
        
        # Core workflow components
        self.workflow_graph = WorkflowGraph()
        self.config_loader = ConfigLoader(config.workflow_directory or ".")
        
        # Step and link management
        self.child_steps: Dict[str, Step] = {}
        self.step_links: Dict[str, LinkBase] = {}
        
        # Execution state
        self.execution_order: List[str] = []
        self.current_step_index: int = 0
        self.is_workflow_complete: bool = False
        self.failed_steps: Set[str] = set()
        self.completed_steps: Set[str] = set()
        
        # Performance tracking
        self.step_execution_times: Dict[str, float] = {}
        self.workflow_start_time: Optional[float] = None
        self.workflow_end_time: Optional[float] = None
        
        # Progress reporting
        self.progress_reporter: Optional[ProgressReporter] = None
        if config.enable_progress_reporting:
            self.progress_reporter = ProgressReporter(
                workflow_id=f"{self.name}_{int(time.time())}",
                workflow_name=self.name,
                session_id=kwargs.get('session_id')
            )
            self.progress_reporter.workflow_progress.batch_interval = config.progress_batch_interval
            self.progress_reporter.workflow_progress.collapsed_by_default = config.progress_collapsed_by_default
            self.progress_reporter.workflow_progress.show_technical_errors = config.progress_show_technical_errors
            self.progress_reporter.workflow_progress.preserve_session_history = config.progress_preserve_session_history
        
        # Workflow-specific logger
        self.workflow_logger = get_logger(f"workflow.{self.name}", debug_mode=config.debug_mode)
        
        self.workflow_logger.info(f"Initialized workflow: {self.name}")
    
    async def initialize(self) -> None:
        """Initialize workflow: load steps, create links, build graph."""
        if self._is_initialized:
            return
        
        async with self.nb_logger.async_execution_context(
            OperationType.STEP_EXECUTE,
            f"{self.name}.initialize_workflow"
        ) as context:
            
            # Initialize as Step first
            await super().initialize()
            
            # Load workflow configuration
            await self._load_workflow_configuration()
            
            # Initialize child steps
            await self._initialize_child_steps()
            
            # Create step links
            await self._create_step_links()
            
            # Build and validate workflow graph
            await self._build_workflow_graph()
            await self._validate_workflow()
            
            # Determine execution order
            self._determine_execution_order()
            
            # Initialize progress reporting
            if self.progress_reporter:
                self.progress_reporter.initialize_steps(self.workflow_config.steps)
                await self.progress_reporter.update_progress(
                    'workflow_init', 100, 'completed', 
                    message="Workflow initialized successfully"
                )
            
            context.metadata['num_steps'] = len(self.child_steps)
            context.metadata['num_links'] = len(self.step_links)
            context.metadata['execution_strategy'] = self.workflow_config.execution_strategy.value
            
        self.workflow_logger.info(
            f"Workflow {self.name} initialized successfully",
            num_steps=len(self.child_steps),
            num_links=len(self.step_links),
            execution_strategy=self.workflow_config.execution_strategy.value
        )
    
    async def process(self, input_data: Dict[str, Any], **kwargs) -> Any:
        """Execute workflow by processing steps according to execution strategy."""
        async with self.nb_logger.async_execution_context(
            OperationType.STEP_EXECUTE,
            f"{self.name}.execute_workflow",
            execution_strategy=self.workflow_config.execution_strategy.value
        ) as context:
            
            self.workflow_start_time = time.time()
            self.is_workflow_complete = False
            self.failed_steps.clear()
            self.completed_steps.clear()
            
            try:
                result = await self._execute_workflow(input_data, **kwargs)
                self.is_workflow_complete = True
                
                self.workflow_end_time = time.time()
                execution_time = self.workflow_end_time - self.workflow_start_time
                
                context.metadata.update({
                    'execution_time_seconds': execution_time,
                    'completed_steps': len(self.completed_steps),
                    'failed_steps': len(self.failed_steps),
                    'success': True
                })
                
                self.workflow_logger.info(
                    f"Workflow {self.name} completed successfully",
                    execution_time_seconds=execution_time,
                    completed_steps=len(self.completed_steps),
                    failed_steps=len(self.failed_steps)
                )
                
                return result
                
            except Exception as e:
                self.workflow_end_time = time.time()
                execution_time = self.workflow_end_time - self.workflow_start_time
                
                context.metadata.update({
                    'execution_time_seconds': execution_time,
                    'completed_steps': len(self.completed_steps),
                    'failed_steps': len(self.failed_steps),
                    'success': False,
                    'error': str(e)
                })
                
                self.workflow_logger.error(
                    f"Workflow {self.name} execution failed",
                    execution_time_seconds=execution_time,
                    completed_steps=len(self.completed_steps),
                    failed_steps=len(self.failed_steps),
                    error=str(e)
                )
                
                raise
    
    async def _load_workflow_configuration(self) -> None:
        """Load step configurations from workflow configuration."""
        self.workflow_logger.debug("Loading workflow step configurations")
        
        # Workflow steps are already specified in the config
        # This method can be extended to load additional configuration
        pass
    
    async def _initialize_child_steps(self) -> None:
        """Initialize all child steps from configuration."""
        self.workflow_logger.info(f"Initializing {len(self.workflow_config.steps)} child steps")
        
        for step_config_dict in self.workflow_config.steps:
            step_id = step_config_dict.get('step_id')
            if not step_id:
                raise ValueError("Step configuration must include 'step_id'")
            
            # Load step configuration if config_file is specified
            if 'config_file' in step_config_dict:
                try:
                    step_config = self.config_loader.load_step_config(
                        step_config_dict['config_file'],
                        self.workflow_config.workflow_directory
                    )
                except FileNotFoundError as e:
                    self.workflow_logger.error(f"Failed to load step config for {step_id}: {e}")
                    raise
            else:
                # Use inline configuration
                config_data = step_config_dict.get('config', {})
                # Ensure name is set for StepConfig
                config_data['name'] = step_id
                config_data['description'] = step_config_dict.get('description', f"Step {step_id}")
                step_config = StepConfig(**config_data)
            
            # Create step instance
            step = await self._create_step_instance(step_id, step_config, step_config_dict)
            
            # Add to workflow
            self.child_steps[step_id] = step
            self.workflow_graph.add_step(step_id, step)
            
            # Initialize the step
            await step.initialize()
            
            self.workflow_logger.debug(f"Initialized child step: {step_id}")
    
    async def _create_step_instance(self, step_id: str, step_config: StepConfig, config_dict: Dict[str, Any]) -> Step:
        """Create a step instance from configuration."""
        # Import here to avoid circular imports
        from .step import create_step
        
        # Determine step type/class
        step_class = config_dict.get('class', config_dict.get('step_type', 'SimpleStep'))
        
        # Update step config with ID and name
        step_config.name = step_id
        
        # Create step instance
        try:
            step = create_step(step_class, step_config, executor=self.executor)
            return step
        except Exception as e:
            self.workflow_logger.error(f"Failed to create step {step_id}: {e}")
            raise
    
    async def _create_step_links(self) -> None:
        """Create links between steps from configuration."""
        self.workflow_logger.info(f"Creating {len(self.workflow_config.links)} step links")
        
        for link_config_dict in self.workflow_config.links:
            link_id = link_config_dict.get('link_id')
            source_id = link_config_dict.get('source')
            target_id = link_config_dict.get('target')
            link_type = link_config_dict.get('link_type', 'direct')
            
            if not all([link_id, source_id, target_id]):
                raise ValueError("Link configuration must include 'link_id', 'source', and 'target'")
            
            if source_id not in self.child_steps:
                raise ValueError(f"Link source step not found: {source_id}")
            
            if target_id not in self.child_steps:
                raise ValueError(f"Link target step not found: {target_id}")
            
            # Get source and target steps
            source_step = self.child_steps[source_id]
            target_step = self.child_steps[target_id]
            
            # Create link configuration
            link_config = LinkConfig(
                link_type=link_type,
                **{k: v for k, v in link_config_dict.items() 
                   if k not in ['link_id', 'source', 'target', 'link_type']}
            )
            
            # Create appropriate link instance based on type
            if link_type == 'conditional':
                # Handle conditional links with proper condition parsing
                from .link import ConditionalLink, parse_condition_from_config
                
                condition_config = link_config_dict.get('condition')
                if not condition_config:
                    raise ValueError(f"Conditional link {link_id} missing condition configuration")
                
                condition_func = parse_condition_from_config(condition_config)
                link = ConditionalLink(source_step, target_step, condition_func, link_config, name=link_id)
                
                self.workflow_logger.info(f"Created conditional link: {link_id} with condition: {condition_config}")
                
            elif link_type == 'transform':
                # Handle transform links
                from .link import TransformLink
                
                transform_function = link_config_dict.get('transform_function')
                if not transform_function:
                    raise ValueError(f"Transform link {link_id} missing transform_function")
                
                # For now, assume transform_function is a simple lambda or function name
                # This could be enhanced to support more complex transformations
                if isinstance(transform_function, str):
                    if transform_function.startswith('lambda'):
                        transform_func = eval(transform_function)
                    else:
                        # Could be enhanced to import and use named functions
                        transform_func = lambda x: x  # Identity function as fallback
                else:
                    transform_func = transform_function
                
                link = TransformLink(source_step, target_step, transform_func, link_config, name=link_id)
                
            else:
                # Default to DirectLink for 'direct' and any other types
                from .link import DirectLink
                link = DirectLink(source_step, target_step, link_config, name=link_id)
            
            # Add to workflow
            self.step_links[link_id] = link
            self.workflow_graph.add_link(link_id, link, source_id, target_id)
            
            # Initialize link
            await link.start()
            
            self.workflow_logger.debug(f"Created step link: {link_id} ({source_id} -> {target_id}) type: {link_type}")
    
    async def _build_workflow_graph(self) -> None:
        """Build the internal workflow graph representation."""
        # Graph is built incrementally in _initialize_child_steps and _create_step_links
        self.workflow_logger.debug("Workflow graph built successfully")
    
    async def _validate_workflow(self) -> None:
        """Validate the workflow graph structure."""
        if not self.workflow_config.validate_graph:
            self.workflow_logger.debug("Graph validation disabled")
            return
        
        is_valid, errors = self.workflow_graph.validate_graph(
            allow_cycles=self.workflow_config.allow_cycles,
            require_connected=self.workflow_config.require_connected_graph
        )
        
        if not is_valid:
            error_msg = f"Workflow graph validation failed:\n" + "\n".join(f"  - {error}" for error in errors)
            self.workflow_logger.error(error_msg)
            raise ValueError(error_msg)
        
        self.workflow_logger.info("Workflow graph validation passed")
    
    def _determine_execution_order(self) -> None:
        """Determine execution order based on execution strategy."""
        if self.workflow_config.execution_strategy == ExecutionStrategy.SEQUENTIAL:
            self.execution_order = self.workflow_graph.get_execution_order()
        elif self.workflow_config.execution_strategy == ExecutionStrategy.PARALLEL:
            # For parallel execution, we still need topological order for dependencies
            self.execution_order = self.workflow_graph.get_execution_order()
        elif self.workflow_config.execution_strategy == ExecutionStrategy.GRAPH_BASED:
            self.execution_order = self.workflow_graph.get_execution_order()
        elif self.workflow_config.execution_strategy == ExecutionStrategy.EVENT_DRIVEN:
            # Event-driven execution doesn't need predetermined order
            self.execution_order = list(self.child_steps.keys())
        
        self.workflow_logger.debug(f"Determined execution order: {self.execution_order}")
    
    async def _execute_workflow(self, input_data: Dict[str, Any], **kwargs) -> Any:
        """Execute workflow based on execution strategy."""
        strategy = self.workflow_config.execution_strategy
        
        if strategy == ExecutionStrategy.SEQUENTIAL:
            return await self._execute_sequential(input_data, **kwargs)
        elif strategy == ExecutionStrategy.PARALLEL:
            return await self._execute_parallel(input_data, **kwargs)
        elif strategy == ExecutionStrategy.GRAPH_BASED:
            return await self._execute_graph_based(input_data, **kwargs)
        elif strategy == ExecutionStrategy.EVENT_DRIVEN:
            return await self._execute_event_driven(input_data, **kwargs)
        else:
            raise ValueError(f"Unknown execution strategy: {strategy}")
    
    async def _execute_sequential(self, input_data: Dict[str, Any], **kwargs) -> Any:
        """Execute steps in sequential order."""
        self.workflow_logger.debug("Executing workflow sequentially")
        
        current_data = input_data
        final_result = None
        
        for step_id in self.execution_order:
            step = self.child_steps[step_id]
            
            try:
                step_start_time = time.time()
                
                self.workflow_logger.debug(f"Executing step: {step_id}")
                
                # Set input data for the step - first step gets workflow input, 
                # subsequent steps get output from previous step or workflow input for each step
                if hasattr(step, 'set_input'):
                    # Set the current data as input for this step
                    await step.set_input(current_data)
                
                # Execute step with kwargs only (input data is set via set_input)
                result = await step.execute(**kwargs)
                
                step_end_time = time.time()
                self.step_execution_times[step_id] = step_end_time - step_start_time
                
                self.completed_steps.add(step_id)
                final_result = result
                
                # Update current_data for next step - but preserve original workflow input
                if result is not None:
                    # Merge result with original input data for next step
                    if isinstance(result, dict) and isinstance(current_data, dict):
                        current_data = {**input_data, **result}  # Keep original input_data, add result
                    else:
                        current_data = result  # Use result as-is
                else:
                    # Keep original workflow input if no result
                    current_data = input_data
                
                # Propagate data through links
                await self._propagate_step_data(step_id, result)
                
                self.workflow_logger.debug(
                    f"Step {step_id} completed",
                    execution_time_seconds=self.step_execution_times[step_id]
                )
                
            except Exception as e:
                self.failed_steps.add(step_id)
                await self._handle_step_error(step_id, e)
        
        return final_result
    
    async def _execute_parallel(self, input_data: Dict[str, Any], **kwargs) -> Any:
        """Execute independent steps in parallel by execution level."""
        self.workflow_logger.debug("Executing workflow in parallel")
        
        execution_levels = self.workflow_graph.get_parallel_execution_levels()
        final_result = None
        
        for level_index, level_steps in enumerate(execution_levels):
            self.workflow_logger.debug(f"Executing level {level_index} with steps: {level_steps}")
            
            # Execute all steps in this level concurrently
            level_tasks = []
            for step_id in level_steps:
                step = self.child_steps[step_id]
                task = self._execute_step_with_tracking(step_id, step, input_data, **kwargs)
                level_tasks.append(task)
            
            # Wait for all steps in this level to complete
            if level_tasks:
                try:
                    results = await asyncio.gather(*level_tasks, return_exceptions=True)
                    
                    # Process results and handle exceptions
                    for step_id, result in zip(level_steps, results):
                        if isinstance(result, Exception):
                            self.failed_steps.add(step_id)
                            await self._handle_step_error(step_id, result)
                        else:
                            self.completed_steps.add(step_id)
                            final_result = result
                            await self._propagate_step_data(step_id, result)
                
                except Exception as e:
                    self.workflow_logger.error(f"Level {level_index} execution failed: {e}")
                    raise
        
        return final_result
    
    async def _execute_graph_based(self, input_data: Dict[str, Any], **kwargs) -> Any:
        """Execute based on data availability and dependencies."""
        self.workflow_logger.debug("Executing workflow using graph-based strategy")
        
        # This is a simplified graph-based execution
        # In a full implementation, this would use a more sophisticated
        # data-driven execution model with event queues
        
        return await self._execute_sequential(input_data, **kwargs)
    
    async def _execute_event_driven(self, input_data: Dict[str, Any], **kwargs) -> Any:
        """Execute using event-driven triggers."""
        self.workflow_logger.debug("Executing workflow using event-driven strategy")
        
        # This would integrate with the trigger system for true event-driven execution
        # For now, fall back to sequential execution
        
        return await self._execute_sequential(input_data, **kwargs)
    
    async def _execute_step_with_tracking(self, step_id: str, step: Step, input_data: Dict[str, Any], **kwargs) -> Any:
        """Execute a single step with performance tracking and progress reporting."""
        step_start_time = time.time()
        
        # Update progress: step started
        if self.progress_reporter:
            await self.progress_reporter.update_progress(
                step_id, 0, 'running', 
                message=f"Starting {step_id}"
            )
        
        try:
            # Set input data for the step
            if hasattr(step, 'set_input'):
                await step.set_input(input_data)
            
            # Execute step with progress updates
            if hasattr(step, 'execute_with_progress') and self.progress_reporter:
                # Step supports progress reporting
                result = await step.execute_with_progress(
                    progress_callback=lambda p, m=None: self._update_step_progress(step_id, p, message=m),
                    **kwargs
                )
            else:
                # Standard execution
                result = await step.execute(**kwargs)
            
            step_end_time = time.time()
            self.step_execution_times[step_id] = step_end_time - step_start_time
            
            # Update progress: step completed
            if self.progress_reporter:
                await self.progress_reporter.update_progress(
                    step_id, 100, 'completed',
                    message=f"Completed {step_id}"
                )
            
            self.workflow_logger.debug(
                f"Step {step_id} completed",
                execution_time_seconds=self.step_execution_times[step_id]
            )
            
            return result
            
        except Exception as e:
            step_end_time = time.time()
            self.step_execution_times[step_id] = step_end_time - step_start_time
            
            # Update progress: step failed
            if self.progress_reporter:
                technical_details = {
                    'error_type': type(e).__name__,
                    'execution_time': self.step_execution_times[step_id],
                    'stack_trace': str(e) if self.workflow_config.progress_show_technical_errors else None
                }
                await self.progress_reporter.update_progress(
                    step_id, 0, 'failed',
                    error=str(e),
                    technical_details=technical_details
                )
            
            self.workflow_logger.error(
                f"Step {step_id} failed",
                execution_time_seconds=self.step_execution_times[step_id],
                error=str(e)
            )
            
            raise
    
    async def _update_step_progress(self, step_id: str, progress: int, message: str = None) -> None:
        """Update step progress during execution."""
        if self.progress_reporter:
            await self.progress_reporter.update_progress(
                step_id, progress, message=message
            )
    
    async def _propagate_step_data(self, step_id: str, step_result: Any) -> None:
        """Propagate step result data through connected links."""
        # Find all links originating from this step
        outgoing_links = [
            link for link in self.step_links.values()
            if hasattr(link, 'source') and getattr(link.source, 'name', None) == step_id
        ]
        
        # Transfer data through each link
        for link in outgoing_links:
            try:
                await link.transfer(step_result)
            except Exception as e:
                self.workflow_logger.error(f"Data propagation failed for link {link.name}: {e}")
    
    async def _handle_step_error(self, step_id: str, error: Exception) -> None:
        """Handle step execution errors according to error handling strategy."""
        error_strategy = self.workflow_config.error_handling
        
        self.workflow_logger.error(f"Step {step_id} failed with error: {error}")
        
        if error_strategy == ErrorHandlingStrategy.STOP:
            raise error
        elif error_strategy == ErrorHandlingStrategy.CONTINUE:
            # Continue with next steps
            self.workflow_logger.warning(f"Continuing workflow despite step {step_id} failure")
        elif error_strategy == ErrorHandlingStrategy.RETRY:
            # Implement retry logic
            await self._retry_step(step_id, error)
        elif error_strategy == ErrorHandlingStrategy.ROLLBACK:
            # Implement rollback logic
            await self._rollback_workflow(step_id, error)
    
    async def _retry_step(self, step_id: str, original_error: Exception) -> None:
        """Retry failed step execution."""
        max_retries = self.workflow_config.retry_attempts
        retry_delay = self.workflow_config.retry_delay
        
        for attempt in range(max_retries):
            self.workflow_logger.info(f"Retrying step {step_id}, attempt {attempt + 1}/{max_retries}")
            
            try:
                await asyncio.sleep(retry_delay)
                step = self.child_steps[step_id]
                result = await step.execute()
                
                # Remove from failed steps if successful
                self.failed_steps.discard(step_id)
                self.completed_steps.add(step_id)
                
                await self._propagate_step_data(step_id, result)
                
                self.workflow_logger.info(f"Step {step_id} succeeded on retry attempt {attempt + 1}")
                return
                
            except Exception as e:
                self.workflow_logger.warning(f"Step {step_id} retry attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    # Final attempt failed
                    raise original_error
    
    async def _rollback_workflow(self, failed_step_id: str, error: Exception) -> None:
        """Rollback workflow state after step failure."""
        self.workflow_logger.warning(f"Rolling back workflow due to step {failed_step_id} failure")
        
        # This is a placeholder for rollback logic
        # In a full implementation, this would:
        # 1. Undo changes made by completed steps
        # 2. Reset data units to previous states
        # 3. Clean up resources
        
        raise error
    
    async def add_step(self, step_id: str, step_config: Union[Dict, StepConfig, str]) -> None:
        """Dynamically add step to workflow."""
        if step_id in self.child_steps:
            raise ValueError(f"Step {step_id} already exists in workflow")
        
        # Handle different step configuration types
        if isinstance(step_config, str):
            # Load from config file
            config = self.config_loader.load_step_config(step_config, self.workflow_config.workflow_directory)
        elif isinstance(step_config, dict):
            config = StepConfig(**step_config)
        else:
            config = step_config
        
        # Create and initialize step
        step = await self._create_step_instance(step_id, config, {'step_id': step_id})
        await step.initialize()
        
        # Add to workflow
        self.child_steps[step_id] = step
        self.workflow_graph.add_step(step_id, step)
        
        # Update execution order
        self._determine_execution_order()
        
        self.workflow_logger.info(f"Added step to workflow: {step_id}")
    
    async def remove_step(self, step_id: str) -> None:
        """Remove step and update graph."""
        if step_id not in self.child_steps:
            raise ValueError(f"Step {step_id} not found in workflow")
        
        # Shutdown the step
        step = self.child_steps[step_id]
        await step.shutdown()
        
        # Remove from workflow
        self.workflow_graph.remove_step(step_id)
        del self.child_steps[step_id]
        
        # Clean up any tracking data
        self.failed_steps.discard(step_id)
        self.completed_steps.discard(step_id)
        self.step_execution_times.pop(step_id, None)
        
        # Update execution order
        self._determine_execution_order()
        
        self.workflow_logger.info(f"Removed step from workflow: {step_id}")
    
    async def shutdown(self) -> None:
        """Shutdown the workflow and cleanup resources."""
        self.workflow_logger.info(f"Shutting down workflow: {self.name}")
        
        # Shutdown all child steps
        for step_id, step in self.child_steps.items():
            try:
                await step.shutdown()
            except Exception as e:
                self.workflow_logger.error(f"Error shutting down step {step_id}: {e}")
        
        # Stop all links
        for link_id, link in self.step_links.items():
            try:
                await link.stop()
            except Exception as e:
                self.workflow_logger.error(f"Error stopping link {link_id}: {e}")
        
        # Clear configuration cache
        self.config_loader.clear_cache()
        
        # Shutdown as Step
        await super().shutdown()
        
        self.workflow_logger.info(f"Workflow {self.name} shutdown complete")
    
    def get_workflow_stats(self) -> Dict[str, Any]:
        """Get comprehensive workflow statistics."""
        stats = {
            "workflow_name": self.name,
            "execution_strategy": self.workflow_config.execution_strategy.value,
            "num_steps": len(self.child_steps),
            "num_links": len(self.step_links),
            "completed_steps": len(self.completed_steps),
            "failed_steps": len(self.failed_steps),
            "is_complete": self.is_workflow_complete,
            "step_execution_times": self.step_execution_times.copy(),
            "graph_stats": self.workflow_graph.get_stats()
        }
        
        if self.workflow_start_time and self.workflow_end_time:
            stats["total_execution_time"] = self.workflow_end_time - self.workflow_start_time
        
        return stats
    
    def get_step(self, step_id: str) -> Optional[Step]:
        """Get a child step by ID."""
        return self.child_steps.get(step_id)
    
    def get_link(self, link_id: str) -> Optional[LinkBase]:
        """Get a link by ID."""
        return self.step_links.get(link_id)
    
    def list_steps(self) -> List[str]:
        """Get list of all step IDs."""
        return list(self.child_steps.keys())
    
    def list_links(self) -> List[str]:
        """Get list of all link IDs."""
        return list(self.step_links.keys())
    
    # Progress reporting methods
    def add_progress_callback(self, callback: Callable) -> None:
        """Add callback for progress updates."""
        if self.progress_reporter:
            self.progress_reporter.add_progress_callback(callback)
    
    def get_progress_summary(self) -> Optional[Dict[str, Any]]:
        """Get current progress summary."""
        if self.progress_reporter:
            return self.progress_reporter.get_progress_summary()
        return None
    
    def get_progress_history(self) -> List[Dict[str, Any]]:
        """Get progress history for session."""
        if self.progress_reporter:
            return self.progress_reporter.progress_history
        return []
    
    async def restore_from_checkpoint(self, step_id: str) -> Optional[Dict[str, Any]]:
        """Restore step from checkpoint."""
        if self.progress_reporter:
            return await self.progress_reporter.restore_from_checkpoint(step_id)
        return None


# Factory function for creating workflows
async def create_workflow(config: Union[WorkflowConfig, Dict[str, Any], str], **kwargs) -> Workflow:
    """
    Create and initialize a workflow.
    
    Args:
        config: Workflow configuration (WorkflowConfig, dict, or path to YAML file)
        **kwargs: Additional arguments passed to Workflow constructor
        
    Returns:
        Initialized Workflow instance
    """
    if isinstance(config, str):
        # Load from YAML file
        loader = ConfigLoader()
        workflow_config = loader.load_workflow_config(config)
    elif isinstance(config, dict):
        workflow_config = WorkflowConfig(**config)
    else:
        workflow_config = config
    
    workflow = Workflow(workflow_config, **kwargs)
    await workflow.initialize()
    
    return workflow 