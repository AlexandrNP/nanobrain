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

from .step import BaseStep, Step, StepConfig
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
    """
    Enhanced Configuration for workflows extending StepConfig - INHERITS constructor prohibition.
    
    ✅ FRAMEWORK COMPLIANCE:
    - Supports class+config patterns for steps, links, and triggers
    - ConfigBase._resolve_nested_objects() automatically instantiates components
    - Complete validation through ConfigBase schemas
    - Pure configuration-driven workflow creation
    
    ❌ FORBIDDEN: WorkflowConfig(name="test", steps=...)
    ✅ REQUIRED: WorkflowConfig.from_config('path/to/config.yml')
    """
    
    # Enhanced workflow configuration supporting class+config patterns
    steps: Dict[str, Any] = Field(
        default_factory=dict,
        description="Step definitions with class+config patterns or step_id configurations"
    )
    links: Dict[str, Any] = Field(
        default_factory=dict,
        description="Link definitions with class+config patterns"
    )
    triggers: Dict[str, Any] = Field(
        default_factory=dict,
        description="Trigger definitions with class+config patterns"
    )
    
    # Workflow execution configuration
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
    
    def add_step(self, step_id: str, step: BaseStep) -> None:
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
        
        # Store link with source/target IDs for validation
        self.edges[link_id] = {
            'link': link,
            'source_id': source_id,
            'target_id': target_id
        }
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
        for link_id, link_info in self.edges.items():
            if link_info['source_id'] == step_id or link_info['target_id'] == step_id:
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
        
        link_info = self.edges[link_id]
        
        # Get source and target step IDs from stored info
        source_id = link_info['source_id']
        target_id = link_info['target_id']
        
        if source_id and target_id:
            self.adjacency[source_id].discard(target_id)
            self.reverse_adjacency[target_id].discard(source_id)
        
        del self.edges[link_id]
        self._invalidate_cache()
        self.logger.debug(f"Removed link from workflow graph: {link_id}")
    
    def get_step(self, step_id: str) -> Optional[BaseStep]:
        """Get a step by ID."""
        return self.nodes.get(step_id)
    
    def get_link(self, link_id: str) -> Optional[LinkBase]:
        """Get a link by ID."""
        link_info = self.edges.get(link_id)
        return link_info['link'] if link_info else None
    
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
        for link_id, link_info in self.edges.items():
            source_id = link_info['source_id']
            target_id = link_info['target_id']
            
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
        return WorkflowConfig.from_config(config_data)
    
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
        return StepConfig.from_config(config_data)
    
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
    
    COMPONENT_TYPE = "workflow"
    REQUIRED_CONFIG_FIELDS = ['name']
    OPTIONAL_CONFIG_FIELDS = {
        'description': '',
        'steps': [],
        'links': [],
        'execution_strategy': 'sequential',
        'error_handling': 'continue',
        'enable_monitoring': True,
        'auto_initialize': True,
        'debug_mode': False
    }
    
    @classmethod
    def _get_config_class(cls):
        """UNIFIED PATTERN: Return WorkflowConfig - ONLY method that differs from other components"""
        return WorkflowConfig
    
    @classmethod
    def from_config(cls, config_path: Union[str, Path], **context) -> 'Workflow':
        """
        Enhanced workflow loading with automatic component instantiation
        
        ✅ FRAMEWORK COMPLIANCE:
        - Leverages ConfigBase._resolve_nested_objects() for automatic component instantiation
        - Steps, links, and triggers created via class+config patterns
        - No manual factory functions or redundant creation logic
        - Complete validation through ConfigBase schemas
        
        Args:
            config_path: Path to workflow configuration file
            **context: Additional context
            
        Returns:
            Fully initialized workflow instance
            
        Example Configuration:
        ```yaml
        name: "enhanced_workflow"
        description: "Workflow with automatic component instantiation"
        
        # Steps created via class+config patterns
        steps:
          data_acquisition:
            class: "nanobrain.library.steps.bv_brc_data_acquisition_step.BVBRCDataAcquisitionStep"
            config: "config/steps/BVBRCDataAcquisitionStep.yml"
          
          analysis:
            class: "nanobrain.library.steps.analysis_step.AnalysisStep"
            config:
              name: "protein_analysis"
              analysis_type: "protein_structure"
        
        # Links created via class+config patterns
        links:
          data_flow:
            class: "nanobrain.core.link.DirectLink"
            config: "config/links/DataFlowLink.yml"
        
        # Triggers created via class+config patterns  
        triggers:
          data_updated:
            class: "nanobrain.core.trigger.DataUpdatedTrigger"
            config:
              data_unit_name: "protein_data"
              threshold: 10
        ```
        
        ✅ FRAMEWORK COMPLIANCE:
        - ConfigBase._resolve_nested_objects() automatically instantiates all components
        - Components validated through their respective ConfigBase schemas
        - No manual component creation or factory dependencies
        - Complete configuration-driven workflow creation
        """
        from pathlib import Path
        
        # Use enhanced WorkflowConfig.from_config() method - automatically resolves class+config patterns
        workflow_config = WorkflowConfig.from_config(config_path, **context)
        
        # ConfigBase._resolve_nested_objects() has already instantiated all components
        # Extract resolved components from the configuration
        resolved_components = cls._extract_resolved_components(workflow_config)
        
        # Create workflow instance from resolved configuration
        workflow = cls._create_from_resolved_config(workflow_config, resolved_components, **context)
        
        return workflow
    
    @classmethod
    def _extract_resolved_components(cls, workflow_config: WorkflowConfig) -> Dict[str, Any]:
        """
        Extract instantiated components from resolved workflow configuration
        
        ConfigBase._resolve_nested_objects() has already instantiated all components
        specified with class+config patterns. This method extracts and validates them.
        
        Args:
            workflow_config: Resolved workflow configuration
            
        Returns:
            Dictionary containing categorized instantiated components
            
        ✅ FRAMEWORK COMPLIANCE:
        - Components already instantiated via ConfigBase._resolve_nested_objects()
        - No manual component creation or factory logic
        - Complete validation through ConfigBase schemas
        - Components immediately available for workflow use
        """
        resolved_components = {
            'steps': {},
            'links': {},
            'triggers': {},
            'data_units': {}
        }
        
        # Extract resolved steps
        steps_config = getattr(workflow_config, 'steps', {})
        for step_id, step_instance in steps_config.items():
            # Validate that it's a proper step instance
            if hasattr(step_instance, 'execute') or hasattr(step_instance, '__class__'):
                # Check if it's an instantiated object (not a dict)
                if not isinstance(step_instance, dict):
                    resolved_components['steps'][step_id] = step_instance
                    logger.debug(f"✅ Extracted resolved step: {step_id} ({step_instance.__class__.__name__})")
                else:
                    # If still a dict, it means it's a legacy configuration that needs manual handling
                    logger.warning(f"⚠️ Step '{step_id}' not resolved via class+config - requires legacy handling")
            else:
                logger.warning(f"⚠️ Skipping invalid step instance: {step_id} (missing execute method)")
        
        # Extract resolved links
        links_config = getattr(workflow_config, 'links', {})
        for link_id, link_instance in links_config.items():
            # Validate that it's a proper link instance
            if hasattr(link_instance, 'transfer') or hasattr(link_instance, '__class__'):
                # Check if it's an instantiated object (not a dict)
                if not isinstance(link_instance, dict):
                    resolved_components['links'][link_id] = link_instance
                    logger.debug(f"✅ Extracted resolved link: {link_id} ({link_instance.__class__.__name__})")
                else:
                    logger.warning(f"⚠️ Link '{link_id}' not resolved via class+config - requires legacy handling")
            else:
                logger.warning(f"⚠️ Skipping invalid link instance: {link_id} (missing transfer method)")
        
        # Extract resolved triggers
        triggers_config = getattr(workflow_config, 'triggers', {})
        for trigger_id, trigger_instance in triggers_config.items():
            # Validate that it's a proper trigger instance
            if hasattr(trigger_instance, 'start') or hasattr(trigger_instance, '__class__'):
                # Check if it's an instantiated object (not a dict)
                if not isinstance(trigger_instance, dict):
                    resolved_components['triggers'][trigger_id] = trigger_instance
                    logger.debug(f"✅ Extracted resolved trigger: {trigger_id} ({trigger_instance.__class__.__name__})")
                else:
                    logger.warning(f"⚠️ Trigger '{trigger_id}' not resolved via class+config - requires legacy handling")
            else:
                logger.warning(f"⚠️ Skipping invalid trigger instance: {trigger_id} (missing start method)")
        
        logger.info(f"✅ Extracted resolved components: {len(resolved_components['steps'])} steps, "
                   f"{len(resolved_components['links'])} links, {len(resolved_components['triggers'])} triggers")
        
        return resolved_components
    
    @classmethod
    def _create_from_resolved_config(cls, workflow_config: WorkflowConfig, resolved_components: Dict[str, Any], **context) -> 'Workflow':
        """
        Create workflow instance from resolved configuration and instantiated components
        
        This method assembles the workflow using components that have already been
        instantiated by ConfigBase._resolve_nested_objects().
        
        Args:
            workflow_config: Resolved workflow configuration
            resolved_components: Dictionary of instantiated components
            **context: Additional context
            
        Returns:
            Fully initialized workflow instance
            
        ✅ FRAMEWORK COMPLIANCE:
        - Uses pre-instantiated components from ConfigBase resolution
        - No manual component creation or factory dependencies
        - Complete configuration-driven workflow assembly
        - Validates component compatibility and integration
        """
        # Create executor if specified in context
        executor = context.get('executor')
        
        # Create workflow instance using the standard from_config_base pattern
        workflow = cls.from_config_base(workflow_config, executor=executor, **context)
        
        # Integrate resolved components into workflow
        workflow._integrate_resolved_components(resolved_components)
        
        # Store resolved components for workflow operation
        workflow._resolved_components = resolved_components
        
        # Validate integrated components
        workflow._validate_integrated_components(resolved_components)
        
        logger.info(f"✅ Created workflow from resolved config: {workflow_config.name}")
        
        return workflow
    
    def _integrate_resolved_components(self, resolved_components: Dict[str, Any]) -> None:
        """
        Integrate resolved components into workflow structure
        
        Replaces the manual component creation with pre-instantiated components
        from ConfigBase._resolve_nested_objects().
        
        Args:
            resolved_components: Dictionary of instantiated components
            
        ✅ FRAMEWORK COMPLIANCE:
        - Uses pre-instantiated components exclusively
        - No manual component creation or factory logic
        - Components already validated through ConfigBase schemas
        - Immediate availability for workflow execution
        """
        # Integrate resolved steps
        for step_id, step_instance in resolved_components['steps'].items():
            self.child_steps[step_id] = step_instance
            self.workflow_graph.add_step(step_id, step_instance)
            
            # Set step integration properties
            if hasattr(step_instance, 'step_id'):
                step_instance.step_id = step_id
            if hasattr(step_instance, 'executor') and not step_instance.executor:
                step_instance.executor = self.executor
            
            logger.debug(f"✅ Integrated step: {step_id}")
        
        # Integrate resolved links
        for link_id, link_instance in resolved_components['links'].items():
            self.step_links[link_id] = link_instance
            
            # Add link to workflow graph if it has source/target information
            if hasattr(link_instance, 'source') and hasattr(link_instance, 'target'):
                source_id = getattr(link_instance.source, 'step_id', None) or getattr(link_instance.source, 'name', None)
                target_id = getattr(link_instance.target, 'step_id', None) or getattr(link_instance.target, 'name', None)
                
                if source_id and target_id:
                    self.workflow_graph.add_link(link_id, link_instance, source_id, target_id)
                    logger.debug(f"✅ Integrated link: {link_id} ({source_id} -> {target_id})")
                else:
                    logger.warning(f"⚠️ Link {link_id} missing source/target identification")
            else:
                logger.warning(f"⚠️ Link {link_id} missing source/target properties")
        
        # Integrate resolved triggers (stored for later activation)
        self._workflow_triggers = resolved_components['triggers']
        for trigger_id, trigger_instance in resolved_components['triggers'].items():
            logger.debug(f"✅ Integrated trigger: {trigger_id}")
        
        logger.info(f"✅ Integrated all resolved components into workflow")
    
    def _validate_integrated_components(self, resolved_components: Dict[str, Any]) -> None:
        """
        Validate that integrated components are compatible and properly configured
        
        Args:
            resolved_components: Dictionary of instantiated components
            
        Raises:
            ValueError: If component integration validation fails
        """
        # Validate steps
        for step_id, step_instance in resolved_components['steps'].items():
            if not hasattr(step_instance, 'execute'):
                raise ValueError(f"❌ Invalid step: {step_id} missing execute method")
            
            # Validate step has required configuration
            if not hasattr(step_instance, 'config') or not hasattr(step_instance, 'name'):
                logger.warning(f"⚠️ Step {step_id} missing standard configuration attributes")
        
        # Validate links reference existing steps
        for link_id, link_instance in resolved_components['links'].items():
            if hasattr(link_instance, 'source') and hasattr(link_instance, 'target'):
                source_step = link_instance.source
                target_step = link_instance.target
                
                # Check if source and target steps exist in workflow
                source_found = any(step == source_step for step in self.child_steps.values())
                target_found = any(step == target_step for step in self.child_steps.values())
                
                if not source_found:
                    logger.warning(f"⚠️ Link {link_id} source step not found in workflow steps")
                if not target_found:
                    logger.warning(f"⚠️ Link {link_id} target step not found in workflow steps")
        
        # Validate triggers have required properties
        for trigger_id, trigger_instance in resolved_components['triggers'].items():
            if not hasattr(trigger_instance, 'start'):
                raise ValueError(f"❌ Invalid trigger: {trigger_id} missing start method")
        
        logger.info("✅ All integrated components validated successfully")
    
    @classmethod
    def extract_component_config(cls, config: WorkflowConfig) -> Dict[str, Any]:
        """Extract Workflow configuration"""
        base_config = super().extract_component_config(config)
        return {
            **base_config,
            'steps': getattr(config, 'steps', []),
            'links': getattr(config, 'links', []),
            'execution_strategy': getattr(config, 'execution_strategy', ExecutionStrategy.SEQUENTIAL),
            'error_handling': getattr(config, 'error_handling', ErrorHandlingStrategy.CONTINUE),
            'enable_monitoring': getattr(config, 'enable_monitoring', True),
            'workflow_directory': getattr(config, 'workflow_directory', None),
            'max_parallel_steps': getattr(config, 'max_parallel_steps', 10),
            'step_timeout': getattr(config, 'step_timeout', 300.0),
            'retry_attempts': getattr(config, 'retry_attempts', 3),
            'retry_delay': getattr(config, 'retry_delay', 1.0),
            'validate_graph': getattr(config, 'validate_graph', True),
            'allow_cycles': getattr(config, 'allow_cycles', False),
            'require_connected_graph': getattr(config, 'require_connected_graph', True),
            'enable_progress_reporting': getattr(config, 'enable_progress_reporting', True),
            'progress_batch_interval': getattr(config, 'progress_batch_interval', 3.0),
            'progress_collapsed_by_default': getattr(config, 'progress_collapsed_by_default', True),
            'progress_show_technical_errors': getattr(config, 'progress_show_technical_errors', True),
            'progress_preserve_session_history': getattr(config, 'progress_preserve_session_history', True)
        }
    
    def _init_from_config(self, config: WorkflowConfig, component_config: Dict[str, Any],
                         dependencies: Dict[str, Any]) -> None:
        """Initialize Workflow with resolved dependencies"""
        super()._init_from_config(config, component_config, dependencies)
        
        # Workflow-specific configuration
        self.workflow_config = config
        
        # Core workflow components
        self.workflow_graph = WorkflowGraph()
        
        # Resolve workflow directory properly
        workflow_dir = component_config.get('workflow_directory') or "."
        if not Path(workflow_dir).is_absolute():
            # If relative path, search for it in common locations
            possible_paths = [
                Path(workflow_dir),  # Relative to current directory
                Path.cwd() / workflow_dir,  # Relative to current working directory
                Path(__file__).parent.parent / workflow_dir,  # Relative to nanobrain root
                Path(__file__).parent.parent.parent / workflow_dir,  # One level up from nanobrain/core/
            ]
            
            for possible_path in possible_paths:
                if possible_path.exists():
                    workflow_dir = str(possible_path.resolve())  # Use absolute path
                    break
            else:
                # If none found, try to find nanobrain package root more systematically
                current = Path(__file__).parent  # nanobrain/core/
                while current.parent != current:  # Go up until filesystem root
                    candidate = current / workflow_dir
                    if candidate.exists():
                        workflow_dir = str(candidate.resolve())
                        break
                    current = current.parent
        
        self.config_loader = ConfigLoader(workflow_dir)
        
        # Step and link management
        self.child_steps: Dict[str, BaseStep] = {}
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
        if component_config.get('enable_progress_reporting', True):
            session_id = dependencies.get('session_id')
            self.progress_reporter = ProgressReporter(
                workflow_id=f"{self.name}_{int(time.time())}",
                workflow_name=self.name,
                session_id=session_id
            )
            self.progress_reporter.workflow_progress.batch_interval = component_config.get('progress_batch_interval', 3.0)
            self.progress_reporter.workflow_progress.collapsed_by_default = component_config.get('progress_collapsed_by_default', True)
            self.progress_reporter.workflow_progress.show_technical_errors = component_config.get('progress_show_technical_errors', True)
            self.progress_reporter.workflow_progress.preserve_session_history = component_config.get('progress_preserve_session_history', True)
        
        # Workflow-specific logger
        self.workflow_logger = get_logger(f"workflow.{self.name}", debug_mode=component_config.get('debug_mode', False))
        
        self.workflow_logger.info(f"Initialized workflow: {self.name}")
    
    # Workflow inherits FromConfigBase.__init__ which prevents direct instantiation
    # Use Workflow.from_config() to create instances
    
    def _legacy_init_workflow_components(self, config: WorkflowConfig, **kwargs):
        """Legacy initialization method - kept for reference but should use _init_from_config"""
        
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
        """
        Initialize child steps from resolved components
        
        Steps are already instantiated by ConfigBase._resolve_nested_objects()
        and integrated into the workflow via _integrate_resolved_components().
        This method initializes the pre-instantiated steps.
        
        ✅ FRAMEWORK COMPLIANCE:
        - Uses pre-instantiated steps from ConfigBase resolution
        - No manual step creation or factory logic
        - Steps already validated through ConfigBase schemas
        - Immediate availability for workflow execution
        """
        if not hasattr(self, '_resolved_components'):
            self.workflow_logger.warning("⚠️ No resolved components found - workflow may not be fully configured")
            return
        
        resolved_steps = self._resolved_components.get('steps', {})
        self.workflow_logger.info(f"Initializing {len(resolved_steps)} pre-instantiated child steps")
        
        # Initialize each resolved step
        for step_id, step_instance in resolved_steps.items():
            try:
                # Initialize the step if not already initialized
                if hasattr(step_instance, 'initialize') and hasattr(step_instance, '_is_initialized'):
                    if not step_instance._is_initialized:
                        await step_instance.initialize()
                        self.workflow_logger.debug(f"✅ Initialized resolved step: {step_id}")
                    else:
                        self.workflow_logger.debug(f"✅ Step already initialized: {step_id}")
                elif hasattr(step_instance, 'initialize'):
                    # Initialize even if _is_initialized attribute is not present
                    await step_instance.initialize()
                    self.workflow_logger.debug(f"✅ Initialized resolved step: {step_id}")
                else:
                    self.workflow_logger.debug(f"✅ Step does not require initialization: {step_id}")
                
                # Ensure step has required workflow integration properties
                if not hasattr(step_instance, 'step_id'):
                    step_instance.step_id = step_id
                
                # Set executor if not already set
                if hasattr(step_instance, 'executor') and not step_instance.executor:
                    step_instance.executor = self.executor
                
                # Set workflow directory context
                if hasattr(step_instance, 'workflow_directory') and not step_instance.workflow_directory:
                    step_instance.workflow_directory = self.workflow_config.workflow_directory
                
            except Exception as e:
                self.workflow_logger.error(f"❌ Failed to initialize resolved step {step_id}: {e}")
                raise ValueError(f"Step initialization failed: {step_id} - {str(e)}") from e
        
        self.workflow_logger.info(f"✅ Initialized {len(resolved_steps)} resolved child steps")
    

    
    async def _create_step_links(self) -> None:
        """
        Initialize step links from resolved components
        
        Links are already instantiated by ConfigBase._resolve_nested_objects()
        and integrated into the workflow via _integrate_resolved_components().
        This method starts the pre-instantiated links.
        
        ✅ FRAMEWORK COMPLIANCE:
        - Uses pre-instantiated links from ConfigBase resolution
        - No manual link creation or factory logic
        - Links already validated through ConfigBase schemas
        - Immediate availability for workflow execution
        """
        if not hasattr(self, '_resolved_components'):
            self.workflow_logger.warning("⚠️ No resolved components found - workflow may not be fully configured")
            return
        
        resolved_links = self._resolved_components.get('links', {})
        self.workflow_logger.info(f"Initializing {len(resolved_links)} pre-instantiated step links")
        
        # Initialize each resolved link
        for link_id, link_instance in resolved_links.items():
            try:
                # Start the link if it has a start method
                if hasattr(link_instance, 'start'):
                    await link_instance.start()
                    self.workflow_logger.debug(f"✅ Started resolved link: {link_id}")
                else:
                    self.workflow_logger.debug(f"✅ Link does not require starting: {link_id}")
                
                # Ensure link has required workflow integration properties
                if not hasattr(link_instance, 'name') or not link_instance.name:
                    if hasattr(link_instance, 'name'):
                        link_instance.name = link_id
                
                # Validate link has source and target
                if not (hasattr(link_instance, 'source') and hasattr(link_instance, 'target')):
                    self.workflow_logger.warning(f"⚠️ Link {link_id} missing source/target properties")
                
            except Exception as e:
                self.workflow_logger.error(f"❌ Failed to initialize resolved link {link_id}: {e}")
                raise ValueError(f"Link initialization failed: {link_id} - {str(e)}") from e
        
        self.workflow_logger.info(f"✅ Initialized {len(resolved_links)} resolved step links")
    
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
            config = StepConfig.from_config(step_config)
        else:
            config = step_config
        
        # Create and initialize step
        step = await self._create_step_instance({'step_id': step_id, 'config_file': step_config})
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
        workflow_config = WorkflowConfig.from_config(config)
    else:
        workflow_config = config
    
    workflow = Workflow(workflow_config, **kwargs)
    await workflow.initialize()
    
    return workflow 