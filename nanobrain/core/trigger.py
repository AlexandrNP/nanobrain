"""
Trigger System for NanoBrain Framework

Provides event-driven processing capabilities for Steps.
Enhanced with mandatory from_config pattern implementation.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from contextvars import ContextVar
from typing import Any, Dict, Literal, Optional, List, Callable, Set, Union
from enum import Enum
from pydantic import Field
from pathlib import Path

from .component_base import FromConfigBase, ComponentConfigurationError
# Import logging system
from .logging_system import get_logger, get_system_log_manager
# Import new ConfigBase for constructor prohibition
from .config.config_base import ConfigBase
# Import event types for proper enum-based event handling
from .event_types import DataUnitEventType, validate_event_type, DEFAULT_DATA_UNIT_EVENT

logger = logging.getLogger(__name__)


# G115 — workflow-scoped task tagging for nested Workflow.run() (2026-05-18)
#
# AsyncTriggerExecutor is a process-singleton with a shared
# background_tasks set. Before G115, nested ``await inner_workflow.run(...)``
# from inside an outer workflow's step.process() deadlocked: the inner
# wait_for_cascade saw the outer's still-awaiting task in the same set
# and never observed drain. Each level cascade-timed-out at 60s.
#
# Fix: every Workflow.run() pushes its workflow_id into this ContextVar
# for the duration of the run. Tasks created during the run inherit the
# value via the asyncio contextvar propagation. wait_for_all_tasks
# accepts an optional workflow_id kwarg and filters background_tasks
# by the ``_nb_workflow_id`` attribute set on each task.
#
# Backward compatibility: legacy callers that don't pass workflow_id
# fall back to the original "wait for ALL tasks" behavior. Tasks not
# tagged (e.g., from tests that create tasks directly) are matched by
# the legacy path, never filtered out.
_active_workflow_id: ContextVar[Optional[str]] = ContextVar(
    "nb_active_workflow_id", default=None
)


def _current_workflow_id() -> Optional[str]:
    """Return the workflow_id of the currently active Workflow.run scope,
    or None if no run is active. Internal helper for G115."""
    return _active_workflow_id.get()


def _tag_task_with_workflow(task: "asyncio.Task") -> None:
    """Stamp the currently-active workflow_id (if any) onto an asyncio
    Task object. Called at task-creation sites so wait_for_all_tasks
    can filter by scope. Idempotent: re-tagging with the same value
    is a no-op."""
    wf_id = _active_workflow_id.get()
    if wf_id is not None:
        task._nb_workflow_id = wf_id  # type: ignore[attr-defined]


class AsyncTriggerExecutor:
    """
    Async Trigger Executor - Non-blocking trigger execution system.

    This class implements the async trigger execution approach to eliminate
    deadlocks while preserving the data-driven architecture of NanoBrain.

    Key Features:
    - Non-blocking trigger execution using asyncio.create_task()
    - Background task management with proper cleanup
    - Maintains data-driven architecture completely
    - Eliminates circular trigger deadlocks
    """

    _instance = None
    _lock = asyncio.Lock()

    def __init__(self):
        self.background_tasks: Set[asyncio.Task] = set()
        self.logger = get_logger("async_trigger_executor")
        self.execution_count = 0
        self.error_count = 0
        # ✅ DEADLOCK FIX: Add circular dependency tracking
        self.execution_stack = set()  # Track currently executing triggers
        self.execution_history = {}   # Track recent executions to prevent rapid re-triggering

    @classmethod
    async def get_instance(cls) -> 'AsyncTriggerExecutor':
        """Get singleton instance of AsyncTriggerExecutor."""
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    async def execute_trigger_async(self, trigger: 'TriggerBase', data: Any = None) -> None:
        """Execute trigger asynchronously without blocking the caller."""
        try:
            # ✅ DEADLOCK FIX: Check for circular dependencies
            trigger_id = f"{trigger.name}_{id(trigger)}"

            # Prevent immediate re-execution of the same trigger
            if trigger_id in self.execution_stack:
                self.logger.warning(f"Preventing circular execution of trigger: {trigger.name}")
                return

            # Check for rapid re-triggering (within 100ms)
            import time
            current_time = time.time()
            if trigger_id in self.execution_history:
                last_execution = self.execution_history[trigger_id]
                if current_time - last_execution < 0.1:  # 100ms threshold
                    self.logger.debug(f"Throttling rapid re-execution of trigger: {trigger.name}")
                    return

            # Update execution tracking
            self.execution_history[trigger_id] = current_time

            # Create async task for trigger execution
            task = asyncio.create_task(
                self._run_trigger_in_background(trigger, data, trigger_id)
            )

            # G115 — stamp the active workflow_id (if any) before
            # adding to the shared set. wait_for_all_tasks(workflow_id=...)
            # uses this tag to filter nested-workflow cascade drains.
            _tag_task_with_workflow(task)

            # Add to background tasks for tracking
            self.background_tasks.add(task)

            # Remove task when done (prevents memory leaks)
            task.add_done_callback(self.background_tasks.discard)

            self.execution_count += 1

            # Log async trigger execution
            if hasattr(trigger, 'nb_logger') and trigger.nb_logger:
                trigger.nb_logger.debug(
                    f"🚀 Async trigger execution initiated: {trigger.name}",
                    operation="async_trigger_start",
                    execution_count=self.execution_count,
                    background_tasks=len(self.background_tasks)
                )

        except Exception as e:
            self.error_count += 1
            self.logger.error(
                f"Failed to initiate async trigger execution: {e}")

            if hasattr(trigger, 'nb_logger') and trigger.nb_logger:
                trigger.nb_logger.error(
                    f"❌ Async trigger execution failed: {trigger.name}",
                    operation="async_trigger_error",
                    error=str(e),
                    error_count=self.error_count
                )

    async def _run_trigger_in_background(self, trigger: 'TriggerBase', data: Any, trigger_id: str) -> None:
        """Run trigger in separate async context to prevent blocking."""
        try:
            # ✅ DEADLOCK FIX: Track execution to prevent circular dependencies
            self.execution_stack.add(trigger_id)

            # Yield control to allow caller to continue
            await asyncio.sleep(0)

            # Execute the trigger's original logic
            await trigger._execute_callbacks(data)

            # Log successful completion
            if hasattr(trigger, 'nb_logger') and trigger.nb_logger:
                trigger.nb_logger.debug(
                    f"✅ Async trigger execution completed: {trigger.name}",
                    operation="async_trigger_complete"
                )

        except Exception as e:
            self.error_count += 1
            self.logger.error(
                f"Background trigger execution failed for {trigger.name}: {e}")

            if hasattr(trigger, 'nb_logger') and trigger.nb_logger:
                trigger.nb_logger.error(
                    f"❌ Background trigger execution error: {trigger.name}",
                    operation="async_trigger_background_error",
                    error=str(e)
                )
        finally:
            # ✅ DEADLOCK FIX: Always remove from execution stack to prevent permanent blocking
            self.execution_stack.discard(trigger_id)

    async def wait_for_all_tasks(
        self,
        timeout: float = 30.0,
        settle_ms: int = 50,
        *,
        workflow_id: Optional[str] = None,
    ) -> bool:
        """Wait for all background tasks to complete.

        Handles **cascading** task creation: when an in-flight task
        spawns another task (e.g. a DataUnitChangeTrigger fires step A,
        which writes to a DataUnit that triggers step B, which spawns
        a fresh background task), this method keeps draining until the
        ``background_tasks`` set stays empty for ``settle_ms``.

        This is the entry point that test code (and graceful shutdown
        paths) use to await the full trigger cascade synchronously
        after a data-driven ``Workflow.process(input)`` call.

        Args:
            timeout: Total wall-clock budget. Returns ``False`` if the
                cascade hasn't drained within this many seconds.
            settle_ms: Quiet-period in milliseconds. After
                ``background_tasks`` is observed empty, wait this long
                and re-check; only return ``True`` if it's still empty.
                Catches the case where one trigger's done callback
                spawns another trigger's task asynchronously.
            workflow_id: G115 (2026-05-18). When set, only consider
                tasks tagged with this workflow_id (via the
                ``_active_workflow_id`` ContextVar at task-creation
                time). Untagged tasks and tasks belonging to a
                DIFFERENT workflow are EXCLUDED from this scope's
                drain — they are some other caller's responsibility.
                When None (legacy default), every task in the set is
                considered (preserves pre-G115 behavior for callers
                that don't pass workflow_id). Enables nested
                ``Workflow.run()`` calls to drain only their own
                cascade without deadlocking on the outer's still-
                awaiting task.

        Returns:
            ``True`` if the cascade fully drained; ``False`` on timeout.
        """
        import time as _time

        def _scoped(tasks: Set[asyncio.Task]) -> List[asyncio.Task]:
            """Return the subset of tasks matching the scope. When
            workflow_id is None, every task matches. When set, a task
            matches if its ``_nb_workflow_id`` tag equals workflow_id.
            Untagged tasks are treated as foreign-to-the-scope and
            EXCLUDED — they belong to some other workflow's run (or
            to a non-workflow caller) and are not this scope's
            responsibility to drain."""
            if workflow_id is None:
                return list(tasks)
            return [
                t for t in tasks
                if getattr(t, "_nb_workflow_id", None) == workflow_id
            ]

        deadline = _time.monotonic() + timeout

        while True:
            remaining = deadline - _time.monotonic()
            if remaining <= 0:
                scoped = _scoped(self.background_tasks)
                if scoped:
                    self.logger.warning(
                        f"Timeout waiting for "
                        f"{len(scoped)} background tasks "
                        f"(workflow_id={workflow_id!r})"
                    )
                    return False
                return True

            scoped = _scoped(self.background_tasks)
            if not scoped:
                # Quiet — but a task may be about to fire from a
                # done-callback chain. Sleep briefly and re-check;
                # only return success if the set stays empty.
                await asyncio.sleep(min(settle_ms / 1000.0, remaining))
                scoped = _scoped(self.background_tasks)
                if not scoped:
                    return True
                continue

            # Snapshot the scoped tasks and await them. Tasks added
            # AFTER this snapshot (in OR out of scope) are picked up
            # on the next iteration.
            snapshot = scoped
            try:
                await asyncio.wait_for(
                    asyncio.gather(*snapshot, return_exceptions=True),
                    timeout=remaining,
                )
            except asyncio.TimeoutError:
                still_scoped = _scoped(self.background_tasks)
                self.logger.warning(
                    f"Timeout waiting for "
                    f"{len(still_scoped)} background tasks "
                    f"(workflow_id={workflow_id!r}, cascading drain)"
                )
                return False

    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return {
            "execution_count": self.execution_count,
            "error_count": self.error_count,
            "active_background_tasks": len(self.background_tasks),
            "success_rate": (self.execution_count - self.error_count) / max(self.execution_count, 1)
        }


class TriggerType(Enum):
    """Types of triggers."""
    DATA_UPDATED = "data_updated"
    ALL_DATA_RECEIVED = "all_data_received"
    TIMER = "timer"
    MANUAL = "manual"
    CONDITION = "condition"
    EVENT = "event"  # G22 — externally-fired event trigger


class TriggerConfig(ConfigBase):
    """
    Configuration for triggers - INHERITS constructor prohibition.

    ❌ FORBIDDEN: TriggerConfig(trigger_type="data_updated", ...)
    ✅ REQUIRED: TriggerConfig.from_config('path/to/config.yml')
    """

    trigger_type: TriggerType = TriggerType.DATA_UPDATED
    debounce_ms: int = Field(default=100, ge=0)
    max_frequency_hz: float = Field(default=10.0, gt=0)
    condition: Optional[str] = None
    timer_interval_ms: Optional[int] = None
    name: str = ""

    # G2 — dynamic expected-set narrowing for AllDataReceivedTrigger.
    # See `apecx-mcp-integration/docs/CONTRACTS.md#g2`.
    # All fields default to None; a trigger that doesn't set them uses the
    # historical static-list semantics. When set, the trigger reads
    # `expected_set_source` (a workflow-level data unit reference of the form
    # "workflow.<unit_name>") on first activation, projects
    # `expected_set_field` against its payload, formats each result through
    # `expected_set_naming`, and intersects with the static `inputs` list to
    # narrow to the active-this-run subset. Eliminates the "publish empty
    # bundle so the trigger fires" workaround in gated layer steps.
    expected_set_source: Optional[str] = Field(
        default=None,
        description="Workflow-level data unit reference of the form "
                    "'workflow.<unit_name>' that carries the active set."
    )
    expected_set_field: Optional[str] = Field(
        default=None,
        description="Dotted field path to project from the source data unit's "
                    "payload (e.g. 'active_layers'). Projection MUST be a "
                    "JSON array of strings."
    )
    expected_set_naming: str = Field(
        default="{value}",
        description="str.format template wrapping each projected string into "
                    "the canonical data-unit name. Default is the identity "
                    "template; the orchestrator-typical pattern is "
                    "'{value}_layer.layer_result_output'."
    )

    # G10 — gate-to-bottom semantics for AllDataReceivedTrigger. See
    # `apecx-mcp-integration/docs/CONTRACTS.md#g10`.
    # When 'publish_empty' (default; legacy), a payload of None blocks
    # firing — the trigger waits indefinitely. When 'gate_to_bottom',
    # the trigger ALSO recognizes the magic string
    # ConditionalLink.GATED_OFF_SENTINEL as a "satisfied but absent"
    # marker: the unit is counted as resolved AND excluded from the
    # outgoing trigger payload, so the downstream step's process()
    # never sees the magic string.
    gate_semantics: str = Field(
        default="publish_empty",
        description="AllDataReceivedTrigger gating semantics. "
                    "'publish_empty' (default; legacy) only counts non-None "
                    "payloads as satisfied. 'gate_to_bottom' additionally "
                    "treats ConditionalLink.GATED_OFF_SENTINEL as satisfied "
                    "and excludes it from the trigger payload."
    )

    # G22 — EventTrigger filter. Optional G1 PredicateConfig dict (or
    # legacy condition shape) that gates whether an incoming event
    # actually fires the trigger. When None, every fire_event() call
    # fires; when set, the predicate is evaluated against the event
    # body and fire_event() returns silently on miss.
    event_filter: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional G1-style predicate dict applied to incoming "
                    "event bodies in EventTrigger.fire_event(). When set, "
                    "fire_event silently returns when the predicate is "
                    "False; when None, every event fires the trigger."
    )

    # G22 Step 3 — missed-schedule policy. Applies to triggers with a
    # cadence (currently TimerTrigger). When the framework detects that
    # one or more scheduled fires were missed (e.g., the process was
    # restarted across a fire boundary), it consults this policy:
    #   - 'skip' (default; legacy): forget missed fires; only fire on
    #     the next regularly-scheduled tick.
    #   - 'catch_up': fire N times in rapid succession, where N is the
    #     count of missed intervals.
    #   - 'merge': fire ONCE regardless of how many were missed.
    # EventTrigger / DataUnitChangeTrigger / ManualTrigger ignore this
    # field (no cadence to miss against).
    on_missed: Literal["skip", "catch_up", "merge"] = Field(
        default="skip",
        description="G22 Step 3 missed-schedule policy for cadenced "
                    "triggers (TimerTrigger). Determines what happens "
                    "when the framework detects N missed fires after a "
                    "process restart. 'skip' (default; legacy) forgets "
                    "them; 'catch_up' fires N times rapidly; 'merge' "
                    "fires once. Non-cadenced triggers ignore this "
                    "field."
    )


class TriggerBase(FromConfigBase, ABC):
    """
    Base Trigger Class - Event-Driven Activation and Workflow Orchestration
    =======================================================================

    The TriggerBase class is the foundational component for event-driven processing
    within the NanoBrain framework. Triggers monitor conditions, data changes, and
    external events to automatically activate steps, workflows, and agents, enabling
    reactive and responsive processing architectures.

    **Core Architecture:**
        Triggers represent intelligent event detection systems that:

        * **Monitor Conditions**: Continuously watch for specific events or state changes
        * **Activate Components**: Automatically trigger step and workflow execution
        * **Manage Timing**: Control execution frequency with debouncing and rate limiting
        * **Enable Reactivity**: Support real-time response to data and system events
        * **Coordinate Workflows**: Orchestrate complex multi-step processing pipelines
        * **Ensure Reliability**: Provide robust event detection with error handling

    **Biological Analogy:**
        Like action potential threshold mechanisms that fire when specific conditions
        are met, triggers activate steps when particular events or conditions are
        satisfied. Neurons accumulate electrical potential and fire when threshold
        is reached, propagating signals through neural networks - exactly how triggers
        monitor conditions and activate processing components when criteria are met,
        propagating execution through workflow networks.

    **Event-Driven Processing Architecture:**

        **Condition Monitoring:**
        * Continuous monitoring of data units and system state
        * Real-time change detection with configurable sensitivity
        * Multi-condition evaluation with logical operators
        * Custom condition scripting and evaluation

        **Activation Patterns:**
        * Immediate activation upon condition detection
        * Debounced activation to prevent excessive triggering
        * Rate-limited activation for performance optimization
        * Scheduled activation with timer-based triggers

        **Event Types:**
        * **Data Updated**: Triggered when data units receive new data
        * **All Data Received**: Triggered when all required inputs are available
        * **Timer**: Triggered on scheduled intervals or specific times
        * **Manual**: Triggered by explicit user or system commands
        * **Condition**: Triggered when custom conditions evaluate to true

        **Response Coordination:**
        * Multi-target activation for parallel processing
        * Sequential activation with dependency management
        * Conditional activation based on runtime state
        * Priority-based activation ordering

    **Framework Integration:**
        Triggers seamlessly integrate with all framework components:

        * **Step Activation**: Automatically trigger step execution when conditions are met
        * **Workflow Orchestration**: Coordinate complex multi-step processing workflows
        * **Agent Integration**: Trigger agent processing based on data availability
        * **Data Unit Monitoring**: Monitor data unit changes and trigger processing
        * **Executor Support**: Triggers work with all execution backends
        * **Monitoring Integration**: Comprehensive logging and performance tracking

    **Trigger Type Implementations:**
        The framework supports various trigger specializations:

        * **DataUpdatedTrigger**: Monitors data unit changes and modifications
        * **AllDataReceivedTrigger**: Waits for all required inputs before activation
        * **TimerTrigger**: Provides scheduled and interval-based activation
        * **ManualTrigger**: Enables user-controlled activation and testing
        * **ConditionalTrigger**: Supports custom condition evaluation and scripting
        * **CompoundTrigger**: Combines multiple triggers with logical operators

    **Configuration Architecture:**
        Triggers follow the framework's configuration-first design:

        ```yaml
        # Data update trigger
        name: "data_change_trigger"
        trigger_type: "data_updated"
        debounce_ms: 500
        max_frequency_hz: 5.0

        # Watch specific data units
        watch_data_units:
          - "input_data"
          - "parameters"

        # Target steps to activate
        target_steps:
          - "data_processing"
          - "validation"

        # Timer trigger
        name: "scheduled_processing"
        trigger_type: "timer"
        timer_interval_ms: 60000  # Every minute

        # Schedule configuration
        schedule:
          type: "interval"
          interval: "1m"
          start_time: "09:00"
          end_time: "17:00"
          timezone: "UTC"

        # Conditional trigger
        name: "threshold_trigger"
        trigger_type: "condition"
        condition: "data.temperature > 25 and data.humidity < 60"

        # Condition evaluation
        evaluation:
          language: "python"
          context_variables:
            - "data"
            - "metadata"
            - "system_state"
          timeout_ms: 1000

        # All data received trigger
        name: "batch_ready_trigger"
        trigger_type: "all_data_received"
        required_data_units:
          - "raw_data"
          - "configuration"
          - "metadata"

        # Activation settings
        activation:
          mode: "once_per_batch"
          reset_on_completion: true
          timeout_ms: 30000
        ```

    **Usage Patterns:**

        **Basic Data Monitoring:**
        ```python
        from nanobrain.core import DataUpdatedTrigger

        # Create trigger from configuration
        trigger = DataUpdatedTrigger.from_config('config/data_trigger.yml')

        # Register callback for activation
        async def on_data_updated(data_unit, old_value, new_value):
            print(f"Data changed: {old_value} -> {new_value}")
            # Trigger step execution
            await step.execute()

        trigger.register_callback(on_data_updated)

        # Start monitoring
        await trigger.start()
        ```

        **Timer-Based Processing:**
        ```python
        # Scheduled processing trigger
        timer_trigger = TimerTrigger.from_config('config/timer_trigger.yml')

        # Register processing callback
        async def scheduled_process():
            # Execute periodic processing
            results = await workflow.execute_batch()
            return results

        timer_trigger.register_callback(scheduled_process)

        # Start scheduled execution
        await timer_trigger.start()
        ```

        **Complex Condition Monitoring:**
        ```python
        # Custom condition trigger
        condition_trigger = ConditionalTrigger.from_config('config/condition_trigger.yml')

        # Advanced condition evaluation
        condition_expression = "data.temperature > threshold and data.trend == 'increasing'"
        condition_trigger.set_condition(condition_expression)

        # Context variables for evaluation
        condition_trigger.set_context({
            'threshold': 30.0,
            'system_state': system_monitor.get_state()
        })

        await condition_trigger.start()
        ```

        **Multi-Target Activation:**
        ```python
        # Trigger multiple steps simultaneously
        multi_trigger = DataUpdatedTrigger.from_config('config/multi_trigger.yml')

        # Register multiple targets
        multi_trigger.add_target(preprocessing_step)
        multi_trigger.add_target(validation_step)
        multi_trigger.add_target(logging_step)

        # All targets activated when trigger fires
        await multi_trigger.start()
        ```

    **Advanced Features:**

        **Debouncing and Rate Limiting:**
        * Configurable debounce periods to prevent excessive triggering
        * Rate limiting to control maximum activation frequency
        * Burst detection and handling for high-frequency events
        * Adaptive rate limiting based on processing capacity

        **Condition Evaluation:**
        * Python expression evaluation for custom conditions
        * Multi-variable context with data and system state
        * Safe evaluation with timeout and resource limits
        * Precompiled expressions for performance optimization

        **Event Aggregation:**
        * Batch event processing for improved efficiency
        * Event correlation and pattern detection
        * Time-window based aggregation
        * Statistical analysis of event patterns

        **Error Handling and Recovery:**
        * Robust error handling with detailed diagnostics
        * Automatic recovery from temporary failures
        * Circuit breaker patterns for unstable conditions
        * Fallback activation mechanisms

    **Performance and Scalability:**

        **Efficient Monitoring:**
        * Low-overhead condition checking with optimized algorithms
        * Event-driven architecture minimizing resource usage
        * Selective monitoring with configurable granularity
        * Batch processing for improved throughput

        **Scalability Features:**
        * Distributed trigger monitoring across multiple nodes
        * Load balancing for high-frequency event processing
        * Horizontal scaling with trigger distribution
        * Resource pooling and optimization

        **Monitoring and Metrics:**
        * Trigger activation frequency and timing analysis
        * Condition evaluation performance monitoring
        * Error rate tracking and optimization recommendations
        * Resource usage analysis and capacity planning

    **Integration Patterns:**

        **Workflow Orchestration:**
        * Event-driven workflow activation and coordination
        * Multi-stage pipeline triggering with dependencies
        * Conditional workflow branching based on trigger conditions
        * Dynamic workflow modification based on events

        **Real-Time Processing:**
        * Stream processing with continuous data monitoring
        * Low-latency response to critical events
        * Real-time analytics and alerting systems
        * Adaptive processing based on data characteristics

        **Batch Processing:**
        * Scheduled batch processing with timer triggers
        * Data availability-based batch activation
        * Resource-aware batch scheduling and optimization
        * Large dataset processing with progress monitoring

    **Event Lifecycle:**
        Triggers follow a well-defined event processing lifecycle:

        1. **Configuration Loading**: Parse and validate trigger configuration
        2. **Condition Setup**: Initialize monitoring conditions and parameters
        3. **Target Registration**: Register callback functions and target components
        4. **Monitoring Initialization**: Setup event listeners and data watchers
        5. **Active Monitoring**: Continuously monitor conditions and events
        6. **Condition Evaluation**: Evaluate trigger conditions when events occur
        7. **Activation Decision**: Determine whether to activate based on conditions
        8. **Target Activation**: Execute registered callbacks and activate targets
        9. **Rate Limiting**: Apply debouncing and frequency controls
        10. **Cleanup**: Handle cleanup and resource management

    **Security and Reliability:**

        **Secure Condition Evaluation:**
        * Safe expression evaluation with sandboxing
        * Input validation and sanitization
        * Resource limits and timeout protection
        * Access control for sensitive data and operations

        **Reliability Features:**
        * Fault tolerance with automatic recovery
        * Event persistence and replay capabilities
        * Redundancy and failover mechanisms
        * Health monitoring and alerting

        **Audit and Compliance:**
        * Comprehensive logging of trigger activations
        * Event history and audit trails
        * Performance metrics and compliance reporting
        * Security event tracking and analysis

    **Development and Testing:**

        **Testing Support:**
        * Mock trigger implementations for testing
        * Event simulation and validation frameworks
        * Trigger performance benchmarking
        * Integration testing with steps and workflows

        **Debugging Features:**
        * Comprehensive logging with event tracing
        * Condition evaluation debugging and analysis
        * Performance profiling and optimization hints
        * Visual event timeline and inspection tools

        **Development Tools:**
        * Trigger configuration validation and linting
        * Condition expression testing and validation
        * Performance monitoring and optimization tools
        * Event pattern analysis and optimization

    Attributes:
        name (str): Trigger identifier for logging and component coordination
        trigger_type (TriggerType): Type of trigger and activation pattern
        debounce_ms (int): Debounce period in milliseconds to prevent excessive activation
        max_frequency_hz (float): Maximum activation frequency in Hz for rate limiting
        condition (str, optional): Custom condition expression for conditional triggers
        callbacks (List[Callable]): Registered callback functions for activation
        is_active (bool): Whether trigger is currently monitoring conditions
        last_trigger_time (float): Timestamp of last activation for rate limiting
        trigger_count (int): Total number of activations since creation
        performance_metrics (Dict): Real-time performance and usage metrics

    Note:
        This is an abstract base class that cannot be instantiated directly.
        Use concrete implementations like DataUpdatedTrigger, TimerTrigger, or
        ConditionalTrigger. All triggers must be created using the from_config
        pattern with proper configuration files following framework patterns.

    Warning:
        Triggers can significantly impact system performance if configured with
        high frequencies or complex conditions. Monitor trigger performance and
        implement appropriate rate limiting and debouncing. Be cautious with
        condition expressions that access external resources or perform expensive operations.

    See Also:
        * :class:`TriggerConfig`: Trigger configuration schema and validation
        * :class:`TriggerType`: Available trigger types and activation patterns
        * :class:`DataUpdatedTrigger`: Data change monitoring and activation
        * :class:`TimerTrigger`: Scheduled and interval-based activation
        * :class:`ConditionalTrigger`: Custom condition evaluation and activation
        * :class:`BaseStep`: Steps that can be activated by triggers
        * :class:`Workflow`: Workflows that coordinate trigger-driven processing
    """

    COMPONENT_TYPE = "trigger"
    REQUIRED_CONFIG_FIELDS = ['trigger_type']
    OPTIONAL_CONFIG_FIELDS = {
        'debounce_ms': 100,
        'max_frequency_hz': 10.0,
        'condition': None,
        'timer_interval_ms': None,
        'name': ''
    }

    @classmethod
    def _get_config_class(cls):
        """UNIFIED PATTERN: Return TriggerConfig - ONLY method that differs from other components"""
        return TriggerConfig

    @classmethod
    def extract_component_config(cls, config: TriggerConfig) -> Dict[str, Any]:
        """Extract Trigger configuration"""
        return {
            'trigger_type': config.trigger_type,
            'debounce_ms': getattr(config, 'debounce_ms', 100),
            'max_frequency_hz': getattr(config, 'max_frequency_hz', 10.0),
            'condition': getattr(config, 'condition', None),
            'timer_interval_ms': getattr(config, 'timer_interval_ms', None),
            'name': getattr(config, 'name', '')
        }

    @classmethod
    def resolve_dependencies(cls, component_config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Resolve Trigger dependencies"""
        return {
            'enable_logging': kwargs.get('enable_logging', True),
            'debug_mode': kwargs.get('debug_mode', False)
        }

    def _init_from_config(self, config: TriggerConfig, component_config: Dict[str, Any],
                          dependencies: Dict[str, Any]) -> None:
        """Initialize Trigger with resolved dependencies"""
        self.config = config
        self.name = component_config.get('name') or self.__class__.__name__
        self._is_active = False
        self._callbacks: List[Callable] = []
        self._last_trigger_time = 0.0
        self._debounce_task: Optional[asyncio.Task] = None

        # Internal state tracking
        self._creation_time = time.time()
        self._trigger_count = 0
        self._rate_limited_count = 0
        self._callback_error_count = 0
        self._total_callback_time = 0.0

        # Initialize centralized logging system
        self.enable_logging = dependencies.get('enable_logging', True)
        if self.enable_logging:
            # Use centralized logging system
            self.nb_logger = get_logger(
                self.name, category="triggers", debug_mode=dependencies.get('debug_mode', False))

            # Register with system log manager
            system_manager = get_system_log_manager()
            system_manager.register_component("triggers", self.name, self, {
                "trigger_type": component_config['trigger_type'].value if hasattr(component_config['trigger_type'], 'value') else str(component_config['trigger_type']),
                "debounce_ms": component_config['debounce_ms'],
                "max_frequency_hz": component_config['max_frequency_hz'],
                "enable_logging": True
            })
        else:
            self.nb_logger = None

    # TriggerBase inherits FromConfigBase.__init__ which prevents direct instantiation

    def _get_internal_state(self) -> Dict[str, Any]:
        """Get comprehensive internal state for logging."""
        uptime = time.time() - self._creation_time
        avg_callback_time = self._total_callback_time / \
            max(self._trigger_count, 1)

        return {
            "is_active": self._is_active,
            "trigger_count": self._trigger_count,
            "rate_limited_count": self._rate_limited_count,
            "callback_error_count": self._callback_error_count,
            "callback_count": len(self._callbacks),
            "uptime_seconds": uptime,
            "last_trigger_time": self._last_trigger_time,
            "avg_callback_time_ms": avg_callback_time * 1000 if self._trigger_count > 0 else 0,
            "trigger_type": self.config.trigger_type.value if hasattr(self.config.trigger_type, 'value') else str(self.config.trigger_type),
            "debounce_ms": self.config.debounce_ms,
            "max_frequency_hz": self.config.max_frequency_hz,
            "success_rate": (self._trigger_count - self._callback_error_count) / max(self._trigger_count, 1)
        }

    @abstractmethod
    async def start_monitoring(self) -> None:
        """Start monitoring for trigger conditions."""
        pass

    @abstractmethod
    async def stop_monitoring(self) -> None:
        """Stop monitoring for trigger conditions."""
        pass

    async def add_callback(self, callback: Callable) -> None:
        """Add a callback to be executed when triggered."""
        if callback not in self._callbacks:
            self._callbacks.append(callback)

            # Log callback addition
            if self.enable_logging and self.nb_logger:
                self.nb_logger.info(f"Callback added to trigger: {self.name}",
                                    operation="add_callback",
                                    callback_count=len(self._callbacks),
                                    callback_name=getattr(
                                        callback, '__name__', str(callback)),
                                    internal_state=self._get_internal_state())

    async def remove_callback(self, callback: Callable) -> None:
        """Remove a callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

            # Log callback removal
            if self.enable_logging and self.nb_logger:
                self.nb_logger.info(f"Callback removed from trigger: {self.name}",
                                    operation="remove_callback",
                                    callback_count=len(self._callbacks),
                                    callback_name=getattr(
                                        callback, '__name__', str(callback)),
                                    internal_state=self._get_internal_state())

    async def trigger(self, data: Any = None) -> None:
        """Execute trigger with rate limiting and debouncing."""
        current_time = asyncio.get_event_loop().time()

        # Check frequency limit
        time_since_last = current_time - self._last_trigger_time
        min_interval = 1.0 / self.config.max_frequency_hz

        if time_since_last < min_interval:
            self._rate_limited_count += 1
            logger.debug(f"Trigger {self.name} rate limited")

            # Log rate limiting
            if self.enable_logging and self.nb_logger:
                self.nb_logger.debug(f"Trigger rate limited: {self.name}",
                                     operation="rate_limited",
                                     time_since_last=time_since_last,
                                     min_interval=min_interval,
                                     rate_limited_count=self._rate_limited_count)
            return

        # Cancel previous debounce task
        if self._debounce_task and not self._debounce_task.done():
            self._debounce_task.cancel()

        # Create debounced execution
        if self.config.debounce_ms > 0:
            self._debounce_task = asyncio.create_task(
                self._debounced_execute(data)
            )
        else:
            await self._execute_callbacks(data)

    async def _debounced_execute(self, data: Any) -> None:
        """Execute callbacks after debounce delay."""
        try:
            await asyncio.sleep(self.config.debounce_ms / 1000.0)
            await self._execute_callbacks(data)
        except asyncio.CancelledError:
            logger.debug(f"Debounced execution cancelled for {self.name}")

            # Log debounce cancellation
            if self.enable_logging and self.nb_logger:
                self.nb_logger.debug(f"Debounced execution cancelled: {self.name}",
                                     operation="debounce_cancelled")

    async def _execute_callbacks(self, data: Any) -> None:
        """Execute all registered callbacks."""
        self._last_trigger_time = asyncio.get_event_loop().time()
        self._trigger_count += 1

        start_time = time.time()
        successful_callbacks = 0

        # Log trigger activation
        if self.enable_logging and self.nb_logger:
            self.nb_logger.log_trigger_activation(
                trigger_name=self.name,
                trigger_type=self.config.trigger_type.value if hasattr(
                    self.config.trigger_type, 'value') else str(self.config.trigger_type),
                conditions={"data_type": type(
                    data).__name__ if data is not None else "None"},
                activated=True
            )

            self.nb_logger.info(f"Trigger activated: {self.name}",
                                operation="trigger_activated",
                                callback_count=len(self._callbacks),
                                data_type=type(
                                    data).__name__ if data is not None else "None",
                                trigger_count=self._trigger_count)

        for i, callback in enumerate(self._callbacks):
            try:
                callback_start = time.time()
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
                callback_duration = time.time() - callback_start
                self._total_callback_time += callback_duration
                successful_callbacks += 1

                # Log successful callback execution
                if self.enable_logging and self.nb_logger:
                    self.nb_logger.debug(f"Callback executed successfully: {self.name}[{i}]",
                                         operation="callback_success",
                                         callback_index=i,
                                         callback_name=getattr(
                                             callback, '__name__', str(callback)),
                                         duration_ms=callback_duration * 1000)

            except Exception as e:
                self._callback_error_count += 1
                logger.error(f"Error in trigger callback: {e}")

                # Log callback error
                if self.enable_logging and self.nb_logger:
                    self.nb_logger.error(f"Callback error in trigger: {self.name}[{i}]",
                                         operation="callback_error",
                                         callback_index=i,
                                         callback_name=getattr(
                                             callback, '__name__', str(callback)),
                                         error=str(e),
                                         error_type=type(e).__name__)

        # Log execution summary
        total_duration = time.time() - start_time
        if self.enable_logging and self.nb_logger:
            self.nb_logger.info(f"Trigger execution completed: {self.name}",
                                operation="trigger_completed",
                                successful_callbacks=successful_callbacks,
                                total_callbacks=len(self._callbacks),
                                total_duration_ms=total_duration * 1000,
                                internal_state=self._get_internal_state())

    @property
    def is_active(self) -> bool:
        """Check if trigger is actively monitoring."""
        return self._is_active


class DataUnitChangeTrigger(TriggerBase):
    """
    Event-driven trigger that fires when data unit changes occur.
    Uses change listener system for immediate response without polling.
    """

    @classmethod
    def from_config(cls, config: Union[str, Path, TriggerConfig, Dict[str, Any]], **kwargs) -> 'DataUnitChangeTrigger':
        """
        Enhanced from_config implementation following standard NanoBrain pattern

        Supports both file paths and inline dictionary configurations as per
        NanoBrain framework standards for DataUnit, Link, and Trigger classes.

        Args:
            config: Configuration file path, TriggerConfig object, or dictionary
            **kwargs: Additional context and dependencies

        Returns:
            Fully initialized DataUnitChangeTrigger instance

        ✅ FRAMEWORK COMPLIANCE:
        - Follows standard Union[str, Path, ConfigClass, Dict] pattern
        - Supports inline dict config as per Trigger rules
        - No hardcoding or simplified solutions
        - Pure configuration-driven instantiation
        """
        logger = get_logger(f"{cls.__name__}.from_config")
        logger.info(f"Creating {cls.__name__} from configuration")

        # Step 1: Normalize input to TriggerConfig object
        if isinstance(config, (str, Path)):
            # File path input - use standard config loading
            config_object = TriggerConfig.from_config(config, **kwargs)
        elif isinstance(config, dict):
            # Dictionary input - create TriggerConfig from dict (inline config support)
            # This is specifically allowed for DataUnit, Link, Trigger classes

            # Normalize legacy trigger type names
            normalized_config = config.copy()
            if normalized_config.get('trigger_type') == 'data_unit_change':
                normalized_config['trigger_type'] = 'data_updated'

            try:
                # Enable direct instantiation for config creation
                TriggerConfig._allow_direct_instantiation = True
                config_object = TriggerConfig(**normalized_config)
            finally:
                TriggerConfig._allow_direct_instantiation = False
        elif isinstance(config, TriggerConfig):
            # Already a TriggerConfig object
            config_object = config
        else:
            # Handle other BaseModel types
            if hasattr(config, 'model_dump'):
                config_dict = config.model_dump()
            elif hasattr(config, 'dict'):
                config_dict = config.dict()
            else:
                raise ValueError(f"Unsupported config type: {type(config)}")

            try:
                TriggerConfig._allow_direct_instantiation = True
                config_object = TriggerConfig(**config_dict)
            finally:
                TriggerConfig._allow_direct_instantiation = False

        # Step 2: Validate configuration schema
        cls.validate_config_schema(config_object)

        # Step 3: Extract component-specific configuration
        component_config = cls.extract_component_config(config_object)

        # Step 4: Resolve dependencies
        dependencies = cls.resolve_dependencies(component_config, **kwargs)

        # Step 5: Create instance
        instance = cls.create_instance(
            config_object, component_config, dependencies)

        # Step 6: Post-creation initialization
        instance._post_config_initialization()

        logger.info(f"Successfully created {cls.__name__}")
        return instance

    @classmethod
    def extract_component_config(cls, config: TriggerConfig) -> Dict[str, Any]:
        """Extract DataUnitChangeTrigger configuration including data_unit field"""
        base_config = super().extract_component_config(config)
        return {
            **base_config,
            'data_unit': getattr(config, 'data_unit', None),
            'event_type': getattr(config, 'event_type', 'set')
        }

    @classmethod
    def resolve_dependencies(cls, component_config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Resolve DataUnitChangeTrigger dependencies with step-scope data unit resolution

        ✅ ARCHITECTURAL COMPLIANCE: Only resolve data_unit string references 
        to actual DataUnit objects from step context (step-scope isolation).
        """
        base_deps = super().resolve_dependencies(component_config, **kwargs)

        # Get data unit reference (may be string or object)
        data_unit_ref = component_config.get('data_unit')

        # ✅ STEP-SCOPE DATA UNIT RESOLUTION
        if isinstance(data_unit_ref, str) and 'step_context' in kwargs:
            step_context = kwargs['step_context']

            # ✅ ARCHITECTURAL COMPLIANCE: Only search within step scope
            resolved_data_unit = None

            # Check step input data units
            step_input_units = step_context.get('step_input_data_units', {})
            if data_unit_ref in step_input_units:
                resolved_data_unit = step_input_units[data_unit_ref]

            # Check step output data units
            if not resolved_data_unit:
                step_output_units = step_context.get(
                    'step_output_data_units', {})
                if data_unit_ref in step_output_units:
                    resolved_data_unit = step_output_units[data_unit_ref]

            if resolved_data_unit:
                # ✅ Successfully resolved string reference to DataUnit object within step scope
                data_unit_ref = resolved_data_unit
                logger = get_logger(f"{cls.__name__}.resolve_dependencies")
                logger.info(
                    f"✅ Resolved data_unit reference: '{component_config.get('data_unit')}' -> {resolved_data_unit.name}")
            else:
                # ✅ STEP-SCOPE ISOLATION: Only show step-local data units
                available_units = (
                    list(step_input_units.keys()) +
                    list(step_output_units.keys())
                )
                step_name = step_context.get('step_name', 'unknown')
                raise ValueError(
                    f"❌ Data unit reference '{data_unit_ref}' not found in step '{step_name}' scope. "
                    f"Available step data units: {available_units}"
                )
        elif isinstance(data_unit_ref, str) and 'workflow_context' in kwargs:
            # ✅ LEGACY SUPPORT: Handle old workflow_context parameter for backward compatibility
            # But prioritize step-scope resolution for proper architecture
            workflow_context = kwargs['workflow_context']

            # Attempt to resolve string reference to actual DataUnit object
            resolved_data_unit = (
                workflow_context.get('step_input_data_units', {}).get(data_unit_ref) or
                workflow_context.get(
                    'step_output_data_units', {}).get(data_unit_ref)
            )

            if resolved_data_unit:
                data_unit_ref = resolved_data_unit
                logger = get_logger(f"{cls.__name__}.resolve_dependencies")
                logger.info(
                    f"✅ Resolved data_unit reference (legacy): '{component_config.get('data_unit')}' -> {resolved_data_unit.name}")
            else:
                available_units = (
                    list(workflow_context.get('step_input_data_units', {}).keys()) +
                    list(workflow_context.get(
                        'step_output_data_units', {}).keys())
                )
                raise ValueError(
                    f"❌ Data unit reference '{data_unit_ref}' not found in context. "
                    f"Available data units: {available_units}"
                )

        return {
            **base_deps,
            'data_unit': data_unit_ref,
            'event_type': component_config.get('event_type', 'data_unit_updated')
        }

    def _init_from_config(self, config: TriggerConfig, component_config: Dict[str, Any],
                          dependencies: Dict[str, Any]) -> None:
        """Initialize DataUnitChangeTrigger with resolved dependencies"""
        super()._init_from_config(config, component_config, dependencies)
        self.data_unit = dependencies.get('data_unit')
        # BRUTAL TRUTH: Fixed inconsistent default - now matches data unit event types
        self.event_type = validate_event_type(
            dependencies.get('event_type', DEFAULT_DATA_UNIT_EVENT),
            DataUnitEventType
        ).value
        self.bound_actions = []

        if not self.data_unit:
            raise ComponentConfigurationError(
                "DataUnitChangeTrigger requires data_unit")

    def bind_action(self, action_func: Callable) -> None:
        """Bind action to trigger for immediate execution on data unit changes"""
        if action_func not in self.bound_actions:
            self.bound_actions.append(action_func)
            if self.enable_logging and self.nb_logger:
                self.nb_logger.debug(f"Bound action to trigger {self.name}")

    def unbind_action(self, action_func: Callable) -> None:
        """Unbind action from trigger"""
        if action_func in self.bound_actions:
            self.bound_actions.remove(action_func)
            if self.enable_logging and self.nb_logger:
                self.nb_logger.debug(
                    f"Unbound action from trigger {self.name}")

    async def start_monitoring(self) -> None:
        """Start monitoring by registering as change listener"""
        if self._is_active:
            return

        self._is_active = True

        # Register with data unit's change listener system
        if self.enable_logging and self.nb_logger:
            self.nb_logger.info(f"🔗 BRUTAL TRUTH: About to register trigger {self.name} with data unit {getattr(self.data_unit, 'name', 'unknown')}")
            self.nb_logger.info(f"🔗 Data unit type: {type(self.data_unit)}")
            self.nb_logger.info(f"🔗 Data unit ID: {id(self.data_unit)}")

        self.data_unit.register_change_listener(self._on_data_unit_changed)

        # BRUTAL TRUTH: Add debugging for change listener registration
        if self.enable_logging and self.nb_logger:
            self.nb_logger.info(f"🔗 BRUTAL TRUTH: Trigger {self.name} registered as change listener on {getattr(self.data_unit, 'name', 'unknown')}")
            self.nb_logger.info(f"🔗 Data unit has {len(self.data_unit._change_listeners)} change listeners")

        if self.enable_logging and self.nb_logger:
            self.nb_logger.info(
                f"Started monitoring data unit {self.data_unit.name}")

    async def stop_monitoring(self) -> None:
        """Stop monitoring by unregistering change listener"""
        if not self._is_active:
            return

        self._is_active = False

        # Unregister from data unit's change listener system
        self.data_unit.unregister_change_listener(self._on_data_unit_changed)

        if self.enable_logging and self.nb_logger:
            self.nb_logger.info(
                f"Stopped monitoring data unit {self.data_unit.name}")

    async def _on_data_unit_changed(self, change_event: Dict[str, Any]) -> None:
        """Handle data unit change event"""
        try:
            # BRUTAL TRUTH: Add debugging for all change events
            if self.enable_logging and self.nb_logger:
                self.nb_logger.info(f"🔗 BRUTAL TRUTH: Trigger {self.name} received change event: {change_event}")

            # Check if event type matches (if specified) - FIXED: Add debug logging
            if hasattr(self, 'event_type') and self.event_type != 'all':
                operation = change_event.get('operation')
                if self.enable_logging and self.nb_logger:
                    self.nb_logger.info(f"🔗 Checking event type: expected '{self.event_type}', got '{operation}'")

                if operation != self.event_type:
                    # BRUTAL TRUTH: Log when triggers are ignored due to event type mismatch
                    if self.enable_logging and self.nb_logger:
                        self.nb_logger.warning(
                            f"🚫 Trigger {self.name} ignored event - expected '{self.event_type}', got '{operation}'"
                        )
                    return

            # Create trigger event
            trigger_event = {
                'trigger_id': getattr(self, 'trigger_id', self.name),
                'event_type': self.event_type,
                'data_unit': self.data_unit.name,
                'change_event': change_event,
                'timestamp': time.time()
            }

            # Execute bound actions immediately (no polling delay)
            logger.info(f"🔥 BRUTAL TRUTH: Trigger {self.name} executing {len(self.bound_actions)} bound actions")
            for action in self.bound_actions:
                logger.info(f"🔥 BRUTAL TRUTH: About to call bound action: {action}")
                try:
                    await action(trigger_event)
                    logger.info(f"🔥 BRUTAL TRUTH: Bound action completed successfully: {action}")
                except Exception as e:
                    logger.error(f"🔥 BRUTAL TRUTH: Bound action failed with exception: {e}")
                    logger.error(f"🔥 BRUTAL TRUTH: Exception type: {type(e).__name__}")
                    import traceback
                    logger.error(f"🔥 BRUTAL TRUTH: Traceback: {traceback.format_exc()}")
                    raise

            # Also execute callbacks for compatibility
            await self._execute_callbacks(change_event)

            if self.enable_logging and self.nb_logger:
                self.nb_logger.info(
                    f"🔥 Trigger {self.name} fired for data unit change")

        except Exception as e:
            if self.enable_logging and self.nb_logger:
                self.nb_logger.error(f"❌ Error in DataUnitChangeTrigger: {e}")


# DataUpdatedTrigger removed - NanoBrain uses pure event-driven architecture
# All triggers now use DataUnitChangeTrigger for immediate event notifications


class AllDataReceivedTrigger(TriggerBase):
    """
    Trigger that fires when all required data units have data.
    """

    @classmethod
    def from_config(cls, config: TriggerConfig, **kwargs) -> 'AllDataReceivedTrigger':
        """Mandatory from_config implementation for AllDataReceivedTrigger"""
        logger = get_logger(f"{cls.__name__}.from_config")
        logger.info(f"Creating {cls.__name__} from configuration")

        # Step 1: Validate configuration schema
        cls.validate_config_schema(config)

        # Step 2: Extract component-specific configuration
        component_config = cls.extract_component_config(config)

        # Step 3: Resolve dependencies
        dependencies = cls.resolve_dependencies(component_config, **kwargs)

        # Step 4: Create instance
        instance = cls.create_instance(config, component_config, dependencies)

        # Step 5: Post-creation initialization
        instance._post_config_initialization()

        logger.info(f"Successfully created {cls.__name__}")
        return instance

    @classmethod
    def resolve_dependencies(cls, component_config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Resolve AllDataReceivedTrigger dependencies"""
        base_deps = super().resolve_dependencies(component_config, **kwargs)
        return {
            **base_deps,
            'data_units': kwargs.get('data_units', [])
        }

    def _init_from_config(self, config: TriggerConfig, component_config: Dict[str, Any],
                          dependencies: Dict[str, Any]) -> None:
        """Initialize AllDataReceivedTrigger with resolved dependencies"""
        super()._init_from_config(config, component_config, dependencies)
        self.data_units = dependencies.get('data_units', [])
        self._monitoring_task: Optional[asyncio.Task] = None

        # G2 — dynamic expected-set fields. Cached at trigger init; resolved
        # against a source data unit on first activation via
        # _resolve_expected_set(). When all three are absent (default), the
        # historical static-list behavior is preserved.
        self._expected_set_source: Optional[str] = getattr(
            config, "expected_set_source", None)
        self._expected_set_field: Optional[str] = getattr(
            config, "expected_set_field", None)
        self._expected_set_naming: str = getattr(
            config, "expected_set_naming", "{value}") or "{value}"
        self._resolved_expected_set: Optional[set[str]] = None  # cached after first resolve

        # G10 — gate-to-bottom semantics. When 'publish_empty' (default;
        # legacy), only non-None payloads count as satisfied. When
        # 'gate_to_bottom', the GATED_OFF_SENTINEL also counts as
        # satisfied AND is excluded from the trigger payload.
        gs = getattr(config, "gate_semantics", "publish_empty")
        if gs not in ("publish_empty", "gate_to_bottom"):
            raise ComponentConfigurationError(
                f"FAIL-FAST: AllDataReceivedTrigger {self.name!r} "
                f"gate_semantics must be 'publish_empty' or 'gate_to_bottom', "
                f"got {gs!r}"
            )
        self.gate_semantics: str = gs

    def _is_satisfied(self, payload: Any) -> tuple[bool, bool]:
        """G10 satisfaction predicate for a single data unit's payload.

        Returns:
            (satisfied, include_in_payload) tuple.
            - satisfied: True if this data unit counts toward
              "all data received" for trigger firing.
            - include_in_payload: True if the value should appear in
              the dict forwarded to downstream consumers; False to
              exclude (used to hide the gated-off sentinel from
              user code, per G10).

        Semantics:
            - payload is None → (False, False): unsatisfied, would
              not include either way.
            - payload is the GATED_OFF_SENTINEL AND gate_semantics
              is 'gate_to_bottom' → (True, False): counts as
              satisfied for firing, but excluded from payload so
              user process() never sees the magic string.
            - payload is the GATED_OFF_SENTINEL AND gate_semantics
              is 'publish_empty' → (False, False): legacy semantics
              treat the sentinel as opaque user data the trigger
              doesn't recognize. The condition is "not received"
              because the magic string is not None — but legacy
              code can't have written it intentionally either. We
              treat it as unsatisfied to preserve the dominant
              v1 deadlock-on-gate failure mode unless the operator
              opts in.
            - any other non-None payload → (True, True).
        """
        # Lazy import to avoid the trigger.py ↔ link.py cycle.
        from .link import ConditionalLink

        if payload is None:
            return (False, False)

        if payload == ConditionalLink.GATED_OFF_SENTINEL:
            if self.gate_semantics == "gate_to_bottom":
                return (True, False)
            return (False, False)

        return (True, True)

    async def _resolve_expected_set(
        self,
        source_data_unit: Any,
        static_inputs: List[str],
    ) -> set[str]:
        """G2 expected-set resolver. Called once on first activation.

        Args:
            source_data_unit: The workflow-level data unit referenced by
                ``expected_set_source``. Caller (workflow loader) MUST
                resolve the ``"workflow.<unit_name>"`` reference and pass
                the actual data unit object.
            static_inputs: The static ``inputs`` list declared on the
                trigger config (the FULL set of possible upstream data
                unit names; the dynamic narrowing intersects with this).

        Returns:
            The set of data unit names that the trigger waits for on the
            current run.

        Raises:
            ComponentConfigurationError: when the source data unit's
                payload is missing the projected field, or the projection
                is not a list of strings, or the projected names are not
                a subset of ``static_inputs``.

        Behavior:
            - When ``expected_set_source`` is None, returns ``set(static_inputs)``
              unchanged (preserves the v1 static-list semantics).
            - When set, reads ``source_data_unit.get()``, walks
              ``expected_set_field`` via the G1 dotted-path resolver,
              formats each projected string through ``expected_set_naming``,
              and intersects with ``static_inputs``.
            - Result is cached on ``self._resolved_expected_set`` so
              repeated calls within the same activation cycle are O(1).
        """
        # Cached ⇒ return cached (covers both no-source and dynamic paths
        # after first call). Cache invariant: once set, never replaced.
        if self._resolved_expected_set is not None:
            return self._resolved_expected_set

        # No dynamic source ⇒ static behavior. Cache and return.
        if not self._expected_set_source:
            self._resolved_expected_set = set(static_inputs)
            return self._resolved_expected_set

        if source_data_unit is None:
            raise ComponentConfigurationError(
                f"FAIL-FAST: AllDataReceivedTrigger {self.name!r} "
                f"expected_set_source={self._expected_set_source!r} "
                f"requires the workflow loader to resolve and pass the "
                f"source data unit object; got None"
            )

        payload = await source_data_unit.get()

        # Reuse G1's dotted-path resolver — same path semantics for
        # consistency. Imported lazily to avoid a hard cycle.
        from .link import get_nested_value_strict, _PATH_MISS

        field_path = self._expected_set_field or ""
        projected = get_nested_value_strict(payload, field_path)
        if projected is _PATH_MISS:
            raise ComponentConfigurationError(
                f"FAIL-FAST: AllDataReceivedTrigger {self.name!r} "
                f"expected_set_field={field_path!r} missing in source data "
                f"unit payload (payload type "
                f"{type(payload).__name__})"
            )

        if not isinstance(projected, (list, tuple)):
            raise ComponentConfigurationError(
                f"FAIL-FAST: AllDataReceivedTrigger {self.name!r} "
                f"expected_set_field={field_path!r} projected to "
                f"{type(projected).__name__}, expected list[str]"
            )
        if not all(isinstance(item, str) for item in projected):
            raise ComponentConfigurationError(
                f"FAIL-FAST: AllDataReceivedTrigger {self.name!r} "
                f"expected_set_field={field_path!r} projected list contains "
                f"non-string elements"
            )

        # Format through the naming template.
        try:
            named = {self._expected_set_naming.format(value=v) for v in projected}
        except (KeyError, IndexError) as e:
            raise ComponentConfigurationError(
                f"FAIL-FAST: AllDataReceivedTrigger {self.name!r} "
                f"expected_set_naming={self._expected_set_naming!r} format failed: {e}"
            ) from e

        # Intersect with static inputs (validate that every projected name
        # actually exists in the trigger's declared input set).
        static_set = set(static_inputs)
        if not named.issubset(static_set):
            offenders = named - static_set
            raise ComponentConfigurationError(
                f"FAIL-FAST: AllDataReceivedTrigger {self.name!r} "
                f"expected_set narrows to {sorted(named)} which is not a "
                f"subset of inputs={sorted(static_set)}; off-DAG names: "
                f"{sorted(offenders)}"
            )

        self._resolved_expected_set = named
        return self._resolved_expected_set

    async def start_monitoring(self) -> None:
        """Start monitoring for all data received."""
        if self._is_active:
            return

        self._is_active = True
        self._monitoring_task = asyncio.create_task(self._monitor_all_data())
        logger.debug(f"AllDataReceivedTrigger {self.name} started monitoring")

    async def stop_monitoring(self) -> None:
        """Stop monitoring."""
        self._is_active = False

        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass

        logger.debug(f"AllDataReceivedTrigger {self.name} stopped monitoring")

    async def _monitor_all_data(self) -> None:
        """Monitor until all data units have data.

        G10 — uses ``_is_satisfied`` to recognize the GATED_OFF_SENTINEL
        as a satisfaction signal under ``gate_semantics='gate_to_bottom'``.
        Sentinel-bearing units are counted toward firing but excluded
        from the outgoing payload so user ``process()`` never sees the
        magic string.
        """
        try:
            while self._is_active:
                all_have_data = True
                data_dict = {}

                for i, data_unit in enumerate(self.data_units):
                    data = await data_unit.get()
                    satisfied, include = self._is_satisfied(data)
                    if not satisfied:
                        all_have_data = False
                        break
                    if include:
                        data_dict[f"input_{i}"] = data

                if all_have_data:
                    await self.trigger(data_dict)
                    # Stop monitoring after successful trigger
                    break

                # Check again after a short delay
                await asyncio.sleep(0.1)

        except asyncio.CancelledError:
            logger.debug(
                f"AllDataReceivedTrigger {self.name} monitoring cancelled")
        except Exception as e:
            logger.error(f"Error in AllDataReceivedTrigger {self.name}: {e}")


class TimerTrigger(TriggerBase):
    """
    Trigger that fires at regular intervals.
    """

    @classmethod
    def from_config(cls, config: Union[str, Path, TriggerConfig, Dict[str, Any]], **kwargs) -> 'TimerTrigger':
        """
        Enhanced from_config implementation following standard NanoBrain pattern

        Supports both file paths and inline dictionary configurations as per
        NanoBrain framework standards for DataUnit, Link, and Trigger classes.

        Args:
            config: Configuration file path, TriggerConfig object, or dictionary
            **kwargs: Additional context and dependencies

        Returns:
            Fully initialized TimerTrigger instance

        ✅ FRAMEWORK COMPLIANCE:
        - Follows standard Union[str, Path, ConfigClass, Dict] pattern
        - Supports inline dict config as per Trigger rules
        - No hardcoding or simplified solutions
        - Pure configuration-driven instantiation
        """
        logger = get_logger(f"{cls.__name__}.from_config")
        logger.info(f"Creating {cls.__name__} from configuration")

        # Step 1: Normalize input to TriggerConfig object
        if isinstance(config, (str, Path)):
            # File path input - use standard config loading
            config_object = TriggerConfig.from_config(config, **kwargs)
        elif isinstance(config, dict):
            # Dictionary input - create TriggerConfig from dict (inline config support)
            # This is specifically allowed for DataUnit, Link, Trigger classes

            # Normalize legacy trigger type names
            normalized_config = config.copy()
            if normalized_config.get('trigger_type') == 'data_unit_change':
                normalized_config['trigger_type'] = 'data_updated'

            try:
                # Enable direct instantiation for config creation
                TriggerConfig._allow_direct_instantiation = True
                config_object = TriggerConfig(**normalized_config)
            finally:
                TriggerConfig._allow_direct_instantiation = False
        elif isinstance(config, TriggerConfig):
            # Already a TriggerConfig object
            config_object = config
        else:
            # Handle other BaseModel types
            if hasattr(config, 'model_dump'):
                config_dict = config.model_dump()
            elif hasattr(config, 'dict'):
                config_dict = config.dict()
            else:
                raise ValueError(f"Unsupported config type: {type(config)}")

            try:
                TriggerConfig._allow_direct_instantiation = True
                config_object = TriggerConfig(**config_dict)
            finally:
                TriggerConfig._allow_direct_instantiation = False

        # Step 2: Validate configuration schema
        cls.validate_config_schema(config_object)

        # Step 3: Extract component-specific configuration
        component_config = cls.extract_component_config(config_object)

        # Step 4: Resolve dependencies
        dependencies = cls.resolve_dependencies(component_config, **kwargs)

        # Step 5: Create instance
        instance = cls.create_instance(
            config_object, component_config, dependencies)

        # Step 6: Post-creation initialization
        instance._post_config_initialization()

        logger.info(f"Successfully created {cls.__name__}")
        return instance

    @classmethod
    def resolve_dependencies(cls, component_config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Resolve TimerTrigger dependencies"""
        base_deps = super().resolve_dependencies(component_config, **kwargs)
        interval_ms = kwargs.get('interval_ms') or component_config.get(
            'timer_interval_ms', 1000)
        return {
            **base_deps,
            'interval_ms': interval_ms
        }

    def _init_from_config(self, config: TriggerConfig, component_config: Dict[str, Any],
                          dependencies: Dict[str, Any]) -> None:
        """Initialize TimerTrigger with resolved dependencies"""
        super()._init_from_config(config, component_config, dependencies)
        self.interval_ms = dependencies.get('interval_ms', 1000)
        self._timer_task: Optional[asyncio.Task] = None

        # G22 Step 3 — missed-schedule policy. Read from TriggerConfig;
        # validated at the Pydantic layer (Literal["skip","catch_up","merge"]).
        self.on_missed: str = getattr(config, "on_missed", "skip")
        # The last-fire wall-clock time (UTC epoch seconds). Set by the
        # restart-recovery hook; None means "no prior fire known".
        self._last_fire_epoch_seconds: Optional[float] = None

    async def replay_missed_fires(
        self,
        last_known_fire_epoch_seconds: float,
        now_epoch_seconds: Optional[float] = None,
    ) -> int:
        """G22 Step 3 — apply ``on_missed`` policy after a restart.

        Compute the number of missed fires between
        ``last_known_fire_epoch_seconds`` (the wall-clock time at which
        the trigger last fired before the restart, persisted by the
        deployment) and ``now_epoch_seconds`` (defaults to current UTC
        epoch). Then act per ``self.on_missed``:

        - ``skip``    → return 0 (no fires).
        - ``merge``   → fire ONCE if N >= 1, else 0; return 1 or 0.
        - ``catch_up`` → fire N times in rapid succession; return N.

        The returned int is the count of fires actually emitted. The
        framework provides the LAST-FIRE persistence in Step 4 (durable
        inner-trigger binding); this method is the policy-application
        primitive that Step 4 builds on.

        Caller responsibility: invoke this method ONCE at trigger
        startup, before ``start_monitoring``. It does not own
        persistence — it only consumes the persisted timestamp.

        Returns:
            int — number of fires emitted.

        Raises:
            ValueError if last_known_fire_epoch_seconds > now (clock
            skew or bad input).
        """
        import time as _time

        if now_epoch_seconds is None:
            now_epoch_seconds = _time.time()

        if last_known_fire_epoch_seconds > now_epoch_seconds:
            raise ValueError(
                f"FAIL-FAST: TimerTrigger {self.name!r} replay_missed_fires: "
                f"last_known_fire_epoch_seconds ({last_known_fire_epoch_seconds}) "
                f"is in the future relative to now ({now_epoch_seconds}); "
                f"check clock skew or persistence bug"
            )

        if self.interval_ms <= 0:
            return 0  # nothing meaningful to replay

        # Integer-millisecond arithmetic dodges the
        # ``1.0 / 0.1 == 9.999...`` floating-point trap. Round the
        # elapsed window (not floor) so a boundary case like
        # exactly-N-intervals counts as N.
        elapsed_ms = round((now_epoch_seconds - last_known_fire_epoch_seconds) * 1000.0)
        missed_count = elapsed_ms // self.interval_ms
        if missed_count <= 0:
            return 0

        # Was active before? Restore for the replay window. Caller
        # is responsible for the regular start_monitoring afterwards.
        was_active = self._is_active
        self._is_active = True
        try:
            if self.on_missed == "skip":
                fires = 0
            elif self.on_missed == "merge":
                # Bypass rate-limit / debounce — catch-up replay is an
                # explicit user opt-in, not a normal cadence fire.
                await self._execute_callbacks(None)
                fires = 1
            elif self.on_missed == "catch_up":
                for _ in range(missed_count):
                    await self._execute_callbacks(None)
                fires = missed_count
            else:
                # Defensive — Pydantic Literal should reject this earlier.
                raise ComponentConfigurationError(
                    f"FAIL-FAST: TimerTrigger {self.name!r} unknown "
                    f"on_missed policy {self.on_missed!r}"
                )
        finally:
            self._is_active = was_active

        self._last_fire_epoch_seconds = now_epoch_seconds
        return fires

    async def start_monitoring(self) -> None:
        """Start timer monitoring."""
        if self._is_active:
            return

        self._is_active = True
        self._timer_task = asyncio.create_task(self._timer_loop())
        logger.debug(
            f"TimerTrigger {self.name} started with {self.interval_ms}ms interval")

    async def stop_monitoring(self) -> None:
        """Stop timer monitoring."""
        self._is_active = False

        if self._timer_task and not self._timer_task.done():
            self._timer_task.cancel()
            try:
                await self._timer_task
            except asyncio.CancelledError:
                pass

        logger.debug(f"TimerTrigger {self.name} stopped")

    async def _timer_loop(self) -> None:
        """Timer loop that triggers at intervals."""
        try:
            while self._is_active:
                await asyncio.sleep(self.interval_ms / 1000.0)
                if self._is_active:  # Check again after sleep
                    await self.trigger()
        except asyncio.CancelledError:
            logger.debug(f"TimerTrigger {self.name} loop cancelled")
        except Exception as e:
            logger.error(f"Error in TimerTrigger {self.name}: {e}")


class ManualTrigger(TriggerBase):
    """
    Trigger that fires only when manually activated.
    """

    @classmethod
    def from_config(cls, config: Union[str, Path, TriggerConfig, Dict[str, Any]], **kwargs) -> 'ManualTrigger':
        """
        Enhanced from_config implementation following standard NanoBrain pattern

        Supports both file paths and inline dictionary configurations as per
        NanoBrain framework standards for DataUnit, Link, and Trigger classes.

        Args:
            config: Configuration file path, TriggerConfig object, or dictionary
            **kwargs: Additional context and dependencies

        Returns:
            Fully initialized ManualTrigger instance

        ✅ FRAMEWORK COMPLIANCE:
        - Follows standard Union[str, Path, ConfigClass, Dict] pattern
        - Supports inline dict config as per Trigger rules
        - No hardcoding or simplified solutions
        - Pure configuration-driven instantiation
        """
        logger = get_logger(f"{cls.__name__}.from_config")
        logger.info(f"Creating {cls.__name__} from configuration")

        # Step 1: Normalize input to TriggerConfig object
        if isinstance(config, (str, Path)):
            # File path input - use standard config loading
            config_object = TriggerConfig.from_config(config, **kwargs)
        elif isinstance(config, dict):
            # Dictionary input - create TriggerConfig from dict (inline config support)
            # This is specifically allowed for DataUnit, Link, Trigger classes

            # Normalize legacy trigger type names
            normalized_config = config.copy()
            if normalized_config.get('trigger_type') == 'data_unit_change':
                normalized_config['trigger_type'] = 'data_updated'

            try:
                # Enable direct instantiation for config creation
                TriggerConfig._allow_direct_instantiation = True
                config_object = TriggerConfig(**normalized_config)
            finally:
                TriggerConfig._allow_direct_instantiation = False
        elif isinstance(config, TriggerConfig):
            # Already a TriggerConfig object
            config_object = config
        else:
            # Handle other BaseModel types
            if hasattr(config, 'model_dump'):
                config_dict = config.model_dump()
            elif hasattr(config, 'dict'):
                config_dict = config.dict()
            else:
                raise ValueError(f"Unsupported config type: {type(config)}")

            try:
                TriggerConfig._allow_direct_instantiation = True
                config_object = TriggerConfig(**config_dict)
            finally:
                TriggerConfig._allow_direct_instantiation = False

        # Step 2: Validate configuration schema
        cls.validate_config_schema(config_object)

        # Step 3: Extract component-specific configuration
        component_config = cls.extract_component_config(config_object)

        # Step 4: Resolve dependencies
        dependencies = cls.resolve_dependencies(component_config, **kwargs)

        # Step 5: Create instance
        instance = cls.create_instance(
            config_object, component_config, dependencies)

        # Step 6: Post-creation initialization
        instance._post_config_initialization()

        logger.info(f"Successfully created {cls.__name__}")
        return instance

    def _init_from_config(self, config: TriggerConfig, component_config: Dict[str, Any],
                          dependencies: Dict[str, Any]) -> None:
        """Initialize ManualTrigger with resolved dependencies"""
        super()._init_from_config(config, component_config, dependencies)
        self.bound_actions = []

    async def start_monitoring(self) -> None:
        """Start monitoring (no-op for manual trigger)."""
        self._is_active = True
        logger.debug(f"ManualTrigger {self.name} ready for manual activation")

    async def stop_monitoring(self) -> None:
        """Stop monitoring."""
        self._is_active = False
        logger.debug(f"ManualTrigger {self.name} deactivated")

    def bind_action(self, action_func: Callable) -> None:
        """Bind action to trigger for execution when manually fired"""
        if action_func not in self.bound_actions:
            self.bound_actions.append(action_func)
            if self.enable_logging and self.nb_logger:
                self.nb_logger.debug(f"Bound action to manual trigger {self.name}")

    def unbind_action(self, action_func: Callable) -> None:
        """Unbind action from trigger"""
        if action_func in self.bound_actions:
            self.bound_actions.remove(action_func)
            if self.enable_logging and self.nb_logger:
                self.nb_logger.debug(f"Unbound action from manual trigger {self.name}")

    async def fire(self, data: Any = None) -> None:
        """Manually fire the trigger."""
        if self._is_active:
            # Execute bound actions
            for action in self.bound_actions:
                try:
                    await action(data)
                except Exception as e:
                    logger.error(f"Error executing bound action in ManualTrigger {self.name}: {e}")

            await self.trigger(data)
        else:
            logger.warning(f"ManualTrigger {self.name} not active")


class EventTrigger(TriggerBase):
    """G22 — externally-fired event trigger.

    Per ``apecx-mcp-integration/docs/CONTRACTS.md#g22``.

    Designed for HTTP webhooks and message-bus subscriptions. The
    framework-provided primitive is **transport-agnostic**: this class
    does not embed an HTTP server or a message-bus client. External
    code (a webhook handler in your deployment, or a message-bus
    consumer task) calls ``await trigger.fire_event(event_body)`` for
    each incoming event. Production deployments wire the transport
    plumbing on top of this primitive.

    The optional ``event_filter`` field on ``TriggerConfig`` is a G1
    declarative predicate dict (e.g.
    ``{"op": "eq", "field": "kind", "value": "novel"}``). When set,
    ``fire_event`` evaluates the predicate against the event body and
    returns silently (no callbacks invoked) on miss. When unset, every
    event fires.

    The trigger holds NO event history; it is fire-and-forget. Callers
    that need replay-on-restart semantics should layer a durable queue
    in front of ``fire_event``.
    """

    @classmethod
    def from_config(cls, config: Union[str, Path, TriggerConfig, Dict[str, Any]],
                    **kwargs) -> "EventTrigger":
        """Standard from_config implementation matching ManualTrigger /
        TimerTrigger; supports file paths, dict, and TriggerConfig input."""
        nb_logger = get_logger(f"{cls.__name__}.from_config")
        nb_logger.info(f"Creating {cls.__name__} from configuration")

        if isinstance(config, (str, Path)):
            config_object = TriggerConfig.from_config(config, **kwargs)
        elif isinstance(config, dict):
            normalized_config = config.copy()
            try:
                TriggerConfig._allow_direct_instantiation = True
                config_object = TriggerConfig(**normalized_config)
            finally:
                TriggerConfig._allow_direct_instantiation = False
        elif isinstance(config, TriggerConfig):
            config_object = config
        else:
            if hasattr(config, "model_dump"):
                config_dict = config.model_dump()
            elif hasattr(config, "dict"):
                config_dict = config.dict()
            else:
                raise ValueError(f"Unsupported config type: {type(config)}")
            try:
                TriggerConfig._allow_direct_instantiation = True
                config_object = TriggerConfig(**config_dict)
            finally:
                TriggerConfig._allow_direct_instantiation = False

        cls.validate_config_schema(config_object)
        component_config = cls.extract_component_config(config_object)
        dependencies = cls.resolve_dependencies(component_config, **kwargs)
        instance = cls.create_instance(config_object, component_config, dependencies)
        instance._post_config_initialization()

        nb_logger.info(f"Successfully created {cls.__name__}")
        return instance

    def _init_from_config(self, config: TriggerConfig, component_config: Dict[str, Any],
                          dependencies: Dict[str, Any]) -> None:
        """Initialize EventTrigger; pre-build the optional event_filter
        predicate so each fire_event call is O(predicate-eval), not
        O(predicate-build + eval)."""
        super()._init_from_config(config, component_config, dependencies)

        self._event_filter_raw = getattr(config, "event_filter", None)
        self._event_filter_func: Optional[Callable[[Any], bool]] = None
        if self._event_filter_raw is not None:
            # Reuse the G1 condition resolver — same dict shape as
            # ConditionalLink's `condition` field. Lazy import to avoid
            # the trigger.py ↔ link.py cycle.
            from .link import parse_condition_from_config
            self._event_filter_func = parse_condition_from_config(self._event_filter_raw)

    async def start_monitoring(self) -> None:
        """No-op: event-driven, not polled."""
        self._is_active = True
        logger.debug(f"EventTrigger {self.name} ready for fire_event() calls")

    async def stop_monitoring(self) -> None:
        """No-op: nothing to cancel."""
        self._is_active = False
        logger.debug(f"EventTrigger {self.name} deactivated")

    async def fire_event(self, event_body: Any) -> bool:
        """Fire the trigger with ``event_body`` if the optional filter
        passes. Returns True if the trigger fired, False if filtered out
        or inactive.

        FAIL-FAST is preserved: predicate evaluation errors propagate
        as ``ComponentConfigurationError`` from the G1 evaluator; we
        do not swallow them.
        """
        if not self._is_active:
            logger.warning(f"EventTrigger {self.name} not active; ignoring event")
            return False

        if self._event_filter_func is not None:
            should_fire = self._event_filter_func(event_body)
            if asyncio.iscoroutine(should_fire):
                should_fire = await should_fire
            if not should_fire:
                logger.debug(
                    f"EventTrigger {self.name} event filtered out by predicate"
                )
                return False

        await self.trigger(event_body)
        return True
