"""
Step System for NanoBrain Framework

Provides event-driven data processing with DataUnit integration.
Enhanced with mandatory from_config pattern implementation.
"""

import asyncio
import importlib
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Union
from pathlib import Path
from pydantic import Field, ConfigDict, model_validator

from .component_base import (
    FromConfigBase, ComponentConfigurationError, import_class_from_path
)
from .executor import LocalExecutor, ExecutorConfig
from .data_unit import DataUnitBase, DataUnitConfig
from .trigger import TriggerBase, TriggerConfig
from .link import LinkBase
from .logging_system import (
    get_logger, OperationType
)
# Import new ConfigBase for constructor prohibition
from .config.config_base import ConfigBase
from .agent_response import AgentResponse, AgentProcessingMetadata, ConversationContext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# G6 — Typed step input/output schemas (added 2026-05-09)
# ---------------------------------------------------------------------------
#
# Per `apecx-mcp-integration/docs/CONTRACTS.md#g6`: every
# step's `process()` may declare an input + output schema. The framework
# validates payloads at the wire boundary: input on the way in, output on
# the way out. The dominant motivation is the silent-failure shape where
# a step's return-dict key doesn't match the declared output_data_units
# name (the framework silently drops the value); typed schemas surface
# this at validation time as a FAIL-FAST.
#
# Reserved-field escape valve (per the gap proposal):
# - `errors`: list[StepError] — always allowed alongside the typed payload
# - `partial`: bool — always allowed; signals upstream degradation
# These two field names are reserved across all step output schemas; the
# framework validates them against fixed shapes and ignores them when
# checking the user's declared schema.
# ---------------------------------------------------------------------------

# Reserved field names that ANY step output may include alongside its
# typed payload. Validated against fixed shapes.
RESERVED_OUTPUT_FIELDS = ("errors", "partial")


# ---------------------------------------------------------------------------
# G12 — Declarative resource envelope on Step.
# ---------------------------------------------------------------------------

class ResourceEnvelope(ConfigBase):
    """G12 — per-step resource projection.

    Each field is OPTIONAL; a step that omits a field is treated as
    "no declared bound" for that dimension. The aggregator
    (Workflow.aggregate_resource_envelope) skips None fields.

    Per-field aggregation rules (workflow rolls up per-step values):

    | Field | Aggregation |
    |---|---|
    | walltime_minutes | sum (worst-case sequential) |
    | cpu_cores | max (steps run on the same allocation; envelope is the peak) |
    | memory_gb | max |
    | capability_tokens | union (every token any step needs) |
    | cost_units | sum (every step's cost adds) |

    Future enhancement: parallel-aware aggregation that knows when the
    workflow runs steps in parallel (then walltime = max not sum). v1
    over-projects, which is the safe direction (operator sees a
    too-high envelope, not a too-low one).
    """
    model_config = ConfigDict(extra="forbid")

    walltime_minutes: Optional[float] = Field(default=None, ge=0.0)
    cpu_cores: Optional[float] = Field(default=None, ge=0.0)
    memory_gb: Optional[float] = Field(default=None, ge=0.0)
    capability_tokens: List[str] = Field(default_factory=list)
    cost_units: Optional[float] = Field(default=None, ge=0.0)


def aggregate_resource_envelopes(
    envelopes: List[ResourceEnvelope],
) -> ResourceEnvelope:
    """G12 — aggregate per-step envelopes into a workflow-level envelope.

    Per-field rule (see ResourceEnvelope docstring):
    - walltime_minutes: sum (worst-case sequential)
    - cpu_cores / memory_gb: max
    - capability_tokens: union
    - cost_units: sum

    Returns a new ResourceEnvelope. Omitted fields (None across all
    inputs) remain None in the output. Empty list returns an empty
    envelope (all None / empty list).
    """
    if not envelopes:
        ResourceEnvelope._allow_direct_instantiation = True
        try:
            return ResourceEnvelope()
        finally:
            ResourceEnvelope._allow_direct_instantiation = False

    # Sum-fields: walltime, cost. None contributes 0; if EVERY input is
    # None for a field, output is also None.
    def _sum_or_none(values: List[Optional[float]]) -> Optional[float]:
        non_none = [v for v in values if v is not None]
        return sum(non_none) if non_none else None

    def _max_or_none(values: List[Optional[float]]) -> Optional[float]:
        non_none = [v for v in values if v is not None]
        return max(non_none) if non_none else None

    walltimes = [e.walltime_minutes for e in envelopes]
    costs = [e.cost_units for e in envelopes]
    cpus = [e.cpu_cores for e in envelopes]
    mems = [e.memory_gb for e in envelopes]
    all_tokens: set[str] = set()
    for e in envelopes:
        all_tokens.update(e.capability_tokens)

    ResourceEnvelope._allow_direct_instantiation = True
    try:
        return ResourceEnvelope(
            walltime_minutes=_sum_or_none(walltimes),
            cpu_cores=_max_or_none(cpus),
            memory_gb=_max_or_none(mems),
            capability_tokens=sorted(all_tokens),
            cost_units=_sum_or_none(costs),
        )
    finally:
        ResourceEnvelope._allow_direct_instantiation = False


class StepError(ConfigBase):
    """Reserved error envelope used in the G6 escape valve. Step output may
    include an `errors: list[StepError]` field even if not declared in the
    output schema; the framework validates against this fixed shape."""
    model_config = ConfigDict(extra="forbid")
    code: str
    detail: str
    source: Optional[str] = None


class SchemaRef(ConfigBase):
    """G6 — declarative schema reference. EXACTLY ONE of `class` (Pydantic
    model dotted path) or `json_schema` (inline JSON Schema dict) must be set.

    The framework imports the Pydantic class at validation time and calls
    `Cls.model_validate(payload)`; for JSON Schema, it uses `jsonschema`
    (already a transitive dependency of the framework via several extras).

    Cross-reference `apecx-mcp-integration/docs/CONTRACTS.md#g6`.
    """
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    class_field: Optional[str] = Field(default=None, alias="class")
    json_schema: Optional[Dict[str, Any]] = None
    # When True, force materialization of DataUnitProxyRef payloads for
    # validation (default: False — proxies validate at .get() time, not
    # at write time). Per G6 spec.
    validate_on_set: bool = False

    @model_validator(mode="after")
    def _validate_one_of(self) -> "SchemaRef":
        has_class = self.class_field is not None
        has_jsonschema = self.json_schema is not None
        if has_class == has_jsonschema:
            raise ValueError(
                "FAIL-FAST: SchemaRef requires EXACTLY ONE of 'class' "
                "(Pydantic model path) or 'json_schema' (inline schema)"
            )
        return self


def validate_payload_against_schema(
    payload: Any,
    schema_ref: SchemaRef,
    *,
    component_name: str,
    direction: str,
) -> None:
    """Validate ``payload`` against ``schema_ref``.

    Raises ``ComponentConfigurationError("FAIL-FAST: ...")`` on mismatch.

    For step OUTPUT direction: reserved fields (`errors`, `partial`) are
    stripped from the payload before validation against the user's declared
    schema, then validated independently against their fixed shapes.
    For step INPUT direction: reserved fields are not recognized.

    `direction` is one of "input" or "output" — used in the error message.
    """
    if not isinstance(payload, dict) and direction == "output":
        # Non-dict outputs cannot carry reserved fields; validate as-is.
        _validate_via_schema(payload, schema_ref, component_name, direction)
        return

    if direction == "output" and isinstance(payload, dict):
        # Strip reserved fields, validate them separately, then validate
        # the remainder against the user's schema.
        reserved = {k: payload[k] for k in RESERVED_OUTPUT_FIELDS if k in payload}
        non_reserved = {k: v for k, v in payload.items()
                        if k not in RESERVED_OUTPUT_FIELDS}

        if "errors" in reserved:
            _validate_reserved_errors(
                reserved["errors"], component_name)
        if "partial" in reserved and not isinstance(reserved["partial"], bool):
            raise ComponentConfigurationError(
                f"FAIL-FAST: step {component_name!r} output reserved field "
                f"'partial' must be bool, got {type(reserved['partial']).__name__}"
            )

        # If the only thing left is non-reserved keys, validate those.
        # If the entire payload was reserved (errors-only or partial-only
        # output with no typed payload), skip user-schema validation.
        if non_reserved:
            _validate_via_schema(non_reserved, schema_ref, component_name, direction)
        return

    # Input direction or non-dict payload — straight validation.
    _validate_via_schema(payload, schema_ref, component_name, direction)


def _validate_reserved_errors(value: Any, component_name: str) -> None:
    if not isinstance(value, list):
        raise ComponentConfigurationError(
            f"FAIL-FAST: step {component_name!r} output reserved field "
            f"'errors' must be a list, got {type(value).__name__}"
        )
    for i, item in enumerate(value):
        try:
            StepError._allow_direct_instantiation = True
            try:
                if isinstance(item, dict):
                    StepError(**item)
                elif not isinstance(item, StepError):
                    raise ComponentConfigurationError(
                        f"FAIL-FAST: step {component_name!r} output errors[{i}] "
                        f"must be a dict or StepError, got {type(item).__name__}"
                    )
            finally:
                StepError._allow_direct_instantiation = False
        except ComponentConfigurationError:
            raise
        except Exception as e:
            raise ComponentConfigurationError(
                f"FAIL-FAST: step {component_name!r} output errors[{i}] "
                f"failed StepError shape: {e}"
            ) from e


def _validate_via_schema(
    payload: Any,
    schema_ref: SchemaRef,
    component_name: str,
    direction: str,
) -> None:
    if schema_ref.class_field:
        try:
            cls = import_class_from_path(schema_ref.class_field)
        except Exception as e:
            raise ComponentConfigurationError(
                f"FAIL-FAST: step {component_name!r} {direction} schema class "
                f"{schema_ref.class_field!r} failed to import: {e}"
            ) from e
        if not (hasattr(cls, "model_validate") or hasattr(cls, "parse_obj")):
            raise ComponentConfigurationError(
                f"FAIL-FAST: step {component_name!r} {direction} schema class "
                f"{schema_ref.class_field!r} is not a Pydantic model "
                f"(no model_validate or parse_obj)"
            )
        try:
            if hasattr(cls, "model_validate"):
                cls.model_validate(payload)
            else:
                cls.parse_obj(payload)
        except Exception as e:
            raise ComponentConfigurationError(
                f"FAIL-FAST: step {component_name!r} {direction} failed schema: {e}"
            ) from e
    elif schema_ref.json_schema is not None:
        try:
            import jsonschema
        except ImportError as e:
            raise ComponentConfigurationError(
                f"FAIL-FAST: step {component_name!r} {direction} schema uses "
                f"json_schema form which requires the 'jsonschema' package. "
                f"Install with: pip install jsonschema. Original: {e}"
            ) from e
        try:
            jsonschema.validate(payload, schema_ref.json_schema)
        except jsonschema.ValidationError as e:
            raise ComponentConfigurationError(
                f"FAIL-FAST: step {component_name!r} {direction} failed schema: "
                f"{e.message} at path {list(e.path)}"
            ) from e


class StepConfig(ConfigBase):
    """
    Configuration for steps - INHERITS constructor prohibition.

    ❌ FORBIDDEN: StepConfig(name="test", class="...")
    ✅ REQUIRED: StepConfig.from_config('path/to/config.yml')
    """

    name: str
    description: str = ""
    executor_config: Optional[ExecutorConfig] = None
    input_configs: Dict[str, DataUnitConfig] = Field(default_factory=dict)
    output_config: Optional[DataUnitConfig] = None
    trigger_config: Optional[TriggerConfig] = None
    auto_initialize: bool = True
    debug_mode: bool = False
    enable_logging: bool = True
    log_data_transfers: bool = True
    log_executions: bool = True

    # NEW: Step-level tool configuration
    tools: Optional[Dict[str, Dict[str, Any]]] = Field(default_factory=dict)

    # Resolved components storage (populated by ConfigBase)
    resolved_tools: Dict[str, Any] = Field(
        default_factory=dict,
        description="Instantiated tool objects from configuration"
    )

    # Step-specific agent configurations (optional, step-dependent)
    extraction_agent: Optional[Any] = Field(
        default=None, description="Extraction agent for specialized processing")
    agents: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Multiple agents for complex processing steps")

    # EVENT-DRIVEN ARCHITECTURE: Step-level data units and triggers
    input_data_units: Optional[Dict[str, Union[Dict[str, Any], 'DataUnitBase']]] = Field(
        default_factory=dict)
    output_data_units: Optional[Dict[str, Union[Dict[str, Any], 'DataUnitBase']]] = Field(
        default_factory=dict)

    # Enhancement 2: Data extraction configuration
    data_extraction: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Configuration for automatic data unit wrapper extraction"
    )
    # ✅ UNIFIED RESOLUTION: Accept both dict configs and resolved trigger objects (workflows ARE steps)
    triggers: Optional[List[Union[Dict[str, Any], 'TriggerBase']]] = Field(
        default_factory=list)

    # G6 — typed input/output schemas (additive; default None preserves
    # historical no-validation behavior).
    # See `apecx-mcp-integration/docs/CONTRACTS.md#g6`.
    step_input_schema: Optional[Union[Dict[str, Any], 'SchemaRef']] = Field(
        default=None,
        description="G6 — declarative schema for the dict passed to process(). "
                    "Either {class: 'pkg.mod.PydanticModel'} OR "
                    "{json_schema: {...}}. When set, the framework validates "
                    "the input on every process() invocation."
    )
    step_output_schema: Optional[Union[Dict[str, Any], 'SchemaRef']] = Field(
        default=None,
        description="G6 — declarative schema for the value returned from "
                    "process(). Reserved fields 'errors' (list[StepError]) "
                    "and 'partial' (bool) are admitted alongside the typed "
                    "payload — see the gap proposal's escape valve."
    )

    # G12 — declarative resource envelope (additive; default None preserves
    # historical un-bounded behavior). Used by HPC bundle exporter + cost
    # gate (HITL §3.4) to project workflow resource needs before execution.
    # See `apecx-mcp-integration/docs/CONTRACTS.md#g12`.
    resource_envelope: Optional[Union[Dict[str, Any], 'ResourceEnvelope']] = Field(
        default=None,
        description="G12 — coarse resource projection: walltime_minutes, "
                    "cpu_cores, memory_gb, capability_tokens, cost_units. "
                    "Per-step values; the workflow aggregates via "
                    "Workflow.aggregate_resource_envelope().",
    )


class AgentStepConfig(StepConfig):
    """Configuration schema for AgentStep wrapper."""

    # Agent specification (required)
    agent_class: str = Field(
        description="Full class path to agent (e.g., 'nanobrain.library.agents.ConversationalAgent')")
    agent_config: Union[str, Dict[str, Any]] = Field(
        description="Agent configuration file path or inline config")

    # Data format bridging strategies
    input_extraction_strategy: str = Field(
        default="auto_detect",
        description="Strategy for extracting agent input: auto_detect, single_text_field, concatenate_all, custom_template"
    )
    output_formatting_strategy: str = Field(
        default="conversational_response",
        description="Strategy for formatting output: conversational_response, simple_response, preserve_input, custom_format"
    )

    # Input extraction configuration
    input_field_name: Optional[str] = Field(
        default=None, description="Specific input field name for single_text_field strategy")
    input_template: Optional[str] = Field(
        default=None, description="Template for custom_template strategy")

    # Output formatting configuration
    output_template: Optional[Dict[str, Any]] = Field(
        default=None, description="Template for custom_format strategy")

    # Conversation management (for chatbot workflows)
    enable_conversation_tracking: bool = Field(
        default=True, description="Enable conversation history tracking")
    max_conversation_length: int = Field(
        default=20, description="Maximum conversation history length")
    conversation_id_field: str = Field(
        default="conversation_id", description="Field name for conversation ID")

    # Agent lifecycle configuration
    auto_initialize_agent: bool = Field(
        default=True, description="Automatically initialize wrapped agent")
    agent_health_check_interval: int = Field(
        default=300, description="Agent health check interval in seconds")


class BaseStep(FromConfigBase, ABC):
    """
    Base Step Class - Event-Driven Data Processing and Workflow Building Blocks
    ==========================================================================

    The BaseStep class is the foundational component for creating data processing
    units within the NanoBrain framework. Steps represent discrete processing
    operations that transform data, execute computations, and coordinate with
    other components through event-driven architecture patterns.

    **Core Architecture:**
        Steps represent autonomous processing units that:

        * **Process Data**: Transform input data through configurable operations
        * **Manage State**: Track processing state and maintain operation history
        * **Coordinate Events**: Respond to triggers and emit events for workflow orchestration
        * **Handle Resources**: Manage computational resources and cleanup operations
        * **Integrate Components**: Seamlessly work with agents, tools, and other steps
        * **Execute Asynchronously**: Support concurrent and parallel processing patterns

    **Biological Analogy:**
        Like functional neural circuits that process specific types of information
        and pass results to other circuits, steps process specific operations and
        pass results to other steps. Neural circuits are specialized for particular
        functions (visual processing, motor control, etc.) and coordinate through
        complex signaling - exactly how steps specialize for specific operations
        and coordinate through data units, links, and triggers.

    **Data Processing Architecture:**

        **Input Management:**
        * Multiple input data units with type validation
        * Streaming data support for real-time processing
        * Batch processing capabilities for large datasets
        * Input dependency tracking and resolution

        **Processing Patterns:**
        * Synchronous processing for immediate results
        * Asynchronous processing for non-blocking operations
        * Parallel processing for computationally intensive tasks
        * Pipeline processing for multi-stage transformations

        **Output Generation:**
        * Multiple output data units with structured results
        * Result validation and quality assurance
        * Output routing to downstream components
        * Error handling with detailed diagnostics

        **State Management:**
        * Processing state tracking and persistence
        * Progress monitoring and reporting
        * Resource usage tracking and optimization
        * Recovery mechanisms for interrupted processing

    **Event-Driven Integration:**
        Steps operate within an event-driven architecture:

        * **Trigger Activation**: Steps respond to various trigger types
            - Data availability triggers when input data is ready
            - Timer triggers for scheduled processing
            - Manual triggers for user-initiated operations
            - Conditional triggers based on processing state

        * **Data Flow Coordination**: Steps coordinate through data units
            - Input data units provide processing inputs
            - Output data units store and share results
            - Shared data units enable cross-step communication
            - Data persistence for workflow continuity

        * **Link Integration**: Steps connect through configurable links
            - Direct links for immediate data transfer
            - Transform links for data format conversion
            - Conditional links for dynamic routing
            - Queue links for asynchronous processing

    **Framework Integration:**
        Steps seamlessly integrate with all framework components:

        * **Agent Integration**: Steps can embed agents for AI-driven processing
        * **Tool Utilization**: Steps can use tools for specialized operations
        * **Workflow Orchestration**: Steps compose complex multi-stage workflows
        * **Executor Support**: Steps run on various execution backends
        * **Configuration Management**: Complete YAML-driven configuration
        * **Monitoring Integration**: Comprehensive logging and performance tracking

    **Step Specializations:**
        The framework supports various step specializations:

        * **Step**: Standard processing step with configurable operations
        * **TransformStep**: Specialized for data transformation operations
        * **ParallelStep**: Parallel processing across multiple execution units
        * **BioinformaticsStep**: Computational biology specialized processing
        * **AgentStep**: Steps that integrate AI agents for processing
        * **ConversationalAgentStep**: Multi-turn conversation processing

    **Configuration Architecture:**
        Steps follow the framework's configuration-first design:

        ```yaml
        # Basic step configuration
        name: "data_processor"
        description: "Processes input data with validation"
        auto_initialize: true
        enable_logging: true

        # Input data units
        input_data_units:
          raw_data:
            class: "nanobrain.core.data_unit.DataUnitFile"
            config:
              file_path: "data/input.json"
              encoding: "utf-8"
          parameters:
            class: "nanobrain.core.data_unit.DataUnitMemory"
            config:
              initial_value: {"threshold": 0.5}

        # Output data units
        output_data_units:
          processed_data:
            class: "nanobrain.core.data_unit.DataUnitMemory"
            config:
              persistent: true
          results:
            class: "nanobrain.core.data_unit.DataUnitFile"
            config:
              file_path: "data/output.json"

        # Agent integration (optional)
        agent:
          class: "nanobrain.core.agent.ConversationalAgent"
          config: "config/processing_agent.yml"

        # Tools integration (optional)
        tools:
          validator:
            class: "nanobrain.library.tools.DataValidator"
            config: "config/validator.yml"

        # Triggers
        triggers:
          - class: "nanobrain.core.trigger.DataUpdatedTrigger"
            config:
              watch_data_units: ["raw_data", "parameters"]

        # Executor configuration
        executor:
          class: "nanobrain.core.executor.LocalExecutor"
          config: "config/local_executor.yml"
        ```

    **Usage Patterns:**

        **Basic Step Processing:**
        ```python
        from nanobrain.core import Step

        # Create step from configuration
        step = Step.from_config('config/data_processor.yml')

        # Execute step processing
        results = await step.execute()
        print(f"Processing complete: {results}")
        ```

        **Step with Agent Integration:**
        ```python
        # Step automatically creates and uses configured agent
        step = Step.from_config('config/ai_processor.yml')

        # Step processes data using embedded agent
        results = await step.execute()
        # Agent provides AI-driven processing within step
        ```

        **Multi-Step Workflow:**
        ```python
        # Steps coordinate through data units and triggers
        ingestion = Step.from_config('config/data_ingestion.yml')
        processing = Step.from_config('config/data_processing.yml')
        output = Step.from_config('config/data_output.yml')

        # Steps automatically coordinate through configured links
        await ingestion.execute()  # Triggers processing step
        # Processing automatically starts when ingestion completes
        # Output automatically starts when processing completes
        ```

    **Data Flow Patterns:**

        **Input Processing:**
        * Multiple data sources with different types and formats
        * Data validation and quality checks before processing
        * Dependency resolution and waiting for required inputs
        * Streaming data support for real-time processing

        **Processing Execution:**
        * Configurable processing logic through agents or tools
        * Error handling with retry mechanisms and fallbacks
        * Progress monitoring and intermediate result storage
        * Resource management and optimization

        **Output Management:**
        * Multiple output destinations with format conversion
        * Result validation and quality assurance
        * Output routing to downstream processing steps
        * Result persistence and retrieval mechanisms

    **Performance and Scalability:**

        **Execution Optimization:**
        * Asynchronous processing for responsive operations
        * Parallel execution for computationally intensive tasks
        * Streaming processing for large datasets
        * Resource pooling and reuse for efficiency

        **Monitoring and Metrics:**
        * Processing time tracking and optimization
        * Resource usage monitoring and alerting
        * Error rate tracking and analysis
        * Throughput measurement and optimization

        **Scalability Features:**
        * Horizontal scaling through parallel step instances
        * Vertical scaling through resource allocation
        * Load balancing across execution backends
        * Distributed processing via Parsl integration

    **Error Handling and Recovery:**
        Comprehensive error handling with graceful degradation:

        * **Input Validation**: Data format and content validation
        * **Processing Errors**: Exception handling with detailed diagnostics
        * **Resource Failures**: Automatic resource recovery and reallocation
        * **Network Issues**: Retry mechanisms with exponential backoff
        * **State Recovery**: Checkpoint and resume capabilities for long-running operations

    **Integration Patterns:**

        **Agent Integration:**
        * Embed agents for AI-driven processing logic
        * Agent tool calling within step processing
        * Multi-agent coordination for complex operations
        * Agent state management and conversation tracking

        **Tool Integration:**
        * Tool selection based on processing requirements
        * Parallel tool execution for complex operations
        * Tool result validation and integration
        * Tool performance monitoring and optimization

        **Workflow Integration:**
        * Step composition into complex workflows
        * Dynamic step routing based on processing results
        * Conditional step execution with branching logic
        * Loop and iteration patterns for repetitive processing

    **Step Lifecycle:**
        Steps follow a well-defined processing lifecycle:

        1. **Configuration Loading**: Parse and validate step configuration
        2. **Dependency Resolution**: Initialize data units, agents, and tools
        3. **Trigger Registration**: Setup event listeners and activation conditions
        4. **Initialization**: Prepare processing resources and state
        5. **Activation**: Respond to triggers and begin processing
        6. **Processing**: Execute configured processing logic
        7. **Output Generation**: Produce results and update output data units
        8. **Cleanup**: Release resources and update processing state

    **Advanced Features:**

        **Dynamic Configuration:**
        * Runtime configuration updates and reloading
        * Parameter tuning based on processing performance
        * Adaptive resource allocation based on workload
        * Dynamic tool selection based on data characteristics

        **State Persistence:**
        * Processing checkpoint creation and restoration
        * State synchronization across distributed execution
        * Result caching for performance optimization
        * Audit trail maintenance for debugging and compliance

        **Quality Assurance:**
        * Input data validation and sanitization
        * Processing result verification and validation
        * Performance benchmarking and optimization
        * Error detection and automated correction

    Attributes:
        name (str): Step identifier for logging and workflow coordination
        description (str): Human-readable step description and purpose
        input_data_units (Dict[str, DataUnitBase]): Input data sources and containers
        output_data_units (Dict[str, DataUnitBase]): Output data destinations and storage
        triggers (List[TriggerBase]): Event triggers that activate step processing
        agent (Agent, optional): Embedded agent for AI-driven processing
        tools (Dict[str, Tool], optional): Available tools for processing operations
        executor (ExecutorBase): Execution backend for step operations
        processing_state (Dict): Current processing state and progress information
        performance_metrics (Dict): Real-time performance and usage metrics

    Note:
        This is an abstract base class that cannot be instantiated directly.
        Use concrete implementations like Step or TransformStep. All steps
        must be created using the from_config pattern with proper configuration
        files following the framework's event-driven architecture patterns.

    Warning:
        Steps may consume significant computational resources depending on
        processing complexity. Monitor resource usage and implement appropriate
        limits and cleanup mechanisms. Ensure proper error handling for
        long-running or resource-intensive operations.

    See Also:
        * :class:`Step`: Standard step implementation with configurable processing
        * :class:`TransformStep`: Specialized step for data transformation
        * :class:`StepConfig`: Step configuration schema and validation
        * :class:`DataUnitBase`: Data container and management system
        * :class:`TriggerBase`: Event trigger system for step activation
        * :class:`LinkBase`: Step connectivity and data flow management
        * :mod:`nanobrain.library.infrastructure.steps`: Specialized step implementations
    """

    COMPONENT_TYPE = "base_step"
    REQUIRED_CONFIG_FIELDS = ['name']
    OPTIONAL_CONFIG_FIELDS = {
        'description': '',
        'auto_initialize': True,
        'debug_mode': False,
        'enable_logging': True,
        'log_data_transfers': True,
        'log_executions': True
    }

    @classmethod
    def _get_config_class(cls):
        """UNIFIED PATTERN: Return StepConfig - ONLY method that differs from other components"""
        return StepConfig

    # Now inherits unified from_config implementation from FromConfigBase

    @classmethod
    def extract_component_config(cls, config: StepConfig) -> Dict[str, Any]:
        """
        Extract BaseStep-specific configuration including resolved objects

        ✅ FRAMEWORK COMPLIANCE: Includes resolved objects from class+config patterns
        so steps can access instantiated agents, tools, and other components.
        """
        # Use model_dump() to get all field values, including resolved objects
        all_config = config.model_dump()

        # Start with core configuration fields
        component_config = {
            'name': config.name,
            'description': getattr(config, 'description', ''),
            'auto_initialize': getattr(config, 'auto_initialize', True),
            'debug_mode': getattr(config, 'debug_mode', False),
            'enable_logging': getattr(config, 'enable_logging', True),
            'log_data_transfers': getattr(config, 'log_data_transfers', True),
            'log_executions': getattr(config, 'log_executions', True),
        }

        # Add all other fields from model_dump() for resolved objects
        # This captures resolved agents, tools, etc. from class+config patterns
        for field_name, field_value in all_config.items():
            if field_name not in component_config:
                component_config[field_name] = field_value

        # Re-attach the PARSED executor_config object. model_dump() above
        # flattens it to a plain dict (executor_type -> string, nested blocks
        # -> dicts); resolve_dependencies needs the live ExecutorConfig so it
        # can dispatch on the typed `executor_type` enum and hand the object
        # straight to the target executor's from_config. See
        # BaseStep.resolve_dependencies precedence #2.
        component_config['executor_config'] = getattr(config, 'executor_config', None)

        return component_config

    @classmethod
    def resolve_dependencies(cls, component_config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Resolve BaseStep dependencies with step-level executor priority

        Priority order (highest first):
        1. A pre-built ``executor`` object in component_config (programmatic
           override) — used as-is.
        2. The step's own ``executor_config`` (a parsed ExecutorConfig from
           the step YAML) — built here via the shared
           ``build_executor_from_config`` dispatch (local / thread / process /
           parsl / globus_compute). FAIL-LOUD on an unknown executor_type or
           an invalid executor-type-specific config block.
        3. Workflow-level executor (passed via kwargs).
        4. Default LocalExecutor (fallback).
        """
        # Priority 1: a pre-built executor object on the component_config.
        step_executor = component_config.get('executor')
        if step_executor is not None:
            # Step has its own pre-built executor object - use it.
            return {
                'executor': step_executor
            }

        # Priority 2: the step's own executor_config block (parsed
        # ExecutorConfig). When present, build + bind that executor using the
        # SAME dispatch the workflow-level factory uses. This is the wiring
        # that makes `executor_config:` in a step YAML actually take effect.
        step_executor_config = component_config.get('executor_config')
        if step_executor_config is not None:
            from .executor import ExecutorConfig, build_executor_from_config
            # `executor_config` may arrive as a parsed ExecutorConfig (the
            # normal path, re-attached by extract_component_config) or, for
            # robustness, as a dict. build_executor_from_config requires the
            # parsed object, so coerce a dict via a temp YAML file (the same
            # file-only-config constraint ExecutorConfig enforces).
            if not isinstance(step_executor_config, ExecutorConfig):
                import tempfile
                import os
                import yaml
                with tempfile.NamedTemporaryFile(
                        mode='w', suffix='.yml', delete=False) as f:
                    yaml.dump(step_executor_config, f)
                    _tmp_path = f.name
                try:
                    step_executor_config = ExecutorConfig.from_config(_tmp_path)
                finally:
                    os.unlink(_tmp_path)
            executor = build_executor_from_config(step_executor_config)
            return {
                'executor': executor
            }

        # Priority 3: workflow-level executor.
        executor = kwargs.get('executor')
        if executor is None:
            # Import here to avoid circular imports
            from .executor import ExecutorConfig

            # Create default LocalExecutor using proper framework pattern
            # ExecutorConfig doesn't support inline dict config, so use direct instantiation
            try:
                ExecutorConfig._allow_direct_instantiation = True
                default_config = ExecutorConfig(executor_type="local")
            finally:
                ExecutorConfig._allow_direct_instantiation = False

            executor = LocalExecutor.from_config(default_config)

        return {
            'executor': executor
        }

    def _init_from_config(self, config: StepConfig, component_config: Dict[str, Any],
                          dependencies: Dict[str, Any]) -> None:
        """Initialize BaseStep with proper sequence"""
        # ✅ FRAMEWORK COMPLIANT - Call parent initialization first to set up enable_logging and other base attributes
        super()._init_from_config(config, component_config, dependencies)

        # StepBase-specific initialization (parent already sets self.config, self.name, self.description)
        # Override with step-specific logger that includes debug_mode
        self.nb_logger = get_logger(
            f"step.{self.name}", category="steps", debug_mode=component_config['debug_mode'])
        self.nb_logger.info(
            f"Initializing step: {self.name}", step_name=self.name, config=config.model_dump())

        # FAIL-FAST: Validate step implementation for common errors
        self._validate_step_implementation()

        # Executor for running the step
        self.executor = dependencies['executor']

        # Data management (legacy step data units)
        # NOTE: input_data_units is now a property that maps to step_input_data_units
        # Do not set self.input_data_units as instance attribute - it overrides the property
        self.output_data_unit: Optional[DataUnitBase] = None
        self.links: Dict[str, LinkBase] = {}

        # Trigger for activation (legacy)
        self.trigger: Optional[TriggerBase] = None

        # Initialize tool registry
        self.tools: Dict[str, Any] = {}

        # ✅ PHASE 1: Initialize step-level data unit containers
        self.step_input_data_units = {}
        self.step_output_data_units = {}

        # ✅ PHASE 1: Create step-level data units via from_config
        self._create_step_data_units(config)

        # ✅ ARCHITECTURAL FIX: Store trigger configurations for later processing in initialize()
        # Triggers will be created AFTER data units exist, ensuring proper resolution
        self.step_trigger_configs = getattr(config, 'triggers', [])
        self.step_triggers = {}  # Will be populated during resolution phase

        # Load tools from external configuration files
        self.step_tools = {}

        # ENHANCED: Check if tools were already resolved by ConfigBase._resolve_nested_objects()
        resolved_tools = getattr(config, 'resolved_tools', {})
        if resolved_tools:
            # Use already-instantiated tools from enhanced class+config resolution
            self.step_tools.update(resolved_tools)
            for tool_name in resolved_tools:
                self.nb_logger.info(f"Using resolved tool: {tool_name}")
        else:
            # Fallback: Load tools from external configuration files (legacy)
            tools_config = getattr(config, 'tools', {})
            for tool_name, tool_ref in tools_config.items():
                config_file = tool_ref.get('config_file')
                if config_file:
                    tool_config = self._load_config_file(config_file)
                    # Use direct from_config pattern instead of ComponentFactory
                    tool_class_path = tool_config['class']
                    module_path, class_name = tool_class_path.rsplit('.', 1)
                    import importlib
                    module = importlib.import_module(module_path)
                    tool_class = getattr(module, class_name)
                    tool = tool_class.from_config(tool_config)
                    self.step_tools[tool_name] = tool
                    self.nb_logger.info(f"Loaded step tool: {tool_name}")

        # Skip legacy tool loading if tools were already resolved by enhanced system
        # Load tools from legacy configuration only if no enhanced tools were found
        if not resolved_tools and hasattr(config, 'tools') and config.tools:
            self._load_tools_from_config(config.tools)

        # State management
        self._is_initialized = False
        self._execution_count = 0
        self._error_count = 0
        self._last_result = None

        # Performance tracking
        self._start_time = time.time()
        self._last_activity_time = time.time()
        self._total_processing_time = 0.0

    # BaseStep inherits FromConfigBase.__init__ which prevents direct instantiation

    def _validate_step_implementation(self) -> None:
        """
        FAIL-FAST: Validate step implementation for common errors.

        This catches implementation bugs at initialization time instead of
        runtime, providing immediate feedback to developers.
        """
        import inspect
        from .component_base import ComponentConfigurationError

        try:
            # Get step source code for analysis
            step_source = inspect.getsource(self.__class__)

            # Check for logger attribute misuse (the bug that caused 3 hours of debugging)
            if 'self.logger' in step_source:
                # Find line numbers where self.logger is used
                lines_with_logger = []
                for i, line in enumerate(step_source.split('\n'), 1):
                    if 'self.logger' in line and 'self.nb_logger' not in line:
                        lines_with_logger.append(i)

                if lines_with_logger:
                    raise ComponentConfigurationError(
                        f"FAIL-FAST: Step {self.name} ({self.__class__.__name__}) uses 'self.logger' "
                        f"on lines {lines_with_logger}. Use 'self.nb_logger' instead for framework compliance. "
                        f"This error prevents workflow execution to avoid AttributeError at runtime."
                    )

            # Check for required process method
            if not hasattr(self, 'process'):
                raise ComponentConfigurationError(
                    f"FAIL-FAST: Step {self.name} ({self.__class__.__name__}) missing required 'process' method. "
                    f"Add 'async def process(self, input_data, **kwargs)' to your step class."
                )

            # Check if process method is async
            if hasattr(self, 'process') and not inspect.iscoroutinefunction(self.process):
                raise ComponentConfigurationError(
                    f"FAIL-FAST: Step {self.name} ({self.__class__.__name__}).process() must be async. "
                    f"Change 'def process' to 'async def process'."
                )

            # ADDITIONAL VALIDATION RULES

            # Check for missing _init_from_config method
            if not hasattr(self.__class__, '_init_from_config'):
                raise ComponentConfigurationError(
                    f"FAIL-FAST: Step {self.name} ({self.__class__.__name__}) missing '_init_from_config' method. "
                    f"Add 'def _init_from_config(self, config, component_config, dependencies)' method."
                )

            # Check for proper inheritance
            if not hasattr(self, 'nb_logger'):
                raise ComponentConfigurationError(
                    f"FAIL-FAST: Step {self.name} ({self.__class__.__name__}) missing 'nb_logger' attribute. "
                    f"Ensure step inherits from Step or BaseStep and calls super()._init_from_config()."
                )

            # Check if nb_logger has required methods (framework compatibility)
            if not hasattr(self.nb_logger, 'info'):
                raise ComponentConfigurationError(
                    f"FAIL-FAST: Step {self.name} nb_logger is not a proper NanoBrainLogger. "
                    f"Use get_logger() with category='steps' parameter."
                )

            # Check for common async/await issues in process method
            if hasattr(self, 'process'):
                process_source = inspect.getsource(self.process)
                if 'await ' not in process_source and 'async def' in process_source:
                    self.nb_logger.warning(
                        f"VALIDATION WARNING: Step {self.name}.process() is async but contains no 'await' statements. "
                        f"This might indicate missing async calls or unnecessary async declaration."
                    )

        except ComponentConfigurationError:
            # Re-raise configuration errors
            raise
        except Exception as e:
            # If validation itself fails, log but don't block initialization
            self.nb_logger.warning(f"Step validation failed for {self.name}: {e}")
            # Don't raise - validation failure shouldn't block valid steps













    def _create_step_data_units(self, config: StepConfig) -> None:
        """
        ✅ FRAMEWORK COMPLIANT: Create step data units via from_config
        Phase 1: Component creation without binding/resolution
        """
        # Create input data units
        input_configs = getattr(config, 'input_data_units', {})
        for unit_name, unit_config in input_configs.items():

            # ✅ from_config COMPLIANCE: All creation via proper pattern
            if isinstance(unit_config, DataUnitBase):
                # Already instantiated DataUnit object from class+config resolution
                data_unit = unit_config
            elif isinstance(unit_config, dict) and 'class' in unit_config:
                # Dictionary configuration with class field - determine class and call its from_config
                data_unit = self._create_data_unit_from_class_config(
                    unit_config)
            elif isinstance(unit_config, dict):
                # Dictionary configuration without class field - use default
                class_path = unit_config.get(
                    'class', 'nanobrain.core.data_unit.DataUnitMemory')
                enhanced_config = unit_config.copy()
                enhanced_config['class'] = class_path
                data_unit = self._create_data_unit_from_class_config(
                    enhanced_config)
            else:
                # DataUnitConfig object - use proper from_config pattern
                data_unit = self._create_data_unit(unit_config)

            self.step_input_data_units[unit_name] = data_unit
            self.nb_logger.debug(f"Created step input data unit: {unit_name}")

        # Create output data units
        output_configs = getattr(config, 'output_data_units', {})
        for unit_name, unit_config in output_configs.items():

            # ✅ from_config COMPLIANCE: All creation via proper pattern
            if isinstance(unit_config, DataUnitBase):
                # Already instantiated DataUnit object from class+config resolution
                data_unit = unit_config
            elif isinstance(unit_config, dict) and 'class' in unit_config:
                # Dictionary configuration with class field - determine class and call its from_config
                data_unit = self._create_data_unit_from_class_config(
                    unit_config)
            elif isinstance(unit_config, dict):
                # Dictionary configuration without class field - use default
                class_path = unit_config.get(
                    'class', 'nanobrain.core.data_unit.DataUnitMemory')
                enhanced_config = unit_config.copy()
                enhanced_config['class'] = class_path
                data_unit = self._create_data_unit_from_class_config(
                    enhanced_config)
            else:
                # DataUnitConfig object - use proper from_config pattern
                data_unit = self._create_data_unit(unit_config)

            self.step_output_data_units[unit_name] = data_unit
            self.nb_logger.debug(f"Created step output data unit: {unit_name}")

    def _create_data_unit_from_class_config(self, unit_config: Dict[str, Any]) -> DataUnitBase:
        """
        ✅ FRAMEWORK COMPLIANT: Create data unit from class+config dictionary
        """
        class_path = unit_config.get('class')
        if not class_path:
            raise ValueError("Data unit configuration missing 'class' field")

        # Import the DataUnit class and call its from_config method directly
        module_path, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        data_unit_class = getattr(module, class_name)

        # DataUnit classes support inline dict configs
        # Pass parent scope for proper component registration
        return data_unit_class.from_config(unit_config, parent_scope=self.name)

    async def initialize(self) -> None:
        """
        ✅ THREE-PHASE INITIALIZATION: Proper sequence for trigger resolution
        Phase 1: Component creation (already done in _init_from_config)
        Phase 2: Data unit initialization 
        Phase 3: Trigger resolution and binding
        """
        if self._is_initialized:
            self.nb_logger.debug(f"Step {self.name} already initialized")
            return

        # FAIL-FAST: Check if logger has async_execution_context method
        if hasattr(self.nb_logger, 'async_execution_context'):
            async with self.nb_logger.async_execution_context(
                OperationType.STEP_EXECUTE,
                f"{self.name}.initialize"
            ) as context:
                await self._initialize_step_internal()
        else:
            # Fallback for loggers without async context support
            self.nb_logger.info(f"Initializing step {self.name} (fallback mode)")
            await self._initialize_step_internal()

    async def _initialize_step_internal(self) -> None:
        """Internal step initialization logic"""
        # Phase 2: Initialize data units (make them ready for binding)
        await self._initialize_step_data_units()

        # Phase 3: Resolve and bind triggers to data units
        await self._resolve_and_bind_step_triggers()

        # Initialize other components (executor, legacy components)
        await self._initialize_other_components()

        self._is_initialized = True

        # Log initialization completion
        self.nb_logger.info(
            f"Step {self.name} initialized successfully - "
            f"inputs: {len(self.input_data_units)}, "
            f"step_inputs: {len(self.step_input_data_units)}, "
            f"step_outputs: {len(self.step_output_data_units)}, "
            f"triggers: {len(self.step_triggers)}, "
            f"has_output: {self.output_data_unit is not None}, "
            f"has_trigger: {self.trigger is not None}"
        )

    async def _initialize_step_data_units(self) -> None:
        """
        ✅ PHASE 2: Initialize data units to make them ready for trigger binding
        """
        # Initialize step input data units
        for unit_name, data_unit in self.step_input_data_units.items():
            await data_unit.initialize()
            self.nb_logger.debug(
                f"Initialized step input data unit: {unit_name}")

        # Initialize step output data units
        for unit_name, data_unit in self.step_output_data_units.items():
            await data_unit.initialize()
            self.nb_logger.debug(
                f"Initialized step output data unit: {unit_name}")

        # NEW: Automatic trigger registration
        await self._register_automatic_input_triggers()
        await self._register_automatic_output_triggers()

        self.nb_logger.info(
            "✅ Phase 2 Complete: All step data units initialized with automatic triggers")

    async def _resolve_and_bind_step_triggers(self) -> None:
        """
        ✅ PHASE 3: Resolve trigger data unit references and bind to actual objects

        ARCHITECTURAL COMPLIANCE:
        - Only resolves within step scope (no cross-step references)
        - Uses actual data unit objects created in previous phases
        - Maintains from_config pattern for trigger creation
        """
        step_context = self._create_step_resolution_context()

        for trigger_config in self.step_trigger_configs:
            # ✅ FRAMEWORK COMPLIANT: Create trigger via from_config with step context
            trigger_instance = await self._create_and_resolve_step_trigger(
                trigger_config, step_context
            )

            if trigger_instance:
                # Bind trigger to step execution
                trigger_instance.bind_action(self._execute_on_trigger)

                # Store resolved trigger
                trigger_id = getattr(
                    trigger_instance, 'trigger_id', f'trigger_{len(self.step_triggers)}')
                self.step_triggers[trigger_id] = trigger_instance

                # Start monitoring
                await trigger_instance.start_monitoring()

                self.nb_logger.info(
                    f"✅ Resolved and bound step trigger: {trigger_id}")

        # ✅ AUTOMATIC OUTPUT TRIGGERS: Create triggers for all output data units
        # This enables event-driven link activation when step writes output data
        await self._create_automatic_output_triggers(step_context)

        self.nb_logger.info(
            f"✅ Phase 3 Complete: All {len(self.step_triggers)} triggers resolved and monitoring")

    async def _create_automatic_output_triggers(self, step_context: Dict[str, Any]) -> None:
        """
        ✅ AUTOMATIC FRAMEWORK FEATURE: Create triggers for all output data units

        This enables event-driven link activation when steps write to output data units.
        When a step writes data to an output data unit, the trigger fires and activates
        any associated links for automatic data transfer.

        ARCHITECTURAL COMPLIANCE:
        - Creates triggers via from_config pattern
        - Uses DataUnitChangeTrigger for immediate event response
        - Does not bind to step execution (these are for link activation only)
        """

        step_output_units = step_context.get('step_output_data_units', {})

        for unit_name, data_unit in step_output_units.items():
            # Create trigger config for output data unit
            output_trigger_config = {
                'trigger_type': 'data_updated',
                'data_unit': unit_name,
                'event_type': 'set',
                'description': f'Automatic output trigger for {unit_name} - activates links when data is written',
                'enable_logging': True
            }

            try:
                # Create trigger via from_config pattern
                trigger_instance = await self._create_and_resolve_step_trigger(
                    output_trigger_config, step_context
                )

                if trigger_instance:
                    # Store with output-specific ID
                    trigger_id = f"output_{unit_name}_trigger"
                    self.step_triggers[trigger_id] = trigger_instance

                    # Start monitoring for link activation
                    await trigger_instance.start_monitoring()

                    self.nb_logger.info(
                        f"✅ Created automatic output trigger: {trigger_id} for data unit {unit_name}")

            except Exception as e:
                self.nb_logger.error(
                    f"❌ Failed to create automatic output trigger for {unit_name}: {e}", exc_info=True)

        if step_output_units:
            self.nb_logger.info(
                f"✅ Created {len(step_output_units)} automatic output triggers for event-driven link activation")

    async def _initialize_other_components(self) -> None:
        """
        ✅ LEGACY SUPPORT: Initialize other components (executor, legacy data units, triggers)
        """
        # Initialize executor
        self.nb_logger.debug(f"Initializing executor for step {self.name}")
        await self.executor.initialize()

        # Initialize legacy input data units (if any)
        self.nb_logger.debug(
            f"Initializing {len(self.config.input_configs)} legacy input data units")
        for input_id, input_config in self.config.input_configs.items():
            data_unit = self._create_data_unit(input_config)
            await data_unit.initialize()
            # Store in step_input_data_units (accessible via input_data_units property)
            self.step_input_data_units[input_id] = data_unit
            self.nb_logger.debug(
                f"Initialized legacy input data unit: {input_id}")

        # Initialize legacy output data unit (if any)
        if self.config.output_config:
            self.nb_logger.debug("Initializing legacy output data unit")
            self.output_data_unit = self._create_data_unit(
                self.config.output_config)
            await self.output_data_unit.initialize()

        # Initialize legacy trigger (if any)
        if self.config.trigger_config:
            self.nb_logger.debug("Initializing legacy trigger")
            self.trigger = self._create_trigger(self.config.trigger_config)

            # Set up trigger callback
            await self.trigger.add_callback(self._on_trigger_activated)

            # Start monitoring
            await self.trigger.start_monitoring()

    def _create_step_resolution_context(self) -> Dict[str, Any]:
        """
        ✅ STEP-SCOPE ONLY: Create resolution context with step-local data units

        ARCHITECTURAL COMPLIANCE:
        - Only includes data units within this step scope
        - No workflow-level or cross-step data units
        - Ensures trigger isolation within step boundaries
        """
        return {
            'step_input_data_units': self.step_input_data_units,
            'step_output_data_units': self.step_output_data_units,
            'step_name': self.name,
            'step_scope_only': True  # Enforce step isolation
        }

    async def _create_and_resolve_step_trigger(self, trigger_config: Any,
                                               step_context: Dict[str, Any]) -> Optional[TriggerBase]:
        """
        ✅ FRAMEWORK COMPLIANT: Create trigger via from_config with step-scope resolution
        """
        try:
            # G118 (2026-05-18) — extract YAML-declared ``data_units``
            # list from the trigger config BEFORE construction. The
            # framework's TriggerBase.resolve_dependencies reads
            # ``kwargs.get('data_units', [])`` only — so the config-side
            # list must be lifted into kwargs OR the AllDataReceivedTrigger
            # ends up with an empty data_units list and never fires
            # (the dominant YAML-AllDataReceivedTrigger silent-failure
            # before this fix). Resolution from string → DataUnit
            # happens after construction below.
            yaml_data_units: Optional[List[str]] = None
            if isinstance(trigger_config, dict):
                raw_units = trigger_config.get('data_units')
                if isinstance(raw_units, list) and raw_units:
                    yaml_data_units = list(raw_units)
            elif hasattr(trigger_config, 'data_units'):
                raw_units = getattr(trigger_config, 'data_units', None)
                if isinstance(raw_units, list) and raw_units:
                    yaml_data_units = list(raw_units)

            # Phase 3A: Create trigger instance via from_config
            if hasattr(trigger_config, '__class__') and hasattr(trigger_config, 'bind_action'):
                # Already instantiated trigger from ConfigBase resolution
                trigger_instance = trigger_config
            else:
                # Create trigger via from_config pattern
                trigger_class = self._get_trigger_class(trigger_config)
                # G118: pass the lifted data_units as a kwarg so
                # resolve_dependencies picks it up. Strings are still
                # strings here — we resolve to DataUnit instances below.
                from_config_kwargs: Dict[str, Any] = {
                    "step_context": step_context,
                }
                if yaml_data_units is not None:
                    from_config_kwargs["data_units"] = yaml_data_units
                trigger_instance = trigger_class.from_config(
                    trigger_config,
                    **from_config_kwargs,
                )

            # Phase 3B: Resolve data unit references within step scope
            if hasattr(trigger_instance, 'data_unit') and isinstance(trigger_instance.data_unit, str):
                resolved_data_unit = self._resolve_step_data_unit_reference(
                    trigger_instance.data_unit, step_context
                )

                if resolved_data_unit:
                    trigger_instance.data_unit = resolved_data_unit
                    self.nb_logger.debug(
                        f"✅ Resolved trigger data unit: {trigger_instance.data_unit.name}")
                else:
                    raise ValueError(
                        f"❌ Data unit '{trigger_instance.data_unit}' not found in step scope")

            # G118 (2026-05-18): AllDataReceivedTrigger uses ``data_units``
            # (LIST), not ``data_unit``. Resolve every string in the list
            # to the corresponding DataUnit instance from step scope.
            # Without this, a YAML-authored AllDataReceivedTrigger ends up
            # with an empty resolved list (the original kwargs.get fallback)
            # and the trigger NEVER FIRES — silent failure. Authors had
            # to work around it by using a single-input DataUnitChangeTrigger
            # on the last-arriving input, which is less expressive.
            data_units_attr = getattr(trigger_instance, 'data_units', None)
            if isinstance(data_units_attr, list) and data_units_attr:
                resolved_units: List[Any] = []
                for ref in data_units_attr:
                    if isinstance(ref, str):
                        resolved = self._resolve_step_data_unit_reference(
                            ref, step_context
                        )
                        if resolved is None:
                            raise ValueError(
                                f"❌ AllDataReceivedTrigger data unit "
                                f"{ref!r} not found in step scope for "
                                f"step {self.name!r}"
                            )
                        resolved_units.append(resolved)
                    else:
                        # Already an instance (rare; pass through).
                        resolved_units.append(ref)
                trigger_instance.data_units = resolved_units
                self.nb_logger.debug(
                    f"✅ Resolved {len(resolved_units)} trigger data_units "
                    f"({', '.join(u.name for u in resolved_units)})"
                )

            return trigger_instance

        except Exception as e:
            self.nb_logger.error(
                f"❌ Failed to create/resolve step trigger: {e}", exc_info=True)
            return None

    def _resolve_step_data_unit_reference(self, data_unit_ref: str,
                                          step_context: Dict[str, Any]) -> Optional[DataUnitBase]:
        """
        ✅ STEP-SCOPE ONLY: Resolve data unit reference within step boundaries

        ARCHITECTURAL COMPLIANCE:
        - Only searches within step input/output data units
        - No cross-step or workflow-level resolution
        - Returns None if not found in step scope (proper behavior)
        """
        # Check step input data units
        step_input_units = step_context.get('step_input_data_units', {})
        if data_unit_ref in step_input_units:
            return step_input_units[data_unit_ref]

        # Check step output data units
        step_output_units = step_context.get('step_output_data_units', {})
        if data_unit_ref in step_output_units:
            return step_output_units[data_unit_ref]

        # Not found in step scope - this is expected behavior for cross-scope references
        available_units = list(step_input_units.keys()) + \
            list(step_output_units.keys())
        self.nb_logger.warning(f"Data unit '{data_unit_ref}' not found in step scope. "
                               f"Available: {available_units}")
        return None

    def _get_trigger_class(self, trigger_config: Any):
        """
        ✅ FRAMEWORK COMPLIANT: Get trigger class for from_config creation

        G118-companion (2026-05-18): also recognize the ``class:`` field
        (dotted class path) when ``trigger_type:`` is absent. Previously
        the framework silently fell back to ``DataUnitChangeTrigger``
        when authors used the more-Pythonic ``class:`` form without
        ``trigger_type:`` — producing an instance of the wrong trigger
        type that never fires correctly. The dotted-path lookup keeps
        the legacy ``trigger_type`` field working AND fixes the
        ``class:``-only YAML pattern.
        """
        # First-priority: ``class:`` field (dotted path). Lets YAML
        # authors use the canonical ``class:`` form without forcing
        # ``trigger_type:`` to be specified.
        class_path: Optional[str] = None
        if isinstance(trigger_config, dict):
            class_path = trigger_config.get('class')
        elif hasattr(trigger_config, 'class_'):
            class_path = getattr(trigger_config, 'class_', None)
        if isinstance(class_path, str) and '.' in class_path:
            try:
                module_path, class_name = class_path.rsplit('.', 1)
                import importlib

                module = importlib.import_module(module_path)
                resolved = getattr(module, class_name, None)
                if resolved is not None:
                    return resolved
            except Exception:
                # Fall through to the legacy trigger_type lookup.
                pass

        # Determine trigger type from configuration
        trigger_type = None
        if isinstance(trigger_config, dict):
            trigger_type = trigger_config.get(
                'trigger_type', 'data_updated')
        elif hasattr(trigger_config, 'trigger_type'):
            trigger_type = trigger_config.trigger_type
        else:
            trigger_type = 'data_updated'

        # Map trigger types to classes - all event-driven
        if trigger_type in ['data_updated', 'data_unit_change']:
            # Use DataUnitChangeTrigger for pure event-driven architecture
            from .trigger import DataUnitChangeTrigger
            return DataUnitChangeTrigger
        elif trigger_type == 'all_data_received':
            from .trigger import AllDataReceivedTrigger
            return AllDataReceivedTrigger
        elif trigger_type == 'timer':
            from .trigger import TimerTrigger
            return TimerTrigger
        elif trigger_type == 'manual':
            from .trigger import ManualTrigger
            return ManualTrigger
        else:
            # Default to event-driven trigger
            from .trigger import DataUnitChangeTrigger
            return DataUnitChangeTrigger

    async def shutdown(self) -> None:
        """Shutdown the step and cleanup resources."""
        async with self.nb_logger.async_execution_context(
            OperationType.STEP_EXECUTE,
            f"{self.name}.shutdown"
        ) as context:
            # Log final statistics
            uptime_seconds = time.time() - self._start_time
            self.nb_logger.info(f"Step {self.name} shutting down",
                                uptime_seconds=uptime_seconds,
                                execution_count=self._execution_count,
                                error_count=self._error_count,
                                total_processing_time=self._total_processing_time)

            # Shutdown trigger
            if self.trigger:
                await self.trigger.stop_monitoring()

            # EVENT-DRIVEN ARCHITECTURE: Shutdown step-level triggers
            for trigger_id, trigger in self.step_triggers.items():
                await trigger.stop_monitoring()
                self.nb_logger.debug(f"Stopped step trigger: {trigger_id}")

            # Shutdown data units
            for data_unit in self.input_data_units.values():
                await data_unit.shutdown()

            if self.output_data_unit:
                await self.output_data_unit.shutdown()

            # Shutdown executor
            await self.executor.shutdown()

            self._is_initialized = False
            context.metadata['final_stats'] = {
                'uptime_seconds': uptime_seconds,
                'execution_count': self._execution_count,
                'error_count': self._error_count,
                'total_processing_time': self._total_processing_time
            }

    def _create_data_unit(self, config: DataUnitConfig) -> DataUnitBase:
        """Create a data unit from configuration using proper from_config pattern."""
        # Import here to avoid circular imports
        class_path = config.class_path

        # Import the DataUnit class and call its from_config method directly
        module_path, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        data_unit_class = getattr(module, class_name)

        # Use the DataUnit class's from_config method
        return data_unit_class.from_config(config)

    def _create_trigger(self, config: TriggerConfig) -> TriggerBase:
        """Create a trigger from configuration using proper from_config pattern."""
        # Import here to avoid circular imports
        trigger_type = config.trigger_type

        # Normalize legacy trigger type names
        if trigger_type == 'data_unit_change':
            trigger_type = 'data_updated'

        # Import the appropriate trigger class and call its from_config method
        if trigger_type == 'data_updated':
            # Use DataUnitChangeTrigger for event-driven architecture
            from .trigger import DataUnitChangeTrigger
            return DataUnitChangeTrigger.from_config(config)
        elif trigger_type == 'all_data_received':
            from .trigger import AllDataReceivedTrigger
            return AllDataReceivedTrigger.from_config(config)
        elif trigger_type == 'timer':
            from .trigger import TimerTrigger
            return TimerTrigger.from_config(config)
        elif trigger_type == 'manual':
            from .trigger import ManualTrigger
            return ManualTrigger.from_config(config)
        else:
            # Default to event-driven trigger for pure event architecture
            from .trigger import DataUnitChangeTrigger
            return DataUnitChangeTrigger.from_config(config)

    async def _execute_on_trigger(self, trigger_event: Dict[str, Any]) -> None:
        """Execute step when triggered by data unit change (EVENT-DRIVEN EXECUTION)"""
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"🔥 BRUTAL TRUTH: _execute_on_trigger called for step {self.name}")
        try:
            # G118 (2026-05-18): tolerate trigger event shapes that
            # don't carry ``trigger_id`` (notably AllDataReceivedTrigger,
            # which fires with a raw ``data_dict``). The step doesn't
            # use the trigger_id beyond logging — gracefully degrade.
            trigger_id_label = (
                trigger_event.get('trigger_id', self.name + '_anon_trigger')
                if isinstance(trigger_event, dict)
                else f"{self.name}_unknown_trigger"
            )
            logger.info(f"🔥 Step {self.name} triggered by {trigger_id_label}")

            # Get input data from triggered data unit
            logger.info(f"🔥 BRUTAL TRUTH: About to collect input data for step {self.name}")
            input_data = {}
            for unit_name, data_unit in self.step_input_data_units.items():
                logger.info(f"🔥 BRUTAL TRUTH: Getting data from unit {unit_name}")
                input_data[unit_name] = await data_unit.get()
                logger.info(f"🔥 BRUTAL TRUTH: Got data from unit {unit_name}, type: {type(input_data[unit_name])}")

            # ✅ CRITICAL DEBUG: Add logging before calling executor
            self.nb_logger.info(f"🚀 ABOUT TO CALL EXECUTOR for step {self.name}")
            self.nb_logger.info(f"🚀 Input data keys: {list(input_data.keys())}")
            self.nb_logger.info(f"🔥 BRUTAL TRUTH: Using executor {type(self.executor).__name__} for triggered execution")

            # Execute step business logic through executor (not direct process call)
            async def execute_wrapper():
                self.nb_logger.info("🔥 BRUTAL TRUTH: Inside triggered execute_wrapper, calling _execute_process")

                # FAIL-FAST: Add execution timeout protection
                timeout_seconds = getattr(self.config, 'execution_timeout', 300)  # 5 minutes default

                try:
                    result = await asyncio.wait_for(
                        self._execute_process(input_data),
                        timeout=timeout_seconds
                    )
                    return result
                except asyncio.TimeoutError:
                    from .component_base import ComponentConfigurationError
                    raise ComponentConfigurationError(
                        f"FAIL-FAST: Step {self.name} exceeded {timeout_seconds}s timeout. "
                        f"This indicates a hanging step. Check for infinite loops, blocking operations, "
                        f"or increase execution_timeout in step configuration."
                    )

            logger.info(f"🔥 BRUTAL TRUTH: About to call executor.execute() for step {self.name}")
            self.nb_logger.info(f"🔥 BRUTAL TRUTH: Calling {type(self.executor).__name__}.execute() from trigger")
            result = await self.executor.execute(execute_wrapper)
            logger.info(f"🔥 BRUTAL TRUTH: MIRACLE! Executor returned result: {type(result)}")
            self.nb_logger.info(f"🔥 BRUTAL TRUTH: Executor returned result: {type(result)}")

            # ✅ CRITICAL DEADLOCK FIX: Update output data units with proper data validation
            logger.info(f"🔥 BRUTAL TRUTH: About to update output data units for step {self.name}")
            logger.info(f"🔥 BRUTAL TRUTH: Step output data units: {list(self.step_output_data_units.keys())}")
            logger.info(f"🔥 BRUTAL TRUTH: Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")

            # Framework only ensures result exists - business logic validation is step responsibility

            # ✅ BRUTAL TRUTH: Call the unified output data unit update method
            await self._update_output_data_units(result)

            # Update execution statistics
            self._execution_count += 1
            self._last_result = result
            self._last_activity_time = time.time()

        except Exception as e:
            self._error_count += 1
            # ✅ CRITICAL DEBUG: Add more detailed error logging
            self.nb_logger.error(f"❌ EXCEPTION IN _execute_on_trigger for step {self.name}: {type(e).__name__}: {str(e)}")
            self.nb_logger.error(f"❌ Exception occurred at step: {self.name}")
            self.nb_logger.error(
                f"❌ Step execution failed: {e}", exc_info=True)
            raise

    async def _on_trigger_activated(self, trigger_data: Dict[str, Any]) -> None:
        """Handle trigger activation."""
        async with self.nb_logger.async_execution_context(
            OperationType.TRIGGER_ACTIVATE,
            f"{self.name}.trigger_activated",
            trigger_type=type(
                self.trigger).__name__ if self.trigger else "unknown"
        ) as context:
            self.nb_logger.log_trigger_activation(
                trigger_name=f"{self.name}.trigger",
                trigger_type=type(
                    self.trigger).__name__ if self.trigger else "unknown",
                conditions=trigger_data,
                activated=True
            )

            context.metadata['trigger_data'] = trigger_data

            # Execute the step
            await self.execute()

    @property
    def input_data_units(self) -> Dict[str, DataUnitBase]:
        """
        Map input_data_units to step_input_data_units for framework compatibility.

        This property ensures that BaseStep.execute() can access input data units
        regardless of which storage location they were created in during initialization.
        """
        return getattr(self, 'step_input_data_units', {})

    @input_data_units.setter
    def input_data_units(self, value: Dict[str, DataUnitBase]) -> None:
        """Set input_data_units and synchronize with step_input_data_units."""
        if not hasattr(self, 'step_input_data_units'):
            self.step_input_data_units = {}
        self.step_input_data_units.update(value)

    @property
    def output_data_units(self) -> Dict[str, DataUnitBase]:
        """
        Map output_data_units to step_output_data_units for framework compatibility.

        This property ensures that BaseStep.execute() can access output data units
        regardless of which storage location they were created in during initialization.
        """
        return getattr(self, 'step_output_data_units', {})

    @output_data_units.setter
    def output_data_units(self, value: Dict[str, DataUnitBase]) -> None:
        """Set output_data_units and synchronize with step_output_data_units."""
        if not hasattr(self, 'step_output_data_units'):
            self.step_output_data_units = {}
        self.step_output_data_units.update(value)

    def register_input_data_unit(self, input_id: str, data_unit: DataUnitBase) -> None:
        """Register an input data unit."""
        # Ensure step_input_data_units exists
        if not hasattr(self, 'step_input_data_units'):
            self.step_input_data_units = {}

        # Store in step_input_data_units (property will make it accessible via input_data_units)
        self.step_input_data_units[input_id] = data_unit
        self.nb_logger.info(f"Registered input data unit: {input_id}",
                            input_id=input_id,
                            data_unit_type=type(data_unit).__name__)

        # If we have a trigger, register the data unit with it
        if self.trigger and hasattr(self.trigger, 'register_data_unit'):
            self.trigger.register_data_unit(input_id, data_unit)

    def register_output_data_unit(self, data_unit: DataUnitBase) -> None:
        """Register the output data unit."""
        # Ensure step_output_data_units exists
        if not hasattr(self, 'step_output_data_units'):
            self.step_output_data_units = {}

        # Store in step_output_data_units (property will make it accessible via output_data_units)
        self.output_data_unit = data_unit
        self.nb_logger.info("Registered output data unit",
                            data_unit_type=type(data_unit).__name__)

    def add_link(self, link_id: str, link: LinkBase) -> None:
        """Add a link to another step."""
        self.links[link_id] = link
        self.nb_logger.info(f"Added link: {link_id}",
                            link_id=link_id,
                            link_type=type(link).__name__)

    async def execute(self, **kwargs) -> Any:
        """Execute the step using the configured executor."""
        if not self._is_initialized:
            await self.initialize()

        async with self.nb_logger.async_execution_context(
            OperationType.STEP_EXECUTE,
            f"{self.name}.execute",
            kwargs_keys=list(kwargs.keys())
        ) as context:
            try:
                # Collect input data
                input_data = {}
                for input_id, data_unit in self.input_data_units.items():
                    data = await data_unit.read()
                    input_data[input_id] = data

                    if self.config.log_data_transfers:
                        self.nb_logger.log_data_transfer(
                            source=f"{input_id}.data_unit",
                            destination=f"{self.name}.input",
                            data_type=type(data).__name__,
                            size_bytes=len(str(data)) if data else 0
                        )

                context.metadata['input_data_keys'] = list(input_data.keys())
                context.metadata['input_data_sizes'] = {
                    k: len(str(v)) if v else 0 for k, v in input_data.items()
                }

                self.nb_logger.debug(f"Collected input data for step {self.name}",
                                     input_keys=list(input_data.keys()))

                # Process using executor
                start_time = time.time()
                # Create a wrapper function that properly handles the input_data argument
                self.nb_logger.info(f"🔥 BRUTAL TRUTH: About to call executor.execute() on {type(self.executor).__name__}")

                async def execute_wrapper():
                    self.nb_logger.info("🔥 BRUTAL TRUTH: Inside execute_wrapper, calling _execute_process")
                    return await self._execute_process(input_data, **kwargs)

                self.nb_logger.info(f"🔥 BRUTAL TRUTH: Calling {type(self.executor).__name__}.execute(execute_wrapper)")
                self.nb_logger.info("🔥 BRUTAL TRUTH: About to await executor.execute() - THIS IS WHERE IT HANGS")

                # CRITICAL DEBUG: Try to identify the exact hang point
                try:
                    result = await self.executor.execute(execute_wrapper)
                    self.nb_logger.info(f"🔥 BRUTAL TRUTH: MIRACLE! Executor returned result: {type(result)}")
                except Exception as e:
                    self.nb_logger.error(f"🔥 BRUTAL TRUTH: Executor await failed with exception: {e}")
                    raise
                processing_time = time.time() - start_time

                self._execution_count += 1
                self._last_activity_time = time.time()
                self._total_processing_time += processing_time
                self._last_result = result

                # Store result in output data unit (legacy single output)
                if self.output_data_unit and result is not None:
                    await self.output_data_unit.write(result)

                    if self.config.log_data_transfers:
                        self.nb_logger.log_data_transfer(
                            source=f"{self.name}.output",
                            destination=f"{self.name}.output_data_unit",
                            data_type=type(result).__name__,
                            size_bytes=len(str(result)) if result else 0
                        )

                # ✅ BRUTAL TRUTH: Update step output data units (modern multiple outputs)
                if self.step_output_data_units and result is not None:
                    await self._update_output_data_units(result)

                # Propagate data through links
                if self.links and result is not None:
                    await self._propagate_through_links(result)

                # Log execution
                if self.config.log_executions:
                    self.nb_logger.log_step_execution(
                        step_name=self.name,
                        inputs=input_data,
                        outputs=result,
                        duration_ms=processing_time * 1000,
                        success=True
                    )

                context.metadata['result_type'] = type(
                    result).__name__ if result else None
                context.metadata['processing_time_ms'] = processing_time * 1000
                context.metadata['execution_count'] = self._execution_count

                self.nb_logger.debug(f"Step {self.name} executed successfully",
                                     execution_count=self._execution_count,
                                     processing_time_ms=processing_time * 1000,
                                     result_type=type(result).__name__ if result else None)

                return result

            except Exception as e:
                self._error_count += 1

                # Log failed execution
                if self.config.log_executions:
                    self.nb_logger.log_step_execution(
                        step_name=self.name,
                        inputs=input_data if 'input_data' in locals() else {},
                        success=False,
                        error=str(e)
                    )

                context.metadata['error_count'] = self._error_count
                self.nb_logger.error(f"Step {self.name} execution failed: {e}",
                                     error_type=type(e).__name__,
                                     error_count=self._error_count,
                                     exc_info=True)
                raise

    async def _execute_process(self, input_data: Dict[str, Any], **kwargs) -> Any:
        """Wrapper for process method to be executed by executor.

        G6 — when ``step_input_schema`` and/or ``step_output_schema`` are
        set on the StepConfig, the framework validates the dict at the
        wire boundary on every invocation. Both default to None, in which
        case no validation runs (preserves historical behavior).

        G21 Step 5 — automatic cooperation with WorkflowRunner pause.
        When this step runs inside a detached workflow context, the
        runner publishes a ``PauseSignal`` contextvar (PEP 567 asyncio-
        task-local). Before invoking ``process()``, we check the signal
        and ``await wait_until_resumed()`` if paused. This makes pause a
        framework-level cooperative protocol — user step code does NOT
        need to consult the contextvar manually for pause to work. The
        check is a no-op when no signal is published (no detached run);
        existing behavior is preserved.

        Pause semantics: the framework gates step BOUNDARIES, not step
        internals. A step that is mid-process when pause is requested
        runs to completion; the NEXT step's _execute_process is what
        blocks. This matches the gap proposal's "soft-pause; in-flight
        steps complete; no new steps started" semantics.

        G4-completion (2026-05-09) — automatic ProvenanceContext
        recording. When a ProvenanceContext is active (via
        ``current_provenance_context()``), the framework records ONE
        invocation per ``process()`` call, capturing inputs, outputs OR
        exception, and timing. The recorder pipeline applies the
        configured redaction list before sinking the record so secrets
        cannot leak. When no context is active, recording is a fast
        no-op — existing behavior preserved. Source:
        ``apecx-mcp-integration/eval_03_nanobrain_gap_inventory.md``
        Round 2 G4-completion;
        ``apecx-mcp-integration/docs/development_roadmap.md`` 8.7.
        """
        await _await_pause_signal_if_present()

        input_schema = self._g6_resolved_input_schema()
        if input_schema is not None:
            validate_payload_against_schema(
                input_data,
                input_schema,
                component_name=self.name,
                direction="input",
            )

        # G4-completion — resolve the active ProvenanceContext (if any)
        # and capture a wall-clock start so the recorded ``timing``
        # field is meaningful. Lazy import keeps core/step.py from
        # depending on core/provenance.py at import time.
        prov_ctx = None
        prov_started_at: Optional[float] = None
        try:
            from .provenance import current_provenance_context

            prov_ctx = current_provenance_context()
        except Exception:
            prov_ctx = None
        if prov_ctx is not None and getattr(prov_ctx, "enabled", False):
            import time as _time

            prov_started_at = _time.monotonic()
        else:
            prov_ctx = None  # disabled OR import failed → skip recording

        # G37 — publish step_start event to any active subscribers.
        # Independent of provenance recording: the contextvar lets
        # subscribers consume events live (provenance is the durable
        # audit trail; step events are the live publish stream).
        # Lazy import to avoid circulars; the time stamp is always
        # captured so step_complete / step_failed can compute duration.
        import time as _time

        _step_event_started_at = _time.monotonic()
        try:
            from .step_events import (
                _make_step_complete_event,
                _make_step_failed_event,
                _make_step_start_event,
                publish_step_event,
            )

            _step_events_enabled = True
        except Exception:
            _step_events_enabled = False
        if _step_events_enabled:
            run_id = self._g37_resolve_run_id()
            publish_step_event(
                _make_step_start_event(
                    step_name=self.name,
                    run_id=run_id,
                    inputs=input_data
                    if isinstance(input_data, dict)
                    else {"_input": input_data},
                )
            )

        try:
            result = await self.process(input_data, **kwargs)
        except Exception as exc:
            # G4-completion — record the exception path BEFORE re-raise.
            # Operators rely on the recorder seeing failures, otherwise
            # a crash silently disappears from the audit trail.
            import time as _time
            import traceback as _tb

            failure_duration = _time.monotonic() - _step_event_started_at
            if prov_ctx is not None:
                try:
                    await prov_ctx.record_step_invocation(
                        step_name=self.name,
                        inputs=input_data if isinstance(input_data, dict) else {"_input": input_data},
                        exception={
                            "type": type(exc).__name__,
                            "message": str(exc),
                            "traceback": _tb.format_exc(limit=10),
                        },
                        timing={"duration_seconds": failure_duration},
                    )
                except Exception:
                    # The recorder MUST NOT mask the original exception.
                    # If the recorder itself raises, swallow it and let
                    # the original ``exc`` propagate.
                    pass
            # G37 — publish step_failed event. Same fail-quiet
            # contract: subscriber errors are swallowed inside
            # publish_step_event so they cannot mask the original.
            if _step_events_enabled:
                publish_step_event(
                    _make_step_failed_event(
                        step_name=self.name,
                        run_id=self._g37_resolve_run_id(),
                        exception_type=type(exc).__name__,
                        exception_message=str(exc),
                        duration_seconds=failure_duration,
                        traceback_text=_tb.format_exc(limit=10),
                    )
                )
            raise

        output_schema = self._g6_resolved_output_schema()
        if output_schema is not None:
            validate_payload_against_schema(
                result,
                output_schema,
                component_name=self.name,
                direction="output",
            )

        # G4-completion — record the success path. Outputs go through
        # the redaction pipeline; operators who want size-bounded
        # records configure ``redact: ['outputs']`` or similar.
        import time as _time

        success_duration = _time.monotonic() - _step_event_started_at
        if prov_ctx is not None:
            try:
                await prov_ctx.record_step_invocation(
                    step_name=self.name,
                    inputs=input_data if isinstance(input_data, dict) else {"_input": input_data},
                    outputs=result if isinstance(result, dict) else {"_result": result},
                    timing={"duration_seconds": success_duration},
                )
            except Exception:
                # Recorder errors are non-fatal — never let provenance
                # bookkeeping break the step's actual return path.
                pass

        # G37 — publish step_complete event.
        if _step_events_enabled:
            publish_step_event(
                _make_step_complete_event(
                    step_name=self.name,
                    run_id=self._g37_resolve_run_id(),
                    outputs=result,
                    duration_seconds=success_duration,
                )
            )

        return result

    def _g37_resolve_run_id(self) -> Optional[str]:
        """G37 helper — resolve the active WorkflowRunContext's run_id
        for step-event tagging. Returns None when no context is active.
        Lazy import keeps core/step.py from depending on library/."""
        try:
            from nanobrain.library.orchestration.run_context import (
                current_run_context,
            )
        except ImportError:
            return None
        ctx = current_run_context()
        if ctx is None:
            return None
        return getattr(ctx, "run_id", None)

    def _g6_resolved_input_schema(self) -> Optional["SchemaRef"]:
        """Resolve the StepConfig.step_input_schema to a SchemaRef instance.
        Lazy + cached so per-invocation overhead is one attribute access."""
        return self._g6_resolve_schema_field("input")

    def _g6_resolved_output_schema(self) -> Optional["SchemaRef"]:
        return self._g6_resolve_schema_field("output")

    def _g6_resolve_schema_field(self, direction: str) -> Optional["SchemaRef"]:
        attr_cache = f"_g6_{direction}_schema_cached"
        if hasattr(self, attr_cache):
            return getattr(self, attr_cache)

        config = getattr(self, "_step_config_object", None) or getattr(self, "config", None)
        field_name = f"step_{direction}_schema"
        raw = getattr(config, field_name, None) if config is not None else None
        if raw is None:
            setattr(self, attr_cache, None)
            return None

        if isinstance(raw, SchemaRef):
            resolved = raw
        elif isinstance(raw, dict):
            try:
                SchemaRef._allow_direct_instantiation = True
                try:
                    resolved = SchemaRef(**raw)
                finally:
                    SchemaRef._allow_direct_instantiation = False
            except Exception as e:
                raise ComponentConfigurationError(
                    f"FAIL-FAST: step {self.name!r} {field_name} failed "
                    f"SchemaRef shape: {e}"
                ) from e
        else:
            raise ComponentConfigurationError(
                f"FAIL-FAST: step {self.name!r} {field_name} must be a "
                f"SchemaRef or dict, got {type(raw).__name__}"
            )

        setattr(self, attr_cache, resolved)
        return resolved

    async def _update_output_data_units(self, result: Any) -> None:
        """Unified method to update output data units from a step
        execution result.

        Routing per output unit:
          - Named-key path: result is a dict carrying the unit's name as
            a key — write ``result[unit_name]`` to the unit.
          - Single-output fallback: result is a dict, the step has exactly
            one output unit, and the unit's name is NOT in the dict —
            write the full dict to that unit. Matches imperative-mode
            parity (workflow.py line 2067-2071).
          - Neither path matches: skip the unit. Previously this branch
            was a silent no-op AND the named-key branch lacked its
            write — a latent silent-failure shape closed 2026-05-11.

        Workflow subclasses override this method to short-circuit the
        status-dict shape (see ``Workflow._update_output_data_units``).
        """
        import logging
        logger = logging.getLogger(__name__)

        # ✅ PARITY FIX (2026-05-05): when a step returns a dict that
        # does NOT contain the output data unit's name as a key, AND
        # the step has exactly one output data unit, write the entire
        # result to that unit. This matches ``_execute_step_imperative``'s
        # behavior (workflow.py line 2067-2071) — without this, a step
        # whose return shape doesn't match the data unit name would
        # silently produce no output in trigger-driven mode while
        # working correctly in imperative mode.
        single_output_fallback = (
            isinstance(result, dict)
            and len(self.step_output_data_units) == 1
            and not any(name in result for name in self.step_output_data_units)
        )

        for unit_name, data_unit in self.step_output_data_units.items():
            logger.info(f"🔥 Processing output data unit: {unit_name}")
            # Determine which slice of the result lands in this unit:
            #   - named-key path: result is a dict that explicitly carries
            #     this unit's name as a key (multi-output case)
            #   - single-output fallback: result is a dict but does NOT
            #     mention this unit's name AND there's exactly one unit,
            #     so the whole result becomes the unit's value
            #   - neither: nothing to do for this unit (skip)
            if unit_name in result:
                logger.info(f"🔥 Named-key path: extracting result[{unit_name!r}]")
                result_data = result[unit_name]
            elif single_output_fallback:
                logger.info(
                    f"🔥 Single-output fallback — writing full result dict "
                    f"to {unit_name}"
                )
                result_data = result
            else:
                # No matching key + not the single-output case → skip this
                # unit entirely (previously this branch was missing AND the
                # write fell through silently — a latent silent-failure
                # shape; 2026-05-11 audit).
                continue

            # ✅ CRITICAL FIX: Prevent storing DataUnit objects as data
            # (applies to both paths; a DataUnit instance is never the
            # right value to .set() into another DataUnit — extract the
            # underlying payload).
            if hasattr(result_data, '__class__') and 'DataUnit' in result_data.__class__.__name__:
                self.nb_logger.warning(
                    f"⚠️ Preventing DataUnit object storage in {unit_name} - extracting actual data")
                if hasattr(result_data, '_data'):
                    result_data = result_data._data
                elif hasattr(result_data, 'get'):
                    try:
                        result_data = await result_data.get()
                    except Exception as e:
                        self.nb_logger.error(f"Failed to extract data from DataUnit: {e}")
                        result_data = None
                else:
                    self.nb_logger.error(f"Cannot extract data from DataUnit object: {type(result_data)}")
                    result_data = None

            # Write — guard against None so cascade-not-fired branches
            # don't blank a unit that already has a real value.
            if result_data is not None:
                try:
                    await data_unit.set(result_data)
                    self.nb_logger.info(
                        f"📤 Updated output data unit: {unit_name} with {type(result_data).__name__}")
                except Exception as e:
                    logger.error(f"🔥 data_unit.set() FAILED for {unit_name}: {e}")
                    raise
            else:
                self.nb_logger.warning(
                    f"⚠️ Skipping output data unit update for {unit_name} - no valid data")

    async def _propagate_through_links(self, data: Any) -> None:
        """Propagate data through all links."""
        for link_id, link in self.links.items():
            try:
                async with self.nb_logger.async_execution_context(
                    OperationType.DATA_TRANSFER,
                    f"{self.name}.link.{link_id}",
                    link_type=type(link).__name__
                ) as context:
                    await link.transfer(data)

                    if self.config.log_data_transfers:
                        self.nb_logger.log_data_transfer(
                            source=f"{self.name}",
                            destination=f"link.{link_id}",
                            data_type=type(data).__name__,
                            size_bytes=len(str(data)) if data else 0
                        )

                    context.metadata['data_type'] = type(data).__name__

            except Exception as e:
                self.nb_logger.error(f"Failed to propagate data through link {link_id}: {e}",
                                     link_id=link_id,
                                     error_type=type(e).__name__,
                                     exc_info=True)

    def _load_tools_from_config(self, tools_config: Dict[str, Dict[str, Any]]) -> None:
        """Load tools configuration from step-level YAML configuration"""
        logger = get_logger(f"step.{self.name}.tools")

        for tool_name, tool_config in tools_config.items():
            try:
                tool_instance = self._create_tool_from_config(
                    tool_name, tool_config)
                self._register_tool(tool_instance, tool_name)
                logger.info(f"Loaded tool: {tool_name}")
            except Exception as e:
                logger.error(
                    f"Failed to load tool {tool_name}: {e}", exc_info=True)
                raise

    def _create_tool_from_config(self, tool_name: str, tool_config: Dict[str, Any]) -> Any:
        """Create tool instance from step-level YAML configuration"""

        # Start with a copy of the local tool config
        merged_config = tool_config.copy()

        # Get tool class from config
        tool_class = tool_config.get('class')

        # If no class specified or config_file is present, load from referenced config file
        config_file = tool_config.get('config_file')
        if config_file:
            try:
                tool_config_data = self._load_config_file(config_file)
                # Merge external config with local config, prioritizing local config
                external_config = tool_config_data.copy()
                # Remove the tool_config section temporarily if it exists in external config
                external_tool_config = external_config.pop('tool_config', {})
                # Merge external config first, then local config overrides it
                merged_config = {**external_config, **
                                 merged_config, **external_tool_config}

                # Get class from external config if not specified locally
                if not tool_class:
                    tool_class = tool_config_data.get('class')
            except Exception as e:
                logger = get_logger(f"step.{self.name}.tools")
                logger.warning(
                    f"Failed to load external config file {config_file}: {e}")

        if not tool_class:
            raise ValueError(
                f"Tool '{tool_name}' configuration missing 'class' field")

        # Update the class in merged config
        merged_config['class'] = tool_class

        # Merge tool_config section if present in local config
        if 'tool_config' in tool_config:
            tool_config_section = merged_config.pop('tool_config', {})
            merged_config = {**merged_config, **
                             tool_config_section, **tool_config['tool_config']}

        # Ensure name field is present (required by many components)
        # For external tools, use 'tool_name' instead of 'name'
        if 'name' not in merged_config and 'tool_name' not in merged_config and tool_name:
            # Check if this is an external tool by checking the class path
            if 'bioinformatics.bv_brc_tool' in tool_class or 'external_tool' in tool_class.lower():
                merged_config['tool_name'] = tool_name
            else:
                merged_config['name'] = tool_name

        # Create tool using direct from_config pattern
        module_path, class_name = tool_class.rsplit('.', 1)
        module = importlib.import_module(module_path)
        tool_cls = getattr(module, class_name)
        return tool_cls.from_config(merged_config)

    def _load_config_file(self, config_file_path: str) -> Dict[str, Any]:
        """Load configuration from file"""
        import yaml

        config_path = Path(config_file_path)
        if not config_path.is_absolute():
            # Make relative to step's workflow directory if available
            workflow_dir = getattr(self.config, 'workflow_directory', '.')
            config_path = Path(workflow_dir) / config_path

        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}")

        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _register_tool(self, tool: Any, name: str) -> None:
        """Register tool with step for easy access"""
        self.tools[name] = tool

    def get_tool(self, name: str) -> Optional[Any]:
        """Get tool by name for step usage"""
        return self.tools.get(name)

    def list_tools(self) -> List[str]:
        """List all available tool names"""
        return list(self.tools.keys())

    @abstractmethod
    async def process(self, input_data: Dict[str, Any], **kwargs) -> Any:
        """
        Process input data and return result.

        Args:
            input_data: Dictionary of input data from registered data units
            **kwargs: Additional parameters

        Returns:
            Processing result
        """
        pass

    # Enhancement 2: Data Unit Wrapper Handling
    async def _process_with_data_extraction(self, input_data: Dict[str, Any], **kwargs) -> Any:
        """
        Enhanced step process method with automatic data unit wrapper extraction.

        This method provides automatic data extraction for steps that want to use it.
        Steps can call this method instead of implementing manual extraction.

        Automatically extracts data from NanoBrain wrappers before processing:
        {'data_unit_name': actual_data} -> actual_data
        """
        try:
            # Automatic data unit wrapper extraction
            extracted_data = self._extract_wrapper_data(input_data)

            # Call step-specific processing
            if hasattr(self, '_process_step_data'):
                # New pattern: step implements _process_step_data for extracted data
                return await self._process_step_data(extracted_data, **kwargs)
            elif hasattr(self, 'execute'):
                # Backward compatibility: call execute with extracted data
                return await self.execute(extracted_data)
            else:
                # Fallback: call the original process with extracted data
                return await self.process(extracted_data, **kwargs)

        except Exception as e:
            if hasattr(self, 'nb_logger') and self.nb_logger:
                self.nb_logger.error(
                    f"❌ Step processing with data extraction failed: {e}", exc_info=True)
            raise

    def _extract_wrapper_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract data from NanoBrain data unit wrappers."""
        if not isinstance(input_data, dict):
            return {}

        # Check if data extraction is enabled
        if not self._should_extract_wrapper_data():
            return input_data.copy()

        # Strategy 1: Single key that matches a step input data unit
        if len(input_data) == 1:
            key, value = next(iter(input_data.items()))
            if (isinstance(value, dict) and
                hasattr(self, 'step_input_data_units') and
                    key in self.step_input_data_units):

                if hasattr(self, 'nb_logger') and self.nb_logger:
                    self.nb_logger.info(
                        f"🔧 Extracted data from '{key}' wrapper")
                return value.copy()

        # Strategy 2: Check explicit wrapper keys from configuration
        wrapper_keys = self._get_expected_wrapper_keys()
        for key in wrapper_keys:
            if key in input_data and isinstance(input_data[key], dict):
                if hasattr(self, 'nb_logger') and self.nb_logger:
                    self.nb_logger.info(
                        f"🔧 Extracted data from configured wrapper '{key}'")
                return input_data[key].copy()

        # Strategy 3: Check all step input data units
        if hasattr(self, 'step_input_data_units'):
            for unit_name in self.step_input_data_units.keys():
                if unit_name in input_data and isinstance(input_data[unit_name], dict):
                    if hasattr(self, 'nb_logger') and self.nb_logger:
                        self.nb_logger.info(
                            f"🔧 Extracted data from '{unit_name}' wrapper")
                    return input_data[unit_name].copy()

        # Strategy 4: No wrapper detected, return as-is
        return input_data.copy()

    def _should_extract_wrapper_data(self) -> bool:
        """Check if automatic wrapper data extraction is enabled."""
        if hasattr(self, 'config') and hasattr(self.config, 'data_extraction'):
            return getattr(self.config.data_extraction, 'enabled', True)
        return True  # Default: enabled

    def _get_expected_wrapper_keys(self) -> List[str]:
        """Get list of expected wrapper keys from configuration."""
        if hasattr(self, 'config') and hasattr(self.config, 'data_extraction'):
            return getattr(self.config.data_extraction, 'expected_wrapper_keys', [])
        return []

    def _unwrap_recursive_workflow_input(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        SYSTEMIC FIX: Unwrap recursive workflow_input nesting in BaseStep.

        This centralizes ALL recursive workflow_input handling in one place,
        preventing the need for individual steps to handle this issue.

        Handles cases where data gets wrapped like:
        {'workflow_input': {'workflow_input': {'user_query': 'test', ...}}}

        Args:
            data: Data that may contain recursive workflow_input nesting

        Returns:
            Data with recursive nesting unwrapped
        """
        try:
            if hasattr(self, 'nb_logger') and self.nb_logger:
                self.nb_logger.info(
                    f"🔧 BaseStep {self.name}: Starting recursive workflow_input unwrapping")

            if not isinstance(data, dict):
                return data

            # CRITICAL: Check for max_depth_exceeded marker FIRST
            if (isinstance(data, dict) and
                'workflow_input' in data and
                    isinstance(data.get('workflow_input'), dict)):

                inner = data['workflow_input']
                if (isinstance(inner, dict) and
                    'workflow_input' in inner and
                        inner.get('workflow_input') == "<max_depth_exceeded: dict>"):
                    if hasattr(self, 'nb_logger') and self.nb_logger:
                        self.nb_logger.warning(
                            f"🔧 BaseStep {self.name}: Detected max_depth_exceeded marker, returning empty dict")
                    return {}

            # Check if this is a recursive workflow_input structure
            if 'workflow_input' in data and len(data) == 1:
                inner_data = data['workflow_input']

                # If the inner data also has workflow_input, we have recursion
                if isinstance(inner_data, dict) and 'workflow_input' in inner_data:
                    # Use iterative approach to prevent recursion errors
                    current = inner_data
                    max_depth = 10  # Much smaller limit to prevent issues
                    depth = 0

                    while (isinstance(current, dict) and
                           'workflow_input' in current and
                           depth < max_depth):

                        next_level = current.get('workflow_input')

                        # Safety check for string markers
                        if isinstance(next_level, str) and "max_depth_exceeded" in next_level:
                            if hasattr(self, 'nb_logger') and self.nb_logger:
                                self.nb_logger.warning(
                                    f"🔧 BaseStep {self.name}: Hit max_depth_exceeded string marker")
                            return {}

                        # If next level has actual data fields, return it
                        if (isinstance(next_level, dict) and
                                any(key in next_level for key in ['user_query', 'session_id', 'request_id', 'timestamp'])):
                            if hasattr(self, 'nb_logger') and self.nb_logger:
                                self.nb_logger.info(
                                    f"🔧 BaseStep {self.name}: Found actual data at depth {depth + 1}")
                            return next_level

                        # Safety check - if next level is not a dict, stop
                        if not isinstance(next_level, dict):
                            break

                        current = next_level
                        depth += 1

                    # Return the unwrapped data or empty dict if too deep
                    if depth >= max_depth:
                        if hasattr(self, 'nb_logger') and self.nb_logger:
                            self.nb_logger.warning(
                                f"🔧 BaseStep {self.name}: Reached max depth {max_depth}, returning empty dict")
                        return {}

                    if hasattr(self, 'nb_logger') and self.nb_logger:
                        self.nb_logger.info(
                            f"🔧 BaseStep {self.name}: Unwrapped recursive workflow_input (depth: {depth})")
                    return current if isinstance(current, dict) else {}

            # No recursive wrapping detected, return as-is
            return data

        except Exception as e:
            if hasattr(self, 'nb_logger') and self.nb_logger:
                self.nb_logger.error(
                    f"🔧 BaseStep {self.name}: Error in unwrapping: {e}")
            # Return empty dict on any error to break cycles
            return {}

    async def get_clean_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simple automatic wrapper extraction and DataUnit resolution.

        Handles:
        - Wrapper key detection using existing data structure
        - DataUnit object resolution via duck typing
        - No field mapping, no validation overhead

        Args:
            input_data: Raw input data from workflow execution

        Returns:
            Dict with extracted data, preserving original field names
        """
        try:
            # Step 1: SYSTEMIC FIX - Handle recursive workflow_input unwrapping FIRST
            unwrapped_data = self._unwrap_recursive_workflow_input(input_data)

            # Step 2: Extract wrapper if needed
            extracted_data = await self._extract_wrapper_if_needed(unwrapped_data)

            # Step 3: Resolve any DataUnit objects in the extracted data
            resolved_data = await self._resolve_data_units_recursive(extracted_data)

            if hasattr(self, 'nb_logger') and self.nb_logger:
                self.nb_logger.debug(
                    f"🔧 Clean data extraction completed for {self.name}")

            return resolved_data

        except RecursionError:
            if hasattr(self, 'nb_logger') and self.nb_logger:
                self.nb_logger.error(
                    "❌ Clean data extraction failed: maximum recursion depth exceeded")
            # Return empty dict to break recursion cycles
            return {}
        except Exception as e:
            if hasattr(self, 'nb_logger') and self.nb_logger:
                self.nb_logger.error(f"❌ Clean data extraction failed: {e}")
            # Fallback to original data if extraction fails
            return input_data.copy() if isinstance(input_data, dict) else {}

    async def _extract_wrapper_if_needed(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simple wrapper extraction logic:

        1. If single key contains nested dict → extract the nested dict
        2. If multiple keys → return as-is
        3. Use existing step input data unit names for detection
        4. No hardcoded patterns or field mapping
        """
        if not isinstance(input_data, dict):
            return {}

        # Strategy 1: Single key with nested dictionary
        if len(input_data) == 1:
            key, value = next(iter(input_data.items()))
            if isinstance(value, dict):
                # Check if this looks like a wrapper (nested dict with multiple fields)
                if len(value) > 1 or any(isinstance(v, (str, int, float, bool)) for v in value.values()):
                    if hasattr(self, 'nb_logger') and self.nb_logger:
                        self.nb_logger.debug(
                            f"🔧 Extracted data from single wrapper key '{key}'")
                    return value.copy()

        # Strategy 2: Use existing step input data unit names
        if hasattr(self, 'step_input_data_units'):
            for unit_name in self.step_input_data_units.keys():
                if unit_name in input_data and isinstance(input_data[unit_name], dict):
                    if hasattr(self, 'nb_logger') and self.nb_logger:
                        self.nb_logger.debug(
                            f"🔧 Extracted data from data unit wrapper '{unit_name}'")
                    return input_data[unit_name].copy()

        # Strategy 3: Return as-is if no wrapper detected
        return input_data.copy()

    async def _resolve_data_units_recursive(self, data: Any) -> Any:
        """
        Recursively resolve DataUnit objects in the data structure.

        Args:
            data: Data that may contain DataUnit objects

        Returns:
            Data with all DataUnit objects resolved to their actual values
        """
        if isinstance(data, dict):
            resolved_dict = {}
            for key, value in data.items():
                resolved_dict[key] = await self._resolve_data_units_recursive(value)
            return resolved_dict
        elif isinstance(data, (list, tuple)):
            resolved_list = []
            for item in data:
                resolved_list.append(await self._resolve_data_units_recursive(item))
            return resolved_list if isinstance(data, list) else tuple(resolved_list)
        else:
            # Try to resolve as DataUnit
            return await self._resolve_single_data_unit(data)

    async def _resolve_single_data_unit(self, data: Any) -> Any:
        """
        Simple DataUnit resolution using duck typing:

        1. Check for get() method existence
        2. Attempt async get() call
        3. Return original if not a DataUnit
        4. No complex fallback logic
        """
        if hasattr(data, 'get') and callable(getattr(data, 'get', None)):
            try:
                resolved = await data.get()
                if hasattr(self, 'nb_logger') and self.nb_logger:
                    self.nb_logger.debug(
                        f"🔧 Resolved DataUnit object to: {type(resolved).__name__}")
                return resolved
            except Exception as e:
                if hasattr(self, 'nb_logger') and self.nb_logger:
                    self.nb_logger.warning(
                        f"⚠️ Failed to resolve DataUnit: {e}")
                pass  # Return original if get() fails

        return data

    def get_result(self) -> Any:
        """Get the last execution result."""
        return self._last_result

    @property
    def is_initialized(self) -> bool:
        """Check if the step is initialized."""
        return self._is_initialized

    @property
    def execution_count(self) -> int:
        """Get the number of executions."""
        return self._execution_count

    @property
    def error_count(self) -> int:
        """Get the number of errors."""
        return self._error_count

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for this step."""
        uptime_seconds = time.time() - self._start_time
        idle_seconds = time.time() - self._last_activity_time
        avg_processing_time = self._total_processing_time / \
            max(self._execution_count, 1)

        return {
            "uptime_seconds": uptime_seconds,
            "idle_seconds": idle_seconds,
            "execution_count": self._execution_count,
            "error_count": self._error_count,
            "success_rate": (self._execution_count - self._error_count) / max(self._execution_count, 1),
            "total_processing_time": self._total_processing_time,
            "avg_processing_time": avg_processing_time,
            "input_data_units": len(self.input_data_units),
            "has_output_data_unit": self.output_data_unit is not None,
            "links_count": len(self.links),
            "has_trigger": self.trigger is not None
        }

    async def get_output(self) -> Any:
        """Convenience method to get output data."""
        if not self.output_data_unit:
            return self._last_result
        return await self.output_data_unit.get()

    @property
    def is_running(self) -> bool:
        """Check if the step is currently running."""
        # For now, just return False as we don't track running state
        # This could be enhanced to track actual execution state
        return False

    # ============================================================================
    # AUTOMATIC TRIGGER SYSTEM - NEW IMPLEMENTATION
    # ============================================================================

    async def _register_automatic_input_triggers(self) -> None:
        """Automatically register input data units and create triggers.

        G117 (2026-05-18): if the step has ANY manual trigger declared
        in its YAML ``triggers:`` list, suppress ALL auto-input triggers
        on this step. The user has explicitly authored the firing
        semantics — adding extra triggers on OTHER input data units
        causes premature firing (e.g., a 2-input step with a trigger
        on the LAST-arriving input would otherwise also fire on the
        FIRST input, with the second input still empty).

        Per-unit suppression is the fallback when the user has no
        manual triggers at all but the step still has input data
        units (the original auto-trigger use case).
        """
        if not hasattr(self, 'step_input_data_units'):
            return

        # G117: any manual trigger -> suppress all auto-input.
        # The check uses the same name-resolution as
        # _has_manual_trigger_for_data_unit; if any input has a
        # manual trigger declared, we conclude the user is driving
        # firing manually for ALL inputs.
        manual_configs = getattr(self, 'step_trigger_configs', None) or []
        any_manual_trigger = len(manual_configs) > 0

        success_count = 0
        for unit_name, data_unit in self.step_input_data_units.items():
            if hasattr(data_unit, 'register_as_input_for_step'):
                if any_manual_trigger:
                    if self.enable_logging and self.nb_logger:
                        self.nb_logger.debug(
                            f"⏭️ Skipped auto-trigger for {unit_name} "
                            f"- step has {len(manual_configs)} manual trigger(s) "
                            f"declared (G117: any-manual suppresses all-auto)"
                        )
                    continue
                success = await data_unit.register_as_input_for_step(self)
                if success:
                    success_count += 1

                    if self.enable_logging and self.nb_logger:
                        self.nb_logger.debug(
                            f"✅ Auto-registered input trigger for {unit_name}")

        if self.enable_logging and self.nb_logger:
            self.nb_logger.info(
                f"✅ Created {success_count} automatic input triggers")

    async def _register_automatic_output_triggers(self) -> None:
        """Automatically register output data units for link activation."""
        if not hasattr(self, 'step_output_data_units'):
            return

        success_count = 0
        for unit_name, data_unit in self.step_output_data_units.items():
            if hasattr(data_unit, 'register_as_output_for_step'):
                success = await data_unit.register_as_output_for_step(self)
                if success:
                    success_count += 1

                    if self.enable_logging and self.nb_logger:
                        self.nb_logger.debug(
                            f"✅ Auto-registered output data unit {unit_name}")

        if self.enable_logging and self.nb_logger:
            self.nb_logger.info(
                f"✅ Registered {success_count} automatic output data units")

    def _has_manual_trigger_for_data_unit(self, data_unit_name: str) -> bool:
        """Check if a manual trigger already exists for the specified data unit.

        G117 fix (2026-05-18): also inspects the raw
        ``step_trigger_configs`` list (the YAML-declared trigger configs)
        because ``_register_automatic_input_triggers`` runs in Phase 2 —
        BEFORE the manual triggers from the YAML are resolved + populated
        into ``self.step_triggers`` (Phase 3). Without this fix, the
        framework auto-creates a duplicate ``auto_input_<step>_<unit>``
        trigger on EVERY input data unit even when the user already
        declared a ``DataUnitChangeTrigger`` for that unit in the YAML
        — the resulting double-firing causes downstream step.process()
        invocations 2x per single upstream data-unit set, which
        deadlocks/breaks multi-step composite workflows (see
        ``apecx-mcp-integration/docs/g117_multi_step_composition_double_firing_2026-05-18.md``).
        """
        # Phase 3 path: trigger already resolved + bound; check by
        # the trigger's resolved data_unit instance.
        if hasattr(self, 'step_triggers'):
            for trigger in self.step_triggers.values():
                if hasattr(trigger, 'data_unit') and hasattr(trigger.data_unit, 'name'):
                    if trigger.data_unit.name == data_unit_name:
                        return True

        # G117 — Phase 2 path: the trigger config is still raw (dict
        # OR Pydantic TriggerConfig); the ``data_unit`` field carries
        # a string reference to the unit name (or a resolved DataUnit
        # instance, or a dict with a ``name`` key). Cover all three.
        configs = getattr(self, 'step_trigger_configs', None) or []
        for trigger_cfg in configs:
            ref = None
            # Dict shape
            if isinstance(trigger_cfg, dict):
                ref = trigger_cfg.get('data_unit')
            # Pydantic config / instantiated trigger
            elif hasattr(trigger_cfg, 'data_unit'):
                ref = trigger_cfg.data_unit
            if ref is None:
                continue
            # Normalize: string name | DataUnit instance | dict with 'name'
            if isinstance(ref, str):
                if ref == data_unit_name:
                    return True
            elif hasattr(ref, 'name') and getattr(ref, 'name', None) == data_unit_name:
                return True
            elif isinstance(ref, dict) and ref.get('name') == data_unit_name:
                return True
        return False

    async def disable_automatic_triggers(self) -> None:
        """Disable automatic trigger creation for this step."""
        # Disable for input data units
        if hasattr(self, 'step_input_data_units'):
            for data_unit in self.step_input_data_units.values():
                if hasattr(data_unit, 'disable_automatic_triggers'):
                    await data_unit.disable_automatic_triggers()

        # Disable for output data units
        if hasattr(self, 'step_output_data_units'):
            for data_unit in self.step_output_data_units.values():
                if hasattr(data_unit, 'disable_automatic_triggers'):
                    await data_unit.disable_automatic_triggers()

        if self.enable_logging and self.nb_logger:
            self.nb_logger.info(f"🚫 Disabled automatic triggers for step {self.name}")

    async def get_automatic_trigger_statistics(self) -> Dict[str, Any]:
        """Get statistics about automatic triggers in this step."""
        stats = {
            'total_automatic_triggers': 0,
            'input_triggers': 0,
            'output_triggers': 0,
            'data_units_with_auto_triggers': 0
        }

        # Count input data unit triggers
        if hasattr(self, 'step_input_data_units'):
            for data_unit in self.step_input_data_units.values():
                if hasattr(data_unit, 'automatic_trigger_count'):
                    trigger_count = data_unit.automatic_trigger_count
                    stats['total_automatic_triggers'] += trigger_count
                    stats['input_triggers'] += len(getattr(data_unit, '_auto_input_triggers', {}))
                    if trigger_count > 0:
                        stats['data_units_with_auto_triggers'] += 1

        # Count output data unit triggers
        if hasattr(self, 'step_output_data_units'):
            for data_unit in self.step_output_data_units.values():
                if hasattr(data_unit, 'automatic_trigger_count'):
                    trigger_count = data_unit.automatic_trigger_count
                    stats['total_automatic_triggers'] += trigger_count
                    stats['output_triggers'] += len(getattr(data_unit, '_auto_output_triggers', {}))
                    if trigger_count > 0:
                        stats['data_units_with_auto_triggers'] += 1

        return stats


class Step(BaseStep):
    """Step (formerly SimpleStep) with mandatory from_config implementation."""

    REQUIRED_CONFIG_FIELDS = ['name']
    OPTIONAL_CONFIG_FIELDS = {
        'description': '',
        'processing_mode': 'combine',
        'add_metadata': True,
        'auto_initialize': True,
        'debug_mode': False,
        'enable_logging': True
    }

    @classmethod
    def extract_component_config(cls, config: StepConfig) -> Dict[str, Any]:
        """Extract Step configuration"""
        base_config = super().extract_component_config(config)
        return {
            **base_config,
            'processing_mode': getattr(config, 'processing_mode', 'combine'),
            'add_metadata': getattr(config, 'add_metadata', True),
        }

    def _init_from_config(self, config: StepConfig, component_config: Dict[str, Any],
                          dependencies: Dict[str, Any]) -> None:
        """Initialize Step with resolved dependencies"""
        super()._init_from_config(config, component_config, dependencies)
        self.processing_mode = component_config['processing_mode']
        self.add_metadata = component_config['add_metadata']

    async def process(self, input_data: Dict[str, Any], **kwargs) -> Any:
        """Process input data with simple transformation."""
        async with self.nb_logger.async_execution_context(
            OperationType.STEP_EXECUTE,
            f"{self.name}.process",
            input_keys=list(input_data.keys()),
            step_type="SimpleStep"
        ) as context:
            # Simple processing: combine all inputs
            if not input_data:
                result = "No input data"
            elif len(input_data) == 1:
                result = list(input_data.values())[0]
            else:
                result = {
                    "processed_at": time.time(),
                    "inputs": input_data,
                    "step_name": self.name
                }

            self.nb_logger.debug(f"Simple step {self.name} processed data",
                                 input_count=len(input_data),
                                 result_type=type(result).__name__)

            context.metadata['result_type'] = type(result).__name__
            context.metadata['input_count'] = len(input_data)

            return result


class TransformStep(BaseStep):
    """Step that applies a transformation function to input data."""

    REQUIRED_CONFIG_FIELDS = ['name']
    OPTIONAL_CONFIG_FIELDS = {
        'description': '',
        'auto_initialize': True,
        'debug_mode': False,
        'enable_logging': True
    }

    @classmethod
    def extract_component_config(cls, config: StepConfig) -> Dict[str, Any]:
        """Extract TransformStep configuration"""
        base_config = super().extract_component_config(config)
        return base_config

    @classmethod
    def resolve_dependencies(cls, component_config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Resolve TransformStep dependencies"""
        base_deps = super().resolve_dependencies(component_config, **kwargs)

        # Get transform function from kwargs
        transform_func = kwargs.get('transform_func')

        return {
            **base_deps,
            'transform_func': transform_func
        }

    def _init_from_config(self, config: StepConfig, component_config: Dict[str, Any],
                          dependencies: Dict[str, Any]) -> None:
        """Initialize TransformStep with resolved dependencies"""
        super()._init_from_config(config, component_config, dependencies)
        self.transform_func = dependencies.get(
            'transform_func') or self._default_transform

        self.nb_logger.debug(f"Transform step {self.name} initialized",
                             has_custom_transform=dependencies.get('transform_func') is not None)

    def _default_transform(self, data: Any) -> Any:
        """Default transformation: convert to string and add metadata."""
        return {
            "original": data,
            "transformed": str(data).upper(),
            "timestamp": time.time(),
            "step": self.name
        }

    async def process(self, input_data: Dict[str, Any], **kwargs) -> Any:
        """Process input data using the transformation function."""
        async with self.nb_logger.async_execution_context(
            OperationType.STEP_EXECUTE,
            f"{self.name}.process",
            input_keys=list(input_data.keys()),
            step_type="TransformStep"
        ) as context:
            if not input_data:
                result = None
            elif len(input_data) == 1:
                # Single input: apply transform directly
                input_value = list(input_data.values())[0]
                result = self.transform_func(input_value)
            else:
                # Multiple inputs: apply transform to each
                result = {}
                for key, value in input_data.items():
                    result[key] = self.transform_func(value)

            self.nb_logger.debug(f"Transform step {self.name} processed data",
                                 input_count=len(input_data),
                                 result_type=type(result).__name__)

            context.metadata['result_type'] = type(result).__name__
            context.metadata['input_count'] = len(input_data)
            context.metadata['transform_func'] = self.transform_func.__name__

            return result


"""
FRAMEWORK CHANGE: Pure Configuration-Driven Step Loading

Steps are now loaded EXCLUSIVELY through configuration files using the
from_config pattern. The create_step factory has been eliminated.

✅ CORRECT USAGE:
   # In workflow configuration (YAML):
   steps:
     - step_id: my_step
       config_file: "config/MyStep/MyStep.yml"
   
   # In workflow code:
   step_config = manager.load_config(config_path, StepConfig)
   step_class = import_class(step_config.class)
   step = step_class.from_config(config_path, executor=executor)

❌ DEPRECATED USAGE:
   step = create_step('module.StepClass', step_config, executor=executor)

REASON: Enforces pure configuration-driven architecture without
        programmatic component creation.
"""


# Legacy factory functions removed as per Phase 3: Legacy Component Removal
# These functions are no longer needed as all step creation is now handled
# via class-specific from_config methods leveraging ConfigBase._resolve_nested_objects()
#
# ✅ FRAMEWORK COMPLIANCE:
# - All component creation uses class-specific from_config methods
# - ConfigBase._resolve_nested_objects() handles automatic instantiation
# - No factory functions or redundant creation logic
# - Pure configuration-driven component creation


class AgentStep(BaseStep):
    """
    Universal wrapper for any NanoBrain Agent.

    Automatically bridges the gap between:
    - Step event-driven architecture (Dict input/output)
    - Agent conversational interface (str input/output)

    Features:
    - Universal agent compatibility
    - Automatic data format conversion
    - Conversation history management
    - Agent lifecycle management
    - Comprehensive error handling
    """

    COMPONENT_TYPE = "agent_step"

    def _init_from_config(self, config: AgentStepConfig,
                          component_config: Dict[str, Any],
                          dependencies: Dict[str, Any]) -> None:
        """Initialize AgentStep with wrapped agent."""
        super()._init_from_config(config, component_config, dependencies)

        # Store configuration
        self.agent_step_config = config

        # Create wrapped agent
        self.wrapped_agent = self._create_wrapped_agent(config)

        # Configuration for data bridging
        self.input_extraction_strategy = config.input_extraction_strategy
        self.output_formatting_strategy = config.output_formatting_strategy
        self.input_field_name = config.input_field_name
        self.input_template = config.input_template
        self.output_template = config.output_template

        # Conversation management
        self.enable_conversation_tracking = config.enable_conversation_tracking
        self.max_conversation_length = config.max_conversation_length
        self.conversation_id_field = config.conversation_id_field
        self.conversation_contexts = {}  # conversation_id -> ConversationContext

        # Agent lifecycle
        self.agent_initialized = False
        self.auto_initialize_agent = config.auto_initialize_agent

        # Performance tracking
        self.total_requests = 0
        self.successful_requests = 0
        self.total_processing_time = 0.0

        self.nb_logger.info(
            f"✅ AgentStep initialized with {type(self.wrapped_agent).__name__}")

    async def initialize(self) -> None:
        """Initialize the step and wrapped agent."""
        await super().initialize()

        if self.auto_initialize_agent:
            await self._initialize_wrapped_agent()

    async def _initialize_wrapped_agent(self) -> None:
        """Initialize the wrapped agent with error handling."""
        try:
            if hasattr(self.wrapped_agent, 'initialize'):
                await self.wrapped_agent.initialize()

            self.agent_initialized = True
            self.nb_logger.info(
                f"✅ Wrapped agent {type(self.wrapped_agent).__name__} initialized")

        except Exception as e:
            self.nb_logger.error(
                f"❌ Failed to initialize wrapped agent: {e}", exc_info=True)
            raise

    def _create_wrapped_agent(self, config: AgentStepConfig):
        """Create the wrapped agent from configuration."""

        try:
            # Import agent class
            agent_class = import_class_from_path(config.agent_class)

            # Create agent instance
            if isinstance(config.agent_config, str):
                # Load from file
                agent = agent_class.from_config(config.agent_config)
            else:
                # Use inline config
                agent = agent_class.from_config(config.agent_config)

            self.nb_logger.info(
                f"✅ Created wrapped agent: {type(agent).__name__}")
            return agent

        except Exception as e:
            self.nb_logger.error(
                f"❌ Failed to create wrapped agent: {e}", exc_info=True)
            raise

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data through wrapped agent using AgentResponse dataclasses."""

        async with self.nb_logger.async_execution_context(
            OperationType.STEP_EXECUTE,
            f"{self.name}.process_agent",
            input_keys=list(input_data.keys()),
            step_type="AgentStep"
        ) as context:

            start_time = time.time()
            self.total_requests += 1

            try:
                # Ensure agent is initialized
                if not self.agent_initialized:
                    await self._initialize_wrapped_agent()

                # 1. Extract agent input from step data
                agent_input, conversation_id = self._extract_agent_input(
                    input_data)

                # 2. Get conversation context if enabled
                conversation_context = self._get_conversation_context(
                    conversation_id) if conversation_id else None
                conversation_history = conversation_context.messages if conversation_context else []

                # 3. Create processing metadata
                metadata = AgentProcessingMetadata(
                    agent_name=getattr(self.wrapped_agent, 'name', self.name),
                    agent_type=type(self.wrapped_agent).__name__,
                    model=getattr(self.wrapped_agent, 'model', 'unknown'),
                    input_length=len(agent_input),
                    conversation_id=conversation_id,
                    conversation_turn=len(
                        conversation_history) // 2 + 1 if conversation_history else 1,
                    history_length=len(conversation_history)
                )

                # 4. Process through wrapped agent
                agent_output = await self._process_with_agent(agent_input, conversation_history, metadata)

                # 5. Update processing metadata
                processing_time = time.time() - start_time
                metadata.processing_time_seconds = processing_time
                metadata.response_length = len(agent_output)

                # 6. Create structured agent response
                if conversation_id:
                    agent_response = AgentResponse.create_success_response(
                        response_text=agent_output,
                        user_input=agent_input,
                        conversation_id=conversation_id,
                        conversation_history=conversation_history,
                        metadata=metadata
                    )

                    # Update conversation context
                    self._update_conversation_context(
                        conversation_id, agent_response.conversation_context)
                else:
                    # No conversation tracking
                    agent_response = AgentResponse(
                        response_text=agent_output,
                        processing_metadata=metadata,
                        response_type="standard"
                    )

                # Update performance metrics
                self.successful_requests += 1
                self.total_processing_time += processing_time

                self.nb_logger.info(
                    f"✅ Agent processing completed in {processing_time:.3f}s")

                # Return structured response as dictionary
                return agent_response.to_dict()

            except Exception as e:
                processing_time = time.time() - start_time
                self.total_processing_time += processing_time

                self.nb_logger.error(
                    f"❌ Agent processing failed: {e}", exc_info=True)

                # Create error response using AgentResponse
                error_response = AgentResponse.create_error_response(
                    error_message=str(e),
                    conversation_id=conversation_id or "",
                    conversation_history=conversation_history if 'conversation_history' in locals() else [],
                    error_type=type(e).__name__
                )

                return error_response.to_dict()



    def _extract_agent_input(self, input_data: Dict[str, Any]) -> tuple[str, Optional[str]]:
        """Extract text input for agent and conversation ID from step data."""

        # Extract conversation ID if present
        conversation_id = input_data.get(self.conversation_id_field)

        # Extract agent input based on strategy
        strategy = self.input_extraction_strategy

        if strategy == "single_text_field":
            field_name = self.input_field_name or 'user_input'
            agent_input = str(input_data.get(field_name, ''))

        elif strategy == "concatenate_all":
            text_parts = []
            for key, value in input_data.items():
                if key != self.conversation_id_field:  # Skip conversation ID
                    if isinstance(value, str):
                        text_parts.append(f"{key}: {value}")
                    else:
                        text_parts.append(f"{key}: {str(value)}")
            agent_input = "\n".join(text_parts)

        elif strategy == "custom_template":
            if self.input_template:
                try:
                    agent_input = self.input_template.format(**input_data)
                except KeyError as e:
                    self.nb_logger.warning(f"Template formatting failed: {e}")
                    agent_input = str(input_data)
            else:
                agent_input = str(input_data)

        else:  # auto_detect
            # Try common field names
            for field_name in ['user_input', 'message', 'query', 'text', 'input', 'prompt']:
                if field_name in input_data:
                    agent_input = str(input_data[field_name])
                    break
            else:
                # Fallback: use first string value or convert dict to string
                for value in input_data.values():
                    if isinstance(value, str) and value.strip():
                        agent_input = value
                        break
                else:
                    agent_input = str(input_data)

        return agent_input, conversation_id

    async def _process_with_agent(self, agent_input: str, conversation_history: List[Dict[str, Any]],
                                  metadata: AgentProcessingMetadata) -> str:
        """Process input through wrapped agent with conversation context."""

        try:
            # Create agent processing task for executor (enables distributed execution)
            async def agent_task():
                # Prepare agent input with conversation context if available
                if conversation_history and hasattr(self.wrapped_agent, 'process_with_history'):
                    # Agent supports conversation history
                    return await self.wrapped_agent.process_with_history(agent_input, conversation_history)
                elif hasattr(self.wrapped_agent, 'process'):
                    # Standard agent process method
                    return await self.wrapped_agent.process(agent_input)
                elif hasattr(self.wrapped_agent, 'execute'):
                    # Alternative execute method
                    return await self.wrapped_agent.execute(agent_input)
                elif hasattr(self.wrapped_agent, 'chat'):
                    # Chat-specific method
                    return await self.wrapped_agent.chat(agent_input)
                else:
                    raise RuntimeError(
                        f"Agent {type(self.wrapped_agent).__name__} has no supported processing method")

            # Execute agent task through executor (enables distributed execution)
            result = await self.executor.execute(agent_task)

            # Handle different return types
            if isinstance(result, dict):
                return result.get('response', result.get('output', str(result)))
            else:
                return str(result)

        except Exception as e:
            self.nb_logger.error(
                f"❌ Agent execution failed: {e}", exc_info=True)
            # Mark metadata as failed
            metadata.add_error(type(e).__name__, str(e))
            return f"I apologize, but I encountered an error while processing your request: {str(e)}"

    def _get_conversation_context(self, conversation_id: str) -> Optional[ConversationContext]:
        """Get conversation context for a specific conversation ID."""

        if not conversation_id or not self.enable_conversation_tracking:
            return None

        return self.conversation_contexts.get(conversation_id)

    def _update_conversation_context(self, conversation_id: str, conversation_context: ConversationContext) -> None:
        """Update conversation context with new exchange."""

        if not conversation_id or not self.enable_conversation_tracking:
            return

        # Trim conversation if too long
        if len(conversation_context.messages) > self.max_conversation_length:
            conversation_context.trim_to_length(self.max_conversation_length)

        # Store updated context
        self.conversation_contexts[conversation_id] = conversation_context

        self.nb_logger.debug(
            f"Updated conversation {conversation_id}: {conversation_context.total_messages} messages")

    def get_conversation_summary(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get summary of a conversation."""

        context = self._get_conversation_context(conversation_id)
        if not context:
            return None

        return {
            'conversation_id': conversation_id,
            'total_messages': context.total_messages,
            'user_messages': context.user_messages,
            'assistant_messages': context.assistant_messages,
            'created_at': context.created_at,
            'last_updated': context.last_updated,
            'conversation_topic': context.conversation_topic,
            'conversation_tags': context.conversation_tags,
            'context_optimized': context.context_optimized,
            'original_length': context.original_length
        }

    def clear_conversation(self, conversation_id: str) -> bool:
        """Clear conversation context for a specific conversation."""

        if conversation_id in self.conversation_contexts:
            del self.conversation_contexts[conversation_id]
            self.nb_logger.info(f"Cleared conversation {conversation_id}")
            return True

        return False

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics for monitoring."""

        success_rate = (self.successful_requests /
                        self.total_requests * 100) if self.total_requests > 0 else 0
        avg_processing_time = (self.total_processing_time /
                               self.total_requests) if self.total_requests > 0 else 0

        # Calculate conversation statistics
        total_conversations = len(self.conversation_contexts)
        active_conversations = sum(1 for ctx in self.conversation_contexts.values()
                                   # Active in last hour
                                   if (time.time() - ctx.last_updated) < 3600)

        return {
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'success_rate_percent': success_rate,
            'average_processing_time_seconds': avg_processing_time,
            'total_processing_time_seconds': self.total_processing_time,
            'total_conversations': total_conversations,
            'active_conversations': active_conversations,
            'agent_type': type(self.wrapped_agent).__name__ if self.wrapped_agent else 'unknown',
            'agent_name': getattr(self.wrapped_agent, 'name', 'unknown'),
            'agent_initialized': self.agent_initialized,
            'conversation_tracking_enabled': self.enable_conversation_tracking,
            'max_conversation_length': self.max_conversation_length
        }

    async def _execute_process(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Override BaseStep._execute_process to properly map AgentResponse keys to output data unit names.

        BRUTAL TRUTH: This fixes the core AgentStep output data unit mapping issue where
        AgentResponse.to_dict() returns {'response': '...'} but BaseStep expects keys
        that match the configured output data unit names (e.g., {'output': '...'}).
        """
        # Call the parent process method to get the AgentResponse dictionary
        agent_result = await self.process(input_data, **kwargs)

        # Map AgentResponse keys to configured output data unit names
        mapped_result = {}

        # Get the configured output data unit names
        output_unit_names = list(self.step_output_data_units.keys())

        if output_unit_names:
            # Map the 'response' key from AgentResponse to the first output data unit
            primary_output_unit = output_unit_names[0]
            if 'response' in agent_result:
                mapped_result[primary_output_unit] = agent_result['response']
                self.nb_logger.info(f"🔥 BRUTAL TRUTH: Mapped 'response' -> '{primary_output_unit}' for AgentStep output data unit")

            # Preserve other fields for additional output data units or metadata
            for key, value in agent_result.items():
                if key != 'response':  # Don't duplicate the main response
                    # Check if there's a matching output data unit for this key
                    if key in output_unit_names:
                        mapped_result[key] = value
                        self.nb_logger.info(f"🔥 BRUTAL TRUTH: Mapped '{key}' -> '{key}' for AgentStep output data unit")
        else:
            # No output data units configured, return original result
            mapped_result = agent_result
            self.nb_logger.warning("⚠️ No output data units configured for AgentStep, returning original result")

        self.nb_logger.info("🔥 BRUTAL TRUTH: AgentStep result mapping complete")
        self.nb_logger.info(f"   Original keys: {list(agent_result.keys())}")
        self.nb_logger.info(f"   Mapped keys: {list(mapped_result.keys())}")

        return mapped_result


# ---------------------------------------------------------------------------
# G21 Step 5 — automatic PauseSignal cooperation
# ---------------------------------------------------------------------------
#
# This helper is consulted at every BaseStep._execute_process call. When a
# step runs inside a detached workflow run, the WorkflowRunner publishes
# a PauseSignal contextvar (PEP 567 asyncio-task-local). If the signal
# is paused, this helper blocks until resumed — making pause a framework-
# level cooperative protocol rather than a per-step opt-in.
#
# Layering rule: nanobrain.core MUST NOT import from nanobrain.library.
# We honor that with a lazy + cached import. If the runtime module is
# unavailable for any reason (older nanobrain layout, partial install),
# the helper degrades to a no-op rather than failing.
#
# Performance: the cached _CURRENT_PAUSE_SIGNAL_GETTER is one attribute
# load per _execute_process call after first use; the contextvar.get()
# call itself is O(1).
# ---------------------------------------------------------------------------

_CURRENT_PAUSE_SIGNAL_GETTER: Any = None  # set on first _await_pause_signal_if_present call
_CURRENT_PAUSE_SIGNAL_PROBED: bool = False


async def _await_pause_signal_if_present() -> None:
    """Block until the current detached run's PauseSignal is resumed,
    or no-op when no pause signal is published in the current contextvar.

    See ``nanobrain.library.runtime.workflow_runner.PauseSignal`` and
    G21 Step 2 + Step 5 for the full cooperative-pause protocol.
    """
    global _CURRENT_PAUSE_SIGNAL_GETTER, _CURRENT_PAUSE_SIGNAL_PROBED

    if not _CURRENT_PAUSE_SIGNAL_PROBED:
        _CURRENT_PAUSE_SIGNAL_PROBED = True
        try:
            from nanobrain.library.runtime.workflow_runner import (
                current_pause_signal,
            )
            _CURRENT_PAUSE_SIGNAL_GETTER = current_pause_signal
        except ImportError:
            # Runtime module not importable — degrade to no-op.
            _CURRENT_PAUSE_SIGNAL_GETTER = None

    if _CURRENT_PAUSE_SIGNAL_GETTER is None:
        return

    signal = _CURRENT_PAUSE_SIGNAL_GETTER()
    if signal is None or not signal.is_paused():
        return

    # Paused — block until resumed. asyncio.Event.wait honors task
    # cancellation, so cancel-during-pause still terminates the step.
    await signal.wait_until_resumed()
