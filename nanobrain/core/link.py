"""
Link System for NanoBrain Framework

Provides dataflow abstractions for connecting Steps together.
Links define how information flows between system components.
Enhanced with mandatory from_config pattern implementation.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Callable, Union, Literal
from enum import Enum
from pathlib import Path
from pydantic import Field, ConfigDict, model_validator

# Async file operations
import aiofiles

from .component_base import FromConfigBase, ComponentConfigurationError
# Import logging system
from .logging_system import get_logger, get_system_log_manager
# Import new ConfigBase for constructor prohibition
from .config.config_base import ConfigBase

logger = logging.getLogger(__name__)


# Sentinel for "field path miss" — distinct from a legitimate None payload value.
_PATH_MISS = object()


def get_nested_value(data: Any, field_path: str) -> Any:
    """
    Extract nested value from dictionary or object attribute path using dot notation.

    Args:
        data: Dict or object to extract from
        field_path: Dot-separated path like "routing_decision.next_step"

    Returns:
        The value at the specified path, or None if not found.

    Note:
        Returns None on miss to preserve backwards compatibility with the legacy
        dict-condition path. NEW code (G1 PredicateConfig evaluator) uses
        ``get_nested_value_strict`` below, which FAIL-FASTs on miss to eliminate
        the silent-failure shape (per ``nanobrain_capability_gaps.md G1``).
    """
    try:
        current = data
        for key in field_path.split('.'):
            if isinstance(current, dict):
                current = current[key]
            else:
                current = getattr(current, key)
        return current
    except (KeyError, TypeError, AttributeError):
        return None


def get_nested_value_strict(data: Any, field_path: str) -> Any:
    """
    G1 strict-mode dotted-path resolver. Walks the path through dicts (via
    ``__getitem__``) and Pydantic models / objects (via ``getattr``).

    Returns:
        The resolved value, or the module-level ``_PATH_MISS`` sentinel when
        the path cannot be resolved. Caller must distinguish the sentinel
        from a legitimate ``None`` payload value.

    Used by:
        ``evaluate_predicate`` (G1 evaluator). For the legacy "soft miss
        returns None" semantics, use ``get_nested_value`` instead.
    """
    if not field_path:
        return data
    current = data
    for key in field_path.split('.'):
        if isinstance(current, dict):
            if key not in current:
                return _PATH_MISS
            current = current[key]
        else:
            if not hasattr(current, key):
                return _PATH_MISS
            current = getattr(current, key)
    return current


# ---------------------------------------------------------------------------
# G1 — Declarative predicate DSL for ConditionalLink
# ---------------------------------------------------------------------------
#
# Per ``nanobrain_capability_gaps.md G1``: a fixed vocabulary of predicate
# operators that an LLM (or human) can author safely in YAML, without needing
# to synthesize a Python callable + import path. The evaluator is pure (no
# side effects) and FAIL-FASTs on dotted-path miss when ``op != "exists"``.
#
# YAML form expected on a ConditionalLink config:
#
#     predicate: {op: contains, field: active_layers, value: structural}
#     # or a combinator:
#     predicate:
#       op: all
#       of:
#         - {op: contains, field: active_layers, value: structural}
#         - {op: eq, field: layer_options.structural.compute_accessibility, value: true}
#
# This is opt-in: existing ConditionalLink configs that use ``condition: {...}``
# with the legacy ``field/operator/value`` keys keep working unchanged. The new
# shape is used when the config dict has an ``op`` key (the new vocabulary).
# ---------------------------------------------------------------------------

PredicateOp = Literal["eq", "ne", "in", "contains", "exists", "all", "any", "not"]
LEAF_OPS = {"eq", "ne", "in", "contains", "exists"}
COMBINATOR_OPS = {"all", "any", "not"}


class PredicateConfig(ConfigBase):
    """G1 declarative predicate over a source data unit's payload.

    Validated at workflow load (``model_validator`` ensures leaf vs. combinator
    fields are populated correctly). Fixed vocabulary: see ``PredicateOp``.

    The evaluator (``evaluate_predicate``) runs the predicate against the
    payload of the link's configured ``source`` data unit. ``field`` is a
    dotted path (e.g. ``"active_layers"`` or
    ``"layer_options.structural.compute_accessibility"``) resolved by
    successive ``__getitem__`` on dicts and ``__getattr__`` on objects.

    Path miss raises ``ComponentConfigurationError("FAIL-FAST: predicate ...
    field <path> missing in payload")`` for ``op != "exists"`` (where
    ``"exists"`` is itself the presence check).

    Cross-reference ``nanobrain_capability_gaps.md G1``.
    """
    model_config = ConfigDict(extra="forbid")

    op: PredicateOp
    field: Optional[str] = None
    value: Any = None
    of: Optional[List["PredicateConfig"]] = None

    @model_validator(mode="after")
    def _validate_shape(self) -> "PredicateConfig":
        if self.op in LEAF_OPS:
            if self.of is not None:
                raise ValueError(
                    f"FAIL-FAST: predicate op={self.op!r} is a leaf op; "
                    f"'of' must not be set"
                )
            if self.op != "exists" and self.field is None:
                raise ValueError(
                    f"FAIL-FAST: predicate op={self.op!r} requires 'field'"
                )
            if self.op in {"eq", "ne", "in", "contains"} and self.value is None:
                # Allow value=None ONLY if the user genuinely wants to test
                # against None; we cannot tell intent from absence. Pydantic
                # treats absent + explicit None identically, so we conservatively
                # accept None values here. A FAIL-FAST on missing 'value' would
                # be a false positive for legitimate `eq: null` configs.
                pass
            if self.op == "exists":
                if self.field is None:
                    raise ValueError(
                        "FAIL-FAST: predicate op='exists' requires 'field'"
                    )
                if self.value is not None:
                    raise ValueError(
                        "FAIL-FAST: predicate op='exists' must not set 'value'"
                    )
        elif self.op in COMBINATOR_OPS:
            if self.field is not None or self.value is not None:
                raise ValueError(
                    f"FAIL-FAST: predicate op={self.op!r} is a combinator; "
                    f"'field' and 'value' must not be set"
                )
            if self.of is None or len(self.of) == 0:
                raise ValueError(
                    f"FAIL-FAST: predicate op={self.op!r} requires 'of' "
                    f"with at least one sub-predicate"
                )
            if self.op == "not" and len(self.of) != 1:
                raise ValueError(
                    "FAIL-FAST: predicate op='not' requires exactly one "
                    "sub-predicate"
                )
        else:
            # Should be unreachable due to Literal type, but defensive.
            raise ValueError(
                f"FAIL-FAST: unknown predicate op={self.op!r}; "
                f"vocabulary is eq | ne | in | contains | exists | all | any | not"
            )
        return self


PredicateConfig.model_rebuild()


def evaluate_predicate(payload: Any, predicate: PredicateConfig) -> bool:
    """G1 predicate evaluator. Pure function — no side effects.

    Args:
        payload: The value held by the ConditionalLink's source data unit
            at the moment the link fires. Typically a dict or a Pydantic
            model.
        predicate: The validated ``PredicateConfig``.

    Returns:
        bool: whether the predicate matches.

    Raises:
        ComponentConfigurationError: when ``op != "exists"`` and the
            predicate's ``field`` cannot be resolved against ``payload``.
            This is the G1 FAIL-FAST contract — silent ``None`` returns from
            path-miss are forbidden because they hide bugs.
    """
    op = predicate.op

    if op in COMBINATOR_OPS:
        if op == "all":
            return all(evaluate_predicate(payload, sub) for sub in predicate.of or [])
        if op == "any":
            return any(evaluate_predicate(payload, sub) for sub in predicate.of or [])
        if op == "not":
            # validator guarantees len(of) == 1
            return not evaluate_predicate(payload, predicate.of[0])

    # Leaf operations — resolve the field path first (except 'exists', which
    # uses path-miss semantics intentionally).
    field_path = predicate.field
    if op == "exists":
        resolved = get_nested_value_strict(payload, field_path)
        return resolved is not _PATH_MISS

    resolved = get_nested_value_strict(payload, field_path)
    if resolved is _PATH_MISS:
        raise ComponentConfigurationError(
            f"FAIL-FAST: predicate op={op!r} field={field_path!r} missing in payload "
            f"(payload type {type(payload).__name__}); use op='exists' if "
            f"a missing field should be a legitimate False rather than an error"
        )

    if op == "eq":
        return resolved == predicate.value
    if op == "ne":
        return resolved != predicate.value
    if op == "in":
        # value is the container; resolved is the needle
        try:
            return resolved in predicate.value
        except TypeError as e:
            raise ComponentConfigurationError(
                f"FAIL-FAST: predicate op='in' value must be iterable, "
                f"got {type(predicate.value).__name__}: {predicate.value!r}"
            ) from e
    if op == "contains":
        # resolved is the container; value is the needle
        try:
            return predicate.value in resolved
        except TypeError as e:
            raise ComponentConfigurationError(
                f"FAIL-FAST: predicate op='contains' field {field_path!r} resolved to "
                f"non-container {type(resolved).__name__}: {resolved!r}"
            ) from e

    # Unreachable due to validator, but defensive.
    raise ComponentConfigurationError(
        f"FAIL-FAST: unhandled predicate op={op!r} (validator should have rejected this)"
    )


def parse_condition_from_config(condition_config: Union[str, Dict[str, Any], "PredicateConfig"]) -> Callable:
    """
    Parse YAML condition configuration into a callable function.

    Three accepted input shapes (in priority order):

    1. **G1 declarative predicate** (NEW; preferred). A dict containing an
       ``op:`` key from the fixed vocabulary
       ``eq | ne | in | contains | exists | all | any | not``. Built into
       a ``PredicateConfig`` (validated), then evaluated via
       ``evaluate_predicate``. FAIL-FASTs on dotted-path miss for ``op !=
       "exists"`` — silent ``False`` returns are forbidden.

    2. **Legacy dict** (deprecated). A dict with ``field/operator/value``
       keys using the old vocabulary
       ``equals | not_equals | contains | greater_than | less_than | exists``.
       Continues to work; logs a deprecation WARNING.

    3. **Legacy string** (deprecated). A bare string used as a substring
       presence check on ``str(payload)``. Continues to work; logs a
       deprecation WARNING.

    Args:
        condition_config: One of the three shapes above, or a pre-built
            ``PredicateConfig`` instance.

    Returns:
        Callable[[Any], bool] — invoked synchronously with the source data
        unit's payload at link-fire time.

    Raises:
        ComponentConfigurationError: when the input is a dict that LOOKS
            like a G1 predicate (has ``op``) but fails ``PredicateConfig``
            validation. Legacy dicts that lack ``op`` continue to use the
            legacy path with a deprecation warning.
    """
    # Shape 1 — already a built PredicateConfig (programmatic callers).
    if isinstance(condition_config, PredicateConfig):
        predicate = condition_config

        def g1_predicate_func(data: Any) -> bool:
            return evaluate_predicate(data, predicate)
        return g1_predicate_func

    # Shape 1 — dict with G1 'op' key. Build PredicateConfig (which
    # validates) and bind a fresh evaluator closure.
    if isinstance(condition_config, dict) and "op" in condition_config:
        try:
            # ConfigBase normally forbids direct construction; the
            # _allow_direct_instantiation flag is the documented backdoor for
            # internal builders (mirrors LinkConfig handling above).
            PredicateConfig._allow_direct_instantiation = True
            try:
                predicate = PredicateConfig(**condition_config)
            finally:
                PredicateConfig._allow_direct_instantiation = False
        except Exception as e:
            raise ComponentConfigurationError(
                f"FAIL-FAST: predicate config validation failed: {e}"
            ) from e

        def g1_predicate_dict_func(data: Any) -> bool:
            return evaluate_predicate(data, predicate)
        return g1_predicate_dict_func

    # Shape 3 — bare string (legacy substring-presence check).
    if isinstance(condition_config, str):
        logger.warning(
            "ConditionalLink condition: bare-string form is deprecated; "
            "migrate to a declarative predicate per nanobrain_capability_gaps.md G1, "
            "e.g. {op: contains, field: <path>, value: %r}",
            condition_config,
        )

        def string_condition_func(data: Any) -> bool:
            return condition_config in str(data)
        return string_condition_func

    # Shape 2 — legacy dict with field/operator/value (deprecated).
    if isinstance(condition_config, dict):
        logger.warning(
            "ConditionalLink condition: legacy field/operator/value form is "
            "deprecated; migrate to a declarative predicate per "
            "nanobrain_capability_gaps.md G1 (use 'op' key with vocabulary "
            "eq/ne/in/contains/exists/all/any/not). Got config: %r",
            condition_config,
        )
        field = condition_config.get('field')
        operator = condition_config.get('operator', 'equals')
        value = condition_config.get('value')

        def dict_condition_func(data: Any) -> bool:
            try:
                field_value = get_nested_value(data, field)

                if operator == 'equals':
                    return field_value == value
                elif operator == 'not_equals':
                    return field_value != value
                elif operator == 'contains':
                    return value in str(field_value) if field_value else False
                elif operator == 'greater_than':
                    return float(field_value) > float(value) if field_value is not None else False
                elif operator == 'less_than':
                    return float(field_value) < float(value) if field_value is not None else False
                elif operator == 'exists':
                    return field_value is not None
                else:
                    logger.warning(
                        f"Unknown operator: {operator}, defaulting to equals")
                    return field_value == value

            except Exception as e:
                logger.debug(f"Condition evaluation failed: {e}")
                return False

        return dict_condition_func

    # Fallback for other types — also deprecated.
    logger.warning(
        "ConditionalLink condition: unsupported config type %s; treating as "
        "boolean coercion of the value (deprecated; will be removed)",
        type(condition_config).__name__,
    )

    def default_condition_func(data: Any) -> bool:
        return bool(condition_config)
    return default_condition_func


def parse_transform_from_config(transform_spec: str) -> Callable:
    """Resolve a YAML-configured ``transform_function`` string to a
    Python callable.

    Grammar: ``"package.module.callable"`` — a fully-qualified dotted
    path. The final segment is looked up as an attribute on the module
    named by the leading segments. Submodule attributes (e.g.
    ``"pkg.mod.Class.staticmethod"``) work because ``getattr`` walks the
    remaining dots after the module resolves.

    Sync callables (``def f(data) -> Any``) and async callables
    (``async def f(data) -> Any``) are both accepted. ``TransformLink``
    dispatches on ``asyncio.iscoroutinefunction`` at transfer time.

    Raises ``ComponentConfigurationError`` when:
      - ``transform_spec`` is not a string (e.g. mis-typed as a dict).
      - The spec has no dots (ambiguous between "top-level module" and
        "top-level callable"; reject so the message is clearer).
      - The leading module path doesn't import (wrong package).
      - The trailing attribute doesn't exist on the resolved object.
      - The resolved attribute isn't callable.

    Design note: we deliberately do NOT use a global registry or
    string-eval — module-path lookup is the same pattern APScheduler /
    Celery / pytest plugins use, and it has no startup-order pitfalls.
    """
    if not isinstance(transform_spec, str):
        raise ComponentConfigurationError(
            "transform_function must be a string 'package.module.callable', "
            f"got {type(transform_spec).__name__}: {transform_spec!r}"
        )
    if "." not in transform_spec:
        raise ComponentConfigurationError(
            f"transform_function {transform_spec!r} has no '.' — must be a "
            "fully-qualified dotted path like 'my_pkg.my_mod.my_func'"
        )

    import importlib

    # Walk dots backward until one prefix is an importable module; the
    # remaining tail is getattr-walked. This lets ``pkg.mod.Class.method``
    # resolve even though ``pkg.mod.Class`` is not itself a module.
    segments = transform_spec.split(".")
    module_obj: Any = None
    split_at = -1
    for i in range(len(segments) - 1, 0, -1):
        candidate = ".".join(segments[:i])
        try:
            module_obj = importlib.import_module(candidate)
            split_at = i
            break
        except ImportError:
            continue

    if module_obj is None:
        # The leading segment isn't even an importable top-level module.
        # Emit the clearest-possible diagnostic naming the top-level name.
        raise ComponentConfigurationError(
            f"transform_function {transform_spec!r} — could not import module "
            f"{segments[0]!r} (tried progressively shorter prefixes down to "
            f"{segments[0]!r})"
        )

    obj: Any = module_obj
    walked: list[str] = []
    for attr in segments[split_at:]:
        try:
            obj = getattr(obj, attr)
        except AttributeError as exc:
            walked_str = ".".join(segments[:split_at] + walked)
            raise ComponentConfigurationError(
                f"transform_function {transform_spec!r} — {walked_str!r} "
                f"has no attribute {attr!r}"
            ) from exc
        walked.append(attr)

    if not callable(obj):
        raise ComponentConfigurationError(
            f"transform_function {transform_spec!r} resolved to "
            f"{type(obj).__name__}, which is not callable"
        )
    return obj


class LinkType(Enum):
    """Types of links."""
    DIRECT = "direct"
    FILE = "file"
    QUEUE = "queue"
    TRANSFORM = "transform"
    CONDITIONAL = "conditional"
    ACADEMY = "academy"  # Academy integration link type


class LinkConfig(ConfigBase):
    """
    Configuration for links - INHERITS constructor prohibition.

    ❌ FORBIDDEN: LinkConfig(link_type="direct", ...)
    ✅ REQUIRED: LinkConfig.from_config('path/to/config.yml')
    """
    # Core link identification
    source: Optional[str] = None
    target: Optional[str] = None

    # Link behavior configuration
    link_type: LinkType = LinkType.DIRECT
    buffer_size: int = Field(default=100, ge=1)
    transform_function: Optional[str] = None
    condition: Optional[Union[str, Dict[str, Any]]] = None
    file_path: Optional[str] = None
    data_mapping: Optional[Dict[str, str]] = None
    auto_transfer: bool = Field(default=False)

    # Academy integration fields
    academy_agent_handle: Optional[str] = None
    action_name: Optional[str] = None
    timeout_seconds: int = Field(default=30, ge=1)
    retry_attempts: int = Field(default=3, ge=1)

    # G10 — gate-to-bottom semantics for ConditionalLink. See
    # `apecx-mcp-integration/docs/nanobrain_capability_gaps.md G10`.
    # When 'publish_empty' (default; legacy), a ConditionalLink whose
    # condition evaluates False is a no-op — the target data unit is
    # left untouched. Downstream AllDataReceivedTrigger will deadlock
    # unless the upstream step explicitly publishes an empty bundle.
    # When 'gate_to_bottom', the ConditionalLink writes the magic
    # string ConditionalLink.GATED_OFF_SENTINEL to the target. An
    # AllDataReceivedTrigger configured with the same gate_semantics
    # treats sentinel-bearing units as satisfied and excludes them
    # from the trigger payload — fan-in proceeds with N-1 keys
    # rather than waiting forever.
    gate_semantics: str = Field(
        default="publish_empty",
        description="ConditionalLink gating semantics. "
                    "'publish_empty' (default; legacy) is a no-op when the "
                    "condition is False. 'gate_to_bottom' writes the "
                    "ConditionalLink.GATED_OFF_SENTINEL marker to the target "
                    "so downstream AllDataReceivedTrigger can fire without "
                    "the upstream step publishing an empty bundle."
    )


class LinkBase(FromConfigBase, ABC):
    """
    Base Link Class - Data Flow Connections and Workflow Communication
    =================================================================

    The LinkBase class is the foundational component for creating data flow connections
    within the NanoBrain framework. Links enable seamless data transfer between steps,
    workflows, and agents with support for data transformation, conditional routing,
    buffering, and various transport mechanisms.

    **Core Architecture:**
        Links represent intelligent data transport systems that:

        * **Connect Components**: Establish data flow paths between steps and workflows
        * **Transform Data**: Apply transformations during data transfer
        * **Route Conditionally**: Support conditional data routing based on content
        * **Buffer Data**: Provide buffering for performance and reliability
        * **Validate Transfer**: Ensure data integrity during transmission
        * **Monitor Performance**: Track data flow performance and throughput

    **Biological Analogy:**
        Like neural pathways that carry information between different brain regions,
        links carry data between different processing components. Neural pathways
        have specialized properties (myelination for speed, neurotransmitter specificity
        for signal type, synaptic plasticity for learning) - exactly how links have
        specialized properties for data transformation, conditional routing, buffering,
        and performance optimization.

    **Data Flow Architecture:**

        **Connection Management:**
        * Direct connections for immediate data transfer
        * Buffered connections for asynchronous processing
        * Queue-based connections for reliable message passing
        * File-based connections for large dataset transfer

        **Data Transformation:**
        * Built-in transformation functions for common operations
        * Custom transformation scripts and functions
        * Data format conversion and normalization
        * Schema mapping and validation

        **Conditional Routing:**
        * Content-based routing with configurable conditions
        * Multi-path routing for complex data flows
        * Dynamic routing based on runtime conditions
        * Fallback routing for error scenarios

        **Performance Optimization:**
        * Intelligent buffering with configurable sizes
        * Compression for large data transfers
        * Parallel transfer for improved throughput
        * Connection pooling and reuse

    **Framework Integration:**
        Links seamlessly integrate with all framework components:

        * **Step Integration**: Connect step outputs to inputs for data flow
        * **Workflow Coordination**: Enable complex multi-step data processing
        * **Agent Communication**: Support data exchange between agents
        * **Data Unit Connectivity**: Connect data units across processing boundaries
        * **Executor Support**: Links work with all execution backends
        * **Monitoring Integration**: Comprehensive logging and performance tracking

    **Link Type Implementations:**
        The framework supports various link specializations:

        * **DirectLink**: Immediate data transfer with minimal overhead
        * **FileLink**: File-based transfer for large datasets and persistence
        * **QueueLink**: Reliable message queuing with persistence and retry
        * **TransformLink**: Data transformation during transfer
        * **ConditionalLink**: Conditional routing based on data content
        * **CompoundLink**: Combination of multiple link types for complex scenarios

    **Configuration Architecture:**
        Links follow the framework's configuration-first design:

        ```yaml
        # Direct link configuration
        name: "data_transfer"
        link_type: "direct"
        buffer_size: 1000

        # Source and target configuration
        source: "step_a.output_data"
        target: "step_b.input_data"

        # Transform link configuration
        name: "data_transformation"
        link_type: "transform"
        transform_function: "normalize_data"

        # Data transformation settings
        transformation:
          function_name: "custom_normalizer"
          parameters:
            scale_factor: 1.0
            remove_outliers: true
          input_schema: "schemas/raw_data.json"
          output_schema: "schemas/normalized_data.json"

        # Conditional link configuration
        name: "conditional_routing"
        link_type: "conditional"

        # Routing conditions
        condition:
          field: "data.category"
          operator: "equals"
          value: "priority"

        # Alternative routing
        routes:
          default:
            target: "standard_processing.input"
          priority:
            target: "priority_processing.input"
            condition:
              field: "data.priority"
              operator: "greater_than"
              value: 5

        # Queue link configuration
        name: "reliable_transfer"
        link_type: "queue"
        buffer_size: 10000

        # Queue settings
        queue_config:
          persistence: true
          retry_attempts: 3
          retry_delay_ms: 1000
          dead_letter_queue: true
          compression: true

        # File link configuration
        name: "large_dataset_transfer"
        link_type: "file"
        file_path: "data/transfer/{timestamp}_{source}_{target}.json"

        # File transfer settings
        file_config:
          compression: "gzip"
          encryption: true
          chunk_size: "10MB"
          cleanup_after_transfer: true
        ```

    **Usage Patterns:**

        **Basic Data Transfer:**
        ```python
        from nanobrain.core import DirectLink

        # Create link from configuration
        link = DirectLink.from_config('config/data_link.yml')

        # Connect data units
        await link.connect(source_data_unit, target_data_unit)

        # Transfer data
        result = await link.transfer(data)
        print(f"Transfer completed: {result}")
        ```

        **Data Transformation:**
        ```python
        # Transform link with custom function
        transform_link = TransformLink.from_config('config/transform_link.yml')

        # Define transformation function
        def normalize_data(data):
            # Custom normalization logic
            normalized = {
                'values': [x / max(data['values']) for x in data['values']],
                'metadata': data.get('metadata', {}),
                'timestamp': time.time()
            }
            return normalized

        # Register transformation
        transform_link.set_transform_function(normalize_data)

        # Data automatically transformed during transfer
        await transform_link.transfer(raw_data)
        ```

        **Conditional Routing:**
        ```python
        # Conditional link for dynamic routing
        conditional_link = ConditionalLink.from_config('config/routing_link.yml')

        # Define routing conditions
        conditional_link.add_route(
            condition=lambda data: data.get('priority', 0) > 5,
            target='high_priority_processor.input'
        )

        conditional_link.add_route(
            condition=lambda data: data.get('category') == 'urgent',
            target='urgent_processor.input'
        )

        # Default route for unmatched conditions
        conditional_link.set_default_route('standard_processor.input')

        # Data automatically routed based on content
        await conditional_link.transfer(data)
        ```

        **Large Dataset Transfer:**
        ```python
        # File-based link for large datasets
        file_link = FileLink.from_config('config/large_data_link.yml')

        # Configure compression and chunking
        file_link.set_compression('gzip')
        file_link.set_chunk_size('100MB')

        # Transfer large dataset efficiently
        await file_link.transfer(large_dataset)

        # Automatic cleanup and compression
        ```

    **Advanced Features:**

        **Performance Optimization:**
        * Intelligent buffering with adaptive sizing
        * Parallel transfer for large datasets
        * Compression for reduced bandwidth usage
        * Connection pooling and reuse for efficiency

        **Reliability and Error Handling:**
        * Automatic retry with exponential backoff
        * Dead letter queues for failed transfers
        * Data integrity validation with checksums
        * Graceful degradation for partial failures

        **Monitoring and Analytics:**
        * Real-time transfer performance monitoring
        * Throughput analysis and optimization recommendations
        * Error rate tracking and alerting
        * Data flow visualization and analysis

        **Security and Privacy:**
        * Data encryption for sensitive transfers
        * Access control and permission management
        * Audit logging for compliance and debugging
        * Data anonymization and privacy protection

    **Data Flow Patterns:**

        **Fan-Out Pattern:**
        * Single source data distributed to multiple targets
        * Parallel processing with result aggregation
        * Load balancing across processing components
        * Broadcast messaging for notifications

        **Fan-In Pattern:**
        * Multiple sources feeding into single target
        * Data aggregation and correlation
        * Result collection from parallel processing
        * Event consolidation and analysis

        **Pipeline Pattern:**
        * Sequential data processing through multiple stages
        * Data transformation at each pipeline stage
        * Error propagation and recovery mechanisms
        * Progress tracking and monitoring

        **Mesh Pattern:**
        * Complex interconnection of multiple components
        * Dynamic routing based on content and conditions
        * Adaptive data flow optimization
        * Distributed processing coordination

    **Integration Patterns:**

        **Workflow Integration:**
        * Links coordinate data flow between workflow steps
        * Automatic activation of downstream processing
        * Data dependency resolution and management
        * Result propagation and collection

        **Agent Coordination:**
        * Inter-agent communication and data sharing
        * Context preservation across agent interactions
        * Result correlation and analysis
        * Collaborative processing workflows

        **External System Integration:**
        * API-based data exchange with external services
        * Database integration for data persistence
        * File system integration for large datasets
        * Message queue integration for reliable messaging

    **Performance and Scalability:**

        **Throughput Optimization:**
        * Asynchronous data transfer for non-blocking operations
        * Batch processing for improved efficiency
        * Connection pooling for resource optimization
        * Adaptive buffering based on load patterns

        **Scalability Features:**
        * Horizontal scaling through link distribution
        * Load balancing across multiple link instances
        * Dynamic resource allocation based on demand
        * Auto-scaling for varying workloads

        **Resource Management:**
        * Memory management for large data transfers
        * Network bandwidth optimization and throttling
        * Storage management for file-based transfers
        * Cleanup and garbage collection

    **Development and Testing:**

        **Testing Support:**
        * Mock link implementations for testing
        * Data flow simulation and validation
        * Performance benchmarking and profiling
        * Integration testing with steps and workflows

        **Debugging Features:**
        * Comprehensive logging with data flow tracing
        * Transfer inspection and analysis tools
        * Performance profiling and optimization hints
        * Visual data flow monitoring and debugging

        **Development Tools:**
        * Link configuration validation and linting
        * Data flow visualization and analysis
        * Performance monitoring and optimization tools
        * Template generation for common link patterns

    Attributes:
        name (str): Link identifier for logging and component coordination
        link_type (LinkType): Type of link and data transfer mechanism
        buffer_size (int): Buffer size for data transfer optimization
        transform_function (str, optional): Transformation function name for data processing
        condition (Union[str, Dict], optional): Routing condition for conditional links
        source (str): Source data unit or component identifier
        target (str): Target data unit or component identifier
        is_connected (bool): Whether link is currently active and connected
        transfer_count (int): Total number of data transfers performed
        performance_metrics (Dict): Real-time performance and usage metrics

    Note:
        This is an abstract base class that cannot be instantiated directly.
        Use concrete implementations like DirectLink, TransformLink, or
        ConditionalLink. All links must be created using the from_config
        pattern with proper configuration files following framework patterns.

    Warning:
        Links may consume significant network and memory resources depending
        on data size and transfer patterns. Monitor resource usage and implement
        appropriate limits, buffering, and cleanup mechanisms. Be cautious with
        file-based links that may consume disk space.

    See Also:
        * :class:`LinkConfig`: Link configuration schema and validation
        * :class:`LinkType`: Available link types and transfer mechanisms
        * :class:`DirectLink`: Direct data transfer with minimal overhead
        * :class:`TransformLink`: Data transformation during transfer
        * :class:`ConditionalLink`: Conditional routing based on data content
        * :class:`BaseStep`: Steps that connect through links for data flow
        * :class:`Workflow`: Workflows that coordinate link-based data processing
    """

    COMPONENT_TYPE = "link"
    REQUIRED_CONFIG_FIELDS = ['link_type']
    OPTIONAL_CONFIG_FIELDS = {
        'source': None,
        'target': None,
        'buffer_size': 100,
        'transform_function': None,
        'condition': None,
        'file_path': None,
        'data_mapping': None,
        'auto_transfer': False
    }

    @classmethod
    def _get_config_class(cls):
        """UNIFIED PATTERN: Return LinkConfig - ONLY method that differs from other components"""
        return LinkConfig

    @classmethod
    def extract_component_config(cls, config: LinkConfig) -> Dict[str, Any]:
        """Extract Link configuration"""
        return {
            'source': config.source,
            'target': config.target,
            'link_type': config.link_type,
            'buffer_size': config.buffer_size,
            'transform_function': config.transform_function,
            'condition': config.condition,
            'file_path': config.file_path,
            'data_mapping': config.data_mapping,
            'auto_transfer': config.auto_transfer
        }

    @classmethod
    def resolve_dependencies(cls, component_config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Resolve Link dependencies - extract source/target from component_config"""
        return {
            'source': component_config.get('source'),
            'target': component_config.get('target'),
            'enable_logging': kwargs.get('enable_logging', True),
            'debug_mode': kwargs.get('debug_mode', False)
        }

    def _init_from_config(self, config: LinkConfig, component_config: Dict[str, Any],
                          dependencies: Dict[str, Any]) -> None:
        """Initialize Link with resolved dependencies"""
        self.config = config
        self._source = dependencies.get('source')
        self._target = dependencies.get('target')

        # ✅ CRITICAL: Validate against self-referencing links at base level
        if self._source and self._target:
            source_name = getattr(self._source, 'name', str(self._source))
            target_name = getattr(self._target, 'name', str(self._target))

            if (self._source is self._target or
                (hasattr(self._source, 'name') and hasattr(self._target, 'name') and
                 source_name == target_name)):
                error_msg = (
                    f"❌ ILLEGAL SELF-REFERENCING LINK: {self.__class__.__name__} cannot connect "
                    f"data unit '{source_name}' to itself. Self-referencing links are "
                    f"prohibited in the workflow architecture as they create infinite "
                    f"trigger loops and prevent proper workflow execution.")
                raise ValueError(error_msg)

        self._update_name()
        self._is_active = False
        self._transfer_count = 0
        self._error_count = 0
        self._creation_time = time.time()
        self._last_transfer_time = None
        self._total_transfer_time = 0.0

        # Store data mapping configuration for transform operations
        self.data_mapping = getattr(config, 'data_mapping', None)

        # FRAMEWORK FIX: Initialize transfer trigger for automatic data flow
        # Will be set up in _setup_automatic_transfer_if_possible
        self.transfer_trigger = None

        # Initialize centralized logging system
        self.enable_logging = dependencies.get('enable_logging', True)
        if self.enable_logging:
            # Use centralized logging system
            self.nb_logger = get_logger(
                self.name, category="links", debug_mode=dependencies.get('debug_mode', False))

            # Register with system log manager
            system_manager = get_system_log_manager()

            # Handle both string identifiers and data unit objects for source/target
            if isinstance(self.source, str):
                source_name = self.source
            else:
                source_name = getattr(self.source, 'name', str(self.source))

            if isinstance(self.target, str):
                target_name = self.target
            else:
                target_name = getattr(self.target, 'name', str(self.target))

            system_manager.register_component("links", self.name, self, {
                "link_type": component_config['link_type'].value if hasattr(component_config['link_type'], 'value') else str(component_config['link_type']),
                "source": source_name,
                "target": target_name,
                "buffer_size": component_config['buffer_size'],
                "enable_logging": True
            })
        else:
            self.nb_logger = None

        # Note: Automatic transfer setup will be called after source/target are set

    # LinkBase inherits FromConfigBase.__init__ which prevents direct instantiation

    def _setup_automatic_transfer_if_possible(self) -> None:
        """Setup automatic transfer if explicitly configured"""
        # Only setup automatic transfer if explicitly requested in configuration
        auto_transfer = getattr(self, 'auto_transfer', False)
        if not auto_transfer:
            if self.enable_logging and self.nb_logger:
                self.nb_logger.debug(
                    f"DirectLink {self.name} - automatic transfer disabled (auto_transfer=False)")
            return

        try:
            if hasattr(self.source, 'register_change_listener'):
                # Register with source data unit for automatic transfer
                self.source.register_change_listener(
                    self._on_source_data_changed)
                self._is_active = True

                if self.enable_logging and self.nb_logger:
                    self.nb_logger.info(
                        f"🔗 DirectLink {self.name} registered for automatic transfer")
            else:
                if self.enable_logging and self.nb_logger:
                    self.nb_logger.warning(
                        f"Source {self.source} does not support change listeners")
        except Exception as e:
            if self.enable_logging and self.nb_logger:
                self.nb_logger.error(
                    f"Failed to setup automatic transfer: {e}")

    def _setup_callback_registration(self) -> None:
        """Setup callback registration for automatic data transfer.

        RE-ENABLED 2026-05-04 (apecx-mcp-integration): the previous comment
        claimed this was disabled because ``DataUnit.register_with_link()``
        was the new mechanism. That method is defined on DataUnit but never
        called anywhere in the framework — i.e., the supposed replacement
        does not exist, so there is no duplicate-transfer risk to guard
        against. Without this registration, inter-step DirectLinks never
        propagate data and multi-step workflows silently produce no output
        past the first step. See
        ``apecx-mcp-integration/src/apecx_integration/synonym_dictionary/workflow/workflow.py``
        for the durable integration test that catches a re-disable
        regression.
        """
        if not self.source or not self.target:
            if self.enable_logging and self.nb_logger:
                self.nb_logger.warning(
                    f"DirectLink {self.name} - missing source or target for callback registration")
            return

        try:
            if hasattr(self.source, 'register_change_listener'):
                # Register with source data unit for automatic transfer.
                # When the source's _notify_change_listeners fires, our
                # _on_source_data_changed handler reads new_data from the
                # event and calls self.transfer(...) → writes to target.
                self.source.register_change_listener(self._on_source_data_changed)
                if self.enable_logging and self.nb_logger:
                    self.nb_logger.info(
                        f"🔗 DirectLink {self.name}: registered change listener on source")
            else:
                if self.enable_logging and self.nb_logger:
                    self.nb_logger.warning(
                        f"⚠️ DirectLink {self.name}: source has no register_change_listener; "
                        "data will not propagate automatically")
        except Exception as e:
            if self.enable_logging and self.nb_logger:
                self.nb_logger.error(
                    f"Failed to setup callback registration: {e}")

    def _verify_callback_registration(self) -> bool:
        """Verify that the link callback is properly registered with the source data unit."""
        if not hasattr(self.source, '_change_listeners'):
            return False

        return self._on_source_data_changed in self.source._change_listeners

    # REMOVED: _bind_to_output_trigger method - DirectLink no longer binds to triggers automatically
    # All data flow should be managed by DataUnitChangeTrigger configurations

    # REMOVED: _find_source_step and _extract_data_unit_name methods - no longer needed
    # DirectLink no longer binds to triggers automatically

    def _update_name(self) -> None:
        """Update link name based on current source and target"""
        # Handle both string identifiers and data unit objects
        if self._source:
            if isinstance(self._source, str):
                source_name = self._source
            else:
                source_name = getattr(self._source, 'name', str(self._source))
        else:
            source_name = 'unknown'

        if self._target:
            if isinstance(self._target, str):
                target_name = self._target
            else:
                target_name = getattr(self._target, 'name', str(self._target))
        else:
            target_name = 'unknown'

        self.name = f"{source_name}->{target_name}"

    @property
    def source(self):
        """Get the source data unit"""
        return self._source

    @source.setter
    def source(self, value):
        """Set the source data unit and update name"""
        self._source = value
        self._update_name()
        # Update logger if it exists
        if hasattr(self, 'nb_logger') and self.nb_logger:
            from nanobrain.core.logging_system import get_logger
            self.nb_logger = get_logger(self.name, category="links")

    @property
    def target(self):
        """Get the target data unit"""
        return self._target

    @target.setter
    def target(self, value):
        """Set the target data unit and update name"""
        self._target = value
        self._update_name()
        # Update logger if it exists
        if hasattr(self, 'nb_logger') and self.nb_logger:
            from nanobrain.core.logging_system import get_logger
            self.nb_logger = get_logger(self.name, category="links")

    async def _on_source_data_changed(self, trigger_event: Dict[str, Any]) -> None:
        """Handle source data unit change - automatically transfer data"""
        try:
            if not self._is_active:
                return

            if self.enable_logging and self.nb_logger:
                self.nb_logger.info(
                    f"🔥 Link {self.name} triggered by source data change")

            # Use event data directly instead of calling source.get()
            source_data = None
            if 'new_data' in trigger_event:
                source_data = trigger_event['new_data']
                if self.enable_logging and self.nb_logger:
                    self.nb_logger.debug(
                        f"✅ Using event data: {type(source_data).__name__}")
            else:
                if self.enable_logging and self.nb_logger:
                    self.nb_logger.debug(
                        "⚠️ No 'new_data' in event, falling back to source.get()")
                # Fallback: Get data from source data unit
                if hasattr(self.source, 'get'):
                    if asyncio.iscoroutinefunction(self.source.get):
                        source_data = await self.source.get()
                    else:
                        source_data = self.source.get()

            # Only transfer if data exists
            if source_data is not None:
                if self.enable_logging and self.nb_logger:
                    self.nb_logger.info(
                        f"🚀 Starting transfer for {self.name} with data type: {type(source_data).__name__}")
                await self.transfer(source_data)

                if self.enable_logging and self.nb_logger:
                    self.nb_logger.info(
                        f"🔄 Automatic transfer completed: {self.name}")
            else:
                if self.enable_logging and self.nb_logger:
                    self.nb_logger.warning(
                        f"❌ No data to transfer for {self.name} - source_data is None")

        except Exception as e:
            if self.enable_logging and self.nb_logger:
                self.nb_logger.error(
                    f"❌ Automatic transfer failed for {self.name}: {e}")

    async def _should_transfer(self, data: Any) -> bool:
        """Check if data should be transferred (apply conditions)"""
        try:
            return self.condition_func(data) if hasattr(self, 'condition_func') else True
        except:
            return True

    async def _transform_data(self, data: Any) -> Any:
        """Apply data transformations (can be overridden by subclasses)"""
        # Apply data mapping if configured
        if hasattr(self, 'data_mapping') and self.data_mapping:
            if isinstance(data, dict):
                transformed = {}
                for target_key, source_key in self.data_mapping.items():
                    if source_key in data:
                        transformed[target_key] = data[source_key]
                return transformed

        return data

    def _parse_data_unit_reference(self, reference: str) -> tuple[str, str]:
        """Parse step.data_unit notation"""
        if '.' not in reference:
            raise ValueError(
                f"Invalid reference: {reference}. Must use 'step.data_unit' format")

        step_name, data_unit_name = reference.split('.', 1)
        return step_name.strip(), data_unit_name.strip()

    def _validate_dot_notation_reference(self, reference: str) -> bool:
        """Validate that reference uses proper step.data_unit format"""
        try:
            step_name, data_unit_name = self._parse_data_unit_reference(
                reference)
            return len(step_name) > 0 and len(data_unit_name) > 0
        except ValueError:
            return False

    def resolve_link_endpoints(self, source_ref: str, target_ref: str,
                               workflow_context: Dict[str, Any]) -> tuple[Any, Any]:
        """Resolve both source and target references using dot notation"""
        # Validate references
        if not self._validate_dot_notation_reference(source_ref):
            raise ValueError(f"Invalid source reference: {source_ref}")
        if not self._validate_dot_notation_reference(target_ref):
            raise ValueError(f"Invalid target reference: {target_ref}")

        # Resolve references
        source_data_unit = self._resolve_data_unit_reference(
            source_ref, workflow_context)
        target_data_unit = self._resolve_data_unit_reference(
            target_ref, workflow_context)

        return source_data_unit, target_data_unit

    @abstractmethod
    async def transfer(self, data: Any) -> None:
        """Transfer data from source to target."""
        pass

    @abstractmethod
    async def start(self) -> None:
        """Start the link."""
        # EVENT-DRIVEN ARCHITECTURE: Start automatic transfer trigger
        if self.transfer_trigger:
            await self.transfer_trigger.start_monitoring()
            if self.enable_logging and self.nb_logger:
                self.nb_logger.info(
                    f"🚀 Started automatic transfer for link {self.name}")
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the link."""
        # EVENT-DRIVEN ARCHITECTURE: Stop automatic transfer trigger
        if self.transfer_trigger:
            await self.transfer_trigger.stop_monitoring()
            if self.enable_logging and self.nb_logger:
                self.nb_logger.info(
                    f"🛑 Stopped automatic transfer for link {self.name}")
        pass

    def _get_internal_state(self) -> Dict[str, Any]:
        """Get comprehensive internal state for logging."""
        uptime = time.time() - self._creation_time
        avg_transfer_time = self._total_transfer_time / \
            max(self._transfer_count, 1)

        return {
            "is_active": self._is_active,
            "transfer_count": self._transfer_count,
            "error_count": self._error_count,
            "success_rate": self._transfer_count / max(self._transfer_count + self._error_count, 1),
            "uptime_seconds": uptime,
            "last_transfer_time": self._last_transfer_time,
            "avg_transfer_time_ms": avg_transfer_time * 1000 if self._transfer_count > 0 else 0,
            "link_type": self.config.link_type.value if hasattr(self.config.link_type, 'value') else str(self.config.link_type),
            "source": getattr(self.source, 'name', str(self.source)),
            "target": getattr(self.target, 'name', str(self.target))
        }

    @property
    def is_active(self) -> bool:
        """Check if link is active."""
        return self._is_active

    @property
    def transfer_count(self) -> int:
        """Get number of successful transfers."""
        return self._transfer_count

    @property
    def error_count(self) -> int:
        """Get number of transfer errors."""
        return self._error_count

    async def _record_transfer(self, success: bool = True, duration_ms: float = None, data_info: Dict[str, Any] = None) -> None:
        """Record transfer statistics with comprehensive logging."""
        transfer_time = time.time()

        if success:
            self._transfer_count += 1
            self._last_transfer_time = transfer_time
            if duration_ms:
                self._total_transfer_time += duration_ms / 1000.0
        else:
            self._error_count += 1

        # Log transfer event
        if self.enable_logging and self.nb_logger:
            self.nb_logger.log_data_transfer(
                source=getattr(self.source, 'name', str(self.source)),
                destination=getattr(self.target, 'name', str(self.target)),
                data_type=data_info.get(
                    'data_type', 'unknown') if data_info else 'unknown',
                size_bytes=data_info.get('size_bytes', 0) if data_info else 0
            )

            self.nb_logger.info(f"Link transfer {'succeeded' if success else 'failed'}: {self.name}",
                                operation="transfer",
                                success=success,
                                duration_ms=duration_ms,
                                data_info=data_info or {},
                                internal_state=self._get_internal_state())


class DirectLink(LinkBase):
    """
    Direct link that immediately transfers data from source to target.
    Enhanced with mandatory from_config pattern implementation.
    """

    COMPONENT_TYPE = "direct_link"
    REQUIRED_CONFIG_FIELDS = ['link_type']
    OPTIONAL_CONFIG_FIELDS = {
        'buffer_size': 100,
        'data_mapping': None,
        'auto_transfer': False
    }

    def __init__(self, *args, **kwargs):
        """Prevent direct instantiation - use from_config instead"""
        raise RuntimeError(
            "Direct instantiation of DirectLink is prohibited. "
            "ALL framework components must use DirectLink.from_config() "
            "as per mandatory framework requirements."
        )

    @classmethod
    def from_config(cls, config: Union[str, Path, LinkConfig, Dict[str, Any]], **kwargs) -> 'DirectLink':
        """Mandatory from_config implementation for DirectLink with dictionary support"""
        # Get logger
        nb_logger = get_logger(f"{cls.__name__}.from_config")
        nb_logger.info(f"Creating {cls.__name__} from configuration")

        # Step 1: Normalize input to LinkConfig object
        if isinstance(config, (str, Path)):
            # File path input - use standard config loading
            config_object = LinkConfig.from_config(config, **kwargs)
        elif isinstance(config, dict):
            # Dictionary input - create LinkConfig from dict (inline config support)
            # This is specifically allowed for DataUnit, Link, Trigger classes
            try:
                # Enable direct instantiation for config creation
                LinkConfig._allow_direct_instantiation = True
                config_object = LinkConfig(**config)
            finally:
                LinkConfig._allow_direct_instantiation = False
        elif isinstance(config, LinkConfig):
            # Already a LinkConfig object
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
                LinkConfig._allow_direct_instantiation = True
                config_object = LinkConfig(**config_dict)
            finally:
                LinkConfig._allow_direct_instantiation = False

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

        nb_logger.info(f"Successfully created {cls.__name__}")
        return instance

    @classmethod
    def extract_component_config(cls, config: LinkConfig) -> Dict[str, Any]:
        """Extract DirectLink configuration"""
        return {
            'source': config.source,
            'target': config.target,
            'link_type': config.link_type,
            'buffer_size': getattr(config, 'buffer_size', 100),
            'data_mapping': getattr(config, 'data_mapping', None),
            'auto_transfer': getattr(config, 'auto_transfer', False)
        }

    @classmethod
    def resolve_dependencies(cls, component_config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Resolve DirectLink dependencies - extract source/target from component_config"""
        return {
            'source': component_config.get('source'),
            'target': component_config.get('target'),
            'enable_logging': kwargs.get('enable_logging', True),
            'debug_mode': kwargs.get('debug_mode', False)
        }

    def _init_from_config(self, config: LinkConfig, component_config: Dict[str, Any],
                          dependencies: Dict[str, Any]) -> None:
        """Initialize DirectLink with resolved dependencies"""
        # Call parent _init_from_config
        super()._init_from_config(config, component_config, dependencies)

        # Set auto_transfer from component configuration
        self.auto_transfer = component_config.get('auto_transfer', False)

        # Debug logging to check configuration
        if self.enable_logging and self.nb_logger:
            self.nb_logger.debug(
                f"DirectLink {self.name} config: auto_transfer={self.auto_transfer}, "
                f"component_config={component_config}")
            # Also check the raw config object
            if hasattr(config, 'auto_transfer'):
                self.nb_logger.debug(
                    f"DirectLink {self.name} raw config.auto_transfer={config.auto_transfer}")
            else:
                self.nb_logger.debug(
                    f"DirectLink {self.name} raw config has no auto_transfer attribute")

        # ✅ CRITICAL: Validate against self-referencing links during initialization
        if hasattr(self, '_source') and hasattr(self, '_target') and self._source and self._target:
            source_name = getattr(self._source, 'name', str(self._source))
            target_name = getattr(self._target, 'name', str(self._target))

            if (self._source is self._target or
                (hasattr(self._source, 'name') and hasattr(self._target, 'name') and
                 source_name == target_name)):
                error_msg = (
                    f"❌ ILLEGAL SELF-REFERENCING LINK: DirectLink cannot connect "
                    f"data unit '{source_name}' to itself. Self-referencing links are "
                    f"prohibited in the workflow architecture as they create infinite "
                    f"trigger loops and prevent proper workflow execution.")
                if hasattr(self, 'nb_logger') and self.nb_logger:
                    self.nb_logger.error(error_msg)
                raise ValueError(error_msg)

    async def start(self) -> None:
        """Start the direct link."""
        # EVENT-DRIVEN ARCHITECTURE: Start automatic transfer trigger first
        await super().start()

        # CRITICAL FIX: Setup automatic transfer callback registration
        # This ensures DirectLink callbacks are registered with source data units
        self._setup_callback_registration()

        # Verify callback registration was successful
        if self._verify_callback_registration():
            if self.enable_logging and self.nb_logger:
                self.nb_logger.info(f"✅ Callback registration verified: {self.name}")
        else:
            if self.enable_logging and self.nb_logger:
                self.nb_logger.warning(f"⚠️ Callback registration not verified: {self.name}")

        self._is_active = True
        logger.debug(f"DirectLink {self.name} started")

        # Log start event
        if self.enable_logging and self.nb_logger:
            self.nb_logger.info(f"Link started: {self.name}",
                                operation="start",
                                internal_state=self._get_internal_state())

    async def stop(self) -> None:
        """Stop the direct link."""
        # EVENT-DRIVEN ARCHITECTURE: Stop automatic transfer trigger first
        await super().stop()

        self._is_active = False
        logger.debug(f"DirectLink {self.name} stopped")

        # Log stop event with final statistics
        if self.enable_logging and self.nb_logger:
            uptime = time.time() - self._creation_time
            self.nb_logger.info(f"Link stopped: {self.name}",
                                operation="stop",
                                uptime_seconds=uptime,
                                final_stats={
                "total_transfers": self._transfer_count,
                "total_errors": self._error_count,
                "success_rate": self._transfer_count / max(self._transfer_count + self._error_count, 1)
            },
                internal_state=self._get_internal_state())

    async def transfer(self, data: Any) -> None:
        """Transfer data directly to target."""
        if not self._is_active:
            logger.warning(f"DirectLink {self.name} not active")
            return

        start_time = time.time()
        transfer_method = None

        try:
            # Prepare data info for logging
            data_info = {
                "data_type": type(data).__name__ if data is not None else "None",
                "size_bytes": len(str(data)) if data is not None else 0,
                "data_value": data if self.enable_logging else None
            }

            # DEBUG: Log target information
            if self.enable_logging and self.nb_logger:
                self.nb_logger.info(
                    f"🎯 Transfer target info for {self.name}: type={type(self.target).__name__}, "
                    f"has_set={hasattr(self.target, 'set')}, "
                    f"set_callable={callable(getattr(self.target, 'set', None))}")

            # Handle different target types
            if hasattr(self.target, 'set') and callable(getattr(self.target, 'set')):
                # Target is a DataUnit - use async non-blocking transfer
                transfer_method = "DataUnit.set"
                if self.enable_logging and self.nb_logger:
                    self.nb_logger.info(
                        f"📝 Initiating async transfer for {self.name}")

                # Use async non-blocking transfer to prevent deadlocks
                await self._transfer_to_dataunit_async(data)

                if self.enable_logging and self.nb_logger:
                    self.nb_logger.info(
                        f"✅ Async transfer initiated for {self.name}")

                logger.debug(
                    f"DirectLink {self.name} initiated async transfer to DataUnit")
            elif hasattr(self.target, 'input_data_units') and self.target.input_data_units:
                # Target is a Step with input data units
                transfer_method = "Step.input_data_units"
                for input_unit in self.target.input_data_units.values():
                    await input_unit.set(data)
                logger.debug(
                    f"DirectLink {self.name} transferred data to Step input units")
            elif hasattr(self.target, 'execute') and callable(getattr(self.target, 'execute')):
                # Target is a Step - trigger execution
                transfer_method = "Step.execute"
                await self.target.execute(data)
                logger.debug(f"DirectLink {self.name} executed target Step")
            elif hasattr(self.target, 'set_input'):
                # Direct method call
                transfer_method = "set_input"
                await self.target.set_input(data)
                logger.debug(
                    f"DirectLink {self.name} called set_input on target")
            else:
                logger.warning(
                    f"Target {getattr(self.target, 'name', str(self.target))} has no compatible input mechanism")

                # Log failed transfer attempt
                if self.enable_logging and self.nb_logger:
                    self.nb_logger.warning(f"Transfer failed - no compatible target mechanism: {self.name}",
                                           operation="transfer_failed",
                                           reason="no_compatible_mechanism",
                                           target_type=type(
                                               self.target).__name__,
                                           data_info=data_info)
                return

            # Calculate duration and record success
            duration_ms = (time.time() - start_time) * 1000
            data_info["transfer_method"] = transfer_method
            await self._record_transfer(True, duration_ms, data_info)
            logger.debug(
                f"DirectLink {self.name} transferred data successfully")

        except Exception as e:
            # Calculate duration and record failure
            duration_ms = (time.time() - start_time) * 1000
            data_info = {
                "data_type": type(data).__name__ if data is not None else "None",
                "size_bytes": len(str(data)) if data is not None else 0,
                "transfer_method": transfer_method,
                "error": str(e),
                "error_type": type(e).__name__
            }

            await self._record_transfer(False, duration_ms, data_info)
            logger.error(f"DirectLink {self.name} transfer failed: {e}")

            # Log detailed error information
            if self.enable_logging and self.nb_logger:
                self.nb_logger.error(f"Transfer error in link: {self.name}",
                                     operation="transfer_error",
                                     error=str(e),
                                     error_type=type(e).__name__,
                                     duration_ms=duration_ms,
                                     data_info=data_info,
                                     internal_state=self._get_internal_state())
            raise

    async def _transfer_to_dataunit_async(self, data: Any) -> None:
        """Transfer data to DataUnit using async non-blocking approach."""
        try:
            # Import AsyncTriggerExecutor here to avoid circular imports
            from .trigger import AsyncTriggerExecutor

            # Get async executor instance
            async_executor = await AsyncTriggerExecutor.get_instance()

            # Create async task for DataUnit.set() operation
            transfer_task = asyncio.create_task(
                self._execute_dataunit_set_async(data)
            )

            # Add to background tasks for tracking
            async_executor.background_tasks.add(transfer_task)
            transfer_task.add_done_callback(
                async_executor.background_tasks.discard)

            # Log async transfer initiation
            if self.enable_logging and self.nb_logger:
                self.nb_logger.info(
                    f"🚀 Async DataUnit transfer initiated: {self.name}",
                    operation="async_transfer_start",
                    target_name=getattr(self.target, 'name', str(self.target))
                )

        except Exception as e:
            if self.enable_logging and self.nb_logger:
                self.nb_logger.error(
                    f"Failed to initiate async DataUnit transfer for {self.name}: {e}",
                    operation="async_transfer_error",
                    error=str(e)
                )
            raise

    async def _execute_dataunit_set_async(self, data: Any) -> None:
        """Execute DataUnit.set() in async context without blocking link operations."""
        try:
            # Yield control to allow link transfer to complete
            await asyncio.sleep(0)

            # Execute the DataUnit.set() operation - DirectLink is PURE TRANSPORT
            await self.target.set(data)

            # Log successful completion
            if self.enable_logging and self.nb_logger:
                self.nb_logger.info(
                    f"✅ Async DataUnit transfer completed: {self.name}",
                    operation="async_transfer_complete",
                    target_name=getattr(self.target, 'name', str(self.target))
                )

        except Exception as e:
            if self.enable_logging and self.nb_logger:
                self.nb_logger.error(
                    f"❌ Async DataUnit transfer error: {self.name} - {e}",
                    operation="async_transfer_execution_error",
                    target_name=getattr(self.target, 'name', str(self.target)),
                    error=str(e)
                )
            # Don't re-raise here as this runs in background


class QueueLink(LinkBase):
    """
    Queue-based link that buffers data between source and target.
    """

    def __init__(self, source: Any, target: Any, config: Optional[LinkConfig] = None, **kwargs):
        config = config or LinkConfig(link_type=LinkType.QUEUE)
        super().__init__(source, target, config, **kwargs)
        self._queue: Optional[asyncio.Queue] = None
        self._consumer_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the queue link."""
        if self._is_active:
            return

        self._queue = asyncio.Queue(maxsize=self.config.buffer_size)
        self._consumer_task = asyncio.create_task(self._consume_queue())
        self._is_active = True
        logger.debug(
            f"QueueLink {self.name} started with buffer size {self.config.buffer_size}")

    async def stop(self) -> None:
        """Stop the queue link."""
        self._is_active = False

        if self._consumer_task and not self._consumer_task.done():
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except asyncio.CancelledError:
                pass

        self._queue = None
        logger.debug(f"QueueLink {self.name} stopped")

    async def transfer(self, data: Any) -> None:
        """Add data to queue for transfer."""
        if not self._is_active or not self._queue:
            logger.warning(f"QueueLink {self.name} not active")
            return

        try:
            await self._queue.put(data)
            logger.debug(f"QueueLink {self.name} queued data")
        except Exception as e:
            await self._record_transfer(False)
            logger.error(f"QueueLink {self.name} queue failed: {e}")
            raise

    async def _consume_queue(self) -> None:
        """Consume data from queue and transfer to target."""
        try:
            while self._is_active and self._queue:
                try:
                    # Wait for data with timeout
                    data = await asyncio.wait_for(self._queue.get(), timeout=1.0)

                    # Transfer to target
                    if hasattr(self.target, 'input_data_units') and self.target.input_data_units:
                        input_unit = self.target.input_data_units[0]
                        await input_unit.set(data)
                    elif hasattr(self.target, 'set_input'):
                        await self.target.set_input(data)

                    await self._record_transfer(True)

                except asyncio.TimeoutError:
                    # Continue loop on timeout
                    continue
                except Exception as e:
                    await self._record_transfer(False)
                    logger.error(f"QueueLink {self.name} consumer error: {e}")

        except asyncio.CancelledError:
            logger.debug(f"QueueLink {self.name} consumer cancelled")


class TransformLink(LinkBase):
    """
    Link that transforms data before transferring to target.

    Enhanced 2026-04-23 with mandatory from_config pattern support:
    YAML loaders now resolve ``transform_function`` (a dotted string
    like ``"my_pkg.my_mod.my_func"``) via ``parse_transform_from_config``
    into a Python callable at link-construction time. Sync and async
    callables are both accepted; the transfer path dispatches on
    ``asyncio.iscoroutinefunction``.

    Example YAML::

        links:
          rename_step1_to_step3a:
            class: "nanobrain.core.link.TransformLink"
            config:
              link_type: "transform"
              source: "entity_extraction.entity_candidates_output"
              target: "synonym_cache_lookup.query_terms_input"
              transform_function: "apecx_integration.composition.transforms.entities_to_query_terms"
    """

    COMPONENT_TYPE = "transform_link"
    REQUIRED_CONFIG_FIELDS = ['link_type', 'transform_function']
    OPTIONAL_CONFIG_FIELDS = {
        'buffer_size': 100,
        'data_mapping': None,
    }

    def __init__(self, *args, **kwargs):
        """Prevent direct instantiation — use from_config instead.

        Matches the framework's mandatory-from_config policy; same shape
        as DirectLink / ConditionalLink. Pre-2026-04-23 code that called
        ``TransformLink(source, target, transform_func)`` directly must
        migrate to ``TransformLink.from_config({...})``.
        """
        raise RuntimeError(
            "Direct instantiation of TransformLink is prohibited. "
            "ALL framework components must use TransformLink.from_config() "
            "as per mandatory framework requirements."
        )

    @classmethod
    def from_config(cls, config: Union[str, Path, LinkConfig, Dict[str, Any]], **kwargs) -> 'TransformLink':
        """Mandatory from_config implementation for TransformLink."""
        nb_logger = get_logger(f"{cls.__name__}.from_config")
        nb_logger.info(f"Creating {cls.__name__} from configuration")

        # Step 1: Normalize input to LinkConfig object (mirrors ConditionalLink).
        if isinstance(config, (str, Path)):
            config_object = LinkConfig.from_config(config, **kwargs)
        elif isinstance(config, dict):
            try:
                LinkConfig._allow_direct_instantiation = True
                config_object = LinkConfig(**config)
            finally:
                LinkConfig._allow_direct_instantiation = False
        elif isinstance(config, LinkConfig):
            config_object = config
        else:
            if hasattr(config, 'model_dump'):
                config_dict = config.model_dump()
            elif hasattr(config, 'dict'):
                config_dict = config.dict()
            else:
                raise ValueError(f"Unsupported config type: {type(config)}")
            try:
                LinkConfig._allow_direct_instantiation = True
                config_object = LinkConfig(**config_dict)
            finally:
                LinkConfig._allow_direct_instantiation = False

        # Step 2: Validate configuration schema.
        cls.validate_config_schema(config_object)

        # Step 3: Extract component-specific configuration.
        component_config = cls.extract_component_config(config_object)

        # Step 4: Resolve dependencies (string → callable happens here).
        dependencies = cls.resolve_dependencies(component_config, **kwargs)

        # Step 5: Create instance.
        instance = cls.create_instance(
            config_object, component_config, dependencies)

        # Step 6: Post-creation initialization.
        instance._post_config_initialization()

        nb_logger.info(f"Successfully created {cls.__name__}")
        return instance

    @classmethod
    def extract_component_config(cls, config: LinkConfig) -> Dict[str, Any]:
        """Extract TransformLink configuration."""
        return {
            'source': config.source,
            'target': config.target,
            'link_type': config.link_type,
            'transform_function': getattr(config, 'transform_function', None),
            'buffer_size': getattr(config, 'buffer_size', 100),
            'data_mapping': getattr(config, 'data_mapping', None),
            'auto_transfer': getattr(config, 'auto_transfer', False),
        }

    @classmethod
    def resolve_dependencies(cls, component_config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Resolve TransformLink dependencies: string → Python callable."""
        transform_spec = component_config.get('transform_function')
        if not transform_spec:
            raise ComponentConfigurationError(
                "TransformLink requires 'transform_function' in its config — "
                "a fully-qualified dotted path like 'my_pkg.my_mod.my_func'"
            )

        # Accept a callable directly for tests / in-process callers that
        # skip the YAML loader. (Not reachable from a YAML file — that
        # pins ``transform_function`` to a string via LinkConfig.)
        if callable(transform_spec):
            transform_func = transform_spec
        else:
            transform_func = parse_transform_from_config(transform_spec)

        return {
            'source': component_config.get('source'),
            'target': component_config.get('target'),
            'transform_func': transform_func,
            'enable_logging': kwargs.get('enable_logging', True),
            'debug_mode': kwargs.get('debug_mode', False),
        }

    def _init_from_config(self, config: LinkConfig, component_config: Dict[str, Any],
                          dependencies: Dict[str, Any]) -> None:
        """Initialize TransformLink with resolved dependencies."""
        super()._init_from_config(config, component_config, dependencies)
        self.transform_func = dependencies['transform_func']

    async def start(self) -> None:
        """Start the transform link."""
        self._is_active = True
        logger.debug(f"TransformLink {self.name} started")

    async def stop(self) -> None:
        """Stop the transform link."""
        self._is_active = False
        logger.debug(f"TransformLink {self.name} stopped")

    async def transfer(self, data: Any) -> None:
        """Transform data and transfer to target."""
        if not self._is_active:
            logger.warning(f"TransformLink {self.name} not active")
            return

        try:
            # Apply transformation — supports both sync and async callables.
            if asyncio.iscoroutinefunction(self.transform_func):
                transformed_data = await self.transform_func(data)
            else:
                transformed_data = self.transform_func(data)

            # Transfer transformed data.
            if hasattr(self.target, 'input_data_units') and self.target.input_data_units:
                input_unit = self.target.input_data_units[0]
                await input_unit.set(transformed_data)
            elif hasattr(self.target, 'set_input'):
                await self.target.set_input(transformed_data)

            await self._record_transfer(True)
            logger.debug(
                f"TransformLink {self.name} transformed and transferred data")

        except Exception as e:
            await self._record_transfer(False)
            logger.error(f"TransformLink {self.name} transform failed: {e}")
            raise


class ConditionalLink(LinkBase):
    """
    Link that transfers data only when condition is met.
    Enhanced with mandatory from_config pattern implementation.

    G10 — gate-to-bottom semantics
    ------------------------------
    See ``apecx-mcp-integration/docs/nanobrain_capability_gaps.md G10``.
    When ``gate_semantics == 'publish_empty'`` (default; legacy), a False
    condition is a no-op — the target data unit is left untouched.
    Downstream ``AllDataReceivedTrigger`` will deadlock unless the upstream
    step explicitly publishes an empty bundle.

    When ``gate_semantics == 'gate_to_bottom'``, a False condition writes
    the magic string :attr:`GATED_OFF_SENTINEL` to the target. An
    ``AllDataReceivedTrigger`` configured with the same flag treats
    sentinel-bearing units as satisfied AND excludes them from the
    payload it forwards — fan-in proceeds with N-1 keys rather than
    waiting indefinitely. User ``process()`` code never sees the magic
    string; only the trigger-satisfaction layer does.
    """

    COMPONENT_TYPE = "conditional_link"
    REQUIRED_CONFIG_FIELDS = ['link_type', 'condition']
    OPTIONAL_CONFIG_FIELDS = {
        'buffer_size': 100,
        'data_mapping': None,
        'gate_semantics': 'publish_empty',
    }

    # G10 sentinel. Distinct from None and any user payload. Reserved
    # string; user code MUST NOT publish this value to a data unit.
    GATED_OFF_SENTINEL = "__nanobrain_gated_off__"

    def __init__(self, *args, **kwargs):
        """Prevent direct instantiation - use from_config instead"""
        raise RuntimeError(
            "Direct instantiation of ConditionalLink is prohibited. "
            "ALL framework components must use ConditionalLink.from_config() "
            "as per mandatory framework requirements."
        )

    @classmethod
    def from_config(cls, config: Union[str, Path, LinkConfig, Dict[str, Any]], **kwargs) -> 'ConditionalLink':
        """Mandatory from_config implementation for ConditionalLink with dictionary support"""
        # Get logger
        nb_logger = get_logger(f"{cls.__name__}.from_config")
        nb_logger.info(f"Creating {cls.__name__} from configuration")

        # Step 1: Normalize input to LinkConfig object
        if isinstance(config, (str, Path)):
            # File path input - use standard config loading
            config_object = LinkConfig.from_config(config, **kwargs)
        elif isinstance(config, dict):
            # Dictionary input - create LinkConfig from dict (inline config support)
            # This is specifically allowed for DataUnit, Link, Trigger classes
            try:
                # Enable direct instantiation for config creation
                LinkConfig._allow_direct_instantiation = True
                config_object = LinkConfig(**config)
            finally:
                LinkConfig._allow_direct_instantiation = False
        elif isinstance(config, LinkConfig):
            # Already a LinkConfig object
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
                LinkConfig._allow_direct_instantiation = True
                config_object = LinkConfig(**config_dict)
            finally:
                LinkConfig._allow_direct_instantiation = False

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

        nb_logger.info(f"Successfully created {cls.__name__}")
        return instance

    @classmethod
    def extract_component_config(cls, config: LinkConfig) -> Dict[str, Any]:
        """Extract ConditionalLink configuration"""
        gate_semantics = getattr(config, 'gate_semantics', 'publish_empty')
        if gate_semantics not in ('publish_empty', 'gate_to_bottom'):
            raise ValueError(
                f"FAIL-FAST: ConditionalLink gate_semantics must be "
                f"'publish_empty' or 'gate_to_bottom', got {gate_semantics!r}"
            )
        return {
            'source': config.source,
            'target': config.target,
            'link_type': config.link_type,
            'condition': getattr(config, 'condition', None),
            'buffer_size': getattr(config, 'buffer_size', 100),
            'data_mapping': getattr(config, 'data_mapping', None),
            'auto_transfer': getattr(config, 'auto_transfer', False),
            'gate_semantics': gate_semantics,
        }

    @classmethod
    def resolve_dependencies(cls, component_config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Resolve ConditionalLink dependencies"""
        condition_config = component_config.get('condition')
        if not condition_config:
            raise ValueError(
                "ConditionalLink requires condition configuration")

        # Parse condition function from config
        condition_func = parse_condition_from_config(condition_config)

        return {
            'source': component_config.get('source'),
            'target': component_config.get('target'),
            'condition_func': condition_func,
            'enable_logging': kwargs.get('enable_logging', True),
            'debug_mode': kwargs.get('debug_mode', False)
        }

    def _init_from_config(self, config: LinkConfig, component_config: Dict[str, Any],
                          dependencies: Dict[str, Any]) -> None:
        """Initialize ConditionalLink with resolved dependencies"""
        # Call parent _init_from_config
        super()._init_from_config(config, component_config, dependencies)

        # Set ConditionalLink-specific attributes
        self.condition_func = dependencies['condition_func']
        self.gate_semantics = component_config.get(
            'gate_semantics', 'publish_empty'
        )

    async def start(self) -> None:
        """Start the conditional link."""
        self._is_active = True
        logger.debug(f"ConditionalLink {self.name} started")

    async def stop(self) -> None:
        """Stop the conditional link."""
        self._is_active = False
        logger.debug(f"ConditionalLink {self.name} stopped")

    async def transfer(self, data: Any) -> None:
        """Transfer data if condition is met."""
        if not self._is_active:
            logger.warning(f"ConditionalLink {self.name} not active")
            return

        try:
            # Check condition
            if asyncio.iscoroutinefunction(self.condition_func):
                should_transfer = await self.condition_func(data)
            else:
                should_transfer = self.condition_func(data)

            if should_transfer:
                # Transfer data
                if hasattr(self.target, 'input_data_units') and self.target.input_data_units:
                    input_unit = self.target.input_data_units[0]
                    await input_unit.set(data)
                elif hasattr(self.target, 'set_input'):
                    await self.target.set_input(data)

                await self._record_transfer(True)
                logger.debug(
                    f"ConditionalLink {self.name} condition met, transferred data")
            elif self.gate_semantics == 'gate_to_bottom':
                # G10: write the gated-off sentinel to the target so the
                # downstream AllDataReceivedTrigger can fire without waiting
                # indefinitely. The trigger's _is_satisfied helper recognizes
                # the sentinel and excludes it from the payload it forwards.
                sentinel = ConditionalLink.GATED_OFF_SENTINEL
                if hasattr(self.target, 'input_data_units') and self.target.input_data_units:
                    input_unit = self.target.input_data_units[0]
                    await input_unit.set(sentinel)
                elif hasattr(self.target, 'set_input'):
                    await self.target.set_input(sentinel)
                else:
                    # The target is itself a data unit (no .set_input,
                    # no .input_data_units list) — set on it directly.
                    await self.target.set(sentinel)

                await self._record_transfer(True)
                logger.debug(
                    f"ConditionalLink {self.name} condition not met, "
                    f"wrote GATED_OFF_SENTINEL to target under gate_to_bottom"
                )
            else:
                # Legacy 'publish_empty' semantics — silent no-op. The
                # downstream AllDataReceivedTrigger will block until the
                # upstream step publishes an empty bundle. This is the
                # documented silent-failure shape that gate_to_bottom is
                # designed to eliminate.
                logger.debug(
                    f"ConditionalLink {self.name} condition not met, skipped transfer "
                    f"(gate_semantics='publish_empty')")

        except Exception as e:
            await self._record_transfer(False)
            logger.error(
                f"ConditionalLink {self.name} condition check failed: {e}")
            raise


class FileLink(LinkBase):
    """
    File-based link that transfers data through file system.
    """

    def __init__(self, source: Any, target: Any, file_path: str,
                 config: Optional[LinkConfig] = None, **kwargs):
        config = config or LinkConfig(
            link_type=LinkType.FILE, file_path=file_path)
        super().__init__(source, target, config, **kwargs)
        self.file_path = file_path
        self._monitor_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the file link."""
        if self._is_active:
            return

        self._monitor_task = asyncio.create_task(self._monitor_file())
        self._is_active = True
        logger.debug(
            f"FileLink {self.name} started monitoring {self.file_path}")

    async def stop(self) -> None:
        """Stop the file link."""
        self._is_active = False

        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        logger.debug(f"FileLink {self.name} stopped")

    async def transfer(self, data: Any) -> None:
        """Write data to file for transfer."""
        if not self._is_active:
            logger.warning(f"FileLink {self.name} not active")
            return

        try:
            from pathlib import Path
            import json

            file_path = Path(self.file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Write data to file (NON-BLOCKING)
            if isinstance(data, (dict, list)):
                content = json.dumps(data, indent=2)
            else:
                content = str(data)

            async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                await f.write(content)
            logger.debug(f"FileLink {self.name} wrote data to file")

        except Exception as e:
            await self._record_transfer(False)
            logger.error(f"FileLink {self.name} file write failed: {e}")
            raise

    async def _monitor_file(self) -> None:
        """Monitor file for changes and transfer to target."""
        from pathlib import Path
        import json

        file_path = Path(self.file_path)
        last_modified = 0.0

        try:
            while self._is_active:
                try:
                    if file_path.exists():
                        current_modified = file_path.stat().st_mtime

                        if current_modified > last_modified:
                            last_modified = current_modified

                            # Read and transfer data
                            content = file_path.read_text()
                            try:
                                data = json.loads(content)
                            except json.JSONDecodeError:
                                data = content

                            # Transfer to target
                            if hasattr(self.target, 'input_data_units') and self.target.input_data_units:
                                input_unit = self.target.input_data_units[0]
                                await input_unit.set(data)
                            elif hasattr(self.target, 'set_input'):
                                await self.target.set_input(data)

                            await self._record_transfer(True)
                            logger.debug(
                                f"FileLink {self.name} detected file change and transferred")

                    await asyncio.sleep(1.0)  # Check every second

                except Exception as e:
                    await self._record_transfer(False)
                    logger.error(f"FileLink {self.name} monitor error: {e}")
                    await asyncio.sleep(5.0)  # Wait longer on error

        except asyncio.CancelledError:
            logger.debug(f"FileLink {self.name} monitor cancelled")
