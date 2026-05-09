"""ToolExecutionStep (G11) — tool-step taxonomy.

Per ``apecx-mcp-integration/docs/nanobrain_capability_gaps.md G11`` and
``apecx-mcp-integration/docs/external_tool_integration.md``: tool dispatch
is a Step concern (not an Agent concern — see audit finding F-4).
``ToolExecutionStep`` is the BaseStep subclass that consumes a UTD
reference (G15) and dispatches to a registered backend adapter.

This module ships the framework primitives:

- :class:`ToolBackendAdapter` — abstract base class for backend adapters
- :class:`ToolBackendRegistry` — process-global adapter registry by name
- :class:`ToolExecutionStep` — the BaseStep that resolves UTD → adapter → result

What this module does NOT do:

- Concrete backend adapters (RheaAdapter, GalaxyAdapter, LocalParslAdapter)
  live elsewhere — Rhea adapter is shipped from the Rhea fork
  (Track C T-RH-04); Galaxy is deferred until availability; LocalParsl
  is implementation work that depends on the workflow's executor
  configuration. We ship the protocol; concrete adapters land
  separately.

Workspace constraints:

- ToolExecutionStep follows the from_config + COMPONENT_TYPE +
  _get_config_class pattern matching LoopController + ApprovalStep.
- The output schema (G6) is automatically derived from the UTD's
  outputs list — no need for the caller to redeclare.
- Result-typing (per gap proposal): UTD output type → DataUnit class
  mapping is handled by the framework's existing data unit system; the
  caller chooses which DataUnit subclass to use for each output.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, List, Optional, Tuple

from pydantic import Field, model_validator

from nanobrain.core.component_base import ComponentConfigurationError
from nanobrain.core.step import BaseStep, StepConfig
from nanobrain.core.unified_tool_descriptor import UnifiedToolDescriptor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backend adapter protocol
# ---------------------------------------------------------------------------

class ToolBackendAdapter(ABC):
    """Abstract base class for tool backend adapters.

    Each concrete adapter wraps one tool-execution backend (Rhea, Galaxy,
    LocalParsl, etc). The adapter knows how to translate a UTD invocation
    into a backend-specific call and translate the backend's response
    back into the framework's standard result shape.

    Concrete adapters ship in their respective integration packages —
    e.g., ``rhea.extensions.apecx_utd_extension.RheaAdapter`` (Track C),
    or apecx-mcp-side adapters for local execution.
    """

    #: Backend identifier — matches the UTD descriptor_id's `<backend>:` prefix.
    #: Concrete adapters MUST set this. Used by the registry to look up
    #: the right adapter for a given UTD.
    BACKEND_NAME: ClassVar[str] = ""

    @abstractmethod
    async def invoke(
        self,
        utd: UnifiedToolDescriptor,
        inputs: Dict[str, Any],
        *,
        run_context_namespace: str = "",
        **kwargs,
    ) -> Dict[str, Any]:
        """Invoke the tool described by ``utd`` with ``inputs``.

        Args:
            utd: The fully-resolved tool descriptor (already validated;
                the adapter MUST NOT re-validate the UTD shape).
            inputs: Tool input parameter dict, keyed by UTD input names.
                Adapter implementations validate against the UTD's
                ``inputs`` list (every required input must be present
                or raise ``ComponentConfigurationError("FAIL-FAST: ...")``).
            run_context_namespace: The current G13 WorkflowRunContext
                namespace (or empty string when no context is active).
                Adapters that talk to a shared ProxyStore use this for
                per-tenant isolation.
            **kwargs: Backend-specific options (Rhea: SSE callback URL;
                Galaxy: history_id; etc).

        Returns:
            Dict keyed by UTD output names. Values are typed per the
            UTD's ``outputs`` list — small payloads inline; large
            payloads as ProxyStore keys.

        Raises:
            ComponentConfigurationError: on UTD/inputs mismatch or
                backend-side configuration error.
            RuntimeError: on transient backend error (timeout, connection
                refused). The framework's retry/circuit-breaker layer
                handles these.
        """
        ...


class ToolBackendRegistry:
    """Process-global registry mapping ``backend_name`` → ``ToolBackendAdapter``.

    Adapters register themselves at import time (or via
    ``register_backend_adapter`` from a startup hook). ``ToolExecutionStep``
    consults the registry to find the right adapter for a given UTD.

    The registry is a ClassVar — there is exactly one per process. Tests
    that want isolation use ``with registry.scope(): ...`` (a context
    manager that snapshots and restores).
    """

    _adapters: ClassVar[Dict[str, ToolBackendAdapter]] = {}

    @classmethod
    def register(cls, adapter: ToolBackendAdapter) -> None:
        """Register an adapter. Re-registering the same name FAIL-FASTs
        (different adapter for the same backend = ambiguous dispatch)."""
        if not adapter.BACKEND_NAME:
            raise ComponentConfigurationError(
                f"FAIL-FAST: ToolBackendAdapter {type(adapter).__name__} "
                f"must set BACKEND_NAME (matched against UTD descriptor_id "
                f"backend prefix)"
            )
        if adapter.BACKEND_NAME in cls._adapters:
            existing = cls._adapters[adapter.BACKEND_NAME]
            if existing is not adapter:
                raise ComponentConfigurationError(
                    f"FAIL-FAST: backend {adapter.BACKEND_NAME!r} already "
                    f"registered with adapter {type(existing).__name__}; "
                    f"cannot replace with {type(adapter).__name__}"
                )
        cls._adapters[adapter.BACKEND_NAME] = adapter

    @classmethod
    def get(cls, backend_name: str) -> ToolBackendAdapter:
        """Look up an adapter by backend name. Raises KeyError when missing."""
        if backend_name not in cls._adapters:
            available = sorted(cls._adapters.keys())
            raise KeyError(
                f"FAIL-FAST: no ToolBackendAdapter registered for backend "
                f"{backend_name!r}; available: {available}"
            )
        return cls._adapters[backend_name]

    @classmethod
    def unregister(cls, backend_name: str) -> None:
        """Remove an adapter. Used by tests for cleanup."""
        cls._adapters.pop(backend_name, None)

    @classmethod
    def list_backends(cls) -> List[str]:
        return list(cls._adapters.keys())


# ---------------------------------------------------------------------------
# ToolExecutionStep
# ---------------------------------------------------------------------------

class ToolExecutionStepConfig(StepConfig):
    """Configuration for ToolExecutionStep.

    Extends StepConfig with a single required nested block: the
    ``tool_descriptor`` (a full UTD) OR a ``tool_descriptor_path`` (a
    path string the loader resolves). Exactly one must be set; the
    model validator enforces this.
    """
    tool_descriptor: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Inline UTD as a dict. Mutually exclusive with "
                    "tool_descriptor_path.",
    )
    tool_descriptor_path: Optional[str] = Field(
        default=None,
        description="Path to a YAML file containing the UTD. Mutually "
                    "exclusive with tool_descriptor.",
    )

    # Optional invocation-level kwargs that the adapter sees as **kwargs.
    backend_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Backend-specific extra kwargs forwarded to "
                    "adapter.invoke(). Adapter docs name the supported "
                    "keys per backend.",
    )

    @model_validator(mode="after")
    def _exactly_one_descriptor_source(self) -> "ToolExecutionStepConfig":
        has_inline = self.tool_descriptor is not None
        has_path = self.tool_descriptor_path is not None
        if has_inline == has_path:
            raise ValueError(
                "FAIL-FAST: ToolExecutionStepConfig requires EXACTLY ONE of "
                "tool_descriptor (inline UTD dict) or tool_descriptor_path "
                "(YAML file path)"
            )
        return self


class ToolExecutionStep(BaseStep):
    """G11 — base class for steps that dispatch to a tool backend.

    Lifecycle per ``process(input_data)``:

    1. Resolve the UTD (inline dict or path) into a typed
       ``UnifiedToolDescriptor`` — validation happens here.
    2. Look up the backend adapter from the registry using the UTD's
       descriptor_backend.
    3. Resolve the current WorkflowRunContext namespace (G13) for the
       adapter's per-tenant isolation.
    4. Call ``adapter.invoke(utd, input_data, run_context_namespace, **backend_kwargs)``.
    5. Return the adapter's result dict — keys match the UTD's outputs.

    The G6 step output schema is auto-derived from the UTD's outputs
    list (each output name → its declared type). The framework's
    _execute_process wrapper (G6) validates the result before it ships
    to the next step.

    Subclassing:

    Concrete steps that wrap a specific tool can subclass to customize
    input pre-processing or output post-processing. The default
    implementation is fine for most cases.
    """

    COMPONENT_TYPE: str = "tool_execution_step"
    REQUIRED_CONFIG_FIELDS = ["name"]

    @classmethod
    def _get_config_class(cls):
        return ToolExecutionStepConfig

    def _init_from_config(
        self,
        config: ToolExecutionStepConfig,
        component_config: Dict[str, Any],
        dependencies: Dict[str, Any],
    ) -> None:
        super()._init_from_config(config, component_config, dependencies)

        # Resolve the UTD once at step init — the descriptor is immutable
        # for the step's lifetime. Re-resolution per process() call would
        # be a perf cost AND a contract drift risk (the UTD might change
        # behind the step's back). UnifiedToolDescriptor.from_dict()
        # handles the nested-model direct-instantiation admittance.
        if config.tool_descriptor is not None:
            try:
                self._utd = UnifiedToolDescriptor.from_dict(config.tool_descriptor)
            except Exception as e:
                raise ComponentConfigurationError(
                    f"FAIL-FAST: ToolExecutionStep {self.name!r} inline "
                    f"tool_descriptor failed UTD shape: {e}"
                ) from e
        else:
            # Path-based UTD load — read the YAML file.
            from pathlib import Path
            import yaml

            path = Path(config.tool_descriptor_path)
            if not path.is_file():
                raise ComponentConfigurationError(
                    f"FAIL-FAST: ToolExecutionStep {self.name!r} "
                    f"tool_descriptor_path {path} not found"
                )
            try:
                utd_data = yaml.safe_load(path.read_text())
            except yaml.YAMLError as e:
                raise ComponentConfigurationError(
                    f"FAIL-FAST: ToolExecutionStep {self.name!r} "
                    f"tool_descriptor_path {path} failed YAML parse: {e}"
                ) from e
            try:
                self._utd = UnifiedToolDescriptor.from_dict(utd_data)
            except Exception as e:
                raise ComponentConfigurationError(
                    f"FAIL-FAST: ToolExecutionStep {self.name!r} UTD at "
                    f"{path} failed shape: {e}"
                ) from e

        self._backend_kwargs = config.backend_kwargs or {}

    @property
    def utd(self) -> UnifiedToolDescriptor:
        """The resolved UnifiedToolDescriptor this step dispatches to."""
        return self._utd

    @property
    def backend_name(self) -> str:
        """Convenience: extract the backend prefix from the UTD."""
        return self._utd.descriptor_backend

    async def process(self, input_data: Any, **kwargs) -> Dict[str, Any]:
        """Dispatch the tool call.

        ``input_data`` is the dict of UTD inputs (keyed by UTD input
        names). The adapter validates required inputs are present.

        Returns the adapter's result dict, keyed by UTD output names.
        """
        if not isinstance(input_data, dict):
            raise ComponentConfigurationError(
                f"FAIL-FAST: ToolExecutionStep {self.name!r} input_data "
                f"must be dict (UTD input names → values), got "
                f"{type(input_data).__name__}"
            )

        # Look up the backend adapter.
        adapter = ToolBackendRegistry.get(self.backend_name)

        # Resolve the active run context namespace (G13). Lazy import to
        # avoid cycles (this module is library; run_context is library too,
        # but at module import time we want minimal coupling).
        ns = ""
        try:
            from nanobrain.library.orchestration.run_context import (
                current_run_context,
            )
            ctx = current_run_context()
            if ctx is not None:
                ns = ctx.proxystore_namespace
        except ImportError:
            pass

        # Merge step-level backend_kwargs with per-call kwargs — call
        # kwargs win (caller provides per-invocation overrides).
        merged_kwargs = dict(self._backend_kwargs)
        merged_kwargs.update(kwargs)

        return await adapter.invoke(
            self._utd, input_data,
            run_context_namespace=ns,
            **merged_kwargs,
        )
