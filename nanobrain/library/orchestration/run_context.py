"""WorkflowRunContext (G13) — per-run scope for multi-tenant isolation.

Per ``apecx-mcp-integration/docs/nanobrain_capability_gaps.md G13``: when
multiple workflow runs share a ProxyStore (Redis-backed in production),
keys from run A can collide with keys from run B without per-run scoping.
``WorkflowRunContext`` carries a ``run_id`` (UUIDv7 in production; any
unique string in tests) plus a derived ``proxystore_namespace`` that
``DataUnitProxyRef`` (and future provenance + checkpoint primitives)
consult for isolation.

Design:

- The context is a contextvar-managed singleton. A workflow's ``run()``
  method (G8) installs the context at run start and tears it down at
  end. Steps inside the run access the context via the module-level
  ``current_run_context()`` accessor.
- The G3 ``DataUnitProxyRef.namespace()`` API today returns a
  configured-at-load namespace prefix; G13 augments this by allowing
  the prefix to be REPLACED with the current run context's value when
  no static prefix is configured. Existing G3 consumers that pin a
  static prefix continue to win — explicit > implicit.
- The contextvar pattern (PEP 567) is asyncio-safe: each Task gets its
  own copy, so concurrent runs in the same event loop don't trample.
"""

from __future__ import annotations

import contextvars
import uuid
from datetime import datetime, timezone
from typing import Iterator, Optional

from contextlib import contextmanager

from pydantic import ConfigDict, Field

from nanobrain.core.config.config_base import ConfigBase
from nanobrain.core.component_base import FromConfigBase


# ---------------------------------------------------------------------------
# The context variable. Process-global module state (one per asyncio Task
# via contextvars). None when no run is active — callers must handle that
# case (no current run = no namespace prefix to apply).
# ---------------------------------------------------------------------------

_current_context: contextvars.ContextVar[Optional["WorkflowRunContext"]] = (
    contextvars.ContextVar("current_workflow_run_context", default=None)
)


class WorkflowRunContextConfig(ConfigBase):
    """Configuration for WorkflowRunContext.

    All fields are optional with sensible defaults — a context built with
    no config is valid (UUIDv4 run_id, default namespace template).
    """
    model_config = ConfigDict(extra="forbid")

    run_id: Optional[str] = Field(
        default=None,
        description="Stable identifier for this run. Defaults to a UUIDv4 "
                    "generated at construction time. Pass an explicit "
                    "value when integrating with an external job-id "
                    "system (e.g., a control-plane Run row's primary key).",
    )
    proxystore_namespace_template: str = Field(
        default="run_${run_id}",
        description="str.Template grammar; the only placeholder is "
                    "${run_id}. The expanded value is what "
                    "DataUnitProxyRef.namespace() returns when no static "
                    "namespace is configured on the data unit itself.",
    )


class WorkflowRunContext(FromConfigBase):
    """G13 — per-run scope carrying run_id + derived namespace.

    Built via standard from_config + lifecycle:

    .. code-block:: python

        ctx = WorkflowRunContext.from_config({"run_id": "abc123"})
        with ctx.activate():
            # Steps inside this block see the context.
            await workflow.run(input_data)
        # Outside the block, current_run_context() returns None.

    The activate() context manager is the canonical way to install the
    context. WorkflowRunner (gap G21) installs the context automatically
    around each detached run; explicit construction is for tests and
    advanced callers.
    """

    @classmethod
    def _get_config_class(cls):
        return WorkflowRunContextConfig

    @classmethod
    def from_config(cls, config=None, **kwargs) -> "WorkflowRunContext":
        """Standard from_config; accepts None (= default config), a dict,
        a Path/str (YAML file), or a WorkflowRunContextConfig instance."""
        from pathlib import Path
        if config is None:
            config = {}
        if isinstance(config, (str, Path)):
            config_object = WorkflowRunContextConfig.from_config(config, **kwargs)
        elif isinstance(config, dict):
            try:
                WorkflowRunContextConfig._allow_direct_instantiation = True
                config_object = WorkflowRunContextConfig(**config)
            finally:
                WorkflowRunContextConfig._allow_direct_instantiation = False
        elif isinstance(config, WorkflowRunContextConfig):
            config_object = config
        else:
            raise ValueError(f"Unsupported config type: {type(config)}")

        # Bypass FromConfigBase __new__ enforcement — this is the
        # canonical from_config entry, so we MUST be the ones constructing.
        cls._allow_direct_instantiation = True
        try:
            instance = cls()
        finally:
            cls._allow_direct_instantiation = False

        # Resolve run_id (default = UUIDv4 hex).
        instance._run_id = config_object.run_id or uuid.uuid4().hex
        instance._namespace_template = config_object.proxystore_namespace_template
        instance._start_time = datetime.now(timezone.utc)
        return instance

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def start_time(self) -> datetime:
        return self._start_time

    @property
    def proxystore_namespace(self) -> str:
        """The namespace prefix downstream consumers (DataUnitProxyRef,
        ProvenanceContext, CheckpointStep) use for per-run isolation.

        Computed by substituting ${run_id} into
        ``proxystore_namespace_template``.
        """
        from string import Template
        return Template(self._namespace_template).safe_substitute(
            run_id=self._run_id
        )

    @contextmanager
    def activate(self) -> Iterator["WorkflowRunContext"]:
        """Install this context as the current run context for the
        duration of the with-block. PEP 567 contextvars make this
        asyncio-safe: concurrent tasks each see their own context.
        """
        token = _current_context.set(self)
        try:
            yield self
        finally:
            _current_context.reset(token)


def current_run_context() -> Optional[WorkflowRunContext]:
    """Return the currently-active WorkflowRunContext, or None if no
    run is active. Steps that want to consult the current run's
    namespace call this from inside their ``process()`` method.
    """
    return _current_context.get()
