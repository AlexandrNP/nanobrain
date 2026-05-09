"""G22 — WorkflowEntryTrigger: wrap any TriggerBase to start a detached
workflow run via WorkflowRunner (G21).

Per ``apecx-mcp-integration/docs/nanobrain_capability_gaps.md G22``.

This module implements the *workflow-start* half of G22. The
*event-source* half (``EventTrigger``) lives in
``nanobrain.core.trigger`` because it is a TriggerBase subclass with no
dependency on the runtime layer; ``WorkflowEntryTrigger`` lives here
because it depends on G21's ``WorkflowRunner``.

G22 Step 4 (2026-05-09) — durable inner-trigger → launch binding:
    Adds an optional state-store hookup so the wrapper can persist
    the last-fire timestamp on every successful inner fire AND read
    it back at restart to drive ``replay_missed_fires`` (G22 Step 3).
    Two state-store backends ship: in-memory (for tests) and file
    (single JSON file per ``entry_id``). Production deployments wire
    the WorkflowRunner's task store or a Postgres backend through
    this interface.

Usage::

    runner = WorkflowRunner.from_config("config/runner.yml")

    inner = TimerTrigger.from_config(...)        # cron-shaped wakeup
    entry = WorkflowEntryTrigger.from_config(
        "config/entry.yml",
        runner=runner,
        workflow_callable=my_workflow.run,
    )
    await inner.start_monitoring()
    await entry.start()  # binds inner -> entry's launch_task

Each inner-trigger fire calls ``runner.run_detached`` with a freshly-built
payload (or ``{}`` when ``payload_factory`` is unset). Task IDs are
auto-generated as ``"{name}-{uuid4}"`` per fire; the caller receives them
via the optional ``on_launch`` callback.

v1 scope:

- The wrapper is constructed against a *concrete* runner + workflow_callable
  passed via kwargs (programmatic + via from_config kwargs).
- ``payload_factory`` is a dotted-path-resolved callable that takes the
  inner trigger event body and returns the payload dict for the workflow.
- ``autonomy_level`` and ``cost_envelope_template`` are accepted as
  forward-compatibility fields; they are passed through into the payload
  dict under keys ``__autonomy_level__`` and ``__cost_envelope_template__``
  so downstream consumers can read them. The framework does NOT enforce
  cost or HITL semantics at this layer (that is apecx-mcp-integration scope).

Deferred to follow-ups:

- Step 2: dotted-path resolution of ``target_workflow`` to a Workflow class
  (today the workflow callable is passed in as kwargs).
- Step 3: missed-schedule policy (``on_missed: catch_up | skip | merge``)
  for cron-shaped inner triggers.
- Step 4: durable inner trigger -> launch binding so a process restart
  re-establishes the binding without operator intervention.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Literal, Optional

from pydantic import ConfigDict, Field

from nanobrain.core.component_base import (
    ComponentConfigurationError,
    FromConfigBase,
)
from nanobrain.core.config.config_base import ConfigBase
from nanobrain.core.trigger import TriggerBase

from .workflow_runner import DetachedTaskHandle, WorkflowRunner


# ---------------------------------------------------------------------------
# G22 Step 4 — durable state store
# ---------------------------------------------------------------------------

class EntryStateStore:
    """Abstract durable state store for WorkflowEntryTrigger bindings.

    The store maps an opaque ``entry_id`` (the wrapper's name) to a
    small dict of bookkeeping fields:

        {
          "last_fire_epoch_seconds": float | None,
          "last_task_id": str | None,
        }

    Two implementations ship:
      - ``InMemoryEntryStateStore`` (tests; one process)
      - ``FileEntryStateStore`` (production; one JSON file per entry_id
        under a configured base directory)

    Production deployments that already have a Postgres / KV backend
    can subclass this and implement get/set/delete in two lines. The
    interface is async to keep the door open for I/O-bound backends.
    """

    async def get(self, entry_id: str) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    async def set(self, entry_id: str, state: Dict[str, Any]) -> None:
        raise NotImplementedError

    async def delete(self, entry_id: str) -> None:
        raise NotImplementedError


class InMemoryEntryStateStore(EntryStateStore):
    """Process-local dict, asyncio.Lock-serialized."""

    def __init__(self) -> None:
        self._states: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def get(self, entry_id: str) -> Optional[Dict[str, Any]]:
        async with self._lock:
            return dict(self._states[entry_id]) if entry_id in self._states else None

    async def set(self, entry_id: str, state: Dict[str, Any]) -> None:
        async with self._lock:
            self._states[entry_id] = dict(state)

    async def delete(self, entry_id: str) -> None:
        async with self._lock:
            self._states.pop(entry_id, None)


class FileEntryStateStore(EntryStateStore):
    """One JSON file per entry_id under a base directory.

    Filename: ``<entry_id>.json``. Atomic writes via tmp+rename.
    Asyncio.Lock-serialized to prevent torn writes from concurrent
    set() calls on the same entry_id.
    """

    def __init__(self, base_dir: str) -> None:
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    def _path_for(self, entry_id: str) -> Path:
        # Sanitize: forbid path separators and dot-segments that would
        # let a hostile entry_id escape the base_dir.
        if not isinstance(entry_id, str) or not entry_id:
            raise ComponentConfigurationError(
                f"FAIL-FAST: FileEntryStateStore entry_id must be a "
                f"non-empty string; got {entry_id!r}"
            )
        if "/" in entry_id or "\\" in entry_id or entry_id in (".", ".."):
            raise ComponentConfigurationError(
                f"FAIL-FAST: FileEntryStateStore entry_id contains "
                f"path separators or traversal sequences: {entry_id!r}"
            )
        return self._base_dir / f"{entry_id}.json"

    async def get(self, entry_id: str) -> Optional[Dict[str, Any]]:
        async with self._lock:
            p = self._path_for(entry_id)
            if not p.is_file():
                return None
            return json.loads(p.read_text(encoding="utf-8"))

    async def set(self, entry_id: str, state: Dict[str, Any]) -> None:
        async with self._lock:
            p = self._path_for(entry_id)
            tmp = p.with_suffix(p.suffix + ".tmp")
            tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
            tmp.replace(p)

    async def delete(self, entry_id: str) -> None:
        async with self._lock:
            p = self._path_for(entry_id)
            if p.is_file():
                p.unlink()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class WorkflowEntryTriggerConfig(ConfigBase):
    """Configuration for ``WorkflowEntryTrigger``.

    The wrapped (inner) trigger is provided programmatically via the
    ``inner_trigger`` kwarg to ``from_config`` — it is NOT loaded from
    this config. This avoids cross-cutting trigger-resolution complexity
    in v1 and matches how WorkflowRunner is also passed by reference.

    ``target_workflow`` and ``payload_factory`` are dotted-path strings
    that production callers can use to locate the workflow callable and
    the payload-building callable. v1 supports ``payload_factory``
    resolution; ``target_workflow`` is reserved-for-future and the
    workflow callable must be passed via kwargs to ``from_config``.
    """

    name: str
    target_workflow: Optional[str] = Field(
        default=None,
        description="Dotted-path string resolving to one of: "
                    "(a) an async callable f(payload) -> Any (preferred), "
                    "(b) a Workflow instance with a .run method (we'll "
                    "    bind .run automatically), "
                    "(c) a class — NOT supported in v2; instantiate via "
                    "    from_config first and pass the resulting "
                    "    instance's .run method via kwarg or via the "
                    "    instance attribute path. "
                    "When set, framework auto-resolves and uses the result "
                    "as the workflow callable. Explicit workflow_callable "
                    "kwarg overrides this (programmatic > YAML)."
    )
    payload_factory: Optional[str] = Field(
        default=None,
        description="Dotted-path to a callable f(event_body) -> dict that "
                    "builds the workflow payload from the inner-trigger "
                    "event body. When unset, the event body is passed "
                    "through as the payload (or {} if it's not a dict)."
    )
    autonomy_level: Literal[
        "strict_hitl", "opt_in_hitl", "pure_autonomous"
    ] = "strict_hitl"
    cost_envelope_template: Optional[str] = None
    task_id_prefix: Optional[str] = Field(
        default=None,
        description="Prefix for auto-generated task IDs. Defaults to the "
                    "trigger's name."
    )

    # G22 Step 3 — missed-schedule policy. Declared on the wrapper so a
    # YAML author can express the policy at the wiring layer rather
    # than reaching into the inner trigger's config (which is also a
    # legal place — both are honored, with the wrapper-level setting
    # winning when both are set, mirroring the framework's
    # explicit > implicit principle for the wrapping layer).
    #
    # Effective only when the inner trigger is cadenced (TimerTrigger).
    # On non-cadenced inners the field is recorded but unused.
    on_missed: Literal["skip", "catch_up", "merge"] = Field(
        default="skip",
        description="G22 Step 3 missed-schedule policy. When set on a "
                    "WorkflowEntryTrigger wrapping a TimerTrigger, this "
                    "policy is applied at restart-recovery time via the "
                    "inner trigger's replay_missed_fires hook. "
                    "Non-cadenced inner triggers ignore the field."
    )

    source_path: Optional[str] = Field(default=None, exclude=True)
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Helper: resolve a dotted-path string to a callable
# ---------------------------------------------------------------------------

def _resolve_dotted_callable(spec: str) -> Callable[[Any], Any]:
    """Resolve a dotted-path spec ('pkg.mod.func') to a callable.

    Raises ``ComponentConfigurationError`` if resolution fails or the
    result is not callable. Mirrors ``parse_transform_from_config``
    in ``nanobrain.core.link``.
    """
    if not isinstance(spec, str) or "." not in spec:
        raise ComponentConfigurationError(
            f"FAIL-FAST: payload_factory must be a dotted-path string "
            f"like 'pkg.mod.func'; got {spec!r}"
        )
    module_path, _, attr_path = spec.partition(":")
    if not attr_path:
        # No colon — split last dot.
        module_path, _, attr_path = spec.rpartition(".")
    try:
        mod = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        raise ComponentConfigurationError(
            f"FAIL-FAST: payload_factory module {module_path!r} not "
            f"importable: {exc}"
        ) from exc
    obj: Any = mod
    for part in attr_path.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError as exc:
            raise ComponentConfigurationError(
                f"FAIL-FAST: payload_factory attribute {attr_path!r} not "
                f"found on module {module_path!r}: {exc}"
            ) from exc
    if not callable(obj):
        raise ComponentConfigurationError(
            f"FAIL-FAST: payload_factory {spec!r} resolved to non-callable "
            f"{type(obj).__name__}"
        )
    return obj


def _resolve_workflow_target(spec: str) -> Callable[..., Awaitable[Any]]:
    """G22 Step 2 — resolve a ``target_workflow`` dotted-path spec to a
    callable suitable for ``WorkflowRunner.run_detached``.

    The spec resolves to one of two acceptable shapes:

    1. **A callable** (async function, coroutine, bound method, etc.) →
       returned directly.
    2. **An instance with a callable ``.run`` attribute** (e.g., a
       ``Workflow`` instance) → ``.run`` is bound and returned.

    Returning a *class* is deliberately rejected: the framework's
    ``from_config`` discipline forbids ad-hoc construction of Workflow
    classes here. If the user wants a class, they must instantiate it
    (via ``from_config``) and pass the instance — either by referring
    to a module-level instance variable in the dotted path, or by
    passing ``workflow_callable=<wf>.run`` programmatically.

    Raises ``ComponentConfigurationError`` (FAIL-FAST) on resolution
    failure or shape mismatch.
    """
    if not isinstance(spec, str) or "." not in spec:
        raise ComponentConfigurationError(
            f"FAIL-FAST: target_workflow must be a dotted-path string "
            f"like 'pkg.mod.func' or 'pkg.mod.workflow_instance'; got "
            f"{spec!r}"
        )
    module_path, _, attr_path = spec.partition(":")
    if not attr_path:
        module_path, _, attr_path = spec.rpartition(".")
    try:
        mod = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        raise ComponentConfigurationError(
            f"FAIL-FAST: target_workflow module {module_path!r} not "
            f"importable: {exc}"
        ) from exc
    obj: Any = mod
    for part in attr_path.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError as exc:
            raise ComponentConfigurationError(
                f"FAIL-FAST: target_workflow attribute {attr_path!r} not "
                f"found on module {module_path!r}: {exc}"
            ) from exc

    # Acceptable shape 1: a callable directly (function, bound method,
    # coroutine function).
    if callable(obj) and not isinstance(obj, type):
        return obj

    # Acceptable shape 2: an instance with a callable .run attribute.
    # Workflow instances are the canonical case; we duck-type so any
    # object that quacks like a workflow (.run(payload)) works.
    run_attr = getattr(obj, "run", None)
    if callable(run_attr) and not isinstance(obj, type):
        return run_attr

    # Reject classes deliberately — the from_config discipline says
    # don't ad-hoc-construct here.
    if isinstance(obj, type):
        raise ComponentConfigurationError(
            f"FAIL-FAST: target_workflow {spec!r} resolved to a class "
            f"({obj.__name__}); the framework's from_config discipline "
            f"forbids ad-hoc class instantiation here. Instantiate via "
            f"{obj.__name__}.from_config(<yaml>) and reference the "
            f"resulting instance (e.g., 'mymodule.my_workflow_instance') "
            f"OR pass workflow_callable=<wf>.run programmatically."
        )

    raise ComponentConfigurationError(
        f"FAIL-FAST: target_workflow {spec!r} resolved to "
        f"{type(obj).__name__}, which is neither callable nor an "
        f"instance with a callable .run method"
    )


# ---------------------------------------------------------------------------
# WorkflowEntryTrigger
# ---------------------------------------------------------------------------

class WorkflowEntryTrigger(FromConfigBase):
    """G22 — wraps an inner TriggerBase to launch a detached workflow run.

    This is NOT a TriggerBase subclass: it does not itself fire callbacks,
    it adapts an inner trigger's callback API to ``WorkflowRunner.run_detached``.
    Production callers compose:

        cron TimerTrigger -> WorkflowEntryTrigger -> WorkflowRunner.run_detached

    or

        EventTrigger      -> WorkflowEntryTrigger -> WorkflowRunner.run_detached

    Construct via ``from_config`` with the runner, inner trigger, and
    workflow callable passed as kwargs (the YAML carries the policy
    fields — payload factory path, autonomy, cost envelope name).
    """

    COMPONENT_TYPE = "workflow_entry_trigger"
    REQUIRED_CONFIG_FIELDS = ["name"]

    @classmethod
    def _get_config_class(cls):
        return WorkflowEntryTriggerConfig

    def _init_from_config(
        self,
        config: WorkflowEntryTriggerConfig,
        component_config: Dict[str, Any],
        dependencies: Dict[str, Any],
    ) -> None:
        self.name = config.name
        self._autonomy_level = config.autonomy_level
        self._cost_envelope_template = config.cost_envelope_template
        self._task_id_prefix = config.task_id_prefix or config.name
        self._on_missed = config.on_missed

        # G22 Step 4 — durable state. Set lazily via bind_durable_state();
        # absent by default so existing callers see no behavior change.
        self._state_store: Optional[EntryStateStore] = None
        self._entry_id: Optional[str] = None

        runner = dependencies.get("runner")
        if not isinstance(runner, WorkflowRunner):
            raise ComponentConfigurationError(
                f"FAIL-FAST: WorkflowEntryTrigger {config.name!r} requires "
                f"a WorkflowRunner instance via the 'runner' kwarg to "
                f"from_config; got {type(runner).__name__}"
            )
        self._runner: WorkflowRunner = runner

        inner_trigger = dependencies.get("inner_trigger")
        if not isinstance(inner_trigger, TriggerBase):
            raise ComponentConfigurationError(
                f"FAIL-FAST: WorkflowEntryTrigger {config.name!r} requires "
                f"a TriggerBase instance via the 'inner_trigger' kwarg to "
                f"from_config; got {type(inner_trigger).__name__}"
            )
        self._inner: TriggerBase = inner_trigger

        # G22 Step 2 — resolution precedence:
        #   1. workflow_callable kwarg wins (programmatic > YAML).
        #   2. Else, if target_workflow YAML field is set, resolve it
        #      via the dotted-path resolver. The resolved object can be:
        #        - a callable directly → use it
        #        - a Workflow instance (anything with a callable .run
        #          attribute) → bind .run as the callable
        #   3. Else, FAIL-FAST.
        workflow_callable = dependencies.get("workflow_callable")
        if workflow_callable is None and config.target_workflow:
            workflow_callable = _resolve_workflow_target(config.target_workflow)
        if not callable(workflow_callable):
            raise ComponentConfigurationError(
                f"FAIL-FAST: WorkflowEntryTrigger {config.name!r} requires "
                f"a callable workflow target. Provide one via the "
                f"'workflow_callable' kwarg to from_config OR set "
                f"'target_workflow' in YAML to a dotted-path resolving to "
                f"a callable or a Workflow instance with a .run method. "
                f"Got: workflow_callable={type(workflow_callable).__name__}, "
                f"target_workflow={config.target_workflow!r}"
            )
        self._workflow_callable: Callable[..., Awaitable[Any]] = workflow_callable
        self._target_workflow_spec = config.target_workflow

        # Optional callback invoked AFTER each successful run_detached call,
        # so the caller can record the task_id without polling list_active.
        self._on_launch: Optional[Callable[[DetachedTaskHandle], Awaitable[None]]] = (
            dependencies.get("on_launch")
        )

        if config.payload_factory:
            self._payload_factory = _resolve_dotted_callable(config.payload_factory)
        else:
            self._payload_factory = None

        self._is_started = False

    @classmethod
    def resolve_dependencies(cls, component_config: Dict[str, Any], **kwargs):
        # Pass through the runtime kwargs — the framework's standard
        # resolve_dependencies does not know about runner/inner_trigger.
        return {
            "runner": kwargs.get("runner"),
            "inner_trigger": kwargs.get("inner_trigger"),
            "workflow_callable": kwargs.get("workflow_callable"),
            "on_launch": kwargs.get("on_launch"),
        }

    async def start(self) -> None:
        """Bind to the inner trigger's callback list and start it."""
        if self._is_started:
            return
        await self._inner.add_callback(self._on_inner_fire)
        await self._inner.start_monitoring()
        self._is_started = True

    async def stop(self) -> None:
        """Stop the inner trigger and unbind."""
        if not self._is_started:
            return
        await self._inner.stop_monitoring()
        # TriggerBase has no remove_callback API today; the inner trigger
        # owns its own lifecycle. Stop suffices to halt fire delivery.
        self._is_started = False

    async def replay_missed_fires(
        self,
        last_known_fire_epoch_seconds: float,
        now_epoch_seconds: Optional[float] = None,
    ) -> int:
        """G22 Step 3 — apply ``on_missed`` policy after a process restart.

        Delegates to the inner trigger's ``replay_missed_fires`` if the
        inner is cadenced (e.g., TimerTrigger). The inner trigger's
        own ``on_missed`` is overridden by the wrapper's policy for the
        duration of the replay — the wrapper-level setting wins so a
        deployment author can configure restart behavior at the wiring
        layer without editing the inner trigger's YAML.

        For non-cadenced inner triggers (EventTrigger, ManualTrigger,
        DataUnitChangeTrigger), there is no schedule to miss; this
        method is a no-op and returns 0. Callers should not invoke
        replay on event-driven workflows; if they do, the framework
        does not silently fire phantom events.

        Caller responsibility: the deployment must have persisted
        ``last_known_fire_epoch_seconds`` from the prior run (Step 4
        scope). Step 3 ships only the policy-application primitive.

        Returns the count of fires emitted.
        """
        replay = getattr(self._inner, "replay_missed_fires", None)
        if replay is None:
            return 0
        original_policy = getattr(self._inner, "on_missed", None)
        try:
            # Wrapper override: stamp the wrapper's policy on the inner
            # for the duration of the replay window.
            self._inner.on_missed = self._on_missed
            return await replay(
                last_known_fire_epoch_seconds=last_known_fire_epoch_seconds,
                now_epoch_seconds=now_epoch_seconds,
            )
        finally:
            if original_policy is not None:
                self._inner.on_missed = original_policy

    async def _on_inner_fire(self, event_body: Any) -> None:
        """Callback registered with the inner trigger. Builds the payload
        and launches a detached workflow run."""
        if self._payload_factory is not None:
            payload = self._payload_factory(event_body)
        elif isinstance(event_body, dict):
            payload = dict(event_body)
        else:
            payload = {"event_body": event_body}

        if not isinstance(payload, dict):
            raise ComponentConfigurationError(
                f"FAIL-FAST: WorkflowEntryTrigger {self.name!r} payload_factory "
                f"returned {type(payload).__name__}; expected dict"
            )

        # Forward-compat metadata that downstream consumers may inspect.
        payload.setdefault("__autonomy_level__", self._autonomy_level)
        if self._cost_envelope_template:
            payload.setdefault(
                "__cost_envelope_template__", self._cost_envelope_template
            )

        task_id = f"{self._task_id_prefix}-{uuid.uuid4().hex[:12]}"
        handle = await self._runner.run_detached(
            self._workflow_callable, task_id, payload
        )

        # G22 Step 4 — persist last-fire bookkeeping. Done AFTER
        # successful run_detached scheduling, BEFORE invoking the
        # caller's on_launch callback. If the durable store is down,
        # the fire still completed (we never roll the workflow back),
        # but the next restart's replay_missed_fires will see the
        # last persisted timestamp instead of the one that just fired.
        # This is the safer failure shape: lost-bookkeeping > lost-work.
        if self._state_store is not None and self._entry_id is not None:
            import time as _time
            try:
                await self._state_store.set(self._entry_id, {
                    "last_fire_epoch_seconds": _time.time(),
                    "last_task_id": task_id,
                })
            except Exception:  # noqa: BLE001
                # Durable-store write failure must NOT crash the fire
                # cascade. Log via the framework logger if available;
                # otherwise swallow silently. The next successful fire
                # will refresh the timestamp.
                pass

        if self._on_launch:
            await self._on_launch(handle)

    # ---- G22 Step 4 — durable binding API ------------------------------

    def bind_durable_state(
        self, store: EntryStateStore, entry_id: Optional[str] = None,
    ) -> None:
        """G22 Step 4 — attach a durable state store to this wrapper.

        After binding, every successful inner-fire persists
        ``{last_fire_epoch_seconds, last_task_id}`` under ``entry_id``
        (defaults to ``self.name``). Deployments call this once at
        startup BEFORE start(), then call recover_from_durable_state()
        to consume any persisted timestamp via replay_missed_fires.

        Idempotent: calling twice is safe; the second call replaces
        the binding. Pass ``store=None`` to detach.
        """
        if store is not None and not isinstance(store, EntryStateStore):
            raise ComponentConfigurationError(
                f"FAIL-FAST: WorkflowEntryTrigger {self.name!r} "
                f"bind_durable_state: store must be an EntryStateStore "
                f"instance, got {type(store).__name__}"
            )
        self._state_store = store
        self._entry_id = entry_id or self.name

    async def recover_from_durable_state(
        self, now_epoch_seconds: Optional[float] = None,
    ) -> int:
        """G22 Step 4 — read the persisted last-fire timestamp and
        delegate to ``replay_missed_fires`` to apply ``on_missed``
        policy.

        Call this at deployment startup AFTER bind_durable_state and
        BEFORE start(). Returns the count of replayed fires (0 when
        no prior state exists OR when the inner trigger is non-cadenced).
        """
        if self._state_store is None or self._entry_id is None:
            return 0
        state = await self._state_store.get(self._entry_id)
        if not state:
            return 0
        last_fire = state.get("last_fire_epoch_seconds")
        if last_fire is None:
            return 0
        return await self.replay_missed_fires(
            last_known_fire_epoch_seconds=float(last_fire),
            now_epoch_seconds=now_epoch_seconds,
        )
