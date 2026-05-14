"""GlobusComputeExecutor (G22 / G24) — dispatch a step to a Globus Compute endpoint.

A nanobrain ``ExecutorBase`` that runs a step's ``process()`` on a remote
Globus Compute endpoint (e.g. a managed endpoint on ALCF Aurora) instead of
locally. It mirrors ``ParslExecutor``'s structure and — crucially — its
*dispatch contract*.

Dispatch design — approach (B), reconstruct-from-config
-------------------------------------------------------
``BaseStep._execute_on_trigger`` (``core/step.py:1544``) hands the executor
ONLY a closure (``execute_wrapper``) that captures ``self`` (the step) and
``input_data``. It does NOT pass the step's class or config path as kwargs.

``ParslExecutor._execute_parsl`` (``core/executor.py:1501-1523``) handles
this by *introspecting the closure* — it walks ``task.__closure__`` cells,
finds the cell holding an object with both ``.config`` and ``.process``
(the step) and the cell holding a ``dict`` (the input data), reads
``step._config_path`` + the step's fully-qualified class name, and ships
``(step_config_path, step_class_name, input_data)`` to the worker. The
worker does ``importlib`` -> ``step_class.from_config(path)`` ->
``step.process(input_data)``.

This executor replicates that exact pattern. We chose approach (B) over
approach (A) "serialize the live closure" because:

  * It is what the framework's own ParslExecutor does — staying consistent
    with the framework's dispatch contract is the load-bearing requirement.
  * It is robust cross-machine: only a config-file path + a class name + a
    plain ``input_data`` dict cross the wire. No live nanobrain step object
    (with its data units, triggers, loggers, asyncio state) is pickled.
  * It requires the remote endpoint to have the SAME nanobrain code and the
    SAME step config file resolvable on its filesystem — that is the
    documented requirement, identical to ParslExecutor's
    ``Step instance missing _config_path`` contract. A managed endpoint on
    a shared HPC filesystem (Aurora's ``/lus/flare``) satisfies this.

The module-level :func:`_run_step_on_endpoint` is what actually executes on
the endpoint worker. It MUST be module-level (a Globus Compute function is
serialized by reference + source, like a Parsl ``@python_app``); capturing
a closure variable would break serialization.

FAIL-LOUD discipline (workspace CLAUDE.md)
------------------------------------------
  * ``globus_compute_sdk`` missing -> ``ComponentConfigurationError``.
  * ``endpoint_id`` missing -> ``ComponentConfigurationError``.
  * auth failure (delegated to the G23 helper) -> ``ComponentConfigurationError``.
  * step instance not extractable from the closure / missing
    ``_config_path`` -> ``RuntimeError`` (same shape as ParslExecutor).
  * a remote exception on the endpoint -> re-raised locally as
    ``RuntimeError`` carrying the endpoint's traceback. It is NEVER
    swallowed: a "successful" dispatch that produced an endpoint-side
    error must surface.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from nanobrain.core.component_base import ComponentConfigurationError
from nanobrain.core.distributed.globus_auth import build_globus_app
from nanobrain.core.executor import ExecutorBase, ExecutorConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level worker function — runs ON the Globus Compute endpoint.
# ---------------------------------------------------------------------------
def _run_step_on_endpoint(
    step_config_path: str,
    step_class_name: str,
    input_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Reconstruct a nanobrain step from config on the endpoint and run it.

    This function is shipped to and executed on the remote Globus Compute
    endpoint worker. It only takes serializable primitives — a config file
    path, a fully-qualified class name, and a plain dict — so nothing
    nanobrain-live is pickled.

    The endpoint worker MUST have the nanobrain package importable and the
    ``step_config_path`` resolvable on its filesystem. This is the same
    requirement ParslExecutor places on its workers.

    Returns a result envelope dict. On the endpoint side we catch every
    exception and return it in the envelope (with the traceback) rather
    than letting it propagate as an opaque Globus Compute task failure —
    the calling side re-raises it FAIL-LOUD with full context.
    """
    import asyncio as _asyncio
    import importlib as _importlib
    import os as _os
    import traceback as _traceback

    try:
        module_name, class_name = step_class_name.rsplit(".", 1)
        module = _importlib.import_module(module_name)
        step_class = getattr(module, class_name)

        step = step_class.from_config(step_config_path)

        if _asyncio.iscoroutinefunction(step.process):
            loop = _asyncio.new_event_loop()
            _asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(step.process(input_data))
            finally:
                loop.close()
        else:
            result = step.process(input_data)

        return {
            "status": "success",
            "result": result,
            "worker_node": _os.uname().nodename,
            "worker_pid": _os.getpid(),
        }
    except Exception as exc:  # noqa: BLE001 - intentionally broad: report everything
        return {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": _traceback.format_exc(),
            "worker_node": _os.uname().nodename,
            "worker_pid": _os.getpid(),
        }


# ---------------------------------------------------------------------------
# Config model for the `globus_compute` block.
# ---------------------------------------------------------------------------
class GlobusComputeConfig(BaseModel):
    """Validates the ``globus_compute`` dict on an ``ExecutorConfig``.

    ``extra='forbid'`` (workspace rule): a YAML typo raises here at
    executor-build time rather than silently using a default.
    """

    model_config = ConfigDict(extra="forbid")

    endpoint_id: str = Field(
        ...,
        description="The Globus Compute endpoint UUID to dispatch steps to.",
    )
    auth_mode: Literal["client_credentials", "native"] = Field(
        default="client_credentials",
        description=(
            "'client_credentials' (default) uses a confidential client; "
            "'native' uses an interactive browser login."
        ),
    )
    client_id: Optional[str] = Field(
        default=None,
        description=(
            "Confidential-client id. Falls back to "
            "$GLOBUS_COMPUTE_CLIENT_ID when omitted."
        ),
    )
    client_secret: Optional[str] = Field(
        default=None,
        description=(
            "Confidential-client secret. Falls back to "
            "$GLOBUS_COMPUTE_CLIENT_SECRET when omitted."
        ),
    )
    resource_specification: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Passed verbatim to globus_compute_sdk.Executor as the default "
            "resource_specification for submitted tasks."
        ),
    )
    user_endpoint_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Passed verbatim to globus_compute_sdk.Executor; per-user "
            "endpoint configuration for multi-user endpoints."
        ),
    )
    task_timeout_seconds: float = Field(
        default=3600.0,
        gt=0.0,
        description=(
            "Maximum wall time to wait for a remote task result before "
            "raising. HPC endpoints can be slow to schedule; 1h default."
        ),
    )

    @classmethod
    def from_executor_config(cls, executor_config: ExecutorConfig) -> "GlobusComputeConfig":
        """Build + validate from the ``globus_compute`` dict on an ExecutorConfig.

        FAIL-LOUD when the block is absent — a GlobusComputeExecutor with no
        ``globus_compute`` config has nothing to dispatch to.
        """
        raw = getattr(executor_config, "globus_compute", None)
        if not raw:
            raise ComponentConfigurationError(
                "FAIL-FAST: GlobusComputeExecutor requires a 'globus_compute' "
                "configuration block on the ExecutorConfig (with at least "
                "'endpoint_id'). None was provided."
            )
        try:
            return cls(**raw)
        except Exception as exc:
            raise ComponentConfigurationError(
                "FAIL-FAST: GlobusComputeExecutor 'globus_compute' config is "
                f"invalid: {exc}"
            ) from exc


# ---------------------------------------------------------------------------
# The executor.
# ---------------------------------------------------------------------------
class GlobusComputeExecutor(ExecutorBase):
    """Execute nanobrain steps on a remote Globus Compute endpoint.

    Created via ``from_config`` only. The relevant config is the
    ``globus_compute`` dict on the ``ExecutorConfig`` (see
    :class:`GlobusComputeConfig`).
    """

    @classmethod
    def from_config(
        cls, config: Union[str, ExecutorConfig], **kwargs
    ) -> "GlobusComputeExecutor":
        """Create a GlobusComputeExecutor from a config path or ExecutorConfig.

        Mirrors ``ParslExecutor.from_config`` / ``LocalExecutor.from_config``:
        accepts a YAML path or an ``ExecutorConfig`` object for framework
        consistency.
        """
        if isinstance(config, str):
            executor_config = ExecutorConfig.from_config(config, **kwargs)
        elif isinstance(config, ExecutorConfig):
            executor_config = config
        else:
            raise ValueError(
                f"Invalid config type: {type(config)}. "
                "Expected str or ExecutorConfig"
            )

        cls.validate_config_schema(executor_config)
        component_config = cls.extract_component_config(executor_config)
        dependencies = cls.resolve_dependencies(component_config, **kwargs)
        instance = cls.create_instance(executor_config, component_config, dependencies)
        instance._post_config_initialization()
        logger.info("Successfully created GlobusComputeExecutor")
        return instance

    def _init_from_config(
        self,
        config: ExecutorConfig,
        component_config: Dict[str, Any],
        dependencies: Dict[str, Any],
    ) -> None:
        """Validate the globus_compute block; defer all network/auth to initialize()."""
        super()._init_from_config(config, component_config, dependencies)
        # Validate the globus_compute block now (FAIL-LOUD on a typo at
        # build time), but do NOT touch the network or Globus Auth here —
        # that is initialize()'s job, so a malformed config fails fast
        # while reachability is a runtime concern.
        self._gc_config: GlobusComputeConfig = GlobusComputeConfig.from_executor_config(
            config
        )
        self._gc_app: Optional[Any] = None
        self._gc_client: Optional[Any] = None
        self._gc_executor: Optional[Any] = None

    @property
    def globus_compute_config(self) -> GlobusComputeConfig:
        """The validated GlobusComputeConfig this executor dispatches with."""
        return self._gc_config

    async def initialize(self) -> None:
        """Lazy-import globus_compute_sdk; build the GlobusApp, Client, Executor.

        FAIL-LOUD on: missing ``globus_compute_sdk``, auth failure, or any
        endpoint/client construction error — each with a distinct,
        actionable message.
        """
        if self._is_initialized:
            return

        try:
            import globus_compute_sdk  # noqa: PLC0415 - intentional lazy import
        except ImportError as exc:
            raise ComponentConfigurationError(
                "FAIL-FAST: GlobusComputeExecutor requires the "
                "'globus_compute_sdk' package, which is not installed. "
                "Install it with `pip install globus-compute-sdk`. This "
                "dependency is only needed when a workflow actually selects "
                f"executor_type: globus_compute. Underlying error: {exc}"
            ) from exc

        cfg = self._gc_config

        # The canonical Globus Compute scope. globus_compute_sdk exposes it
        # as Client.FUNCX_SCOPE (env-overridable via $GLOBUS_COMPUTE_SCOPE)
        # — verified against globus_compute_sdk 4.11.0. We do NOT hardcode
        # a guessed scope string.
        compute_scope = globus_compute_sdk.Client.FUNCX_SCOPE

        # Build the GlobusApp via the shared G23 helper. We pre-declare the
        # Compute scope; globus_compute_sdk.Client(app=...) would also
        # auto-register it, but declaring it up front means the first token
        # acquisition already covers Compute.
        try:
            self._gc_app = build_globus_app(
                auth_mode=cfg.auth_mode,
                scopes=[compute_scope],
                client_id=cfg.client_id,
                client_secret=cfg.client_secret,
                app_name="nanobrain-globus-compute-executor",
            )
        except ComponentConfigurationError:
            raise
        except Exception as exc:
            raise ComponentConfigurationError(
                "FAIL-FAST: GlobusComputeExecutor failed to build the Globus "
                f"auth app (auth_mode={cfg.auth_mode!r}): {exc}"
            ) from exc

        try:
            self._gc_client = globus_compute_sdk.Client(app=self._gc_app)
        except Exception as exc:
            raise ComponentConfigurationError(
                "FAIL-FAST: GlobusComputeExecutor failed to construct the "
                f"globus_compute_sdk.Client (auth/connectivity issue): {exc}"
            ) from exc

        try:
            self._gc_executor = globus_compute_sdk.Executor(
                endpoint_id=cfg.endpoint_id,
                client=self._gc_client,
                resource_specification=cfg.resource_specification,
                user_endpoint_config=cfg.user_endpoint_config,
            )
        except Exception as exc:
            raise ComponentConfigurationError(
                "FAIL-FAST: GlobusComputeExecutor failed to construct the "
                f"globus_compute_sdk.Executor for endpoint_id="
                f"{cfg.endpoint_id!r}: {exc}"
            ) from exc

        self._is_initialized = True
        logger.info(
            "GlobusComputeExecutor initialized (endpoint_id=%s, auth_mode=%s)",
            cfg.endpoint_id,
            cfg.auth_mode,
        )

    @staticmethod
    def _extract_step_and_input(task: Any) -> tuple:
        """Pull the step instance + input_data dict out of the step closure.

        ``BaseStep._execute_on_trigger`` hands the executor an
        ``execute_wrapper`` closure that captures ``self`` (the step) and
        ``input_data``. This walks ``task.__closure__`` the exact same way
        ``ParslExecutor._execute_parsl`` does (``core/executor.py:1507``):
        the step is the cell whose contents have both ``.config`` and
        ``.process``; ``input_data`` is the cell that is a plain dict.
        """
        step_instance = None
        input_data: Optional[Dict[str, Any]] = None
        closure = getattr(task, "__closure__", None)
        if closure:
            for cell in closure:
                contents = cell.cell_contents
                if hasattr(contents, "config") and hasattr(contents, "process"):
                    step_instance = contents
                elif isinstance(contents, dict):
                    input_data = contents
        return step_instance, input_data

    async def execute(
        self,
        task: Any,
        resource_specification: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Any:
        """Dispatch a step's ``process()`` to the Globus Compute endpoint.

        ``task`` is the ``execute_wrapper`` closure handed in by
        ``BaseStep._execute_on_trigger``. We introspect it for the step +
        input data (approach B), ship ``(config_path, class_name,
        input_data)`` to the endpoint, bridge the returned
        ``concurrent.futures.Future`` to asyncio, and FAIL-LOUD on any
        remote error.

        ``resource_specification`` is accepted for ``ExecutorBase``
        contract compatibility; the endpoint-level resource spec is set on
        the ``globus_compute_sdk.Executor`` at construction time.
        """
        if not self._is_initialized:
            await self.initialize()

        step_instance, input_data = self._extract_step_and_input(task)

        if step_instance is None:
            raise RuntimeError(
                "GlobusComputeExecutor: cannot extract a step instance from "
                "the task closure. This executor dispatches nanobrain steps "
                "(it introspects the step's execute_wrapper closure the same "
                "way ParslExecutor does); it cannot run an arbitrary "
                "callable."
            )

        # The remote endpoint reconstructs the step via
        # step_class.from_config(<path>), so we need the step's config-file
        # path. ParslExecutor reads `step._config_path` (set by the
        # workflow loader when steps come from class:+config: references).
        # We also accept `step.config.source_path` (the ConfigBase tracking
        # attribute set on ANY from_config(path) load) so a step loaded
        # directly — not only through a workflow — is still dispatchable.
        step_config_path = getattr(step_instance, "_config_path", None)
        if step_config_path is None:
            step_config_obj = getattr(step_instance, "config", None)
            step_config_path = getattr(step_config_obj, "source_path", None)
        if step_config_path is None:
            raise RuntimeError(
                "GlobusComputeExecutor: step instance "
                f"{step_instance.__class__.__name__!r} has neither "
                "'_config_path' nor 'config.source_path'. The remote "
                "endpoint reconstructs the step via "
                "step_class.from_config(<path>), so the step MUST be loaded "
                "from a YAML config file, not an inline dict. (Same "
                "requirement as ParslExecutor.)"
            )

        step_class_name = (
            f"{step_instance.__class__.__module__}."
            f"{step_instance.__class__.__name__}"
        )

        logger.info(
            "GlobusComputeExecutor: dispatching %s (config=%s) to endpoint %s",
            step_class_name,
            step_config_path,
            self._gc_config.endpoint_id,
        )

        # Submit to the endpoint. globus_compute_sdk.Executor.submit returns
        # a concurrent.futures.Future. Bridge it to asyncio without blocking
        # the event loop, and honour task_timeout_seconds.
        try:
            cf_future = self._gc_executor.submit(
                _run_step_on_endpoint,
                str(step_config_path),
                step_class_name,
                input_data or {},
            )
        except Exception as exc:
            raise RuntimeError(
                "GlobusComputeExecutor: failed to submit task to endpoint "
                f"{self._gc_config.endpoint_id!r}: {type(exc).__name__}: {exc}"
            ) from exc

        aio_future = asyncio.wrap_future(cf_future)
        try:
            envelope = await asyncio.wait_for(
                aio_future, timeout=self._gc_config.task_timeout_seconds
            )
        except asyncio.TimeoutError as exc:
            cf_future.cancel()
            raise RuntimeError(
                "GlobusComputeExecutor: remote task on endpoint "
                f"{self._gc_config.endpoint_id!r} exceeded "
                f"task_timeout_seconds={self._gc_config.task_timeout_seconds}. "
                "Increase task_timeout_seconds, or check the endpoint is "
                "online and scheduling tasks."
            ) from exc
        except Exception as exc:
            # Globus Compute itself raised (serialization error, endpoint
            # offline, auth expired, etc). Surface it FAIL-LOUD.
            raise RuntimeError(
                "GlobusComputeExecutor: remote execution on endpoint "
                f"{self._gc_config.endpoint_id!r} failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        # The endpoint-side worker catches its own exceptions and returns
        # them in the envelope. A status=='error' envelope is a remote
        # failure — re-raise it locally with the endpoint's traceback. It
        # is NEVER swallowed.
        if not isinstance(envelope, dict) or "status" not in envelope:
            raise RuntimeError(
                "GlobusComputeExecutor: remote worker returned an "
                f"unrecognized result envelope: {envelope!r}. Expected a "
                "dict with a 'status' key."
            )

        if envelope["status"] == "error":
            raise RuntimeError(
                "GlobusComputeExecutor: step execution FAILED on endpoint "
                f"{self._gc_config.endpoint_id!r} "
                f"(worker {envelope.get('worker_node')}, "
                f"pid {envelope.get('worker_pid')}).\n"
                f"Remote error: {envelope.get('error')}\n"
                f"Remote traceback:\n{envelope.get('traceback')}"
            )

        logger.info(
            "GlobusComputeExecutor: step completed on endpoint %s (worker %s)",
            self._gc_config.endpoint_id,
            envelope.get("worker_node"),
        )
        return envelope["result"]

    async def shutdown(self) -> None:
        """Shut down the underlying globus_compute_sdk.Executor."""
        if self._gc_executor is not None:
            try:
                self._gc_executor.shutdown()
            except Exception as exc:  # noqa: BLE001 - shutdown best-effort
                logger.warning(
                    "GlobusComputeExecutor: error during executor shutdown: %s",
                    exc,
                )
            self._gc_executor = None
        self._gc_client = None
        self._gc_app = None
        self._is_initialized = False
        logger.info("GlobusComputeExecutor shutdown complete")


__all__ = [
    "GlobusComputeExecutor",
    "GlobusComputeConfig",
]
