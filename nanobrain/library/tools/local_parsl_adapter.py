"""LocalParslAdapter — G11-completion concrete backend adapter.

eval_03 Round 2: G11 was "shipped" (the abstract ``ToolBackendAdapter``
+ ``ToolBackendRegistry`` + ``ToolExecutionStep`` are real in
``library/steps/tool_execution_step.py``), but the concrete in-tree
LocalParsl adapter the gap proposal listed as a sibling shipment was
deferred. RheaAdapter shipped from the Rhea fork; Galaxy is still
deferred (no testable endpoint); LocalParsl was the missing in-tree
piece for any ``ToolExecutionStep`` whose backend is local-process
Python execution (i.e. the dominant case for development +
single-host deployments).

Post-G11-completion: ``LocalParslAdapter`` (BACKEND_NAME = "local_parsl")
materializes a Python callable from a UTD's ``provenance_pin.class_path``
(or an explicit kwarg) and dispatches it via Parsl, returning the
typed dict-shaped result.

## P0++b decision (open question from roadmap §8.7)

Three candidate Parsl executor presets:

  * ``thread``  — ``parsl.executors.ThreadPoolExecutor``
                  Lowest-overhead; fork-safe; no cluster prereqs.
                  THIS IS THE DEFAULT.
  * ``process`` — ``parsl.executors.HighThroughputExecutor`` with a
                  ``LocalProvider`` (HTEX is the only Parsl executor
                  with real-process parallelism on a single node;
                  ProcessPoolExecutor was deprecated upstream).
  * ``htex``    — alias for ``process`` (different name for clarity
                  in HPC-shaped configs).

ThreadPoolExecutor is the chosen default because:

  1. It has zero cluster prereqs (no provider, no launcher).
  2. It is fork-safe — important on macOS where the default
     start method changed away from fork in py3.14.
  3. It plays well with the tool's typical workload: I/O-bound
     domain calls (HTTP, DB, file I/O), not CPU-bound NumPy.
  4. Tests can spin up + tear down deterministically without
     leaving zombie workers.

Operators who want process-level parallelism flip
``executor_kind="process"``; HPC operators mounting their own
launchers swap in a custom Parsl ``Config`` via the
``parsl_config`` kwarg.

## Lifecycle

Parsl's ``parsl.load(config)`` is process-global — calling it twice
without a clear in between raises ``ConfigurationError``. The adapter
defends against this:

  * lazy init: first ``invoke()`` call loads Parsl
  * ``close()`` clears Parsl so a fresh process / test can re-load
  * tests use the ``adapter.scope()`` context to guarantee cleanup

## UTD contract

The adapter expects the UTD's invocation hint to indicate where to
find the Python callable:

  * ``provenance_pin.class_path`` is a dotted import path to the
    callable (e.g. ``"my_pkg.my_module.my_function"``)
  * the callable's signature drives input binding — extra inputs
    not in the signature FAIL-FAST (mirrors PythonCallableDispatcher)
  * the callable returns a dict; values for unknown UTD output keys
    are passed through verbatim

A future revision can support shell commands by routing through
``parsl.bash_app`` instead of ``parsl.python_app``; the adapter
shape is intentionally extensible.

Source: ``apecx-mcp-integration/eval_03_nanobrain_gap_inventory.md``
Round 2 G11-completion;
``apecx-mcp-integration/docs/development_roadmap.md`` 8.7 (P0++b).
"""
from __future__ import annotations

import asyncio
import importlib
import inspect
import logging
import threading
from contextlib import contextmanager
from typing import Any, Callable, ClassVar, Dict, Iterator, Literal, Optional

from nanobrain.core.component_base import ComponentConfigurationError
from nanobrain.core.unified_tool_descriptor import UnifiedToolDescriptor
from nanobrain.library.steps.tool_execution_step import (
    ToolBackendAdapter,
    ToolBackendRegistry,
)

logger = logging.getLogger(__name__)


ExecutorKind = Literal["thread", "process", "htex"]


class LocalParslAdapter(ToolBackendAdapter):
    """ToolBackendAdapter that dispatches Python callables via Parsl.

    Args:
        executor_kind: One of ``"thread"`` (default; ThreadPoolExecutor),
            ``"process"`` / ``"htex"`` (HighThroughputExecutor + LocalProvider).
        max_workers: Parallelism for the executor. Defaults to 4 — large
            enough to exercise concurrency in tests + small enough not
            to thrash a developer laptop.
        parsl_config: Optional pre-built ``parsl.config.Config`` instance
            to override the executor preset. HPC operators with custom
            launchers / providers pass their config here. When provided,
            ``executor_kind`` and ``max_workers`` are ignored.
        explicit_callable: Optional callable to use for ALL invocations.
            Bypasses ``provenance_pin.class_path`` resolution. Useful for
            tests + single-tool deployments. When provided, the adapter
            dispatches every invoke() call to this callable regardless
            of the UTD's class_path.
    """

    BACKEND_NAME: ClassVar[str] = "local_parsl"

    # Process-global "is parsl loaded" guard. parsl.load can only be
    # called once between clear() calls, so we coordinate across all
    # adapter instances in the same process.
    _parsl_loaded: ClassVar[bool] = False
    _load_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(
        self,
        *,
        executor_kind: ExecutorKind = "thread",
        max_workers: int = 4,
        parsl_config: Optional[Any] = None,
        explicit_callable: Optional[Callable[..., Any]] = None,
    ) -> None:
        if executor_kind not in ("thread", "process", "htex"):
            raise ComponentConfigurationError(
                f"FAIL-FAST: LocalParslAdapter executor_kind must be "
                f"'thread', 'process', or 'htex'; got {executor_kind!r}"
            )
        self._executor_kind: ExecutorKind = executor_kind
        self._max_workers: int = int(max_workers)
        self._parsl_config = parsl_config
        self._explicit_callable = explicit_callable
        # Cache the @python_app-wrapped callable per UTD class_path so we
        # don't pay the wrap overhead on every invocation.
        self._app_cache: Dict[str, Any] = {}
        # Resolution-cache for raw callables (un-wrapped) — used for
        # signature introspection.
        self._fn_cache: Dict[str, Callable[..., Any]] = {}
        # Track our own load state so close() is idempotent.
        self._loaded_by_self: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def invoke(
        self,
        utd: UnifiedToolDescriptor,
        inputs: Dict[str, Any],
        *,
        run_context_namespace: str = "",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Dispatch ``utd`` with ``inputs`` via Parsl.

        Returns whatever the wrapped callable returned, packaged as a
        dict (callables that return non-dict values are wrapped under
        ``{"_result": <value>}`` for shape-uniformity, mirroring
        BaseStep._update_output_data_units).

        Raises:
            ComponentConfigurationError: on UTD resolution failure
                (class_path missing or unresolvable), input binding
                mismatch, or executor-config error.
            RuntimeError: on Parsl-side failures (worker died, etc.).
                The framework's retry layer handles these.
        """
        if not isinstance(inputs, dict):
            raise ComponentConfigurationError(
                f"FAIL-FAST: LocalParslAdapter.invoke inputs must be a "
                f"dict; got {type(inputs).__name__}"
            )

        fn = self._resolve_callable(utd)
        bound_inputs = self._bind_inputs(fn, inputs)

        # Lazy-load Parsl on first invocation.
        await self._ensure_parsl_loaded()

        # Build (or fetch from cache) the python_app wrapper.
        cache_key = self._cache_key_for_utd(utd)
        app = self._app_cache.get(cache_key)
        if app is None:
            app = self._wrap_as_python_app(fn)
            self._app_cache[cache_key] = app

        # Dispatch. Parsl's app(**kwargs) returns a concurrent.futures.
        # Future. We wrap into asyncio so the calling coroutine awaits
        # naturally without blocking the event loop.
        future = app(**bound_inputs)
        result = await asyncio.wrap_future(asyncio.ensure_future(
            asyncio.to_thread(future.result)
        ))

        # Normalize result shape: dict or single-key fallback.
        if isinstance(result, dict):
            return result
        return {"_result": result}

    @contextmanager
    def scope(self) -> Iterator["LocalParslAdapter"]:
        """Test convenience: ensure parsl is cleared on exit.

        Use::

            with LocalParslAdapter().scope() as adapter:
                ...

        Guarantees ``close()`` is called even if the test body raises.
        """
        try:
            yield self
        finally:
            self.close()

    def close(self) -> None:
        """Clear Parsl + local app cache. Idempotent.

        Tests must call this in teardown OR use the ``scope()`` context
        manager — Parsl's ``parsl.load`` cannot be called a second time
        without a ``parsl.dfk().cleanup()`` in between.
        """
        self._app_cache.clear()
        self._fn_cache.clear()
        if not self._loaded_by_self:
            return
        try:
            import parsl

            try:
                # Parsl 2024+: dfk() returns the active DataFlowKernel.
                # cleanup() drains pending work and shuts down executors.
                parsl.dfk().cleanup()
            except Exception:
                pass
            try:
                parsl.clear()
            except Exception:
                pass
        finally:
            self._loaded_by_self = False
            type(self)._parsl_loaded = False

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _cache_key_for_utd(self, utd: UnifiedToolDescriptor) -> str:
        """Stable cache key per UTD. Falls back to id(utd) if class_path
        is missing — prevents two UTDs sharing a wrapper accidentally.
        """
        if self._explicit_callable is not None:
            return f"explicit:{id(self._explicit_callable)}"
        pin = getattr(utd, "provenance_pin", None)
        class_path = getattr(pin, "class_path", None) if pin else None
        if class_path:
            return f"class_path:{class_path}"
        return f"id:{id(utd)}"

    def _resolve_callable(
        self, utd: UnifiedToolDescriptor
    ) -> Callable[..., Any]:
        """Get the Python callable for this UTD.

        Resolution order:
          1. ``self._explicit_callable`` if provided to __init__
          2. ``utd.provenance_pin.class_path`` resolved via importlib

        FAIL-FASTs when neither yields a callable.
        """
        if self._explicit_callable is not None:
            return self._explicit_callable

        pin = getattr(utd, "provenance_pin", None)
        class_path = getattr(pin, "class_path", None) if pin else None
        if not class_path:
            raise ComponentConfigurationError(
                f"FAIL-FAST: LocalParslAdapter cannot resolve a callable "
                f"for UTD {getattr(utd, 'descriptor_id', '<unknown>')!r}: "
                f"provenance_pin.class_path is missing AND no "
                f"explicit_callable was passed at adapter construction. "
                f"Either set provenance_pin.class_path to a dotted "
                f"import path OR construct the adapter with "
                f"LocalParslAdapter(explicit_callable=fn)."
            )

        cached = self._fn_cache.get(class_path)
        if cached is not None:
            return cached

        try:
            module_path, _, attr_name = class_path.rpartition(".")
            if not module_path or not attr_name:
                raise ImportError(
                    f"class_path {class_path!r} is not a dotted import "
                    f"path (expected 'pkg.module.symbol')"
                )
            module = importlib.import_module(module_path)
            fn = getattr(module, attr_name)
        except (ImportError, AttributeError) as exc:
            raise ComponentConfigurationError(
                f"FAIL-FAST: LocalParslAdapter could not resolve UTD "
                f"class_path {class_path!r}: "
                f"{type(exc).__name__}: {exc}. Check the dotted import "
                f"path is correct AND the package is installed in the "
                f"runtime venv."
            ) from exc
        if not callable(fn):
            raise ComponentConfigurationError(
                f"FAIL-FAST: LocalParslAdapter resolved {class_path!r} "
                f"to {type(fn).__name__} which is not callable."
            )
        self._fn_cache[class_path] = fn
        return fn

    def _bind_inputs(
        self,
        fn: Callable[..., Any],
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Validate inputs match the callable's signature. Mirrors
        PythonCallableDispatcher._bind_payload — same FAIL-FAST surface
        so adapters stay consistent."""
        try:
            sig = inspect.signature(fn)
        except (ValueError, TypeError):
            return dict(inputs)  # builtins / C-extensions

        params = sig.parameters
        accepts_var_kw = any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
        )

        if not accepts_var_kw:
            extras = [k for k in inputs if k not in params]
            if extras:
                raise ComponentConfigurationError(
                    f"FAIL-FAST: LocalParslAdapter inputs has unexpected "
                    f"keys {extras!r} not in callable signature; "
                    f"declared params: {list(params.keys())}"
                )

        missing = [
            pname
            for pname, param in params.items()
            if pname not in ("self", "cls")
            and param.kind not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            )
            and param.default is inspect.Parameter.empty
            and pname not in inputs
        ]
        if missing:
            raise ComponentConfigurationError(
                f"FAIL-FAST: LocalParslAdapter inputs missing required "
                f"parameters {missing!r}; provided: {list(inputs.keys())}"
            )

        return {
            k: v
            for k, v in inputs.items()
            if k not in ("self", "cls")
            and (k in params or accepts_var_kw)
        }

    async def _ensure_parsl_loaded(self) -> None:
        """Lazy-load Parsl with the configured executor preset.

        Parsl's ``parsl.load`` is process-global; we coordinate across
        adapter instances via the class-level lock + flag. If parsl was
        loaded externally before we got here (e.g., the host application
        manages its own DFK), we respect that and do not reload.
        """
        if type(self)._parsl_loaded:
            return
        with type(self)._load_lock:
            if type(self)._parsl_loaded:
                return
            # Best-effort detection: if a DFK is already alive, skip load.
            import parsl

            try:
                parsl.dfk()
                # An external DFK exists. Mark loaded but don't claim ownership.
                type(self)._parsl_loaded = True
                return
            except (RuntimeError, Exception):
                # No DFK active — proceed with our load.
                pass

            cfg = self._parsl_config or self._build_default_config()
            try:
                parsl.load(cfg)
            except Exception as exc:
                raise ComponentConfigurationError(
                    f"FAIL-FAST: LocalParslAdapter parsl.load failed: "
                    f"{type(exc).__name__}: {exc}. "
                    f"executor_kind={self._executor_kind!r}, "
                    f"max_workers={self._max_workers}"
                ) from exc

            type(self)._parsl_loaded = True
            self._loaded_by_self = True

    def _build_default_config(self) -> Any:
        """Build the default Parsl Config for the requested preset.

        Local + thread (default): ThreadPoolExecutor (no provider).
        Local + process / htex: HighThroughputExecutor + LocalProvider.
        """
        from parsl.config import Config

        if self._executor_kind == "thread":
            from parsl.executors.threads import ThreadPoolExecutor

            return Config(
                executors=[
                    ThreadPoolExecutor(
                        max_threads=self._max_workers,
                        label="local_parsl_thread",
                    )
                ],
                run_dir=".parsl_runinfo_local_thread",
            )

        # process / htex — same underlying executor.
        from parsl.executors import HighThroughputExecutor
        from parsl.providers import LocalProvider

        return Config(
            executors=[
                HighThroughputExecutor(
                    label="local_parsl_htex",
                    max_workers_per_node=self._max_workers,
                    provider=LocalProvider(
                        init_blocks=1,
                        min_blocks=1,
                        max_blocks=1,
                    ),
                )
            ],
            run_dir=".parsl_runinfo_local_htex",
        )

    def _wrap_as_python_app(self, fn: Callable[..., Any]) -> Any:
        """Wrap ``fn`` as a Parsl python_app. The decorator returns an
        ``AppBase`` callable; calling it with kwargs returns a Future."""
        from parsl.app.app import python_app

        # Preserve callable identity inside Parsl by binding the original
        # function as a default kwarg — Parsl's serialization handles
        # this on ThreadPoolExecutor (no pickling). For HighThroughputExecutor,
        # the function must be importable; `_resolve_callable` already
        # ensured that via importlib lookup.
        return python_app(fn)


# ---------------------------------------------------------------------------
# Module-level convenience: register a default LocalParslAdapter so any
# UTD whose backend is "local_parsl" can be invoked via ToolExecutionStep
# without explicit registration. Tests that need a custom configured
# adapter call ToolBackendRegistry.unregister("local_parsl") first.
# ---------------------------------------------------------------------------


def register_default_local_parsl_adapter() -> LocalParslAdapter:
    """Idempotent default-adapter registration. Returns the registered
    adapter (whether freshly constructed OR previously registered).
    """
    try:
        return ToolBackendRegistry.get(LocalParslAdapter.BACKEND_NAME)  # type: ignore[return-value]
    except KeyError:
        adapter = LocalParslAdapter()
        ToolBackendRegistry.register(adapter)
        return adapter


__all__ = [
    "ExecutorKind",
    "LocalParslAdapter",
    "register_default_local_parsl_adapter",
]
