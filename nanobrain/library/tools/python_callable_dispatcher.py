"""PythonCallableDispatcher + ToolBase.from_python_callable shortcut.

The ergonomic counterpart to ``RheaMCPDispatcher``: where the latter
wraps a remote MCP tool, this one wraps an in-process Python callable.
Both produce ``ToolBase`` instances; both are descriptor-driven; both
flow through ``ToolBase.from_descriptor``.

## Usage

    from nanobrain.core.tool import ToolBase

    def search_my_corpus(query: str, max_results: int = 10) -> dict:
        '''Search the local corpus for query terms.'''
        return {"hits": [...]}

    tool = ToolBase.from_python_callable(search_my_corpus)
    result = await tool.execute({"query": "hello", "max_results": 5})

    # The descriptor is auto-built; access via:
    tool.descriptor  # type: UnifiedToolDescriptor

    # Override any UTD field at construction:
    tool = ToolBase.from_python_callable(
        search_my_corpus,
        backend="apecx",
        version="2.0.0",
        side_effects="ro_external",
        cost_estimate={"estimated_seconds": 0.5, "confidence": "medium"},
    )

## Design rationale

The descriptor's ``provenance_pin.class_path`` points at the
PythonCallableDispatcher class itself (the dispatcher is the
implementation), and the **callable is passed as a kwarg** to
``from_descriptor``. Compared to a registry-based design:

- A different process can reconstruct the descriptor from JSON, but
  CANNOT reconstruct the live callable without the kwarg. Honest.
- For HPC bundle replay (per ``CONTRACTS.md#hpc-determinism``), the
  descriptor records the callable's importable path via the auto-
  derived ``UnifiedToolDescriptor.from_python_callable``'s
  ``provenance_class_path``-style hint — but that hint goes into
  the dispatcher's own state, not into the descriptor's
  ``provenance_pin`` (which always points at the dispatcher class).

## Sync vs async callables

- Async callable (``async def``) → awaited directly.
- Sync callable → run via ``asyncio.to_thread`` so the event loop
  is not blocked. This is critical for downstream Workflow callers
  who run the tool inside an asyncio task.

## Payload binding

The callable's signature is honored: ``execute(payload)`` calls
``fn(**payload)``. Extra keys in ``payload`` that don't match a
parameter raise ``ComponentConfigurationError`` (FAIL-FAST per
the workspace policy on silent failures). Missing required
parameters also FAIL-FAST.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any, Awaitable, Callable, Dict, Optional, Union

from nanobrain.core.component_base import ComponentConfigurationError
from nanobrain.core.tool import ToolBase, ToolConfig
from nanobrain.core.unified_tool_descriptor import UnifiedToolDescriptor

logger = logging.getLogger(__name__)


class PythonCallableDispatcher(ToolBase):
    """ToolBase that wraps a Python callable.

    Construct via ``ToolBase.from_python_callable(fn, ...)`` (the
    ergonomic API), or via ``ToolBase.from_descriptor(utd,
    python_callable=fn, ...)`` (the descriptor-driven API).

    Both paths produce equivalent dispatchers; the latter is what
    a bundle-replay deployment uses when the descriptor was loaded
    from JSON and the callable is reconstructed from
    ``provenance_pin.class_path`` lookup at runtime.
    """

    COMPONENT_TYPE = "python_callable_dispatcher"
    REQUIRED_CONFIG_FIELDS = ["name"]

    @classmethod
    def _get_config_class(cls):
        return ToolConfig

    def _init_from_config(
        self,
        config: ToolConfig,
        component_config: Dict[str, Any],
        dependencies: Dict[str, Any],
    ) -> None:
        super()._init_from_config(config, component_config, dependencies)
        fn = dependencies.get("python_callable")
        if not callable(fn):
            raise ComponentConfigurationError(
                f"FAIL-FAST: PythonCallableDispatcher {config.name!r} "
                f"requires the 'python_callable' kwarg (a callable); "
                f"got {fn!r}"
            )
        self._fn: Callable[..., Any] = fn
        self._is_coro_fn: bool = asyncio.iscoroutinefunction(fn)
        try:
            self._sig: Optional[inspect.Signature] = inspect.signature(fn)
        except (ValueError, TypeError):
            self._sig = None  # builtins / C-extensions
        # Optional descriptor passed through (set by from_python_callable)
        self.descriptor: Optional[UnifiedToolDescriptor] = (
            dependencies.get("descriptor")
        )

    @classmethod
    def resolve_dependencies(
        cls, component_config: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        return {
            "python_callable": kwargs.get("python_callable"),
            "descriptor": kwargs.get("descriptor"),
        }

    # ---- Public API -----------------------------------------------------

    async def execute(self, payload: Dict[str, Any]) -> Any:
        """Dispatch ``payload`` to the wrapped callable.

        - Sync callables run in ``asyncio.to_thread`` so they don't
          block the event loop.
        - Async callables are awaited directly.
        - Missing required parameters / unexpected kwargs FAIL-FAST.
        """
        if not isinstance(payload, dict):
            raise ComponentConfigurationError(
                f"FAIL-FAST: PythonCallableDispatcher.execute payload must "
                f"be a dict; got {type(payload).__name__}"
            )

        bound = self._bind_payload(payload)
        if self._is_coro_fn:
            return await self._fn(**bound)
        # Sync fn — offload to a thread so the loop stays responsive.
        return await asyncio.to_thread(self._fn, **bound)

    # ---- Internals ------------------------------------------------------

    def _bind_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that ``payload`` matches the callable's signature.

        FAIL-FAST on:
        - extra kwargs (unless the callable accepts ``**kwargs``)
        - missing required parameters

        Returns the payload as-is when the signature was unintrospectable
        (builtins / C-extensions); the dispatch then succeeds or fails
        per the callable's runtime contract.
        """
        if self._sig is None:
            return dict(payload)

        params = self._sig.parameters
        accepts_var_kw = any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
        accepts_var_pos = any(
            p.kind is inspect.Parameter.VAR_POSITIONAL for p in params.values()
        )

        # Extra-kwarg check
        if not accepts_var_kw:
            extras = [k for k in payload if k not in params]
            if extras:
                raise ComponentConfigurationError(
                    f"FAIL-FAST: PythonCallableDispatcher payload has "
                    f"unexpected keys {extras!r} not in the callable's "
                    f"signature; declared params: {list(params.keys())}"
                )

        # Missing-required check
        missing = []
        for pname, param in params.items():
            if pname in ("self", "cls"):
                continue
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            if param.default is inspect.Parameter.empty and pname not in payload:
                missing.append(pname)
        if missing:
            raise ComponentConfigurationError(
                f"FAIL-FAST: PythonCallableDispatcher payload missing "
                f"required parameters {missing!r}; provided: "
                f"{list(payload.keys())}"
            )

        # Drop self/cls from the bound payload (defensive; user shouldn't
        # ever pass them, but if they do it's a programming error we
        # surface)
        return {k: v for k, v in payload.items() if k not in ("self", "cls")
                and (k in params or accepts_var_kw)}


# ---------------------------------------------------------------------------
# Public factory — installed onto ToolBase by tool.py at import time
# (see the ToolBase.from_python_callable injection at the bottom of tool.py)
# ---------------------------------------------------------------------------

def _from_python_callable_impl(
    fn: Callable[..., Any],
    *,
    backend: str = "native",
    version: str = "0.1.0",
    provenance_class_path: Optional[str] = None,
    **utd_overrides: Any,
) -> ToolBase:
    """Build a ``ToolBase`` that dispatches to ``fn``.

    Internally:
    1. Build the UTD via ``UnifiedToolDescriptor.from_python_callable``.
    2. Override the descriptor's ``provenance_pin.class_path`` to point
       at ``PythonCallableDispatcher`` (the dispatcher IS the
       implementation, regardless of where ``fn`` lives).
    3. Materialize the dispatcher via ``ToolBase.from_descriptor``,
       passing ``python_callable=fn`` + ``descriptor=utd`` as kwargs.

    Args mirror ``UnifiedToolDescriptor.from_python_callable``: any
    UTD field can be overridden via ``**utd_overrides`` (display_name,
    summary, cost_estimate, side_effects, etc.).
    """
    utd = UnifiedToolDescriptor.from_python_callable(
        fn,
        backend=backend,
        version=version,
        provenance_class_path=provenance_class_path,
        **utd_overrides,
    )

    # Override the descriptor's provenance_pin to point at the dispatcher
    # CLASS (not the callable's import path — ``provenance_class_path``
    # records the callable's path; the dispatcher class is what
    # from_descriptor instantiates).
    #
    # We rebuild the descriptor with the corrected provenance_pin so the
    # auto-derived descriptor_hash stays consistent.
    utd_dict = utd.model_dump(mode="json")
    utd_dict["provenance_pin"]["class_path"] = (
        "nanobrain.library.tools.python_callable_dispatcher.PythonCallableDispatcher"
    )
    # Strip the descriptor_hash so the validator recomputes it (otherwise
    # the original hash from the auto-build won't match the new
    # provenance_pin and the cross-check would surface a stale hash).
    utd_dict.pop("descriptor_hash", None)
    utd_with_dispatcher = UnifiedToolDescriptor.from_dict(utd_dict)

    return ToolBase.from_descriptor(
        utd_with_dispatcher,
        python_callable=fn,
        descriptor=utd_with_dispatcher,
    )


# Public re-export of the implementation function (the canonical
# entry point is now ToolBase.from_python_callable in nanobrain.core.tool,
# which lazily imports this module). This module-level alias is preserved
# for callers who prefer the explicit import.
build_tool_from_python_callable = _from_python_callable_impl
