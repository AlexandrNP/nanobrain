"""Distributed execution components for NanoBrain framework.

This package wraps Parsl-based distributed execution AND Globus
Compute / Globus Auth integrations. Each of those backends has heavy
optional dependencies (``parsl``, ``globus-compute-sdk``,
``globus-sdk``) that aren't part of the framework's core install — so
this ``__init__`` MUST NOT eagerly import any submodule that pulls
them.

Why this matters
----------------
Earlier this file did ``from .workflow_execution import
execute_workflow_distributed`` at module level. ``workflow_execution``
imports ``parsl.app.app`` at top level, so ANY code path that touched
``nanobrain.core.distributed`` (even transitively via
``nanobrain.library.steps`` re-exports → ``globus_transfer_step`` →
``globus_auth``) would crash with ``ModuleNotFoundError: No module
named 'parsl'`` when parsl wasn't installed — which is the default
state for the ``nanobrain[test]`` CI extra. Result: pytest collection
itself failed for ~10 unrelated test modules and the entire suite
exited with non-zero.

The fix is PEP 562 module-level ``__getattr__``: the
``execute_workflow_distributed`` import surface is preserved
(``from nanobrain.core.distributed import execute_workflow_distributed``
still works), but the import is deferred until ATTRIBUTE ACCESS
rather than fired at module load.

If you genuinely want the parsl-backed function, you'll already need
parsl installed at access time — same hard requirement as before,
just deferred to the moment it actually matters.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover — type-checker only
    from .workflow_execution import execute_workflow_distributed  # noqa: F401

__all__ = ["execute_workflow_distributed"]


def __getattr__(name: str) -> Any:
    """PEP 562 lazy attribute access for optional-deps re-exports.

    ``from nanobrain.core.distributed import execute_workflow_distributed``
    triggers this lazily — parsl is only imported at that moment.
    Importing the package itself (which happens transitively via
    ``library.steps`` and ``library.steps.globus_transfer_step``)
    does NOT pull parsl in.
    """
    if name == "execute_workflow_distributed":
        from .workflow_execution import execute_workflow_distributed

        return execute_workflow_distributed
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}. Available "
        f"lazy attributes: {__all__!r}."
    )
