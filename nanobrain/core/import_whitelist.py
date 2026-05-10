"""Class-path import whitelist (G20).

Per ``apecx-mcp-integration/docs/nanobrain_capability_gaps.md G20`` and
``apecx-mcp-integration/docs/security_threat_model.md §5.8 T-CL-1``: the
framework's ``class:`` field accepts any dotted Python path and the
loader imports it. A malicious skeleton or UTD with
``class: attacker.module.Backdoor`` triggers arbitrary code at workflow
load. This module provides the operator-configurable allowlist that
gates class imports.

Default behavior:
- The whitelist is **off by default** to preserve backward compatibility.
  Operators opt in via process-level config (``set_class_import_whitelist``)
  or per-workflow opt-in (passing ``class_import_whitelist`` to
  ``Workflow.from_config``).
- When the whitelist is set AND a class-path doesn't match any prefix
  in the allowlist, the loader FAIL-FASTs with a clear message
  naming the rejected path.

Workspace constraints honored:
- Allowlist entries are **module-prefix patterns**, not full paths
  (e.g., ``"nanobrain.*"`` matches ``nanobrain.core.step.BaseStep``).
- The check is a simple prefix match — no regex, no glob — so it's
  fast and predictable. Future enhancement (G20 v2) may add glob
  semantics; v1 is intentionally minimal.
- Process-level whitelist is process-global and thread-safe;
  per-call whitelist is additive (extends the process-level set).

Threat model:
- Defense-in-depth: this is one layer; the others are filesystem
  trust on config files (operator policy), signed configs (G19),
  and runtime sandbox isolation (T13b Docker sandbox).
- Bypass risk: a whitelisted module that internally calls
  ``import_module`` on attacker input. The whitelist must be
  applied transitively — that's a deployment audit task, not
  enforced by this module.

Layering with apecx-mcp-integration's AST scanner (G36 closure, 2026-05-09):
This module is **Stage 2** of a two-layer whitelist. Stage 1 is the
integration's static AST scanner at
``apecx-mcp-integration/src/apecx_integration/composition/sandbox.py``,
which scans LLM-emitted PYTHON SOURCE before it becomes an Artifact.
Stage 1 catches dynamic-import escapes (``importlib.import_module``,
``exec``, ``eval``); Stage 2 catches malicious YAML ``class:`` paths
at framework load time. The two layers are intentionally complementary —
folding them into one would leave attack surfaces open at one of the
two entry points (LLM-emit boundary OR YAML load boundary). See
``apecx-mcp-integration/docs/whitelist_layering.md`` for the full
contract + per-deployment audit checklist.
"""

from __future__ import annotations

import logging
import threading
from typing import Iterable, List, Optional

logger = logging.getLogger(__name__)


# Process-global whitelist. None means "no whitelist set; allow everything"
# (legacy behavior). An empty list means "deny everything except what
# you explicitly add" — useful for tests of the deny path.
_GLOBAL_WHITELIST: Optional[List[str]] = None
_WHITELIST_LOCK = threading.Lock()


def set_class_import_whitelist(prefixes: Optional[Iterable[str]]) -> None:
    """Configure the process-global class-import whitelist.

    Args:
        prefixes: Iterable of module-prefix patterns (e.g.,
            ``["nanobrain.", "apecx_integration."]``). Pass ``None``
            to disable the whitelist (legacy "allow all" behavior).
            Pass ``[]`` to deny all (useful for tests).

    Notes:
        - Each prefix is a string the import-path must START WITH.
          Trailing-dot is recommended (``"nanobrain."``) so
          ``"nanobrain"`` doesn't accidentally match
          ``"nanobrain_evil"``.
        - The whitelist applies process-wide. Tests that change it
          MUST restore the previous value in teardown — see
          ``with_class_import_whitelist`` for a context manager.
    """
    global _GLOBAL_WHITELIST
    with _WHITELIST_LOCK:
        if prefixes is None:
            _GLOBAL_WHITELIST = None
        else:
            _GLOBAL_WHITELIST = list(prefixes)
        logger.debug(
            "class_import_whitelist set to %s (None = allow all; "
            "[] = deny all)",
            _GLOBAL_WHITELIST,
        )


def get_class_import_whitelist() -> Optional[List[str]]:
    """Return the current process-global whitelist (or None when disabled)."""
    with _WHITELIST_LOCK:
        return None if _GLOBAL_WHITELIST is None else list(_GLOBAL_WHITELIST)


class _WhitelistContext:
    """Context manager that temporarily overrides the whitelist.

    Used by tests (and by operator-script callers that want a scoped
    override). Restores the previous value on exit, even on exception.
    """

    def __init__(self, prefixes: Optional[Iterable[str]]) -> None:
        self._new = list(prefixes) if prefixes is not None else None
        self._old: Optional[List[str]] = None

    def __enter__(self) -> "_WhitelistContext":
        global _GLOBAL_WHITELIST
        with _WHITELIST_LOCK:
            self._old = (
                None if _GLOBAL_WHITELIST is None else list(_GLOBAL_WHITELIST)
            )
            _GLOBAL_WHITELIST = self._new
        return self

    def __exit__(self, *exc) -> None:
        global _GLOBAL_WHITELIST
        with _WHITELIST_LOCK:
            _GLOBAL_WHITELIST = self._old


def with_class_import_whitelist(
    prefixes: Optional[Iterable[str]],
) -> _WhitelistContext:
    """Context-manager API for scoped whitelist overrides.

    .. code-block:: python

        with with_class_import_whitelist(["nanobrain."]):
            wf = Workflow.from_config(path)  # only nanobrain.* classes load
        # Outside: previous whitelist restored.
    """
    return _WhitelistContext(prefixes)


def check_class_import_allowed(
    class_path: str,
    *,
    extra_whitelist: Optional[Iterable[str]] = None,
) -> None:
    """Raise ``ImportError`` if ``class_path`` is not allowed by the
    current whitelist.

    Args:
        class_path: The dotted Python path being imported (e.g.,
            ``"nanobrain.core.step.BaseStep"``).
        extra_whitelist: Per-call additional prefixes to allow. Useful
            for ``Workflow.from_config(class_import_whitelist=[...])`` —
            the per-workflow opt-in extends the process-level set.

    Behavior:
        - Process-global whitelist is ``None`` AND no extra: allow
          (legacy behavior).
        - Otherwise: combine process-global + extra; allow only if
          ``class_path`` starts with any prefix in the combined set.

    Raises:
        ImportError: with a clear message naming the rejected path
            and the allowed prefixes.
    """
    process_wl = get_class_import_whitelist()
    extra_list = list(extra_whitelist) if extra_whitelist else []

    # No whitelist at all → legacy "allow everything" behavior.
    if process_wl is None and not extra_list:
        return

    # Combine process + extra into one allowed-prefix set.
    allowed: List[str] = []
    if process_wl is not None:
        allowed.extend(process_wl)
    allowed.extend(extra_list)

    for prefix in allowed:
        if class_path.startswith(prefix):
            return

    raise ImportError(
        f"FAIL-FAST: class_import_whitelist rejected {class_path!r}. "
        f"Allowed prefixes: {allowed}. "
        f"To allow this class, add a matching prefix to the whitelist "
        f"via set_class_import_whitelist(...) OR pass it via the "
        f"workflow's class_import_whitelist option. "
        f"See nanobrain_capability_gaps.md G20 + security_threat_model.md "
        f"§5.8 T-CL-1."
    )
