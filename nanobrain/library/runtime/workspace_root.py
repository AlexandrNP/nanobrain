"""locate_workflow_root — G40 framework helper for workspace-root resolution.

eval_03 Round 4 G40: pre-G40 every consumer that needed to find the
workspace root walked Path(__file__).parents[N] manually with
hard-coded depths. The brittle assumption broke when:
  * the package was moved up or down a directory level (depth changed)
  * the package was installed into site-packages (no workspace root
    exists at all)
  * the workspace was checked out into a non-canonical layout

apecx-mcp-integration's ``_workspace.py`` (commit 01bd0ba — "consolidate
workspace-root resolution into shared utility") solved this for the
integration; G40 lifts that pattern into the framework so every
consumer (the integration, future demos, downstream operators) gets
the same primitive.

## Resolution strategy

Walk upward from a starting directory looking for ANY of a set of
canonical marker files / directories. The first ancestor containing
ANY marker is returned as the root.

Default markers (most common first):

  - ``pyproject.toml``        — Python project root
  - ``.git``                  — git repo root
  - ``setup.py``              — legacy Python project root
  - ``CLAUDE.md``             — workspace-config marker
  - ``apecx-mcp-integration`` — workspace sibling marker (apecx-cowork
                                layout where the workflow lives in a
                                sibling repo of the workspace root)

Operators with non-canonical layouts pass ``markers=`` explicitly.

## Environment-variable override

``$NANOBRAIN_WORKSPACE_ROOT`` (when set) short-circuits the walk and
returns the env-var value verbatim (after Path expansion). Operators
who deploy in non-walkable layouts (Docker bind mounts that don't
reach the marker, CI runners with file-system isolation) set this
explicitly.

## Failure mode

When NO ancestor contains a marker AND no env-var override is set,
``locate_workflow_root`` returns ``None`` rather than raising. The
caller decides whether absence is fatal — most callers want a clear
error message (FAIL-FAST), some want a tempdir fallback (e.g., log
sinks). The helper is intentionally permissive; the FAIL-FAST is
the caller's policy.

## Usage::

    from nanobrain.library.runtime.workspace_root import locate_workflow_root

    root = locate_workflow_root()
    if root is None:
        raise RuntimeError("FAIL-FAST: workspace root not found...")
    config_dir = root / "configs"

    # With explicit start + markers:
    root = locate_workflow_root(
        start=Path(__file__).parent,
        markers=["my_workspace_marker.toml"],
    )

Source: ``apecx-mcp-integration/eval_03_nanobrain_gap_inventory.md``
Round 4 G40;
``apecx-mcp-integration/docs/development_roadmap.md`` 8.7 Tier 4.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

logger = logging.getLogger(__name__)


# The default marker set. Order matters for fast-path detection: the
# most common project-root signals come first so the walk short-
# circuits on the first match.
DEFAULT_MARKERS: tuple[str, ...] = (
    "pyproject.toml",
    ".git",
    "setup.py",
    "CLAUDE.md",
    "apecx-mcp-integration",
)


def locate_workflow_root(
    *,
    start: Optional[Path] = None,
    markers: Optional[Sequence[str]] = None,
    env_var: str = "NANOBRAIN_WORKSPACE_ROOT",
) -> Optional[Path]:
    """Find the workspace root by walking upward from ``start``.

    Args:
        start: Starting directory (or any path inside it). Defaults
            to ``Path.cwd()`` when None.
        markers: Filenames / directory names to look for. The first
            ancestor containing ANY listed marker is returned.
            Defaults to ``DEFAULT_MARKERS``.
        env_var: Name of the env var that, when set, overrides the
            walk. Defaults to ``"NANOBRAIN_WORKSPACE_ROOT"``.

    Returns:
        ``Path`` of the resolved workspace root, OR ``None`` when
        no ancestor contains any marker AND no env var is set.

    The walk terminates at the filesystem root (``/``). When the
    starting directory does not exist, the walk starts from its
    closest existing ancestor — operators sometimes pass
    ``Path(__file__).parent`` from a frozen module that no longer
    exists on disk; the helper degrades gracefully.
    """
    # Env-var fast path. Handle empty string and explicit None.
    if env_var:
        raw = os.environ.get(env_var)
        if raw:
            resolved = Path(raw).expanduser()
            logger.debug(
                "locate_workflow_root: env var %s -> %s",
                env_var,
                resolved,
            )
            return resolved

    marker_set: tuple[str, ...] = (
        tuple(markers) if markers is not None else DEFAULT_MARKERS
    )
    if not marker_set:
        # Empty marker list = caller is opting out of marker-based
        # detection. Without an env var override, we cannot answer.
        return None

    if start is None:
        start = Path.cwd()
    candidate = Path(start).resolve()

    # If the starting path doesn't exist, fall back to its closest
    # existing ancestor.
    while candidate != candidate.parent and not candidate.exists():
        candidate = candidate.parent

    # Walk upward.
    for ancestor in [candidate] + list(candidate.parents):
        if _has_any_marker(ancestor, marker_set):
            logger.debug(
                "locate_workflow_root: matched %s via marker (start=%s)",
                ancestor,
                start,
            )
            return ancestor

    return None


def require_workflow_root(
    *,
    start: Optional[Path] = None,
    markers: Optional[Sequence[str]] = None,
    env_var: str = "NANOBRAIN_WORKSPACE_ROOT",
) -> Path:
    """Like ``locate_workflow_root`` but FAIL-FAST when no root is
    found. Callers that consider absence fatal use this entrypoint
    so the error message is uniform across the codebase.
    """
    root = locate_workflow_root(
        start=start, markers=markers, env_var=env_var
    )
    if root is None:
        marker_set: Iterable[str] = (
            markers if markers is not None else DEFAULT_MARKERS
        )
        raise RuntimeError(
            f"FAIL-FAST: locate_workflow_root could not find a "
            f"workspace root by walking upward from "
            f"{start or Path.cwd()!s}. Searched markers: "
            f"{list(marker_set)}. Set the {env_var!r} env var to "
            f"the absolute path OR add a marker file to your "
            f"workspace root."
        )
    return root


def _has_any_marker(directory: Path, markers: Sequence[str]) -> bool:
    """Return True if ``directory`` contains any of the listed
    marker files / directories. Marker membership is by direct
    child existence — we don't recurse, since the marker is a
    workspace-root signal."""
    if not directory.is_dir():
        return False
    for m in markers:
        if (directory / m).exists():
            return True
    return False


__all__ = [
    "DEFAULT_MARKERS",
    "locate_workflow_root",
    "require_workflow_root",
]
