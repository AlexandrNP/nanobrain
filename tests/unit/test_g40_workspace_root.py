"""G40 — pin the locate_workflow_root contract.

eval_03 Round 4 G40: pre-G40 every consumer that needed to find the
workspace root walked Path(__file__).parents[N] manually. Brittle
when the package was moved up or down a directory level OR installed
into site-packages.

Post-G40 ships locate_workflow_root + require_workflow_root.

This test pins:
  1. locate finds an ancestor containing a default marker
  2. locate returns the FIRST matching ancestor (closest, not root)
  3. locate returns None when no marker found AND no env var
  4. require raises with diagnostic message when no marker found
  5. env var override short-circuits the walk
  6. custom markers list is honored (default markers ignored)
  7. empty markers list returns None (caller opted out)
  8. starting from a non-existent path falls back to closest existing
     ancestor
  9. marker can be a directory (e.g. .git) not just a file

Source: ``apecx-mcp-integration/eval_03_nanobrain_gap_inventory.md``
Round 4 G40;
``apecx-mcp-integration/docs/development_roadmap.md`` 8.7 Tier 4.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from nanobrain.library.runtime.workspace_root import (
    DEFAULT_MARKERS,
    locate_workflow_root,
    require_workflow_root,
)


def _make_workspace(tmp_path: Path, marker: str = "pyproject.toml") -> Path:
    """Build a small workspace tree with a marker at root + nested
    sub-directories the caller can locate from."""
    root = tmp_path / "ws"
    root.mkdir()
    (root / marker).touch()
    sub = root / "src" / "pkg" / "sub"
    sub.mkdir(parents=True)
    return root


def test_locate_finds_ancestor_with_default_marker(tmp_path):
    root = _make_workspace(tmp_path, marker="pyproject.toml")
    sub = root / "src" / "pkg" / "sub"
    found = locate_workflow_root(start=sub)
    assert found == root


def test_locate_returns_closest_ancestor(tmp_path):
    """When two ancestors both have a marker, locate returns the
    CLOSEST one (deepest) — operators expect "this is my project
    root", not "the highest ancestor I can find"."""
    outer = tmp_path / "outer"
    inner = outer / "inner"
    inner.mkdir(parents=True)
    (outer / "pyproject.toml").touch()
    (inner / "pyproject.toml").touch()
    deeper = inner / "src"
    deeper.mkdir()
    found = locate_workflow_root(start=deeper)
    assert found == inner, (
        f"closest ancestor should win; got {found}, expected {inner}"
    )


def test_locate_returns_none_when_no_marker(tmp_path, monkeypatch):
    """No markers anywhere in the tree → None. Caller decides
    whether absence is fatal."""
    monkeypatch.delenv("NANOBRAIN_WORKSPACE_ROOT", raising=False)
    sub = tmp_path / "isolated" / "sub"
    sub.mkdir(parents=True)
    found = locate_workflow_root(start=sub)
    assert found is None


def test_require_raises_with_diagnostic(tmp_path, monkeypatch):
    """require_workflow_root FAIL-FASTs when no root found, with
    a message that names the start path + searched markers + env-var
    override."""
    monkeypatch.delenv("NANOBRAIN_WORKSPACE_ROOT", raising=False)
    sub = tmp_path / "isolated"
    sub.mkdir()
    with pytest.raises(RuntimeError) as excinfo:
        require_workflow_root(start=sub)
    msg = str(excinfo.value)
    assert "FAIL-FAST" in msg
    assert "NANOBRAIN_WORKSPACE_ROOT" in msg
    assert "pyproject.toml" in msg or ".git" in msg


def test_env_var_overrides_walk(tmp_path, monkeypatch):
    """``$NANOBRAIN_WORKSPACE_ROOT`` short-circuits — the env-var
    value wins regardless of marker presence elsewhere."""
    target = tmp_path / "operator_chosen"
    target.mkdir()
    monkeypatch.setenv("NANOBRAIN_WORKSPACE_ROOT", str(target))
    # Walk starts from a different directory; env var still wins.
    sub = tmp_path / "elsewhere" / "deep"
    sub.mkdir(parents=True)
    found = locate_workflow_root(start=sub)
    assert found == target


def test_custom_markers_honored(tmp_path, monkeypatch):
    """Operators with non-canonical layouts pass markers= explicitly.
    Default markers are NOT consulted when custom markers are given."""
    monkeypatch.delenv("NANOBRAIN_WORKSPACE_ROOT", raising=False)
    root = tmp_path / "ws"
    root.mkdir()
    # Default marker present but should be IGNORED (custom markers only).
    (root / "pyproject.toml").touch()
    # Custom marker.
    (root / "WORKSPACE_MARKER").touch()

    sub = root / "deep" / "subdir"
    sub.mkdir(parents=True)

    found = locate_workflow_root(
        start=sub, markers=["WORKSPACE_MARKER"]
    )
    assert found == root

    # Now: with custom markers that DON'T match, even though
    # pyproject.toml is present, we get None.
    found_none = locate_workflow_root(
        start=sub, markers=["nope_does_not_exist"]
    )
    assert found_none is None


def test_empty_markers_returns_none(tmp_path, monkeypatch):
    """Empty marker list = caller opted out of marker-based detection.
    Without an env-var override, we cannot answer."""
    monkeypatch.delenv("NANOBRAIN_WORKSPACE_ROOT", raising=False)
    root = _make_workspace(tmp_path)
    found = locate_workflow_root(start=root, markers=[])
    assert found is None


def test_nonexistent_start_falls_back_to_closest_ancestor(tmp_path):
    """Operators sometimes pass Path(__file__).parent from a frozen
    module that no longer exists on disk; the helper walks up from
    the closest existing ancestor."""
    root = _make_workspace(tmp_path)
    # Construct a path that doesn't exist but whose ancestors do.
    fake = root / "src" / "pkg" / "frozen_module_dir" / "removed_subdir"
    found = locate_workflow_root(start=fake)
    assert found == root


def test_marker_can_be_directory_not_just_file(tmp_path, monkeypatch):
    """Directory markers (e.g., .git) should be detected the same as
    file markers."""
    monkeypatch.delenv("NANOBRAIN_WORKSPACE_ROOT", raising=False)
    root = tmp_path / "git_repo"
    root.mkdir()
    (root / ".git").mkdir()  # directory marker
    sub = root / "src"
    sub.mkdir()
    found = locate_workflow_root(start=sub)
    assert found == root


def test_default_markers_set_includes_canonical_signals():
    """Sanity check: the default marker set covers the most common
    workspace-root signals. If somebody removes one, this fires."""
    assert "pyproject.toml" in DEFAULT_MARKERS
    assert ".git" in DEFAULT_MARKERS
