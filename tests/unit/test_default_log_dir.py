"""G33 — pin the writable-default log directory contract.

Pre-G33 the framework defaulted both ``async_logging.py`` and
``logging_system.py`` to ``Path("logs")``, which is *cwd-relative*. Any
caller launched with a read-only cwd (Claude Desktop on macOS launches
MCP servers with cwd=``/``) crashed with
``[Errno 30] Read-only file system: 'logs'`` at logger initialization
time. The crash happened before any user code ran, so the workflow
appeared to "fail to load" with no traceable cause.

Post-G33 ``_default_writable_log_dir()`` resolves to:

  1. ``$NANOBRAIN_LOG_DIR`` (operator override)
  2. ``~/.cache/nanobrain/logs/`` (writable on every POSIX user account)
  3. ``$TMPDIR/nanobrain-logs`` (last-resort fallback)

This test pins:

  * the function returns an *absolute* path (cwd-independent)
  * the function never returns the legacy ``Path("logs")`` literal
  * the env-var override is honored
  * the home-directory branch resolves under ``~/.cache``
  * the tempdir fallback is reachable when home does not exist

If somebody reverts to ``Path("logs")`` to "make tests cleaner" or some
similar shortcut, every assertion below fires.

Source: ``apecx-mcp-integration/eval_03_nanobrain_gap_inventory.md``
Round 4 G33; ``apecx-mcp-integration/docs/development_roadmap.md`` 8.6.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from nanobrain.core.async_logging import (
    _default_writable_log_dir as async_default,
)
from nanobrain.core.logging_system import (
    _default_writable_log_dir as system_default,
)

# Both modules ship the same helper; pin both so a refactor that makes
# them diverge is caught by this test.
_DEFAULT_FNS = pytest.mark.parametrize(
    "default_fn",
    [async_default, system_default],
    ids=["async_logging", "logging_system"],
)


@_DEFAULT_FNS
def test_default_log_dir_is_absolute(default_fn, monkeypatch):
    """Result must be cwd-independent. The pre-G33 ``Path('logs')``
    default was relative and broke under read-only cwd."""
    # Force the home-branch by clearing the env-var override.
    monkeypatch.delenv("NANOBRAIN_LOG_DIR", raising=False)
    result = default_fn()
    assert result.is_absolute(), (
        f"{default_fn.__name__} returned a relative path "
        f"({result!r}); G33 contract requires an absolute path."
    )


@_DEFAULT_FNS
def test_default_log_dir_is_not_legacy_logs_literal(default_fn, monkeypatch):
    """The legacy ``Path('logs')`` was the silent-failure source. If a
    refactor restores it, this test fires regardless of which branch
    in ``_default_writable_log_dir`` was hit."""
    monkeypatch.delenv("NANOBRAIN_LOG_DIR", raising=False)
    result = default_fn()
    assert result != Path("logs"), (
        f"{default_fn.__name__} reverted to the cwd-relative "
        f"``Path('logs')`` default that G33 was meant to retire."
    )
    assert "logs" not in (result.parts[:1] if not result.is_absolute() else ()), (
        f"{default_fn.__name__} returned a path whose first component "
        f"is the bare ``logs`` literal; that was the cwd-relative shape."
    )


@_DEFAULT_FNS
def test_default_log_dir_honors_env_var(default_fn, monkeypatch, tmp_path):
    """Operators must be able to override the default location via
    ``NANOBRAIN_LOG_DIR``. This is the documented operator-side fix
    for any deployment whose home/tmp choices are unsuitable."""
    target = tmp_path / "operator-chosen-log-dir"
    monkeypatch.setenv("NANOBRAIN_LOG_DIR", str(target))
    result = default_fn()
    assert result == target, (
        f"{default_fn.__name__} did not honor NANOBRAIN_LOG_DIR; "
        f"expected {target!r}, got {result!r}."
    )


@_DEFAULT_FNS
def test_default_log_dir_uses_home_cache_when_available(
    default_fn, monkeypatch
):
    """When ``NANOBRAIN_LOG_DIR`` is unset and home exists, the path
    must resolve under ``~/.cache/nanobrain/logs/`` (XDG-style cache
    semantics — writable on every POSIX user account)."""
    monkeypatch.delenv("NANOBRAIN_LOG_DIR", raising=False)
    result = default_fn()
    home = Path.home()
    if not home.exists():
        pytest.skip(
            "home directory does not exist on this runner; "
            "the tempdir-fallback test exercises that branch instead."
        )
    expected = home / ".cache" / "nanobrain" / "logs"
    assert result == expected, (
        f"{default_fn.__name__}: expected ~/.cache/nanobrain/logs "
        f"branch ({expected!r}); got {result!r}."
    )


@_DEFAULT_FNS
def test_default_log_dir_falls_back_to_tempdir_when_no_home(
    default_fn, monkeypatch
):
    """Last-resort branch: when home does not exist (some CI sandboxes,
    container images without /root mounted), the function must still
    return a writable path rather than raising."""
    monkeypatch.delenv("NANOBRAIN_LOG_DIR", raising=False)
    fake_missing_home = Path("/this/path/intentionally/does/not/exist")
    with patch.object(Path, "home", classmethod(lambda cls: fake_missing_home)):
        result = default_fn()
    expected = Path(tempfile.gettempdir()) / "nanobrain-logs"
    assert result == expected, (
        f"{default_fn.__name__}: expected tempdir fallback "
        f"({expected!r}); got {result!r}."
    )


@_DEFAULT_FNS
def test_default_log_dir_is_unaffected_by_cwd(default_fn, monkeypatch, tmp_path):
    """The G33 silent-failure shape: a cwd change must NOT change the
    result. Pre-G33 the default was ``Path('logs')`` and changing cwd
    changed where logs landed. The post-G33 default is anchored to
    home (or env or tempdir), all of which are cwd-independent."""
    monkeypatch.delenv("NANOBRAIN_LOG_DIR", raising=False)
    saved_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        result_under_tmp = default_fn()
        os.chdir(saved_cwd)
        result_under_saved = default_fn()
    finally:
        # Belt-and-suspenders cwd restore in case anything raised
        # mid-block; matters because pytest runs many tests in sequence.
        if os.getcwd() != saved_cwd:
            os.chdir(saved_cwd)
    assert result_under_tmp == result_under_saved, (
        f"{default_fn.__name__} returned different paths from "
        f"different cwds — G33 regression. tmp={result_under_tmp!r}, "
        f"saved={result_under_saved!r}."
    )
