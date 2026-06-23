"""T3: `config_search_paths` resolver in `ConfigBase._resolve_config_path`.

Adds an extra LAST resolution strategy (after every in-tree strategy) so a
relative `config:` reference can resolve against caller-injected roots — the
seam an executor uses to stage a composed workflow that reuses wrappers from
multiple catalog dirs. Real tmp trees, no mocks.

Scope note: these unit-test the resolver + its precedence/ambiguity contract.
The end-to-end threading (config_search_paths kwarg -> nested from_config
loads) + the routing fact that a workflow's nested step `config:` refs go
through THIS context-aware resolver (config_base) rather than the context-blind
`component_base._resolve_config_file_path` are exercised dynamically by the
apecx executor e2e (multi-dir reuse -> RUN_COMPLETED) and the resolution
migration audit. The routing was also confirmed by code-read: the executor
loads via `Workflow.from_config(str(absolute_staged_yaml))`, whose absolute top
path is a no-op in component_base (:722-724), so nested refs route here.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from nanobrain.core.config.config_base import ConfigLoadingContext
from nanobrain.core.workflow import WorkflowConfig  # representative ConfigBase subclass


def _ctx(base_path: Path, search_paths: list[str] | None = None) -> ConfigLoadingContext:
    return ConfigLoadingContext(
        base_path=Path(base_path),
        resolution_stack=set(),
        loading_timestamp=datetime.now(),
        config_search_paths=search_paths,
    )


def _write(p: Path, text: str = "name: x\n") -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def test_resolves_via_search_root_when_in_tree_strategies_miss(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    target = _write(tmp_path / "rootA" / "steps" / "x.yml")
    resolved = WorkflowConfig._resolve_config_path("steps/x.yml", _ctx(base, [str(tmp_path / "rootA")]))
    assert Path(resolved) == target.resolve()


def test_base_path_wins_over_search_root(tmp_path):
    # Precedence: a co-located ref resolves via base_path (Strategy 2); the
    # search-path strategy is LAST and must NOT be consulted here.
    base = tmp_path / "base"
    base_target = _write(base / "steps" / "x.yml", "name: base\n")
    _write(tmp_path / "rootA" / "steps" / "x.yml", "name: search\n")
    resolved = WorkflowConfig._resolve_config_path("steps/x.yml", _ctx(base, [str(tmp_path / "rootA")]))
    assert Path(resolved) == base_target.resolve()


def test_ambiguous_match_across_two_roots_raises(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    _write(tmp_path / "rootA" / "steps" / "x.yml", "name: a\n")
    _write(tmp_path / "rootB" / "steps" / "x.yml", "name: b\n")
    ctx = _ctx(base, [str(tmp_path / "rootA"), str(tmp_path / "rootB")])
    with pytest.raises(ValueError, match="AMBIGUOUS"):
        WorkflowConfig._resolve_config_path("steps/x.yml", ctx)


def test_same_file_via_two_roots_dedups_not_ambiguous(tmp_path):
    # Two roots resolving to the SAME file (overlapping roots) -> de-dup -> resolve.
    base = tmp_path / "base"
    base.mkdir()
    target = _write(tmp_path / "rootA" / "steps" / "x.yml")
    root = str(tmp_path / "rootA")
    resolved = WorkflowConfig._resolve_config_path("steps/x.yml", _ctx(base, [root, root]))
    assert Path(resolved) == target.resolve()


def test_no_search_paths_is_noop(tmp_path):
    # Empty/None roots -> new strategy is inert; a genuinely-missing ref still raises.
    base = tmp_path / "base"
    base.mkdir()
    with pytest.raises(FileNotFoundError):
        WorkflowConfig._resolve_config_path("steps/missing.yml", _ctx(base, None))


def test_search_root_miss_still_raises_filenotfound(tmp_path):
    # Roots present but the ref isn't under any of them -> fall through to the
    # existing not-found error (search-path only turns failure->success).
    base = tmp_path / "base"
    base.mkdir()
    (tmp_path / "rootA").mkdir()
    with pytest.raises(FileNotFoundError):
        WorkflowConfig._resolve_config_path("steps/missing.yml", _ctx(base, [str(tmp_path / "rootA")]))
