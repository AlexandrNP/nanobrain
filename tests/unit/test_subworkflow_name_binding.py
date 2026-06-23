"""Tests for SubworkflowStep ``inner_workflow_name`` binding (the third
inner-workflow source alongside path + builder).

The name seam lets a workflow reference a reusable inner workflow (e.g. a
reasoning-pattern workflow like ``tdr_loop``) BY NAME, resolved against
``workflow_search_paths`` using the SAME YAML-precedence as the application's
workflow discovery. Resolution folds into the existing path branch, so the
load+cache+gate lifecycle is identical to path binding.

These tests use REAL files on a tmp dir (no mocks) and the REAL ``from_config``
path for the mutual-exclusion checks. The end-to-end "a name binds + loads + runs
a real reasoning-pattern workflow" is verified application-side (apecx) against
the real ``tdr_loop`` / ``rag_e2e_synthesis`` workflows — nanobrain stays generic
and does not depend on any application's workflow layout.
"""

from __future__ import annotations

import pytest

from nanobrain.core.component_base import ComponentConfigurationError
from nanobrain.library.steps.subworkflow_step import SubworkflowStep

_R = SubworkflowStep._resolve_inner_workflow_name


def _mk_workflow_dir(root, name: str, yaml_filename: str) -> None:
    """Create a real ``<root>/<name>/<yaml_filename>`` with minimal content."""
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / yaml_filename).write_text(f"name: {name}\n")


# --- resolution (real files) ------------------------------------------------

def test_resolve_star_workflow_fallback(tmp_path):
    # The tdr_loop shape: dir name != YAML stem; matched via *_workflow.yml.
    _mk_workflow_dir(tmp_path, "tdr_loop", "tdr_refine_workflow.yml")
    resolved = _R("tdr_loop", [str(tmp_path)])
    assert resolved.name == "tdr_refine_workflow.yml"
    assert resolved.is_file()


def test_resolve_prefers_workflow_yml(tmp_path):
    _mk_workflow_dir(tmp_path, "flow", "workflow.yml")
    _mk_workflow_dir(tmp_path, "flow", "flow_workflow.yml")
    assert _R("flow", [str(tmp_path)]).name == "workflow.yml"


def test_resolve_prefers_named_over_other_star(tmp_path):
    _mk_workflow_dir(tmp_path, "flow", "flow_workflow.yml")
    _mk_workflow_dir(tmp_path, "flow", "other_workflow.yml")
    assert _R("flow", [str(tmp_path)]).name == "flow_workflow.yml"


def test_resolve_ambiguous_multiple_star_fails_loud(tmp_path):
    # Real shape (the apecx code_writing/ dir): several *_workflow.yml and NO
    # canonical workflow.yml / <name>_workflow.yml must FAIL LOUD, not silently
    # pick the alphabetically-first (the silent-wrong-bind the review caught).
    _mk_workflow_dir(tmp_path, "coll", "alpha_workflow.yml")
    _mk_workflow_dir(tmp_path, "coll", "beta_workflow.yml")
    with pytest.raises(ComponentConfigurationError) as exc:
        _R("coll", [str(tmp_path)])
    msg = str(exc.value)
    assert "MULTIPLE candidate workflow" in msg
    assert "alpha_workflow.yml" in msg and "beta_workflow.yml" in msg


def test_resolve_canonical_workflow_yml_breaks_ambiguity(tmp_path):
    # workflow.yml present among multiple *_workflow.yml → unambiguous, no raise.
    _mk_workflow_dir(tmp_path, "coll", "alpha_workflow.yml")
    _mk_workflow_dir(tmp_path, "coll", "beta_workflow.yml")
    _mk_workflow_dir(tmp_path, "coll", "workflow.yml")
    assert _R("coll", [str(tmp_path)]).name == "workflow.yml"


def test_resolve_named_breaks_ambiguity(tmp_path):
    # <name>_workflow.yml present among other *_workflow.yml → picks the named one.
    _mk_workflow_dir(tmp_path, "coll", "alpha_workflow.yml")
    _mk_workflow_dir(tmp_path, "coll", "coll_workflow.yml")
    assert _R("coll", [str(tmp_path)]).name == "coll_workflow.yml"


def test_resolve_first_search_path_wins(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    _mk_workflow_dir(a, "flow", "flow_workflow.yml")
    _mk_workflow_dir(b, "flow", "flow_workflow.yml")
    resolved = _R("flow", [str(a), str(b)])
    assert str(resolved).startswith(str(a))


def test_resolve_found_in_second_search_path(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    _mk_workflow_dir(b, "flow", "flow_workflow.yml")
    resolved = _R("flow", [str(a), str(b)])
    assert str(resolved).startswith(str(b))


# --- FAIL-LOUD (never silent / never a wrong-file guess) ---------------------

def test_empty_search_paths_fails_loud(tmp_path):
    with pytest.raises(ComponentConfigurationError, match="workflow_search_paths is empty"):
        _R("flow", [])


def test_unknown_name_fails_loud_and_lists_available(tmp_path):
    _mk_workflow_dir(tmp_path, "tdr_loop", "tdr_refine_workflow.yml")
    _mk_workflow_dir(tmp_path, "best_of_n_loop", "best_of_n_workflow.yml")
    with pytest.raises(ComponentConfigurationError) as exc:
        _R("does_not_exist", [str(tmp_path)])
    msg = str(exc.value)
    assert "Available names" in msg
    assert "tdr_loop" in msg and "best_of_n_loop" in msg


def test_dir_without_workflow_yaml_fails_loud(tmp_path):
    # A <name>/ dir that exists but holds no *_workflow.yml must FAIL LOUD,
    # not silently fall through to "not found" or grab an unrelated yaml.
    # (Filename must NOT end in _workflow.yml, else it IS a valid match.)
    _mk_workflow_dir(tmp_path, "flow", "config.yml")
    with pytest.raises(ComponentConfigurationError) as exc:
        _R("flow", [str(tmp_path)])
    assert "no workflow YAML" in str(exc.value)
    assert "config.yml" in str(exc.value)


# --- default search-paths classmethod (application supplies its own) --------

def test_default_workflow_search_paths_contract():
    # Base is application-agnostic ([]); a subclass supplies its own dirs.
    assert SubworkflowStep._default_workflow_search_paths() == []

    class _Sub(SubworkflowStep):
        COMPONENT_TYPE = "test_dsp_step"

        @classmethod
        def _default_workflow_search_paths(cls):
            return ["/some/app/workflows"]

    assert _Sub._default_workflow_search_paths() == ["/some/app/workflows"]


# --- 3-way mutual exclusion (real from_config) ------------------------------

def _excl_step_yaml(tmp_path, body: str):
    p = tmp_path / "excl_step.yml"
    p.write_text("name: excl_step\n" + body)
    return p


def test_name_and_path_mutually_exclusive(tmp_path):
    p = _excl_step_yaml(
        tmp_path,
        "inner_workflow_path: /tmp/nonexistent_wf.yml\n"
        "inner_workflow_name: tdr_loop\n",
    )
    with pytest.raises(ComponentConfigurationError, match="more than one"):
        SubworkflowStep.from_config(str(p))


def test_name_and_builder_mutually_exclusive(tmp_path):
    p = _excl_step_yaml(
        tmp_path,
        "inner_workflow_builder: some.module.build\n"
        "inner_workflow_name: tdr_loop\n",
    )
    with pytest.raises(ComponentConfigurationError, match="more than one"):
        SubworkflowStep.from_config(str(p))


def test_no_inner_source_fails_loud(tmp_path):
    p = _excl_step_yaml(tmp_path, "")
    with pytest.raises(ComponentConfigurationError, match="requires an"):
        SubworkflowStep.from_config(str(p))
