"""Tests for G7 Step 1+2 — config_version field + auto_transfer deprecation WARNING.

Per ``apecx-mcp-integration/docs/nanobrain_capability_gaps.md G7``: the
`LinkBase.auto_transfer` field defaults to False, which causes the dominant
silent-failure shape (workflow loads, every step runs, but no data ever
transfers because every link silently no-ops).

Step 2 of the G7 migration plan (this commit) is to emit a WARNING at workflow
load when a v1 workflow has a Link config that omits `auto_transfer`. Step 3
(land v2 semantics) and Step 4 (flip new-workflow default) are deferred per
the gap proposal.

These tests cover ONLY the WARNING-emission helper and the WorkflowConfig
field — they do NOT load full workflows (which would need the heavy
infrastructure). The integration of the helper into `Workflow._init_from_config`
is verified by the helper's signature being callable with the right arg shape;
end-to-end coverage will land with the apecx-mcp-integration MC-X-01 inventory
and a full workflow-load smoke test.
"""

from __future__ import annotations

import logging

import pytest

from nanobrain.core.workflow import (
    WorkflowConfig,
    _link_class_needs_auto_transfer_check,
    _link_inline_config_omits_auto_transfer,
    _warn_on_implicit_auto_transfer,
)


# ---------------------------------------------------------------------------
# WorkflowConfig.config_version field
# ---------------------------------------------------------------------------

class TestWorkflowConfigVersionField:

    def _build(self, **kwargs):
        WorkflowConfig._allow_direct_instantiation = True
        try:
            return WorkflowConfig(**kwargs)
        finally:
            WorkflowConfig._allow_direct_instantiation = False

    def test_default_is_v2(self):
        """G7 Step 4 (2026-05-09) — workspace-wide default flipped from
        v1 to v2. Authors who want legacy semantics must declare
        ``config_version: 1`` explicitly."""
        cfg = self._build(name="test")
        assert cfg.config_version == 2

    def test_explicit_v1_accepted(self):
        cfg = self._build(name="test", config_version=1)
        assert cfg.config_version == 1

    def test_explicit_v2_accepted(self):
        cfg = self._build(name="test", config_version=2)
        assert cfg.config_version == 2

    def test_explicit_v3_accepted(self):
        # Project A Step 2 — v3 opts into binding contract enforcement.
        cfg = self._build(name="test", config_version=3)
        assert cfg.config_version == 3

    def test_unknown_version_rejected(self):
        with pytest.raises(Exception):
            self._build(name="test", config_version=4)

    def test_string_version_rejected(self):
        with pytest.raises(Exception):
            self._build(name="test", config_version="v2")


# ---------------------------------------------------------------------------
# _link_class_needs_auto_transfer_check
# ---------------------------------------------------------------------------

class TestLinkClassDetection:

    def test_directlink(self):
        assert _link_class_needs_auto_transfer_check("nanobrain.core.link.DirectLink") is True

    def test_transformlink(self):
        assert _link_class_needs_auto_transfer_check("nanobrain.core.link.TransformLink") is True

    def test_conditionallink(self):
        assert _link_class_needs_auto_transfer_check("nanobrain.core.link.ConditionalLink") is True

    def test_filelink(self):
        assert _link_class_needs_auto_transfer_check("nanobrain.core.link.FileLink") is True

    def test_queuelink(self):
        assert _link_class_needs_auto_transfer_check("nanobrain.core.link.QueueLink") is True

    def test_academylink_skipped(self):
        """AcademyLink is intentionally not in the warning set — its config
        consistently sets auto_transfer explicitly per the existing demos."""
        assert _link_class_needs_auto_transfer_check(
            "nanobrain.academy_integration.academy_link.AcademyLink"
        ) is False

    def test_unknown_class_skipped(self):
        assert _link_class_needs_auto_transfer_check("custom.UnknownLink") is False

    def test_non_string_skipped(self):
        assert _link_class_needs_auto_transfer_check(None) is False
        assert _link_class_needs_auto_transfer_check(42) is False


# ---------------------------------------------------------------------------
# _link_inline_config_omits_auto_transfer
# ---------------------------------------------------------------------------

class TestLinkInlineConfigOmits:

    def test_nested_inline_omits(self):
        entry = {"class": "nanobrain.core.link.DirectLink",
                 "config": {"source": "a.x", "target": "b.x"}}
        assert _link_inline_config_omits_auto_transfer(entry) is True

    def test_nested_inline_present_true(self):
        entry = {"class": "nanobrain.core.link.DirectLink",
                 "config": {"source": "a.x", "target": "b.x", "auto_transfer": True}}
        assert _link_inline_config_omits_auto_transfer(entry) is False

    def test_nested_inline_present_false_is_explicit(self):
        """An explicit `auto_transfer: false` is still 'present' — the author
        made a deliberate choice. No warning."""
        entry = {"class": "nanobrain.core.link.DirectLink",
                 "config": {"source": "a.x", "target": "b.x", "auto_transfer": False}}
        assert _link_inline_config_omits_auto_transfer(entry) is False

    def test_legacy_top_level_omits(self):
        entry = {"class": "nanobrain.core.link.DirectLink",
                 "source": "a.x", "target": "b.x"}
        assert _link_inline_config_omits_auto_transfer(entry) is True

    def test_legacy_top_level_present(self):
        entry = {"class": "nanobrain.core.link.DirectLink",
                 "source": "a.x", "target": "b.x", "auto_transfer": True}
        assert _link_inline_config_omits_auto_transfer(entry) is False

    def test_path_reference_indeterminate(self):
        entry = {"class": "nanobrain.core.link.DirectLink",
                 "config": "config/my_link.yml"}
        assert _link_inline_config_omits_auto_transfer(entry) is None

    def test_resolved_linkbase_indeterminate(self):
        """Already-resolved LinkBase instances cannot be statically inspected
        for original YAML intent."""
        class FakeLink:
            pass
        assert _link_inline_config_omits_auto_transfer(FakeLink()) is None


# ---------------------------------------------------------------------------
# _warn_on_implicit_auto_transfer — the actual warning emitter
# ---------------------------------------------------------------------------

class TestWarnOnImplicitAutoTransfer:

    def test_v1_emits_warning_for_omitted_auto_transfer(self, caplog):
        links = {
            "my_link": {
                "class": "nanobrain.core.link.DirectLink",
                "config": {"source": "a.x", "target": "b.x"},
            },
        }
        with caplog.at_level(logging.WARNING, logger="nanobrain.core.workflow"):
            _warn_on_implicit_auto_transfer("test_wf", links, config_version=1)
        # Exactly one WARNING record naming the link and the workflow.
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 1
        msg = warnings[0].getMessage()
        assert "test_wf" in msg
        assert "my_link" in msg
        assert "auto_transfer" in msg
        # Cross-references included so the user can follow the breadcrumb:
        assert "G7" in msg

    def test_v2_suppresses_warning(self, caplog):
        links = {
            "my_link": {
                "class": "nanobrain.core.link.DirectLink",
                "config": {"source": "a.x", "target": "b.x"},
            },
        }
        with caplog.at_level(logging.WARNING, logger="nanobrain.core.workflow"):
            _warn_on_implicit_auto_transfer("test_wf", links, config_version=2)
        # No WARNING — v2 promises new default makes omission safe.
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 0

    def test_explicit_true_suppresses_warning(self, caplog):
        links = {
            "good_link": {
                "class": "nanobrain.core.link.DirectLink",
                "config": {"source": "a.x", "target": "b.x", "auto_transfer": True},
            },
        }
        with caplog.at_level(logging.WARNING, logger="nanobrain.core.workflow"):
            _warn_on_implicit_auto_transfer("test_wf", links, config_version=1)
        assert len([r for r in caplog.records if r.levelname == "WARNING"]) == 0

    def test_explicit_false_suppresses_warning(self, caplog):
        """Explicit auto_transfer: false is a deliberate author choice."""
        links = {
            "intentional_noop": {
                "class": "nanobrain.core.link.DirectLink",
                "config": {"source": "a.x", "target": "b.x", "auto_transfer": False},
            },
        }
        with caplog.at_level(logging.WARNING, logger="nanobrain.core.workflow"):
            _warn_on_implicit_auto_transfer("test_wf", links, config_version=1)
        assert len([r for r in caplog.records if r.levelname == "WARNING"]) == 0

    def test_academylink_does_not_trigger_warning(self, caplog):
        """AcademyLink is excluded from the static check (its templates set
        the flag explicitly)."""
        links = {
            "academy": {
                "class": "nanobrain.academy_integration.academy_link.AcademyLink",
                "config": {"source": "a.x", "target": "b.x"},
            },
        }
        with caplog.at_level(logging.WARNING, logger="nanobrain.core.workflow"):
            _warn_on_implicit_auto_transfer("test_wf", links, config_version=1)
        assert len([r for r in caplog.records if r.levelname == "WARNING"]) == 0

    def test_path_reference_emits_debug_not_warning(self, caplog):
        """Path-reference link configs cannot be statically checked.
        We emit a DEBUG note (not a WARNING) — discoverable but not loud."""
        links = {
            "ref_link": {
                "class": "nanobrain.core.link.DirectLink",
                "config": "config/my_link.yml",
            },
        }
        with caplog.at_level(logging.DEBUG, logger="nanobrain.core.workflow"):
            _warn_on_implicit_auto_transfer("test_wf", links, config_version=1)
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        debugs = [r for r in caplog.records if r.levelname == "DEBUG"]
        assert len(warnings) == 0
        assert any("path-reference" in r.getMessage() for r in debugs)

    def test_mixed_workflow_only_warns_on_implicit(self, caplog):
        links = {
            "ok": {
                "class": "nanobrain.core.link.DirectLink",
                "config": {"source": "a.x", "target": "b.x", "auto_transfer": True},
            },
            "bad": {
                "class": "nanobrain.core.link.TransformLink",
                "config": {"source": "b.x", "target": "c.x", "transform_function": "x.y.z"},
            },
        }
        with caplog.at_level(logging.WARNING, logger="nanobrain.core.workflow"):
            _warn_on_implicit_auto_transfer("test_wf", links, config_version=1)
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 1
        assert "bad" in warnings[0].getMessage()
        assert "ok" not in warnings[0].getMessage()

    def test_empty_links_no_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="nanobrain.core.workflow"):
            _warn_on_implicit_auto_transfer("test_wf", {}, config_version=1)
        assert len([r for r in caplog.records if r.levelname == "WARNING"]) == 0

    def test_resolved_linkbase_skipped(self, caplog):
        """An already-resolved LinkBase in the links dict is skipped (we
        cannot statically inspect the original YAML)."""
        class FakeResolved:
            pass

        links = {"resolved": FakeResolved()}
        with caplog.at_level(logging.WARNING, logger="nanobrain.core.workflow"):
            _warn_on_implicit_auto_transfer("test_wf", links, config_version=1)
        # No warning, no error — gracefully degraded.
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 0


# ---------------------------------------------------------------------------
# G7 Step 3 — _apply_v2_link_defaults
# Tests for the model_validator that injects auto_transfer=True into inline
# link configs when config_version >= 2.
# ---------------------------------------------------------------------------

class TestV2LinkDefaultsApplied:
    """Verify that v2 inline link configs gain auto_transfer=True
    automatically while v1 preserves the historical False default."""

    def _build(self, **kwargs):
        WorkflowConfig._allow_direct_instantiation = True
        try:
            return WorkflowConfig(**kwargs)
        finally:
            WorkflowConfig._allow_direct_instantiation = False

    def test_v1_does_not_mutate_inline_link(self):
        """Step 3 must not affect v1 workflows — legacy semantics preserved."""
        link_dict = {
            "class": "nanobrain.core.link.DirectLink",
            "source": "a.x",
            "target": "b.x",
        }
        cfg = self._build(name="test", config_version=1, links={"l": link_dict})
        # Field is absent (NOT injected) — would cause LinkConfig to use False.
        assert "auto_transfer" not in cfg.links["l"]

    def test_v2_injects_auto_transfer_flat_shape(self):
        """Flat-shape inline link config (no nested 'config' key) gets the
        injection at the top level."""
        link_dict = {
            "class": "nanobrain.core.link.DirectLink",
            "source": "a.x",
            "target": "b.x",
        }
        cfg = self._build(name="test", config_version=2, links={"l": link_dict})
        assert cfg.links["l"]["auto_transfer"] is True

    def test_v2_injects_auto_transfer_nested_shape(self):
        """Nested-shape inline link config ({class, config: {...}}) gets
        the injection inside the 'config' dict."""
        link_dict = {
            "class": "nanobrain.core.link.DirectLink",
            "config": {"source": "a.x", "target": "b.x"},
        }
        cfg = self._build(name="test", config_version=2, links={"l": link_dict})
        assert cfg.links["l"]["config"]["auto_transfer"] is True

    def test_v2_does_not_override_explicit_true(self):
        """Explicit True is left alone — setdefault is a no-op when the key exists."""
        link_dict = {
            "class": "nanobrain.core.link.DirectLink",
            "source": "a.x", "target": "b.x",
            "auto_transfer": True,
        }
        cfg = self._build(name="test", config_version=2, links={"l": link_dict})
        assert cfg.links["l"]["auto_transfer"] is True

    def test_v2_does_not_override_explicit_false(self):
        """Explicit False is THE non-obvious case: an author wrote
        auto_transfer: False intentionally and the v2 default must NOT
        override that. setdefault is the right primitive precisely
        because it preserves explicit values of any truthiness."""
        link_dict = {
            "class": "nanobrain.core.link.DirectLink",
            "source": "a.x", "target": "b.x",
            "auto_transfer": False,
        }
        cfg = self._build(name="test", config_version=2, links={"l": link_dict})
        assert cfg.links["l"]["auto_transfer"] is False

    def test_v2_skips_unknown_link_class(self):
        """Unknown link classes (e.g., AcademyLink, custom) are not
        mutated — Step 3 only touches the known auto_transfer-bearing
        classes whitelisted by _link_class_needs_auto_transfer_check."""
        link_dict = {
            "class": "some.custom.UnknownLink",
            "source": "a.x", "target": "b.x",
        }
        cfg = self._build(name="test", config_version=2, links={"l": link_dict})
        assert "auto_transfer" not in cfg.links["l"]

    def test_v2_rewrites_path_reference_config(self, tmp_path):
        """G7 Step 4 — path-reference link configs are now LOADED and
        REWRITTEN to nested-inline form when config_version >= 2.
        The external YAML is read; auto_transfer is injected if absent;
        the entry's 'config' value flips from a string to a dict."""
        # Create a real external link YAML in tmp_path
        ext = tmp_path / "some_link.yml"
        ext.write_text("source: a.x\ntarget: b.x\n")

        link_dict = {
            "class": "nanobrain.core.link.DirectLink",
            "config": str(ext),
        }
        cfg = self._build(
            name="test", config_version=2,
            links={"l": link_dict},
        )
        # 'config' was rewritten from a string to the loaded dict
        assert isinstance(cfg.links["l"]["config"], dict)
        assert cfg.links["l"]["config"]["auto_transfer"] is True
        assert cfg.links["l"]["config"]["source"] == "a.x"
        assert cfg.links["l"]["config"]["target"] == "b.x"

    def test_v1_still_skips_path_reference(self, tmp_path):
        """v1 must NOT load path-reference configs (Step 4 is v2-only)."""
        ext = tmp_path / "some_link.yml"
        ext.write_text("source: a.x\ntarget: b.x\n")

        link_dict = {
            "class": "nanobrain.core.link.DirectLink",
            "config": str(ext),
        }
        cfg = self._build(name="test", config_version=1,
                          links={"l": link_dict})
        # v1: still a string; not loaded.
        assert cfg.links["l"]["config"] == str(ext)

    def test_v2_path_reference_preserves_explicit_value(self, tmp_path):
        """An external YAML that already declares auto_transfer: false
        must NOT be overridden by v2 default injection — author intent
        wins (the same setdefault semantics as the inline path)."""
        ext = tmp_path / "some_link.yml"
        ext.write_text("source: a.x\ntarget: b.x\nauto_transfer: false\n")

        link_dict = {
            "class": "nanobrain.core.link.DirectLink",
            "config": str(ext),
        }
        cfg = self._build(name="test", config_version=2,
                          links={"l": link_dict})
        assert cfg.links["l"]["config"]["auto_transfer"] is False

    def test_v2_path_reference_missing_file_fails_fast(self):
        link_dict = {
            "class": "nanobrain.core.link.DirectLink",
            "config": "/nonexistent/path/ghost.yml",
        }
        with pytest.raises(Exception) as exc_info:
            self._build(name="test", config_version=2,
                        links={"l": link_dict})
        assert "FAIL-FAST" in str(exc_info.value)

    def test_v2_handles_multiple_links_independently(self):
        """Each link in the dict is processed independently; mixed-shape
        links coexist."""
        cfg = self._build(name="test", config_version=2, links={
            "flat": {
                "class": "nanobrain.core.link.DirectLink",
                "source": "a.x", "target": "b.x",
            },
            "nested": {
                "class": "nanobrain.core.link.TransformLink",
                "config": {"source": "b.x", "target": "c.x",
                           "transform_function": "x.y"},
            },
            "explicit_false": {
                "class": "nanobrain.core.link.ConditionalLink",
                "source": "c.x", "target": "d.x",
                "condition": "true_only",
                "auto_transfer": False,
            },
            "academy": {
                "class": "nanobrain.academy_integration.academy_link.AcademyLink",
                "source": "d.x", "target": "e.x",
            },
        })
        assert cfg.links["flat"]["auto_transfer"] is True
        assert cfg.links["nested"]["config"]["auto_transfer"] is True
        assert cfg.links["explicit_false"]["auto_transfer"] is False
        assert "auto_transfer" not in cfg.links["academy"]  # not whitelisted

    def test_v2_with_empty_links_no_error(self):
        cfg = self._build(name="test", config_version=2, links={})
        assert cfg.links == {}

    def test_v2_with_resolved_linkbase_in_dict_skipped(self):
        """If 'links' contains an already-resolved object (programmatic),
        we skip it gracefully."""
        class FakeResolvedLink:
            pass
        cfg = self._build(name="test", config_version=2, links={
            "resolved": FakeResolvedLink(),
        })
        assert isinstance(cfg.links["resolved"], FakeResolvedLink)


class TestV2WarningSuppressed:
    """When config_version >= 2, the deprecation WARNING is suppressed
    because v2 has actively flipped the default; no recommendation needed."""

    def test_v2_emits_no_warning_for_omission(self, caplog):
        # An omitted-auto_transfer DirectLink under v2 must NOT warn.
        links = {
            "l": {
                "class": "nanobrain.core.link.DirectLink",
                "source": "a.x", "target": "b.x",
            }
        }
        with caplog.at_level(logging.WARNING, logger="nanobrain.core.workflow"):
            _warn_on_implicit_auto_transfer("test_wf", links, config_version=2)
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 0
