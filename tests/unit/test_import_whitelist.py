"""Tests for G20 — class-path import whitelist.

Per ``apecx-mcp-integration/docs/nanobrain_capability_gaps.md G20`` and
``apecx-mcp-integration/docs/security_threat_model.md §5.8 T-CL-1``.

Tests cover:
1. Default behavior (no whitelist set = allow all)
2. set_class_import_whitelist + get_class_import_whitelist roundtrip
3. with_class_import_whitelist context manager scoping
4. check_class_import_allowed FAIL-FAST shape
5. Per-call extra_whitelist additivity
6. Empty whitelist denies everything (when set to [])
7. Thread isolation considerations (whitelist is process-global by design)
"""

from __future__ import annotations

import pytest

from nanobrain.core.import_whitelist import (
    check_class_import_allowed,
    get_class_import_whitelist,
    set_class_import_whitelist,
    with_class_import_whitelist,
)


# Auto-cleanup: every test starts with whitelist=None and ends with
# whitelist=None, regardless of test outcome.
@pytest.fixture(autouse=True)
def _whitelist_cleanup():
    set_class_import_whitelist(None)
    yield
    set_class_import_whitelist(None)


# ---------------------------------------------------------------------------
# 1. Default behavior — no whitelist = allow all
# ---------------------------------------------------------------------------

class TestDefaultAllowAll:

    def test_no_whitelist_allows_anything(self):
        check_class_import_allowed("absolutely.evil.Backdoor")
        check_class_import_allowed("__main__.Whatever")
        check_class_import_allowed("nanobrain.core.step.BaseStep")

    def test_get_returns_none_initially(self):
        assert get_class_import_whitelist() is None


# ---------------------------------------------------------------------------
# 2. set + get roundtrip
# ---------------------------------------------------------------------------

class TestSetAndGet:

    def test_set_list_roundtrip(self):
        set_class_import_whitelist(["a.", "b."])
        assert get_class_import_whitelist() == ["a.", "b."]

    def test_set_to_none_disables(self):
        set_class_import_whitelist(["a."])
        set_class_import_whitelist(None)
        assert get_class_import_whitelist() is None

    def test_set_to_empty_list_denies_all(self):
        set_class_import_whitelist([])
        assert get_class_import_whitelist() == []

    def test_set_returns_copy_not_alias(self):
        wl = ["a."]
        set_class_import_whitelist(wl)
        # Mutating the original list MUST NOT affect the whitelist.
        wl.append("b.")
        assert get_class_import_whitelist() == ["a."]

    def test_get_returns_copy_not_alias(self):
        set_class_import_whitelist(["a."])
        wl = get_class_import_whitelist()
        wl.append("b.")
        # Mutating the returned list MUST NOT affect the whitelist.
        assert get_class_import_whitelist() == ["a."]


# ---------------------------------------------------------------------------
# 3. Context manager
# ---------------------------------------------------------------------------

class TestContextManager:

    def test_scoped_override_restored_on_exit(self):
        set_class_import_whitelist(["original."])
        with with_class_import_whitelist(["scoped."]):
            assert get_class_import_whitelist() == ["scoped."]
        assert get_class_import_whitelist() == ["original."]

    def test_scoped_override_restored_on_exception(self):
        set_class_import_whitelist(["original."])
        with pytest.raises(ValueError):
            with with_class_import_whitelist(["scoped."]):
                raise ValueError("intentional")
        assert get_class_import_whitelist() == ["original."]

    def test_nested_context_managers(self):
        set_class_import_whitelist(["outer_initial."])
        with with_class_import_whitelist(["outer."]):
            assert get_class_import_whitelist() == ["outer."]
            with with_class_import_whitelist(["inner."]):
                assert get_class_import_whitelist() == ["inner."]
            assert get_class_import_whitelist() == ["outer."]
        assert get_class_import_whitelist() == ["outer_initial."]

    def test_disabling_via_context(self):
        set_class_import_whitelist(["a."])
        with with_class_import_whitelist(None):
            # Disabled — anything allowed.
            check_class_import_allowed("evil.Backdoor")
        # Restored.
        with pytest.raises(ImportError):
            check_class_import_allowed("evil.Backdoor")


# ---------------------------------------------------------------------------
# 4. check_class_import_allowed
# ---------------------------------------------------------------------------

class TestCheckImportAllowed:

    def test_allowed_prefix_passes(self):
        set_class_import_whitelist(["nanobrain.", "apecx_integration."])
        check_class_import_allowed("nanobrain.core.step.BaseStep")
        check_class_import_allowed("apecx_integration.composition.composer")

    def test_disallowed_prefix_fails_fast(self):
        set_class_import_whitelist(["nanobrain."])
        with pytest.raises(ImportError) as exc_info:
            check_class_import_allowed("attacker.evil.Backdoor")
        msg = str(exc_info.value)
        assert "FAIL-FAST" in msg
        assert "attacker.evil.Backdoor" in msg
        assert "nanobrain." in msg

    def test_error_message_lists_allowed_prefixes(self):
        set_class_import_whitelist(["a.", "b.", "c."])
        with pytest.raises(ImportError) as exc_info:
            check_class_import_allowed("z.Z")
        msg = str(exc_info.value)
        assert "a." in msg
        assert "b." in msg
        assert "c." in msg

    def test_empty_whitelist_denies_everything(self):
        set_class_import_whitelist([])
        with pytest.raises(ImportError):
            check_class_import_allowed("nanobrain.core.step.BaseStep")
        with pytest.raises(ImportError):
            check_class_import_allowed("anything.Else")

    def test_dotless_prefix_can_match_too_much(self):
        """Operator-policy reminder: prefix without trailing dot can
        match too much. The framework doesn't enforce trailing dot —
        that's a deployment audit concern. We document the behavior."""
        set_class_import_whitelist(["nano"])  # no trailing dot
        check_class_import_allowed("nanobrain.core")  # matches
        check_class_import_allowed("nano_evil.attack")  # ALSO matches
        # Test asserts the (potentially-surprising) behavior so the
        # operator who reads this knows to use trailing dots.


# ---------------------------------------------------------------------------
# 5. Per-call extra_whitelist
# ---------------------------------------------------------------------------

class TestPerCallExtra:

    def test_extra_extends_process_whitelist(self):
        set_class_import_whitelist(["nanobrain."])
        # Extra adds plugin.* for this call only.
        check_class_import_allowed(
            "plugin.MyTool",
            extra_whitelist=["plugin."],
        )
        # Without extra, plugin.* is rejected:
        with pytest.raises(ImportError):
            check_class_import_allowed("plugin.MyTool")

    def test_extra_only_when_process_unset(self):
        """When process whitelist is None but extra is set, the per-call
        extra DOES enforce checking (this opts in to whitelist mode for
        this call only)."""
        # Default — no process whitelist:
        check_class_import_allowed("evil.Backdoor")  # legacy allow
        # With per-call extra, behavior switches to enforce:
        check_class_import_allowed("ok.Foo", extra_whitelist=["ok."])
        with pytest.raises(ImportError):
            check_class_import_allowed("evil.Backdoor", extra_whitelist=["ok."])

    def test_empty_extra_iterable_is_legacy_allow(self):
        """An empty/None extra with no process whitelist = allow all."""
        check_class_import_allowed("evil.Backdoor", extra_whitelist=[])
        check_class_import_allowed("evil.Backdoor", extra_whitelist=None)
