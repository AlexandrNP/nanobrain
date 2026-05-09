"""Tests for G14 — PromptTemplate primitive (extended).

Per ``apecx-mcp-integration/docs/nanobrain_capability_gaps.md G14`` and
``apecx-mcp-integration/docs/llm_prompt_contracts.md §3``: the G14
contract on top of the existing PromptTemplate primitive.

These tests cover:
1. Backward compatibility — legacy PromptTemplate (no template_id) still works
2. template_id grammar validation
3. compute_template_content_hash determinism + canonical ordering
4. Auto-resolution of content_hash on load
5. is_g14_compliant flag
6. PromptHole shape validation
7. render() — system_prompt + user_template substitution
8. render() FAIL-FAST on missing required holes
"""

from __future__ import annotations

import pytest

from nanobrain.core.prompt_template_manager import (
    PromptHole,
    PromptTemplate,
    compute_template_content_hash,
)


# ---------------------------------------------------------------------------
# 1. Backward compatibility — legacy PromptTemplate
# ---------------------------------------------------------------------------

class TestLegacyV1Compat:

    def test_legacy_template_loads_unchanged(self):
        pt = PromptTemplate(
            template="Hello $name",
            description="legacy",
            required_params=["name"],
        )
        assert pt.template == "Hello $name"
        assert pt.template_id is None
        assert pt.is_g14_compliant is False
        assert pt.content_hash is None  # auto-hash skipped for legacy

    def test_legacy_template_render_via_manager_path_unaffected(self):
        """The class itself doesn't change behavior for legacy users —
        PromptTemplateManager.get_prompt continues to operate on the
        `template` field. We don't re-test the manager here; this test
        just doc-anchors the behavior."""
        pt = PromptTemplate(template="$x")
        assert pt.template == "$x"


# ---------------------------------------------------------------------------
# 2. template_id grammar
# ---------------------------------------------------------------------------

class TestTemplateIdGrammar:

    def test_canonical_id_accepted(self):
        pt = PromptTemplate(template_id="phase0_planning.default@1.4.0")
        assert pt.template_id == "phase0_planning.default@1.4.0"
        assert pt.template_family == "phase0_planning"
        assert pt.template_semver == "1.4.0"

    def test_prerelease_semver_accepted(self):
        pt = PromptTemplate(template_id="repair.default@2.0.1-rc.1")
        assert pt.template_semver == "2.0.1-rc.1"

    def test_complex_prerelease_accepted(self):
        pt = PromptTemplate(template_id="x.y@1.0.0-alpha.beta.10")
        assert pt.template_semver == "1.0.0-alpha.beta.10"

    def test_uppercase_family_rejected(self):
        with pytest.raises(Exception) as exc_info:
            PromptTemplate(template_id="Phase0.default@1.0.0")
        assert "FAIL-FAST" in str(exc_info.value)

    def test_missing_semver_rejected(self):
        with pytest.raises(Exception) as exc_info:
            PromptTemplate(template_id="phase0.default")
        assert "FAIL-FAST" in str(exc_info.value)

    def test_non_semver_version_rejected(self):
        with pytest.raises(Exception):
            PromptTemplate(template_id="phase0.default@v1")

    def test_missing_family_rejected(self):
        with pytest.raises(Exception):
            PromptTemplate(template_id=".default@1.0.0")

    def test_legacy_id_none_passes(self):
        pt = PromptTemplate(template="x")
        assert pt.template_id is None
        assert pt.template_family is None
        assert pt.template_semver is None


# ---------------------------------------------------------------------------
# 3. compute_template_content_hash — determinism + canonical ordering
# ---------------------------------------------------------------------------

class TestContentHashDeterminism:

    def test_same_inputs_same_hash(self):
        h1 = compute_template_content_hash(
            system_prompt="s", user_template="u",
            holes={"a": {"type": "string"}}, output_schema_ref={"class": "x"},
        )
        h2 = compute_template_content_hash(
            system_prompt="s", user_template="u",
            holes={"a": {"type": "string"}}, output_schema_ref={"class": "x"},
        )
        assert h1 == h2
        # SHA-256 hex = 64 chars.
        assert len(h1) == 64

    def test_hole_key_order_does_not_affect_hash(self):
        h1 = compute_template_content_hash(
            holes={"a": {"type": "string"}, "b": {"type": "integer"}},
        )
        h2 = compute_template_content_hash(
            holes={"b": {"type": "integer"}, "a": {"type": "string"}},
        )
        assert h1 == h2

    def test_different_system_prompt_different_hash(self):
        h1 = compute_template_content_hash(system_prompt="a")
        h2 = compute_template_content_hash(system_prompt="b")
        assert h1 != h2

    def test_different_user_template_different_hash(self):
        h1 = compute_template_content_hash(user_template="x")
        h2 = compute_template_content_hash(user_template="y")
        assert h1 != h2

    def test_different_holes_different_hash(self):
        h1 = compute_template_content_hash(holes={"a": {"type": "string"}})
        h2 = compute_template_content_hash(holes={"a": {"type": "integer"}})
        assert h1 != h2

    def test_different_output_schema_different_hash(self):
        h1 = compute_template_content_hash(output_schema_ref={"class": "x"})
        h2 = compute_template_content_hash(output_schema_ref={"class": "y"})
        assert h1 != h2

    def test_empty_inputs_have_stable_hash(self):
        h = compute_template_content_hash()
        assert len(h) == 64


# ---------------------------------------------------------------------------
# 4. Auto-resolution of content_hash on load
# ---------------------------------------------------------------------------

class TestAutoHashResolution:

    def test_hash_auto_computed_when_sentinel(self):
        pt = PromptTemplate(
            template_id="x.y@1.0.0",
            system_prompt="hello",
            user_template="$q",
            content_hash="<computed-at-load>",
        )
        assert pt.content_hash != "<computed-at-load>"
        assert len(pt.content_hash) == 64

    def test_hash_auto_computed_when_empty(self):
        pt = PromptTemplate(
            template_id="x.y@1.0.0",
            system_prompt="hello",
        )
        assert pt.content_hash is not None
        assert len(pt.content_hash) == 64

    def test_explicit_hash_preserved(self):
        """User-provided hash is preserved (does not auto-recompute).
        Useful for verification: the user pins an expected hash and the
        loader does not silently overwrite it. Mismatch detection is a
        separate concern — see the registry-side gate."""
        pt = PromptTemplate(
            template_id="x.y@1.0.0",
            system_prompt="hello",
            content_hash="abc123",  # not the canonical form, but preserved
        )
        assert pt.content_hash == "abc123"

    def test_legacy_template_no_hash_computation(self):
        """Templates without template_id are legacy v1 and skip the
        G14 hash-computation path entirely."""
        pt = PromptTemplate(template="legacy", content_hash=None)
        assert pt.content_hash is None


# ---------------------------------------------------------------------------
# 5. is_g14_compliant
# ---------------------------------------------------------------------------

class TestG14ComplianceFlag:

    def test_full_g14_compliant(self):
        pt = PromptTemplate(
            template_id="x.y@1.0.0",
            system_prompt="s",
            user_template="u",
        )
        assert pt.is_g14_compliant is True

    def test_id_without_body_not_compliant(self):
        pt = PromptTemplate(template_id="x.y@1.0.0")
        assert pt.is_g14_compliant is False

    def test_body_without_id_not_compliant(self):
        pt = PromptTemplate(system_prompt="s")
        assert pt.is_g14_compliant is False

    def test_legacy_not_compliant(self):
        pt = PromptTemplate(template="legacy")
        assert pt.is_g14_compliant is False


# ---------------------------------------------------------------------------
# 6. PromptHole shape
# ---------------------------------------------------------------------------

class TestPromptHole:

    def test_minimal_hole(self):
        h = PromptHole()
        assert h.type == "string"
        assert h.required is True
        assert h.default is None

    def test_hole_with_all_fields(self):
        h = PromptHole(
            type="integer",
            required=False,
            default=42,
            description="A number",
        )
        assert h.type == "integer"
        assert h.required is False
        assert h.default == 42

    def test_unknown_type_rejected(self):
        with pytest.raises(Exception):
            PromptHole(type="ghost_type")

    def test_extra_field_rejected(self):
        with pytest.raises(Exception):
            PromptHole(type="string", ghost="bad")


# ---------------------------------------------------------------------------
# 7-8. render() — substitution + FAIL-FAST
# ---------------------------------------------------------------------------

class TestRender:

    def test_basic_substitution(self):
        pt = PromptTemplate(
            template_id="x.y@1.0.0",
            system_prompt="You are a helper.",
            user_template="Find $entity in $database",
            holes={
                "entity": PromptHole(type="string", required=True),
                "database": PromptHole(type="string", required=True),
            },
        )
        out = pt.render({"entity": "tomato", "database": "VegDB"})
        assert out["system"] == "You are a helper."
        assert out["user"] == "Find tomato in VegDB"

    def test_only_system_prompt(self):
        pt = PromptTemplate(
            template_id="x.y@1.0.0",
            system_prompt="Just a system prompt.",
        )
        out = pt.render()
        assert out["system"] == "Just a system prompt."
        assert out["user"] == ""

    def test_only_user_template(self):
        pt = PromptTemplate(
            template_id="x.y@1.0.0",
            user_template="$q",
            holes={"q": PromptHole(type="string", required=True)},
        )
        out = pt.render({"q": "hello"})
        assert out["system"] == ""
        assert out["user"] == "hello"

    def test_missing_required_hole_fails_fast(self):
        pt = PromptTemplate(
            template_id="x.y@1.0.0",
            user_template="$x",
            holes={"x": PromptHole(type="string", required=True)},
        )
        with pytest.raises(ValueError) as exc_info:
            pt.render()
        assert "FAIL-FAST" in str(exc_info.value)
        assert "missing required hole" in str(exc_info.value)

    def test_optional_hole_with_default(self):
        pt = PromptTemplate(
            template_id="x.y@1.0.0",
            user_template="Limit: $limit",
            holes={
                "limit": PromptHole(
                    type="integer", required=False, default=10),
            },
        )
        out = pt.render()
        assert out["user"] == "Limit: 10"

    def test_optional_hole_no_default_substitutes_empty(self):
        pt = PromptTemplate(
            template_id="x.y@1.0.0",
            user_template="Filter: '$filter'",
            holes={"filter": PromptHole(required=False)},
        )
        out = pt.render()
        assert out["user"] == "Filter: ''"

    def test_legacy_template_render_raises(self):
        """Legacy templates without template_id cannot use render() —
        they must use the PromptTemplateManager.get_prompt path."""
        pt = PromptTemplate(template="legacy $x")
        with pytest.raises(ValueError) as exc_info:
            pt.render({"x": "v"})
        assert "G14" in str(exc_info.value)
