"""Tests for G9 — Skeleton primitive.

Per ``apecx-mcp-integration/docs/nanobrain_capability_gaps.md G9`` and
``apecx-mcp-integration/docs/agent_workflow_authoring.md §4.1``.

Tests cover:
1. Skeleton + SkeletonHole shape validation
2. skeleton_version grammar
3. content_hash auto-resolution + canonical-form invariance
4. find_inline_hole_tokens — parses {{name: type | default=val}} grammar
5. validate_against_schema — cross-checks body vs holes schema
6. SkeletonRegistry — register + lookup by hash/prefix/semver
7. load_from_directory convenience
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from nanobrain.library.orchestration import (
    Skeleton,
    SkeletonHole,
    SkeletonRegistry,
    compute_skeleton_body_hash,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_skel(**overrides) -> Skeleton:
    base = dict(
        skeleton_id="test_skel",
        skeleton_version="v1.0.0",
        description="test",
        body=(
            "class: nanobrain.core.workflow.Workflow\n"
            "config:\n"
            "  name: test\n"
            "  steps:\n"
            "    one:\n"
            "      class: x.y.Z\n"
            "      param: '{{the_param: string}}'\n"
        ),
        holes={"the_param": {"type": "string", "required": True}},
    )
    base.update(overrides)
    return Skeleton.from_config(base)


# ---------------------------------------------------------------------------
# 1. SkeletonHole shape
# ---------------------------------------------------------------------------

class TestSkeletonHoleShape:

    def _build(self, **kwargs):
        SkeletonHole._allow_direct_instantiation = True
        try:
            return SkeletonHole(**kwargs)
        finally:
            SkeletonHole._allow_direct_instantiation = False

    def test_minimal(self):
        h = self._build()
        assert h.type == "string"
        assert h.required is True
        assert h.default is None

    def test_full(self):
        h = self._build(
            type="integer", required=False, default=42, description="num")
        assert h.type == "integer"
        assert h.default == 42

    def test_tool_descriptor_ref_type_accepted(self):
        h = self._build(type="tool_descriptor_ref")
        assert h.type == "tool_descriptor_ref"

    def test_unknown_type_rejected(self):
        with pytest.raises(Exception):
            self._build(type="ghost_type")

    def test_extra_field_rejected(self):
        with pytest.raises(Exception):
            self._build(type="string", ghost="bad")


# ---------------------------------------------------------------------------
# 2. skeleton_version grammar
# ---------------------------------------------------------------------------

class TestSkeletonVersionGrammar:

    def test_12_char_hex_accepted(self):
        sk = _build_skel(skeleton_version="a7c2e4f91b03")
        assert sk.skeleton_version == "a7c2e4f91b03"

    def test_64_char_hex_accepted(self):
        full_hash = "a" * 64
        sk = _build_skel(skeleton_version=full_hash)
        assert sk.skeleton_version == full_hash

    def test_semver_accepted(self):
        sk = _build_skel(skeleton_version="1.2.3-rc.1+build.456")
        assert sk.skeleton_version == "1.2.3-rc.1+build.456"

    def test_empty_rejected(self):
        with pytest.raises(Exception) as exc_info:
            _build_skel(skeleton_version="")
        assert "FAIL-FAST" in str(exc_info.value)

    def test_whitespace_rejected(self):
        with pytest.raises(Exception):
            _build_skel(skeleton_version="   ")
        with pytest.raises(Exception):
            _build_skel(skeleton_version="abc def")


# ---------------------------------------------------------------------------
# 3. content_hash auto-resolution + canonical-form invariance
# ---------------------------------------------------------------------------

class TestContentHashCanonical:

    def test_hash_auto_computed(self):
        sk = _build_skel()
        assert sk.content_hash is not None
        assert len(sk.content_hash) == 64

    def test_explicit_hash_preserved(self):
        sk = _build_skel(content_hash="explicitly_set")
        assert sk.content_hash == "explicitly_set"

    def test_sentinel_triggers_computation(self):
        sk = _build_skel(content_hash="<computed-at-load>")
        assert sk.content_hash != "<computed-at-load>"
        assert len(sk.content_hash) == 64

    def test_canonical_form_invariant_under_yaml_reformatting(self):
        """Two YAML bodies that parse to the same data structure hash
        identically — even if whitespace and key order differ."""
        body1 = "a: 1\nb:\n  c: 2\n"
        body2 = "b:\n  c: 2\na: 1\n"  # keys reordered
        h1 = compute_skeleton_body_hash(body1)
        h2 = compute_skeleton_body_hash(body2)
        assert h1 == h2

    def test_different_body_different_hash(self):
        h1 = compute_skeleton_body_hash("a: 1")
        h2 = compute_skeleton_body_hash("a: 2")
        assert h1 != h2

    def test_malformed_yaml_falls_back_to_raw_hash(self):
        """A body that doesn't parse as YAML still hashes; the hash is
        then sensitive to whitespace (the docstring contract)."""
        bad_yaml = "a: 1\n  bad\n}}}{{}"
        # Should not raise; should return SOME hash.
        h = compute_skeleton_body_hash(bad_yaml)
        assert len(h) == 64

    def test_description_change_does_not_affect_body_hash(self):
        """Hash is over body only; description is metadata."""
        sk1 = _build_skel(description="alpha")
        sk2 = _build_skel(description="beta")
        assert sk1.content_hash == sk2.content_hash


# ---------------------------------------------------------------------------
# 4. find_inline_hole_tokens
# ---------------------------------------------------------------------------

class TestFindInlineHoleTokens:

    def test_simple_token(self):
        sk = _build_skel()
        tokens = sk.find_inline_hole_tokens()
        assert ("the_param", "string", None) in tokens

    def test_token_with_default(self):
        body = "x: '{{my_count: integer | default=10}}'"
        sk = _build_skel(body=body, holes={
            "my_count": {"type": "integer", "required": False, "default": 10}
        })
        tokens = sk.find_inline_hole_tokens()
        assert tokens == [("my_count", "integer", "10")]

    def test_multiple_tokens(self):
        body = "x: '{{a: string}}'\ny: '{{b: integer}}'\n"
        sk = _build_skel(body=body, holes={
            "a": {"type": "string"}, "b": {"type": "integer"}
        })
        tokens = sk.find_inline_hole_tokens()
        names = [t[0] for t in tokens]
        assert "a" in names
        assert "b" in names

    def test_malformed_token_skipped(self):
        """A {{...}} block that doesn't match the strict grammar is
        skipped (lowering pipeline reports it with better context)."""
        body = "x: '{{not a real hole}}'\ny: '{{good: string}}'"
        sk = _build_skel(body=body, holes={"good": {"type": "string"}})
        tokens = sk.find_inline_hole_tokens()
        names = [t[0] for t in tokens]
        assert "good" in names
        assert "not" not in names  # malformed skipped

    def test_no_tokens_in_body(self):
        sk = _build_skel(body="just some text\n", holes={})
        assert sk.find_inline_hole_tokens() == []


# ---------------------------------------------------------------------------
# 5. validate_against_schema
# ---------------------------------------------------------------------------

class TestValidateAgainstSchema:

    def test_clean_skeleton_no_issues(self):
        sk = _build_skel()
        assert sk.validate_against_schema() == []

    def test_undeclared_token_in_body(self):
        body = "x: '{{ghost: string}}'"
        sk = _build_skel(body=body, holes={})
        issues = sk.validate_against_schema()
        assert any("ghost" in i and "not declared" in i for i in issues)

    def test_unused_hole_in_schema(self):
        body = "x: 'no holes here'"
        sk = _build_skel(body=body, holes={
            "unused": {"type": "string"},
        })
        issues = sk.validate_against_schema()
        assert any("unused" in i and "not used" in i for i in issues)

    def test_type_mismatch(self):
        body = "x: '{{the_param: integer}}'"
        sk = _build_skel(body=body, holes={
            "the_param": {"type": "string"},  # schema says string, body says integer
        })
        issues = sk.validate_against_schema()
        assert any("type mismatch" in i for i in issues)

    def test_required_with_inline_default_flagged(self):
        body = "x: '{{the_param: string | default=hello}}'"
        sk = _build_skel(body=body, holes={
            "the_param": {"type": "string", "required": True},
        })
        issues = sk.validate_against_schema()
        assert any("required" in i and "inline default" in i for i in issues)


# ---------------------------------------------------------------------------
# 6. SkeletonRegistry
# ---------------------------------------------------------------------------

class TestSkeletonRegistry:

    def test_register_and_lookup_by_full_hash(self):
        reg = SkeletonRegistry()
        sk = _build_skel()
        reg.register(sk)
        result = reg.lookup("test_skel", sk.content_hash)
        assert result is sk

    def test_register_and_lookup_by_12_char_prefix(self):
        reg = SkeletonRegistry()
        sk = _build_skel()
        reg.register(sk)
        result = reg.lookup("test_skel", sk.content_hash[:12])
        assert result is sk

    def test_register_and_lookup_by_semver_tag(self):
        reg = SkeletonRegistry()
        sk = _build_skel(skeleton_version="1.0.0")
        reg.register(sk)
        result = reg.lookup("test_skel", "1.0.0")
        assert result is sk

    def test_lookup_missing_skeleton_id_raises(self):
        reg = SkeletonRegistry()
        with pytest.raises(KeyError) as exc_info:
            reg.lookup("does_not_exist", "1.0.0")
        assert "FAIL-FAST" in str(exc_info.value)
        assert "not in registry" in str(exc_info.value)

    def test_lookup_missing_version_lists_available(self):
        reg = SkeletonRegistry()
        sk = _build_skel(skeleton_version="1.0.0")
        reg.register(sk)
        with pytest.raises(KeyError) as exc_info:
            reg.lookup("test_skel", "ghost_version")
        assert "available versions" in str(exc_info.value)
        assert "1.0.0" in str(exc_info.value)

    def test_register_idempotent_on_same_content_hash(self):
        """Registering the SAME skeleton twice is a no-op."""
        reg = SkeletonRegistry()
        sk = _build_skel(skeleton_version="1.0.0")
        reg.register(sk)
        reg.register(sk)  # second call — no error
        assert reg.versions_of("test_skel") == ["1.0.0"]

    def test_register_same_version_different_hash_fails_fast(self):
        """Two skeletons claiming the same version_string but with
        different bodies = contract drift = FAIL-FAST."""
        reg = SkeletonRegistry()
        sk1 = _build_skel(skeleton_version="1.0.0", body="a: 1")
        sk2 = _build_skel(skeleton_version="1.0.0", body="a: 2")
        reg.register(sk1)
        with pytest.raises(ValueError) as exc_info:
            reg.register(sk2)
        assert "FAIL-FAST" in str(exc_info.value)
        assert "contract drift" in str(exc_info.value)

    def test_list_skeletons_and_versions(self):
        reg = SkeletonRegistry()
        sk_a_v1 = _build_skel(skeleton_id="a", skeleton_version="1.0.0", body="x: 1")
        sk_a_v2 = _build_skel(skeleton_id="a", skeleton_version="2.0.0", body="x: 2")
        sk_b = _build_skel(skeleton_id="b", skeleton_version="1.0.0", body="y: 1")
        reg.register(sk_a_v1)
        reg.register(sk_a_v2)
        reg.register(sk_b)
        assert sorted(reg.list_skeletons()) == ["a", "b"]
        assert sorted(reg.versions_of("a")) == ["1.0.0", "2.0.0"]


# ---------------------------------------------------------------------------
# 7. load_from_directory
# ---------------------------------------------------------------------------

class TestLoadFromDirectory:

    def test_load_with_both_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            skel_dir = Path(tmp) / "my_skel"
            skel_dir.mkdir()
            (skel_dir / "skeleton.yml").write_text(
                "config:\n  param: '{{my_hole: string}}'\n"
            )
            (skel_dir / "skeleton.schema.json").write_text(json.dumps({
                "holes": {"my_hole": {"type": "string", "required": True}},
                "metadata": {"description": "loaded from disk"},
            }))
            sk = Skeleton.load_from_directory(skel_dir)
            assert sk.skeleton_id == "my_skel"
            assert sk.description == "loaded from disk"
            assert "my_hole" in sk.holes
            assert sk.skeleton_version == sk.content_hash[:12]

    def test_load_body_only(self):
        """No skeleton.schema.json is OK — the loader uses an empty
        holes dict. validate_against_schema() will then flag every
        inline token as undeclared, but loading itself succeeds."""
        with tempfile.TemporaryDirectory() as tmp:
            skel_dir = Path(tmp) / "body_only"
            skel_dir.mkdir()
            (skel_dir / "skeleton.yml").write_text("config: {x: 1}\n")
            sk = Skeleton.load_from_directory(skel_dir)
            assert sk.skeleton_id == "body_only"
            assert sk.holes == {}

    def test_load_missing_body_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            skel_dir = Path(tmp) / "empty"
            skel_dir.mkdir()
            with pytest.raises(FileNotFoundError) as exc_info:
                Skeleton.load_from_directory(skel_dir)
            assert "FAIL-FAST" in str(exc_info.value)
            assert "skeleton body not found" in str(exc_info.value)

    def test_explicit_id_and_version_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            skel_dir = Path(tmp) / "default_name"
            skel_dir.mkdir()
            (skel_dir / "skeleton.yml").write_text("config: {x: 1}\n")
            sk = Skeleton.load_from_directory(
                skel_dir, skeleton_id="custom_id", skeleton_version="2.0.0",
            )
            assert sk.skeleton_id == "custom_id"
            assert sk.skeleton_version == "2.0.0"
