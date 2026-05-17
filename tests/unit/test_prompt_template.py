"""Unit tests for G14 PromptTemplate primitive.

Pins the surface required by the G25 PromptRegressionHarness:
  * template_id property
  * content_hash property
  * regression_fixtures property
  * render(params) -> {"system": ..., "user": ...} method

Plus the file-loading + substitution + validation contracts.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from nanobrain.core.component_base import ComponentConfigurationError
from nanobrain.core.prompt_template import PromptTemplate


def _build_template(cfg: dict) -> PromptTemplate:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
        yaml.safe_dump(cfg, f)
        path = f.name
    return PromptTemplate.from_config(path)


class TestInlineTemplates:
    def test_inline_system_only(self):
        t = _build_template({
            "template_id": "test1",
            "system_prompt": "You are a code writer.",
        })
        rendered = t.render({})
        assert rendered["system"] == "You are a code writer."
        assert rendered["user"] == ""

    def test_inline_user_only(self):
        t = _build_template({
            "template_id": "test2",
            "user_template": "Write a function named {name}.",
        })
        rendered = t.render({"name": "fib"})
        assert rendered["system"] == ""
        assert rendered["user"] == "Write a function named fib."

    def test_inline_both_with_substitution(self):
        t = _build_template({
            "template_id": "test3",
            "system_prompt": "You are an expert in {language}.",
            "user_template": "Implement {algorithm}.",
        })
        rendered = t.render({"language": "Python", "algorithm": "quicksort"})
        assert rendered["system"] == "You are an expert in Python."
        assert rendered["user"] == "Implement quicksort."


class TestFileTemplates:
    def test_file_loading_works(self, tmp_path):
        prompt_file = tmp_path / "system.md"
        prompt_file.write_text("File-loaded system prompt.")
        t = _build_template({
            "template_id": "file1",
            "system_prompt_file": str(prompt_file),
        })
        rendered = t.render({})
        assert rendered["system"] == "File-loaded system prompt."

    def test_relative_file_resolves_against_yaml_dir(self, tmp_path):
        prompt_file = tmp_path / "prompt.md"
        prompt_file.write_text("Relative load works.")
        # Build config IN tmp_path so the relative path resolves correctly.
        config_file = tmp_path / "config.yml"
        with config_file.open("w") as f:
            yaml.safe_dump({
                "template_id": "rel1",
                "system_prompt_file": "prompt.md",  # relative to config_file's dir
            }, f)
        t = PromptTemplate.from_config(str(config_file))
        assert t.render({})["system"] == "Relative load works."

    def test_missing_file_raises_config_error(self):
        with pytest.raises(ComponentConfigurationError, match="failed to read"):
            _build_template({
                "template_id": "missing_file",
                "system_prompt_file": "/does/not/exist.md",
            })

    def test_empty_file_raises_config_error(self, tmp_path):
        prompt_file = tmp_path / "empty.md"
        prompt_file.write_text("   \n  ")  # whitespace only
        with pytest.raises(ComponentConfigurationError, match="empty"):
            _build_template({
                "template_id": "empty1",
                "system_prompt_file": str(prompt_file),
            })


class TestValidation:
    def test_both_inline_and_file_for_system_rejected(self, tmp_path):
        prompt_file = tmp_path / "p.md"
        prompt_file.write_text("hello")
        with pytest.raises(Exception):  # ComponentConfigurationError or pydantic ValidationError
            _build_template({
                "template_id": "conflict1",
                "system_prompt": "inline",
                "system_prompt_file": str(prompt_file),
            })

    def test_no_template_at_all_rejected(self):
        with pytest.raises(Exception):
            _build_template({"template_id": "empty"})

    def test_extra_unknown_field_rejected(self):
        with pytest.raises(Exception):  # extra='forbid'
            _build_template({
                "template_id": "with_typo",
                "system_prompt": "x",
                "typo_field": "oops",
            })


class TestRendering:
    def test_missing_variable_raises_keyerror_with_name(self):
        t = _build_template({
            "template_id": "needs_var",
            "user_template": "Write {fn} for {language}.",
        })
        with pytest.raises(KeyError, match="fn"):
            t.render({"language": "Python"})  # missing fn

    def test_extra_params_silently_ignored(self):
        """Pythonic str.format ignores extra kwargs — we preserve that."""
        t = _build_template({
            "template_id": "no_vars",
            "system_prompt": "Static.",
        })
        rendered = t.render({"unused": "x", "also_unused": 42})
        assert rendered["system"] == "Static."

    def test_render_with_non_dict_raises_typeerror(self):
        t = _build_template({"template_id": "x", "system_prompt": "y"})
        with pytest.raises(TypeError, match="must be a dict"):
            t.render("not a dict")  # type: ignore[arg-type]


class TestContentHash:
    def test_hash_is_deterministic_for_same_content(self):
        t1 = _build_template({
            "template_id": "a", "system_prompt": "fixed"
        })
        t2 = _build_template({
            "template_id": "b", "system_prompt": "fixed"  # different ID, same content
        })
        # Same content → same hash (content hash is over content, not ID).
        assert t1.content_hash == t2.content_hash

    def test_hash_changes_when_content_changes(self):
        t1 = _build_template({"template_id": "x", "system_prompt": "v1"})
        t2 = _build_template({"template_id": "x", "system_prompt": "v2"})
        assert t1.content_hash != t2.content_hash

    def test_hash_format_is_sha256_prefix(self):
        t = _build_template({"template_id": "x", "system_prompt": "y"})
        assert t.content_hash.startswith("sha256:")
        assert len(t.content_hash) == len("sha256:") + 64  # SHA-256 = 64 hex chars


class TestG25HarnessCompatibility:
    """Pin the surface the G25 PromptRegressionHarness reads."""

    def test_has_template_id_attribute(self):
        t = _build_template({"template_id": "for_g25", "system_prompt": "x"})
        assert t.template_id == "for_g25"

    def test_has_content_hash_attribute(self):
        t = _build_template({"template_id": "x", "system_prompt": "y"})
        assert isinstance(t.content_hash, str)

    def test_has_render_method(self):
        t = _build_template({"template_id": "x", "user_template": "hi"})
        assert callable(t.render)
        result = t.render({})
        assert isinstance(result, dict)
        assert "system" in result
        assert "user" in result

    def test_regression_fixtures_default_empty_list(self):
        t = _build_template({"template_id": "x", "system_prompt": "y"})
        assert t.regression_fixtures == []

    def test_regression_fixtures_pass_through(self):
        fixtures = [
            {"input": {"var": "v1"}, "contract": {"contains": "v1"}},
            {"input": {"var": "v2"}, "contract": {"contains": "v2"}},
        ]
        t = _build_template({
            "template_id": "x",
            "user_template": "Has {var}",
            "regression_fixtures": fixtures,
        })
        assert t.regression_fixtures == fixtures
        # Returns a copy — mutations don't leak.
        t.regression_fixtures.append({"hack": True})
        assert len(t.regression_fixtures) == 2


class TestRawAccessors:
    """Callers that don't need substitution can read raw template
    strings via system_prompt / user_template properties."""

    def test_raw_system_prompt(self):
        t = _build_template({"template_id": "x", "system_prompt": "raw {var}"})
        # Raw == unsubstituted.
        assert t.system_prompt == "raw {var}"

    def test_raw_user_template(self):
        t = _build_template({
            "template_id": "x", "user_template": "raw {var}",
        })
        assert t.user_template == "raw {var}"
