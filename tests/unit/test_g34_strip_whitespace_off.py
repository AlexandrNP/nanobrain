"""G34 — pin that ConfigBase preserves whitespace in string fields.

eval_03 Round 4 G34: pre-G34 (commit unknown) the framework's
ConfigBase had ``str_strip_whitespace=True``, which silently
dropped meaningful-whitespace string values like ``delimiter: "\\t"``
from a CSV-reader step YAML — the field arrived as an empty string.
The integration's apecx-mcp-integration/composition/steps/file_readers.py
worked around this with a named-format enum (``csv`` / ``tsv``) that
mapped to the real delimiter internally.

The fix landed earlier at ``config_base.py:684`` (str_strip_whitespace=False).
This test exists as a REGRESSION GUARD: a refactor that re-enables
strip_whitespace would re-introduce the silent-failure shape, so we
pin the contract here.

Pinned contracts:
  1. ConfigBase.model_config has str_strip_whitespace=False
  2. Tab characters survive a YAML round-trip through a ConfigBase
     subclass (StepConfig)
  3. Trailing whitespace survives (operators may have semantically
     meaningful trailing spaces — e.g., a separator like ``", "``)
  4. Leading whitespace survives
  5. Multiple-character whitespace survives (CR LF, mixed tabs)

Source: ``apecx-mcp-integration/eval_03_nanobrain_gap_inventory.md``
Round 4 G34;
``apecx-mcp-integration/docs/development_roadmap.md`` 8.7 Tier 4.
"""
from __future__ import annotations

import tempfile

import pytest
import yaml

from nanobrain.core.config.config_base import ConfigBase
from nanobrain.core.step import StepConfig


def test_config_base_str_strip_whitespace_is_off():
    """The base ConfigBase model_config MUST set str_strip_whitespace=False.
    A refactor that flips it back to True is a silent-failure regression."""
    assert ConfigBase.model_config.get("str_strip_whitespace") is False, (
        "ConfigBase.model_config must have str_strip_whitespace=False; "
        "True silently drops meaningful-whitespace string values"
    )


@pytest.mark.parametrize(
    "raw_value,description",
    [
        ("\t", "single tab"),
        ("trailing  ", "trailing two-space"),
        ("  leading", "leading two-space"),
        ("\r\n", "CR LF"),
        ("\t\t", "double tab"),
        (" mid space ", "leading + trailing space"),
        (",", "regular comma — sanity check"),
    ],
)
def test_step_config_subclass_preserves_whitespace(raw_value, description):
    """StepConfig subclasses inherit ConfigBase's model_config, so a
    field declared as ``delimiter: str`` MUST receive the literal
    YAML value, including whitespace."""

    class _DelimitedConfig(StepConfig):
        delimiter: str = ","

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yml", delete=False
    ) as f:
        yaml.safe_dump({"name": "x", "delimiter": raw_value}, f)
        path = f.name

    cfg = _DelimitedConfig.from_config(path)
    assert cfg.delimiter == raw_value, (
        f"whitespace stripped for {description!r}: "
        f"expected {raw_value!r}, got {cfg.delimiter!r}"
    )


def test_direct_dict_construction_preserves_whitespace():
    """Direct dict construction (via _allow_direct_instantiation
    backdoor) also preserves whitespace — Pydantic field-level
    behavior is the same regardless of load path."""

    class _DelimitedConfig(StepConfig):
        delimiter: str = ","

    _DelimitedConfig._allow_direct_instantiation = True
    try:
        cfg = _DelimitedConfig(name="x", delimiter="\t")
    finally:
        _DelimitedConfig._allow_direct_instantiation = False

    assert cfg.delimiter == "\t"
