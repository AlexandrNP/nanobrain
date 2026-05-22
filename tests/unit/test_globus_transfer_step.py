"""Unit tests for GlobusTransferStep (G28).

Unconditional — no network, no real Globus auth. FAIL-LOUD paths are
exercised by monkeypatching the ``globus_sdk`` import and feeding malformed
input payloads; that is testing the error contract, not mocking a real
transfer.

Covers:
  * ``GlobusTransferStepConfig`` validates; ``extra='forbid'`` rejects
    typos; required fields enforced; the ``class:`` key is stripped.
  * ``GlobusTransferStep`` loads via ``from_config``; ``process()`` not
    ``execute()`` is the entry point.
  * Trigger-envelope self-unwrap.
  * FAIL-LOUD: malformed / empty ``items`` payload.
  * FAIL-LOUD: ``globus_sdk`` not importable.
"""

from __future__ import annotations

import asyncio
import builtins
import tempfile
from pathlib import Path

import pytest

from nanobrain.core.component_base import ComponentConfigurationError
from nanobrain.library.steps.globus_transfer_step import (
    GlobusTransferStep,
    GlobusTransferStepConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _write_step_yaml(extra: dict | None = None, *, with_class: bool = False) -> str:
    lines = []
    if with_class:
        lines.append(
            "class: nanobrain.library.steps.globus_transfer_step."
            "GlobusTransferStep"
        )
    lines += [
        "name: stage_inputs",
        "description: 'test transfer step'",
        "source_endpoint_id: src-endpoint-uuid",
        "dest_endpoint_id: dst-endpoint-uuid",
    ]
    if extra:
        for k, v in extra.items():
            lines.append(f"{k}: {v!r}")
    f = tempfile.NamedTemporaryFile(
        "w", suffix=".yml", delete=False, encoding="utf-8"
    )
    f.write("\n".join(lines) + "\n")
    f.close()
    return f.name


# ---------------------------------------------------------------------------
# Config validation. GlobusTransferStepConfig is a ConfigBase subclass, so
# it is file-only — direct construction is framework-forbidden. We exercise
# validation via from_config(<yaml path>), the supported path.
# ---------------------------------------------------------------------------
def test_config_validates_minimal():
    cfg = GlobusTransferStepConfig.from_config(_write_step_yaml())
    assert cfg.source_endpoint_id == "src-endpoint-uuid"
    assert cfg.sync_level == "checksum"
    assert cfg.verify_checksum is True
    assert cfg.poll_timeout_seconds == 600.0


def test_config_extra_forbid_rejects_typo():
    path = _write_step_yaml({"sync_levell": "typo"})  # typo on purpose
    with pytest.raises(Exception) as exc:
        GlobusTransferStepConfig.from_config(path)
    assert "sync_levell" in str(exc.value) or "extra" in str(exc.value).lower()


def test_config_strips_class_key():
    """The framework-loader ``class`` key must not trip extra='forbid'."""
    cfg = GlobusTransferStepConfig.from_config(_write_step_yaml(with_class=True))
    assert cfg.name == "stage_inputs"


# ---------------------------------------------------------------------------
# from_config
# ---------------------------------------------------------------------------
def test_step_loads_from_config_path():
    step = GlobusTransferStep.from_config(_write_step_yaml())
    assert isinstance(step, GlobusTransferStep)
    assert step.transfer_config.source_endpoint_id == "src-endpoint-uuid"


def test_step_loads_from_config_with_class_key():
    step = GlobusTransferStep.from_config(_write_step_yaml(with_class=True))
    assert isinstance(step, GlobusTransferStep)


def test_step_direct_construction_forbidden():
    with pytest.raises(RuntimeError) as exc:
        GlobusTransferStep()
    assert "Direct instantiation" in str(exc.value)


def test_step_implements_process_not_execute():
    """The step's business-logic entry point is process(), per the framework."""
    # process is defined on GlobusTransferStep itself; execute is inherited.
    assert "process" in GlobusTransferStep.__dict__
    assert "execute" not in GlobusTransferStep.__dict__


# ---------------------------------------------------------------------------
# Trigger-envelope self-unwrap
# ---------------------------------------------------------------------------
def test_unwrap_trigger_envelope_strips_input_du_key():
    step = GlobusTransferStep.from_config(_write_step_yaml())
    payload = {"items": [{"source_path": "/a", "dest_path": "/b"}]}
    envelope = {"transfer_request": payload}  # input data unit name as key
    assert step._unwrap_trigger_envelope(envelope) == payload


def test_unwrap_trigger_envelope_passes_through_direct_payload():
    step = GlobusTransferStep.from_config(_write_step_yaml())
    payload = {"items": [{"source_path": "/a", "dest_path": "/b"}]}
    # A dict whose lone key IS 'items' is a direct payload — pass through.
    assert step._unwrap_trigger_envelope(payload) == payload


def test_unwrap_trigger_envelope_passes_through_non_dict():
    step = GlobusTransferStep.from_config(_write_step_yaml())
    assert step._unwrap_trigger_envelope(["not", "a", "dict"]) == [
        "not",
        "a",
        "dict",
    ]


# ---------------------------------------------------------------------------
# items coercion FAIL-LOUD
# ---------------------------------------------------------------------------
def test_coerce_items_empty_fails_loud():
    with pytest.raises(ComponentConfigurationError) as exc:
        GlobusTransferStep._coerce_items({"items": []})
    assert "FAIL-FAST" in str(exc.value)


def test_coerce_items_missing_fails_loud():
    with pytest.raises(ComponentConfigurationError) as exc:
        GlobusTransferStep._coerce_items({})
    assert "FAIL-FAST" in str(exc.value)


def test_coerce_items_non_dict_item_fails_loud():
    with pytest.raises(ComponentConfigurationError) as exc:
        GlobusTransferStep._coerce_items({"items": ["not-a-dict"]})
    assert "FAIL-FAST" in str(exc.value)


def test_coerce_items_missing_source_path_fails_loud():
    with pytest.raises(ComponentConfigurationError) as exc:
        GlobusTransferStep._coerce_items(
            {"items": [{"dest_path": "/b"}]}
        )
    assert "FAIL-FAST" in str(exc.value)
    assert "source_path" in str(exc.value)


def test_coerce_items_missing_dest_path_fails_loud():
    with pytest.raises(ComponentConfigurationError) as exc:
        GlobusTransferStep._coerce_items(
            {"items": [{"source_path": "/a"}]}
        )
    assert "FAIL-FAST" in str(exc.value)
    assert "dest_path" in str(exc.value)


def test_coerce_items_valid_returns_normalized():
    items = GlobusTransferStep._coerce_items(
        {
            "items": [
                {"source_path": "/a", "dest_path": "/b"},
                {"source_path": "/c/", "dest_path": "/d/", "extra": "ignored"},
            ]
        }
    )
    # recursive defaults to False and is always present in the normalized shape.
    assert items == [
        {"source_path": "/a", "dest_path": "/b", "recursive": False},
        {"source_path": "/c/", "dest_path": "/d/", "recursive": False},
    ]


def test_coerce_items_preserves_recursive_flag():
    items = GlobusTransferStep._coerce_items(
        {
            "items": [
                {"source_path": "/dir/", "dest_path": "/out/", "recursive": True},
                {"source_path": "/f", "dest_path": "/g", "recursive": False},
            ]
        }
    )
    assert items[0]["recursive"] is True
    assert items[1]["recursive"] is False


def test_coerce_items_non_bool_recursive_fails_loud():
    with pytest.raises(ComponentConfigurationError, match="recursive"):
        GlobusTransferStep._coerce_items(
            {"items": [{"source_path": "/a", "dest_path": "/b", "recursive": "yes"}]}
        )


# ---------------------------------------------------------------------------
# process() FAIL-LOUD paths (no network)
# ---------------------------------------------------------------------------
def test_process_bad_payload_fails_loud():
    step = GlobusTransferStep.from_config(_write_step_yaml())
    with pytest.raises(ComponentConfigurationError) as exc:
        asyncio.run(step.process("not a dict"))
    assert "FAIL-FAST" in str(exc.value)


def test_process_empty_items_fails_loud_before_any_network():
    step = GlobusTransferStep.from_config(_write_step_yaml())
    with pytest.raises(ComponentConfigurationError) as exc:
        asyncio.run(step.process({"items": []}))
    assert "FAIL-FAST" in str(exc.value)


def test_process_missing_globus_sdk_fails_loud(monkeypatch):
    step = GlobusTransferStep.from_config(_write_step_yaml())
    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "globus_sdk" or name.startswith("globus_sdk."):
            raise ImportError("simulated: globus_sdk not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    with pytest.raises(ComponentConfigurationError) as exc:
        asyncio.run(
            step.process(
                {"items": [{"source_path": "/a", "dest_path": "/b"}]}
            )
        )
    msg = str(exc.value)
    assert "FAIL-FAST" in msg
    assert "globus_sdk" in msg
    assert "pip install" in msg
