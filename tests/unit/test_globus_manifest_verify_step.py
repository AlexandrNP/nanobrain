"""Unit tests for GlobusManifestVerifyStep (G127).

Two layers, per the workspace mock-parity rule:

  * Config + envelope + coercion + FAIL-LOUD-contract tests are unconditional
    (no network).
  * The operation_ls VERIFICATION LOGIC (present / missing / 404-parent /
    non-404-error / per-parent-grouping) is exercised against a fake
    ``TransferClient`` whose ``operation_ls`` returns canned listings or raises
    REAL ``globus_sdk`` error classes. The matching real-dependency test is
    the gated live test in
    ``tests/integration/test_globus_manifest_verify_live.py`` (and, apecx-side,
    the real /public transfer integration test).
"""

from __future__ import annotations

import asyncio
import builtins
import tempfile

import pytest

import nanobrain.library.steps.globus_manifest_verify_step as gmv
from nanobrain.core.component_base import ComponentConfigurationError
from nanobrain.library.steps.globus_manifest_verify_step import (
    OUTPUT_MANIFEST_KEY,
    GlobusManifestVerifyStep,
    GlobusManifestVerifyStepConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _write_step_yaml(extra: dict | None = None, *, with_class: bool = False) -> str:
    lines = []
    if with_class:
        lines.append(
            "class: nanobrain.library.steps.globus_manifest_verify_step.GlobusManifestVerifyStep"
        )
    lines += [
        "name: verify_sources",
        "description: 'test verify step'",
        "source_endpoint_id: src-endpoint-uuid",
    ]
    if extra:
        for k, v in extra.items():
            lines.append(f"{k}: {v!r}")
    f = tempfile.NamedTemporaryFile("w", suffix=".yml", delete=False, encoding="utf-8")
    f.write("\n".join(lines) + "\n")
    f.close()
    return f.name


class _FakeTransferAPIError(Exception):
    """Stands in for globus_sdk.TransferAPIError. Subclasses the real class at
    runtime (see _make_api_error) so the step's ``except`` clause catches it;
    we set only the attributes the step reads (``http_status``, ``code``)."""


def _make_api_error(http_status: int, code: str = "ClientError.NotFound", message: str = ""):
    import globus_sdk

    text = message or f"simulated {http_status}"

    class _Err(globus_sdk.TransferAPIError):
        def __init__(self):
            # Set only the attributes the step reads. ``message`` is a
            # read-only property on the real class — do NOT assign it.
            self.http_status = http_status
            self.code = code

        def __str__(self):
            return f"{self.code}: {text}"

    return _Err()


class _FakeTC:
    """Fake TransferClient: operation_ls returns canned listings or raises."""

    def __init__(self, listings: dict, errors: dict | None = None):
        self._listings = listings
        self._errors = errors or {}
        self.ls_calls: list[str] = []

    def operation_ls(self, endpoint_id, path):
        self.ls_calls.append(path)
        if path in self._errors:
            raise self._errors[path]
        if path not in self._listings:
            raise _make_api_error(404)
        return self._listings[path]


def _patch_globus(monkeypatch, fake_tc: _FakeTC):
    """Patch build_globus_app + globus_sdk.TransferClient to use the fake."""
    import globus_sdk

    monkeypatch.setattr(gmv, "build_globus_app", lambda **kw: object())
    monkeypatch.setattr(globus_sdk, "TransferClient", lambda app=None: fake_tc)


# ---------------------------------------------------------------------------
# Config validation (ConfigBase → file-only)
# ---------------------------------------------------------------------------
def test_config_validates_minimal():
    cfg = GlobusManifestVerifyStepConfig.from_config(_write_step_yaml())
    assert cfg.source_endpoint_id == "src-endpoint-uuid"
    assert cfg.auth_mode == "client_credentials"


def test_config_extra_forbid_rejects_typo():
    path = _write_step_yaml({"auth_modee": "typo"})
    with pytest.raises(Exception) as exc:
        GlobusManifestVerifyStepConfig.from_config(path)
    assert "auth_modee" in str(exc.value) or "extra" in str(exc.value).lower()


def test_config_strips_class_key():
    cfg = GlobusManifestVerifyStepConfig.from_config(_write_step_yaml(with_class=True))
    assert cfg.name == "verify_sources"


# ---------------------------------------------------------------------------
# from_config + framework compliance
# ---------------------------------------------------------------------------
def test_step_loads_from_config_path():
    step = GlobusManifestVerifyStep.from_config(_write_step_yaml())
    assert isinstance(step, GlobusManifestVerifyStep)
    assert step.verify_config.source_endpoint_id == "src-endpoint-uuid"


def test_step_direct_construction_forbidden():
    with pytest.raises(RuntimeError) as exc:
        GlobusManifestVerifyStep()
    assert "Direct instantiation" in str(exc.value)


def test_step_implements_process_not_execute():
    assert "process" in GlobusManifestVerifyStep.__dict__
    assert "execute" not in GlobusManifestVerifyStep.__dict__


# ---------------------------------------------------------------------------
# Trigger-envelope self-unwrap
# ---------------------------------------------------------------------------
def test_unwrap_strips_input_du_key():
    step = GlobusManifestVerifyStep.from_config(_write_step_yaml())
    payload = {"items": [{"source_path": "/a", "dest_path": "/b"}]}
    assert step._unwrap_trigger_envelope({"manifest_in": payload}) == payload


def test_unwrap_passes_through_direct_payload():
    step = GlobusManifestVerifyStep.from_config(_write_step_yaml())
    payload = {"items": [{"source_path": "/a", "dest_path": "/b"}]}
    assert step._unwrap_trigger_envelope(payload) == payload


# ---------------------------------------------------------------------------
# items coercion FAIL-LOUD
# ---------------------------------------------------------------------------
def test_coerce_empty_fails_loud():
    with pytest.raises(ComponentConfigurationError):
        GlobusManifestVerifyStep._coerce_items({"items": []})


def test_coerce_missing_source_path_fails_loud():
    with pytest.raises(ComponentConfigurationError) as exc:
        GlobusManifestVerifyStep._coerce_items({"items": [{"dest_path": "/b"}]})
    assert "source_path" in str(exc.value)


def test_coerce_keeps_dest_path_for_passthrough():
    items = GlobusManifestVerifyStep._coerce_items(
        {"items": [{"source_path": "/a", "dest_path": "/b"}]}
    )
    assert items == [{"source_path": "/a", "dest_path": "/b"}]


# ---------------------------------------------------------------------------
# process() FAIL-LOUD without network
# ---------------------------------------------------------------------------
def test_process_bad_payload_fails_loud():
    step = GlobusManifestVerifyStep.from_config(_write_step_yaml())
    with pytest.raises(ComponentConfigurationError):
        asyncio.run(step.process("not a dict"))


def test_process_missing_globus_sdk_fails_loud(monkeypatch):
    step = GlobusManifestVerifyStep.from_config(_write_step_yaml())
    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "globus_sdk" or name.startswith("globus_sdk."):
            raise ImportError("simulated: globus_sdk not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    with pytest.raises(ComponentConfigurationError) as exc:
        asyncio.run(step.process({"items": [{"source_path": "/a", "dest_path": "/b"}]}))
    assert "globus_sdk" in str(exc.value)


# ---------------------------------------------------------------------------
# process() verification LOGIC against a fake TransferClient
# ---------------------------------------------------------------------------
def test_process_all_present_passthrough(monkeypatch):
    step = GlobusManifestVerifyStep.from_config(_write_step_yaml())
    items = [
        {"source_path": "/p/data/violin/Vaccine_Information.csv", "dest_path": "/d/v.csv"},
        {"source_path": "/p/data/violin/Gene_Information.csv", "dest_path": "/d/g.csv"},
        {"source_path": "/p/data/BV-BRC/BVBRC_genome_alphavirus.csv", "dest_path": "/d/b.csv"},
    ]
    fake = _FakeTC(
        listings={
            "/p/data/violin": [
                {"name": "Vaccine_Information.csv"},
                {"name": "Gene_Information.csv"},
                {"name": "Pathogen_Information.csv"},
            ],
            "/p/data/BV-BRC": [{"name": "BVBRC_genome_alphavirus.csv"}],
        }
    )
    _patch_globus(monkeypatch, fake)
    out = asyncio.run(step.process({"items": items}))
    # Passthrough under the exact output-DU key.
    assert out == {OUTPUT_MANIFEST_KEY: {"items": items}}


def test_process_one_parent_listed_once(monkeypatch):
    """5 files under the same dir cost ONE operation_ls call, not 5."""
    step = GlobusManifestVerifyStep.from_config(_write_step_yaml())
    names = [
        "Vaccine_Information.csv",
        "Pathogen_Information.csv",
        "Gene_Information.csv",
        "Vaccine_Pathogen_Information.csv",
        "Gene_Vaccine_Pathogen_Information.csv",
    ]
    items = [{"source_path": f"/p/violin/{n}", "dest_path": f"/d/{n}"} for n in names]
    fake = _FakeTC(listings={"/p/violin": [{"name": n} for n in names]})
    _patch_globus(monkeypatch, fake)
    asyncio.run(step.process({"items": items}))
    assert fake.ls_calls.count("/p/violin") == 1


def test_process_missing_file_fails_loud(monkeypatch):
    step = GlobusManifestVerifyStep.from_config(_write_step_yaml())
    items = [
        {"source_path": "/p/data/BV-BRC/present.csv", "dest_path": "/d/p.csv"},
        {"source_path": "/p/data/BV-BRC/MISSING.csv", "dest_path": "/d/m.csv"},
    ]
    fake = _FakeTC(listings={"/p/data/BV-BRC": [{"name": "present.csv"}]})
    _patch_globus(monkeypatch, fake)
    with pytest.raises(ComponentConfigurationError) as exc:
        asyncio.run(step.process({"items": items}))
    msg = str(exc.value)
    assert "MISSING" in msg
    assert "/p/data/BV-BRC/MISSING.csv" in msg
    # The present file must NOT be reported missing.
    assert "/p/data/BV-BRC/present.csv" not in msg


def test_process_404_parent_marks_all_missing(monkeypatch):
    step = GlobusManifestVerifyStep.from_config(_write_step_yaml())
    items = [
        {"source_path": "/p/gone/a.csv", "dest_path": "/d/a.csv"},
        {"source_path": "/p/gone/b.csv", "dest_path": "/d/b.csv"},
    ]
    fake = _FakeTC(listings={})  # /p/gone not present → 404
    _patch_globus(monkeypatch, fake)
    with pytest.raises(ComponentConfigurationError) as exc:
        asyncio.run(step.process({"items": items}))
    msg = str(exc.value)
    assert "/p/gone/a.csv" in msg and "/p/gone/b.csv" in msg


def test_process_403_surfaced_as_authorization_with_group_hint(monkeypatch):
    """A 403 (no effective ACL) is NOT 'file missing' — it's an authorization
    error, and the hint must mention Globus Group membership (the exact wall an
    operator hits for group-gated data)."""
    step = GlobusManifestVerifyStep.from_config(_write_step_yaml())
    items = [{"source_path": "/restricted/a.csv", "dest_path": "/d/a.csv"}]
    fake = _FakeTC(
        listings={},
        errors={"/restricted": _make_api_error(403, code="PermissionDenied")},
    )
    _patch_globus(monkeypatch, fake)
    with pytest.raises(ComponentConfigurationError) as exc:
        asyncio.run(step.process({"items": items}))
    msg = str(exc.value)
    assert "Authorization error" in msg
    assert "Globus Group" in msg
    assert "MISSING" not in msg  # not the missing-file message


def test_process_path_not_allowed_surfaced_as_path_restriction(monkeypatch):
    """A 500 'Path not allowed' must say the path is outside the collection's
    namespace (a different collection serves it) — distinct from authorization
    and from missing-file."""
    step = GlobusManifestVerifyStep.from_config(_write_step_yaml())
    items = [{"source_path": "/elsewhere/a.csv", "dest_path": "/d/a.csv"}]

    err = _make_api_error(500, code="ExternalError", message="Path not allowed.")
    fake = _FakeTC(listings={}, errors={"/elsewhere": err})
    _patch_globus(monkeypatch, fake)
    with pytest.raises(ComponentConfigurationError) as exc:
        asyncio.run(step.process({"items": items}))
    msg = str(exc.value)
    assert "Path-restriction error" in msg
    assert "DIFFERENT collection" in msg
    assert "MISSING" not in msg
