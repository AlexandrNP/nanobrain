"""Gated live integration test for GlobusManifestVerifyStep (G127).

Real-dependency parity for the mocked unit tests in
``tests/unit/test_globus_manifest_verify_step.py`` (workspace mock-parity
rule: any behavior a unit mock asserts must also be exercised against the real
dependency).

Auto-skips unless the operator provides a real source endpoint + a known-present
path via env vars, AND confidential-client credentials are resolvable
(``build_globus_app`` succeeds). Nanobrain stays framework-general — no
apecx-specific endpoint is hardcoded.

Run it::

    export NANOBRAIN_GLOBUS_TEST_SOURCE_EP=<source-collection-uuid>
    export NANOBRAIN_GLOBUS_TEST_EXISTING_PATH=/abs/path/to/a/real/file.csv
    # creds via apecx-globus-setup store / $GLOBUS_COMPUTE_CLIENT_ID+SECRET
    pytest tests/integration/test_globus_manifest_verify_live.py
"""

from __future__ import annotations

import os
import posixpath

import pytest

from nanobrain.core.component_base import ComponentConfigurationError

_SOURCE_EP = os.environ.get("NANOBRAIN_GLOBUS_TEST_SOURCE_EP", "").strip()
_EXISTING = os.environ.get("NANOBRAIN_GLOBUS_TEST_EXISTING_PATH", "").strip()


def _creds_available() -> bool:
    try:
        from nanobrain.core.distributed.globus_auth import build_globus_app

        build_globus_app(
            auth_mode="client_credentials",
            scopes=["urn:globus:auth:scope:transfer.api.globus.org:all"],
            app_name="nanobrain-globus-verify-live-test",
        )
        return True
    except Exception:  # noqa: BLE001 — any failure means "can't run live"
        return False


pytestmark = pytest.mark.skipif(
    not (_SOURCE_EP and _EXISTING and _creds_available()),
    reason=(
        "live Globus verify test needs NANOBRAIN_GLOBUS_TEST_SOURCE_EP + "
        "NANOBRAIN_GLOBUS_TEST_EXISTING_PATH set and resolvable "
        "confidential-client credentials"
    ),
)


def _build_step():
    import tempfile

    from nanobrain.library.steps.globus_manifest_verify_step import (
        GlobusManifestVerifyStep,
    )

    f = tempfile.NamedTemporaryFile("w", suffix=".yml", delete=False, encoding="utf-8")
    f.write(f"name: verify_live\nsource_endpoint_id: {_SOURCE_EP}\nauth_mode: client_credentials\n")
    f.close()
    return GlobusManifestVerifyStep.from_config(f.name)


@pytest.mark.asyncio
async def test_existing_source_passes_through():
    step = _build_step()
    items = [{"source_path": _EXISTING, "dest_path": "/tmp/verify_live_dest"}]
    out = await step.process({"items": items})
    assert out["verified_manifest"]["items"] == items


@pytest.mark.asyncio
async def test_missing_source_fails_loud():
    step = _build_step()
    bogus = posixpath.join(posixpath.dirname(_EXISTING), "__nanobrain_does_not_exist__.dat")
    items = [
        {"source_path": _EXISTING, "dest_path": "/tmp/ok"},
        {"source_path": bogus, "dest_path": "/tmp/bad"},
    ]
    with pytest.raises(ComponentConfigurationError) as exc:
        await step.process({"items": items})
    assert bogus in str(exc.value)
    # The genuinely-present file must NOT be reported missing.
    assert _EXISTING not in str(exc.value).split("Missing:")[-1]
