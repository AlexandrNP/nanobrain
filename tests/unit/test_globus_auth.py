"""Unit tests for the shared Globus Auth helper (G23).

Unconditional — no network, no real Globus auth. The FAIL-LOUD paths are
exercised by monkeypatching the environment and the ``globus_sdk`` import;
that is testing the error contract, not mocking a dependency.

Covers:
  * ``build_globus_app`` in ``client_credentials`` mode builds a ClientApp
    from explicit args and from env vars.
  * ``build_globus_app`` in ``native`` mode builds a UserApp.
  * FAIL-LOUD when ``client_credentials`` mode has no credentials.
  * FAIL-LOUD when ``native`` mode has no client_id.
  * FAIL-LOUD on an unknown ``auth_mode``.
  * FAIL-LOUD when ``globus_sdk`` is not importable.
  * scope strings are grouped by resource server.
"""

from __future__ import annotations

import builtins
import importlib

import pytest

from nanobrain.core.component_base import ComponentConfigurationError
from nanobrain.core.distributed import globus_auth
from nanobrain.core.distributed.globus_auth import (
    ENV_CLIENT_ID,
    ENV_CLIENT_SECRET,
    build_globus_app,
)

_COMPUTE_SCOPE = "https://auth.globus.org/scopes/facd7ccc-c5f4-42aa-916b-a0e270e2c2a9/all"
_TRANSFER_SCOPE = "urn:globus:auth:scope:transfer.api.globus.org:all"


@pytest.fixture(autouse=True)
def _isolate_keyring(monkeypatch):
    """Isolate every test from the OS keyring (tier-3 credential source).

    ``build_globus_app`` resolves credentials args -> env -> keyring. The
    FAIL-LOUD "missing credentials" tests delete the env vars and expect a
    raise — but on a developer machine that has real creds stored under the
    ``nanobrain-globus`` keyring service, the tier-3 lookup would find them and
    the expected raise never fires (the suite passes in CI's empty keyring but
    fails locally). Stub the tier-3 loader to "nothing stored" so the tests
    assert the contract deterministically regardless of the host keyring. A
    test that specifically wants keyring creds can re-patch it."""
    monkeypatch.setattr(globus_auth, "_load_keyring_credentials", lambda: (None, None))


# ---------------------------------------------------------------------------
# client_credentials mode
# ---------------------------------------------------------------------------
def test_client_credentials_from_explicit_args(monkeypatch):
    monkeypatch.delenv(ENV_CLIENT_ID, raising=False)
    monkeypatch.delenv(ENV_CLIENT_SECRET, raising=False)
    import globus_sdk

    app = build_globus_app(
        auth_mode="client_credentials",
        scopes=[_COMPUTE_SCOPE],
        client_id="explicit-id",
        client_secret="explicit-secret",
    )
    assert isinstance(app, globus_sdk.ClientApp)


def test_client_credentials_from_env_vars(monkeypatch):
    monkeypatch.setenv(ENV_CLIENT_ID, "env-id")
    monkeypatch.setenv(ENV_CLIENT_SECRET, "env-secret")
    import globus_sdk

    app = build_globus_app(auth_mode="client_credentials")
    assert isinstance(app, globus_sdk.ClientApp)


def test_client_credentials_missing_id_fails_loud(monkeypatch):
    monkeypatch.delenv(ENV_CLIENT_ID, raising=False)
    monkeypatch.setenv(ENV_CLIENT_SECRET, "env-secret")
    with pytest.raises(ComponentConfigurationError) as exc:
        build_globus_app(auth_mode="client_credentials")
    msg = str(exc.value)
    assert "FAIL-FAST" in msg
    assert "client_id" in msg


def test_client_credentials_missing_secret_fails_loud(monkeypatch):
    monkeypatch.setenv(ENV_CLIENT_ID, "env-id")
    monkeypatch.delenv(ENV_CLIENT_SECRET, raising=False)
    with pytest.raises(ComponentConfigurationError) as exc:
        build_globus_app(auth_mode="client_credentials")
    msg = str(exc.value)
    assert "FAIL-FAST" in msg
    assert "client_secret" in msg


def test_client_credentials_missing_both_fails_loud(monkeypatch):
    monkeypatch.delenv(ENV_CLIENT_ID, raising=False)
    monkeypatch.delenv(ENV_CLIENT_SECRET, raising=False)
    with pytest.raises(ComponentConfigurationError) as exc:
        build_globus_app()  # client_credentials is the default
    assert "FAIL-FAST" in str(exc.value)


# ---------------------------------------------------------------------------
# native mode
# ---------------------------------------------------------------------------
def test_native_mode_builds_user_app(monkeypatch):
    monkeypatch.delenv(ENV_CLIENT_ID, raising=False)
    import globus_sdk

    app = build_globus_app(
        auth_mode="native",
        scopes=[_TRANSFER_SCOPE],
        client_id="native-client-id",
    )
    assert isinstance(app, globus_sdk.UserApp)


def test_native_mode_requests_refresh_tokens(monkeypatch):
    """Native UserApp MUST request refresh tokens (offline access) — otherwise
    the persisted token is online-only and dies in ~2 days, breaking the
    (now-default) interactive-login install path. Regression for the 2026-05-21
    native-default flip."""
    monkeypatch.delenv(ENV_CLIENT_ID, raising=False)

    app = build_globus_app(auth_mode="native", client_id="native-client-id")
    assert app.config.request_refresh_tokens is True


def test_native_mode_client_id_from_env(monkeypatch):
    monkeypatch.setenv(ENV_CLIENT_ID, "native-env-id")
    import globus_sdk

    app = build_globus_app(auth_mode="native")
    assert isinstance(app, globus_sdk.UserApp)


def test_native_mode_missing_client_id_fails_loud(monkeypatch):
    monkeypatch.delenv(ENV_CLIENT_ID, raising=False)
    with pytest.raises(ComponentConfigurationError) as exc:
        build_globus_app(auth_mode="native")
    msg = str(exc.value)
    assert "FAIL-FAST" in msg
    assert "client_id" in msg


# ---------------------------------------------------------------------------
# unknown auth_mode
# ---------------------------------------------------------------------------
def test_unknown_auth_mode_fails_loud():
    with pytest.raises(ComponentConfigurationError) as exc:
        build_globus_app(auth_mode="oauth_pkce_dance")
    assert "FAIL-FAST" in str(exc.value)
    assert "auth_mode" in str(exc.value)


# ---------------------------------------------------------------------------
# globus_sdk not installed — monkeypatch the import to raise ImportError
# ---------------------------------------------------------------------------
def test_missing_globus_sdk_fails_loud(monkeypatch):
    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "globus_sdk" or name.startswith("globus_sdk."):
            raise ImportError("simulated: globus_sdk not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    with pytest.raises(ComponentConfigurationError) as exc:
        build_globus_app(
            auth_mode="client_credentials",
            client_id="id",
            client_secret="secret",
        )
    msg = str(exc.value)
    assert "FAIL-FAST" in msg
    assert "globus_sdk" in msg
    assert "pip install" in msg


# ---------------------------------------------------------------------------
# scope grouping
# ---------------------------------------------------------------------------
def test_scope_resource_server_extraction():
    # urn-form scope
    assert globus_auth._scope_resource_server(_TRANSFER_SCOPE) == "transfer.api.globus.org"
    # https-form scope
    assert (
        globus_auth._scope_resource_server(_COMPUTE_SCOPE) == "facd7ccc-c5f4-42aa-916b-a0e270e2c2a9"
    )


def test_both_scopes_grouped_on_one_app(monkeypatch):
    """One ClientApp can carry BOTH Compute and Transfer scopes."""
    import globus_sdk

    grouped = globus_auth._build_scope_requirements(globus_sdk, [_COMPUTE_SCOPE, _TRANSFER_SCOPE])
    assert set(grouped.keys()) == {
        "facd7ccc-c5f4-42aa-916b-a0e270e2c2a9",
        "transfer.api.globus.org",
    }
    # And the app builds with both.
    app = build_globus_app(
        auth_mode="client_credentials",
        scopes=[_COMPUTE_SCOPE, _TRANSFER_SCOPE],
        client_id="id",
        client_secret="secret",
    )
    assert isinstance(app, globus_sdk.ClientApp)


def test_empty_scope_string_fails_loud():
    import globus_sdk

    with pytest.raises(ComponentConfigurationError) as exc:
        globus_auth._build_scope_requirements(globus_sdk, ["", "  "])
    assert "FAIL-FAST" in str(exc.value)


def test_module_reimports_cleanly():
    """Importing the helper module must NOT force globus_sdk at import time."""
    mod = importlib.reload(globus_auth)
    assert hasattr(mod, "build_globus_app")
