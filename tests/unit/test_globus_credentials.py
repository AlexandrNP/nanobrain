"""Unit tests for the keyring-backed Globus credential store (G30).

Unconditional — no network, no real OS keychain. Tests install a real
*test* keyring backend (``keyring.set_keyring`` with an in-memory
``KeyringBackend`` subclass) and restore the original in teardown. That
test backend is a real keyring backend, NOT a mock of the framework —
it exercises the real ``keyring.get_password`` / ``set_password`` /
``delete_password`` code paths. The real OS keychain is NEVER touched.

Covers:
  * store / load / clear round-trip against a test backend.
  * ``clear_credentials`` is idempotent (no error when absent).
  * ``load_credentials`` returns ``(None, None)`` when nothing stored.
  * ``credential_status`` shape + never-reveals-secret contract.
  * the insecure-backend guard FAIL-LOUDs ``store_credentials`` when the
    active backend is ``keyring.backends.fail.Keyring``.
  * missing-``keyring`` FAIL-LOUDs every public function.
  * ``build_globus_app`` 3-tier precedence: args beat env beat keyring.
"""

from __future__ import annotations

import builtins

import pytest

# The whole module exercises the keyring-backed credential store via
# real keyring API (an in-memory test backend, not a mock). Without
# the ``keyring`` package installed there is no useful test surface —
# skip the whole file via the standard pytest gate instead of letting
# per-test fixture ``import keyring`` raise during setup. ``keyring``
# is declared in the apecx-mcp-integration ``hpc`` extra; framework
# users not on the HPC/Globus path don't install it.
pytest.importorskip(
    "keyring",
    reason=(
        "keyring not installed — globus_credentials store requires it. "
        "Install with: pip install keyring  (or via the apecx hpc extra)."
    ),
)

from nanobrain.core.component_base import ComponentConfigurationError
from nanobrain.core.distributed import globus_credentials
from nanobrain.core.distributed.globus_credentials import (
    KEYRING_SERVICE,
    clear_credentials,
    credential_status,
    load_credentials,
    store_credentials,
)


# ---------------------------------------------------------------------------
# Test keyring backends — real keyring.backend.KeyringBackend subclasses,
# not mocks of the framework. They exercise the real keyring API surface.
# ---------------------------------------------------------------------------
def _make_memory_backend():
    """A real in-memory KeyringBackend (secure-classified, not in the
    insecure denylist)."""
    from keyring.backend import KeyringBackend
    from keyring.errors import PasswordDeleteError

    class _MemoryTestKeyring(KeyringBackend):
        """In-memory test backend. NOT in _INSECURE_BACKEND_QUALNAMES, so
        the credential store treats it as a secure store for tests."""

        priority = 1  # type: ignore[assignment]

        def __init__(self):
            super().__init__()
            self._store: dict = {}

        def get_password(self, service, username):
            return self._store.get((service, username))

        def set_password(self, service, username, password):
            self._store[(service, username)] = password

        def delete_password(self, service, username):
            if (service, username) not in self._store:
                raise PasswordDeleteError("not found")
            del self._store[(service, username)]

    return _MemoryTestKeyring()


@pytest.fixture
def memory_keyring():
    """Install an in-memory test backend; restore the original in teardown."""
    import keyring

    original = keyring.get_keyring()
    backend = _make_memory_backend()
    keyring.set_keyring(backend)
    try:
        yield backend
    finally:
        keyring.set_keyring(original)


@pytest.fixture
def fail_keyring():
    """Install the real ``fail.Keyring`` backend (no usable secure store)."""
    import keyring
    import keyring.backends.fail

    original = keyring.get_keyring()
    keyring.set_keyring(keyring.backends.fail.Keyring())
    try:
        yield
    finally:
        keyring.set_keyring(original)


# ---------------------------------------------------------------------------
# store / load / clear round-trip
# ---------------------------------------------------------------------------
def test_store_then_load_round_trip(memory_keyring):
    store_credentials("client-abc", "secret-xyz")
    client_id, client_secret = load_credentials()
    assert client_id == "client-abc"
    assert client_secret == "secret-xyz"


def test_load_when_nothing_stored_returns_none_pair(memory_keyring):
    # Absence is a normal state — load must NOT raise.
    assert load_credentials() == (None, None)


def test_clear_removes_both(memory_keyring):
    store_credentials("client-abc", "secret-xyz")
    clear_credentials()
    assert load_credentials() == (None, None)


def test_clear_is_idempotent_when_absent(memory_keyring):
    # Clearing an already-clear store is a success, not an error.
    clear_credentials()
    clear_credentials()
    assert load_credentials() == (None, None)


def test_store_rejects_empty_client_id(memory_keyring):
    with pytest.raises(ComponentConfigurationError) as exc:
        store_credentials("", "secret")
    assert "FAIL-FAST" in str(exc.value)
    assert "client_id" in str(exc.value)


def test_store_rejects_empty_client_secret(memory_keyring):
    with pytest.raises(ComponentConfigurationError) as exc:
        store_credentials("client-abc", "   ")
    assert "FAIL-FAST" in str(exc.value)
    assert "client_secret" in str(exc.value)


def test_keys_land_under_the_service_name(memory_keyring):
    store_credentials("client-abc", "secret-xyz")
    # Verify the literal keyring entries directly through the backend.
    assert memory_keyring.get_password(KEYRING_SERVICE, "client_id") == "client-abc"
    assert memory_keyring.get_password(KEYRING_SERVICE, "client_secret") == "secret-xyz"


# ---------------------------------------------------------------------------
# credential_status — shape + never-reveals-secret
# ---------------------------------------------------------------------------
def test_credential_status_shape_when_stored(memory_keyring):
    store_credentials("client-abc", "secret-xyz")
    status = credential_status()
    assert set(status.keys()) == {
        "client_id",
        "client_secret_set",
        "keyring_backend",
        "backend_secure",
    }
    assert status["client_id"] == "client-abc"
    assert status["client_secret_set"] is True
    assert status["backend_secure"] is True
    assert "_MemoryTestKeyring" in status["keyring_backend"]


def test_credential_status_when_absent(memory_keyring):
    status = credential_status()
    assert status["client_id"] is None
    assert status["client_secret_set"] is False


def test_credential_status_never_contains_the_secret(memory_keyring):
    store_credentials("client-abc", "super-secret-value")
    status = credential_status()
    # The secret value must not appear anywhere in the status payload.
    assert "super-secret-value" not in repr(status)
    assert "super-secret-value" not in str(status.values())
    # Only the boolean is exposed.
    assert status["client_secret_set"] is True


# ---------------------------------------------------------------------------
# insecure-backend guard — FAIL-LOUD on fail.Keyring
# ---------------------------------------------------------------------------
def test_store_fails_loud_on_insecure_backend(fail_keyring):
    with pytest.raises(ComponentConfigurationError) as exc:
        store_credentials("client-abc", "secret-xyz")
    msg = str(exc.value)
    assert "FAIL-FAST" in msg
    assert "not a secure credential store" in msg
    # The actionable message must mention the env-var escape hatch.
    assert "GLOBUS_COMPUTE_CLIENT_ID" in msg


def test_credential_status_reports_insecure_backend(fail_keyring):
    status = credential_status()
    assert status["backend_secure"] is False
    assert status["keyring_backend"] == "keyring.backends.fail.Keyring"


# ---------------------------------------------------------------------------
# missing keyring — FAIL-LOUD on every public function
# ---------------------------------------------------------------------------
@pytest.fixture
def no_keyring(monkeypatch):
    """Monkeypatch the import so ``import keyring`` raises ImportError."""
    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "keyring" or name.startswith("keyring."):
            raise ImportError("simulated: keyring not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)


def test_store_fails_loud_without_keyring(no_keyring):
    with pytest.raises(ComponentConfigurationError) as exc:
        store_credentials("client-abc", "secret-xyz")
    msg = str(exc.value)
    assert "FAIL-FAST" in msg
    assert "keyring" in msg
    assert "pip install keyring" in msg


def test_load_fails_loud_without_keyring(no_keyring):
    with pytest.raises(ComponentConfigurationError) as exc:
        load_credentials()
    assert "pip install keyring" in str(exc.value)


def test_clear_fails_loud_without_keyring(no_keyring):
    with pytest.raises(ComponentConfigurationError) as exc:
        clear_credentials()
    assert "pip install keyring" in str(exc.value)


def test_status_fails_loud_without_keyring(no_keyring):
    with pytest.raises(ComponentConfigurationError) as exc:
        credential_status()
    assert "pip install keyring" in str(exc.value)


# ---------------------------------------------------------------------------
# build_globus_app 3-tier precedence: args -> env -> keyring
# ---------------------------------------------------------------------------
def test_build_globus_app_keyring_tier(memory_keyring, monkeypatch):
    """With no args and no env vars, credentials come from the keyring."""
    from nanobrain.core.distributed.globus_auth import (
        ENV_CLIENT_ID,
        ENV_CLIENT_SECRET,
        build_globus_app,
    )
    import globus_sdk

    monkeypatch.delenv(ENV_CLIENT_ID, raising=False)
    monkeypatch.delenv(ENV_CLIENT_SECRET, raising=False)
    store_credentials("keyring-id", "keyring-secret")

    app = build_globus_app(auth_mode="client_credentials")
    assert isinstance(app, globus_sdk.ClientApp)
    assert app.client_id == "keyring-id"


def test_build_globus_app_env_beats_keyring(memory_keyring, monkeypatch):
    """Env vars take precedence over the keyring tier."""
    from nanobrain.core.distributed.globus_auth import (
        ENV_CLIENT_ID,
        ENV_CLIENT_SECRET,
        build_globus_app,
    )
    import globus_sdk

    monkeypatch.setenv(ENV_CLIENT_ID, "env-id")
    monkeypatch.setenv(ENV_CLIENT_SECRET, "env-secret")
    store_credentials("keyring-id", "keyring-secret")

    app = build_globus_app(auth_mode="client_credentials")
    assert isinstance(app, globus_sdk.ClientApp)
    assert app.client_id == "env-id"


def test_build_globus_app_args_beat_env_and_keyring(memory_keyring, monkeypatch):
    """Explicit args take precedence over both env vars and the keyring."""
    from nanobrain.core.distributed.globus_auth import (
        ENV_CLIENT_ID,
        ENV_CLIENT_SECRET,
        build_globus_app,
    )
    import globus_sdk

    monkeypatch.setenv(ENV_CLIENT_ID, "env-id")
    monkeypatch.setenv(ENV_CLIENT_SECRET, "env-secret")
    store_credentials("keyring-id", "keyring-secret")

    app = build_globus_app(
        auth_mode="client_credentials",
        client_id="explicit-id",
        client_secret="explicit-secret",
    )
    assert isinstance(app, globus_sdk.ClientApp)
    assert app.client_id == "explicit-id"


def test_build_globus_app_per_field_tier_resolution(memory_keyring, monkeypatch):
    """Each field resolves independently: env id + keyring secret combine."""
    from nanobrain.core.distributed.globus_auth import (
        ENV_CLIENT_ID,
        ENV_CLIENT_SECRET,
        build_globus_app,
    )
    import globus_sdk

    monkeypatch.setenv(ENV_CLIENT_ID, "env-id")
    monkeypatch.delenv(ENV_CLIENT_SECRET, raising=False)
    store_credentials("keyring-id", "keyring-secret")

    app = build_globus_app(auth_mode="client_credentials")
    assert isinstance(app, globus_sdk.ClientApp)
    # id from env (higher tier), secret from keyring (lower tier).
    assert app.client_id == "env-id"


def test_build_globus_app_still_fails_loud_when_no_tier_has_creds(monkeypatch):
    """With nothing in args, env, OR keyring, the existing FAIL-LOUD fires.

    The keyring tier is empty here because no test backend is installed
    in this test — whatever the host keyring is, it has no
    ``nanobrain-globus`` entry. The missing-keyring path inside
    ``_load_keyring_credentials`` must not mask this error.
    """
    from nanobrain.core.distributed.globus_auth import (
        ENV_CLIENT_ID,
        ENV_CLIENT_SECRET,
        build_globus_app,
    )

    monkeypatch.delenv(ENV_CLIENT_ID, raising=False)
    monkeypatch.delenv(ENV_CLIENT_SECRET, raising=False)
    # Force the keyring tier to contribute nothing, deterministically.
    monkeypatch.setattr(
        "nanobrain.core.distributed.globus_auth._load_keyring_credentials",
        lambda: (None, None),
    )
    with pytest.raises(ComponentConfigurationError) as exc:
        build_globus_app(auth_mode="client_credentials")
    assert "FAIL-FAST" in str(exc.value)


def test_build_globus_app_missing_keyring_does_not_mask_creds_error(monkeypatch):
    """A missing-keyring ImportError must not surface in place of the real
    'no credentials found' FAIL-LOUD."""
    from nanobrain.core.distributed.globus_auth import (
        ENV_CLIENT_ID,
        ENV_CLIENT_SECRET,
        build_globus_app,
    )

    monkeypatch.delenv(ENV_CLIENT_ID, raising=False)
    monkeypatch.delenv(ENV_CLIENT_SECRET, raising=False)

    # Make globus_credentials.load_credentials raise as if keyring is absent.
    def _raise_missing(*_a, **_k):
        raise ComponentConfigurationError("FAIL-FAST: ... keyring ... pip install keyring")

    monkeypatch.setattr(globus_credentials, "load_credentials", _raise_missing)

    with pytest.raises(ComponentConfigurationError) as exc:
        build_globus_app(auth_mode="client_credentials")
    msg = str(exc.value)
    # The error must be the credentials-missing one, not the keyring one.
    assert "client_id" in msg or "client_secret" in msg
    assert "confidential-client credentials" in msg
