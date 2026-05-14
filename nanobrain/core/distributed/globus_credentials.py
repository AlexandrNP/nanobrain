"""Keyring-backed secure credential store for Globus credentials (G30).

The Globus Compute executor and Transfer step authenticate with a
confidential client — a ``client_id`` / ``client_secret`` pair. Before
this module the only resolution tiers were explicit arguments and the
``$GLOBUS_COMPUTE_CLIENT_ID`` / ``$GLOBUS_COMPUTE_CLIENT_SECRET``
environment variables (see ``globus_auth.py``). Environment variables
leak into process listings, shell history, and CI logs. This module
adds a third, lowest-precedence tier: the OS secure credential store
(macOS Keychain, Windows Credential Locker, Secret Service on Linux),
accessed via the ``keyring`` package.

It lives in ``core/`` deliberately, alongside ``globus_auth.py``: it is
consumed by the auth helper (a ``core`` module) and by the
``apecx-globus-setup`` CLI (apecx-side). ``core`` MUST NOT depend on
``library``, and a helper consumed by both has to sit in ``core``.

The ``keyring`` import is lazy (inside each function) and FAIL-LOUD —
importing this module never forces the ``keyring`` dependency on a
workflow that resolves credentials from explicit args or env vars.

Insecure-backend guard
----------------------
``keyring`` always returns *some* backend from ``get_keyring()``. In an
environment with no usable secure store it returns
``keyring.backends.fail.Keyring`` — a backend whose ``set_password``
raises ``NoKeyringError``. ``store_credentials`` checks for this case
(and the ``keyrings.alt`` plaintext/obfuscated backends, which are
insecure-by-design) BEFORE writing, and raises
``ComponentConfigurationError`` rather than ever silently persisting a
secret in plaintext. This guard is an anti-silent-failure measure and
is treated as load-bearing, not optional.

Public API
----------
``store_credentials(client_id, client_secret)``
    Persist both values under the ``KEYRING_SERVICE`` service name.
    FAIL-LOUD if the active backend is insecure.
``load_credentials() -> (client_id | None, client_secret | None)``
    Read both back. Absence returns ``(None, None)`` — it does NOT
    raise, because "no credentials stored yet" is a normal state.
``clear_credentials()``
    Delete both keys. Idempotent — no error if either is absent.
``credential_status() -> dict``
    A non-secret status snapshot: the client_id (or None), whether a
    secret is set (bool, never the value), the backend class name, and
    whether the backend is considered secure.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

from nanobrain.core.component_base import ComponentConfigurationError

# The keyring "service" name all nanobrain Globus credentials live under.
# Two entries are stored under this service, keyed by username:
KEYRING_SERVICE = "nanobrain-globus"
_KEY_CLIENT_ID = "client_id"
_KEY_CLIENT_SECRET = "client_secret"

# Backend class names (module-qualified) that are insecure by design and
# must never receive a stored secret. ``fail.Keyring`` is the "no backend
# available" sentinel; the ``keyrings.alt`` plaintext / encrypted-with-a-
# local-key backends are insecure-by-design (their own docs say so).
_INSECURE_BACKEND_QUALNAMES = frozenset(
    {
        "keyring.backends.fail.Keyring",
        "keyrings.alt.file.PlaintextKeyring",
        "keyrings.alt.file.EncryptedKeyring",
        "keyrings.alt.Windows.RegistryKeyring",
    }
)


def _import_keyring() -> "Any":
    """Lazy-import ``keyring``; FAIL-LOUD with an actionable message."""
    try:
        import keyring  # noqa: PLC0415 - intentional lazy import
    except ImportError as exc:
        raise ComponentConfigurationError(
            "FAIL-FAST: the secure Globus credential store requires the "
            "'keyring' package, which is not installed. Install it with "
            "`pip install keyring`. The credential store is only needed "
            "when you store Globus credentials in the OS secure store "
            "(rather than passing them explicitly or via the "
            "$GLOBUS_COMPUTE_CLIENT_ID / $GLOBUS_COMPUTE_CLIENT_SECRET "
            f"environment variables). Underlying error: {exc}"
        ) from exc
    return keyring


def _backend_qualname(backend: "Any") -> str:
    """Module-qualified class name of a keyring backend instance."""
    cls = type(backend)
    return f"{cls.__module__}.{cls.__qualname__}"


def _backend_is_secure(backend: "Any") -> bool:
    """Whether a keyring backend is considered safe to store a secret in.

    Anything in :data:`_INSECURE_BACKEND_QUALNAMES` is rejected. We do
    NOT attempt an exhaustive positive classification of every secure
    backend — that would be brittle over-engineering. The contract is:
    the known-insecure backends are rejected; everything else (the OS
    Keychain / Credential Locker / Secret Service backends, and any
    third-party backend an operator deliberately installed) is trusted.
    """
    return _backend_qualname(backend) not in _INSECURE_BACKEND_QUALNAMES


def store_credentials(client_id: str, client_secret: str) -> None:
    """Store a Globus confidential-client pair in the OS secure store.

    Args:
        client_id: The confidential-client UUID.
        client_secret: The confidential-client secret.

    Raises:
        ComponentConfigurationError: ``keyring`` is not installed, the
            active keyring backend is insecure (no usable secure store —
            we never silently persist a secret in plaintext), or
            ``client_id`` / ``client_secret`` is empty.
    """
    if not client_id or not str(client_id).strip():
        raise ComponentConfigurationError(
            "FAIL-FAST: store_credentials requires a non-empty client_id."
        )
    if not client_secret or not str(client_secret).strip():
        raise ComponentConfigurationError(
            "FAIL-FAST: store_credentials requires a non-empty client_secret."
        )

    keyring = _import_keyring()
    backend = keyring.get_keyring()
    if not _backend_is_secure(backend):
        raise ComponentConfigurationError(
            "FAIL-FAST: refusing to store Globus credentials — the active "
            f"keyring backend ({_backend_qualname(backend)}) is not a "
            "secure credential store. Storing a client_secret there would "
            "either fail or persist it in plaintext. To fix this:\n"
            "  - macOS / Windows: the OS Keychain / Credential Locker "
            "backend should be available by default; this usually means "
            "keyring is being run in a stripped environment.\n"
            "  - Linux: install a Secret Service provider (e.g. "
            "`gnome-keyring` or `kwallet`) plus the `secretstorage` "
            "Python package, or run inside a desktop session.\n"
            "  - Headless / CI: do NOT use the credential store — pass "
            "the credentials explicitly or via the "
            "$GLOBUS_COMPUTE_CLIENT_ID / $GLOBUS_COMPUTE_CLIENT_SECRET "
            "environment variables instead."
        )

    keyring.set_password(KEYRING_SERVICE, _KEY_CLIENT_ID, str(client_id))
    keyring.set_password(KEYRING_SERVICE, _KEY_CLIENT_SECRET, str(client_secret))


def load_credentials() -> Tuple[Optional[str], Optional[str]]:
    """Load the stored Globus confidential-client pair.

    Returns:
        ``(client_id, client_secret)``. Either element is ``None`` when
        absent from the store; ``(None, None)`` when nothing is stored.
        Absence is a normal state, NOT an error — this function does not
        raise on a missing entry. It DOES raise (via :func:`_import_keyring`)
        if ``keyring`` itself is not installed.

    Raises:
        ComponentConfigurationError: ``keyring`` is not installed.
    """
    keyring = _import_keyring()
    backend = keyring.get_keyring()
    # An insecure backend (e.g. ``fail.Keyring`` — keyring installed but
    # no usable secure store) raises on get_password. Treat that as "no
    # credentials stored" rather than propagating: absence is a normal
    # state for this function, and there is genuinely nothing readable.
    if not _backend_is_secure(backend):
        return None, None
    client_id = keyring.get_password(KEYRING_SERVICE, _KEY_CLIENT_ID)
    client_secret = keyring.get_password(KEYRING_SERVICE, _KEY_CLIENT_SECRET)
    return client_id, client_secret


def clear_credentials() -> None:
    """Delete the stored Globus credentials. Idempotent.

    Deleting an absent key is not an error — ``keyring`` raises
    ``PasswordDeleteError`` for a missing entry, which we swallow so the
    overall operation is idempotent (clearing an already-clear store is
    a success).

    Raises:
        ComponentConfigurationError: ``keyring`` is not installed.
    """
    keyring = _import_keyring()
    # keyring.errors is always importable once keyring is.
    from keyring.errors import PasswordDeleteError  # noqa: PLC0415

    for username in (_KEY_CLIENT_ID, _KEY_CLIENT_SECRET):
        try:
            keyring.delete_password(KEYRING_SERVICE, username)
        except PasswordDeleteError:
            # Absent entry — idempotent clear, not a failure.
            pass


def credential_status() -> dict:
    """Return a non-secret snapshot of the credential-store state.

    Returns:
        A dict with::

            {
                "client_id": <the stored client_id or None>,
                "client_secret_set": <bool — whether a secret is stored>,
                "keyring_backend": <module-qualified backend class name>,
                "backend_secure": <bool — whether the backend is secure>,
            }

        The ``client_secret`` value is NEVER included — only the boolean
        ``client_secret_set``. The ``client_id`` is not a secret (it is a
        public UUID) and is returned as-is for operator confirmation.

    Raises:
        ComponentConfigurationError: ``keyring`` is not installed.
    """
    keyring = _import_keyring()
    backend = keyring.get_keyring()
    secure = _backend_is_secure(backend)

    # An insecure backend (notably ``fail.Keyring``) raises on
    # get_password — there is nothing to read. Report the backend state
    # without attempting a read, so ``status`` stays informative rather
    # than crashing on exactly the environment the operator most needs
    # diagnostics for.
    if secure:
        client_id = keyring.get_password(KEYRING_SERVICE, _KEY_CLIENT_ID)
        client_secret = keyring.get_password(KEYRING_SERVICE, _KEY_CLIENT_SECRET)
    else:
        client_id = None
        client_secret = None

    return {
        "client_id": client_id,
        "client_secret_set": bool(client_secret),
        "keyring_backend": _backend_qualname(backend),
        "backend_secure": secure,
    }


__all__ = [
    "KEYRING_SERVICE",
    "store_credentials",
    "load_credentials",
    "clear_credentials",
    "credential_status",
]
