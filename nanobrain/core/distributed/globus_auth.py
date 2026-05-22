"""Shared Globus Auth helper (G23).

Both the Globus Compute executor (``globus_compute_executor.py``, core-side)
and the Globus Transfer step (``library/steps/globus_transfer_step.py``,
library-side) need to build a ``globus_sdk.GlobusApp`` in a consistent way.
This module is that single source of truth.

It lives in ``core/`` deliberately: ``core`` MUST NOT depend on ``library``,
so a helper consumed by both sides has to sit in ``core``. The module does
NOT import ``globus_sdk`` at module scope — the import is lazy (inside
:func:`build_globus_app`) and FAIL-LOUD, so importing this module never
forces the Globus dependency on a workflow that doesn't use Globus.

Auth modes
----------
``client_credentials`` (default — the workspace's primary mode)
    Builds a ``globus_sdk.ClientApp`` from a confidential-client
    ``client_id`` + ``client_secret`` pair. Credentials are resolved in
    a strict 3-tier precedence, highest first:

      1. **Explicit arguments** — ``client_id`` / ``client_secret``
         passed to :func:`build_globus_app`.
      2. **Environment variables** — ``$GLOBUS_COMPUTE_CLIENT_ID`` /
         ``$GLOBUS_COMPUTE_CLIENT_SECRET`` (the names
         ``globus_compute_sdk`` itself reads).
      3. **OS secure credential store (keyring)** — whatever
         ``globus_credentials.load_credentials()`` returns from the OS
         Keychain / Credential Locker / Secret Service. This is the
         lowest tier; it is only consulted when both tiers above came
         up empty. The ``keyring`` import is lazy: a caller who passes
         explicit credentials or sets the env vars never needs
         ``keyring`` installed, and a missing-``keyring`` ImportError
         is swallowed at this tier so it cannot mask the real
         "credentials not found" error below.

    Each of ``client_id`` and ``client_secret`` is resolved
    independently through the three tiers. If either is still missing
    after all three, this raises ``ComponentConfigurationError`` with a
    ``FAIL-FAST:`` message — a confidential client with no secret is
    never silently downgraded to interactive login.

``native``
    Builds a ``globus_sdk.UserApp`` (interactive / browser login). Used
    for developer workstations. ``client_id`` is optional here; when
    omitted the Globus-issued public client id is used by the SDK
    default.

Scope wiring
------------
``globus_sdk`` resource clients (``TransferClient``, and the web client
inside ``globus_compute_sdk.Client``) auto-register their own
``default_scope_requirements`` onto a ``GlobusApp`` when constructed with
``app=<the app>`` — verified against ``globus_sdk`` 4.5.0
(``BaseClient.attach_globus_app``). So passing ``app=`` is by itself
sufficient for a single-service app. The ``scopes`` argument here lets a
caller PRE-declare scope requirements anyway — the intended use is one
shared ``ClientApp`` that authorizes BOTH Compute and Transfer, where the
caller passes both scope strings up front so the very first token
acquisition covers both services. ``scopes`` is a flat list of scope
strings; they are grouped by Globus resource-server for the
``scope_requirements`` mapping the GlobusApp expects.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Iterable, List, Optional

from nanobrain.core.component_base import ComponentConfigurationError

if TYPE_CHECKING:  # pragma: no cover - typing only
    import globus_sdk


# Environment variable names. These match what ``globus_compute_sdk`` reads
# for its own confidential-client path, so an operator who already set
# them for the Globus Compute CLI does not need to set anything new.
ENV_CLIENT_ID = "GLOBUS_COMPUTE_CLIENT_ID"
ENV_CLIENT_SECRET = "GLOBUS_COMPUTE_CLIENT_SECRET"

VALID_AUTH_MODES = ("client_credentials", "native")


def _import_globus_sdk() -> "Any":
    """Lazy-import ``globus_sdk``; FAIL-LOUD with an actionable message."""
    try:
        import globus_sdk  # noqa: PLC0415 - intentional lazy import
    except ImportError as exc:
        raise ComponentConfigurationError(
            "FAIL-FAST: build_globus_app requires the 'globus_sdk' package, "
            "which is not installed. Install it with "
            "`pip install globus-sdk` (or `pip install globus-compute-sdk`, "
            "which depends on it). Globus auth is only needed for the "
            "GlobusComputeExecutor and GlobusTransferStep — workflows that "
            f"use neither never reach this code path. Underlying error: {exc}"
        ) from exc
    return globus_sdk


def _load_keyring_credentials() -> "tuple[Optional[str], Optional[str]]":
    """Tier-3 credential lookup: the OS secure store, via ``globus_credentials``.

    This is the lowest-precedence resolution tier. The ``keyring``
    dependency is intentionally soft here: a caller who supplies
    explicit credentials or sets the environment variables must never
    be forced to install ``keyring``. So a missing-``keyring``
    ImportError (surfaced as ``ComponentConfigurationError`` by
    ``globus_credentials._import_keyring``) is swallowed at this tier —
    it MUST NOT mask the real "credentials not found in any tier"
    FAIL-LOUD that fires downstream when args + env are also empty.

    The import of ``globus_credentials`` itself is local so that
    importing this module never pulls it in.
    """
    try:
        from nanobrain.core.distributed import globus_credentials
    except ImportError:  # pragma: no cover - module is in-tree
        return None, None
    try:
        return globus_credentials.load_credentials()
    except ComponentConfigurationError:
        # keyring not installed — tier-3 simply contributes nothing.
        return None, None
    except Exception:  # noqa: BLE001 - a broken keyring must not block tiers 1-2
        return None, None


def _scope_resource_server(scope: str) -> str:
    """Derive the Globus resource-server key a scope belongs to.

    GlobusApp's ``scope_requirements`` is a mapping keyed by resource
    server. Globus scope strings come in two shapes:

      * ``urn:globus:auth:scope:<resource_server>:<name>``
      * ``https://auth.globus.org/scopes/<resource_server>/<name>``

    Both forms embed the resource server; we extract it so callers can
    pass a flat list of scope strings and we group them correctly.
    Anything that does not match either shape is bucketed under its own
    full string — still valid (a one-entry list), just not grouped.
    """
    s = scope.strip()
    if s.startswith("urn:globus:auth:scope:"):
        rest = s[len("urn:globus:auth:scope:") :]
        return rest.split(":", 1)[0]
    marker = "://auth.globus.org/scopes/"
    if marker in s:
        rest = s.split(marker, 1)[1]
        return rest.split("/", 1)[0]
    return s


def _build_scope_requirements(
    globus_sdk: "Any", scopes: Optional[Iterable[str]]
) -> Optional["dict"]:
    """Group a flat list of scope strings into the GlobusApp mapping shape."""
    if not scopes:
        return None
    grouped: "dict[str, List[Any]]" = {}
    for scope in scopes:
        if not isinstance(scope, str) or not scope.strip():
            raise ComponentConfigurationError(
                "FAIL-FAST: build_globus_app 'scopes' must be a list of "
                f"non-empty scope strings; got {scope!r}"
            )
        rs = _scope_resource_server(scope)
        grouped.setdefault(rs, []).append(globus_sdk.Scope(scope))
    return grouped


def build_globus_app(
    *,
    auth_mode: str = "client_credentials",
    scopes: Optional[Iterable[str]] = None,
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    app_name: str = "nanobrain",
) -> "globus_sdk.GlobusApp":
    """Build a ``globus_sdk.GlobusApp`` for Compute and/or Transfer.

    Args:
        auth_mode: ``"client_credentials"`` (default) builds a confidential
            ``ClientApp``; ``"native"`` builds an interactive ``UserApp``.
        scopes: Optional flat list of Globus scope strings to pre-declare
            on the app's ``scope_requirements``. Callers that share one
            app across Compute + Transfer should pass both scope strings.
        client_id: Confidential-client id. ``client_credentials`` mode
            falls back to ``$GLOBUS_COMPUTE_CLIENT_ID``. Optional in
            ``native`` mode.
        client_secret: Confidential-client secret. ``client_credentials``
            mode falls back to ``$GLOBUS_COMPUTE_CLIENT_SECRET``.
        app_name: Human-readable app name recorded by Globus Auth.

    Returns:
        A configured ``globus_sdk.GlobusApp`` (``ClientApp`` or ``UserApp``).

    Raises:
        ComponentConfigurationError: ``globus_sdk`` not installed; unknown
            ``auth_mode``; or ``client_credentials`` mode with a missing
            ``client_id`` / ``client_secret``.
    """
    if auth_mode not in VALID_AUTH_MODES:
        raise ComponentConfigurationError(
            f"FAIL-FAST: build_globus_app unknown auth_mode {auth_mode!r}. "
            f"Valid values: {VALID_AUTH_MODES}."
        )

    globus_sdk = _import_globus_sdk()
    scope_requirements = _build_scope_requirements(globus_sdk, scopes)

    if auth_mode == "client_credentials":
        # 3-tier precedence: explicit args -> env vars -> keyring.
        # The keyring tier is only consulted when something is still
        # missing after args + env, so the lazy keyring import is
        # avoided entirely on the common (args/env) paths.
        resolved_id = client_id or os.environ.get(ENV_CLIENT_ID)
        resolved_secret = client_secret or os.environ.get(ENV_CLIENT_SECRET)
        if not resolved_id or not resolved_secret:
            keyring_id, keyring_secret = _load_keyring_credentials()
            resolved_id = resolved_id or keyring_id
            resolved_secret = resolved_secret or keyring_secret
        missing = []
        if not resolved_id:
            missing.append(f"client_id (or ${ENV_CLIENT_ID})")
        if not resolved_secret:
            missing.append(f"client_secret (or ${ENV_CLIENT_SECRET})")
        if missing:
            raise ComponentConfigurationError(
                "FAIL-FAST: build_globus_app auth_mode='client_credentials' "
                f"requires confidential-client credentials. Missing: "
                f"{', '.join(missing)}. Provide them in the component config "
                f"or export the environment variables ${ENV_CLIENT_ID} / "
                f"${ENV_CLIENT_SECRET}. A confidential client is never "
                "silently downgraded to interactive login."
            )
        return globus_sdk.ClientApp(
            app_name=app_name,
            client_id=resolved_id,
            client_secret=resolved_secret,
            scope_requirements=scope_requirements,
        )

    # auth_mode == "native": interactive browser login. globus_sdk's
    # UserApp requires a client_id (it cannot be derived). Resolve it
    # through the same 3-tier precedence: explicit arg -> env var ->
    # keyring. (native mode needs no client_secret.) FAIL-LOUD if absent.
    resolved_id = client_id or os.environ.get(ENV_CLIENT_ID)
    if not resolved_id:
        keyring_id, _ = _load_keyring_credentials()
        resolved_id = resolved_id or keyring_id
    if not resolved_id:
        raise ComponentConfigurationError(
            "FAIL-FAST: build_globus_app auth_mode='native' requires a "
            f"client_id (or ${ENV_CLIENT_ID}) to set up the interactive "
            "login client. globus_sdk.UserApp cannot derive one. Register "
            "a native app at https://app.globus.org/settings/developers and "
            "supply its client_id."
        )
    kwargs: "dict[str, Any]" = {"app_name": app_name, "client_id": resolved_id}
    if scope_requirements:
        kwargs["scope_requirements"] = scope_requirements
    # Request a REFRESH token (offline access). globus_sdk's default
    # (request_refresh_tokens=False) persists an online-only access token that
    # expires in ~2 days with no way to renew — fatal for an unattended /
    # default install path whose tokens silently die days after setup. With a
    # refresh token globus_sdk renews automatically while it stays valid. This
    # mirrors the apecx-side ``apecx-globus-setup login`` fix; both the login
    # flow AND the run-time app must request refresh tokens so persisted tokens
    # keep working at transfer time.
    kwargs["config"] = globus_sdk.GlobusAppConfig(request_refresh_tokens=True)
    return globus_sdk.UserApp(**kwargs)


__all__ = [
    "build_globus_app",
    "ENV_CLIENT_ID",
    "ENV_CLIENT_SECRET",
    "VALID_AUTH_MODES",
]
