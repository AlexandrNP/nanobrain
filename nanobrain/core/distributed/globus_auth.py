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
    ``client_id`` + ``client_secret`` pair. Credentials come from the
    explicit arguments, else from the environment variables
    ``$GLOBUS_COMPUTE_CLIENT_ID`` / ``$GLOBUS_COMPUTE_CLIENT_SECRET``
    (the names ``globus_compute_sdk`` itself reads). If either is
    missing this raises ``ComponentConfigurationError`` with a
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
        rest = s[len("urn:globus:auth:scope:"):]
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
        resolved_id = client_id or os.environ.get(ENV_CLIENT_ID)
        resolved_secret = client_secret or os.environ.get(ENV_CLIENT_SECRET)
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
    # UserApp requires a client_id (it cannot be derived). Fall back to
    # $GLOBUS_COMPUTE_CLIENT_ID so a developer who set it for the Globus
    # Compute CLI does not need to set anything new; FAIL-LOUD if absent.
    resolved_id = client_id or os.environ.get(ENV_CLIENT_ID)
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
    return globus_sdk.UserApp(**kwargs)


__all__ = [
    "build_globus_app",
    "ENV_CLIENT_ID",
    "ENV_CLIENT_SECRET",
    "VALID_AUTH_MODES",
]
