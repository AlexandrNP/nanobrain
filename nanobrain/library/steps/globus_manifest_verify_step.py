"""GlobusManifestVerifyStep (G127) — fail-loud pre-transfer source existence check.

A nanobrain ``BaseStep`` that verifies every source file/directory named in a
transfer manifest actually exists on the SOURCE Globus collection BEFORE a
:class:`~nanobrain.library.steps.globus_transfer_step.GlobusTransferStep`
submits the transfer. It turns two silent / late failure shapes —

  * "transfer reported SUCCEEDED but moved zero files" (empty/skip sync), and
  * "cryptic per-file failure surfaced only after submit + a long poll" —

into an EARLY, actionable, FAIL-LOUD error that names exactly which source
paths are missing, before any transfer task is created.

Why a separate step (not a flag on GlobusTransferStep)
------------------------------------------------------
Composition over a monolith. Verification is independently useful (any staging
workflow wants "do my inputs exist before I pay for a multi-minute transfer?"),
and wiring it as ``verify -> transfer`` with a ``DirectLink`` makes the gate an
explicit, auditable edge in the workflow DAG. On success this step PASSES THE
MANIFEST THROUGH so the downstream transfer step receives the exact same
``items`` it validated.

Input / output contract
------------------------
Input payload (the same ``items`` shape ``GlobusTransferStep`` consumes)::

    {"items": [{"source_path": "...", "dest_path": "..."}, ...]}

On success returns::

    {"verified_manifest": {"items": [...same items...]}}

The single output key ``verified_manifest`` MUST be the name of this step's
output data unit in the wrapper YAML — the framework silently drops a returned
key that matches no declared output data unit (see the ``nanobrain-step-authoring``
skill). Downstream, a ``DirectLink`` carries ``verified_manifest`` into the
transfer step's input data unit; the transfer step's envelope-unwrap then sees
``{"items": [...]}`` again.

FAIL-LOUD discipline
--------------------
  * ``globus_sdk`` missing -> ``ComponentConfigurationError``.
  * auth failure -> ``ComponentConfigurationError``.
  * malformed input payload (no transfer items) -> ``ComponentConfigurationError``.
  * ANY source path missing on the source endpoint ->
    ``ComponentConfigurationError`` listing EVERY missing path.
  * a non-404 Globus error while listing (auth / connectivity / endpoint
    path-restriction) is surfaced FAIL-LOUD and is NOT silently treated as
    "file missing".
"""

from __future__ import annotations

import logging
import posixpath
from typing import Any, Dict, List, Optional

from pydantic import ConfigDict, Field, model_validator

from nanobrain.core.component_base import ComponentConfigurationError
from nanobrain.core.distributed.globus_auth import build_globus_app
from nanobrain.core.step import BaseStep, StepConfig

logger = logging.getLogger(__name__)

# The Globus Transfer scope — operation_ls lives under the same Transfer API
# scope as submit_transfer, so a verify->transfer pair shares one token grant.
_TRANSFER_SCOPE_ALL = "urn:globus:auth:scope:transfer.api.globus.org:all"

# The single output key this step returns. The wrapper YAML's output data unit
# MUST be named this, or the framework drops the value silently.
OUTPUT_MANIFEST_KEY = "verified_manifest"


class GlobusManifestVerifyStepConfig(StepConfig):
    """Configuration for :class:`GlobusManifestVerifyStep`.

    ``extra='forbid'`` (workspace rule): a YAML typo raises at config load
    rather than silently using a default. Shares the SOURCE-side auth fields
    with ``GlobusTransferStep`` (this step only reads the source — it needs no
    ``dest_endpoint_id``).
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=False)

    # Framework tracking attribute populated by ConfigBase.from_config —
    # declared so extra='forbid' doesn't reject it.
    source_path: Optional[str] = Field(default=None)

    source_endpoint_id: str = Field(
        ...,
        description="Globus collection UUID whose source files are verified.",
    )
    auth_mode: str = Field(
        default="client_credentials",
        description=(
            "'client_credentials' (default) uses a confidential client; "
            "'native' uses an interactive browser login."
        ),
    )
    client_id: Optional[str] = Field(
        default=None,
        description=(
            "Confidential-client id. Falls back to $GLOBUS_COMPUTE_CLIENT_ID when omitted."
        ),
    )
    client_secret: Optional[str] = Field(
        default=None,
        description=(
            "Confidential-client secret. Falls back to $GLOBUS_COMPUTE_CLIENT_SECRET when omitted."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _strip_framework_keys(cls, data: Any) -> Any:
        """Drop the framework-loader-only ``class`` key (same pattern as
        ``GlobusTransferStepConfig``) so ``extra='forbid'`` keeps typo
        protection for every OTHER field."""
        if isinstance(data, dict):
            data.pop("class", None)
        return data


class GlobusManifestVerifyStep(BaseStep):
    """Verify source paths exist on a Globus collection before a transfer.

    Expected ``process()`` input payload::

        {"items": [{"source_path": "/data/in.fasta", "dest_path": "..."}, ...]}

    (``dest_path`` is carried through untouched — this step never reads the
    destination; only ``source_path`` is verified.)

    When wired into a workflow the payload arrives wrapped as
    ``{<input_du_name>: <payload>}`` — the step self-unwraps that envelope.

    Return shape (passthrough so the downstream transfer step gets the same
    validated manifest)::

        {"verified_manifest": {"items": [...]}}
    """

    COMPONENT_TYPE: str = "globus_manifest_verify_step"
    REQUIRED_CONFIG_FIELDS = ["name", "source_endpoint_id"]

    @classmethod
    def _get_config_class(cls):
        return GlobusManifestVerifyStepConfig

    def _init_from_config(
        self,
        config: GlobusManifestVerifyStepConfig,
        component_config: Dict[str, Any],
        dependencies: Dict[str, Any],
    ) -> None:
        super()._init_from_config(config, component_config, dependencies)
        # No network / auth work at init: a malformed config fails at
        # config-load; Globus reachability is a runtime concern verified
        # (and FAIL-LOUD) in process().
        self._gmv_config = config

    @property
    def verify_config(self) -> GlobusManifestVerifyStepConfig:
        """The resolved config this step verifies against."""
        return self._gmv_config

    def _unwrap_trigger_envelope(self, input_data: Any) -> Any:
        """Strip the ``{<input_du_name>: payload}`` trigger envelope.

        Mirrors ``GlobusTransferStep._unwrap_trigger_envelope``: the real
        payload carries an ``items`` key. A single-key dict whose lone key is
        NOT ``items`` and whose value is itself a dict is the trigger envelope
        (the key is the input data unit's name). Anything else passes through.
        """
        if not isinstance(input_data, dict) or len(input_data) != 1:
            return input_data
        (only_key,) = input_data.keys()
        if only_key == "items":
            return input_data
        value = input_data[only_key]
        if not isinstance(value, dict):
            return input_data
        logger.debug(
            "GlobusManifestVerifyStep %r: unwrapped trigger envelope key %r",
            self.name,
            only_key,
        )
        return value

    @staticmethod
    def _coerce_items(payload: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extract + validate the list of ``{source_path, dest_path}`` items.

        FAIL-LOUD on a missing / empty / malformed ``items`` list — verifying a
        manifest with nothing in it is a configuration error, not a no-op.
        Only ``source_path`` is required to be present and non-empty here
        (``dest_path`` is carried through for the downstream transfer step but
        is irrelevant to source verification).
        """
        items = payload.get("items")
        if not isinstance(items, list) or not items:
            raise ComponentConfigurationError(
                "FAIL-FAST: GlobusManifestVerifyStep input must carry a "
                "non-empty 'items' list of {source_path, dest_path} dicts; got "
                f"{items!r}"
            )
        coerced: List[Dict[str, str]] = []
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                raise ComponentConfigurationError(
                    f"FAIL-FAST: GlobusManifestVerifyStep items[{idx}] must be a "
                    f"dict, got {type(item).__name__}"
                )
            src = item.get("source_path")
            if not isinstance(src, str) or not src:
                raise ComponentConfigurationError(
                    f"FAIL-FAST: GlobusManifestVerifyStep items[{idx}] missing a "
                    "non-empty 'source_path' string"
                )
            coerced.append(dict(item))
        return coerced

    @staticmethod
    def _classify_ls_error(exc: Any) -> str:
        """Map a non-404 ``operation_ls`` Globus error to an actionable hint.

        The raw Globus error is always appended by the caller; this adds the
        "what do I DO about it" sentence so an operator isn't left decoding
        GridFTP/HTTP codes. Distinguishes the common access-denied shapes:

          * 401/403 (PermissionDenied / no effective ACL) — the transfer
            identity is not authorized for this path. For Group-gated data the
            identity must be a MEMBER of the granting Globus Group (ask the
            data steward to add the confidential client's
            ``<client_id>@clients.auth.globus.org`` identity).
          * 500 "Path not allowed" — the path is OUTSIDE this collection's
            allowed namespace (a different collection/guest-collection serves
            it); using a different ``source_endpoint_id`` is required, not a
            credential change.
          * "not currently connected" — the source collection (often a Globus
            Connect Personal endpoint) is offline; start it.
        """
        status = getattr(exc, "http_status", None)
        text = f"{getattr(exc, 'code', '')} {exc}".lower()
        if status in (401, 403) or "permissiondenied" in text or "no effective acl" in text:
            return (
                "Authorization error: the transfer identity is not authorized "
                "for this path. If the data is gated by a Globus Group, the "
                "identity must be ADDED as a member of that Group (an admin "
                "action — ask the data steward); credentials alone are not "
                "enough."
            )
        if "path not allowed" in text or "path_not_allowed" in text:
            return (
                "Path-restriction error: this path is outside the collection's "
                "allowed namespace, so a DIFFERENT collection serves it. Point "
                "source_endpoint_id at the collection that exposes this path "
                "(a credential change will not help)."
            )
        if "not currently connected" in text or "endpoint is not active" in text:
            return (
                "The source collection is offline/inactive (e.g. a Globus "
                "Connect Personal endpoint that isn't running). Start it and "
                "retry."
            )
        return (
            "Auth / connectivity error reaching the source collection. Verify "
            "credentials, endpoint UUID, and network reachability."
        )

    async def process(self, input_data: Any, **kwargs) -> Dict[str, Any]:
        """List each source parent dir once and assert every item is present.

        Groups source paths by parent directory so N files under the same
        directory cost ONE ``operation_ls`` call, not N. A 404 on a parent
        means every item under it is missing; any other Globus error is a
        connectivity/auth/path-restriction failure and is surfaced FAIL-LOUD
        rather than being miscounted as "file missing".
        """
        import asyncio

        cfg = self._gmv_config
        payload = self._unwrap_trigger_envelope(input_data)
        if not isinstance(payload, dict):
            raise ComponentConfigurationError(
                "FAIL-FAST: GlobusManifestVerifyStep input must be a dict with "
                f"an 'items' list, got {type(payload).__name__}"
            )
        items = self._coerce_items(payload)

        try:
            import globus_sdk  # noqa: PLC0415 - intentional lazy import
        except ImportError as exc:
            raise ComponentConfigurationError(
                "FAIL-FAST: GlobusManifestVerifyStep requires the 'globus_sdk' "
                "package, which is not installed. Install it with "
                "`pip install globus-sdk`. This step is intentionally "
                f"Globus-coupled. Underlying error: {exc}"
            ) from exc

        try:
            app = build_globus_app(
                auth_mode=cfg.auth_mode,
                scopes=[_TRANSFER_SCOPE_ALL],
                client_id=cfg.client_id,
                client_secret=cfg.client_secret,
                app_name="nanobrain-globus-manifest-verify-step",
            )
            transfer_client = globus_sdk.TransferClient(app=app)
        except ComponentConfigurationError:
            raise
        except Exception as exc:
            raise ComponentConfigurationError(
                "FAIL-FAST: GlobusManifestVerifyStep failed to build the Globus "
                f"auth app / TransferClient (auth_mode={cfg.auth_mode!r}): "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        # Group basenames by parent directory. Normalize trailing slashes so a
        # directory item ("/data/db/") and a file item are treated uniformly:
        # we ls the parent and check the basename is present (as file OR dir).
        parent_to_names: Dict[str, List[str]] = {}
        for item in items:
            src = item["source_path"].rstrip("/")
            parent = posixpath.dirname(src) or "/"
            name = posixpath.basename(src)
            parent_to_names.setdefault(parent, []).append(name)

        missing: List[str] = []
        for parent, expected_names in parent_to_names.items():
            try:
                listing = await asyncio.to_thread(
                    lambda p=parent: list(
                        transfer_client.operation_ls(cfg.source_endpoint_id, path=p)
                    )
                )
            except globus_sdk.TransferAPIError as exc:
                if exc.http_status == 404:
                    # Parent dir absent -> every expected file under it missing.
                    missing.extend(f"{parent}/{n}" for n in expected_names)
                    continue
                # Classify the non-404 error so the operator gets an ACTIONABLE
                # message, not a generic "auth/connectivity" bucket. These are
                # NOT missing-file conditions — the file may well exist but the
                # transfer identity can't see it.
                hint = self._classify_ls_error(exc)
                raise ComponentConfigurationError(
                    "FAIL-FAST: GlobusManifestVerifyStep could not list source "
                    f"directory {parent!r} on endpoint {cfg.source_endpoint_id} "
                    "(NOT a missing-file condition). "
                    f"{hint} Underlying Globus error: {exc.code}: {exc}"
                ) from exc
            except globus_sdk.GlobusError as exc:
                raise ComponentConfigurationError(
                    "FAIL-FAST: GlobusManifestVerifyStep failed to list source "
                    f"directory {parent!r} on endpoint "
                    f"{cfg.source_endpoint_id}: {type(exc).__name__}: {exc}"
                ) from exc

            present = {entry["name"] for entry in listing}
            for name in expected_names:
                if name not in present:
                    missing.append(f"{parent}/{name}")

        if missing:
            raise ComponentConfigurationError(
                "FAIL-FAST: GlobusManifestVerifyStep found "
                f"{len(missing)} of {len(items)} source path(s) MISSING on "
                f"endpoint {cfg.source_endpoint_id}. Refusing to proceed to "
                "transfer (a transfer of missing sources moves zero or a "
                "partial subset of files — the silent-failure shape this step "
                "exists to refuse). Missing:\n  " + "\n  ".join(sorted(missing))
            )

        self.nb_logger.info(
            "GlobusManifestVerifyStep %r: verified %d source path(s) present "
            "on endpoint %s across %d directory listing(s)",
            self.name,
            len(items),
            cfg.source_endpoint_id,
            len(parent_to_names),
        )
        # Passthrough: hand the SAME validated manifest to the downstream
        # transfer step. The output data unit MUST be named OUTPUT_MANIFEST_KEY.
        return {OUTPUT_MANIFEST_KEY: {"items": items}}


__all__ = [
    "GlobusManifestVerifyStep",
    "GlobusManifestVerifyStepConfig",
    "OUTPUT_MANIFEST_KEY",
]
