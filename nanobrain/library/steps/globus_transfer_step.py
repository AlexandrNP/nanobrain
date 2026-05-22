"""GlobusTransferStep (G28) — stage data between filesystems via Globus Transfer.

A nanobrain ``BaseStep`` that submits a Globus Transfer task to move one or
more files/directories between two Globus collections (endpoints) and polls
it to completion. Used to stage workflow inputs onto a remote HPC filesystem
before a ``GlobusComputeExecutor`` step runs there, and to stage results
back afterwards.

Framework-native packaging (workspace + nanobrain rules)
--------------------------------------------------------
  * Subclasses ``BaseStep``; implements ``async def process``; never
    overrides ``execute()`` (nanobrain Method Responsibility Matrix).
  * Config extends ``StepConfig`` with ``extra='forbid'`` so a YAML typo
    fails at config-load rather than silently using a default.
  * ``from_config`` only — direct construction is framework-forbidden.
  * Strips the framework-loader-only ``class`` key in a ``model_validator``
    (same pattern as ``RheaFileToolStepConfig``).
  * Self-unwraps the ``{<input_du_name>: payload}`` trigger envelope so it
    works both as a direct ``process(payload)`` call and inside a workflow
    cascade — same discriminator approach as ``RheaFileToolStep``.

Auth
----
Builds the ``globus_sdk.GlobusApp`` via the shared G23 helper
(``nanobrain.core.distributed.globus_auth.build_globus_app``) so auth is
consistent with ``GlobusComputeExecutor``. One confidential ``ClientApp``
can authorize both Compute and Transfer.

FAIL-LOUD discipline
--------------------
  * ``globus_sdk`` missing -> ``ComponentConfigurationError``.
  * auth failure -> ``ComponentConfigurationError``.
  * malformed input payload (no transfer items) -> ``ComponentConfigurationError``.
  * Globus reports the transfer task ``FAILED`` -> ``ComponentConfigurationError``
    carrying Globus's fatal-error detail.
  * the poll times out before the task reaches a terminal state ->
    ``ComponentConfigurationError``.
  A "submitted" transfer whose final status is anything other than
  ``SUCCEEDED`` is the silent-failure shape this step exists to refuse.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from pydantic import ConfigDict, Field, model_validator

from nanobrain.core.component_base import ComponentConfigurationError
from nanobrain.core.distributed.globus_auth import build_globus_app
from nanobrain.core.step import BaseStep, StepConfig

logger = logging.getLogger(__name__)

# The canonical Globus Transfer scope (verified against globus_sdk 4.5.0:
# `globus_sdk.scopes.TransferScopes.all`). Declared here as a constant so
# the value is auditable; build_globus_app pre-registers it on the app.
_TRANSFER_SCOPE_ALL = "urn:globus:auth:scope:transfer.api.globus.org:all"


class GlobusTransferStepConfig(StepConfig):
    """Configuration for :class:`GlobusTransferStep`.

    ``extra='forbid'`` (workspace rule): a YAML typo raises at config
    load rather than silently using a default.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=False)

    # Framework tracking attribute populated by ConfigBase.from_config —
    # declared so extra='forbid' doesn't reject it.
    source_path: Optional[str] = Field(default=None)

    source_endpoint_id: str = Field(
        ...,
        description="Globus collection UUID the files are transferred FROM.",
    )
    dest_endpoint_id: str = Field(
        ...,
        description="Globus collection UUID the files are transferred TO.",
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
            "Confidential-client id. Falls back to "
            "$GLOBUS_COMPUTE_CLIENT_ID when omitted."
        ),
    )
    client_secret: Optional[str] = Field(
        default=None,
        description=(
            "Confidential-client secret. Falls back to "
            "$GLOBUS_COMPUTE_CLIENT_SECRET when omitted."
        ),
    )
    sync_level: Optional[str] = Field(
        default="checksum",
        description=(
            "Globus TransferData sync_level: exists | size | mtime | "
            "checksum. 'checksum' is the safest (and the default)."
        ),
    )
    verify_checksum: bool = Field(
        default=True,
        description="Verify a checksum of each file after transfer.",
    )
    poll_timeout_seconds: float = Field(
        default=600.0,
        gt=0.0,
        description=(
            "Maximum wall time to poll for the transfer to reach a terminal "
            "state before raising."
        ),
    )
    poll_interval_seconds: float = Field(
        default=10.0,
        gt=0.0,
        description="Seconds between transfer-task status polls.",
    )
    transfer_label: str = Field(
        default="nanobrain-globus-transfer",
        description="Human-readable label recorded on the Globus task.",
    )

    @model_validator(mode="before")
    @classmethod
    def _strip_framework_keys(cls, data: Any) -> Any:
        """Drop the framework-loader-only ``class`` key.

        When this config is loaded from a step YAML carrying a top-level
        ``class:`` (the auto-delegation target), the loader passes that key
        through. ``extra='forbid'`` would otherwise reject it; stripping it
        here keeps typo protection for every OTHER field intact.
        """
        if isinstance(data, dict):
            data.pop("class", None)
        return data


class GlobusTransferStep(BaseStep):
    """Stage files between two Globus collections via the Globus Transfer API.

    Expected ``process()`` input payload::

        {
            "items": [
                {"source_path": "/data/in.fasta", "dest_path": "/scratch/in.fasta"},
                {"source_path": "/data/db/",      "dest_path": "/scratch/db/"},
            ]
        }

    When wired into a workflow the payload arrives wrapped as
    ``{<input_du_name>: <payload>}`` — the step self-unwraps that envelope.

    Return shape::

        {
            "task_id": "<globus task uuid>",
            "status": "SUCCEEDED",
            "items_transferred": 2,
            "source_endpoint_id": "<uuid>",
            "dest_endpoint_id": "<uuid>",
        }
    """

    COMPONENT_TYPE: str = "globus_transfer_step"
    REQUIRED_CONFIG_FIELDS = ["name", "source_endpoint_id", "dest_endpoint_id"]

    @classmethod
    def _get_config_class(cls):
        return GlobusTransferStepConfig

    def _init_from_config(
        self,
        config: GlobusTransferStepConfig,
        component_config: Dict[str, Any],
        dependencies: Dict[str, Any],
    ) -> None:
        super()._init_from_config(config, component_config, dependencies)
        # Stash the typed config — process() reads every field off it.
        # No network / auth work at init: a malformed config fails at
        # config-load; Globus reachability is a runtime concern verified
        # (and FAIL-LOUD) in process().
        self._gts_config = config

    @property
    def transfer_config(self) -> GlobusTransferStepConfig:
        """The resolved GlobusTransferStepConfig this step dispatches with."""
        return self._gts_config

    def _unwrap_trigger_envelope(self, input_data: Any) -> Any:
        """Strip the ``{<input_du_name>: payload}`` trigger envelope.

        Mirrors ``RheaFileToolStep._unwrap_trigger_envelope``: the real
        payload carries an ``items`` key. A single-key dict whose lone key
        is NOT ``items`` and whose value is itself a dict is the trigger
        envelope (the key is the input data unit's name). Anything else
        passes through untouched.
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
            "GlobusTransferStep %r: unwrapped trigger envelope key %r",
            self.name,
            only_key,
        )
        return value

    @staticmethod
    def _coerce_items(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract + validate the list of transfer items.

        Each item is ``{source_path, dest_path}`` plus an optional
        ``recursive: bool`` (default False). ``recursive: true`` transfers a
        whole directory tree (Globus requires it for directory sources;
        without it a directory source fails). FAIL-LOUD on a missing / empty
        / malformed ``items`` list — a transfer step with nothing to transfer
        is a configuration error, not a no-op.
        """
        items = payload.get("items")
        if not isinstance(items, list) or not items:
            raise ComponentConfigurationError(
                "FAIL-FAST: GlobusTransferStep input must carry a non-empty "
                "'items' list of {source_path, dest_path} dicts; got "
                f"{items!r}"
            )
        coerced: List[Dict[str, Any]] = []
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                raise ComponentConfigurationError(
                    f"FAIL-FAST: GlobusTransferStep items[{idx}] must be a "
                    f"dict, got {type(item).__name__}"
                )
            src = item.get("source_path")
            dst = item.get("dest_path")
            if not isinstance(src, str) or not src:
                raise ComponentConfigurationError(
                    f"FAIL-FAST: GlobusTransferStep items[{idx}] missing a "
                    "non-empty 'source_path' string"
                )
            if not isinstance(dst, str) or not dst:
                raise ComponentConfigurationError(
                    f"FAIL-FAST: GlobusTransferStep items[{idx}] missing a "
                    "non-empty 'dest_path' string"
                )
            recursive = item.get("recursive", False)
            if not isinstance(recursive, bool):
                raise ComponentConfigurationError(
                    f"FAIL-FAST: GlobusTransferStep items[{idx}] 'recursive' "
                    f"must be a bool, got {type(recursive).__name__}"
                )
            coerced.append(
                {"source_path": src, "dest_path": dst, "recursive": recursive}
            )
        return coerced

    async def process(self, input_data: Any, **kwargs) -> Dict[str, Any]:
        """Submit a Globus Transfer task and poll it to completion.

        See the class docstring for the input / output shapes.
        """
        import asyncio

        cfg = self._gts_config
        payload = self._unwrap_trigger_envelope(input_data)
        if not isinstance(payload, dict):
            raise ComponentConfigurationError(
                "FAIL-FAST: GlobusTransferStep input must be a dict with an "
                f"'items' list, got {type(payload).__name__}"
            )
        items = self._coerce_items(payload)

        # Lazy-import globus_sdk. FAIL-LOUD with an actionable message:
        # this step is intentionally Globus-coupled.
        try:
            import globus_sdk  # noqa: PLC0415 - intentional lazy import
        except ImportError as exc:
            raise ComponentConfigurationError(
                "FAIL-FAST: GlobusTransferStep requires the 'globus_sdk' "
                "package, which is not installed. Install it with "
                "`pip install globus-sdk`. This step is intentionally "
                f"Globus-coupled. Underlying error: {exc}"
            ) from exc

        # Build the GlobusApp via the shared G23 helper (Transfer scope).
        try:
            app = build_globus_app(
                auth_mode=cfg.auth_mode,
                scopes=[_TRANSFER_SCOPE_ALL],
                client_id=cfg.client_id,
                client_secret=cfg.client_secret,
                app_name="nanobrain-globus-transfer-step",
            )
        except ComponentConfigurationError:
            raise
        except Exception as exc:
            raise ComponentConfigurationError(
                "FAIL-FAST: GlobusTransferStep failed to build the Globus "
                f"auth app (auth_mode={cfg.auth_mode!r}): {exc}"
            ) from exc

        try:
            transfer_client = globus_sdk.TransferClient(app=app)
        except Exception as exc:
            raise ComponentConfigurationError(
                "FAIL-FAST: GlobusTransferStep failed to construct the "
                f"globus_sdk.TransferClient (auth/connectivity issue): {exc}"
            ) from exc

        # Build the TransferData and add every item.
        transfer_data = globus_sdk.TransferData(
            source_endpoint=cfg.source_endpoint_id,
            destination_endpoint=cfg.dest_endpoint_id,
            label=cfg.transfer_label,
            sync_level=cfg.sync_level,
            verify_checksum=cfg.verify_checksum,
        )
        for item in items:
            transfer_data.add_item(
                item["source_path"],
                item["dest_path"],
                recursive=item.get("recursive", False),
            )

        self.nb_logger.info(
            "GlobusTransferStep %r: submitting transfer of %d item(s) "
            "%s -> %s",
            self.name,
            len(items),
            cfg.source_endpoint_id,
            cfg.dest_endpoint_id,
        )

        # submit_transfer is a blocking HTTP call — run it off the event
        # loop. globus_sdk raises GlobusAPIError on a bad endpoint /
        # permission / auth problem; surface it FAIL-LOUD.
        try:
            submit_result = await asyncio.to_thread(
                transfer_client.submit_transfer, transfer_data
            )
        except globus_sdk.GlobusError as exc:
            raise ComponentConfigurationError(
                "FAIL-FAST: GlobusTransferStep failed to submit the transfer "
                f"task ({cfg.source_endpoint_id} -> {cfg.dest_endpoint_id}): "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        task_id = submit_result["task_id"]
        self.nb_logger.info(
            "GlobusTransferStep %r: submitted task_id=%s, polling "
            "(timeout=%ss, interval=%ss)",
            self.name,
            task_id,
            cfg.poll_timeout_seconds,
            cfg.poll_interval_seconds,
        )

        # Poll to a terminal state. globus_sdk's task_wait blocks; call it
        # in a thread, in a loop, so the total wait is bounded by
        # poll_timeout_seconds and the event loop is never blocked.
        deadline = asyncio.get_running_loop().time() + cfg.poll_timeout_seconds
        final_status: Optional[str] = None
        while True:
            try:
                done = await asyncio.to_thread(
                    transfer_client.task_wait,
                    task_id,
                    timeout=cfg.poll_interval_seconds,
                    polling_interval=cfg.poll_interval_seconds,
                )
            except globus_sdk.GlobusError as exc:
                raise ComponentConfigurationError(
                    "FAIL-FAST: GlobusTransferStep failed while polling "
                    f"transfer task {task_id}: {type(exc).__name__}: {exc}"
                ) from exc

            if done:
                task_info = await asyncio.to_thread(
                    transfer_client.get_task, task_id
                )
                final_status = task_info.get("status")
                break

            if asyncio.get_running_loop().time() >= deadline:
                raise ComponentConfigurationError(
                    "FAIL-FAST: GlobusTransferStep transfer task "
                    f"{task_id} did not reach a terminal state within "
                    f"poll_timeout_seconds={cfg.poll_timeout_seconds}. The "
                    "transfer may still be running on the Globus service — "
                    "increase poll_timeout_seconds or check the task in the "
                    "Globus web app."
                )

        if final_status != "SUCCEEDED":
            # A submitted transfer whose terminal status is not SUCCEEDED
            # is the silent-failure shape this step exists to refuse.
            fatal_detail = ""
            try:
                task_info = await asyncio.to_thread(
                    transfer_client.get_task, task_id
                )
                fatal_detail = task_info.get("fatal_error") or ""
            except globus_sdk.GlobusError:
                pass
            raise ComponentConfigurationError(
                "FAIL-FAST: GlobusTransferStep transfer task "
                f"{task_id} ended with status={final_status!r} "
                f"(expected 'SUCCEEDED'). Fatal error detail: "
                f"{fatal_detail!r}"
            )

        self.nb_logger.info(
            "GlobusTransferStep %r: transfer task %s SUCCEEDED (%d item(s))",
            self.name,
            task_id,
            len(items),
        )
        return {
            "task_id": task_id,
            "status": final_status,
            "items_transferred": len(items),
            "source_endpoint_id": cfg.source_endpoint_id,
            "dest_endpoint_id": cfg.dest_endpoint_id,
        }


__all__ = ["GlobusTransferStep", "GlobusTransferStepConfig"]
