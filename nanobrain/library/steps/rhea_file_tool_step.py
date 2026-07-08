"""RheaFileToolStep — run any Rhea file-input Galaxy tool over MCP.

This step generalizes the ad-hoc proof script ``rhea/scripts/run_muscle_e2e.py``
into a proper nanobrain ``BaseStep``. It drives the full Rhea file-input
protocol for ANY Galaxy ``type="data"`` tool (MUSCLE is just one
configuration of it):

  1. Stage the input file bytes into the Rhea server via its redis-direct
     HTTP ``POST {base}/upload`` endpoint — this returns a ``key``
     (a ``meta:<uuid>`` string) that names the staged file.
  2. ``find_tools`` over MCP with a semantic query so the Rhea server
     populates the session-scoped tool catalog.
  3. ``tools/call`` the named tool with ``{file_input_arg: key,
     **static_tool_args}``.
  4. Parse Rhea's ``RheaOutput`` JSON, fetch each requested output
     file back via ``GET {base}/download?key=<key>``, decode to text.
  5. Best-effort ``POST {base}/delete?key=<key>`` for the input key and
     every output key so the server's redis store does not grow
     run-over-run.

This is a THIN HTTP client: it imports no rhea-side, object-store, or
pickle libraries. The rhea server owns the redis-backed file transport;
this step only speaks HTTP + MCP to it. The base URL is derived from the
MCP URL (strip the trailing ``/mcp/``) or set explicitly via
``http_base_url``.

Honesty / silent-failure discipline (workspace CLAUDE.md):

  - ``parse_tool_call_result`` raises ``ComponentConfigurationError``
    on the MCP ``isError=True`` envelope — that is the FAIL-LOUD path.
  - A non-zero ``return_code`` raises.
  - A "successful" call that produced ZERO usable output files raises:
    a green call with no output is the silent-failure shape this step
    exists to refuse.
  - An HTTP upload/download failure raises (``raise_for_status``); only
    the ``delete`` eviction is best-effort (teardown must never fail the
    call).

Framework-native packaging:
  - Subclasses ``BaseStep``; implements ``async def process``; never
    overrides ``execute()`` (nanobrain Method Responsibility Matrix).
  - Config extends ``StepConfig``; ``extra='forbid'`` so YAML typos
    fail at load time (workspace rule).
  - ``from_config`` only — direct construction is framework-forbidden.
  - Self-unwraps the ``{<input_du_name>: payload}`` trigger envelope
    the same way ``ToolExecutionStep`` does, so it works both as a
    direct ``process(payload)`` call and inside a workflow cascade.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

import httpx
from pydantic import ConfigDict, Field, model_validator

from nanobrain.core.component_base import ComponentConfigurationError
from nanobrain.core.step import BaseStep, StepConfig
from nanobrain.core.unified_tool_descriptor import UnifiedToolDescriptor
from nanobrain.library.tools._mcp_transport import (
    MCPTransport,
    parse_tool_call_result,
)
from nanobrain.library.tools.rhea_discovery import RheaMCPDiscovery
from nanobrain.library.tools.tool_discovery import find_and_establish_tool

logger = logging.getLogger(__name__)


class RheaFileToolStepConfig(StepConfig):
    """Configuration for :class:`RheaFileToolStep`.

    ``extra='forbid'`` (workspace rule): a YAML typo raises at config
    load rather than silently using a default.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=False)

    # Framework tracking attribute populated by ConfigBase.from_config —
    # declared so extra='forbid' doesn't reject it.
    source_path: str | None = Field(default=None)

    mcp_url: str = Field(
        default="http://localhost:3001/mcp/",
        description="The Rhea MCP streamable-HTTP endpoint.",
    )
    http_base_url: str | None = Field(
        default=None,
        description=(
            "Base URL for the Rhea server's redis-direct file HTTP "
            "endpoints (/upload, /download, /delete). When None, it is "
            "derived from mcp_url by stripping the trailing '/mcp/' "
            "(e.g. 'http://localhost:3001/mcp/' -> 'http://localhost:3001')."
        ),
    )
    tool_name: str = Field(
        ...,
        description="The Rhea/Galaxy tool name to call (e.g. 'muscle').",
    )
    find_tools_query: str = Field(
        ...,
        description=(
            "Semantic query passed to Rhea's find_tools so the tool is "
            "surfaced into the session catalog before tools/call."
        ),
    )
    file_input_arg: str = Field(
        ...,
        description=(
            "Name of the tool argument that receives the staged file's "
            "key (e.g. 'input_seqs' for MUSCLE)."
        ),
    )
    static_tool_args: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Other tool arguments passed verbatim in tools/call "
            "(e.g. {diags: false, run: '16', cluster: 'upgmb', "
            "outputFormat: 'fasta'} for MUSCLE)."
        ),
    )
    output_file_args: List[str] = Field(
        default_factory=list,
        description=(
            "Names of output files to fetch back from the server. "
            "Empty list = fetch all files in the result."
        ),
    )
    timeout_seconds: float = Field(
        default=900.0,
        gt=0.0,
        description=(
            "Per-call MCP + HTTP timeout. Galaxy tools can be slow; 900s "
            "is a safe default for an alignment-class tool."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _strip_framework_keys(cls, data: Any) -> Any:
        """Drop the framework-loader-only ``class`` key.

        When this config is loaded from a step YAML that carries a
        top-level ``class:`` (the auto-delegation target), the loader
        passes that key through to the config dict. ``extra='forbid'``
        would otherwise reject it. Stripping it here keeps typo
        protection for every OTHER field intact.
        """
        if isinstance(data, dict):
            data.pop("class", None)
        return data


class RheaFileToolStep(BaseStep):
    """Run any Rhea file-input Galaxy tool over MCP.

    Expected ``process()`` input payload::

        {
            "fasta_name": "seqtest.fasta",   # logical file name
            "fasta_bytes": b">seq1\\n...",   # raw bytes
        }

    ``fasta_text: str`` is accepted as an alternative to ``fasta_bytes``
    (it is utf-8 encoded). When wired into a workflow, the payload
    arrives wrapped as ``{<input_du_name>: <payload>}`` — the step
    self-unwraps that envelope.

    Return shape::

        {
            "tool_name": "muscle",
            "return_code": 0,
            "stdout": "...",
            "stderr": "...",
            "output_files": {"out_align": "<aligned FASTA text>", ...},
        }
    """

    COMPONENT_TYPE: str = "rhea_file_tool_step"
    REQUIRED_CONFIG_FIELDS = ["name", "tool_name", "find_tools_query", "file_input_arg"]

    @classmethod
    def _get_config_class(cls):
        return RheaFileToolStepConfig

    def _init_from_config(
        self,
        config: RheaFileToolStepConfig,
        component_config: Dict[str, Any],
        dependencies: Dict[str, Any],
    ) -> None:
        super()._init_from_config(config, component_config, dependencies)
        # Stash the typed config — process() reads every field off it.
        # No network work at init: a malformed config should fail at
        # config-load, but Rhea server reachability is a runtime concern
        # verified (and FAIL-LOUD) in process().
        self._rfts_config = config

    @property
    def rhea_config(self) -> RheaFileToolStepConfig:
        """The resolved RheaFileToolStepConfig this step dispatches with."""
        return self._rfts_config

    @staticmethod
    def _resolve_base_url(cfg: RheaFileToolStepConfig) -> str:
        """Resolve the Rhea file-HTTP base URL from config.

        Prefers an explicit ``http_base_url``; otherwise derives it from
        ``mcp_url`` by stripping the trailing ``/mcp`` segment. Handles
        both ``.../mcp/`` and ``.../mcp`` shapes.
        """
        if cfg.http_base_url:
            return cfg.http_base_url.rstrip("/")
        return cfg.mcp_url.rstrip("/").removesuffix("/mcp")

    def _unwrap_trigger_envelope(self, input_data: Any) -> Any:
        """Strip the ``{<input_du_name>: payload}`` trigger envelope.

        Mirrors ``ToolExecutionStep._unwrap_trigger_envelope`` but the
        discriminator here is the payload's own shape: the real payload
        carries ``fasta_name`` / ``fasta_bytes`` / ``fasta_text`` keys.
        A single-key dict whose lone key is none of those, and whose
        value is itself a dict, is the trigger envelope (the key is the
        input data unit's name). Anything else passes through.
        """
        payload_keys = {"fasta_name", "fasta_bytes", "fasta_text"}
        if not isinstance(input_data, dict) or len(input_data) != 1:
            return input_data
        (only_key,) = input_data.keys()
        if only_key in payload_keys:
            return input_data
        value = input_data[only_key]
        if not isinstance(value, dict):
            return input_data
        logger.debug(
            "RheaFileToolStep %r: unwrapped trigger envelope key %r",
            self.name,
            only_key,
        )
        return value

    @staticmethod
    def _coerce_file_bytes(payload: Dict[str, Any]) -> tuple[str, bytes]:
        """Extract ``(fasta_name, fasta_bytes)`` from the payload.

        Accepts ``fasta_bytes`` (bytes) or ``fasta_text`` (str, utf-8
        encoded). FAIL-LOUD when neither is present or usable.
        """
        name = payload.get("fasta_name")
        if not isinstance(name, str) or not name:
            raise ComponentConfigurationError(
                "FAIL-FAST: RheaFileToolStep payload missing a non-empty "
                "'fasta_name' string"
            )
        raw = payload.get("fasta_bytes")
        if raw is not None:
            if not isinstance(raw, (bytes, bytearray)):
                raise ComponentConfigurationError(
                    "FAIL-FAST: RheaFileToolStep 'fasta_bytes' must be "
                    f"bytes, got {type(raw).__name__}"
                )
            return name, bytes(raw)
        text = payload.get("fasta_text")
        if isinstance(text, str) and text:
            return name, text.encode("utf-8")
        raise ComponentConfigurationError(
            "FAIL-FAST: RheaFileToolStep payload must carry 'fasta_bytes' "
            "(bytes) or 'fasta_text' (str)"
        )

    async def process(self, input_data: Any, **kwargs) -> Dict[str, Any]:
        """Stage the file, dispatch the tool over MCP, fetch outputs.

        See the class docstring for the input / output shapes.
        """
        cfg = self._rfts_config
        payload = self._unwrap_trigger_envelope(input_data)
        if not isinstance(payload, dict):
            raise ComponentConfigurationError(
                "FAIL-FAST: RheaFileToolStep input must be a dict "
                f"({{fasta_name, fasta_bytes|fasta_text}}), got "
                f"{type(payload).__name__}"
            )
        fasta_name, fasta_bytes = self._coerce_file_bytes(payload)

        base = self._resolve_base_url(cfg)
        transport = MCPTransport(
            mcp_url=cfg.mcp_url,
            timeout_seconds=cfg.timeout_seconds,
            client_name="nanobrain-rhea-file-tool-step",
        )

        # Per-call keys staged on the rhea server, evicted in finally so the
        # server's redis store does not grow run-over-run (the rhea SERVER
        # stays online; only the per-call ephemera are torn down).
        input_key: str | None = None
        output_keys: List[str] = []
        async with httpx.AsyncClient() as http:
            try:
                # 1. Stage the file via the server's redis-direct upload
                #    endpoint. The returned "key" (a meta:<uuid> string)
                #    is what the tool receives as its file argument.
                upload_resp = await http.post(
                    f"{base}/upload",
                    content=fasta_bytes,
                    headers={"x-filename": fasta_name},
                    timeout=cfg.timeout_seconds,
                )
                upload_resp.raise_for_status()
                input_key = upload_resp.json()["key"]
                self.nb_logger.info(
                    "RheaFileToolStep %r: staged %d bytes as %r -> key=%r",
                    self.name,
                    len(fasta_bytes),
                    fasta_name,
                    input_key,
                )

                # 2. find-and-establish: surface the matching tools into the
                #    Rhea session-scoped catalog via the unified seam, then
                #    VALIDATE the configured tool was actually surfaced. A
                #    find_tools query that does NOT surface cfg.tool_name would
                #    otherwise fail later with a confusing tools/call error;
                #    failing loud HERE names the real problem.
                surfaced = await find_and_establish_tool(
                    cfg.find_tools_query, rhea_transport=transport
                )
                self._assert_tool_surfaced(surfaced, cfg.tool_name)

                # 3. tools/call the named tool with the staged file.
                tool_args: Dict[str, Any] = {cfg.file_input_arg: input_key}
                tool_args.update(cfg.static_tool_args)
                self.nb_logger.info(
                    "RheaFileToolStep %r: calling tool %r with args %r",
                    self.name,
                    cfg.tool_name,
                    tool_args,
                )
                raw_result = await transport.call(
                    "tools/call",
                    {"name": cfg.tool_name, "arguments": tool_args},
                )
                # parse_tool_call_result raises ComponentConfigurationError
                # on isError=True — that is the FAIL-LOUD we want.
                rhea_output = parse_tool_call_result(raw_result, cfg.tool_name)

                if isinstance(rhea_output, str):
                    # parse_tool_call_result returns a str when the text
                    # content was not JSON — Rhea always returns RheaOutput
                    # JSON, so a bare string is a contract violation worth
                    # FAIL-LOUDing.
                    try:
                        rhea_output = json.loads(rhea_output)
                    except json.JSONDecodeError as exc:
                        raise ComponentConfigurationError(
                            f"FAIL-FAST: RheaFileToolStep {self.name!r} got a "
                            f"non-JSON tool result for {cfg.tool_name!r}: "
                            f"{rhea_output[:300]!r}"
                        ) from exc
                if not isinstance(rhea_output, dict):
                    raise ComponentConfigurationError(
                        f"FAIL-FAST: RheaFileToolStep {self.name!r} expected a "
                        f"RheaOutput JSON object, got {type(rhea_output).__name__}"
                    )

                return_code = rhea_output.get("return_code")
                stdout = rhea_output.get("stdout", "") or ""
                stderr = rhea_output.get("stderr", "") or ""
                files = rhea_output.get("files") or []

                if return_code != 0:
                    raise ComponentConfigurationError(
                        f"FAIL-FAST: RheaFileToolStep {self.name!r}: tool "
                        f"{cfg.tool_name!r} returned non-zero return_code="
                        f"{return_code!r}. stderr head: {stderr[:600]!r}"
                    )

                # 4. Fetch the requested output files back from the server's
                #    redis-direct download endpoint and decode them to text.
                wanted = set(cfg.output_file_args)
                output_files: Dict[str, str] = {}
                for file_entry in files:
                    if not isinstance(file_entry, dict):
                        continue
                    file_name = file_entry.get("name")
                    if wanted and file_name not in wanted:
                        continue
                    key_field = file_entry.get("key")
                    # New redis-direct shape: "key" is a plain string. Handle
                    # the legacy {"redis_key": ...} dict shape defensively.
                    out_key = (
                        key_field["redis_key"]
                        if isinstance(key_field, dict)
                        else key_field
                    )
                    if not out_key:
                        continue
                    output_keys.append(out_key)
                    download_resp = await http.get(
                        f"{base}/download",
                        params={"key": out_key},
                        timeout=cfg.timeout_seconds,
                    )
                    download_resp.raise_for_status()
                    output_files[str(file_name)] = download_resp.content.decode(
                        "utf-8", "ignore"
                    )

                if not output_files:
                    # A green call with no usable output is the silent-failure
                    # shape this step exists to refuse.
                    raise ComponentConfigurationError(
                        f"FAIL-FAST: RheaFileToolStep {self.name!r}: tool "
                        f"{cfg.tool_name!r} reported return_code=0 but produced no "
                        f"usable output files (requested={sorted(wanted) or 'ALL'}, "
                        f"result files={[f.get('name') for f in files if isinstance(f, dict)]}). "
                        f"A successful call with no output is a silent failure."
                    )

                self.nb_logger.info(
                    "RheaFileToolStep %r: tool %r succeeded, %d output file(s): %s",
                    self.name,
                    cfg.tool_name,
                    len(output_files),
                    sorted(output_files),
                )
                return {
                    "tool_name": cfg.tool_name,
                    "return_code": return_code,
                    "stdout": stdout,
                    "stderr": stderr,
                    "output_files": output_files,
                }
            finally:
                # Close the MCP transport, but never let that skip key eviction
                # (the more fragile teardown) — the server keys would leak.
                try:
                    await transport.aclose()
                except Exception as exc:  # noqa: BLE001 — transport close must not skip eviction
                    self.nb_logger.warning(
                        "RheaFileToolStep %r: MCP transport close failed: %s", self.name, exc
                    )
                # Per-execution teardown: evict the input key + every output
                # key from the server's redis store (idempotent server-side).
                # Best-effort — a delete failure must NEVER fail the tool call.
                evict_keys = ([input_key] if input_key else []) + output_keys
                for key in evict_keys:
                    try:
                        await http.post(
                            f"{base}/delete",
                            params={"key": key},
                            timeout=cfg.timeout_seconds,
                        )
                    except Exception as exc:  # noqa: BLE001 — teardown never fails the call
                        self.nb_logger.warning(
                            "RheaFileToolStep %r: failed to evict server key %r: %s",
                            self.name,
                            key,
                            exc,
                        )

    def _assert_tool_surfaced(
        self, surfaced: List[UnifiedToolDescriptor], tool_name: str
    ) -> None:
        """FAIL-LOUD when ``find_tools`` did not surface ``tool_name``.

        ``find_and_establish_tool`` returns every tool the Rhea session
        catalog now exposes for the query. ``cfg.tool_name`` is the raw
        Rhea/Galaxy tool name; a surfaced UTD carries it as the
        descriptor's ``tool_id`` (sanitized) AND verbatim under
        ``provenance_pin.mcp_support['rhea_tool_name']``. Match against
        the raw name, its sanitized form, and the descriptor ids so the
        check is robust to the discovery sanitizer.
        """
        candidates: set[str] = set()
        for utd in surfaced:
            candidates.add(utd.descriptor_id)
            candidates.add(utd.descriptor_tool_id)
            pin = getattr(utd, "provenance_pin", None)
            mcp_support = getattr(pin, "mcp_support", None) or {}
            raw_name = mcp_support.get("rhea_tool_name")
            if raw_name:
                candidates.add(str(raw_name))
        sanitized = RheaMCPDiscovery._sanitize_tool_id(tool_name)
        if tool_name in candidates or sanitized in candidates:
            return
        raise ComponentConfigurationError(
            f"FAIL-FAST: RheaFileToolStep {self.name!r}: find_tools query "
            f"{self._rfts_config.find_tools_query!r} did NOT surface the "
            f"configured tool {tool_name!r} into the Rhea session catalog. "
            f"Surfaced tool ids: {sorted(candidates)}. Adjust find_tools_query "
            f"so it matches the tool, or verify tool_name is correct."
        )

    def _evict_rhea_keys(self, redis_client: Any, keys: list[str]) -> None:
        """Best-effort delete of Redis keys via a redis client. NEVER raises
        (teardown is observability, not correctness — a failed evict must not
        fail the tool call).

        Retained for the isolated teardown unit test; ``process()`` now evicts
        via the server's HTTP ``/delete`` endpoint (this step no longer holds a
        redis client of its own).
        """
        for key in keys:
            if not key:
                continue
            try:
                redis_client.delete(key)
            except Exception as exc:  # noqa: BLE001
                self.nb_logger.warning(
                    "RheaFileToolStep %r: failed to evict Redis key %r: %s",
                    self.name,
                    key,
                    exc,
                )

    @staticmethod
    def _close_rhea_store(store: Any) -> None:
        """Best-effort release of a store object's connection resources.

        Retained for the isolated teardown unit test; ``process()`` no longer
        holds a ProxyStore Store of its own.
        """
        close = getattr(store, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001 — teardown must never fail the call
                pass


__all__ = ["RheaFileToolStep", "RheaFileToolStepConfig"]
