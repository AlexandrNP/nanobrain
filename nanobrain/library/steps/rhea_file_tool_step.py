"""RheaFileToolStep — run any Rhea file-input Galaxy tool over MCP.

This step generalizes the ad-hoc proof script ``rhea/scripts/run_muscle_e2e.py``
into a proper nanobrain ``BaseStep``. It drives the full Rhea file-input
protocol for ANY Galaxy ``type="data"`` tool (MUSCLE is just one
configuration of it):

  1. Stage the input file bytes into Rhea's ``rhea-input`` ProxyStore
     via ``RheaFileProxy.from_buffer(...).to_proxy(store)`` — this
     returns a redis_key.
  2. ``find_tools`` over MCP with a semantic query so the Rhea server
     populates the session-scoped tool catalog.
  3. ``tools/call`` the named tool with ``{file_input_arg: redis_key,
     **static_tool_args}``.
  4. Parse Rhea's ``RheaOutput`` JSON, fetch each requested output
     file back out of the ``rhea-output`` ProxyStore, decode to text.

Honesty / silent-failure discipline (workspace CLAUDE.md):

  - ``RheaFileProxy`` is lazy-imported INSIDE ``process()`` and the
    ``ImportError`` is re-raised FAIL-LOUD: this step is intentionally
    Rhea-coupled and the runtime needs the ``rhea`` repo on PYTHONPATH.
    The class is genuinely imported (not vendored) because cloudpickle
    pickles by module reference — the rhea-server deserializes the
    exact ``rhea.utils.proxy.RheaFileProxy`` class.
  - ``parse_tool_call_result`` raises ``ComponentConfigurationError``
    on the MCP ``isError=True`` envelope — that is the FAIL-LOUD path.
  - A non-zero ``return_code`` raises.
  - A "successful" call that produced ZERO usable output files raises:
    a green call with no output is the silent-failure shape this step
    exists to refuse.

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

from pydantic import ConfigDict, Field, model_validator

from nanobrain.core.component_base import ComponentConfigurationError
from nanobrain.core.step import BaseStep, StepConfig
from nanobrain.library.tools._mcp_transport import (
    MCPTransport,
    parse_tool_call_result,
)

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
    redis_host: str = Field(default="localhost")
    redis_port: int = Field(default=6379)
    input_store_name: str = Field(
        default="rhea-input",
        description="ProxyStore name the input file is staged into.",
    )
    output_store_name: str = Field(
        default="rhea-output",
        description="ProxyStore name the tool's output files are read from.",
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
            "redis_key (e.g. 'input_seqs' for MUSCLE)."
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
            "Names of output files to fetch back from the output "
            "ProxyStore. Empty list = fetch all files in the result."
        ),
    )
    timeout_seconds: float = Field(
        default=900.0,
        gt=0.0,
        description=(
            "Per-call MCP timeout. Galaxy tools can be slow; 900s is a "
            "safe default for an alignment-class tool."
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
        # No network / Redis work at init: a malformed config should
        # fail at config-load, but Rhea/Redis reachability is a runtime
        # concern verified (and FAIL-LOUD) in process().
        self._rfts_config = config

    @property
    def rhea_config(self) -> RheaFileToolStepConfig:
        """The resolved RheaFileToolStepConfig this step dispatches with."""
        return self._rfts_config

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

        # Lazy-import the Rhea-side classes. This step is intentionally
        # Rhea-coupled: cloudpickle pickles by module reference, so the
        # rhea-server can only deserialize the genuine
        # rhea.utils.proxy.RheaFileProxy class — it MUST be importable
        # here, not vendored.
        try:
            import cloudpickle
            from proxystore.connectors.redis import RedisConnector, RedisKey
            from proxystore.store import Store
            from redis import Redis

            from rhea.utils.proxy import RheaFileProxy
        except ImportError as exc:  # noqa: PERF203 — single import block
            raise ComponentConfigurationError(
                "FAIL-FAST: RheaFileToolStep requires the 'rhea' repo on "
                "PYTHONPATH plus proxystore/redis/cloudpickle installed. "
                "This step is intentionally Rhea-coupled — cloudpickle "
                "pickles RheaFileProxy by module reference, so the genuine "
                f"class must be importable. Underlying error: {exc}"
            ) from exc

        redis_client = Redis(host=cfg.redis_host, port=cfg.redis_port)
        input_store = Store(
            name=cfg.input_store_name,
            connector=RedisConnector(cfg.redis_host, cfg.redis_port),
            serializer=cloudpickle.dumps,
            deserializer=cloudpickle.loads,
        )

        # 1. Stage the file into the rhea-input ProxyStore.
        proxy = RheaFileProxy.from_buffer(fasta_name, fasta_bytes, redis_client)
        redis_key = proxy.to_proxy(input_store)
        self.nb_logger.info(
            "RheaFileToolStep %r: staged %d bytes as %r -> redis_key=%r",
            self.name,
            len(fasta_bytes),
            fasta_name,
            redis_key,
        )

        transport = MCPTransport(
            mcp_url=cfg.mcp_url,
            timeout_seconds=cfg.timeout_seconds,
            client_name="nanobrain-rhea-file-tool-step",
        )
        try:
            # 2. find_tools so the Rhea server surfaces the tool into
            #    the session-scoped catalog.
            await transport.call(
                "tools/call",
                {
                    "name": "find_tools",
                    "arguments": {"query": cfg.find_tools_query},
                },
            )

            # 3. tools/call the named tool with the staged file.
            tool_args: Dict[str, Any] = {cfg.file_input_arg: redis_key}
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
        finally:
            await transport.aclose()

        if isinstance(rhea_output, str):
            # parse_tool_call_result returns a str when the text content
            # was not JSON — Rhea always returns RheaOutput JSON, so a
            # bare string is a contract violation worth FAIL-LOUDing.
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

        # 4. Fetch the requested output files back out of the
        #    rhea-output ProxyStore and decode them to text.
        output_store = Store(
            name=cfg.output_store_name,
            connector=RedisConnector(cfg.redis_host, cfg.redis_port),
            serializer=cloudpickle.dumps,
            deserializer=cloudpickle.loads,
        )
        wanted = set(cfg.output_file_args)
        output_files: Dict[str, str] = {}
        for file_entry in files:
            if not isinstance(file_entry, dict):
                continue
            file_name = file_entry.get("name")
            if wanted and file_name not in wanted:
                continue
            key_field = file_entry.get("key")
            out_redis_key = (
                key_field["redis_key"]
                if isinstance(key_field, dict)
                else key_field
            )
            if not out_redis_key:
                continue
            out_proxy = RheaFileProxy.from_proxy(
                RedisKey(redis_key=out_redis_key), output_store
            )
            handle = out_proxy.open(redis_client)
            data = handle.read()
            output_files[str(file_name)] = data.decode("utf-8", "ignore")

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


__all__ = ["RheaFileToolStep", "RheaFileToolStepConfig"]
