"""Step-level provenance threading (G4).

Per ``apecx-mcp-integration/docs/CONTRACTS.md#g4``: the
framework wraps every ``process()`` call with a recorder that captures
inputs, outputs, executor metadata, code identity, timing, and
exceptions into a configurable sink (JSONL by default).

Without this, provenance is whatever each step's author chose to log —
uneven coverage by construction. With this, every step gets the same
provenance shape, and the bundle exporter can stitch records into a
deterministic graph for replay.

Redact vocabulary
-----------------

Per the gap proposal: every record passes through a redaction filter
BEFORE the sink writes it. Each primitive replaces a sensitive field
with a typed marker that preserves shape + audit signal:

| Primitive | Effect |
|---|---|
| ``payload`` | Strip materialized payload from input/output values; preserve keys + hashes + sizes |
| ``tool_args`` | Strip structured tool call args; preserve descriptor + arg-shape fingerprint |
| ``prompts`` | Strip ``prompt_text`` from LLM call records; preserve template_id + content_hash |
| ``llm_completions`` | Strip ``completion_text``; preserve length + token count + hash |
| ``executor_env`` | Strip env-var VALUES from executor_metadata; preserve names |
| ``path:<dotted>`` | Surgical: replace one dotted path with a marker |

Defaults:
- When ``redact:`` is omitted, the framework applies
  ``["prompts", "executor_env"]`` — the two cases that are dangerous to
  record in clear in any production deployment.
- Workflows that want full prompt + env capture must EXPLICITLY set
  ``redact: []`` (empty list), and the bundle exporter records that
  the omission was deliberate.

This module ships:
- ``ProvenanceContext`` — the recorder primitive (FromConfigBase).
- ``JsonlSink`` — writes one JSON object per line to a file.
- Redact-primitive helpers + the apply_redaction function.
- ``current_provenance_context()`` — contextvar accessor.
- Step-side integration is a follow-up (BaseStep wraps _execute_process
  to call the recorder); this module ships the primitive surface.
"""

from __future__ import annotations

import asyncio
import contextvars
import hashlib
import json
import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

from pydantic import ConfigDict, Field

from .component_base import FromConfigBase
from .config.config_base import ConfigBase

logger = logging.getLogger(__name__)


# Reserved redact primitive names (excluding the path:<dotted> shape).
_FIXED_REDACT_PRIMITIVES = {
    "payload",
    "tool_args",
    "prompts",
    "llm_completions",
    "executor_env",
}

# Default redactions when redact: is omitted from the config.
_DEFAULT_REDACTIONS = ["prompts", "executor_env"]


# ---------------------------------------------------------------------------
# Sink protocol + JsonlSink reference implementation
# ---------------------------------------------------------------------------

class ProvenanceSinkBase:
    """Abstract sink. Concrete sinks implement ``write_record``."""

    async def write_record(self, record: Dict[str, Any]) -> None:
        raise NotImplementedError

    async def flush(self) -> None:
        """Optional override for sinks that buffer."""
        return None

    async def close(self) -> None:
        """Optional override for sinks that own resources."""
        return None


class JsonlSink(ProvenanceSinkBase):
    """Reference sink — appends one JSON object per line to a file.

    Thread-safe via an asyncio Lock. Buffered writes flush after every
    ``flush_every`` records (default 10) OR explicit ``await flush()``.
    """

    def __init__(self, path: Union[str, Path], flush_every: int = 10):
        self._path = Path(path)
        self._flush_every = max(1, flush_every)
        self._lock = asyncio.Lock()
        self._pending: List[str] = []
        # Ensure the parent directory exists. We DON'T create it lazily
        # in write_record because asyncio + filesystem operations are
        # less predictable.
        self._path.parent.mkdir(parents=True, exist_ok=True)

    async def write_record(self, record: Dict[str, Any]) -> None:
        async with self._lock:
            line = json.dumps(record, default=str, sort_keys=True)
            self._pending.append(line)
            if len(self._pending) >= self._flush_every:
                self._flush_locked()

    async def flush(self) -> None:
        async with self._lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        if not self._pending:
            return
        with self._path.open("a") as f:
            for line in self._pending:
                f.write(line + "\n")
        self._pending.clear()


# ---------------------------------------------------------------------------
# Redaction primitives
# ---------------------------------------------------------------------------

def _redact_payload_value(value: Any) -> Dict[str, Any]:
    """Replace a materialized payload with a typed marker.

    For dicts/lists, computes a stable size + a SHA-256 hash of the
    canonical JSON form. For other types, hashes the str() form.
    """
    try:
        canonical = json.dumps(value, default=str, sort_keys=True)
    except (TypeError, ValueError):
        canonical = str(value)
    return {
        "redacted": "payload",
        "size_bytes": len(canonical.encode("utf-8")),
        "hash": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


def _apply_payload_redaction(record: Dict[str, Any]) -> None:
    """Mutates record in place: replace inputs[*].value and
    outputs[*].value with a redaction marker."""
    for direction in ("inputs", "outputs"):
        items = record.get(direction)
        if not isinstance(items, dict):
            continue
        for unit_name, unit_payload in items.items():
            if isinstance(unit_payload, dict) and "value" in unit_payload:
                items[unit_name] = {
                    **{k: v for k, v in unit_payload.items() if k != "value"},
                    "value": _redact_payload_value(unit_payload["value"]),
                }


def _apply_tool_args_redaction(record: Dict[str, Any]) -> None:
    """Strip structured args of every tool call but preserve tool ID +
    descriptor hash + arg-shape fingerprint."""
    for tool_call in record.get("tool_calls", []):
        if isinstance(tool_call, dict) and "args" in tool_call:
            args = tool_call["args"]
            try:
                schema_canonical = json.dumps(
                    _fingerprint_shape(args), sort_keys=True
                )
            except (TypeError, ValueError):
                schema_canonical = str(type(args))
            tool_call["args"] = {
                "redacted": "tool_args",
                "schema_hash": hashlib.sha256(
                    schema_canonical.encode("utf-8")
                ).hexdigest(),
            }


def _fingerprint_shape(value: Any) -> Any:
    """Recursively replace leaf values with their type names — captures
    the SHAPE of args without the values."""
    if isinstance(value, dict):
        return {k: _fingerprint_shape(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_fingerprint_shape(v) for v in value]
    return type(value).__name__


def _apply_prompts_redaction(record: Dict[str, Any]) -> None:
    """Strip prompt_text from every LLM invocation; preserve template
    fingerprints."""
    for llm_call in record.get("llm_calls", []):
        if isinstance(llm_call, dict) and "prompt_text" in llm_call:
            template_id = llm_call.get("prompt_template_id")
            template_version = llm_call.get("template_version")
            param_hash = llm_call.get("param_hash")
            del llm_call["prompt_text"]
            llm_call["redacted_prompt"] = {
                "redacted": "prompts",
                "template_id": template_id,
                "template_version": template_version,
                "param_hash": param_hash,
            }


def _apply_llm_completions_redaction(record: Dict[str, Any]) -> None:
    """Strip completion_text from every LLM record; preserve hash + count."""
    for llm_call in record.get("llm_calls", []):
        if isinstance(llm_call, dict) and "completion_text" in llm_call:
            text = llm_call["completion_text"]
            llm_call["completion_text"] = {
                "redacted": "llm_completions",
                "char_count": len(text) if isinstance(text, str) else 0,
                "token_count": llm_call.get("token_count", 0),
                "hash": hashlib.sha256(
                    str(text).encode("utf-8")).hexdigest(),
            }


def _apply_executor_env_redaction(record: Dict[str, Any]) -> None:
    """Strip env-var VALUES from executor_metadata.env (if present);
    preserve env-var NAMES."""
    em = record.get("executor_metadata", {})
    if isinstance(em, dict) and isinstance(em.get("env"), dict):
        env_names = sorted(em["env"].keys())
        em["env"] = {
            "redacted": "executor_env",
            "names": env_names,
        }


def _apply_path_redaction(record: Dict[str, Any], dotted_path: str) -> None:
    """Replace the value at a single dotted path with a marker."""
    parts = dotted_path.split(".")
    cur: Any = record
    for p in parts[:-1]:
        if not isinstance(cur, dict) or p not in cur:
            return  # path doesn't exist; nothing to redact
        cur = cur[p]
    if not isinstance(cur, dict) or parts[-1] not in cur:
        return
    cur[parts[-1]] = {
        "redacted": "path",
        "path": dotted_path,
    }


def apply_redactions(
    record: Dict[str, Any],
    redactions: List[str],
) -> Dict[str, Any]:
    """Apply each redaction primitive in declared order. Mutates a copy
    and returns it. The original record is left intact so callers can
    log un-redacted versions to a more-trusted sink if they want."""
    # Deep copy via JSON roundtrip — guarantees isolation from the
    # caller and serializes for free. Cost: payload values must be
    # JSON-serializable, which is usually true for provenance records
    # by construction.
    try:
        result = json.loads(json.dumps(record, default=str))
    except (TypeError, ValueError):
        # Fallback: a shallow copy. Less safe but better than nothing.
        result = dict(record)

    for redaction in redactions:
        if redaction == "payload":
            _apply_payload_redaction(result)
        elif redaction == "tool_args":
            _apply_tool_args_redaction(result)
        elif redaction == "prompts":
            _apply_prompts_redaction(result)
        elif redaction == "llm_completions":
            _apply_llm_completions_redaction(result)
        elif redaction == "executor_env":
            _apply_executor_env_redaction(result)
        elif redaction.startswith("path:"):
            _apply_path_redaction(result, redaction[len("path:"):])
        else:
            logger.warning(
                "Unknown redaction primitive %r — ignored. "
                "Vocabulary: %s | path:<dotted_path>",
                redaction, sorted(_FIXED_REDACT_PRIMITIVES),
            )
    return result


def normalize_redactions(redactions: Optional[List[str]]) -> List[str]:
    """Resolve the user-configured redactions list to its effective form.

    - None → default redactions (prompts + executor_env)
    - Explicit empty list ([]) → no redactions (operator opt-out)
    - Non-empty list → that list as-is
    """
    if redactions is None:
        return list(_DEFAULT_REDACTIONS)
    return list(redactions)


# ---------------------------------------------------------------------------
# Provenance context
# ---------------------------------------------------------------------------

class ProvenanceContextConfig(ConfigBase):
    """Configuration for ProvenanceContext."""
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    sink_path: Optional[str] = Field(
        default=None,
        description="Path to a JSONL file for the default JsonlSink. "
                    "When set, the context auto-builds a JsonlSink "
                    "writing to this path. Mutually exclusive with "
                    "the lower-level sink injection in __init__.",
    )
    flush_every: int = Field(default=10, ge=1)
    redact: Optional[List[str]] = Field(
        default=None,
        description="Redaction primitives to apply to every record. "
                    "None = default ['prompts', 'executor_env']. "
                    "Empty list = no redactions (deliberate).",
    )


# Contextvar carrying the active ProvenanceContext.
_current_provenance: contextvars.ContextVar[Optional["ProvenanceContext"]] = (
    contextvars.ContextVar("current_provenance_context", default=None)
)


class ProvenanceContext(FromConfigBase):
    """G4 — per-run provenance recorder.

    Sinks one JSON object per ``record_step_invocation`` call. Records
    pass through the configured redaction filter BEFORE reaching the
    sink, so an operator who configures the wrong sink can't leak
    secrets.

    Lifecycle:
    - Created via from_config or via the convenience kwargs path.
    - ``activate()`` installs the context as the process-current
      ProvenanceContext (contextvar-managed; PEP 567 means concurrent
      asyncio Tasks see their own context).
    - Steps inside the run consult ``current_provenance_context()`` to
      get the recorder; if no context is active, recording is a no-op.

    The framework's ``BaseStep._execute_process`` integration that
    automatically calls ``record_step_invocation`` is a sibling task
    (NB-G4-02 in the implementation_task_graph.md) and is gated on this
    primitive shipping cleanly first. THIS module ships the primitive +
    sink + redaction; the wrap-on-_execute_process work follows once
    the primitive's API has settled.
    """

    @classmethod
    def _get_config_class(cls):
        return ProvenanceContextConfig

    @classmethod
    def from_config(cls, config=None, **kwargs) -> "ProvenanceContext":
        """Override to admit dict / None inputs (this is a config primitive,
        not a Step or DataUnit). Sink instance can be passed via kwargs
        as 'sink' to override the auto-built JsonlSink."""
        from pathlib import Path
        if config is None:
            config = {}
        if isinstance(config, (str, Path)):
            config_object = ProvenanceContextConfig.from_config(config)
        elif isinstance(config, dict):
            try:
                ProvenanceContextConfig._allow_direct_instantiation = True
                config_object = ProvenanceContextConfig(**config)
            finally:
                ProvenanceContextConfig._allow_direct_instantiation = False
        elif isinstance(config, ProvenanceContextConfig):
            config_object = config
        else:
            raise ValueError(f"Unsupported config type: {type(config)}")

        cls._allow_direct_instantiation = True
        try:
            instance = cls()
        finally:
            cls._allow_direct_instantiation = False

        instance._enabled = config_object.enabled
        instance._redactions = normalize_redactions(config_object.redact)

        # Sink resolution: explicit kwarg > sink_path > no sink (no-op).
        explicit_sink = kwargs.get("sink")
        if explicit_sink is not None:
            instance._sink = explicit_sink
        elif config_object.sink_path:
            instance._sink = JsonlSink(
                config_object.sink_path,
                flush_every=config_object.flush_every,
            )
        else:
            instance._sink = None

        return instance

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def redactions(self) -> List[str]:
        """The effective redaction list (after default-resolution)."""
        return list(self._redactions)

    @property
    def sink(self) -> Optional[ProvenanceSinkBase]:
        return self._sink

    async def record_step_invocation(
        self,
        *,
        step_name: str,
        inputs: Optional[Dict[str, Any]] = None,
        outputs: Optional[Dict[str, Any]] = None,
        executor_metadata: Optional[Dict[str, Any]] = None,
        code_identity: Optional[Dict[str, Any]] = None,
        timing: Optional[Dict[str, float]] = None,
        exception: Optional[Dict[str, Any]] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        llm_calls: Optional[List[Dict[str, Any]]] = None,
        run_id: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record one step invocation. Apply redactions; write to sink.

        When the context is disabled OR no sink is configured, this is
        a fast no-op (still applies the redaction logic for testability;
        future optimization may skip).
        """
        if not self._enabled:
            return
        record: Dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "step_name": step_name,
            "inputs": inputs or {},
            "outputs": outputs or {},
            "executor_metadata": executor_metadata or {},
            "code_identity": code_identity or {},
            "timing": timing or {},
        }
        if exception is not None:
            record["exception"] = exception
        if tool_calls:
            record["tool_calls"] = list(tool_calls)
        if llm_calls:
            record["llm_calls"] = list(llm_calls)
        if run_id:
            record["run_id"] = run_id
        if extra:
            record["extra"] = extra

        redacted = apply_redactions(record, self._redactions)

        if self._sink is not None:
            await self._sink.write_record(redacted)

    @contextmanager
    def activate(self) -> Iterator["ProvenanceContext"]:
        """Install this context as current for the duration of the
        with-block."""
        token = _current_provenance.set(self)
        try:
            yield self
        finally:
            _current_provenance.reset(token)

    async def flush(self) -> None:
        if self._sink is not None:
            await self._sink.flush()

    async def close(self) -> None:
        if self._sink is not None:
            await self._sink.close()


def current_provenance_context() -> Optional[ProvenanceContext]:
    """Return the currently-active ProvenanceContext (or None)."""
    return _current_provenance.get()
