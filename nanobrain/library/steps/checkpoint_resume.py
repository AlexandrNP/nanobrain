"""CheckpointStep + ResumeStep (G5).

Per ``apecx-mcp-integration/docs/nanobrain_capability_gaps.md G5``: ship
the runtime primitives that snapshot a partial workflow run and re-enter
it. The workflow-runner-side "skip upstream execution" hook is a
follow-up task; this module ships the primitives and the manifest
contract so apecx-mcp + Track B Phase 4 (HPC bundle replay) can
adopt them today.

Design summary (per the gap proposal):

- ``CheckpointStep.process(input)``: snapshot every value in ``input``
  to either a ProxyStore (preferred — uses G3 + G13) or a filesystem
  path; write a JSON manifest listing the captured names + content
  hashes + storage backend; emit ``{manifest_path, captured: [...]}``.
- ``ResumeStep.process(input)``: read the manifest at ``input['manifest_path']``;
  restore every captured value; emit a dict keyed by the original
  unit names. The downstream workflow can then bypass the
  upstream steps that originally produced those values.
- The two primitives share a small JSON-schema contract for the
  manifest body so a checkpoint written by one process can be
  resumed by a different process (HPC bundle replay scenario).

What this commit does NOT do:

- Workflow-runner integration (``ResumeStep`` doesn't yet TELL the
  runner to skip upstream execution). The integration is gap
  G5-Step-3, queued as follow-up. v1 lets apecx-mcp use the primitives
  in custom orchestration code (manually structuring the workflow
  YAML so a resume path skips upstream steps via a ConditionalLink
  on a "checkpoint_present" predicate).
- Container-digest mismatch policy (G5 spec). v1 records
  ``code_identity`` in the manifest; the operator reviews it on
  resume. The strict-FAIL-FAST-on-container-mismatch behavior is
  follow-up.
- ``DataUnitStream`` rejection at workflow-load time (G5 spec).
  v1's CheckpointStep simply errors at process time if asked to
  snapshot a stream-shaped value.

Storage backends:

- ``filesystem`` — write each value as a JSON file in a directory.
  Default; works without G3/G13. Good for tests + single-host
  development.
- ``proxystore`` — write each value to a configured ProxyStore Store;
  the manifest records ProxyStore Keys. Requires G3 (DataUnitProxyRef
  shipped) and benefits from G13 (run-context namespacing) for
  multi-tenant isolation.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Literal, Optional, Union

from pydantic import Field, model_validator

from nanobrain.core.component_base import ComponentConfigurationError
from nanobrain.core.step import BaseStep, StepConfig

logger = logging.getLogger(__name__)


# Manifest schema version. Bump when the on-disk format changes in a
# way that older readers can't safely interpret.
_MANIFEST_VERSION = 1


# ---------------------------------------------------------------------------
# Storage backend protocol
# ---------------------------------------------------------------------------

class _CheckpointStorage:
    """Abstract storage. Concrete: filesystem (default) + proxystore."""

    async def write_value(self, value: Any, hint: str) -> Dict[str, Any]:
        """Write a value. Return a dict that uniquely identifies the
        stored bytes (this dict goes into the manifest)."""
        raise NotImplementedError

    async def read_value(self, descriptor: Dict[str, Any]) -> Any:
        """Read a value back from a descriptor produced by ``write_value``."""
        raise NotImplementedError


class _FilesystemStorage(_CheckpointStorage):
    """JSON-file-per-value storage. Each value goes to
    ``<base_dir>/<value_name>.json``. Hash-keyed for content-addressing."""

    def __init__(self, base_dir: Union[str, Path]):
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)

    async def write_value(self, value: Any, hint: str) -> Dict[str, Any]:
        # Reject stream-shaped values per G5 spec.
        if hasattr(value, "__aiter__") and not isinstance(value, (str, bytes, dict, list)):
            raise ComponentConfigurationError(
                f"FAIL-FAST: CheckpointStep cannot snapshot stream-shaped "
                f"value {hint!r} (async iterator); per G5 spec streams "
                f"are not snapshottable"
            )
        # Canonicalize + hash. JSON-roundtrip means non-JSON values
        # coerce via default=str.
        try:
            canonical = json.dumps(value, sort_keys=True, default=str)
        except TypeError as e:
            raise ComponentConfigurationError(
                f"FAIL-FAST: CheckpointStep value {hint!r} not "
                f"JSON-serializable (consider switching backend to "
                f"proxystore for non-JSON payloads): {e}"
            ) from e
        content_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        target_path = self._base_dir / f"{content_hash}.json"
        # Idempotent: if the file already exists with our hash, no rewrite.
        if not target_path.exists():
            target_path.write_text(canonical)
        return {
            "backend": "filesystem",
            "path": str(target_path),
            "content_hash": content_hash,
            "size_bytes": len(canonical.encode("utf-8")),
        }

    async def read_value(self, descriptor: Dict[str, Any]) -> Any:
        path = Path(descriptor["path"])
        if not path.is_file():
            raise ComponentConfigurationError(
                f"FAIL-FAST: ResumeStep cannot read filesystem checkpoint "
                f"at {path} — file missing"
            )
        text = path.read_text()
        # Verify hash matches (defense against tampering).
        actual_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        expected_hash = descriptor.get("content_hash")
        if expected_hash and actual_hash != expected_hash:
            raise ComponentConfigurationError(
                f"FAIL-FAST: ResumeStep checkpoint at {path} content_hash "
                f"mismatch — expected {expected_hash}, got {actual_hash}; "
                f"the file may have been tampered with"
            )
        return json.loads(text)


class _ProxyStoreStorage(_CheckpointStorage):
    """ProxyStore-backed storage. Each value is put() into a configured
    Store; the manifest records the typed Key."""

    def __init__(self, store_name: str, connector_kind: str = "file",
                 store_dir: Optional[str] = None):
        # Lazy import — proxystore is optional per G3.
        try:
            from proxystore.store import Store, register_store, get_store
            from proxystore.connectors.file import FileConnector
        except ImportError as e:
            raise ComponentConfigurationError(
                f"FAIL-FAST: CheckpointStep proxystore backend requires "
                f"proxystore. Install with: pip install proxystore. "
                f"Original: {e}"
            ) from e

        existing = get_store(store_name)
        if existing is None:
            if connector_kind == "file":
                if not store_dir:
                    raise ComponentConfigurationError(
                        f"FAIL-FAST: CheckpointStep proxystore file "
                        f"connector requires store_dir"
                    )
                connector = FileConnector(store_dir)
            else:
                raise ComponentConfigurationError(
                    f"FAIL-FAST: CheckpointStep proxystore connector "
                    f"{connector_kind!r} not yet supported (v1: 'file')"
                )
            store = Store(store_name, connector)
            register_store(store)
        else:
            store = existing
        self._store = store

    async def write_value(self, value: Any, hint: str) -> Dict[str, Any]:
        if hasattr(value, "__aiter__") and not isinstance(value, (str, bytes, dict, list)):
            raise ComponentConfigurationError(
                f"FAIL-FAST: CheckpointStep cannot snapshot stream-shaped "
                f"value {hint!r} (async iterator); per G5 spec streams "
                f"are not snapshottable"
            )
        proxy_key = self._store.put(value)
        # Hash the canonical-JSON form for idempotency / audit.
        try:
            canonical = json.dumps(value, sort_keys=True, default=str)
            content_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        except (TypeError, ValueError):
            content_hash = ""  # non-JSON payload; hash unavailable
        return {
            "backend": "proxystore",
            # ProxyStore Key serializes via its own dataclass shape; we
            # store the repr() so the manifest is self-describing. The
            # actual round-trip uses the stored Key object via store.get.
            "key_repr": str(proxy_key),
            "store_name": self._store.name,
            "content_hash": content_hash,
        }

    async def read_value(self, descriptor: Dict[str, Any]) -> Any:
        # The manifest's key_repr is the Key's str() form. We can't
        # reconstruct the typed Key from a string — proxystore's Key
        # types are dataclasses with private state. v1 limitation:
        # in-process resume only (the Store instance must still hold
        # the Key). Cross-process resume requires the manifest to
        # carry the Key's pickled form, which is a follow-up.
        raise ComponentConfigurationError(
            f"FAIL-FAST: ResumeStep proxystore backend in v1 requires "
            f"the original CheckpointStep instance to still be alive "
            f"in-process (the typed Key cannot be reconstructed from "
            f"the manifest's key_repr alone). For cross-process resume, "
            f"use the filesystem backend in v1; ProxyStore cross-process "
            f"is a follow-up that requires Key serialization."
        )


# ---------------------------------------------------------------------------
# CheckpointStep
# ---------------------------------------------------------------------------

class CheckpointStepConfig(StepConfig):
    """Configuration for CheckpointStep.

    Per G5 proposal: declare the storage backend + namespace + the list
    of input keys to capture. ``capture: ["*"]`` is shorthand for "every
    key in the input dict".
    """
    backend: Literal["filesystem", "proxystore"] = "filesystem"
    base_dir: Optional[str] = Field(
        default=None,
        description="Base directory for filesystem backend. Required when "
                    "backend='filesystem'. Created if missing.",
    )
    proxystore_store_name: Optional[str] = Field(
        default=None,
        description="Store name for proxystore backend. Required when "
                    "backend='proxystore'.",
    )
    proxystore_store_dir: Optional[str] = Field(
        default=None,
        description="Filesystem dir for the proxystore file connector. "
                    "Required when backend='proxystore' and connector_kind='file'.",
    )
    proxystore_connector_kind: Literal["file"] = "file"
    capture: List[str] = Field(
        default_factory=lambda: ["*"],
        description="Input keys to capture. ['*'] = all keys. Otherwise "
                    "list the specific keys.",
    )
    manifest_path: Optional[str] = Field(
        default=None,
        description="Where to write the manifest JSON. Required.",
    )

    @model_validator(mode="after")
    def _validate_backend_fields(self) -> "CheckpointStepConfig":
        if self.backend == "filesystem" and not self.base_dir:
            raise ValueError(
                "FAIL-FAST: CheckpointStep backend='filesystem' requires "
                "base_dir"
            )
        if self.backend == "proxystore" and not self.proxystore_store_name:
            raise ValueError(
                "FAIL-FAST: CheckpointStep backend='proxystore' requires "
                "proxystore_store_name"
            )
        if not self.manifest_path:
            raise ValueError(
                "FAIL-FAST: CheckpointStep requires manifest_path"
            )
        return self


class CheckpointStep(BaseStep):
    """G5 — snapshot upstream data unit values to durable storage.

    Input dict (process input):
        Any keys from upstream — the step captures the keys named in
        ``capture:`` (or all keys if ``capture: ['*']``).

    Output dict:
        ``manifest_path``: str — where the manifest was written
        ``captured``: list of captured key names
        ``manifest``: the manifest dict (for in-process consumers)
        plus an echo of the original input (so the workflow can
        continue downstream from the same data).

    Idempotency: re-running with the same input produces the same
    manifest (assuming deterministic input). The filesystem backend is
    content-addressed by hash; identical values reuse the same file.
    """

    COMPONENT_TYPE: str = "checkpoint_step"
    REQUIRED_CONFIG_FIELDS = ["name", "manifest_path"]

    @classmethod
    def _get_config_class(cls):
        return CheckpointStepConfig

    def _init_from_config(
        self,
        config: CheckpointStepConfig,
        component_config: Dict[str, Any],
        dependencies: Dict[str, Any],
    ) -> None:
        super()._init_from_config(config, component_config, dependencies)
        self._cfg: CheckpointStepConfig = config
        if config.backend == "filesystem":
            self._storage: _CheckpointStorage = _FilesystemStorage(config.base_dir)
        elif config.backend == "proxystore":
            self._storage = _ProxyStoreStorage(
                config.proxystore_store_name,
                connector_kind=config.proxystore_connector_kind,
                store_dir=config.proxystore_store_dir,
            )

    async def process(self, input_data: Any, **kwargs) -> Dict[str, Any]:
        if not isinstance(input_data, dict):
            raise ComponentConfigurationError(
                f"FAIL-FAST: CheckpointStep {self.name!r} input_data "
                f"must be dict, got {type(input_data).__name__}"
            )

        # Determine which keys to capture.
        if self._cfg.capture == ["*"]:
            keys = list(input_data.keys())
        else:
            keys = list(self._cfg.capture)
            missing = [k for k in keys if k not in input_data]
            if missing:
                raise ComponentConfigurationError(
                    f"FAIL-FAST: CheckpointStep {self.name!r} configured "
                    f"to capture {keys} but input_data missing: "
                    f"{sorted(missing)}; provided keys: "
                    f"{sorted(input_data.keys())}"
                )

        # Snapshot each value. Manifest entries are dicts with backend
        # type + descriptor for that backend.
        entries: Dict[str, Dict[str, Any]] = {}
        for key in keys:
            entries[key] = await self._storage.write_value(input_data[key], hint=key)

        # Build manifest.
        manifest = {
            "manifest_version": _MANIFEST_VERSION,
            "step_name": self.name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "backend": self._cfg.backend,
            "captured": keys,
            "entries": entries,
            # Code identity per G5 spec — what the operator needs to
            # decide whether the resume environment matches the snapshot.
            "code_identity": _capture_code_identity(),
        }

        # Write the manifest atomically (write to temp + rename).
        manifest_path = Path(self._cfg.manifest_path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        tmp_path.replace(manifest_path)

        # Echo the input so the workflow continues downstream cleanly.
        result: Dict[str, Any] = dict(input_data)
        result.update({
            "manifest_path": str(manifest_path),
            "captured": keys,
            "manifest": manifest,
        })
        return result


# ---------------------------------------------------------------------------
# ResumeStep
# ---------------------------------------------------------------------------

class ResumeStepConfig(StepConfig):
    """Configuration for ResumeStep.

    G5 Step 3 (2026-05-09) — ``on_missing='rebuild'`` is now supported.
    The rebuild path resolves a dotted-path callable (``rebuild_callable``),
    invokes it to regenerate the data, AND writes a fresh manifest at
    ``manifest_path`` so subsequent resumes hit the cache. The rebuild
    primitive is the "fail-to-cache, fall back to compute" pattern;
    operationally equivalent to a memoized expensive computation.
    """
    on_missing: Literal["fail", "skip", "rebuild"] = "fail"
    accept_code_identity_mismatch: bool = Field(
        default=False,
        description="When True, code-identity mismatch (different git sha "
                    "etc) is logged WARNING and resume proceeds. When "
                    "False (default), code-identity changes are surfaced "
                    "in the result dict's 'code_identity_warning' field "
                    "but resume still proceeds — the operator decides "
                    "whether to abort downstream.",
    )

    # G5 Step 3 — rebuild path. When on_missing='rebuild' AND the
    # manifest is missing, the framework resolves this dotted-path
    # callable, invokes it with the input_data dict, and treats its
    # return value as the resumed payload. A fresh manifest is then
    # written to manifest_path so subsequent resumes hit the cache.
    rebuild_callable: Optional[str] = Field(
        default=None,
        description="Dotted-path string resolving to a callable "
                    "f(input_data: dict) -> dict that regenerates the "
                    "checkpointed data. Required when on_missing='rebuild'. "
                    "May be sync or async; the framework awaits coroutine "
                    "return values."
    )
    rebuild_base_dir: Optional[str] = Field(
        default=None,
        description="Base directory for the manifest written after a "
                    "successful rebuild. Required when on_missing='rebuild'. "
                    "Mirrors CheckpointStepConfig.base_dir."
    )

    @model_validator(mode="after")
    def _validate_rebuild_fields(self) -> "ResumeStepConfig":
        if self.on_missing == "rebuild":
            if not self.rebuild_callable:
                raise ValueError(
                    "FAIL-FAST: ResumeStepConfig on_missing='rebuild' "
                    "requires rebuild_callable"
                )
            if not self.rebuild_base_dir:
                raise ValueError(
                    "FAIL-FAST: ResumeStepConfig on_missing='rebuild' "
                    "requires rebuild_base_dir"
                )
        return self


class ResumeStep(BaseStep):
    """G5 — restore data unit values from a checkpoint manifest.

    Input dict (process input):
        ``manifest_path``: str — path to the manifest written by a
            CheckpointStep
        Optionally: ``checkpoint_step`` — the CheckpointStep instance
            to use for proxystore backend resume (in-process resume only)

    Output dict:
        Every captured key from the manifest, with its restored value.
        Plus:
        ``_resumed_from_manifest``: str — manifest_path
        ``_resumed_at``: ISO timestamp
        ``_code_identity_warning``: str (only present on mismatch)

    Behavior on missing manifest:
    - on_missing='fail' (default): FAIL-FAST.
    - on_missing='skip': return empty dict; downstream sees no values.
    - on_missing='rebuild': raise NotImplementedError in v1 — the
      "rebuild from upstream" semantic is the workflow-runner's job
      (G5 Step 3), not this primitive's.
    """

    COMPONENT_TYPE: str = "resume_step"
    REQUIRED_CONFIG_FIELDS = ["name"]

    @classmethod
    def _get_config_class(cls):
        return ResumeStepConfig

    def _init_from_config(
        self,
        config: ResumeStepConfig,
        component_config: Dict[str, Any],
        dependencies: Dict[str, Any],
    ) -> None:
        super()._init_from_config(config, component_config, dependencies)
        self._cfg: ResumeStepConfig = config

    async def process(self, input_data: Any, **kwargs) -> Dict[str, Any]:
        if not isinstance(input_data, dict):
            raise ComponentConfigurationError(
                f"FAIL-FAST: ResumeStep {self.name!r} input_data "
                f"must be dict, got {type(input_data).__name__}"
            )

        manifest_path_str = input_data.get("manifest_path")
        if not manifest_path_str:
            raise ComponentConfigurationError(
                f"FAIL-FAST: ResumeStep {self.name!r} input_data missing "
                f"'manifest_path' key"
            )
        manifest_path = Path(manifest_path_str)
        if not manifest_path.is_file():
            if self._cfg.on_missing == "fail":
                raise ComponentConfigurationError(
                    f"FAIL-FAST: ResumeStep {self.name!r} manifest at "
                    f"{manifest_path} not found (on_missing='fail')"
                )
            elif self._cfg.on_missing == "skip":
                logger.warning(
                    "ResumeStep %s: manifest at %s missing; on_missing='skip' "
                    "→ returning empty dict",
                    self.name, manifest_path,
                )
                return {}
            elif self._cfg.on_missing == "rebuild":
                logger.warning(
                    "ResumeStep %s: manifest at %s missing; on_missing="
                    "'rebuild' → invoking rebuild_callable=%r",
                    self.name, manifest_path, self._cfg.rebuild_callable,
                )
                rebuilt = await self._do_rebuild(input_data, manifest_path)
                return rebuilt

        manifest = json.loads(manifest_path.read_text())
        if manifest.get("manifest_version") != _MANIFEST_VERSION:
            raise ComponentConfigurationError(
                f"FAIL-FAST: ResumeStep {self.name!r} manifest at "
                f"{manifest_path} has version "
                f"{manifest.get('manifest_version')!r}, expected "
                f"{_MANIFEST_VERSION}"
            )

        backend = manifest.get("backend")
        entries = manifest.get("entries", {})

        # For filesystem backend, we can read entries directly without
        # needing the original CheckpointStep instance.
        if backend == "filesystem":
            storage = _FilesystemStorage(
                # We pass any temp dir; _FilesystemStorage.read_value
                # uses the path stored in the descriptor.
                base_dir=manifest_path.parent,
            )
        elif backend == "proxystore":
            # ProxyStore in-process resume requires the original
            # CheckpointStep. For v1, we fail with a clear message.
            checkpoint_step = input_data.get("checkpoint_step")
            if checkpoint_step is None:
                raise ComponentConfigurationError(
                    f"FAIL-FAST: ResumeStep {self.name!r} manifest backend "
                    f"is 'proxystore' but no 'checkpoint_step' instance "
                    f"provided in input_data; v1 ProxyStore resume requires "
                    f"the original CheckpointStep alive in-process"
                )
            storage = checkpoint_step._storage
        else:
            raise ComponentConfigurationError(
                f"FAIL-FAST: ResumeStep {self.name!r} unknown manifest "
                f"backend {backend!r}"
            )

        restored: Dict[str, Any] = {}
        for key, descriptor in entries.items():
            restored[key] = await storage.read_value(descriptor)

        # Code-identity check.
        code_identity_warning: Optional[str] = None
        snapshot_identity = manifest.get("code_identity") or {}
        current_identity = _capture_code_identity()
        for k in ("python_version", "git_sha"):
            snap = snapshot_identity.get(k)
            cur = current_identity.get(k)
            if snap and cur and snap != cur:
                code_identity_warning = (
                    f"code_identity[{k}] mismatch: snapshot={snap!r}, "
                    f"current={cur!r}"
                )
                if self._cfg.accept_code_identity_mismatch:
                    logger.warning(
                        "ResumeStep %s: %s; accept_code_identity_mismatch=True",
                        self.name, code_identity_warning,
                    )
                else:
                    logger.warning(
                        "ResumeStep %s: %s; surfacing in result for operator review",
                        self.name, code_identity_warning,
                    )

        restored["_resumed_from_manifest"] = str(manifest_path)
        restored["_resumed_at"] = datetime.now(timezone.utc).isoformat()
        if code_identity_warning is not None:
            restored["_code_identity_warning"] = code_identity_warning

        return restored

    async def _do_rebuild(
        self, input_data: Dict[str, Any], manifest_path: Path,
    ) -> Dict[str, Any]:
        """G5 Step 3 — execute the rebuild path: resolve the configured
        callable, invoke it, then write a fresh manifest at
        ``manifest_path`` so subsequent resumes hit the cache.

        Returns the rebuilt-and-resumed payload dict (with _resumed_*
        bookkeeping fields, mirroring the normal resume return shape).
        """
        callable_obj = _resolve_dotted_callable(self._cfg.rebuild_callable)
        result = callable_obj(input_data)
        if asyncio.iscoroutine(result):
            result = await result

        if not isinstance(result, dict):
            raise ComponentConfigurationError(
                f"FAIL-FAST: ResumeStep {self.name!r} rebuild_callable "
                f"{self._cfg.rebuild_callable!r} returned "
                f"{type(result).__name__}, expected dict"
            )

        # Write a fresh manifest at manifest_path so the next resume
        # hits the cache. We mirror CheckpointStep's filesystem-backend
        # logic directly here rather than delegating, to avoid coupling
        # ResumeStep's rebuild path to the full CheckpointStep config
        # surface (operator only declares rebuild_base_dir).
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        base_dir = Path(self._cfg.rebuild_base_dir)
        base_dir.mkdir(parents=True, exist_ok=True)
        storage = _FilesystemStorage(base_dir=base_dir)

        entries: Dict[str, Any] = {}
        for key, value in result.items():
            if key.startswith("_"):
                # Bookkeeping keys (e.g., _resumed_at) — pass through to
                # the response but don't snapshot them.
                continue
            descriptor = await storage.write_value(value=value, hint=key)
            entries[key] = descriptor

        manifest_body = {
            "manifest_version": _MANIFEST_VERSION,
            "step_name": self.name,
            "backend": "filesystem",
            "captured": list(entries.keys()),
            "entries": entries,
            "code_identity": _capture_code_identity(),
            "rebuilt_from_callable": self._cfg.rebuild_callable,
            "rebuilt_at": datetime.now(timezone.utc).isoformat(),
        }
        # Atomic write: write to .tmp then rename.
        tmp_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(manifest_body, indent=2))
        tmp_path.replace(manifest_path)

        # Compose the response in the same shape as the normal resume
        # path, plus a marker indicating this was a rebuild.
        response = dict(result)
        response["_resumed_from_manifest"] = str(manifest_path)
        response["_resumed_at"] = datetime.now(timezone.utc).isoformat()
        response["_rebuilt"] = True
        return response


# ---------------------------------------------------------------------------
# Dotted-path callable resolver (mirrors entry_triggers helper)
# ---------------------------------------------------------------------------

def _resolve_dotted_callable(spec: str):
    """Resolve a dotted-path spec to a callable. Mirrors
    nanobrain.library.runtime.entry_triggers._resolve_dotted_callable;
    duplicated here to avoid the steps → runtime import dependency."""
    import importlib

    if not isinstance(spec, str) or "." not in spec:
        raise ComponentConfigurationError(
            f"FAIL-FAST: rebuild_callable must be a dotted-path string "
            f"like 'pkg.mod.func'; got {spec!r}"
        )
    module_path, _, attr_path = spec.rpartition(".")
    try:
        mod = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        raise ComponentConfigurationError(
            f"FAIL-FAST: rebuild_callable module {module_path!r} not "
            f"importable: {exc}"
        ) from exc
    obj: Any = mod
    for part in attr_path.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError as exc:
            raise ComponentConfigurationError(
                f"FAIL-FAST: rebuild_callable attribute {attr_path!r} "
                f"not found on module {module_path!r}: {exc}"
            ) from exc
    if not callable(obj):
        raise ComponentConfigurationError(
            f"FAIL-FAST: rebuild_callable {spec!r} resolved to non-callable "
            f"{type(obj).__name__}"
        )
    return obj


# ---------------------------------------------------------------------------
# Code identity capture
# ---------------------------------------------------------------------------

def _capture_code_identity() -> Dict[str, Any]:
    """Per G5 spec: capture python_version + git_sha (when available) +
    container_image_digest (env-var-driven)."""
    import os
    import sys

    identity: Dict[str, Any] = {
        "python_version": sys.version.split()[0],
    }
    git_sha = _read_git_sha()
    if git_sha:
        identity["git_sha"] = git_sha
    container_digest = os.environ.get("NANOBRAIN_CONTAINER_DIGEST")
    if container_digest:
        identity["container_image_digest"] = container_digest
    return identity


def _read_git_sha() -> Optional[str]:
    """Best-effort git SHA detection. Returns None when not in a git repo
    or git is unavailable (e.g., container without git installed)."""
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=2,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass
    return None
