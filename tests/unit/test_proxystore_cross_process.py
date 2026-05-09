"""Tests for G5 Step 2 — ProxyStore Key cross-process serialization.

Per ``apecx-mcp-integration/docs/nanobrain_capability_gaps.md G5``.

Coverage:
1. _serialize_proxy_key returns class_path + fields for a NamedTuple Key.
2. Round-trip via JSON: serialize → json.dumps → json.loads → deserialize → equal.
3. Non-NamedTuple Key: FAIL-FAST.
4. _resolve_proxystore_storage finds an already-registered store.
5. _resolve_proxystore_storage rebuilds a Store from manifest hints when
   not pre-registered (the cross-process happy path).
6. End-to-end: CheckpointStep writes a proxystore manifest with key_serialized
   + connector_kind + store_dir; a fresh ResumeStep (no checkpoint_step
   kwarg) resumes successfully.
7. Manifest legacy compatibility: a manifest WITHOUT key_serialized
   FAIL-FASTs with a clear "legacy v1 manifest" message.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import uuid
from pathlib import Path

import pytest
import yaml

from nanobrain.core.component_base import ComponentConfigurationError
from nanobrain.library.steps import CheckpointStep, ResumeStep
from nanobrain.library.steps.checkpoint_resume import (
    _serialize_proxy_key,
    _deserialize_proxy_key,
    _resolve_proxystore_storage,
)


# ---------------------------------------------------------------------------
# 1-3. Key serialization
# ---------------------------------------------------------------------------

class TestKeySerialization:

    def test_filekey_round_trip(self):
        """The dominant case: ProxyStore's FileKey is a NamedTuple."""
        from proxystore.connectors.file import FileKey
        k = FileKey(filename="abc-123")
        blob = _serialize_proxy_key(k)
        assert "class_path" in blob
        assert blob["class_path"].endswith("FileKey")
        assert blob["fields"] == {"filename": "abc-123"}

        # JSON round-trip
        s = json.dumps(blob)
        blob2 = json.loads(s)
        k2 = _deserialize_proxy_key(blob2)
        assert k == k2

    def test_non_namedtuple_key_fails_fast(self):
        class _NotNamedTuple:
            pass
        with pytest.raises(ComponentConfigurationError) as exc_info:
            _serialize_proxy_key(_NotNamedTuple())
        assert "FAIL-FAST" in str(exc_info.value)
        assert "NamedTuple" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 4-5. Storage resolver
# ---------------------------------------------------------------------------

class TestStorageResolver:

    def test_finds_already_registered_store(self):
        """When the operator pre-registered the store via deployment
        startup, the resolver reuses it without consulting the manifest."""
        from proxystore.store import Store, register_store
        from proxystore.connectors.file import FileConnector

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            store_name = f"prereg-{uuid.uuid4().hex[:8]}"
            connector = FileConnector(str(tmp / "pxs"))
            store = Store(store_name, connector)
            register_store(store)

            # No manifest needed; resolver short-circuits on the
            # already-registered store.
            storage = _resolve_proxystore_storage(
                store_name, manifest_path=tmp / "no-such-manifest.json",
            )
            assert storage._store.name == store_name

    def test_rebuilds_from_manifest_hints(self):
        """The cross-process happy path: the deployment did NOT
        pre-register the store; the resolver reads the manifest's
        connector_kind + store_dir and re-registers an equivalent
        FileConnector under the same store_name."""
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                manifest_path = tmp / "manifest.json"
                store_name = f"rebuild-{uuid.uuid4().hex[:8]}"
                # Hand-write a minimal manifest with proxystore hints
                manifest_path.write_text(json.dumps({
                    "manifest_version": 1,
                    "step_name": "cp",
                    "backend": "proxystore",
                    "captured": ["x"],
                    "entries": {
                        "x": {
                            "backend": "proxystore",
                            "store_name": store_name,
                            "connector_kind": "file",
                            "store_dir": str(tmp / "pxs"),
                            "key_serialized": {"class_path": "proxystore.connectors.file.FileKey", "fields": {"filename": "fake"}},
                            "key_repr": "FileKey(filename='fake')",
                            "content_hash": "",
                        },
                    },
                }))
                storage = _resolve_proxystore_storage(store_name, manifest_path)
                assert storage._store.name == store_name
        asyncio.run(run())

    def test_rebuild_fails_fast_when_no_hints(self):
        """A bare manifest with no proxystore entries (or no hints)
        should FAIL-FAST rather than silently returning a broken
        storage."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            manifest_path = tmp / "manifest.json"
            manifest_path.write_text(json.dumps({
                "manifest_version": 1,
                "entries": {},  # no entries
            }))
            with pytest.raises(ComponentConfigurationError) as exc_info:
                _resolve_proxystore_storage(
                    f"unknown-{uuid.uuid4().hex[:8]}", manifest_path,
                )
            assert "FAIL-FAST" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 6. End-to-end cross-process resume
# ---------------------------------------------------------------------------

class TestEndToEndCrossProcessResume:

    def test_fresh_resume_step_no_checkpoint_kwarg(self):
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                store_dir = tmp / "pxs"
                manifest_path = tmp / "manifest.json"
                store_name = f"e2e-{uuid.uuid4().hex[:8]}"

                # CheckpointStep
                cp_yml = tmp / "cp.yml"
                cp_yml.write_text(yaml.safe_dump({
                    "name": "cp",
                    "backend": "proxystore",
                    "proxystore_store_name": store_name,
                    "proxystore_store_dir": str(store_dir),
                    "capture": ["*"],
                    "manifest_path": str(manifest_path),
                }))
                cp = CheckpointStep.from_config(str(cp_yml))
                await cp.process({"a": 1, "b": [2, 3], "c": "hello"})

                # Verify the manifest carries the new fields
                manifest = json.loads(manifest_path.read_text())
                first_entry = next(iter(manifest["entries"].values()))
                assert "key_serialized" in first_entry
                assert "connector_kind" in first_entry
                assert "store_dir" in first_entry

                # Fresh ResumeStep, no checkpoint_step kwarg
                rs_yml = tmp / "rs.yml"
                rs_yml.write_text(yaml.safe_dump({"name": "rs"}))
                rs = ResumeStep.from_config(str(rs_yml))
                result = await rs.process({"manifest_path": str(manifest_path)})
                assert result["a"] == 1
                assert result["b"] == [2, 3]
                assert result["c"] == "hello"
        asyncio.run(run())


# ---------------------------------------------------------------------------
# 7. Legacy manifest detection
# ---------------------------------------------------------------------------

class TestLegacyManifest:

    def test_legacy_manifest_without_key_serialized_fails_fast(self):
        """A manifest written by a pre-G5-Step-2 CheckpointStep won't
        have key_serialized; resume must FAIL-FAST with a clear
        upgrade hint."""
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                manifest_path = tmp / "manifest.json"
                store_name = f"legacy-{uuid.uuid4().hex[:8]}"
                # Pre-register a store so the resolver doesn't fall
                # through to "no store registered" instead.
                from proxystore.store import Store, register_store
                from proxystore.connectors.file import FileConnector
                connector = FileConnector(str(tmp / "pxs"))
                register_store(Store(store_name, connector))

                # Hand-write a "legacy v1" manifest WITHOUT key_serialized.
                manifest_path.write_text(json.dumps({
                    "manifest_version": 1,
                    "step_name": "cp",
                    "backend": "proxystore",
                    "captured": ["x"],
                    "entries": {
                        "x": {
                            "backend": "proxystore",
                            "store_name": store_name,
                            "key_repr": "FileKey(filename='abc')",
                            # NO key_serialized
                            "content_hash": "",
                        },
                    },
                }))
                rs_yml = tmp / "rs.yml"
                rs_yml.write_text(yaml.safe_dump({"name": "rs"}))
                rs = ResumeStep.from_config(str(rs_yml))
                with pytest.raises(ComponentConfigurationError) as exc_info:
                    await rs.process({"manifest_path": str(manifest_path)})
                assert "FAIL-FAST" in str(exc_info.value)
                assert "key_serialized" in str(exc_info.value)
        asyncio.run(run())
