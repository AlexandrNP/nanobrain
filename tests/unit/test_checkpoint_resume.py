"""Tests for G5 — CheckpointStep + ResumeStep primitives.

Per ``apecx-mcp-integration/docs/nanobrain_capability_gaps.md G5``.

Tests cover:
1. CheckpointStepConfig validation
2. CheckpointStep filesystem backend round-trip
3. CheckpointStep idempotency (re-run with same input is a no-op write)
4. CheckpointStep capture: ['*'] vs explicit list
5. CheckpointStep tampering detection
6. ResumeStep on_missing semantics (fail / skip / rebuild)
7. ResumeStep code-identity warning surfacing
8. End-to-end checkpoint → resume → matching values
9. Stream-shaped values rejected per G5 spec
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

from nanobrain.core.component_base import ComponentConfigurationError
from nanobrain.library.steps import (
    CheckpointStep,
    CheckpointStepConfig,
    ResumeStep,
    ResumeStepConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_cp(tmp: Path, **overrides) -> CheckpointStep:
    cfg = {
        "name": "cp",
        "backend": "filesystem",
        "base_dir": str(tmp / "snapshots"),
        "capture": ["*"],
        "manifest_path": str(tmp / "manifest.json"),
    }
    cfg.update(overrides)
    yml = tmp / f"{cfg['name']}.yml"
    yml.write_text(yaml.safe_dump(cfg))
    return CheckpointStep.from_config(str(yml))


def _build_rs(tmp: Path, **overrides) -> ResumeStep:
    cfg = {"name": "rs"}
    cfg.update(overrides)
    yml = tmp / f"{cfg['name']}.yml"
    yml.write_text(yaml.safe_dump(cfg))
    return ResumeStep.from_config(str(yml))


# ---------------------------------------------------------------------------
# 1. CheckpointStepConfig validation
# ---------------------------------------------------------------------------

class TestCheckpointConfigValidation:

    def _build(self, **kwargs):
        CheckpointStepConfig._allow_direct_instantiation = True
        try:
            return CheckpointStepConfig(**kwargs)
        finally:
            CheckpointStepConfig._allow_direct_instantiation = False

    def test_filesystem_minimal(self):
        cfg = self._build(
            name="t", backend="filesystem", base_dir="/tmp/snap",
            manifest_path="/tmp/m.json",
        )
        assert cfg.backend == "filesystem"

    def test_filesystem_missing_base_dir_fails_fast(self):
        with pytest.raises(Exception) as exc_info:
            self._build(name="t", backend="filesystem", manifest_path="/tmp/m.json")
        assert "FAIL-FAST" in str(exc_info.value)
        assert "base_dir" in str(exc_info.value)

    def test_proxystore_missing_store_name_fails_fast(self):
        with pytest.raises(Exception) as exc_info:
            self._build(name="t", backend="proxystore", manifest_path="/tmp/m.json")
        assert "FAIL-FAST" in str(exc_info.value)
        assert "proxystore_store_name" in str(exc_info.value)

    def test_missing_manifest_path_fails_fast(self):
        with pytest.raises(Exception) as exc_info:
            self._build(name="t", backend="filesystem", base_dir="/tmp/snap")
        assert "FAIL-FAST" in str(exc_info.value)
        assert "manifest_path" in str(exc_info.value)

    def test_unknown_backend_rejected(self):
        with pytest.raises(Exception):
            self._build(name="t", backend="ghost",
                        base_dir="/tmp", manifest_path="/tmp/m.json")


# ---------------------------------------------------------------------------
# 2. CheckpointStep filesystem round-trip
# ---------------------------------------------------------------------------

class TestCheckpointFilesystemRoundTrip:

    def test_basic_capture_and_restore(self):
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                cp = _build_cp(tmp)
                cp_result = await cp.process({"a": 1, "b": [2, 3], "c": {"x": "y"}})
                assert sorted(cp_result["captured"]) == ["a", "b", "c"]

                rs = _build_rs(tmp)
                restored = await rs.process({"manifest_path": cp_result["manifest_path"]})
                assert restored["a"] == 1
                assert restored["b"] == [2, 3]
                assert restored["c"] == {"x": "y"}
                assert restored["_resumed_from_manifest"] == cp_result["manifest_path"]
        asyncio.run(run())

    def test_capture_list_filters(self):
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                cp = _build_cp(tmp, capture=["a", "c"])
                cp_result = await cp.process({"a": 1, "b": 2, "c": 3})
                assert sorted(cp_result["captured"]) == ["a", "c"]

                rs = _build_rs(tmp)
                restored = await rs.process({"manifest_path": cp_result["manifest_path"]})
                assert restored["a"] == 1
                assert restored["c"] == 3
                assert "b" not in restored
        asyncio.run(run())

    def test_capture_missing_keys_fails_fast(self):
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                cp = _build_cp(tmp, capture=["a", "ghost_key"])
                with pytest.raises(ComponentConfigurationError) as exc_info:
                    await cp.process({"a": 1})
                assert "FAIL-FAST" in str(exc_info.value)
                assert "ghost_key" in str(exc_info.value)
        asyncio.run(run())

    def test_input_echoed_through(self):
        """CheckpointStep is transparent: it adds manifest fields but
        echoes the original input so the workflow continues downstream."""
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                cp = _build_cp(tmp)
                cp_result = await cp.process({"x": 1, "y": "hello"})
                assert cp_result["x"] == 1
                assert cp_result["y"] == "hello"
                # Plus checkpoint fields:
                assert "manifest_path" in cp_result
                assert "captured" in cp_result
                assert "manifest" in cp_result
        asyncio.run(run())


# ---------------------------------------------------------------------------
# 3. Idempotency
# ---------------------------------------------------------------------------

class TestCheckpointIdempotency:

    def test_same_input_same_content_files(self):
        """Re-running with the same input should NOT create new content
        files (content-addressed)."""
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                cp = _build_cp(tmp)
                await cp.process({"x": "value"})
                files_after_first = sorted((tmp / "snapshots").glob("*.json"))

                # Different CheckpointStep instance, same input:
                cp2 = _build_cp(tmp, manifest_path=str(tmp / "manifest2.json"))
                await cp2.process({"x": "value"})
                files_after_second = sorted((tmp / "snapshots").glob("*.json"))

                # No new content file — content hash is the same.
                assert files_after_first == files_after_second
        asyncio.run(run())


# ---------------------------------------------------------------------------
# 4. Tampering detection
# ---------------------------------------------------------------------------

class TestTamperingDetection:

    def test_tampered_content_file_fails_fast(self):
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                cp = _build_cp(tmp)
                cp_result = await cp.process({"x": "original"})

                # Tamper with the snapshot file:
                manifest = json.loads(Path(cp_result["manifest_path"]).read_text())
                snapshot_path = Path(manifest["entries"]["x"]["path"])
                snapshot_path.write_text(json.dumps("tampered"))

                rs = _build_rs(tmp)
                with pytest.raises(ComponentConfigurationError) as exc_info:
                    await rs.process({"manifest_path": cp_result["manifest_path"]})
                assert "FAIL-FAST" in str(exc_info.value)
                assert "content_hash mismatch" in str(exc_info.value)
                assert "tampered" in str(exc_info.value)
        asyncio.run(run())

    def test_missing_snapshot_file_fails_fast(self):
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                cp = _build_cp(tmp)
                cp_result = await cp.process({"x": "value"})

                # Delete the snapshot:
                manifest = json.loads(Path(cp_result["manifest_path"]).read_text())
                snapshot_path = Path(manifest["entries"]["x"]["path"])
                snapshot_path.unlink()

                rs = _build_rs(tmp)
                with pytest.raises(ComponentConfigurationError) as exc_info:
                    await rs.process({"manifest_path": cp_result["manifest_path"]})
                assert "FAIL-FAST" in str(exc_info.value)
                assert "file missing" in str(exc_info.value)
        asyncio.run(run())


# ---------------------------------------------------------------------------
# 5. ResumeStep on_missing semantics
# ---------------------------------------------------------------------------

class TestResumeOnMissing:

    def test_on_missing_fail(self):
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                rs = _build_rs(tmp, on_missing="fail")
                with pytest.raises(ComponentConfigurationError) as exc_info:
                    await rs.process({"manifest_path": str(tmp / "ghost.json")})
                assert "FAIL-FAST" in str(exc_info.value)
                assert "not found" in str(exc_info.value)
        asyncio.run(run())

    def test_on_missing_skip_returns_empty(self):
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                rs = _build_rs(tmp, on_missing="skip")
                result = await rs.process({"manifest_path": str(tmp / "ghost.json")})
                assert result == {}
        asyncio.run(run())

    def test_on_missing_rebuild_raises_not_implemented(self):
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                rs = _build_rs(tmp, on_missing="rebuild")
                with pytest.raises(NotImplementedError) as exc_info:
                    await rs.process({"manifest_path": str(tmp / "ghost.json")})
                assert "G5 Step 3" in str(exc_info.value)
        asyncio.run(run())

    def test_missing_manifest_path_in_input(self):
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                rs = _build_rs(tmp)
                with pytest.raises(ComponentConfigurationError) as exc_info:
                    await rs.process({})  # no manifest_path
                assert "FAIL-FAST" in str(exc_info.value)
                assert "manifest_path" in str(exc_info.value)
        asyncio.run(run())


# ---------------------------------------------------------------------------
# 6. Manifest version mismatch
# ---------------------------------------------------------------------------

class TestManifestVersionMismatch:

    def test_unknown_manifest_version_fails_fast(self):
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                # Hand-write a manifest with a future version:
                manifest_path = tmp / "manifest.json"
                manifest_path.write_text(json.dumps({
                    "manifest_version": 999,
                    "step_name": "x",
                    "backend": "filesystem",
                    "captured": [],
                    "entries": {},
                }))
                rs = _build_rs(tmp)
                with pytest.raises(ComponentConfigurationError) as exc_info:
                    await rs.process({"manifest_path": str(manifest_path)})
                assert "FAIL-FAST" in str(exc_info.value)
                assert "999" in str(exc_info.value)
        asyncio.run(run())


# ---------------------------------------------------------------------------
# 7. End-to-end multi-key
# ---------------------------------------------------------------------------

class TestEndToEnd:

    def test_complex_payload_round_trip(self):
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                cp = _build_cp(tmp)
                payload = {
                    "small_dict": {"a": 1, "b": [1, 2, 3]},
                    "list_of_dicts": [{"x": 1}, {"y": 2}, {"z": 3}],
                    "nullable": None,
                    "string_value": "hello world",
                    "number": 3.14,
                    "boolean": True,
                }
                cp_result = await cp.process(payload)
                rs = _build_rs(tmp)
                restored = await rs.process({"manifest_path": cp_result["manifest_path"]})
                for key, expected in payload.items():
                    assert restored[key] == expected, f"{key} mismatch"
        asyncio.run(run())


# ---------------------------------------------------------------------------
# 8. Stream rejection
# ---------------------------------------------------------------------------

class TestStreamRejection:

    def test_async_iterator_rejected(self):
        """G5 spec: streams aren't snapshottable."""
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                cp = _build_cp(tmp)

                async def _async_gen():
                    yield 1

                with pytest.raises(ComponentConfigurationError) as exc_info:
                    await cp.process({"stream": _async_gen()})
                assert "FAIL-FAST" in str(exc_info.value)
                assert "stream-shaped" in str(exc_info.value)
        asyncio.run(run())


# ---------------------------------------------------------------------------
# 9. Code identity capture
# ---------------------------------------------------------------------------

class TestCodeIdentityCapture:

    def test_manifest_records_python_version(self):
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                cp = _build_cp(tmp)
                cp_result = await cp.process({"x": 1})
                manifest = cp_result["manifest"]
                assert "code_identity" in manifest
                assert "python_version" in manifest["code_identity"]
                assert "." in manifest["code_identity"]["python_version"]
        asyncio.run(run())
