"""Tests for G5 Step 3 — ResumeStep on_missing='rebuild' path.

Per ``apecx-mcp-integration/docs/nanobrain_capability_gaps.md G5``.

Coverage:
1. ResumeStepConfig: rebuild requires rebuild_callable + rebuild_base_dir.
2. rebuild_callable resolution (sync + async).
3. End-to-end: missing manifest → rebuild → fresh manifest written →
   subsequent resume hits the cache.
4. rebuild_callable returning non-dict: FAIL-FAST.
5. rebuild_callable resolution failure: FAIL-FAST.
6. rebuild result dict's '_'-prefixed keys are not snapshotted (avoid
   self-recursion in _resumed_at, etc.).
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

import pytest
import yaml

from nanobrain.core.component_base import ComponentConfigurationError
from nanobrain.library.steps import ResumeStep, ResumeStepConfig


# ---------------------------------------------------------------------------
# Module-level rebuild callables (must be importable for dotted-path)
# ---------------------------------------------------------------------------

async def _async_rebuild(input_data):
    return {"a": 1, "b": [2, 3], "c": {"x": "y"}}


def _sync_rebuild(input_data):
    return {"sync": True, "input_keys": list(input_data.keys())}


def _bad_rebuild_returns_int(input_data):
    return 42


# Counts how many times the rebuild was invoked — used to verify
# subsequent resumes hit the cache, not the rebuild path.
_rebuild_invocations = []


def _counting_rebuild(input_data):
    _rebuild_invocations.append(input_data)
    return {"count": len(_rebuild_invocations)}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_resume(tmp: Path, **overrides) -> ResumeStep:
    cfg = {
        "name": "rs",
        "on_missing": "rebuild",
        "rebuild_callable": __name__ + "._async_rebuild",
        "rebuild_base_dir": str(tmp / "rebuild_snapshots"),
    }
    cfg.update(overrides)
    yml = tmp / "rs.yml"
    yml.write_text(yaml.safe_dump(cfg))
    return ResumeStep.from_config(str(yml))


# ---------------------------------------------------------------------------
# 1. Config validation
# ---------------------------------------------------------------------------

class TestRebuildConfig:

    def _build(self, **kwargs):
        ResumeStepConfig._allow_direct_instantiation = True
        try:
            return ResumeStepConfig(**kwargs)
        finally:
            ResumeStepConfig._allow_direct_instantiation = False

    def test_rebuild_requires_callable(self):
        with pytest.raises(Exception) as exc_info:
            self._build(name="rs", on_missing="rebuild",
                        rebuild_base_dir="/tmp/x")
        assert "FAIL-FAST" in str(exc_info.value)
        assert "rebuild_callable" in str(exc_info.value)

    def test_rebuild_requires_base_dir(self):
        with pytest.raises(Exception) as exc_info:
            self._build(name="rs", on_missing="rebuild",
                        rebuild_callable="some.fn")
        assert "FAIL-FAST" in str(exc_info.value)
        assert "rebuild_base_dir" in str(exc_info.value)

    def test_fail_skip_dont_require_rebuild_fields(self):
        cfg1 = self._build(name="rs", on_missing="fail")
        cfg2 = self._build(name="rs", on_missing="skip")
        assert cfg1.rebuild_callable is None
        assert cfg2.rebuild_callable is None


# ---------------------------------------------------------------------------
# 2-3. Rebuild path end-to-end
# ---------------------------------------------------------------------------

class TestRebuildEndToEnd:

    def test_async_rebuild_writes_manifest(self):
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                rs = _build_resume(tmp)
                manifest_path = tmp / "manifest.json"
                result = await rs.process({"manifest_path": str(manifest_path)})
                assert result["a"] == 1
                assert result["b"] == [2, 3]
                assert result["c"] == {"x": "y"}
                assert result["_rebuilt"] is True
                assert manifest_path.is_file()

                # Verify manifest contents are well-formed
                manifest = json.loads(manifest_path.read_text())
                assert manifest["manifest_version"] == 1
                assert sorted(manifest["captured"]) == ["a", "b", "c"]
                assert "rebuilt_from_callable" in manifest
                assert manifest["rebuilt_from_callable"].endswith("_async_rebuild")
        asyncio.run(run())

    def test_sync_rebuild_works(self):
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                rs = _build_resume(tmp,
                                   rebuild_callable=__name__ + "._sync_rebuild")
                manifest_path = tmp / "manifest.json"
                result = await rs.process({"manifest_path": str(manifest_path),
                                            "extra": "ignore_me"})
                assert result["sync"] is True
                assert "manifest_path" in result["input_keys"]
        asyncio.run(run())

    def test_subsequent_resume_hits_cache(self):
        """The value-add of rebuild: once the manifest is written, the
        next process() call restores from cache instead of invoking
        the rebuild callable again."""
        async def run():
            global _rebuild_invocations
            _rebuild_invocations = []
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                rs = _build_resume(tmp,
                                   rebuild_callable=__name__ + "._counting_rebuild")
                manifest_path = tmp / "manifest.json"

                # First call — rebuild fires
                r1 = await rs.process({"manifest_path": str(manifest_path)})
                assert r1["count"] == 1
                assert len(_rebuild_invocations) == 1

                # Second call — cache hit
                r2 = await rs.process({"manifest_path": str(manifest_path)})
                assert r2["count"] == 1, \
                    "rebuild fired again instead of using cached manifest"
                assert len(_rebuild_invocations) == 1
                # The cached resume returns _resumed_from_manifest but NOT
                # _rebuilt (only the rebuild path sets that marker).
                assert "_resumed_from_manifest" in r2
                assert r2.get("_rebuilt") is None
        asyncio.run(run())


# ---------------------------------------------------------------------------
# 4-5. Failure modes
# ---------------------------------------------------------------------------

class TestRebuildFailures:

    def test_callable_returns_non_dict_fails_fast(self):
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                rs = _build_resume(tmp,
                                   rebuild_callable=__name__ + "._bad_rebuild_returns_int")
                manifest_path = tmp / "manifest.json"
                with pytest.raises(ComponentConfigurationError) as exc_info:
                    await rs.process({"manifest_path": str(manifest_path)})
                assert "FAIL-FAST" in str(exc_info.value)
                assert "rebuild_callable" in str(exc_info.value)
                assert "expected dict" in str(exc_info.value)
        asyncio.run(run())

    def test_unresolvable_callable_fails_fast(self):
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                rs = _build_resume(tmp,
                                   rebuild_callable="nonexistent_pkg_z2.fn")
                manifest_path = tmp / "manifest.json"
                with pytest.raises(ComponentConfigurationError) as exc_info:
                    await rs.process({"manifest_path": str(manifest_path)})
                assert "FAIL-FAST" in str(exc_info.value)
                assert "not importable" in str(exc_info.value)
        asyncio.run(run())


# ---------------------------------------------------------------------------
# 6. Bookkeeping keys not snapshotted
# ---------------------------------------------------------------------------

async def _rebuild_with_bookkeeping(input_data):
    return {
        "real_value": "data",
        "_internal_marker": "should_not_snapshot",
    }


class TestBookkeepingKeysIgnored:

    def test_underscore_keys_not_in_manifest(self):
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                rs = _build_resume(tmp,
                                   rebuild_callable=__name__ + "._rebuild_with_bookkeeping")
                manifest_path = tmp / "manifest.json"
                result = await rs.process({"manifest_path": str(manifest_path)})
                # The result still has the _-prefixed key (pass-through):
                assert result["_internal_marker"] == "should_not_snapshot"
                # But the manifest must NOT have it:
                manifest = json.loads(manifest_path.read_text())
                assert "real_value" in manifest["captured"]
                assert "_internal_marker" not in manifest["captured"]
        asyncio.run(run())
