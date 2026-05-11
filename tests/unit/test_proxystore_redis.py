"""Tests for the RedisKey rebuild path (extends G5 Step 2 coverage).

Per ``apecx-mcp-integration/docs/nanobrain_capability_gaps.md G5``.

Coverage:
1. CheckpointStepConfig: connector_kind='redis' requires redis_host + redis_port.
2. CheckpointStepConfig: connector_kind='file' still requires store_dir.
3. CheckpointStep with redis backend writes connector_kind+host+port to manifest.
4. INTEGRATION: end-to-end cross-process resume against a real Redis when
   ``REDIS_TEST_HOST``/``REDIS_TEST_PORT`` env vars are set; otherwise SKIPPED.
5. INTEGRATION: a fresh ResumeStep (no checkpoint_step kwarg) reads the
   manifest, re-registers the Redis store from hints, and restores values.

CI / local-dev recipe for the integration tests:

    docker run --rm -d --name nb-redis-test -p 6380:6379 redis:7
    REDIS_TEST_HOST=localhost REDIS_TEST_PORT=6380 \\
        .venv/bin/python -m pytest \\
        tests/unit/test_proxystore_redis.py -v
    docker rm -f nb-redis-test
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import uuid
from pathlib import Path

import pytest
import yaml

from nanobrain.library.steps import (
    CheckpointStep,
    CheckpointStepConfig,
    ResumeStep,
)


_REDIS_HOST = os.environ.get("REDIS_TEST_HOST")
_REDIS_PORT = os.environ.get("REDIS_TEST_PORT")


def _redis_client_importable() -> bool:
    """The proxystore Redis connector requires the ``redis`` Python
    client AND ``proxystore[redis]``. CI runners may ship the server
    container but not the Python deps; tests must skip cleanly in
    that case rather than erroring at fixture-setup time.
    Silent-failure-prevention check — guards against the 2026-05-11
    CI failure where ``REDIS_TEST_HOST`` was set but ``redis`` was
    not pip-installed."""
    try:
        import redis  # noqa: F401
        return True
    except ImportError:
        return False


_redis_skip = pytest.mark.skipif(
    not (_REDIS_HOST and _REDIS_PORT and _redis_client_importable()),
    reason=(
        "Redis tests skipped — need REDIS_TEST_HOST + REDIS_TEST_PORT "
        "env vars AND `pip install redis` (Python client). CI ships "
        "the Redis server container but does not pip-install the "
        "client by default; add ``redis`` to your install "
        "(e.g. ``pip install redis proxystore[redis]``) to enable."
    ),
)


# ---------------------------------------------------------------------------
# 1-2. Config validation
# ---------------------------------------------------------------------------

class TestRedisConfigValidation:

    def _build(self, **kwargs):
        CheckpointStepConfig._allow_direct_instantiation = True
        try:
            return CheckpointStepConfig(**kwargs)
        finally:
            CheckpointStepConfig._allow_direct_instantiation = False

    def test_redis_minimal_accepted(self):
        cfg = self._build(
            name="t", backend="proxystore",
            proxystore_store_name="s",
            proxystore_connector_kind="redis",
            proxystore_redis_host="localhost",
            proxystore_redis_port=6379,
            manifest_path="/tmp/m.json",
        )
        assert cfg.proxystore_connector_kind == "redis"
        assert cfg.proxystore_redis_host == "localhost"
        assert cfg.proxystore_redis_port == 6379

    def test_redis_missing_host_fails_fast(self):
        with pytest.raises(Exception) as exc_info:
            self._build(
                name="t", backend="proxystore",
                proxystore_store_name="s",
                proxystore_connector_kind="redis",
                proxystore_redis_port=6379,
                manifest_path="/tmp/m.json",
            )
        assert "FAIL-FAST" in str(exc_info.value)
        assert "redis_host" in str(exc_info.value)

    def test_redis_missing_port_fails_fast(self):
        with pytest.raises(Exception) as exc_info:
            self._build(
                name="t", backend="proxystore",
                proxystore_store_name="s",
                proxystore_connector_kind="redis",
                proxystore_redis_host="localhost",
                manifest_path="/tmp/m.json",
            )
        assert "FAIL-FAST" in str(exc_info.value)
        assert "redis_port" in str(exc_info.value)

    def test_file_still_requires_store_dir(self):
        """Make sure the new validator branch didn't regress the
        existing file-connector requirement."""
        with pytest.raises(Exception) as exc_info:
            self._build(
                name="t", backend="proxystore",
                proxystore_store_name="s",
                proxystore_connector_kind="file",
                manifest_path="/tmp/m.json",
            )
        assert "FAIL-FAST" in str(exc_info.value)
        assert "store_dir" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 3. Manifest hint round-trip (no real Redis required — config-only)
# ---------------------------------------------------------------------------

class TestManifestHints:

    @_redis_skip
    def test_manifest_records_redis_host_and_port(self):
        """Even though this asserts on manifest content (not Redis I/O),
        we still need a real Redis to write the entry — CheckpointStep
        actually puts the value into the Store. So gate this on the
        env vars too."""
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                manifest_path = tmp / "manifest.json"
                store_name = f"test-{uuid.uuid4().hex[:8]}"
                yml = tmp / "cp.yml"
                yml.write_text(yaml.safe_dump({
                    "name": "cp",
                    "backend": "proxystore",
                    "proxystore_store_name": store_name,
                    "proxystore_connector_kind": "redis",
                    "proxystore_redis_host": _REDIS_HOST,
                    "proxystore_redis_port": int(_REDIS_PORT),
                    "capture": ["*"],
                    "manifest_path": str(manifest_path),
                }))
                cp = CheckpointStep.from_config(str(yml))
                await cp.process({"x": 1})
                manifest = json.loads(manifest_path.read_text())
                first = next(iter(manifest["entries"].values()))
                assert first["connector_kind"] == "redis"
                assert first["redis_host"] == _REDIS_HOST
                assert first["redis_port"] == int(_REDIS_PORT)
                # File-only hints should NOT appear:
                assert "store_dir" not in first
        asyncio.run(run())


# ---------------------------------------------------------------------------
# 4-5. Cross-process resume against real Redis
# ---------------------------------------------------------------------------

@_redis_skip
class TestCrossProcessResumeRedis:

    def test_fresh_resume_step_no_checkpoint_kwarg(self):
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                manifest_path = tmp / "manifest.json"
                store_name = f"e2e-redis-{uuid.uuid4().hex[:8]}"

                cp_yml = tmp / "cp.yml"
                cp_yml.write_text(yaml.safe_dump({
                    "name": "cp",
                    "backend": "proxystore",
                    "proxystore_store_name": store_name,
                    "proxystore_connector_kind": "redis",
                    "proxystore_redis_host": _REDIS_HOST,
                    "proxystore_redis_port": int(_REDIS_PORT),
                    "capture": ["*"],
                    "manifest_path": str(manifest_path),
                }))
                cp = CheckpointStep.from_config(str(cp_yml))
                await cp.process({
                    "a": 1, "b": [2, 3], "c": "hello",
                    "nested": {"deep": {"v": True}},
                })

                # Fresh ResumeStep — different Step instance; no
                # checkpoint_step kwarg. The resolver must read the
                # manifest's redis hints and re-register the Store.
                rs_yml = tmp / "rs.yml"
                rs_yml.write_text(yaml.safe_dump({"name": "rs"}))
                rs = ResumeStep.from_config(str(rs_yml))
                result = await rs.process({
                    "manifest_path": str(manifest_path),
                })
                assert result["a"] == 1
                assert result["b"] == [2, 3]
                assert result["c"] == "hello"
                assert result["nested"] == {"deep": {"v": True}}
        asyncio.run(run())

    def test_filekey_serialization_for_redis_keys(self):
        """The G5 Step 2 generic serializer should handle RedisKey
        identically to FileKey because both are NamedTuples."""
        from proxystore.connectors.redis import RedisKey
        from nanobrain.library.steps.checkpoint_resume import (
            _serialize_proxy_key, _deserialize_proxy_key,
        )
        k = RedisKey(redis_key="some-uuid-here")
        blob = _serialize_proxy_key(k)
        assert blob["class_path"].endswith("RedisKey")
        assert blob["fields"] == {"redis_key": "some-uuid-here"}
        # JSON round-trip
        s = json.dumps(blob)
        k2 = _deserialize_proxy_key(json.loads(s))
        assert k == k2
