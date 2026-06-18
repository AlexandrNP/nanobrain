"""Per-execution teardown helpers for RheaFileToolStep — evict the per-call ProxyStore Redis
keys + release the Stores when a tool execution ends, so Redis does not grow run-over-run (the
rhea SERVER stays online; only the ephemera are torn down). Best-effort: teardown NEVER raises.

Tests the helpers in isolation (no rhea/proxystore import needed) via a fake-self carrying an
``nb_logger`` — the same shape the bound method sees at runtime.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

from nanobrain.library.steps.rhea_file_tool_step import RheaFileToolStep


class _FakeRedis:
    def __init__(self, *, boom: bool = False):
        self.deleted: list[str] = []
        self._boom = boom

    def delete(self, key):
        if self._boom:
            raise RuntimeError("redis down")
        self.deleted.append(key)


def _fake_self():
    return SimpleNamespace(name="rfts", nb_logger=logging.getLogger("test.rfts"))


def test_evict_deletes_each_nonempty_key():
    r = _FakeRedis()
    RheaFileToolStep._evict_rhea_keys(_fake_self(), r, ["in-key", "out-1", "out-2"])
    assert r.deleted == ["in-key", "out-1", "out-2"]


def test_evict_skips_empty_keys():
    r = _FakeRedis()
    RheaFileToolStep._evict_rhea_keys(_fake_self(), r, ["", None, "real"])  # type: ignore[list-item]
    assert r.deleted == ["real"]


def test_evict_never_raises_on_redis_error():
    r = _FakeRedis(boom=True)
    # A teardown failure must NOT propagate (observability, not correctness).
    RheaFileToolStep._evict_rhea_keys(_fake_self(), r, ["k"])
    assert r.deleted == []


def test_close_store_calls_close_and_swallows_errors():
    closed = {"n": 0}

    class _Store:
        def close(self):
            closed["n"] += 1

    RheaFileToolStep._close_rhea_store(_Store())
    assert closed["n"] == 1

    class _Boom:
        def close(self):
            raise RuntimeError("boom")

    RheaFileToolStep._close_rhea_store(_Boom())  # must not raise
    RheaFileToolStep._close_rhea_store(object())  # no close() attr -> no-op, no raise
