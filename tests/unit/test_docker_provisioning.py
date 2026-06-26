"""Unit tests for the shared docker-provisioning helpers (no real docker — a stateful fake).

Covers the one real gap (auto-build-if-absent) + the load-bearing concurrency invariants: the
per-tag build-lock must never double-build, and the container-admission slot must cap concurrency.
"""

from __future__ import annotations

import asyncio

import pytest

from nanobrain.library.runtime import container_admission as ca
from nanobrain.library.runtime import docker_image_builder as dib


class _FakeDocker:
    """Stateful stand-in for ``docker_image_builder._run``: tracks image presence + counts builds."""

    def __init__(self, *, present=False, daemon_up=True, build_ok=True, build_delay=0.0):
        self.present = present
        self.daemon_up = daemon_up
        self.build_ok = build_ok
        self.build_delay = build_delay
        self.build_count = 0

    async def run(self, *argv, timeout):
        sub = argv[1] if len(argv) > 1 else ""
        if sub == "version":
            return (0 if self.daemon_up else 1), "28.1.1\n"
        if argv[1:3] == ("image", "inspect"):
            return (0, "sha256:abc\n") if self.present else (1, "No such image")
        if sub == "build":
            self.build_count += 1
            if self.build_delay:
                await asyncio.sleep(self.build_delay)
            if self.build_ok:
                self.present = True
                return 0, "Successfully built\n"
            return 1, "build error: step 3/5 failed\n"
        return 1, ""


def _patch(monkeypatch, fake):
    monkeypatch.setattr(dib, "_run", fake.run)
    dib._reset_for_test()


@pytest.mark.asyncio
async def test_build_if_absent(monkeypatch):
    fake = _FakeDocker(present=False)
    _patch(monkeypatch, fake)
    await dib.ensure_docker_image_built(dockerfile_path=__file__, build_context=".", image_tag="x:1")
    assert fake.build_count == 1 and fake.present


@pytest.mark.asyncio
async def test_skip_if_present(monkeypatch):
    fake = _FakeDocker(present=True)
    _patch(monkeypatch, fake)
    await dib.ensure_docker_image_built(dockerfile_path=__file__, build_context=".", image_tag="x:1")
    assert fake.build_count == 0


@pytest.mark.asyncio
async def test_no_double_build_under_concurrency(monkeypatch):
    """The load-bearing invariant: N concurrent callers needing the SAME tag build it exactly ONCE."""
    fake = _FakeDocker(present=False, build_delay=0.05)
    _patch(monkeypatch, fake)
    await asyncio.gather(
        *[
            dib.ensure_docker_image_built(dockerfile_path=__file__, build_context=".", image_tag="x:1")
            for _ in range(8)
        ]
    )
    assert fake.build_count == 1


@pytest.mark.asyncio
async def test_fail_loud_daemon_down(monkeypatch):
    _patch(monkeypatch, _FakeDocker(daemon_up=False))
    with pytest.raises(dib.DockerImageBuildError, match="daemon"):
        await dib.ensure_docker_image_built(dockerfile_path=__file__, build_context=".", image_tag="x:1")


@pytest.mark.asyncio
async def test_fail_loud_build_failure(monkeypatch):
    _patch(monkeypatch, _FakeDocker(present=False, build_ok=False))
    with pytest.raises(dib.DockerImageBuildError, match="build error"):
        await dib.ensure_docker_image_built(dockerfile_path=__file__, build_context=".", image_tag="x:1")


@pytest.mark.asyncio
async def test_container_slot_caps_concurrency(monkeypatch):
    monkeypatch.setenv(ca.ENV_VAR, "2")
    ca._reset_for_test()
    state = {"cur": 0, "max": 0}

    async def worker():
        async with ca.acquire_container_slot():
            state["cur"] += 1
            state["max"] = max(state["max"], state["cur"])
            await asyncio.sleep(0.02)
            state["cur"] -= 1

    await asyncio.gather(*[worker() for _ in range(6)])
    assert state["max"] == 2


@pytest.mark.asyncio
async def test_container_slot_bad_cap_fail_loud(monkeypatch):
    monkeypatch.setenv(ca.ENV_VAR, "0")
    ca._reset_for_test()
    with pytest.raises(ValueError, match=">= 1"):
        ca.acquire_container_slot()
