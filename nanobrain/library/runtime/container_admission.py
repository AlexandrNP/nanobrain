"""Process-wide admission control for one-shot ``docker run`` spawns.

Bound the number of containers running SIMULTANEOUSLY across the whole process so N concurrent
callers each driving a container (a containerized tool — PyMOL SASA, a sandbox, …) cannot pin the
host's RAM with no cap. Hold a slot for the container's WHOLE lifetime (spawn → ``communicate()``
returns / is killed), because the host RAM is consumed while it runs::

    from nanobrain.library.runtime.container_admission import acquire_container_slot

    async with acquire_container_slot():
        proc = await asyncio.create_subprocess_exec(*argv, ...)
        await proc.communicate()

Ported from apecx (where it guarded the open-endpoint PyMOL/sandbox spawns) into the framework so
every docker-backed nanobrain tool shares one count-based cap (default 4; override with
``NANOBRAIN_MAX_CONCURRENT_DOCKER_RUNS``). The goal is to turn "host OOM" into "container waits its
turn", not to perfectly account for heterogeneous container sizes.
"""

from __future__ import annotations

import asyncio
import os

#: Env var that overrides the default simultaneous-container cap.
ENV_VAR = "NANOBRAIN_MAX_CONCURRENT_DOCKER_RUNS"
_DEFAULT_MAX = 4

# An ``asyncio.Semaphore`` binds to the loop running when it is created; a module-import-time instance
# would bind to the wrong loop (or none). Create it lazily and rebind if the running loop changes
# (mirrors ``Workflow._get_run_lock``'s per-loop-lazy pattern). The cap binds across spawns sharing ONE
# loop — the realistic single-loop server topology; a multi-loop driver would get one semaphore per loop.
_semaphore: asyncio.Semaphore | None = None
_semaphore_loop: asyncio.AbstractEventLoop | None = None


def _max_slots() -> int:
    """Resolve the cap from the env var (fail-loud on garbage / non-positive)."""
    raw = os.environ.get(ENV_VAR)
    if raw is None:
        return _DEFAULT_MAX
    value = int(raw)  # ValueError on non-int — fail loud, do not silently default
    if value < 1:
        raise ValueError(f"{ENV_VAR} must be >= 1, got {value}")
    return value


def acquire_container_slot() -> asyncio.Semaphore:
    """Return the process-wide code-exec-container semaphore for ``async with``.

    ``async with acquire_container_slot():`` blocks until a slot is free, holds it for the body, and
    releases on exit (including on exception). Must be called from within a running event loop.
    """
    global _semaphore, _semaphore_loop
    loop = asyncio.get_running_loop()
    if _semaphore is None or _semaphore_loop is not loop:
        _semaphore = asyncio.Semaphore(_max_slots())
        _semaphore_loop = loop
    return _semaphore


def _reset_for_test() -> None:
    """Drop the cached semaphore so the next acquire re-reads ``ENV_VAR``. Tests only."""
    global _semaphore, _semaphore_loop
    _semaphore = None
    _semaphore_loop = None
