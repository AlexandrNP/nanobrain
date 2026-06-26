"""Build a Docker image from a Dockerfile if it is not already present — the auto-provision gap
``DockerMCPWorker`` leaves (it FAIL-LOUDs on image-absent and never builds/pulls).

Generic + shared: any containerized nanobrain tool calls :func:`ensure_docker_image_built` before its
first ``docker run`` so the image self-provisions on first use (no separate install script). Idempotent
and concurrency-safe — a per-image-tag build-lock means two concurrent callers needing the SAME tag
serialize (one builds, the rest wait then skip), while different tags build in parallel. Local build
only: NO ``docker pull`` (a registry pull cannot produce a locally-defined image, and would risk a
name-squat surprise) — the image is defined by its Dockerfile.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from pathlib import Path

log = logging.getLogger(__name__)


class DockerImageBuildError(RuntimeError):
    """FAIL-LOUD: the image could not be made present (docker missing / daemon down / build failed)."""


# Per-image-tag build lock, per-event-loop-lazy (mirrors container_admission's semaphore shape) so the
# build of one tag is never issued twice concurrently; distinct tags do not block each other.
_locks: dict[str, asyncio.Lock] = {}
_locks_loop: asyncio.AbstractEventLoop | None = None


def _build_lock(image_tag: str) -> asyncio.Lock:
    global _locks, _locks_loop
    loop = asyncio.get_running_loop()
    if _locks_loop is not loop:
        _locks = {}
        _locks_loop = loop
    return _locks.setdefault(image_tag, asyncio.Lock())


async def _run(*argv: str, timeout: float) -> tuple[int, str]:
    """Run a docker command, capturing combined stdout+stderr. Never blocks the event loop."""
    import shutil

    if shutil.which("docker") is None:
        raise DockerImageBuildError("the docker CLI is not on PATH — install Docker")
    proc = await asyncio.create_subprocess_exec(
        *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        proc.kill()
        await proc.communicate()
        raise DockerImageBuildError(f"docker command timed out after {timeout:.0f}s: {' '.join(argv)}")
    return proc.returncode or 0, (out or b"").decode("utf-8", "replace")


async def _image_present(image_tag: str) -> bool:
    rc, _ = await _run("docker", "image", "inspect", image_tag, timeout=20)
    return rc == 0


async def ensure_docker_image_built(
    *,
    dockerfile_path: str,
    build_context: str,
    image_tag: str,
    build_args: dict[str, str] | None = None,
    timeout: float = 1800.0,
    on_progress: Callable[[str], None] | None = None,
) -> None:
    """Ensure ``image_tag`` is present locally; build it from ``dockerfile_path`` (with context
    ``build_context``) if absent. Idempotent + concurrency-safe.

    Raises :class:`DockerImageBuildError` (FAIL-LOUD) on: docker CLI absent, daemon not running,
    empty tag, missing Dockerfile / context, or a failed build (with the build log tail attached).
    Returns silently when the image is already present (or built successfully).
    """
    if not image_tag:
        raise DockerImageBuildError("image_tag must be non-empty")
    rc, _ = await _run("docker", "version", "--format", "{{.Server.Version}}", timeout=20)
    if rc != 0:
        raise DockerImageBuildError("the Docker daemon is not running — start Docker")
    if await _image_present(image_tag):
        return
    progress = on_progress or log.info
    async with _build_lock(image_tag):
        if await _image_present(image_tag):  # a concurrent caller built it while we waited
            return
        df, ctx = Path(dockerfile_path), Path(build_context)
        if not df.is_file():
            raise DockerImageBuildError(f"Dockerfile not found: {dockerfile_path}")
        if not ctx.is_dir():
            raise DockerImageBuildError(f"build context not a directory: {build_context}")
        argv = ["docker", "build", "-t", image_tag, "-f", str(df)]
        for key, value in (build_args or {}).items():
            argv += ["--build-arg", f"{key}={value}"]
        argv.append(str(ctx))
        progress(f"building image {image_tag} from {dockerfile_path} (first use; this may take minutes)")
        rc, out = await _run(*argv, timeout=timeout)
        if rc != 0 or not await _image_present(image_tag):
            tail = "\n".join(out.strip().splitlines()[-20:])
            raise DockerImageBuildError(f"docker build for {image_tag} failed (rc={rc}):\n{tail}")
        progress(f"image {image_tag} built")


async def image_digest(image_tag: str) -> str | None:
    """The image's content digest (``sha256:...`` Id) for provenance pinning, or None if unavailable."""
    rc, out = await _run("docker", "image", "inspect", "--format", "{{.Id}}", image_tag, timeout=20)
    digest = out.strip()
    return digest if rc == 0 and digest.startswith("sha256:") else None


def _reset_for_test() -> None:
    """Drop the cached build-locks. Tests only."""
    global _locks, _locks_loop
    _locks = {}
    _locks_loop = None
