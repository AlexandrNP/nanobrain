"""DockerMCPWorker — spawn + health-check + tear down an MCP worker container.

A nanobrain-native lifecycle manager for MCP servers that ship as
Docker images (Rhea is the motivating case, but nothing here is
Rhea-specific — the image, port, env, and MCP URL are all caller
config).

Why this exists
---------------

A pipeline that depends on an MCP worker should not assume "someone
else already started it." It should own the worker's lifecycle:
check whether one is already up, spawn one if not, wait until it is
genuinely answering the MCP handshake, and tear down what it spawned.
That is exactly what this class does, and it does it
FAIL-LOUD at every step:

* Docker not installed / daemon not running -> ``ComponentConfigurationError``
  with the exact remediation.
* The image is not present locally -> ``ComponentConfigurationError``
  naming the image and how to build/pull it. (We do NOT silently
  ``docker pull`` — that can be a multi-GB surprise.)
* The container starts but never answers the MCP handshake within the
  timeout -> ``ComponentConfigurationError`` with the container's last
  log lines attached, so the operator sees WHY.
* The container dies during startup -> same, with logs.

A worker that "looks up" but never answers the handshake is the
silent-failure shape this class is built to prevent — ``ensure_running``
returns only when an ``initialize`` round-trip has actually succeeded.

Reuse vs. spawn
---------------

``ensure_running`` first probes the MCP URL. If an MCP server already
answers there, it is REUSED (``was_spawned`` stays False) and ``stop``
is a no-op — we never tear down a worker we did not start. Only when
nothing answers do we ``docker run`` a fresh container; then ``stop``
(and ``__aexit__``) tear down exactly that container.

Scope (v1)
----------

* Docker only (the ``docker`` CLI). Podman / k8s would be a sibling
  class.
* One container per manager instance.
* The health check is the MCP streamable-HTTP ``initialize`` handshake
  via :class:`~nanobrain.library.tools._mcp_transport.MCPTransport` —
  the same transport every Rhea-facing component uses, so "healthy"
  here means "healthy for the components that will use it."
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
from typing import Any

from nanobrain.core.component_base import ComponentConfigurationError

logger = logging.getLogger(__name__)


class DockerMCPWorker:
    """Owns the lifecycle of one Docker-hosted MCP worker.

    Construct with the image + container name + the MCP URL the worker
    will answer on; call :meth:`ensure_running` before using the
    worker and :meth:`stop` (or use the async context manager) after.
    """

    def __init__(
        self,
        *,
        image: str,
        container_name: str,
        mcp_url: str,
        host_port: int,
        container_port: int | None = None,
        env: dict[str, str] | None = None,
        extra_run_args: list[str] | None = None,
        health_timeout_seconds: float = 120.0,
        health_poll_interval_seconds: float = 4.0,
    ) -> None:
        if not image or not isinstance(image, str):
            raise ComponentConfigurationError(
                f"FAIL-FAST: DockerMCPWorker requires a non-empty image; "
                f"got {image!r}"
            )
        if not container_name or not isinstance(container_name, str):
            raise ComponentConfigurationError(
                f"FAIL-FAST: DockerMCPWorker requires a non-empty "
                f"container_name; got {container_name!r}"
            )
        if not mcp_url or not isinstance(mcp_url, str):
            raise ComponentConfigurationError(
                f"FAIL-FAST: DockerMCPWorker requires a non-empty mcp_url; "
                f"got {mcp_url!r}"
            )
        self._image = image
        self._container_name = container_name
        self._mcp_url = mcp_url
        self._host_port = int(host_port)
        self._container_port = int(container_port or host_port)
        self._env = dict(env or {})
        self._extra_run_args = list(extra_run_args or [])
        self._health_timeout = float(health_timeout_seconds)
        self._health_poll = float(health_poll_interval_seconds)
        self._was_spawned = False

    # ---- properties -----------------------------------------------------

    @property
    def mcp_url(self) -> str:
        return self._mcp_url

    @property
    def container_name(self) -> str:
        return self._container_name

    @property
    def was_spawned(self) -> bool:
        """True iff THIS manager started the container (vs. reused one)."""
        return self._was_spawned

    # ---- lifecycle ------------------------------------------------------

    async def ensure_running(self) -> str:
        """Ensure an MCP worker is answering at ``mcp_url``.

        If one already answers, it is reused. Otherwise a fresh
        container is spawned and we block until it answers the MCP
        ``initialize`` handshake (or FAIL-LOUD on timeout / death).

        Returns the ``mcp_url``. Idempotent.
        """
        if await self._is_responding():
            logger.info(
                "DockerMCPWorker: an MCP server already answers at %s — "
                "reusing it (will NOT tear it down)",
                self._mcp_url,
            )
            self._was_spawned = False
            return self._mcp_url

        self._require_docker()
        self._require_image()
        self._docker_run()
        self._was_spawned = True

        # Poll until the worker answers the MCP handshake.
        deadline = asyncio.get_event_loop().time() + self._health_timeout
        while asyncio.get_event_loop().time() < deadline:
            if not self._container_is_up():
                logs = self._container_logs(tail=30)
                raise ComponentConfigurationError(
                    f"FAIL-FAST: DockerMCPWorker container "
                    f"{self._container_name!r} died during startup. "
                    f"Last logs:\n{logs}"
                )
            if await self._is_responding():
                logger.info(
                    "DockerMCPWorker: %s is up and answering the MCP "
                    "handshake at %s",
                    self._container_name,
                    self._mcp_url,
                )
                return self._mcp_url
            await asyncio.sleep(self._health_poll)

        logs = self._container_logs(tail=30)
        raise ComponentConfigurationError(
            f"FAIL-FAST: DockerMCPWorker container {self._container_name!r} "
            f"started but never answered the MCP handshake at "
            f"{self._mcp_url} within {self._health_timeout:.0f}s. The "
            f"container is running but the MCP endpoint is unreachable — "
            f"check the bind host (the server may be listening on the "
            f"container's loopback instead of 0.0.0.0). Last logs:\n{logs}"
        )

    async def stop(self) -> None:
        """Stop + remove the container — but ONLY if this manager spawned it.

        A reused worker is left untouched. Idempotent.
        """
        if not self._was_spawned:
            return
        subprocess.run(
            ["docker", "rm", "-f", self._container_name],
            capture_output=True,
            text=True,
            check=False,
        )
        self._was_spawned = False
        logger.info(
            "DockerMCPWorker: stopped + removed container %s",
            self._container_name,
        )

    async def __aenter__(self) -> "DockerMCPWorker":
        await self.ensure_running()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.stop()

    # ---- internals ------------------------------------------------------

    async def _is_responding(self) -> bool:
        """True iff an MCP ``initialize`` round-trip succeeds at mcp_url."""
        from nanobrain.library.tools._mcp_transport import MCPTransport  # noqa: PLC0415

        transport = MCPTransport(
            mcp_url=self._mcp_url,
            timeout_seconds=8.0,
            client_name="nanobrain-mcp-worker-healthcheck",
        )
        try:
            # tools/list forces the full initialize handshake AND a real
            # JSON-RPC round-trip — a bound-but-broken server fails here.
            await transport.call("tools/list", {})
            return True
        except Exception:  # noqa: BLE001 — any failure == not (yet) healthy
            return False
        finally:
            await transport.aclose()

    @staticmethod
    def _require_docker() -> None:
        if shutil.which("docker") is None:
            raise ComponentConfigurationError(
                "FAIL-FAST: DockerMCPWorker needs the `docker` CLI on PATH, "
                "and it was not found. Install Docker / Docker Desktop, or "
                "point the pipeline at an already-running MCP worker."
            )
        probe = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            text=True,
            check=False,
        )
        if probe.returncode != 0:
            raise ComponentConfigurationError(
                "FAIL-FAST: the `docker` CLI is installed but the Docker "
                "daemon is not reachable (`docker info` failed). Start "
                f"Docker Desktop / the daemon. stderr: {probe.stderr.strip()[:200]}"
            )

    def _require_image(self) -> None:
        probe = subprocess.run(
            ["docker", "image", "inspect", self._image],
            capture_output=True,
            text=True,
            check=False,
        )
        if probe.returncode != 0:
            raise ComponentConfigurationError(
                f"FAIL-FAST: DockerMCPWorker image {self._image!r} is not "
                f"present locally. We deliberately do NOT `docker pull` "
                f"automatically — that can be a multi-GB surprise. Build or "
                f"pull the image first, then retry."
            )

    def _docker_run(self) -> None:
        # Remove any stale container with the same name (a previous run
        # that was not cleaned up). This is safe: same name == our own
        # prior container.
        subprocess.run(
            ["docker", "rm", "-f", self._container_name],
            capture_output=True,
            text=True,
            check=False,
        )
        cmd = [
            "docker", "run", "-d",
            "--name", self._container_name,
            "-p", f"{self._host_port}:{self._container_port}",
        ]
        for key, value in self._env.items():
            cmd += ["-e", f"{key}={value}"]
        cmd += self._extra_run_args
        cmd.append(self._image)
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise ComponentConfigurationError(
                f"FAIL-FAST: DockerMCPWorker `docker run` failed for "
                f"{self._container_name!r}: {result.stderr.strip()[:400]}"
            )
        logger.info(
            "DockerMCPWorker: spawned container %s from %s",
            self._container_name,
            self._image,
        )

    def _container_is_up(self) -> bool:
        probe = subprocess.run(
            [
                "docker", "ps", "--filter", f"name=^{self._container_name}$",
                "--format", "{{.Names}}",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        return self._container_name in probe.stdout

    def _container_logs(self, *, tail: int = 30) -> str:
        probe = subprocess.run(
            ["docker", "logs", "--tail", str(tail), self._container_name],
            capture_output=True,
            text=True,
            check=False,
        )
        return (probe.stdout + probe.stderr).strip() or "(no logs)"


__all__ = ["DockerMCPWorker"]
