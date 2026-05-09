#!/usr/bin/env python3
"""
Shared vLLM Server for Nanobrain
=================================

Provides a shared vLLM server resource that can be accessed by multiple
Parsl workers concurrently. Uses @shared decorator for singleton pattern.

NO FALLBACKS - If vLLM fails, this FAILS LOUDLY.

Created: 2025-12-03
"""

import asyncio
import logging
import socket
import time
from typing import Optional, Dict, Any
from threading import Lock
from parsl import python_app

from nanobrain.core.shared_resource import shared, get_worker_id

logger = logging.getLogger(__name__)


# ============================================================================
# Parsl Apps for vLLM Server Management
# ============================================================================

@python_app
def start_vllm_server_app(
    model_name: str,
    port: int = 8000,
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    max_model_len: int = 4096,
    server_url_file: str = "/tmp/vllm_server_url.txt"
):
    """
    Start vLLM server on compute node (Parsl @python_app).

    This runs on a Parsl worker (compute node) and blocks for the
    duration of the PBS job.

    Args:
        model_name: HuggingFace model name
        port: Server port
        tensor_parallel_size: Number of XPU tiles for tensor parallelism
        pipeline_parallel_size: Number of nodes for pipeline parallelism
        max_model_len: Maximum sequence length
        server_url_file: File to write server URL

    Returns:
        Server URL

    Raises:
        RuntimeError: If vLLM server fails to start
    """
    import subprocess
    import socket

    # Get compute node hostname
    hostname = socket.gethostname()
    server_url = f"http://{hostname}:{port}/v1"

    # Write server URL to file (for client discovery)
    with open(server_url_file, 'w') as f:
        f.write(server_url)

    print(f"🔥 Starting vLLM server on {hostname}:{port}")
    print(f"📦 Model: {model_name}")
    print(f"⚙️  Tensor parallel: {tensor_parallel_size}")
    print(f"⚙️  Pipeline parallel: {pipeline_parallel_size}")
    print(f"📏 Max model length: {max_model_len}")

    # Build vLLM command
    cmd = [
        "vllm", "serve", model_name,
        "--host", "0.0.0.0",
        "--port", str(port),
        "--dtype", "float16",
        "--max-model-len", str(max_model_len),
        "--tensor-parallel-size", str(tensor_parallel_size)
    ]

    # Add pipeline parallelism if > 1
    if pipeline_parallel_size > 1:
        cmd.extend(["--pipeline-parallel-size", str(pipeline_parallel_size)])

    print(f"🚀 Command: {' '.join(cmd)}")

    # Start vLLM server (blocks until PBS job ends)
    # If this fails, the Parsl task fails - NO FALLBACK
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"vLLM server failed to start: {e}")

    return server_url


# ============================================================================
# Shared vLLM Server Resource
# ============================================================================

@shared(resource_type='vllm_server', auto_register=True)
class SharedvLLMServer:
    """
    Shared vLLM server resource for Nanobrain workflows.

    Key features:
    - Singleton pattern via @shared decorator
    - Initialized once, accessed by multiple workers
    - Thread-safe access to server URL
    - Worker ID tracking for all operations
    - Configurable parallelism (tensor + pipeline)

    NO FALLBACKS - Fails loudly if server startup fails.
    """

    def __init__(
        self,
        model_name: str = "BioMistral/BioMistral-7B-DARE",
        port: int = 8000,
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        max_model_len: int = 4096,
        health_check_timeout: int = 300
    ):
        """
        Initialize shared vLLM server (does NOT start server yet).

        Args:
            model_name: HuggingFace model name
            port: Server port
            tensor_parallel_size: Number of XPU tiles (1 for BioMistral-7B)
            pipeline_parallel_size: Number of nodes for multi-node models
            max_model_len: Maximum sequence length
            health_check_timeout: Health check timeout in seconds
        """
        self.model_name = model_name
        self.port = port
        self.tensor_parallel_size = tensor_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.max_model_len = max_model_len
        self.health_check_timeout = health_check_timeout

        # Server state
        self.server_url = None
        self.server_future = None
        self.is_initialized = False
        self._init_lock = Lock()

        # Tracking
        self.access_count = 0
        self.worker_access_log = []

        # Server URL file for discovery
        self.server_url_file = f"/tmp/vllm_server_{port}.txt"

        logger.info(f"🔧 SharedvLLMServer created (model={model_name}, "
                   f"TP={tensor_parallel_size}, PP={pipeline_parallel_size})")

    async def initialize(self) -> str:
        """
        Initialize vLLM server (start if not already running).

        Thread-safe initialization - only starts server once.

        Returns:
            Server URL

        Raises:
            TimeoutError: If server fails to start within timeout
            RuntimeError: If server startup fails
        """
        with self._init_lock:
            if self.is_initialized:
                logger.info(f"✅ vLLM server already initialized at {self.server_url}")
                return self.server_url

            logger.info(f"🚀 Starting vLLM server for {self.model_name}...")

            # Submit vLLM server startup to Parsl
            self.server_future = start_vllm_server_app(
                model_name=self.model_name,
                port=self.port,
                tensor_parallel_size=self.tensor_parallel_size,
                pipeline_parallel_size=self.pipeline_parallel_size,
                max_model_len=self.max_model_len,
                server_url_file=self.server_url_file
            )

            # Wait for server URL file (10 minutes to account for PBS queue + startup + model loading)
            server_url = await self._wait_for_server_url_file(timeout=600)

            logger.info(f"📍 vLLM server URL discovered: {server_url}")

            # Wait for server health check
            await self._wait_for_server_ready(server_url, timeout=self.health_check_timeout)

            self.server_url = server_url
            self.is_initialized = True

            logger.info(f"✅ vLLM server ready at {server_url}")
            logger.info(f"   Resource ID: {self.get_resource_id()}")
            logger.info(f"   Model: {self.model_name}")
            logger.info(f"   Tensor parallel: {self.tensor_parallel_size}")
            logger.info(f"   Pipeline parallel: {self.pipeline_parallel_size}")

            # Register in distributed registry for cross-node discovery
            try:
                await self._register_in_distributed_registry()
            except Exception as e:
                logger.warning(f"⚠️  Failed to register in distributed registry: {e}")
                logger.warning("   Server is running but won't be discoverable by other workers")

            return server_url

    async def _wait_for_server_url_file(self, timeout: int = 30) -> str:
        """
        Wait for server URL file to be written by compute node.

        Args:
            timeout: Timeout in seconds

        Returns:
            Server URL

        Raises:
            TimeoutError: If URL file not written within timeout
        """
        from pathlib import Path

        logger.info(f"⏳ Waiting for server URL file: {self.server_url_file}")

        start_time = time.time()

        while time.time() - start_time < timeout:
            if Path(self.server_url_file).exists():
                server_url = Path(self.server_url_file).read_text().strip()
                logger.info(f"✅ Server URL file found: {server_url}")
                return server_url

            await asyncio.sleep(0.5)

        raise TimeoutError(
            f"Server URL file {self.server_url_file} not created within {timeout}s. "
            "vLLM server may have failed to start."
        )

    async def _wait_for_server_ready(self, server_url: str, timeout: int = 300):
        """
        Poll vLLM server health endpoint until ready.

        vLLM takes 2-5 minutes to load model and start server.

        Args:
            server_url: Server base URL
            timeout: Timeout in seconds

        Raises:
            TimeoutError: If server not ready within timeout
        """
        import httpx

        health_url = f"{server_url}/health"
        logger.info(f"⏳ Waiting for vLLM server health check: {health_url}")
        logger.info("   (This may take 2-5 minutes for model loading)")

        start_time = time.time()

        async with httpx.AsyncClient(timeout=10.0) as client:
            attempt = 0
            while time.time() - start_time < timeout:
                attempt += 1

                try:
                    response = await client.get(health_url)

                    if response.status_code == 200:
                        elapsed = time.time() - start_time
                        logger.info(f"✅ Server ready after {elapsed:.1f}s ({attempt} attempts)")
                        return

                    logger.debug(f"Health check attempt {attempt}: status {response.status_code}")

                except httpx.RequestError as e:
                    logger.debug(f"Health check attempt {attempt}: {e}")

                # Exponential backoff (1s, 2s, 4s, ..., max 10s)
                wait_time = min(2 ** (attempt // 10), 10)
                await asyncio.sleep(wait_time)

        raise TimeoutError(
            f"vLLM server at {server_url} failed to become ready within {timeout}s. "
            "Check server logs for errors."
        )

    async def _register_in_distributed_registry(self):
        """
        Register vLLM server in distributed ProxyStore registry.

        Enables cross-node discovery of this server by other workers.
        NO FALLBACKS - fails if registry unavailable.

        Raises:
            ImportError: If ProxyStore not installed
            Exception: If registration fails
        """
        import os
        from nanobrain.core.distributed_resource_registry import DistributedResourceRegistry

        # Get registry configuration from environment
        backend = os.getenv('NANOBRAIN_REGISTRY_BACKEND', 'file')
        store_dir = os.getenv('NANOBRAIN_REGISTRY_DIR', '/lus/flare/nanobrain_resources')
        redis_host = os.getenv('NANOBRAIN_REGISTRY_REDIS_HOST', 'localhost')
        redis_port = int(os.getenv('NANOBRAIN_REGISTRY_REDIS_PORT', '6379'))

        logger.info(f"📝 Registering vLLM server in distributed registry (backend={backend})")

        # Create registry
        self._distributed_registry = DistributedResourceRegistry(
            backend=backend,
            store_dir=store_dir,
            redis_host=redis_host,
            redis_port=redis_port
        )

        # Register this server
        resource_id = self.get_resource_id()
        self._distributed_registry.register(
            resource_id=resource_id,
            resource_type='vllm_server',
            metadata={
                'server_url': self.server_url,
                'model_name': self.model_name,
                'tensor_parallel_size': self.tensor_parallel_size,
                'pipeline_parallel_size': self.pipeline_parallel_size,
                'max_model_len': self.max_model_len,
                'port': self.port,
                'hostname': socket.gethostname()
            }
        )

        logger.info(f"✅ Registered {resource_id} in distributed registry")
        logger.info(f"   URL: {self.server_url}")
        logger.info(f"   Backend: {backend}")
        if backend == 'file':
            logger.info(f"   Store dir: {store_dir}")

    def get_server_url(self, worker_id: Optional[str] = None) -> Optional[str]:
        """
        Get vLLM server URL (with worker tracking).

        Args:
            worker_id: ID of worker requesting URL

        Returns:
            Server URL, or None if not initialized
        """
        if worker_id is None:
            worker_id = get_worker_id()

        # Track access
        self.access_count += 1
        self.worker_access_log.append({
            'worker_id': worker_id,
            'operation': 'get_server_url'
        })

        if not self.is_initialized:
            logger.warning(f"⚠️  Worker {worker_id} requested URL but server not initialized")
            return None

        logger.debug(f"📦 Worker {worker_id} got server URL: {self.server_url}")

        return self.server_url

    def get_config(self) -> Dict[str, Any]:
        """
        Get vLLM server configuration.

        Returns:
            Dictionary with server configuration
        """
        return {
            'model_name': self.model_name,
            'port': self.port,
            'tensor_parallel_size': self.tensor_parallel_size,
            'pipeline_parallel_size': self.pipeline_parallel_size,
            'max_model_len': self.max_model_len,
            'server_url': self.server_url,
            'is_initialized': self.is_initialized
        }

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the shared vLLM server.

        Returns:
            Dictionary with server statistics
        """
        # Get shared resource stats
        resource_stats = self.get_resource_stats()

        # Combine with server stats
        return {
            'server_stats': {
                'model_name': self.model_name,
                'server_url': self.server_url,
                'is_initialized': self.is_initialized,
                'access_count': self.access_count,
                'unique_workers': len(set(log['worker_id'] for log in self.worker_access_log)),
                'tensor_parallel_size': self.tensor_parallel_size,
                'pipeline_parallel_size': self.pipeline_parallel_size
            },
            'shared_resource_stats': resource_stats,
            'recent_accesses': self.worker_access_log[-10:]
        }

    async def shutdown(self):
        """Shutdown vLLM server and unregister from pool."""
        logger.info(f"🛑 Shutting down vLLM server {self.get_resource_id()}")

        # Print final stats
        stats = self.get_stats()
        logger.info(f"   Total accesses: {stats['server_stats']['access_count']}")
        logger.info(f"   Unique workers: {stats['server_stats']['unique_workers']}")

        # Unregister from distributed registry
        if hasattr(self, '_distributed_registry'):
            try:
                resource_id = self.get_resource_id()
                self._distributed_registry.unregister(resource_id)
                self._distributed_registry.close()
                logger.info(f"🗑️  Unregistered {resource_id} from distributed registry")
            except Exception as e:
                logger.warning(f"Failed to unregister from distributed registry: {e}")

        # Cancel Parsl task (server will stop when PBS job ends)
        if self.server_future and not self.server_future.done():
            self.server_future.cancel()

        # Unregister from pool
        self.unregister()

        # Clear state
        self.server_url = None
        self.is_initialized = False
