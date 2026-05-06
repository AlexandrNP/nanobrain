#!/usr/bin/env python3
"""
vLLM Server Manager - Parsl Integration

Manages vLLM server lifecycle on Aurora compute nodes via Parsl.

NO FALLBACKS - If vLLM fails, this FAILS LOUDLY.
NO MOCKS - Real vLLM server or error.
NO SIMULATED DATA - Actual inference or failure.

Created: 2025-12-03
"""

import asyncio
import logging
import socket
import time
from pathlib import Path
from typing import Optional
from parsl import python_app

logger = logging.getLogger(__name__)


@python_app
def start_vllm_server(
    model_name: str,
    port: int = 8000,
    tensor_parallel_size: int = 1,
    max_model_len: int = 4096,
    server_url_file: str = "/tmp/vllm_server_url.txt"
):
    """
    Start vLLM server on compute node.

    This function runs on the PBS-allocated compute node.
    It starts the vLLM server and writes the server URL to a file.

    Args:
        model_name: HuggingFace model name
        port: Server port
        tensor_parallel_size: Number of XPU tiles (1 for BioMistral-7B)
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

    # Write server URL to file immediately (for client discovery)
    with open(server_url_file, 'w') as f:
        f.write(server_url)

    print(f"🔥 Starting vLLM server on {hostname}:{port}")
    print(f"📦 Model: {model_name}")
    print(f"⚙️  Tensor parallel: {tensor_parallel_size}")
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

    print(f"🚀 Command: {' '.join(cmd)}")

    # Start vLLM server (blocks until PBS job ends)
    # If this fails, the Parsl task fails - NO FALLBACK
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"vLLM server failed to start: {e}")

    return server_url


class vLLMServerManager:
    """
    Manages vLLM server lifecycle via Parsl.

    NO FALLBACKS - Fails loudly if server doesn't start.
    """

    def __init__(self, parsl_executor=None):
        """
        Args:
            parsl_executor: Parsl executor for server task (optional if Parsl already loaded)
        """
        self.executor = parsl_executor
        self.server_future = None
        self.server_url = None
        self.server_url_file = "/tmp/vllm_server_url.txt"

    async def start_server(
        self,
        model_name: str = "BioMistral/BioMistral-7B-DARE",
        port: int = 8000,
        tensor_parallel_size: int = 1,
        max_model_len: int = 4096,
        health_check_timeout: int = 300
    ) -> str:
        """
        Start vLLM server on compute node via Parsl.

        Args:
            model_name: HuggingFace model name
            port: Server port
            tensor_parallel_size: Number of XPU tiles
            max_model_len: Maximum sequence length
            health_check_timeout: Timeout for server health check (seconds)

        Returns:
            Server URL (http://hostname:port/v1)

        Raises:
            TimeoutError: If server fails to start within timeout
            RuntimeError: If server startup fails
        """
        logger.info(f"🚀 Starting vLLM server for model: {model_name}")

        # Submit vLLM server startup task to Parsl
        self.server_future = start_vllm_server(
            model_name=model_name,
            port=port,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            server_url_file=self.server_url_file
        )

        # Wait for server URL file to be written
        server_url = await self._wait_for_server_url_file(timeout=30)

        logger.info(f"📍 vLLM server URL discovered: {server_url}")

        # Wait for server to be ready (health check)
        await self._wait_for_server_ready(server_url, timeout=health_check_timeout)

        self.server_url = server_url
        logger.info(f"✅ vLLM server ready at {server_url}")

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
        logger.info(f"   (This may take 2-5 minutes for model loading)")

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

    async def shutdown_server(self):
        """
        Shutdown vLLM server (cancel Parsl task).

        Server will automatically shutdown when PBS job ends.
        """
        if self.server_future:
            logger.info("🛑 Shutting down vLLM server")
            self.server_future.cancel()
            self.server_url = None

    def __del__(self):
        """Cleanup on deletion"""
        if self.server_future and not self.server_future.done():
            logger.warning("vLLMServerManager deleted with active server - cancelling")
            self.server_future.cancel()
