#!/usr/bin/env python3
"""
LocalLLM Client - OpenAI-Interchangeable Interface with Shared vLLM Server Support
===================================================================================

Provides OpenAI-compatible interface for local LLMs running via vLLM.

KEY FEATURES:
- Checks for existing shared vLLM server before creating new one
- OpenAI-compatible interface (same code works for both)
- Automatic shared server discovery via resource pool
- Worker ID tracking for all operations

NO FALLBACKS - If server unavailable, this FAILS.

Usage:
    # OpenAI
    from openai import AsyncOpenAI
    llm = AsyncOpenAI(api_key="sk-...")

    # LocalLLM - SAME INTERFACE
    from nanobrain.library.llms import LocalLLM
    llm = LocalLLM(base_url="http://compute-node:8000/v1")
    # OR let it auto-discover shared server:
    llm = await LocalLLM.from_shared_server(model_name="BioMistral/BioMistral-7B-DARE")

    # Same code for both
    response = await llm.chat.completions.create(
        model="BioMistral/BioMistral-7B-DARE",
        messages=[{" role": "user", "content": "..."}]
    )

Created: 2025-12-03
"""

import logging
from typing import Optional, Dict, Any

from nanobrain.core.shared_resource import get_resource_pool, get_worker_id

logger = logging.getLogger(__name__)


class LocalLLM:
    """
    Local LLM client with OpenAI-compatible interface.

    This is a thin wrapper around openai.AsyncOpenAI that points to vLLM server
    instead of api.openai.com.

    vLLM implements the OpenAI API spec, so we use the OpenAI client library.

    KEY FEATURE: Automatically discovers and uses shared vLLM servers from
    the global resource pool.

    NO FALLBACKS - If vLLM server is down, requests FAIL.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: str = "EMPTY",
        timeout: float = 60.0,
        max_retries: int = 0,  # NO RETRIES - fail immediately
        shared_server: Optional[Any] = None
    ):
        """
        Initialize LocalLLM client.

        Args:
            base_url: vLLM server URL (e.g., "http://x4201c0s5b0n0:8000/v1")
                     If None and shared_server provided, will get URL from shared server
            api_key: API key (vLLM doesn't check this, but OpenAI library requires it)
            timeout: Request timeout in seconds
            max_retries: Number of retries (0 = no retries, fail immediately)
            shared_server: Optional SharedvLLMServer instance for auto-discovery

        Raises:
            ImportError: If openai library not installed
            ValueError: If neither base_url nor shared_server provided
        """
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError(
                "openai library required for LocalLLM client. "
                "Install with: pip install openai"
            )

        # Handle shared server
        self.shared_server = shared_server

        if base_url is None:
            if shared_server is None:
                raise ValueError(
                    "Either base_url or shared_server must be provided. "
                    "Use LocalLLM.from_shared_server() for auto-discovery."
                )
            # Get URL from shared server
            base_url = shared_server.get_server_url(worker_id=get_worker_id())
            if base_url is None:
                raise RuntimeError(
                    "Shared vLLM server not initialized. "
                    "Call await shared_server.initialize() first."
                )

        self.base_url = base_url

        # Create OpenAI client pointed at vLLM server
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,  # vLLM doesn't check this
            timeout=timeout,
            max_retries=max_retries  # NO RETRIES
        )

        logger.info(f"LocalLLM client initialized: {base_url}")
        if shared_server:
            logger.info(f"  Using shared server: {shared_server.get_resource_id()}")

    @classmethod
    async def from_shared_server(
        cls,
        model_name: str = "BioMistral/BioMistral-7B-DARE",
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        max_model_len: int = 4096,
        port: int = 8000,
        create_if_missing: bool = True,
        **kwargs
    ) -> "LocalLLM":
        """
        Create LocalLLM client from shared vLLM server (auto-discovery).

        This method checks the global SharedResourcePool for an existing vLLM server.
        If found, it reuses the existing server.
        If not found and create_if_missing=True, it creates a new shared server.

        Args:
            model_name: HuggingFace model name
            tensor_parallel_size: Number of XPU tiles for TP
            pipeline_parallel_size: Number of nodes for PP
            max_model_len: Maximum sequence length
            port: Server port
            create_if_missing: Create new server if not found (default: True)
            **kwargs: Additional arguments passed to LocalLLM __init__

        Returns:
            LocalLLM client connected to shared vLLM server

        Raises:
            RuntimeError: If server not found and create_if_missing=False
        """
        from .shared_vllm_server import SharedvLLMServer
        from nanobrain.core.distributed_resource_registry import DistributedResourceRegistry
        import os

        worker_id = get_worker_id()
        logger.info(f"🔍 Worker {worker_id} searching for shared vLLM server...")

        # STEP 1: Check in-memory pool first (same-node, fast)
        resource_pool = get_resource_pool()
        all_resources = resource_pool.list_resources()
        vllm_servers = [
            (res_id, stats)
            for res_id, stats in all_resources.items()
            if stats and stats.get('resource_type') == 'vllm_server'
        ]

        if vllm_servers:
            # Found server in same-node memory pool
            logger.info(f"✅ Found {len(vllm_servers)} vLLM server(s) in local pool (same node)")

            server_id, server_stats = vllm_servers[0]
            logger.info(f"   Using server: {server_id}")
            logger.info(f"   Accesses: {server_stats.get('access_count', 0)}")

            # Get the actual server object from pool
            shared_server = resource_pool.get_resource(
                server_id,
                worker_id=worker_id,
                worker_type='local'
            )

            if shared_server is None:
                raise RuntimeError(f"Server {server_id} found in pool but not retrievable")

            # Ensure server is initialized
            if not shared_server.is_initialized:
                logger.info(f"⏳ Initializing found server {server_id}...")
                await shared_server.initialize()

            # Create client with shared server object (same node)
            return cls(shared_server=shared_server, **kwargs)

        # STEP 2: Check distributed registry (cross-node, slower)
        logger.info(f"❌ No vLLM servers in local pool")
        logger.info(f"🔍 Checking distributed registry for cross-node servers...")

        try:
            backend = os.getenv('NANOBRAIN_REGISTRY_BACKEND', 'file')
            store_dir = os.getenv('NANOBRAIN_REGISTRY_DIR', '/lus/flare/nanobrain_resources')

            registry = DistributedResourceRegistry(backend=backend, store_dir=store_dir)

            # Discover servers matching model
            filters = {'model_name': model_name} if model_name else None
            discovered_resources = registry.discover(
                resource_type='vllm_server',
                filters=filters
            )

            if discovered_resources:
                # Found server(s) in distributed registry
                logger.info(f"✅ Found {len(discovered_resources)} vLLM server(s) in distributed registry")

                # Use first matching server
                resource = discovered_resources[0]
                server_url = resource['metadata']['server_url']
                resource_id = resource['resource_id']

                logger.info(f"   Using server: {resource_id}")
                logger.info(f"   URL: {server_url}")
                logger.info(f"   Model: {resource['metadata']['model_name']}")
                logger.info(f"   Tensor parallel: {resource['metadata']['tensor_parallel_size']}")

                # Create client with URL (cross-node, HTTP-based)
                return cls(base_url=server_url, **kwargs)

            else:
                logger.info(f"❌ No vLLM servers found in distributed registry")

        except ImportError as e:
            logger.warning(f"⚠️  ProxyStore not available: {e}")
            logger.warning("   Skipping distributed registry check")
        except Exception as e:
            logger.warning(f"⚠️  Failed to check distributed registry: {e}")
            logger.warning("   Continuing without cross-node discovery")

        # STEP 3: No existing server found anywhere
        logger.info(f"❌ No existing vLLM servers found (local or distributed)")

        if not create_if_missing:
            raise RuntimeError(
                "No shared vLLM server found and create_if_missing=False. "
                "Create a SharedvLLMServer first or set create_if_missing=True."
            )

        # Create new shared server
        logger.info(f"🔧 Creating new shared vLLM server...")
        logger.info(f"   Model: {model_name}")
        logger.info(f"   TP: {tensor_parallel_size}")
        logger.info(f"   PP: {pipeline_parallel_size}")

        shared_server = SharedvLLMServer(
            model_name=model_name,
            port=port,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            max_model_len=max_model_len
        )

        # Initialize server (will auto-register in distributed registry)
        logger.info(f"🚀 Initializing new vLLM server...")
        await shared_server.initialize()

        # Create client
        return cls(shared_server=shared_server, **kwargs)

    @property
    def chat(self):
        """
        Expose chat.completions interface (OpenAI-compatible).

        This allows:
            llm.chat.completions.create(...)

        Same as OpenAI client.
        """
        return self._client.chat

    @property
    def completions(self):
        """
        Expose completions interface (OpenAI-compatible).

        For non-chat models.
        """
        return self._client.completions

    async def health_check(self) -> bool:
        """
        Check if vLLM server is healthy.

        Returns:
            True if server is healthy

        Raises:
            Exception: If server is not healthy
        """
        import httpx

        health_url = f"{self.base_url}/health"

        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(health_url)
            response.raise_for_status()

        logger.info(f"vLLM server health check passed: {self.base_url}")
        return True

    def get_server_config(self) -> Optional[Dict[str, Any]]:
        """
        Get configuration of the shared vLLM server.

        Returns:
            Server config dict, or None if not using shared server
        """
        if self.shared_server:
            return self.shared_server.get_config()
        return None

    def get_server_stats(self) -> Optional[Dict[str, Any]]:
        """
        Get statistics of the shared vLLM server.

        Returns:
            Server stats dict, or None if not using shared server
        """
        if self.shared_server:
            return self.shared_server.get_stats()
        return None

    def __repr__(self):
        if self.shared_server:
            return f"LocalLLM(base_url='{self.base_url}', shared_server='{self.shared_server.get_resource_id()}')"
        return f"LocalLLM(base_url='{self.base_url}')"
