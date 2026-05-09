"""
Distributed Resource Registry using ProxyStore

Provides transparent distributed resource discovery for SharedResourcePool.
Workers can register and discover resources across compute nodes using ProxyStore.

BRUTAL TRUTH: This is the right level of engineering - not too simple (manual files),
not too complex (Academy agents). ProxyStore is purpose-built for this exact problem.

Created: 2025-12-03
"""
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DistributedResourceRegistry:
    """
    Distributed resource registry using ProxyStore.

    Provides transparent cross-node resource discovery via proxy objects.
    Workers register resources as proxies, other workers get the same proxies
    and can transparently access the actual data.

    ARCHITECTURE:
    - FileConnector backend uses shared filesystem (/lus/flare on Aurora)
    - Resources stored as JSON files in registry directory
    - ProxyStore handles serialization, storage, and retrieval
    - Transparent to application code (users see normal Python dicts)
    """

    def __init__(
        self,
        backend: str = "file",
        store_dir: str = "/lus/flare/nanobrain_resources",
        redis_host: str = "localhost",
        redis_port: int = 6379,
        store_name: str = "nanobrain-shared-resources"
    ):
        """
        Initialize distributed registry.

        Args:
            backend: Storage backend ('file', 'redis')
            store_dir: Directory for file-based storage (if backend='file')
            redis_host: Redis host (if backend='redis')
            redis_port: Redis port (if backend='redis')
            store_name: Name for the ProxyStore instance

        Raises:
            ImportError: If ProxyStore not installed
            ValueError: If unsupported backend specified
        """
        self.backend = backend
        self.store_dir = store_dir
        self.store_name = store_name

        # Import ProxyStore (may not be installed)
        try:
            from proxystore.store import Store
            from proxystore.connectors.file import FileConnector
        except ImportError:
            raise ImportError(
                "ProxyStore not installed. "
                "Install with: pip install 'nanobrain[academy]' or pip install proxystore"
            )

        # Create ProxyStore instance
        if backend == "file":
            # File-based connector (uses shared filesystem)
            Path(store_dir).mkdir(parents=True, exist_ok=True)
            connector = FileConnector(store_dir)
            logger.info(f"📁 Using file-based ProxyStore at {store_dir}")

        elif backend == "redis":
            # Redis connector (requires Redis server)
            try:
                from proxystore.connectors.redis import RedisConnector
                connector = RedisConnector(hostname=redis_host, port=redis_port)
                logger.info(f"🔴 Using Redis ProxyStore at {redis_host}:{redis_port}")
            except ImportError:
                raise ImportError(
                    "ProxyStore Redis connector not available. "
                    "Install with: pip install 'proxystore[redis]'"
                )

        else:
            raise ValueError(f"Unsupported backend: {backend}")

        # Create store
        self.store = Store(
            name=store_name,
            connector=connector,
            cache_size=0,  # No local cache (always fetch from backend)
            metrics=True   # Enable metrics for debugging
        )

        logger.info(f"✅ ProxyStore registry initialized (backend={backend})")

    def register(
        self,
        resource_id: str,
        resource_type: str,
        metadata: Dict[str, Any]
    ) -> str:
        """
        Register a resource in distributed registry.

        Args:
            resource_id: Unique resource identifier
            resource_type: Type of resource (e.g., 'vllm_server')
            metadata: Resource metadata to store

        Returns:
            Resource ID (same as input, for convenience)

        Note:
            ProxyStore uses auto-generated keys (UUIDs). We store resource_id
            in the metadata and discover resources by filtering.
        """
        # Create resource record
        resource_record = {
            'resource_id': resource_id,
            'resource_type': resource_type,
            'metadata': metadata
        }

        # Store in ProxyStore (auto-generated UUID key)
        # We discover resources by filtering metadata, not by key
        proxystore_key = self.store.put(resource_record)

        logger.info(
            f"📝 Registered {resource_type} resource: {resource_id} "
            f"(ProxyStore key={proxystore_key})"
        )

        return resource_id

    def discover(
        self,
        resource_type: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Discover resources in distributed registry.

        Args:
            resource_type: Filter by resource type (e.g., 'vllm_server')
            filters: Additional filters applied to metadata
                     (e.g., {'model_name': 'BioMistral-7B'})

        Returns:
            List of matching resource records

        Example:
            >>> registry.discover(
            ...     resource_type='vllm_server',
            ...     filters={'model_name': 'BioMistral/BioMistral-7B-DARE'}
            ... )
            [{'resource_id': 'vllm_biomistral_7b', ...}]
        """
        # Get all keys from connector
        # FileConnector doesn't have keys() method, so we list files directly
        all_keys = []

        if self.backend == 'file':
            # List files from FileConnector directory
            from pathlib import Path
            from proxystore.connectors.file import FileKey

            store_dir = Path(self.store.connector.store_dir)
            if store_dir.exists():
                # Each file is a stored object (UUID filename)
                for filepath in store_dir.glob('*'):
                    if filepath.is_file():
                        # Create FileKey from filename
                        all_keys.append(FileKey(filename=filepath.name))
        else:
            # For other connectors (e.g., Redis), try keys() method
            try:
                all_keys = list(self.store.connector.keys())
            except AttributeError:
                logger.warning(f"Backend {self.backend} doesn't support keys(), using empty list")
                return []

        matching_resources = []

        for key in all_keys:
            try:
                # Get proxy (lightweight operation)
                proxy = self.store.get(key)
                if proxy is None:
                    continue

                # Dereference proxy to get actual data
                # ProxyStore handles this transparently
                try:
                    resource_record = proxy.__wrapped__
                except AttributeError:
                    # Not a proxy, assume it's the actual data
                    resource_record = proxy

                # Apply filters
                if resource_type and resource_record.get('resource_type') != resource_type:
                    continue

                if filters:
                    metadata = resource_record.get('metadata', {})
                    if not all(metadata.get(k) == v for k, v in filters.items()):
                        continue

                matching_resources.append(resource_record)

            except Exception as e:
                logger.warning(f"Failed to retrieve resource {key}: {e}")
                continue

        logger.info(
            f"🔍 Discovered {len(matching_resources)} resources "
            f"(type={resource_type}, filters={filters})"
        )

        return matching_resources

    def get(self, resource_id: str) -> Optional[Dict[str, Any]]:
        """
        Get specific resource by ID.

        Args:
            resource_id: Resource identifier

        Returns:
            Resource record, or None if not found

        Note:
            Since ProxyStore uses UUID keys, we need to discover by resource_id.
        """
        # Find resource by filtering (inefficient but necessary with UUID keys)
        resources = self.discover(resource_type=None, filters=None)

        for resource in resources:
            if resource.get('resource_id') == resource_id:
                return resource

        logger.debug(f"Resource {resource_id} not found in registry")
        return None

    def unregister(self, resource_id: str) -> bool:
        """
        Unregister a resource.

        Args:
            resource_id: Resource identifier

        Returns:
            True if successfully unregistered, False otherwise

        Note:
            Since ProxyStore uses UUID keys, we need to find the key first.
        """
        try:
            # Get all keys (using same logic as discover)
            all_keys = []

            if self.backend == 'file':
                from pathlib import Path
                from proxystore.connectors.file import FileKey

                store_dir = Path(self.store.connector.store_dir)
                if store_dir.exists():
                    for filepath in store_dir.glob('*'):
                        if filepath.is_file():
                            all_keys.append(FileKey(filename=filepath.name))
            else:
                try:
                    all_keys = list(self.store.connector.keys())
                except AttributeError:
                    logger.warning(f"Backend {self.backend} doesn't support keys()")
                    return False

            for key in all_keys:
                try:
                    resource = self.store.get(key)
                    if resource is None:
                        continue

                    # Check if this is our resource
                    try:
                        resource_data = resource.__wrapped__
                    except AttributeError:
                        resource_data = resource

                    if resource_data.get('resource_id') == resource_id:
                        # Found it - evict using ProxyStore key
                        self.store.evict(key)
                        logger.info(f"🗑️  Unregistered resource: {resource_id}")
                        return True

                except Exception as e:
                    logger.debug(f"Error checking key {key}: {e}")
                    continue

            logger.warning(f"Resource {resource_id} not found for unregistration")
            return False

        except Exception as e:
            logger.warning(f"Failed to unregister {resource_id}: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get registry statistics.

        Returns:
            Dictionary with registry stats (total resources, metrics, etc.)
        """
        # Count resources using same logic as discover
        total_keys = 0

        if self.backend == 'file':
            from pathlib import Path

            store_dir = Path(self.store.connector.store_dir)
            if store_dir.exists():
                total_keys = len(list(store_dir.glob('*')))
        else:
            try:
                total_keys = len(list(self.store.connector.keys()))
            except Exception:
                total_keys = 0

        # Get metrics (API varies by ProxyStore version)
        metrics_dict = {}
        if self.store.metrics:
            try:
                # Try to extract some basic metrics
                metrics_dict = {
                    'metrics_available': True,
                    'metrics_str': str(self.store.metrics)[:200]  # First 200 chars
                }
            except Exception:
                metrics_dict = {'metrics_available': False}

        return {
            'backend': self.backend,
            'store_name': self.store_name,
            'total_resources': total_keys,
            'store_metrics': metrics_dict
        }

    def close(self):
        """Close ProxyStore connection"""
        try:
            self.store.close()
            logger.info("🔒 ProxyStore registry closed")
        except Exception as e:
            logger.warning(f"Error closing registry: {e}")
