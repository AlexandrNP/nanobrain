#!/usr/bin/env python3
"""
Shared Resource Management for Nanobrain Framework
===================================================

Provides decorators and utilities for managing shared resources (like vector databases)
that are accessed by multiple parallel workers/executors.

Key Features:
- @shared decorator for marking shared resource classes
- Automatic resource pooling and lifecycle management
- Worker ID tracking for parallel execution
- Thread-safe access to shared resources
- Integration with data unit->link->trigger execution strategy
"""

import asyncio
import logging
import uuid
from typing import Any, Dict, Optional, Type, Callable
from functools import wraps
from threading import Lock
from dataclasses import dataclass, field
from datetime import datetime


logger = logging.getLogger(__name__)


@dataclass
class WorkerContext:
    """Context information for a worker accessing a shared resource."""
    worker_id: str
    worker_type: str  # 'parsl', 'thread', 'local', etc.
    created_at: datetime = field(default_factory=datetime.now)
    request_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SharedResourceMetadata:
    """Metadata for a shared resource."""
    resource_id: str
    resource_type: str
    created_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    active_workers: Dict[str, WorkerContext] = field(default_factory=dict)
    is_initialized: bool = False
    initialization_lock: Lock = field(default_factory=Lock)


class SharedResourcePool:
    """
    Global pool for managing shared resources.
    
    Ensures that shared resources (like vector databases) are:
    - Initialized only once
    - Safely accessed by multiple workers
    - Properly tracked and monitored
    - Cleaned up when no longer needed
    """
    
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        """Singleton pattern for global resource pool."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the resource pool."""
        if self._initialized:
            return
        
        self._resources: Dict[str, Any] = {}
        self._metadata: Dict[str, SharedResourceMetadata] = {}
        self._access_lock = Lock()
        self._initialized = True
        
        logger.info("🔧 SharedResourcePool initialized")
    
    def register_resource(
        self,
        resource_id: str,
        resource: Any,
        resource_type: str
    ) -> None:
        """
        Register a shared resource in the pool.
        
        Args:
            resource_id: Unique identifier for the resource
            resource: The actual resource object
            resource_type: Type/category of the resource
        """
        with self._access_lock:
            if resource_id in self._resources:
                logger.warning(f"⚠️  Resource {resource_id} already registered, updating")
            
            self._resources[resource_id] = resource
            self._metadata[resource_id] = SharedResourceMetadata(
                resource_id=resource_id,
                resource_type=resource_type,
                is_initialized=True
            )
            
            logger.info(f"✅ Registered shared resource: {resource_id} ({resource_type})")
    
    def get_resource(
        self,
        resource_id: str,
        worker_id: Optional[str] = None,
        worker_type: str = 'unknown'
    ) -> Optional[Any]:
        """
        Get a shared resource from the pool.
        
        Args:
            resource_id: Unique identifier for the resource
            worker_id: ID of the worker requesting the resource
            worker_type: Type of worker (parsl, thread, etc.)
            
        Returns:
            The shared resource, or None if not found
        """
        with self._access_lock:
            if resource_id not in self._resources:
                logger.warning(f"⚠️  Resource {resource_id} not found in pool")
                return None
            
            # Track access
            metadata = self._metadata[resource_id]
            metadata.access_count += 1
            
            # Track worker
            if worker_id:
                if worker_id not in metadata.active_workers:
                    metadata.active_workers[worker_id] = WorkerContext(
                        worker_id=worker_id,
                        worker_type=worker_type
                    )
                metadata.active_workers[worker_id].request_count += 1
            
            logger.debug(f"📦 Resource {resource_id} accessed by worker {worker_id} "
                        f"(total accesses: {metadata.access_count})")
            
            return self._resources[resource_id]
    
    def unregister_resource(self, resource_id: str) -> None:
        """
        Unregister a shared resource from the pool.
        
        Args:
            resource_id: Unique identifier for the resource
        """
        with self._access_lock:
            if resource_id in self._resources:
                del self._resources[resource_id]
                del self._metadata[resource_id]
                logger.info(f"🗑️  Unregistered shared resource: {resource_id}")
    
    def get_resource_stats(self, resource_id: str) -> Optional[Dict[str, Any]]:
        """
        Get statistics for a shared resource.
        
        Args:
            resource_id: Unique identifier for the resource
            
        Returns:
            Dictionary with resource statistics
        """
        with self._access_lock:
            if resource_id not in self._metadata:
                return None
            
            metadata = self._metadata[resource_id]
            return {
                'resource_id': resource_id,
                'resource_type': metadata.resource_type,
                'access_count': metadata.access_count,
                'active_workers': len(metadata.active_workers),
                'worker_ids': list(metadata.active_workers.keys()),
                'created_at': metadata.created_at.isoformat(),
                'is_initialized': metadata.is_initialized
            }
    
    def list_resources(self) -> Dict[str, Dict[str, Any]]:
        """
        List all registered shared resources.
        
        Returns:
            Dictionary mapping resource IDs to their statistics
        """
        with self._access_lock:
            return {
                resource_id: self.get_resource_stats(resource_id)
                for resource_id in self._resources.keys()
            }
    
    def clear(self) -> None:
        """Clear all resources from the pool."""
        with self._access_lock:
            self._resources.clear()
            self._metadata.clear()
            logger.info("🗑️  Cleared all shared resources")


# Global resource pool instance
_resource_pool = SharedResourcePool()


def shared(
    resource_type: Optional[str] = None,
    resource_id: Optional[str] = None,
    auto_register: bool = True
):
    """
    Decorator to mark a class as a shared resource.
    
    Shared resources are:
    - Initialized only once
    - Safely accessed by multiple workers
    - Automatically registered in the global resource pool
    - Tracked for monitoring and debugging
    
    Args:
        resource_type: Type/category of the resource (default: class name)
        resource_id: Unique identifier (default: auto-generated)
        auto_register: Whether to automatically register in the pool
    
    Example:
        ```python
        @shared(resource_type='vector_database')
        class VectorDatabase:
            def __init__(self, config):
                self.config = config
                self.index = None
            
            async def initialize(self):
                # Initialize once, shared by all workers
                self.index = await self._build_index()
            
            async def search(self, query, worker_id=None):
                # Called by multiple workers
                results = await self.index.search(query)
                return {
                    'results': results,
                    'worker_id': worker_id  # Track which worker processed this
                }
        ```
    
    Usage in workflow:
        ```python
        # Vector DB is shared across all parallel workers
        vector_db = VectorDatabase(config)
        await vector_db.initialize()
        
        # Multiple PARSL workers can access it concurrently
        results = await parsl_executor.execute(
            lambda: vector_db.search(query, worker_id=get_worker_id())
        )
        ```
    """
    def decorator(cls: Type) -> Type:
        """Actual decorator that wraps the class."""
        
        # Determine resource type and ID
        res_type = resource_type or cls.__name__
        res_id = resource_id or f"{res_type}_{uuid.uuid4().hex[:8]}"
        
        # Store original __init__
        original_init = cls.__init__
        
        @wraps(original_init)
        def new_init(self, *args, **kwargs):
            """Wrapped __init__ that adds shared resource metadata."""
            # Call original __init__
            original_init(self, *args, **kwargs)
            
            # Add shared resource metadata
            self._shared_resource_id = res_id
            self._shared_resource_type = res_type
            self._shared_auto_register = auto_register
            
            # Auto-register if enabled
            if auto_register:
                _resource_pool.register_resource(res_id, self, res_type)
                logger.info(f"🔧 Auto-registered shared resource: {res_id} ({res_type})")
        
        # Replace __init__
        cls.__init__ = new_init
        
        # Add helper methods
        def get_resource_id(self) -> str:
            """Get the unique resource ID."""
            return getattr(self, '_shared_resource_id', res_id)
        
        def get_resource_stats(self) -> Optional[Dict[str, Any]]:
            """Get statistics for this shared resource."""
            return _resource_pool.get_resource_stats(self.get_resource_id())
        
        def unregister(self) -> None:
            """Unregister this resource from the pool."""
            _resource_pool.unregister_resource(self.get_resource_id())
        
        # Add methods to class
        cls.get_resource_id = get_resource_id
        cls.get_resource_stats = get_resource_stats
        cls.unregister = unregister
        
        # Mark class as shared
        cls._is_shared_resource = True
        cls._shared_resource_type = res_type
        
        logger.debug(f"🔧 Decorated class {cls.__name__} as shared resource ({res_type})")
        
        return cls
    
    return decorator


def get_worker_id() -> str:
    """
    Get the current worker ID.
    
    Generates a unique ID for the current worker/executor context.
    This ID is used to track which worker is accessing shared resources.
    
    Returns:
        Unique worker ID (UUID hash)
    """
    # Try to get from asyncio task context
    try:
        task = asyncio.current_task()
        if task:
            task_name = task.get_name()
            return f"worker_{hash(task_name) & 0xFFFFFFFF:08x}"
    except RuntimeError:
        pass
    
    # Fallback: generate new UUID
    return f"worker_{uuid.uuid4().hex[:8]}"


def get_resource_pool() -> SharedResourcePool:
    """
    Get the global shared resource pool.
    
    Returns:
        The global SharedResourcePool instance
    """
    return _resource_pool

