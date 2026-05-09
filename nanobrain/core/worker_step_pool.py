#!/usr/bin/env python3
"""
Worker Step Pool for PARSL Parallel Execution
==============================================

Manages per-worker step instances for PARSL executor, ensuring:
1. Each worker has its own step instance
2. Each instance has a unique worker_id
3. @shared resources are NOT duplicated
4. Non-shared resources ARE duplicated per worker
"""

import logging
import uuid
from typing import Any, Dict, Type, Optional, List
from dataclasses import dataclass, field



logger = logging.getLogger(__name__)


@dataclass
class WorkerInstanceInfo:
    """Information about a worker's step instance."""
    worker_id: str
    step_instance: Any
    shared_resources: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=lambda: __import__('time').time())


class WorkerStepPool:
    """
    Manages per-worker step instances for PARSL execution.
    
    Features:
    - Creates one step instance per PARSL worker
    - Assigns unique worker_id to each instance
    - Shares @shared resources across instances
    - Isolates non-shared state per worker
    
    Example:
        ```python
        # Create pool for 4 workers
        pool = WorkerStepPool(
            step_class=QueryEnhancementStep,
            step_config=config,
            num_workers=4
        )
        
        # Initialize all worker instances
        await pool.initialize_workers()
        
        # Get instance for specific worker
        step_instance = pool.get_worker_instance('worker_abc123')
        ```
    """
    
    def __init__(
        self,
        step_class: Type,
        step_config: Dict[str, Any],
        num_workers: int,
        step_id: Optional[str] = None
    ):
        """
        Initialize worker step pool.
        
        Args:
            step_class: Class of the step to instantiate
            step_config: Configuration for step instances
            num_workers: Number of worker instances to create
            step_id: Optional step identifier
        """
        self.step_class = step_class
        self.step_config = step_config
        self.num_workers = num_workers
        self.step_id = step_id or f"step_{uuid.uuid4().hex[:8]}"
        
        self.worker_instances: Dict[str, WorkerInstanceInfo] = {}
        self.shared_resources: Dict[str, Any] = {}
        self.is_initialized = False
        
        logger.info(f"🔧 Created WorkerStepPool for {step_class.__name__} with {num_workers} workers")
    
    def _detect_shared_resources(self, obj: Any) -> Dict[str, Any]:
        """
        Detect all @shared resources in an object.
        
        Args:
            obj: Object to inspect
            
        Returns:
            Dict mapping resource_id to resource instance
        """
        shared_resources = {}
        
        # Inspect all attributes
        for attr_name in dir(obj):
            if attr_name.startswith('_'):
                continue
            
            try:
                attr = getattr(obj, attr_name)
                
                # Check if class is marked with @shared
                if hasattr(attr, '__class__') and hasattr(attr.__class__, '_is_shared_resource'):
                    if hasattr(attr, 'get_resource_id'):
                        resource_id = attr.get_resource_id()
                        shared_resources[resource_id] = attr
                        logger.debug(f"   Found @shared resource: {resource_id}")
            except Exception:
                # Skip attributes that can't be accessed
                pass
        
        return shared_resources
    
    def _apply_shared_resources(self, instance: Any, shared_resources: Dict[str, Any]):
        """
        Apply shared resources to an instance.
        
        Args:
            instance: Step instance to modify
            shared_resources: Dict of shared resources to apply
        """
        for attr_name in dir(instance):
            if attr_name.startswith('_'):
                continue
            
            try:
                attr = getattr(instance, attr_name)
                
                # Check if this attribute should be replaced with shared version
                if hasattr(attr, '__class__') and hasattr(attr.__class__, '_is_shared_resource'):
                    if hasattr(attr, 'get_resource_id'):
                        resource_id = attr.get_resource_id()
                        
                        # Replace with shared instance
                        if resource_id in shared_resources:
                            setattr(instance, attr_name, shared_resources[resource_id])
                            logger.debug(f"   Applied shared resource {resource_id} to {attr_name}")
            except Exception:
                pass
    
    async def initialize_workers(self):
        """
        Create and initialize step instance for each worker.
        
        This method:
        1. Creates num_workers step instances
        2. Assigns unique worker_id to each
        3. Detects @shared resources from first instance
        4. Shares those resources across all instances
        5. Initializes all instances
        """
        if self.is_initialized:
            logger.warning("WorkerStepPool already initialized")
            return
        
        logger.info(f"🔧 Initializing {self.num_workers} worker instances for {self.step_class.__name__}")
        
        # Create first instance to detect shared resources
        first_worker_id = f"worker_{uuid.uuid4().hex[:8]}"

        # Try to create instance using from_config if available and config is a file path
        if hasattr(self.step_class, 'from_config') and isinstance(self.step_config, (str, __import__('pathlib').Path)):
            first_instance = self.step_class.from_config(self.step_config)
        else:
            # Direct instantiation for test classes or dict configs
            first_instance = self.step_class(self.step_config)

        # Set worker_id on instance
        first_instance.worker_id = first_worker_id

        # Initialize first instance
        if hasattr(first_instance, 'initialize'):
            await first_instance.initialize()
        
        # Detect shared resources from first instance
        self.shared_resources = self._detect_shared_resources(first_instance)
        
        if self.shared_resources:
            logger.info(f"   Found {len(self.shared_resources)} @shared resources:")
            for res_id in self.shared_resources.keys():
                logger.info(f"      - {res_id}")
        
        # Store first instance
        self.worker_instances[first_worker_id] = WorkerInstanceInfo(
            worker_id=first_worker_id,
            step_instance=first_instance,
            shared_resources=self.shared_resources.copy()
        )
        
        # Create remaining worker instances
        for i in range(1, self.num_workers):
            worker_id = f"worker_{uuid.uuid4().hex[:8]}"

            # Create new instance using from_config if available and config is a file path
            if hasattr(self.step_class, 'from_config') and isinstance(self.step_config, (str, __import__('pathlib').Path)):
                instance = self.step_class.from_config(self.step_config)
            else:
                # Direct instantiation for test classes or dict configs
                instance = self.step_class(self.step_config)
            instance.worker_id = worker_id
            
            # Apply shared resources from first instance
            if self.shared_resources:
                self._apply_shared_resources(instance, self.shared_resources)
            
            # Initialize instance
            if hasattr(instance, 'initialize'):
                await instance.initialize()
            
            # Store instance
            self.worker_instances[worker_id] = WorkerInstanceInfo(
                worker_id=worker_id,
                step_instance=instance,
                shared_resources=self.shared_resources.copy()
            )
            
            logger.debug(f"   Created worker instance {i+1}/{self.num_workers}: {worker_id}")
        
        self.is_initialized = True
        logger.info(f"✅ Initialized {len(self.worker_instances)} worker instances")
    
    def get_worker_instance(self, worker_id: Optional[str] = None) -> Any:
        """
        Get step instance for specific worker.
        
        Args:
            worker_id: Worker ID (if None, returns first available)
            
        Returns:
            Step instance for the worker
        """
        if not self.is_initialized:
            raise RuntimeError("WorkerStepPool not initialized. Call initialize_workers() first.")
        
        if worker_id and worker_id in self.worker_instances:
            return self.worker_instances[worker_id].step_instance
        
        # Return first available instance if worker_id not specified
        if self.worker_instances:
            return next(iter(self.worker_instances.values())).step_instance
        
        raise ValueError(f"No worker instance found for worker_id: {worker_id}")
    
    def get_available_worker_id(self) -> Optional[str]:
        """
        Get an available worker ID (round-robin).
        
        Returns:
            Worker ID or None if no workers available
        """
        if not self.worker_instances:
            return None
        
        # Simple round-robin: return first worker ID
        # In production, could implement more sophisticated load balancing
        return next(iter(self.worker_instances.keys()))
    
    def get_all_worker_ids(self) -> List[str]:
        """
        Get all worker IDs.
        
        Returns:
            List of worker IDs
        """
        return list(self.worker_instances.keys())
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the worker pool.
        
        Returns:
            Dictionary with pool statistics
        """
        return {
            'step_class': self.step_class.__name__,
            'step_id': self.step_id,
            'num_workers': self.num_workers,
            'initialized': self.is_initialized,
            'worker_ids': self.get_all_worker_ids(),
            'shared_resources': list(self.shared_resources.keys()),
            'num_shared_resources': len(self.shared_resources)
        }
    
    async def shutdown(self):
        """Shutdown all worker instances."""
        logger.info(f"🗑️  Shutting down WorkerStepPool for {self.step_class.__name__}")
        
        for worker_id, info in self.worker_instances.items():
            try:
                if hasattr(info.step_instance, 'shutdown'):
                    await info.step_instance.shutdown()
                logger.debug(f"   Shutdown worker instance: {worker_id}")
            except Exception as e:
                logger.error(f"   Failed to shutdown worker {worker_id}: {e}")
        
        self.worker_instances.clear()
        self.shared_resources.clear()
        self.is_initialized = False
        
        logger.info("✅ WorkerStepPool shutdown complete")

