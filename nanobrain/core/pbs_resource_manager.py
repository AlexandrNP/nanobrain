#!/usr/bin/env python3
"""
PBS Resource Manager for Workflow Divergence

CRITICAL PRODUCTION COMPONENT:
- Tracks available PBS nodes and cores
- Manages resource allocation across multiple workflows
- ALWAYS reserves at least 1 core per node for system tasks
- Prevents resource oversubscription disasters
- Integrates with SharedResourcePool pattern

BRUTAL TRUTH: This should have been implemented from day one!
Without this, 70 workflows will crash the PBS system.
"""

import asyncio
import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import threading
import logging

from nanobrain.core.shared_resource import shared, get_worker_id


logger = logging.getLogger(__name__)


@dataclass
class NodeResource:
    """Represents a PBS node and its resource allocation"""
    node_id: str
    total_cores: int
    reserved_cores: int = 1  # Always reserve 1 core for system tasks
    allocated_cores: int = 0
    active_workflows: Set[str] = field(default_factory=set)
    last_updated: float = field(default_factory=time.time)
    
    @property
    def available_cores(self) -> int:
        """Available cores = total - reserved - allocated"""
        return max(0, self.total_cores - self.reserved_cores - self.allocated_cores)
    
    @property
    def utilization(self) -> float:
        """Core utilization percentage"""
        if self.total_cores == 0:
            return 0.0
        return (self.allocated_cores / (self.total_cores - self.reserved_cores)) * 100


@dataclass
class ResourceAllocation:
    """Represents a resource allocation for a workflow"""
    workflow_uuid: str
    node_id: str
    allocated_cores: int
    max_workers: int
    allocation_time: float = field(default_factory=time.time)


@shared(resource_type='pbs_resource_manager', auto_register=True)
class PBSResourceManager:
    """
    Shared PBS Resource Manager
    
    CRITICAL FUNCTIONS:
    1. Track available PBS nodes and cores
    2. Allocate resources to workflows without oversubscription
    3. ALWAYS reserve 1+ cores per node for system tasks
    4. Generate appropriate Parsl executor configurations
    5. Handle resource cleanup when workflows complete
    
    BRUTAL TRUTH: This is what prevents 70 workflows from crashing PBS!
    """
    
    def __init__(self):
        self.nodes: Dict[str, NodeResource] = {}
        self.allocations: Dict[str, ResourceAllocation] = {}
        self._lock = threading.RLock()
        self._last_discovery = 0
        self._discovery_interval = 300  # 5 minutes
        self._initialized = False
        
        logger.info("🏗️  PBSResourceManager initialized")
    
    async def initialize(self) -> bool:
        """Initialize the resource manager by discovering PBS nodes"""
        try:
            await self._discover_pbs_nodes()
            self._initialized = True
            logger.info(f"✅ PBSResourceManager initialized with {len(self.nodes)} nodes")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize PBSResourceManager: {e}")
            return False
    
    async def _discover_pbs_nodes(self) -> None:
        """Discover available PBS nodes and their core counts"""
        try:
            # Try to get PBS node information
            result = await asyncio.create_subprocess_exec(
                'pbsnodes', '-a', '-F', 'json',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()
            
            if result.returncode == 0:
                # Parse PBS node information
                pbs_data = json.loads(stdout.decode())
                await self._parse_pbs_nodes(pbs_data)
            else:
                logger.warning(f"⚠️  PBS node discovery failed: {stderr.decode()}")
                # Fallback to environment-based detection
                await self._fallback_node_discovery()
                
        except Exception as e:
            logger.warning(f"⚠️  PBS discovery error: {e}, using fallback")
            await self._fallback_node_discovery()
        
        self._last_discovery = time.time()
    
    async def _parse_pbs_nodes(self, pbs_data: Dict) -> None:
        """Parse PBS node data and update node resources"""
        with self._lock:
            for node_name, node_info in pbs_data.get('nodes', {}).items():
                if node_info.get('state') in ['free', 'job-exclusive', 'job-sharing']:
                    # Extract core count
                    resources = node_info.get('resources_available', {})
                    ncpus = int(resources.get('ncpus', 8))  # Default to 8 cores
                    
                    if node_name not in self.nodes:
                        self.nodes[node_name] = NodeResource(
                            node_id=node_name,
                            total_cores=ncpus,
                            reserved_cores=max(1, ncpus // 16)  # Reserve 1 core per 16, minimum 1
                        )
                        logger.info(f"📊 Discovered node {node_name}: {ncpus} cores "
                                  f"({self.nodes[node_name].reserved_cores} reserved)")
    
    async def _fallback_node_discovery(self) -> None:
        """Fallback node discovery when PBS commands are not available"""
        with self._lock:
            # Use current hostname as a single node
            hostname = os.environ.get('PBS_NODEFILE_HOST', 'localhost')
            if hostname not in self.nodes:
                # Default Aurora node configuration
                total_cores = int(os.environ.get('PBS_NCPUS', '8'))
                self.nodes[hostname] = NodeResource(
                    node_id=hostname,
                    total_cores=total_cores,
                    reserved_cores=1  # Always reserve 1 core
                )
                logger.info(f"📊 Fallback node {hostname}: {total_cores} cores (1 reserved)")
    
    async def allocate_resources(self, workflow_uuid: str, requested_cores: int) -> Optional[ResourceAllocation]:
        """
        Allocate resources for a workflow
        
        Args:
            workflow_uuid: Unique workflow identifier
            requested_cores: Number of cores requested
            
        Returns:
            ResourceAllocation if successful, None if insufficient resources
        """
        if not self._initialized:
            await self.initialize()
        
        # Refresh node information if needed
        if time.time() - self._last_discovery > self._discovery_interval:
            await self._discover_pbs_nodes()
        
        with self._lock:
            # Find a node with sufficient available cores
            for node_id, node in self.nodes.items():
                if node.available_cores >= requested_cores:
                    # Allocate resources
                    allocation = ResourceAllocation(
                        workflow_uuid=workflow_uuid,
                        node_id=node_id,
                        allocated_cores=requested_cores,
                        max_workers=requested_cores  # 1 worker per core
                    )
                    
                    # Update node allocation
                    node.allocated_cores += requested_cores
                    node.active_workflows.add(workflow_uuid)
                    
                    # Store allocation
                    self.allocations[workflow_uuid] = allocation
                    
                    logger.info(f"✅ Allocated {requested_cores} cores on {node_id} for workflow {workflow_uuid[:8]}...")
                    logger.info(f"📊 Node {node_id}: {node.available_cores} cores remaining "
                              f"({node.utilization:.1f}% utilized)")
                    
                    return allocation
            
            # No suitable node found
            logger.warning(f"⚠️  Insufficient resources for workflow {workflow_uuid[:8]}... "
                         f"(requested: {requested_cores} cores)")
            return None
    
    async def release_resources(self, workflow_uuid: str) -> bool:
        """Release resources allocated to a workflow"""
        with self._lock:
            if workflow_uuid not in self.allocations:
                logger.warning(f"⚠️  No allocation found for workflow {workflow_uuid[:8]}...")
                return False
            
            allocation = self.allocations[workflow_uuid]
            node = self.nodes.get(allocation.node_id)
            
            if node:
                # Release resources
                node.allocated_cores -= allocation.allocated_cores
                node.active_workflows.discard(workflow_uuid)
                
                logger.info(f"✅ Released {allocation.allocated_cores} cores on {allocation.node_id} "
                          f"from workflow {workflow_uuid[:8]}...")
                logger.info(f"📊 Node {allocation.node_id}: {node.available_cores} cores available "
                          f"({node.utilization:.1f}% utilized)")
            
            # Remove allocation
            del self.allocations[workflow_uuid]
            return True
    
    def get_resource_summary(self) -> Dict:
        """Get summary of current resource allocation"""
        with self._lock:
            summary = {
                'total_nodes': len(self.nodes),
                'total_cores': sum(node.total_cores for node in self.nodes.values()),
                'reserved_cores': sum(node.reserved_cores for node in self.nodes.values()),
                'allocated_cores': sum(node.allocated_cores for node in self.nodes.values()),
                'available_cores': sum(node.available_cores for node in self.nodes.values()),
                'active_workflows': len(self.allocations),
                'nodes': {
                    node_id: {
                        'total_cores': node.total_cores,
                        'reserved_cores': node.reserved_cores,
                        'allocated_cores': node.allocated_cores,
                        'available_cores': node.available_cores,
                        'utilization': node.utilization,
                        'active_workflows': len(node.active_workflows)
                    }
                    for node_id, node in self.nodes.items()
                }
            }
            return summary

    def get_allocation(self, workflow_uuid: str) -> Optional[ResourceAllocation]:
        """Get resource allocation for a specific workflow"""
        with self._lock:
            return self.allocations.get(workflow_uuid)

    def get_recommended_allocation(self, total_workflows: int) -> Tuple[int, int]:
        """
        Get recommended resource allocation for workflow divergence

        Args:
            total_workflows: Total number of workflows to spawn

        Returns:
            Tuple of (cores_per_workflow, max_concurrent_workflows)
        """
        if not self._initialized:
            return (1, 1)  # Conservative fallback

        with self._lock:
            total_available = sum(node.available_cores for node in self.nodes.values())

            if total_available == 0:
                logger.warning("⚠️  No available cores for workflow allocation")
                return (1, 1)

            # FIXED: Conservative resource allocation strategy
            # For production workflows, use reasonable core counts per workflow

            # Strategy 1: Conservative allocation - max 8 cores per workflow
            max_cores_per_workflow = min(8, total_available // max(1, total_workflows))
            cores_per_workflow = max(1, max_cores_per_workflow)

            # Strategy 2: Limit concurrent workflows to prevent oversubscription
            max_concurrent = min(total_workflows, total_available // cores_per_workflow)

            # Strategy 3: For testing, use even more conservative limits
            if total_workflows <= 10:  # Testing scenario
                cores_per_workflow = min(2, cores_per_workflow)
                max_concurrent = min(5, max_concurrent)

            logger.info(f"📊 Resource recommendation for {total_workflows} workflows:")
            logger.info(f"   - Cores per workflow: {cores_per_workflow}")
            logger.info(f"   - Max concurrent workflows: {max_concurrent}")
            logger.info(f"   - Total available cores: {total_available}")
            logger.info(f"   - Conservative allocation to prevent oversubscription")

            return (cores_per_workflow, max_concurrent)

    async def wait_for_resources(self, workflow_uuid: str, requested_cores: int,
                                timeout: float = 300.0) -> Optional[ResourceAllocation]:
        """
        Wait for resources to become available

        Args:
            workflow_uuid: Workflow identifier
            requested_cores: Number of cores needed
            timeout: Maximum wait time in seconds

        Returns:
            ResourceAllocation if successful, None if timeout
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            allocation = await self.allocate_resources(workflow_uuid, requested_cores)
            if allocation:
                return allocation

            # Wait before retrying
            await asyncio.sleep(10)

            # Refresh node information
            if time.time() - self._last_discovery > 60:  # Refresh every minute when waiting
                await self._discover_pbs_nodes()

        logger.warning(f"⏰ Timeout waiting for {requested_cores} cores for workflow {workflow_uuid[:8]}...")
        return None


# Convenience function to get the shared PBS resource manager
def get_pbs_resource_manager() -> PBSResourceManager:
    """Get the shared PBS resource manager instance"""
    # The @shared decorator with auto_register=True handles creation automatically
    return PBSResourceManager()
