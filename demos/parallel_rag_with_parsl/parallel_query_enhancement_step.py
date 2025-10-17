#!/usr/bin/env python3
"""
Parallel Query Enhancement Step with PARSL Executor
====================================================

Custom step that uses PARSL executor for parallel query enhancement.
Demonstrates:
- PARSL executor integration
- Worker ID tracking in outputs
- Shared resource access (vector database)
"""

import asyncio
import logging
from typing import Any, Dict

from nanobrain.library.workflows.rag.steps import QueryEnhancementStep
from nanobrain.core.executor import ParslExecutor, ExecutorConfig, ExecutorType
from nanobrain.core.shared_resource import get_worker_id, get_resource_pool


class ParallelQueryEnhancementStep(QueryEnhancementStep):
    """
    Query Enhancement Step with PARSL parallel execution.
    
    Extends the standard QueryEnhancementStep to use PARSL executor
    for parallel processing of multiple queries.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with PARSL executor."""
        super().__init__(config)

        self.parsl_executor = None
        
        # PARSL configuration
        self.parsl_config = {
            'executor_type': 'parsl',
            'max_workers': 4,
            'timeout': 60,
            'parsl_config': {
                'strategy': None,
                'app_cache': True,
                'checkpoint_mode': None,
                'retries': 1,
                'executors': [{
                    'label': 'htex_local_parallel',
                    'class': 'parsl.executors.HighThroughputExecutor',
                    'max_workers_per_node': 4,
                    'worker_debug': False,
                    'heartbeat_period': 5,
                    'heartbeat_threshold': 10,
                    'provider_config': {
                        'class': 'parsl.providers.LocalProvider',
                        'min_blocks': 1,
                        'init_blocks': 1,
                        'max_blocks': 1,
                        'worker_init': '',
                        'parallelism': 1.0
                    }
                }]
            }
        }
    
    async def initialize(self):
        """Initialize the step and PARSL executor."""
        await super().initialize()

        # Create and initialize PARSL executor
        try:
            if self.nb_logger:
                self.nb_logger.info("Initializing PARSL executor for parallel query processing")

            executor_config = ExecutorConfig(**self.parsl_config)
            self.parsl_executor = ParslExecutor(config=executor_config)
            await self.parsl_executor.initialize()

            if self.nb_logger:
                self.nb_logger.info("✅ PARSL executor initialized successfully")
                self.nb_logger.info(f"   Max workers: {self.parsl_config['max_workers']}")

        except Exception as e:
            if self.nb_logger:
                self.nb_logger.warning(f"Failed to initialize PARSL executor: {e}")
                self.nb_logger.warning("Falling back to standard execution")
            self.parsl_executor = None
    
    async def process(self, *args, **kwargs):
        """
        Process query using PARSL executor if available.

        Adds worker ID tracking to outputs for monitoring parallel execution.
        """
        if self.parsl_executor and self.parsl_executor.is_initialized:
            if self.nb_logger:
                self.nb_logger.info("🎯 Using PARSL executor for parallel processing")

            # Get worker ID for this execution
            worker_id = get_worker_id()

            # Use PARSL executor for processing
            try:
                # Execute with worker ID tracking enabled
                result = await self.parsl_executor.execute(
                    lambda: super().process(*args, **kwargs),
                    add_worker_id=True,  # Enable worker ID tracking
                    **kwargs
                )

                # Log worker ID
                if self.nb_logger:
                    result_worker_id = result.get('_worker_id', 'unknown') if isinstance(result, dict) else 'unknown'
                    self.nb_logger.info(f"✅ PARSL execution completed by worker: {result_worker_id}")

                # Log shared resource access
                resource_pool = get_resource_pool()
                resources = resource_pool.list_resources()
                if resources and self.nb_logger:
                    self.nb_logger.info(f"📊 Shared resources in pool: {len(resources)}")
                    for res_id, stats in resources.items():
                        self.nb_logger.info(f"   - {res_id}: {stats['access_count']} accesses, "
                                          f"{stats['active_workers']} active workers")

                return result
            except Exception as e:
                if self.nb_logger:
                    self.nb_logger.warning(f"PARSL execution failed: {e}, falling back to standard execution")
                return await super().process(*args, **kwargs)
        else:
            # Fall back to standard processing
            result = await super().process(*args, **kwargs)

            # Add worker ID even in fallback mode
            if isinstance(result, dict):
                result['_worker_id'] = get_worker_id()
                result['_executor_type'] = 'fallback'

            return result
    
    async def shutdown(self):
        """Shutdown the step and PARSL executor."""
        if self.parsl_executor:
            await self.parsl_executor.shutdown()
        
        await super().shutdown()

