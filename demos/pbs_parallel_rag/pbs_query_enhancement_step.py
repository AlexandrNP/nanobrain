#!/usr/bin/env python3
"""
PBS Query Enhancement Step with PARSL PBS Provider
====================================================

Custom step that uses PARSL with PBS provider for parallel query enhancement.
Demonstrates:
- PARSL with PBS provider integration
- Worker ID tracking in outputs
- Shared resource access (vector database)
"""

import asyncio
import logging
from typing import Any, Dict

from nanobrain.library.workflows.rag.steps import QueryEnhancementStep
from nanobrain.core.executor import ParslExecutor, ExecutorConfig, ExecutorType
from nanobrain.core.shared_resource import get_worker_id, get_resource_pool


class PBSQueryEnhancementStep(QueryEnhancementStep):
    """
    Query Enhancement Step with PARSL using PBS provider.
    
    Extends the standard QueryEnhancementStep to use PARSL with PBS provider
    for parallel processing of multiple queries.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with PARSL PBS executor."""
        super().__init__(config)

        self.parsl_executor = None

        # Debug: Print the config to see what's being passed
        if self.nb_logger:
            self.nb_logger.info(f"🔍 PBSQueryEnhancementStep config keys: {list(config.keys())}")
            self.nb_logger.info(f"🔍 Looking for executor_config_path in config...")

        # Get executor config path from step config
        self.executor_config_path = config.get('executor_config_path')

        if self.nb_logger:
            self.nb_logger.info(f"🔍 Found executor_config_path: {self.executor_config_path}")

        # If not in config, try to construct path relative to step config
        if not self.executor_config_path:
            # Try to find it in the step's directory structure
            from pathlib import Path
            step_dir = Path(__file__).parent
            potential_path = step_dir / "config" / "executors" / "pbs_executor.yml"
            if potential_path.exists():
                self.executor_config_path = str(potential_path)
                if self.nb_logger:
                    self.nb_logger.info(f"🔍 Using fallback path: {self.executor_config_path}")
            else:
                if self.nb_logger:
                    self.nb_logger.warning(f"🔍 Fallback path not found: {potential_path}")

        # Use external configuration if provided, otherwise fall back to default
        if self.executor_config_path:
            # Will load from external config file
            self.parsl_config = None
        else:
            # Fallback PARSL configuration (old behavior)
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
                        'label': 'htex_pbs',
                        'class': 'parsl.executors.HighThroughputExecutor',
                        'max_workers_per_node': 4,
                        'worker_debug': False,
                        'heartbeat_period': 5,
                        'heartbeat_threshold': 10,
                        "provider_config": {
                            "class": "parsl.providers.PBSProvider",
                            "queue": "batch",
                            "nodes_per_block": 1,
                            "cpus_per_node": 2,
                            "walltime": "00:30:00",
                            "scheduler_options": "",
                            "worker_init": "",
                            "parallelism": 1.0
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

            # Try to get executor_config_path from multiple sources
            executor_config_path = None

            # First, try the attribute set in __init__
            if hasattr(self, 'executor_config_path') and self.executor_config_path:
                executor_config_path = self.executor_config_path
                if self.nb_logger:
                    self.nb_logger.info(f"Using executor_config_path from attribute: {executor_config_path}")

            # If not found, try to get it from the step's config directly
            elif hasattr(self, 'config') and self.config and 'executor_config_path' in self.config:
                executor_config_path = self.config['executor_config_path']
                if self.nb_logger:
                    self.nb_logger.info(f"Using executor_config_path from config: {executor_config_path}")

            # If still not found, try a relative path
            else:
                from pathlib import Path
                step_dir = Path(__file__).parent
                potential_path = step_dir / "config" / "executors" / "pbs_executor.yml"
                if potential_path.exists():
                    executor_config_path = str(potential_path)
                    if self.nb_logger:
                        self.nb_logger.info(f"Using fallback executor_config_path: {executor_config_path}")

            if executor_config_path:
                # Load from external configuration file (MPI-enabled)
                if self.nb_logger:
                    self.nb_logger.info(f"Loading Parsl config from: {executor_config_path}")

                # Load the executor configuration properly
                from nanobrain.core.executor import ExecutorConfig
                executor_config = ExecutorConfig.from_config(executor_config_path)
                self.parsl_executor = ParslExecutor.from_config(executor_config)
            else:
                # Use internal configuration (fallback)
                from nanobrain.core.executor import ExecutorConfig
                executor_config = ExecutorConfig(**self.parsl_config)
                self.parsl_executor = ParslExecutor.from_config(executor_config)

            await self.parsl_executor.initialize()

            if self.nb_logger:
                self.nb_logger.info("✅ PARSL executor initialized successfully")
                max_workers = executor_config.max_workers if hasattr(executor_config, 'max_workers') else 'unknown'
                self.nb_logger.info(f"   Max workers: {max_workers}")

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

