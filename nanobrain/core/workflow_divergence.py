"""
Workflow Divergence Management System

This module provides the core infrastructure for workflow divergence - the ability
to spawn multiple child workflows from a single configuration change point.

Key Features:
- Configuration-triggered workflow spawning
- Unique UUID generation and tracking
- Workflow genealogy management
- Directory structure creation for executed workflows
- Integration with existing Nanobrain infrastructure
"""

import asyncio
import json
import uuid
import yaml
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from nanobrain.core.shared_resource import shared
from nanobrain.core.logging_system import get_logger
from nanobrain.core.pbs_resource_manager import get_pbs_resource_manager, ResourceAllocation
from nanobrain.core.dynamic_executor_config import DynamicExecutorConfigGenerator


@dataclass
class DivergencePoint:
    """Represents a point where workflows diverge into multiple child workflows"""
    divergence_id: str
    parent_config: Dict[str, Any]
    divergence_config: Dict[str, Any]
    created_at: datetime
    child_workflows: List[str]  # List of child workflow UUIDs
    status: str = "active"  # active, completed, failed


@dataclass
class WorkflowInfo:
    """Information about a spawned workflow"""
    workflow_uuid: str
    parent_divergence_id: str
    config: Dict[str, Any]
    status: str  # pending, running, completed, failed
    created_at: datetime
    pbs_job_id: Optional[str] = None
    results_path: Optional[str] = None


@shared(resource_type='workflow_divergence_manager', auto_register=True)
class WorkflowDivergenceManager:
    """
    Core manager for workflow divergence operations.
    
    This is a SHARED singleton that manages all workflow divergence across the system.
    It handles UUID generation, directory creation, workflow tracking, and coordination
    with aggregation steps.
    """
    
    def __init__(self):
        self.logger = get_logger(f"{__name__}.WorkflowDivergenceManager")
        self.divergence_points: Dict[str, DivergencePoint] = {}
        self.workflow_genealogy: Dict[str, str] = {}  # child_uuid -> parent_divergence_id
        self.active_workflows: Dict[str, WorkflowInfo] = {}
        self.base_executed_workflows_dir = Path("executed_workflows")

        # Resource management components
        self.pbs_resource_manager = get_pbs_resource_manager()
        self.executor_config_generator = DynamicExecutorConfigGenerator()
        self.workflow_allocations: Dict[str, ResourceAllocation] = {}  # workflow_uuid -> allocation

        # Ensure base directory exists
        self.base_executed_workflows_dir.mkdir(exist_ok=True)

        self.logger.info("WorkflowDivergenceManager initialized with resource management")
    
    async def register_divergence_point(
        self, 
        divergence_id: str,
        parent_workflow_config: Dict[str, Any],
        divergence_config: Dict[str, Any]
    ) -> str:
        """
        Register a new divergence point in the system.
        
        Args:
            divergence_id: Unique identifier for this divergence point
            parent_workflow_config: Base workflow configuration
            divergence_config: Configuration that defines how to diverge
            
        Returns:
            Divergence point ID (same as input for validation)
        """
        if divergence_id in self.divergence_points:
            self.logger.warning(f"Divergence point {divergence_id} already exists, updating")
        
        divergence_point = DivergencePoint(
            divergence_id=divergence_id,
            parent_config=parent_workflow_config,
            divergence_config=divergence_config,
            created_at=datetime.now(),
            child_workflows=[]
        )
        
        self.divergence_points[divergence_id] = divergence_point
        
        self.logger.info(f"✅ Registered divergence point: {divergence_id}")
        return divergence_id
    
    async def spawn_diverged_workflows(
        self,
        divergence_id: str,
        spawn_parameters: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Spawn multiple workflows from a divergence point with resource management.

        CRITICAL RESOURCE MANAGEMENT:
        - Initialize PBS Resource Manager
        - Allocate resources per workflow
        - Generate unique executor configs with proper maximum_workers
        - Reserve cores for system tasks
        - Handle resource allocation failures gracefully

        Args:
            divergence_id: The divergence point to spawn from
            spawn_parameters: List of parameter sets for each workflow

        Returns:
            List of spawned workflow UUIDs
        """
        if divergence_id not in self.divergence_points:
            raise ValueError(f"Divergence point {divergence_id} not found")

        divergence_point = self.divergence_points[divergence_id]
        spawned_workflows = []

        self.logger.info(f"🚀 Spawning {len(spawn_parameters)} workflows from divergence point: {divergence_id}")

        # CRITICAL: Initialize PBS Resource Manager
        await self.pbs_resource_manager.initialize()

        # Get resource recommendations for this divergence
        cores_per_workflow, max_concurrent = self.pbs_resource_manager.get_recommended_allocation(
            len(spawn_parameters)
        )

        self.logger.info("📊 Resource allocation strategy:")
        self.logger.info(f"   - Workflows to spawn: {len(spawn_parameters)}")
        self.logger.info(f"   - Cores per workflow: {cores_per_workflow}")
        self.logger.info(f"   - Max concurrent workflows: {max_concurrent}")

        # Process workflows with resource allocation
        for i, params in enumerate(spawn_parameters):
            try:
                # Generate unique workflow UUID
                workflow_uuid = str(uuid.uuid4())

                self.logger.info(f"🔄 Processing workflow {i+1}/{len(spawn_parameters)}: {workflow_uuid[:8]}...")

                # CRITICAL: Allocate PBS resources for this workflow
                allocation = await self.pbs_resource_manager.wait_for_resources(
                    workflow_uuid, cores_per_workflow, timeout=300.0
                )

                if not allocation:
                    self.logger.error(f"❌ Failed to allocate {cores_per_workflow} cores for workflow {workflow_uuid[:8]}...")
                    continue

                # Store allocation for cleanup
                self.workflow_allocations[workflow_uuid] = allocation

                # Create workflow configuration by merging base + parameters
                workflow_config = self._merge_configurations(
                    divergence_point.parent_config,
                    params
                )

                # CRITICAL FIX: Use the SAME executor configuration from parent workflow
                # For single shared WorkQueue, all workflows must use the same executor config
                # DO NOT generate separate executor configs - that creates multiple PBS jobs!
                parent_executor = divergence_point.parent_config.get('executor', {})
                if 'config_file' in parent_executor:
                    # Use the shared executor configuration specified in orchestrator
                    shared_executor_config_path = parent_executor['config_file']
                    self.logger.info(f"🔗 Using shared executor config: {shared_executor_config_path}")

                    workflow_config['executor'] = {
                        'class': parent_executor.get('class', 'nanobrain.core.executor.ParslExecutor'),
                        'config': shared_executor_config_path
                    }

                    # Create workflow directory structure and duplicate configs
                    # CRITICAL: This must happen AFTER setting executor config so it gets duplicated too
                    workflow_dir = await self._create_workflow_directory(workflow_uuid, workflow_config)

                    # CRITICAL FIX: Get the updated workflow config with corrected paths from directory creation
                    # The _create_workflow_directory method updates paths to point to duplicated configs
                    updated_config_file = workflow_dir / "workflow_config.yml"
                    with open(updated_config_file, 'r') as f:
                        workflow_config = yaml.safe_load(f)

                else:
                    # Fallback: Generate workflow-specific config (old behavior)
                    self.logger.warning("⚠️ No shared executor config found, generating workflow-specific config")

                    # Create workflow directory structure first
                    workflow_dir = await self._create_workflow_directory(workflow_uuid, workflow_config)

                    executor_config_path = self.executor_config_generator.generate_workflow_executor_config(
                        workflow_uuid, allocation, workflow_dir
                    )
                    workflow_config['executor'] = {
                        'class': 'nanobrain.core.executor.ParslExecutor',
                        'config': executor_config_path
                    }

                # Add workflow metadata including resource allocation
                workflow_config['workflow_metadata'] = {
                    'workflow_uuid': workflow_uuid,
                    'parent_divergence_id': divergence_id,
                    'spawn_index': i,
                    'spawn_timestamp': datetime.now().isoformat(),
                    'resource_allocation': {
                        'node_id': allocation.node_id,
                        'allocated_cores': allocation.allocated_cores,
                        'max_workers': allocation.max_workers,
                        'executor_config_path': workflow_config['executor']['config'],
                        'shared_executor': 'config_file' in parent_executor
                    },
                    **params.get('workflow_metadata', {})
                }

                # Save updated workflow configuration
                await self._save_workflow_config(workflow_uuid, workflow_config)

                # Create workflow info object
                workflow_info = WorkflowInfo(
                    workflow_uuid=workflow_uuid,
                    parent_divergence_id=divergence_id,
                    config=workflow_config,
                    status="resource_allocated",
                    created_at=datetime.now()
                )

                # Register workflow in tracking systems
                self.workflow_genealogy[workflow_uuid] = divergence_id
                self.active_workflows[workflow_uuid] = workflow_info
                divergence_point.child_workflows.append(workflow_uuid)

                # CRITICAL: ACTUALLY EXECUTE THE WORKFLOW - NO MOCKS!
                await self._execute_real_workflow(workflow_uuid, workflow_config, allocation)

                spawned_workflows.append(workflow_uuid)

                self.logger.info(f"✅ Spawned workflow {i+1}/{len(spawn_parameters)}: {workflow_uuid[:8]}... "
                               f"(Node: {allocation.node_id}, Cores: {allocation.allocated_cores})")

            except Exception as e:
                self.logger.error(f"❌ Failed to spawn workflow {i+1}: {e}")
                # Clean up any partial allocation
                if workflow_uuid in self.workflow_allocations:
                    await self.pbs_resource_manager.release_resources(workflow_uuid)
                    del self.workflow_allocations[workflow_uuid]
                continue

        self.logger.info(f"✅ Successfully spawned {len(spawned_workflows)}/{len(spawn_parameters)} workflows with resource allocation")
        return spawned_workflows

    async def _execute_real_workflow(self, workflow_uuid: str, workflow_config: Dict[str, Any], allocation: ResourceAllocation):
        """
        Execute the actual workflow - NO MOCKS OR SYNTHETIC DATA!

        This method launches the real workflow execution using the allocated PBS resources
        and the generated executor configuration.

        Args:
            workflow_uuid: Unique workflow identifier
            workflow_config: Complete workflow configuration
            allocation: PBS resource allocation for this workflow
        """
        try:
            self.logger.info(f"🚀 Executing real workflow {workflow_uuid[:8]}... on node {allocation.node_id}")

            # Update workflow status to running
            await self.update_workflow_status(workflow_uuid, "running")

            # Get the workflow directory
            workflow_dir = self.base_executed_workflows_dir / workflow_uuid
            config_path = workflow_dir / "workflow_config.yml"

            # Execute the workflow using Nanobrain's standard workflow execution
            from nanobrain.core.workflow import Workflow

            # Load workflow from the generated configuration
            workflow = Workflow.from_config(str(config_path))
            await workflow.initialize()

            # Execute the workflow asynchronously (fire-and-forget)
            # This allows multiple workflows to run in parallel
            asyncio.create_task(self._run_workflow_async(workflow, workflow_uuid, workflow_config))

            self.logger.info(f"✅ Launched real workflow execution for {workflow_uuid[:8]}...")

        except Exception as e:
            self.logger.error(f"❌ Failed to execute workflow {workflow_uuid[:8]}: {e}")
            await self.update_workflow_status(workflow_uuid, "failed")
            raise

    async def _run_workflow_async(self, workflow: 'Workflow', workflow_uuid: str, workflow_config: Dict[str, Any]):
        """
        Run workflow asynchronously and handle completion/failure

        Args:
            workflow: Nanobrain Workflow instance
            workflow_uuid: Workflow identifier
            workflow_config: Workflow configuration containing species data
        """
        try:
            # Prepare input data from the workflow configuration
            # Extract species information for real data processing
            species_name = workflow_config.get('species_name', 'Unknown')

            input_data = {
                'species_name': species_name,
                'workflow_uuid': workflow_uuid,
                'timestamp': datetime.now().isoformat(),
                'real_data_processing': True  # Flag to ensure no mocks
            }

            self.logger.info(f"🔄 Processing real data for species: {species_name} (workflow {workflow_uuid[:8]})")

            # Execute the workflow with real data processing
            results = await workflow.process(input_data)

            # Save results to the workflow directory
            workflow_dir = self.base_executed_workflows_dir / workflow_uuid
            results_dir = workflow_dir / "results"
            results_dir.mkdir(exist_ok=True)

            results_file = results_dir / "workflow_results.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)

            # Update status to completed
            await self.update_workflow_status(workflow_uuid, "completed")

            self.logger.info(f"✅ Workflow {workflow_uuid[:8]} completed successfully for species: {species_name}")

        except Exception as e:
            self.logger.error(f"❌ Workflow {workflow_uuid[:8]} failed: {e}")
            await self.update_workflow_status(workflow_uuid, "failed")

        finally:
            # Clean up resources
            if workflow_uuid in self.workflow_allocations:
                await self.pbs_resource_manager.release_resources(workflow_uuid)
                del self.workflow_allocations[workflow_uuid]

    def _merge_configurations(self, base_config: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge base configuration with spawn parameters.

        Args:
            base_config: Base workflow configuration
            parameters: Parameters specific to this workflow spawn

        Returns:
            Merged configuration
        """
        import copy
        merged_config = copy.deepcopy(base_config)

        # Apply parameter overrides
        for key, value in parameters.items():
            if key == 'workflow_metadata':
                continue  # Handle separately

            # Handle nested configuration updates
            if isinstance(value, dict) and key in merged_config:
                if isinstance(merged_config[key], dict):
                    merged_config[key].update(value)
                else:
                    merged_config[key] = value
            else:
                merged_config[key] = value

        return merged_config

    async def _create_workflow_directory(self, workflow_uuid: str, workflow_config: Dict[str, Any]) -> Path:
        """
        Create directory structure for a workflow.

        Creates:
        executed_workflows/{workflow_uuid}/
        ├── workflow_config.yml
        ├── step_configs/
        ├── logs/
        └── results/

        CRITICAL: Recursively duplicates ALL referenced configuration files!
        This ensures each workflow has its own complete configuration set.

        Returns:
            Path to the created workflow directory
        """
        workflow_dir = self.base_executed_workflows_dir / workflow_uuid
        workflow_dir.mkdir(exist_ok=True)

        # Create subdirectories
        (workflow_dir / "step_configs").mkdir(exist_ok=True)
        (workflow_dir / "logs").mkdir(exist_ok=True)
        (workflow_dir / "results").mkdir(exist_ok=True)

        # CRITICAL FIX: Recursively duplicate ALL referenced configuration files
        updated_workflow_config = await self._duplicate_referenced_configs(
            workflow_config, workflow_dir, workflow_uuid
        )

        # Save complete workflow configuration with updated paths
        config_file = workflow_dir / "workflow_config.yml"
        with open(config_file, 'w') as f:
            yaml.dump(updated_workflow_config, f, default_flow_style=False, indent=2)

        # Save workflow metadata
        metadata_file = workflow_dir / "workflow_metadata.json"
        metadata = {
            'workflow_uuid': workflow_uuid,
            'created_at': datetime.now().isoformat(),
            'config_file': str(config_file),
            'status': 'created'
        }
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        self.logger.debug(f"Created directory structure for workflow: {workflow_uuid}")
        return workflow_dir

    async def _duplicate_referenced_configs(
        self,
        config: Dict[str, Any],
        workflow_dir: Path,
        workflow_uuid: str
    ) -> Dict[str, Any]:
        """
        Recursively duplicate all configuration files referenced in the workflow config.

        This is CRITICAL for workflow divergence - each workflow must have its own
        complete set of configuration files, not shared references.

        Args:
            config: Configuration dictionary to process
            workflow_dir: Target workflow directory
            workflow_uuid: Workflow UUID for logging

        Returns:
            Updated configuration with local file paths
        """
        import copy

        updated_config = copy.deepcopy(config)

        # Process different types of config references
        await self._process_config_references(updated_config, workflow_dir, workflow_uuid)

        return updated_config

    async def _process_config_references(
        self,
        config_dict: Dict[str, Any],
        workflow_dir: Path,
        workflow_uuid: str,
        path_prefix: str = ""
    ) -> None:
        """
        Recursively process configuration references and duplicate files.

        Handles:
        - Step configurations: steps.*.config
        - Executor configurations: executor.config
        - Agent configurations: agents.*.config
        - Nested configurations at any level
        """

        for key, value in config_dict.items():
            current_path = f"{path_prefix}.{key}" if path_prefix else key

            if isinstance(value, dict):
                # Recursively process nested dictionaries
                await self._process_config_references(value, workflow_dir, workflow_uuid, current_path)

            elif isinstance(value, str) and key in ['config', 'config_file']:
                # Found a config file reference - duplicate it!
                await self._duplicate_config_file(
                    value, config_dict, key, workflow_dir, workflow_uuid, current_path
                )

            elif isinstance(value, list):
                # Process lists that might contain config references
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        await self._process_config_references(
                            item, workflow_dir, workflow_uuid, f"{current_path}[{i}]"
                        )

    async def _duplicate_config_file(
        self,
        config_path: str,
        parent_dict: Dict[str, Any],
        config_key: str,
        workflow_dir: Path,
        workflow_uuid: str,
        reference_path: str
    ) -> None:
        """
        Duplicate a single configuration file and update the reference.

        Args:
            config_path: Original config file path
            parent_dict: Dictionary containing the config reference
            config_key: Key in parent_dict that contains the config path
            workflow_dir: Target workflow directory
            workflow_uuid: Workflow UUID for logging
            reference_path: Path in config hierarchy for logging
        """
        import shutil
        from pathlib import Path

        try:
            # Resolve the source config file path
            source_path = Path(config_path)
            if not source_path.is_absolute():
                # Relative to repository root
                source_path = Path.cwd() / source_path

            if not source_path.exists():
                self.logger.error(f"❌ Config file not found: {source_path} (referenced in {reference_path})")
                return

            # Create target path in workflow directory
            relative_path = Path(config_path)
            if relative_path.is_absolute():
                # Convert absolute path to relative
                relative_path = relative_path.relative_to(Path.cwd())

            target_path = workflow_dir / "step_configs" / relative_path.name
            target_path.parent.mkdir(parents=True, exist_ok=True)

            # Copy the configuration file
            shutil.copy2(source_path, target_path)

            # Update the reference to point to the local copy
            # Use relative path from workflow directory
            local_config_path = f"step_configs/{relative_path.name}"
            parent_dict[config_key] = local_config_path

            self.logger.info(f"📋 Duplicated config: {config_path} -> {local_config_path} (workflow {workflow_uuid[:8]})")

        except Exception as e:
            self.logger.error(f"❌ Failed to duplicate config {config_path}: {e}")

    async def _save_workflow_config(self, workflow_uuid: str, workflow_config: Dict[str, Any]) -> None:
        """
        Save updated workflow configuration to file.

        Args:
            workflow_uuid: Workflow identifier
            workflow_config: Updated workflow configuration
        """
        workflow_dir = self.base_executed_workflows_dir / workflow_uuid
        config_file = workflow_dir / "workflow_config.yml"

        with open(config_file, 'w') as f:
            yaml.dump(workflow_config, f, default_flow_style=False, indent=2)

        self.logger.debug(f"Updated workflow configuration for: {workflow_uuid}")

    def get_divergence_point(self, divergence_id: str) -> Optional[DivergencePoint]:
        """Get divergence point by ID"""
        return self.divergence_points.get(divergence_id)

    def get_workflow_info(self, workflow_uuid: str) -> Optional[WorkflowInfo]:
        """Get workflow info by UUID"""
        return self.active_workflows.get(workflow_uuid)

    def get_child_workflows(self, divergence_id: str) -> List[str]:
        """Get all child workflow UUIDs for a divergence point"""
        divergence_point = self.divergence_points.get(divergence_id)
        return divergence_point.child_workflows if divergence_point else []

    async def cleanup_workflow_resources(self, workflow_uuid: str) -> bool:
        """
        Clean up resources allocated to a completed workflow.

        Args:
            workflow_uuid: Workflow identifier

        Returns:
            True if cleanup successful, False otherwise
        """
        try:
            # Release PBS resources
            if workflow_uuid in self.workflow_allocations:
                success = await self.pbs_resource_manager.release_resources(workflow_uuid)
                if success:
                    del self.workflow_allocations[workflow_uuid]
                    self.logger.info(f"✅ Released resources for workflow {workflow_uuid[:8]}...")
                else:
                    self.logger.warning(f"⚠️  Failed to release resources for workflow {workflow_uuid[:8]}...")
                    return False

            # Update workflow status
            if workflow_uuid in self.active_workflows:
                self.active_workflows[workflow_uuid].status = "completed"

            return True

        except Exception as e:
            self.logger.error(f"❌ Error cleaning up workflow {workflow_uuid[:8]}...: {e}")
            return False

    def get_resource_summary(self) -> Dict[str, Any]:
        """Get summary of current resource allocation across all workflows"""
        pbs_summary = self.pbs_resource_manager.get_resource_summary()

        workflow_summary = {
            'total_workflows': len(self.active_workflows),
            'allocated_workflows': len(self.workflow_allocations),
            'divergence_points': len(self.divergence_points),
            'workflow_allocations': {
                workflow_uuid: {
                    'node_id': allocation.node_id,
                    'allocated_cores': allocation.allocated_cores,
                    'max_workers': allocation.max_workers,
                    'allocation_time': allocation.allocation_time
                }
                for workflow_uuid, allocation in self.workflow_allocations.items()
            }
        }

        return {
            'pbs_resources': pbs_summary,
            'workflow_resources': workflow_summary
        }

    async def update_workflow_status(self, workflow_uuid: str, status: str, **kwargs):
        """Update workflow status and metadata"""
        if workflow_uuid in self.active_workflows:
            workflow_info = self.active_workflows[workflow_uuid]
            workflow_info.status = status

            # Update additional fields
            for key, value in kwargs.items():
                if hasattr(workflow_info, key):
                    setattr(workflow_info, key, value)

            # Update metadata file
            workflow_dir = self.base_executed_workflows_dir / workflow_uuid
            metadata_file = workflow_dir / "workflow_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                metadata['status'] = status
                metadata['last_updated'] = datetime.now().isoformat()
                metadata.update(kwargs)
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)

            self.logger.info(f"Updated workflow {workflow_uuid} status: {status}")

    def get_status_summary(self) -> Dict[str, Any]:
        """Get summary of all divergence points and workflows"""
        summary = {
            'total_divergence_points': len(self.divergence_points),
            'total_workflows': len(self.active_workflows),
            'divergence_points': {},
            'workflow_status_counts': {}
        }

        # Count workflow statuses
        status_counts = {}
        for workflow_info in self.active_workflows.values():
            status = workflow_info.status
            status_counts[status] = status_counts.get(status, 0) + 1
        summary['workflow_status_counts'] = status_counts

        # Summarize divergence points
        for div_id, div_point in self.divergence_points.items():
            summary['divergence_points'][div_id] = {
                'child_workflow_count': len(div_point.child_workflows),
                'created_at': div_point.created_at.isoformat(),
                'status': div_point.status
            }

        return summary
