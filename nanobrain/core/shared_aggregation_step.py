"""
Shared Result Aggregation Step

This module provides the SharedResultAggregationStep - a SHARED step that collects
results from multiple diverged workflows. There is ONE instance per divergence point
that waits for all child workflows to complete and aggregates their results.

Key Features:
- Shared singleton per divergence point
- Asynchronous workflow completion waiting
- Result collection from executed_workflows/{uuid}/results/
- Consolidated result aggregation
- Fault tolerance for partial failures
"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set

from nanobrain.core.step import BaseStep, StepConfig
from nanobrain.core.logging_system import get_logger


class SharedResultAggregationStep(BaseStep):
    """
    SHARED aggregation step - ONE instance per divergence point.
    
    This step automatically discovers child workflows from a divergence point,
    waits asynchronously for their completion, collects results, and produces
    consolidated output for all diverged workflows.
    
    Key Features:
    - Automatically discovers child workflows from divergence manager
    - Waits asynchronously for workflow completion
    - Collects results from executed_workflows/{uuid}/results/
    - Handles partial failures gracefully
    - Produces consolidated timing and result reports
    """
    
    COMPONENT_TYPE = "shared_result_aggregation_step"
    REQUIRED_CONFIG_FIELDS = ['name', 'divergence_id']
    OPTIONAL_CONFIG_FIELDS = {
        'description': 'Shared result aggregation step',
        'auto_initialize': True,
        'debug_mode': False,
        'enable_logging': True
    }

    @classmethod
    def extract_component_config(cls, config: StepConfig) -> Dict[str, Any]:
        """Extract SharedResultAggregationStep configuration"""
        base_config = super().extract_component_config(config)
        return {
            **base_config,
            'divergence_id': getattr(config, 'divergence_id', None),
        }

    def _init_from_config(self, config: StepConfig, component_config: Dict[str, Any], dependencies: Dict[str, Any]) -> None:
        """Initialize SharedResultAggregationStep with proper framework compliance"""
        super()._init_from_config(config, component_config, dependencies)

        self.divergence_id = component_config['divergence_id']
        self.divergence_manager = dependencies.get('divergence_manager')
        self.expected_workflows: Set[str] = set()
        self.completed_workflows: Set[str] = set()
        self.collected_results: Dict[str, Any] = {}
        self.aggregation_lock = asyncio.Lock()
        self.start_time = time.time()

        # Set up logging
        self.logger = get_logger(f"{__name__}.SharedResultAggregationStep.{self.divergence_id}")

        self.logger.info(f"SharedResultAggregationStep initialized for divergence: {self.divergence_id}")
    
    async def register_expected_workflows(self, workflow_uuids: List[str]):
        """Register the workflows this aggregation step should wait for"""
        async with self.aggregation_lock:
            self.expected_workflows.update(workflow_uuids)
            self.logger.info(f"Aggregation step {self.divergence_id} expecting {len(self.expected_workflows)} workflows")
            self.logger.info(f"Expected workflow UUIDs: {list(self.expected_workflows)}")
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main aggregation process - waits for all workflows and aggregates results.
        
        This method is called by the framework and handles the entire aggregation lifecycle.
        """
        self.logger.info(f"🚀 Starting result aggregation for divergence point: {self.divergence_id}")
        
        # If no workflows registered yet, try to discover them from divergence manager
        if not self.expected_workflows and self.divergence_manager:
            await self._discover_expected_workflows()
        
        if not self.expected_workflows:
            self.logger.warning("No workflows to aggregate - returning empty results")
            return {
                'divergence_id': self.divergence_id,
                'status': 'no_workflows',
                'message': 'No workflows found to aggregate'
            }
        
        # Wait for all expected workflows to complete
        await self._wait_for_workflow_completion()
        
        # Collect results from all completed workflows
        await self._collect_all_results()
        
        # Aggregate and consolidate results
        consolidated_results = await self._consolidate_results()
        
        # Save aggregated results
        await self._save_aggregated_results(consolidated_results)
        
        self.logger.info(f"✅ Aggregation complete for divergence point: {self.divergence_id}")
        return consolidated_results
    
    async def _discover_expected_workflows(self):
        """Discover expected workflows from the divergence manager"""
        if not self.divergence_manager:
            self.logger.warning("No divergence manager available for workflow discovery")
            return
        
        child_workflows = self.divergence_manager.get_child_workflows(self.divergence_id)
        if child_workflows:
            await self.register_expected_workflows(child_workflows)
            self.logger.info(f"Discovered {len(child_workflows)} workflows from divergence manager")
        else:
            self.logger.warning(f"No child workflows found for divergence point: {self.divergence_id}")
    
    async def _wait_for_workflow_completion(self):
        """Wait for all expected workflows to complete"""
        self.logger.info(f"⏳ Waiting for {len(self.expected_workflows)} workflows to complete...")
        
        max_wait_time = 3600  # 1 hour timeout
        check_interval = 30   # Check every 30 seconds
        start_time = time.time()
        
        while len(self.completed_workflows) < len(self.expected_workflows):
            if time.time() - start_time > max_wait_time:
                self.logger.warning(f"⏰ Timeout waiting for workflows. Completed: {len(self.completed_workflows)}/{len(self.expected_workflows)}")
                break
            
            # Check for newly completed workflows
            newly_completed = await self._check_workflow_completion()
            if newly_completed:
                async with self.aggregation_lock:
                    self.completed_workflows.update(newly_completed)
                    self.logger.info(f"📈 Progress: {len(self.completed_workflows)}/{len(self.expected_workflows)} workflows completed")
            
            await asyncio.sleep(check_interval)
        
        self.logger.info(f"✅ Workflow completion check finished: {len(self.completed_workflows)}/{len(self.expected_workflows)} completed")
    
    async def _check_workflow_completion(self) -> Set[str]:
        """Check which workflows have completed since last check"""
        newly_completed = set()
        
        for workflow_uuid in self.expected_workflows:
            if workflow_uuid not in self.completed_workflows:
                # Check if workflow has results file
                results_file = Path(f"executed_workflows/{workflow_uuid}/results/workflow_results.json")
                if results_file.exists():
                    newly_completed.add(workflow_uuid)
                    self.logger.debug(f"Found completed workflow: {workflow_uuid}")
        
        return newly_completed
    
    async def _collect_all_results(self):
        """Collect results from all completed workflows"""
        self.logger.info(f"📊 Collecting results from {len(self.completed_workflows)} completed workflows")
        
        for workflow_uuid in self.completed_workflows:
            try:
                results = await self._load_workflow_results(workflow_uuid)
                self.collected_results[workflow_uuid] = results
                self.logger.debug(f"Collected results from workflow: {workflow_uuid}")
            except Exception as e:
                self.logger.error(f"❌ Failed to collect results from workflow {workflow_uuid}: {e}")
                self.collected_results[workflow_uuid] = {
                    'status': 'failed',
                    'error': str(e),
                    'workflow_uuid': workflow_uuid
                }
        
        self.logger.info(f"✅ Result collection complete: {len(self.collected_results)} workflows processed")
    
    async def _load_workflow_results(self, workflow_uuid: str) -> Dict[str, Any]:
        """Load results from a specific workflow"""
        results_file = Path(f"executed_workflows/{workflow_uuid}/results/workflow_results.json")
        
        if not results_file.exists():
            raise FileNotFoundError(f"Results file not found: {results_file}")
        
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        return results

    async def _consolidate_results(self) -> Dict[str, Any]:
        """Consolidate results from all workflows into final output"""
        self.logger.info("🔄 Consolidating results from all workflows")

        successful_results = {}
        failed_results = {}
        timing_data = {}

        for workflow_uuid, results in self.collected_results.items():
            if results.get('status') == 'failed':
                failed_results[workflow_uuid] = results
            else:
                # Extract species name from results or workflow metadata
                species = results.get('species', results.get('workflow_metadata', {}).get('species', workflow_uuid))
                successful_results[species] = {
                    'workflow_uuid': workflow_uuid,
                    'data': results.get('data', {}),
                    'timing': results.get('timing', {}),
                    'metadata': results.get('metadata', {})
                }

                # Collect timing data
                if 'timing' in results:
                    timing_data[species] = results['timing']

        # Create consolidated timing report
        consolidated_timing = self._create_consolidated_timing_report(timing_data)

        return {
            'divergence_id': self.divergence_id,
            'aggregation_timestamp': datetime.now().isoformat(),
            'total_workflows_expected': len(self.expected_workflows),
            'total_workflows_completed': len(self.completed_workflows),
            'successful_workflows': len(successful_results),
            'failed_workflows': len(failed_results),
            'species_results': successful_results,
            'failed_workflows_details': failed_results,
            'consolidated_timing': consolidated_timing,
            'aggregation_metadata': {
                'aggregation_duration': time.time() - self.start_time,
                'completion_rate': len(self.completed_workflows) / len(self.expected_workflows) if self.expected_workflows else 0,
                'success_rate': len(successful_results) / len(self.completed_workflows) if self.completed_workflows else 0
            }
        }

    def _create_consolidated_timing_report(self, timing_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create consolidated timing report from all workflows"""
        if not timing_data:
            return {'message': 'No timing data available'}

        total_times = []
        step_times = {}

        for species, timing in timing_data.items():
            if 'total_time' in timing:
                total_times.append(timing['total_time'])

            # Collect step-specific timing
            for step_name, step_timing in timing.items():
                if step_name != 'total_time' and isinstance(step_timing, (int, float)):
                    if step_name not in step_times:
                        step_times[step_name] = []
                    step_times[step_name].append(step_timing)

        # Calculate statistics
        consolidated = {
            'total_species_processed': len(timing_data),
            'total_execution_time': {
                'min': min(total_times) if total_times else 0,
                'max': max(total_times) if total_times else 0,
                'avg': sum(total_times) / len(total_times) if total_times else 0,
                'sum': sum(total_times) if total_times else 0
            },
            'step_timing_statistics': {}
        }

        # Calculate step-specific statistics
        for step_name, times in step_times.items():
            consolidated['step_timing_statistics'][step_name] = {
                'min': min(times),
                'max': max(times),
                'avg': sum(times) / len(times),
                'count': len(times)
            }

        return consolidated

    async def _save_aggregated_results(self, consolidated_results: Dict[str, Any]):
        """Save aggregated results to file"""
        # Create aggregation results directory
        aggregation_dir = Path(f"executed_workflows/aggregation_{self.divergence_id}")
        aggregation_dir.mkdir(exist_ok=True)

        # Save consolidated results
        results_file = aggregation_dir / "consolidated_results.json"
        with open(results_file, 'w') as f:
            json.dump(consolidated_results, f, indent=2)

        # Save timing report separately
        timing_file = aggregation_dir / "timing_report.json"
        with open(timing_file, 'w') as f:
            json.dump(consolidated_results['consolidated_timing'], f, indent=2)

        # Save aggregation metadata
        metadata_file = aggregation_dir / "aggregation_metadata.json"
        metadata = {
            'divergence_id': self.divergence_id,
            'aggregation_completed_at': datetime.now().isoformat(),
            'results_file': str(results_file),
            'timing_file': str(timing_file),
            'summary': {
                'total_workflows_expected': consolidated_results['total_workflows_expected'],
                'total_workflows_completed': consolidated_results['total_workflows_completed'],
                'successful_workflows': consolidated_results['successful_workflows'],
                'failed_workflows': consolidated_results['failed_workflows']
            }
        }
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(f"💾 Aggregated results saved to: {aggregation_dir}")
        self.logger.info(f"   - Results: {results_file}")
        self.logger.info(f"   - Timing: {timing_file}")
        self.logger.info(f"   - Metadata: {metadata_file}")

    def get_aggregation_status(self) -> Dict[str, Any]:
        """Get current aggregation status"""
        return {
            'divergence_id': self.divergence_id,
            'expected_workflows': len(self.expected_workflows),
            'completed_workflows': len(self.completed_workflows),
            'collected_results': len(self.collected_results),
            'completion_rate': len(self.completed_workflows) / len(self.expected_workflows) if self.expected_workflows else 0,
            'expected_workflow_uuids': list(self.expected_workflows),
            'completed_workflow_uuids': list(self.completed_workflows)
        }
