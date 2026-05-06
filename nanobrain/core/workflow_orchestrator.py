"""
Configuration-Driven Workflow Orchestrator

This module provides the main orchestrator that handles user configuration and
spawns workflows automatically. The user only needs to modify a configuration
file and run a single command - everything else happens automatically.

Key Features:
- Configuration-only user interaction
- Automatic workflow divergence and spawning
- Integrated result aggregation
- Resource-aware execution
- Seamless user experience
"""

import asyncio
import json
import yaml
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

from nanobrain.core.workflow_divergence import WorkflowDivergenceManager
from nanobrain.core.shared_resource import get_resource_pool
from nanobrain.core.logging_system import get_logger
from test_aggregation_simple import SimpleResultAggregator


class AlphavirusWorkflowOrchestrator:
    """
    Main orchestrator that handles user configuration and spawns workflows.
    
    This is the ONLY component the user interacts with through configuration.
    Everything else happens automatically:
    - Load species list
    - Create divergence point
    - Spawn workflows
    - Monitor completion
    - Aggregate results
    - Save consolidated output
    """
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.logger = get_logger(f"{__name__}.AlphavirusWorkflowOrchestrator")
        
        # Get shared divergence manager
        self.divergence_manager = WorkflowDivergenceManager()
        
        self.logger.info(f"AlphavirusWorkflowOrchestrator initialized with config: {config_path}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate required configuration sections
        required_sections = ['species_config', 'base_workflow', 'divergence_config']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Required configuration section missing: {section}")
        
        return config
    
    async def execute_alphavirus_analysis(self) -> Dict[str, Any]:
        """
        Main entry point - user calls this ONE method.
        Everything else happens automatically.
        """
        self.logger.info("🚀 Starting Alphavirus analysis with workflow divergence")
        
        try:
            # Phase 1: Load species list from configuration
            species_list = await self._load_species_list()
            self.logger.info(f"📋 Loaded {len(species_list)} species for analysis")
            
            # Phase 2: Register divergence point
            divergence_id = await self._register_divergence_point()
            self.logger.info(f"🔀 Registered divergence point: {divergence_id}")
            
            # Phase 3: Create spawn parameters for each species
            spawn_parameters = self._create_spawn_parameters(species_list, divergence_id)
            self.logger.info(f"⚙️ Created spawn parameters for {len(spawn_parameters)} workflows")
            
            # Phase 4: Spawn all workflows (this happens automatically)
            workflow_uuids = await self.divergence_manager.spawn_diverged_workflows(
                divergence_id=divergence_id,
                spawn_parameters=spawn_parameters
            )
            self.logger.info(f"🚀 Spawned {len(workflow_uuids)} workflows for species analysis")
            
            # Phase 5: Set up result aggregation
            aggregator = await self._setup_result_aggregation(divergence_id, workflow_uuids)
            
            # Phase 6: Wait for results and aggregate (this handles everything automatically)
            final_results = await self._wait_for_results_and_aggregate(aggregator)
            
            # Phase 7: Save final results
            await self._save_final_results(final_results)
            
            self.logger.info("✅ Alphavirus analysis completed successfully")
            return final_results
            
        except Exception as e:
            self.logger.error(f"❌ Alphavirus analysis failed: {e}")
            raise
    
    async def _load_species_list(self) -> List[str]:
        """Load species list from configuration"""
        species_config = self.config['species_config']

        if species_config['source'] == 'file':
            species_file = Path(species_config['species_file'])

            # If relative path, make it relative to config file directory
            if not species_file.is_absolute():
                species_file = self.config_path.parent / species_file

            if not species_file.exists():
                raise FileNotFoundError(f"Species file not found: {species_file}")

            with open(species_file, 'r') as f:
                species_data = json.load(f)

            species_list = species_data['species_list']

            # Apply limit if specified
            if 'limit' in species_config and species_config['limit'] is not None:
                species_list = species_list[:species_config['limit']]
                self.logger.info(f"Applied species limit: {species_config['limit']}")

            return species_list
        
        elif species_config['source'] == 'bvbrc_query':
            # TODO: Implement BV-BRC query-based species loading
            raise NotImplementedError("BV-BRC query-based species loading not yet implemented")
        
        else:
            raise ValueError(f"Unknown species source: {species_config['source']}")
    
    async def _register_divergence_point(self) -> str:
        """Register divergence point with the manager"""
        divergence_id = f"alphavirus_species_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        await self.divergence_manager.register_divergence_point(
            divergence_id=divergence_id,
            parent_workflow_config=self.config['base_workflow'],
            divergence_config=self.config['divergence_config']
        )
        
        return divergence_id
    
    def _create_spawn_parameters(self, species_list: List[str], divergence_id: str) -> List[Dict[str, Any]]:
        """Create spawn parameters for each species"""
        spawn_parameters = []
        
        for i, species in enumerate(species_list):
            # Normalize species name for file paths
            species_normalized = species.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
            
            spawn_parameters.append({
                'species_name': species,
                'cache_key': f"alphavirus_data_{species_normalized}",
                'output_file': f"data/alphavirus/{species_normalized}_data.json",
                'workflow_metadata': {
                    'species': species,
                    'species_index': i,
                    'divergence_point': divergence_id,
                    'orchestrator_run': True,
                    'spawn_timestamp': datetime.now().isoformat()
                }
            })
        
        return spawn_parameters
    
    async def _setup_result_aggregation(self, divergence_id: str, workflow_uuids: List[str]) -> SimpleResultAggregator:
        """Set up result aggregation for the spawned workflows"""
        aggregator = SimpleResultAggregator(divergence_id)
        await aggregator.register_expected_workflows(workflow_uuids)
        
        self.logger.info(f"📊 Set up result aggregation for {len(workflow_uuids)} workflows")
        return aggregator
    
    async def _wait_for_results_and_aggregate(self, aggregator: SimpleResultAggregator) -> Dict[str, Any]:
        """Wait for workflow completion and aggregate results"""
        self.logger.info("⏳ Waiting for workflows to complete and aggregating results...")

        # REAL WORKFLOW EXECUTION - NO MOCKS!
        # Wait for actual workflow execution to complete
        await self._wait_for_real_workflow_completion(aggregator)

        # Check for completed workflows
        newly_completed = await aggregator.check_workflow_completion()
        aggregator.completed_workflows.update(newly_completed)

        # Collect and consolidate results
        await aggregator.collect_all_results()
        consolidated_results = await aggregator.consolidate_results()

        self.logger.info(f"📊 Aggregated results from {consolidated_results['total_workflows_completed']} workflows")
        return consolidated_results
    
    async def _wait_for_real_workflow_completion(self, aggregator: SimpleResultAggregator):
        """Wait for real workflow execution to complete - NO MOCKS!"""
        import asyncio
        from pathlib import Path

        self.logger.info("🔄 Monitoring real workflow execution...")

        # Monitor workflow completion by checking for result files
        max_wait_time = 4 * 3600  # 4 hours as configured
        check_interval = 30  # Check every 30 seconds
        start_time = asyncio.get_event_loop().time()

        while True:
            current_time = asyncio.get_event_loop().time()
            elapsed = current_time - start_time

            if elapsed > max_wait_time:
                self.logger.warning(f"⚠️ Timeout reached ({max_wait_time}s), proceeding with available results")
                break

            # Check how many workflows have completed
            completed_count = 0
            for workflow_uuid in aggregator.expected_workflows:
                result_file = Path(f"executed_workflows/{workflow_uuid}/results/workflow_results.json")
                if result_file.exists():
                    completed_count += 1

            completion_rate = completed_count / len(aggregator.expected_workflows) * 100
            self.logger.info(f"📊 Workflow completion: {completed_count}/{len(aggregator.expected_workflows)} ({completion_rate:.1f}%)")

            # If all workflows completed, break
            if completed_count == len(aggregator.expected_workflows):
                self.logger.info("✅ All workflows completed successfully!")
                break

            # Wait before next check
            await asyncio.sleep(check_interval)

        self.logger.info(f"🏁 Workflow monitoring completed after {elapsed:.1f}s")
    
    async def _save_final_results(self, final_results: Dict[str, Any]):
        """Save final consolidated results"""
        # Create output directory
        output_dir = Path("data/alphavirus_analysis")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save consolidated results
        results_file = output_dir / f"consolidated_results_{final_results['divergence_id']}.json"
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        # Save timing report
        timing_file = output_dir / f"timing_report_{final_results['divergence_id']}.json"
        with open(timing_file, 'w') as f:
            json.dump(final_results['consolidated_timing'], f, indent=2)
        
        # Save summary report
        summary_file = output_dir / f"summary_{final_results['divergence_id']}.json"
        summary = {
            'analysis_completed_at': datetime.now().isoformat(),
            'divergence_id': final_results['divergence_id'],
            'total_species_analyzed': final_results['total_workflows_expected'],
            'successful_analyses': final_results['successful_workflows'],
            'failed_analyses': final_results['failed_workflows'],
            'success_rate': final_results['successful_workflows'] / final_results['total_workflows_expected'] if final_results['total_workflows_expected'] > 0 else 0,
            'results_file': str(results_file),
            'timing_file': str(timing_file)
        }
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"💾 Final results saved:")
        self.logger.info(f"   - Results: {results_file}")
        self.logger.info(f"   - Timing: {timing_file}")
        self.logger.info(f"   - Summary: {summary_file}")
