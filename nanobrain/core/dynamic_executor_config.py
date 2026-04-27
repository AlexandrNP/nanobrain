#!/usr/bin/env python3
"""
Dynamic Executor Configuration Generator

CRITICAL PRODUCTION COMPONENT:
- Generates unique Parsl executor configurations per workflow
- Integrates with PBS Resource Manager for proper resource allocation
- Creates workflow-specific executor configs with correct maximum_workers
- Prevents resource oversubscription disasters
- Handles Aurora PBS-specific configuration

BRUTAL TRUTH: Each diverged workflow MUST have its own executor config!
Using the same config for 70 workflows = guaranteed PBS crash.
"""

import json
import os
import time
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

from nanobrain.core.pbs_resource_manager import ResourceAllocation


logger = logging.getLogger(__name__)


class DynamicExecutorConfigGenerator:
    """
    Generates workflow-specific Parsl executor configurations
    
    CRITICAL FUNCTIONS:
    1. Create unique executor config per workflow
    2. Set proper maximum_workers based on resource allocation
    3. Configure PBS provider with correct resource limits
    4. Handle Aurora-specific PBS configuration
    5. Save configs to workflow-specific directories
    """
    
    def __init__(self, base_config_path: Optional[str] = None):
        """
        Initialize the dynamic executor config generator
        
        Args:
            base_config_path: Path to base executor configuration template
        """
        self.base_config_path = base_config_path or "demos/viral_pssm_workflow/config/parsl_aurora_executor.yml"
        self.base_config = None
        self._load_base_config()
    
    def _load_base_config(self) -> None:
        """Load the base executor configuration template"""
        try:
            config_path = Path(self.base_config_path)
            if config_path.exists():
                with open(config_path, 'r') as f:
                    self.base_config = yaml.safe_load(f)
                logger.info(f"📋 Loaded base executor config: {config_path}")
            else:
                logger.warning(f"⚠️  Base config not found: {config_path}, using defaults")
                self._create_default_config()
        except Exception as e:
            logger.error(f"❌ Failed to load base config: {e}")
            self._create_default_config()
    
    def _create_default_config(self) -> None:
        """Create default Aurora PBS executor configuration"""
        self.base_config = {
            'name': 'aurora_parsl_executor_template',
            'executor_type': 'parsl',
            'max_workers': 1,  # Will be overridden
            'timeout': 600,
            'parsl_config': {
                'strategy': None,
                'app_cache': True,
                'checkpoint_mode': 'task_exit',
                'checkpoint_period': '00:05:00',
                'retries': 2,
                'executors': [{
                    'label': 'aurora_pbs_htex',
                    'class': 'parsl.executors.HighThroughputExecutor',
                    'max_workers_per_node': 1,  # Will be overridden
                    'cores_per_worker': 1,
                    'worker_debug': True,
                    'heartbeat_period': 10,
                    'heartbeat_threshold': 30,
                    'provider_config': {
                        'class': 'parsl.providers.PBSProProvider',
                        'queue': 'debug',
                        'account': 'FoundEpidem',
                        'nodes_per_block': 1,
                        'cpus_per_node': 8,  # Will be overridden
                        'walltime': '00:30:00',
                        'scheduler_options': '#PBS -l filesystems=home:flare',
                        'worker_init': self._get_default_worker_init(),
                        'parallelism': 1.0
                    }
                }],
                'fallback': {
                    'enable_fallback': False,
                    'executor_type': 'local',
                    'max_workers': 1
                }
            }
        }
    
    def _get_default_worker_init(self) -> str:
        """Get default worker initialization script for Aurora"""
        return """module load frameworks
source /home/onarykov/miniconda3/etc/profile.d/conda.sh
conda activate nanobrain-upd
export PYTHONPATH="/home/onarykov/nanobrain:$PYTHONPATH"
echo "🔥 Starting workflow-specific worker on Aurora node $(hostname)"
echo "📋 PBS_JOBID: $PBS_JOBID"
echo "🆔 Workflow UUID: $NANOBRAIN_WORKFLOW_UUID"
"""
    
    def generate_workflow_executor_config(self, 
                                        workflow_uuid: str,
                                        allocation: ResourceAllocation,
                                        workflow_dir: Path) -> str:
        """
        Generate workflow-specific executor configuration
        
        Args:
            workflow_uuid: Unique workflow identifier
            allocation: Resource allocation from PBS Resource Manager
            workflow_dir: Directory to save the config file
            
        Returns:
            Path to the generated executor configuration file
        """
        if not self.base_config:
            raise RuntimeError("Base configuration not loaded")
        
        # Create workflow-specific config
        config = self._deep_copy_config(self.base_config)
        
        # Update configuration with resource allocation
        self._update_config_with_allocation(config, workflow_uuid, allocation)
        
        # Save configuration to workflow directory
        config_filename = f"executor_config_{workflow_uuid}.yml"
        config_path = workflow_dir / config_filename
        
        # Ensure directory exists
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        logger.info(f"✅ Generated executor config for workflow {workflow_uuid[:8]}...")
        logger.info(f"   📁 Config file: {config_path}")
        logger.info(f"   🔧 Max workers: {allocation.max_workers}")
        logger.info(f"   🖥️  Node: {allocation.node_id}")
        logger.info(f"   ⚙️  Cores: {allocation.allocated_cores}")
        
        return str(config_path)
    
    def _deep_copy_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Deep copy configuration dictionary"""
        return json.loads(json.dumps(config))
    
    def _update_config_with_allocation(self, 
                                     config: Dict[str, Any], 
                                     workflow_uuid: str,
                                     allocation: ResourceAllocation) -> None:
        """Update configuration with resource allocation details"""
        
        # Update top-level configuration
        config['name'] = f"workflow_executor_{workflow_uuid[:8]}"
        config['max_workers'] = allocation.max_workers
        
        # Update executor configuration
        executor_config = config['parsl_config']['executors'][0]
        executor_config['max_workers_per_node'] = allocation.allocated_cores
        
        # Update provider configuration
        provider_config = executor_config['provider_config']
        provider_config['cpus_per_node'] = allocation.allocated_cores
        
        # Add workflow-specific worker initialization
        worker_init = provider_config.get('worker_init', '')
        worker_init += f'\nexport NANOBRAIN_WORKFLOW_UUID="{workflow_uuid}"'
        worker_init += f'\necho "🆔 Workflow UUID: {workflow_uuid}"'
        worker_init += f'\necho "🖥️  Allocated to node: {allocation.node_id}"'
        worker_init += f'\necho "⚙️  Using {allocation.allocated_cores} cores"'
        provider_config['worker_init'] = worker_init
        
        # Update metadata
        if 'metadata' not in config:
            config['metadata'] = {}
        
        config['metadata'].update({
            'workflow_uuid': workflow_uuid,
            'node_id': allocation.node_id,
            'allocated_cores': allocation.allocated_cores,
            'max_workers': allocation.max_workers,
            'generated_at': time.time(),
            'resource_allocation': {
                'workflow_uuid': allocation.workflow_uuid,
                'node_id': allocation.node_id,
                'allocated_cores': allocation.allocated_cores,
                'max_workers': allocation.max_workers,
                'allocation_time': allocation.allocation_time
            }
        })
    
    def generate_fallback_config(self, workflow_uuid: str, workflow_dir: Path) -> str:
        """
        Generate fallback local executor configuration
        
        Args:
            workflow_uuid: Unique workflow identifier
            workflow_dir: Directory to save the config file
            
        Returns:
            Path to the generated fallback configuration file
        """
        config = {
            'name': f"fallback_executor_{workflow_uuid[:8]}",
            'executor_type': 'local',
            'max_workers': 1,
            'timeout': 300,
            'metadata': {
                'workflow_uuid': workflow_uuid,
                'fallback_mode': True,
                'generated_at': time.time()
            }
        }
        
        config_filename = f"fallback_executor_config_{workflow_uuid}.yml"
        config_path = workflow_dir / config_filename
        
        # Ensure directory exists
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        logger.warning(f"⚠️  Generated fallback executor config for workflow {workflow_uuid[:8]}...")
        logger.warning(f"   📁 Config file: {config_path}")
        
        return str(config_path)
