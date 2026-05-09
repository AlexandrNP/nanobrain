"""
Parameter Grid Generator for Distributed Hyperparameter Search

This module provides utilities for generating parameter grids and workflow configurations
for distributed hyperparameter search across parameter combinations.

Key Features:
- Generate all combinations from parameter ranges
- Create workflow configuration files for each combination
- Support for template-based YAML generation
- Metadata tracking for parameter combinations

Usage:
    generator = ParameterGridGenerator()

    # Generate parameter grid
    param_ranges = {
        'column_fraction': [0.5, 0.6, 0.7],
        'nterm_conservation': [0.7, 0.8],
        'cterm_conservation': [0.7, 0.8]
    }
    combinations = generator.generate_grid(param_ranges)

    # Generate workflow configs for each combination
    config_paths = generator.generate_configs(
        combinations=combinations,
        template_path='config/templates/pssm_generation_subworkflow.yml',
        output_dir='config/generated/parameter_grid'
    )
"""

import itertools
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import yaml
import hashlib
import json

from nanobrain.core.logging_system import get_logger

logger = get_logger(__name__)


@dataclass
class ParameterCombination:
    """
    Represents a single parameter combination for hyperparameter search.

    Attributes:
        parameters: Dictionary of parameter names to values
        combination_id: Unique identifier for this combination
        metadata: Additional metadata (e.g., performance metrics, results)
    """
    parameters: Dict[str, Any]
    combination_id: str = field(init=False)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Generate unique combination ID based on parameter values."""
        # Create deterministic hash from sorted parameters
        param_str = json.dumps(self.parameters, sort_keys=True)
        self.combination_id = hashlib.md5(param_str.encode()).hexdigest()[:12]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'parameters': self.parameters,
            'combination_id': self.combination_id,
            'metadata': self.metadata
        }

    def get_param(self, param_name: str, default: Any = None) -> Any:
        """Get parameter value by name."""
        return self.parameters.get(param_name, default)

    def update_metadata(self, **kwargs):
        """Update metadata with new key-value pairs."""
        self.metadata.update(kwargs)


class ParameterGridGenerator:
    """
    Generates parameter grids and workflow configurations for distributed execution.

    This class supports creating all possible parameter combinations from ranges
    and generating individual workflow configuration files for each combination.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize ParameterGridGenerator.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.logger = get_logger(f"{__name__}.ParameterGridGenerator")

    def generate_grid(
        self,
        parameter_ranges: Dict[str, List[Any]],
        constraints: Optional[Dict[str, Any]] = None
    ) -> List[ParameterCombination]:
        """
        Generate all parameter combinations from parameter ranges.

        Args:
            parameter_ranges: Dictionary mapping parameter names to lists of values
                Example: {'param1': [1, 2, 3], 'param2': ['a', 'b']}
            constraints: Optional constraints to filter combinations
                Example: {'max_total': lambda p: p['param1'] + p['param2'] <= 10}

        Returns:
            List of ParameterCombination objects

        Example:
            >>> generator = ParameterGridGenerator()
            >>> param_ranges = {
            ...     'column_fraction': [0.5, 0.6, 0.7],
            ...     'nterm_conservation': [0.7, 0.8],
            ...     'cterm_conservation': [0.7, 0.8]
            ... }
            >>> combinations = generator.generate_grid(param_ranges)
            >>> len(combinations)
            12
        """
        if not parameter_ranges:
            self.logger.warning("Empty parameter ranges provided")
            return []

        # Extract parameter names and value lists
        param_names = list(parameter_ranges.keys())
        param_values = [parameter_ranges[name] for name in param_names]

        # Generate all combinations using Cartesian product
        all_combinations = []
        for combo_values in itertools.product(*param_values):
            # Create parameter dictionary
            param_dict = dict(zip(param_names, combo_values))

            # Apply constraints if provided
            if constraints and not self._check_constraints(param_dict, constraints):
                continue

            # Create ParameterCombination object
            combination = ParameterCombination(parameters=param_dict)
            all_combinations.append(combination)

        self.logger.info(
            f"Generated {len(all_combinations)} parameter combinations from "
            f"{len(param_names)} parameters"
        )

        return all_combinations

    def _check_constraints(
        self,
        parameters: Dict[str, Any],
        constraints: Dict[str, Any]
    ) -> bool:
        """
        Check if parameter combination satisfies constraints.

        Args:
            parameters: Parameter dictionary to check
            constraints: Dictionary of constraint functions

        Returns:
            True if all constraints satisfied, False otherwise
        """
        for constraint_name, constraint_func in constraints.items():
            try:
                if not constraint_func(parameters):
                    self.logger.debug(
                        f"Parameter combination {parameters} failed constraint: {constraint_name}"
                    )
                    return False
            except Exception as e:
                self.logger.error(
                    f"Error evaluating constraint {constraint_name}: {e}"
                )
                return False
        return True

    def generate_configs(
        self,
        combinations: List[ParameterCombination],
        template_path: Union[str, Path],
        output_dir: Union[str, Path],
        additional_config: Optional[Dict[str, Any]] = None,
        config_name_pattern: str = "config_{combination_id}.yml"
    ) -> List[Path]:
        """
        Generate workflow configuration files for each parameter combination.

        Args:
            combinations: List of ParameterCombination objects
            template_path: Path to YAML template file
            output_dir: Directory to save generated configs
            additional_config: Additional configuration to merge into each config
            config_name_pattern: Pattern for config file names (supports {combination_id})

        Returns:
            List of paths to generated configuration files

        Example:
            >>> combinations = generator.generate_grid(param_ranges)
            >>> config_paths = generator.generate_configs(
            ...     combinations=combinations,
            ...     template_path='config/templates/pssm_subworkflow.yml',
            ...     output_dir='config/generated/grid'
            ... )
        """
        template_path = Path(template_path)
        output_dir = Path(output_dir)

        # Validate template exists
        if not template_path.exists():
            raise FileNotFoundError(f"Template file not found: {template_path}")

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load template
        with open(template_path, 'r') as f:
            template_config = yaml.safe_load(f)

        generated_paths = []

        for combination in combinations:
            try:
                # Create config for this combination
                config = self._generate_single_config(
                    template=template_config,
                    combination=combination,
                    additional_config=additional_config
                )

                # Generate output filename
                filename = config_name_pattern.format(
                    combination_id=combination.combination_id
                )
                output_path = output_dir / filename

                # Write config file
                with open(output_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

                generated_paths.append(output_path)

                self.logger.debug(
                    f"Generated config for combination {combination.combination_id}: {output_path}"
                )

            except Exception as e:
                self.logger.error(
                    f"Failed to generate config for combination {combination.combination_id}: {e}"
                )
                continue

        self.logger.info(
            f"Generated {len(generated_paths)} configuration files in {output_dir}"
        )

        return generated_paths

    def _generate_single_config(
        self,
        template: Dict[str, Any],
        combination: ParameterCombination,
        additional_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate a single configuration by substituting parameters into template.

        Args:
            template: Template configuration dictionary
            combination: ParameterCombination to substitute
            additional_config: Additional config to merge

        Returns:
            Generated configuration dictionary
        """
        # Deep copy template to avoid modifying original
        import copy
        config = copy.deepcopy(template)

        # Substitute parameters in template
        config = self._substitute_parameters(config, combination.parameters)

        # Add combination metadata
        if 'metadata' not in config:
            config['metadata'] = {}

        config['metadata']['combination_id'] = combination.combination_id
        config['metadata']['parameters'] = combination.parameters

        # Merge additional config if provided
        if additional_config:
            config = self._deep_merge(config, additional_config)

        return config

    def _substitute_parameters(
        self,
        obj: Any,
        parameters: Dict[str, Any]
    ) -> Any:
        """
        Recursively substitute parameter placeholders in configuration.

        Supports placeholder format: ${PARAMETER_NAME} or {{PARAMETER_NAME}}

        Args:
            obj: Object to process (dict, list, str, or other)
            parameters: Parameter values for substitution

        Returns:
            Object with substituted values
        """
        if isinstance(obj, dict):
            return {k: self._substitute_parameters(v, parameters) for k, v in obj.items()}

        elif isinstance(obj, list):
            return [self._substitute_parameters(item, parameters) for item in obj]

        elif isinstance(obj, str):
            # Replace ${PARAM} and {{PARAM}} placeholders
            result = obj
            for param_name, param_value in parameters.items():
                # Support both ${} and {{}} syntax
                result = result.replace(f"${{{param_name}}}", str(param_value))
                result = result.replace(f"{{{{{param_name}}}}}", str(param_value))
            return result

        else:
            return obj

    def _deep_merge(
        self,
        base: Dict[str, Any],
        override: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Deep merge two dictionaries, with override taking precedence.

        Args:
            base: Base dictionary
            override: Dictionary to merge in (takes precedence)

        Returns:
            Merged dictionary
        """
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def save_grid_metadata(
        self,
        combinations: List[ParameterCombination],
        output_path: Union[str, Path],
        include_results: bool = False
    ):
        """
        Save parameter grid metadata to a file for tracking and analysis.

        Args:
            combinations: List of ParameterCombination objects
            output_path: Path to save metadata file
            include_results: Whether to include result metadata
        """
        output_path = Path(output_path)

        metadata = {
            'total_combinations': len(combinations),
            'parameters': list(combinations[0].parameters.keys()) if combinations else [],
            'combinations': []
        }

        for combo in combinations:
            combo_data = {
                'combination_id': combo.combination_id,
                'parameters': combo.parameters
            }
            if include_results:
                combo_data['metadata'] = combo.metadata

            metadata['combinations'].append(combo_data)

        with open(output_path, 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False)

        self.logger.info(f"Saved grid metadata to {output_path}")

    @staticmethod
    def load_grid_metadata(metadata_path: Union[str, Path]) -> List[ParameterCombination]:
        """
        Load parameter grid from metadata file.

        Args:
            metadata_path: Path to metadata file

        Returns:
            List of ParameterCombination objects
        """
        metadata_path = Path(metadata_path)

        with open(metadata_path, 'r') as f:
            metadata = yaml.safe_load(f)

        combinations = []
        for combo_data in metadata['combinations']:
            combo = ParameterCombination(parameters=combo_data['parameters'])
            if 'metadata' in combo_data:
                combo.metadata = combo_data['metadata']
            combinations.append(combo)

        return combinations

    # ============================================================================
    # PBS/Aurora-Specific Methods for HPC Execution
    # ============================================================================

    def estimate_pbs_resources(
        self,
        num_combinations: int,
        estimated_runtime_per_combo: int = 300,  # 5 minutes default
        max_workers_per_node: int = 4,
        max_nodes: int = 10
    ) -> Dict[str, Any]:
        """
        Estimate PBS resources needed for parameter grid execution on Aurora.

        BRUTAL TRUTH: This is critical for avoiding PBS queue abuse. You can't
        just submit 50 individual PBS jobs for 50 parameter combinations.

        Args:
            num_combinations: Number of parameter combinations to execute
            estimated_runtime_per_combo: Estimated seconds per combination
            max_workers_per_node: Maximum workers per PBS node
            max_nodes: Maximum nodes to request

        Returns:
            Dictionary with PBS resource estimates:
                - nodes_per_block: Recommended nodes per Parsl block
                - blocks: Number of PBS jobs to submit
                - workers_per_node: Workers per node
                - estimated_walltime: Estimated walltime string (HH:MM:SS)
                - total_workers: Total parallel workers
                - parallelism: Parallelism factor
        """
        # Calculate optimal worker distribution
        total_workers_needed = min(num_combinations, max_nodes * max_workers_per_node)

        # Determine nodes and workers
        if total_workers_needed <= max_workers_per_node:
            # Small grid: single node
            nodes_per_block = 1
            workers_per_node = total_workers_needed
        else:
            # Larger grid: distribute across nodes
            workers_per_node = max_workers_per_node
            nodes_per_block = min(
                math.ceil(total_workers_needed / max_workers_per_node),
                max_nodes
            )

        total_workers = nodes_per_block * workers_per_node

        # Estimate walltime
        # With parallel execution, time = (num_combinations / total_workers) * runtime_per_combo
        parallel_time = math.ceil(num_combinations / total_workers) * estimated_runtime_per_combo

        # Add 20% buffer for overhead
        walltime_seconds = int(parallel_time * 1.2)

        # Convert to HH:MM:SS
        hours = walltime_seconds // 3600
        minutes = (walltime_seconds % 3600) // 60
        seconds = walltime_seconds % 60
        walltime_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        # Determine number of PBS blocks
        # For Aurora, typically 1 block is sufficient as it scales workers within the block
        blocks = 1

        return {
            'nodes_per_block': nodes_per_block,
            'blocks': blocks,
            'workers_per_node': workers_per_node,
            'estimated_walltime': walltime_str,
            'total_workers': total_workers,
            'parallelism': total_workers / num_combinations if num_combinations > 0 else 0,
            'max_concurrent_combinations': total_workers,
            'estimated_total_runtime_seconds': walltime_seconds
        }

    def generate_aurora_parsl_config(
        self,
        num_combinations: int,
        output_path: Union[str, Path],
        account: str = "FoundEpidem",
        queue: str = "workq",
        estimated_runtime_per_combo: int = 300,
        conda_env: str = "nanobrain-upd",
        base_config_path: Optional[Union[str, Path]] = None
    ) -> Path:
        """
        Generate Aurora-specific Parsl executor configuration for parameter grid execution.

        BRUTAL TRUTH: This is where we actually connect to Aurora PBS. Without this,
        the parameter grid is just floating in the void.

        Args:
            num_combinations: Number of parameter combinations
            output_path: Path to save generated config
            account: PBS account (default: FoundEpidem)
            queue: PBS queue (default: workq, use 'debug' for testing)
            estimated_runtime_per_combo: Estimated seconds per combination
            conda_env: Conda environment name
            base_config_path: Optional base config to extend

        Returns:
            Path to generated Parsl config file
        """
        output_path = Path(output_path)

        # Estimate PBS resources
        resources = self.estimate_pbs_resources(
            num_combinations=num_combinations,
            estimated_runtime_per_combo=estimated_runtime_per_combo
        )

        self.logger.info(
            f"Generating Aurora Parsl config for {num_combinations} combinations:\n"
            f"  - Nodes per block: {resources['nodes_per_block']}\n"
            f"  - Workers per node: {resources['workers_per_node']}\n"
            f"  - Total workers: {resources['total_workers']}\n"
            f"  - Estimated walltime: {resources['estimated_walltime']}\n"
            f"  - Parallelism: {resources['parallelism']:.2f}"
        )

        # Load base config if provided
        if base_config_path:
            base_config_path = Path(base_config_path)
            if base_config_path.exists():
                with open(base_config_path, 'r') as f:
                    config = yaml.safe_load(f)
            else:
                self.logger.warning(f"Base config not found: {base_config_path}, using default")
                config = self._get_default_aurora_config()
        else:
            config = self._get_default_aurora_config()

        # Update with grid-specific resources
        executor_config = config['parsl_config']['executors'][0]

        executor_config['max_workers_per_node'] = resources['workers_per_node']
        executor_config['cores_per_worker'] = 1

        provider_config = executor_config['provider_config']
        provider_config['queue'] = queue
        provider_config['account'] = account
        provider_config['nodes_per_block'] = resources['nodes_per_block']
        provider_config['cpus_per_node'] = resources['workers_per_node'] * 2  # Hyperthreading
        provider_config['walltime'] = resources['estimated_walltime']

        # Update worker init with correct conda env
        provider_config['worker_init'] = f"""module load frameworks
source /home/onarykov/miniconda3/etc/profile.d/conda.sh
conda activate {conda_env}
export PYTHONPATH="/home/onarykov/nanobrain:$PYTHONPATH"
echo "🔥 Starting parameter grid worker on Aurora node $(hostname)"
echo "📋 PBS_JOBID: $PBS_JOBID"
echo "📊 Processing parameter combinations with {resources['total_workers']} workers"
"""

        # Update executor name and description
        config['name'] = f"aurora_parsl_executor_grid_{num_combinations}"
        config['description'] = (
            f"Aurora Parsl executor for {num_combinations} parameter combinations "
            f"({resources['total_workers']} workers, {resources['estimated_walltime']} walltime)"
        )
        config['max_workers'] = resources['total_workers']

        # Add metadata
        config['metadata'] = {
            'grid_size': num_combinations,
            'pbs_resources': resources,
            'generated_at': str(Path(__file__).resolve()),
            'purpose': 'parameter_grid_search'
        }

        # Write config
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        self.logger.info(f"Generated Aurora Parsl config: {output_path}")
        return output_path

    def _get_default_aurora_config(self) -> Dict[str, Any]:
        """
        Get default Aurora Parsl configuration template.

        Returns:
            Default configuration dictionary
        """
        return {
            'name': 'aurora_parsl_executor',
            'description': 'Parsl executor for Aurora PBS system',
            'executor_type': 'parsl',
            'max_workers': 8,
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
                    'max_workers_per_node': 4,
                    'cores_per_worker': 1,
                    'worker_debug': True,
                    'heartbeat_period': 10,
                    'heartbeat_threshold': 30,
                    'provider_config': {
                        'class': 'parsl.providers.PBSProProvider',
                        'queue': 'workq',
                        'account': 'FoundEpidem',
                        'nodes_per_block': 1,
                        'cpus_per_node': 4,
                        'walltime': '00:30:00',
                        'scheduler_options': '#PBS -l filesystems=home:flare',
                        'worker_init': '',
                        'parallelism': 1.0
                    }
                }]
            },
            'fallback': {
                'enable_fallback': False,
                'executor_type': 'local',
                'max_workers': 4
            }
        }

    def create_batched_execution_plan(
        self,
        combinations: List[ParameterCombination],
        max_concurrent: int = 16
    ) -> List[List[ParameterCombination]]:
        """
        Create batched execution plan to avoid overwhelming PBS queue.

        BRUTAL TRUTH: Submitting 50 workflows at once will either:
        1. Overwhelm the PBS queue
        2. Hit job submission limits
        3. Waste resources on queued jobs

        Better to batch into groups that match available workers.

        Args:
            combinations: List of parameter combinations
            max_concurrent: Maximum concurrent executions (matches Parsl workers)

        Returns:
            List of batches, each containing parameter combinations
        """
        batches = []
        for i in range(0, len(combinations), max_concurrent):
            batch = combinations[i:i + max_concurrent]
            batches.append(batch)

        self.logger.info(
            f"Created {len(batches)} batches from {len(combinations)} combinations "
            f"(max {max_concurrent} concurrent)"
        )

        return batches
