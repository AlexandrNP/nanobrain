"""
Simplified Alphavirus Workflow

Refactored workflow that delegates business logic to individual steps.
This replaces the complex AlphavirusWorkflow class.
"""

import asyncio
import time
from typing import Dict, Any, List, Optional
from pathlib import Path

from nanobrain.core.workflow import Workflow, WorkflowConfig
from nanobrain.core.logging_system import get_logger
from nanobrain.core.step import StepConfig

# Import the new result management steps
from .steps import (
    BVBRCDataAcquisitionStep,
    AnnotationMappingStep, 
    SequenceCurationStep,
    ClusteringStep,
    AlignmentStep,
    PSSMAnalysisStep,
    DataAggregationStep,
    ResultCollectionStep,
    ViralPSSMGenerationStep
)


class AlphavirusWorkflow(Workflow):
    """
    Viral Protein Analysis Workflow
    
    Comprehensive viral protein analysis workflow supporting any viral species.
    Originally designed for Alphavirus analysis but supports all viral types.
    
    This workflow orchestrates steps and delegates all business logic
    to individual step implementations for clean separation of concerns.
    """
    
    def _init_from_config(self, config: WorkflowConfig, component_config: Dict[str, Any],
                         dependencies: Dict[str, Any]) -> None:
        """Initialize SimplifiedAlphavirusWorkflow with resolved dependencies"""
        super()._init_from_config(config, component_config, dependencies)
        
        # Generic virus support attributes
        config_dict = getattr(config, 'config', {})
        self.virus_name = config_dict.get('virus_name', 'Viral species')
        self.pipeline_config = config_dict.get('pipeline_config', {
            'enable_parallel_processing': True,
            'timeout_per_step': 300,
            'max_retries': 3
        })
        self.error_handling_config = config_dict.get('error_handling', {
            'continue_on_failure': True,
            'detailed_error_reporting': True
        })
        
        # Step execution results
        self.step_results = {}
        
        # Step configurations
        self.main_steps_config = getattr(config, 'main_steps_config', {
            'data_acquisition': {'enabled': True},
            'clustering': {'enabled': True},
            'pssm_analysis': {'enabled': True}
        })
        self.result_steps_config = getattr(config, 'result_steps_config', {
            'data_aggregation': {'enabled': True},
            'result_collection': {'enabled': True},
            'viral_pssm_generation': {'enabled': True}
        })
        
        self.workflow_logger.info(f"🧬 AlphavirusWorkflow initialized for viral protein analysis")
    
    async def execute(self, input_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute the simplified Alphavirus workflow
        
        Args:
            input_data: Optional input data (uses default if None)
            
        Returns:
            Dict containing workflow results
        """
        
        workflow_start_time = time.time()
        
        try:
            self.workflow_logger.info("🚀 Starting viral protein analysis workflow execution")
            
            # Initialize input data if not provided
            if input_data is None:
                input_data = await self._create_default_input()
            
            # Execute the main analysis pipeline
            pipeline_results = await self._execute_analysis_pipeline(input_data)
            
            # Execute result management pipeline
            final_results = await self._execute_result_management_pipeline(pipeline_results)
            
            # Calculate overall workflow metrics
            workflow_execution_time = time.time() - workflow_start_time
            
            self.workflow_logger.info(f"✅ Viral protein analysis workflow completed in {workflow_execution_time:.2f} seconds")
            
            # Build execution summary
            successful_steps = sum(1 for result in self.step_results.values() 
                                  if result.get('success', False))
            total_steps = len(self.step_results)
            
            return {
                'success': True,
                'workflow_name': 'AlphavirusWorkflow',
                'execution_time': workflow_execution_time,
                'step_results': self.step_results,
                'main_pipeline_results': {k: v for k, v in self.step_results.items() 
                                        if k in ['data_acquisition', 'clustering', 'pssm_analysis']},
                'result_pipeline_results': {k: v for k, v in self.step_results.items() 
                                          if k in ['data_aggregation', 'result_collection', 'viral_pssm_generation']},
                'final_results': final_results,
                'execution_summary': {
                    'total_steps_executed': total_steps,
                    'successful_steps': successful_steps,
                    'failed_steps': total_steps - successful_steps,
                    'total_execution_time': workflow_execution_time,
                    'framework_version': '2.0.0'
                },
                'workflow_metadata': {
                    'total_steps_executed': total_steps,
                    'successful_steps': successful_steps,
                    'framework_version': '2.0.0'
                }
            }
            
        except Exception as e:
            workflow_execution_time = time.time() - workflow_start_time
            self.workflow_logger.error(f"❌ Viral protein analysis workflow failed: {e}")
            
            return {
                'success': False,
                'error': str(e),
                'execution_time': workflow_execution_time,
                'step_results': self.step_results
            }
    
    async def _create_default_input(self) -> Dict[str, Any]:
        """Create default input data for the workflow"""
        
        return {
            'virus_name': 'Viral species',
            'analysis_parameters': {
                'min_genome_length': 8000,
                'max_genome_length': 15000,
                'clustering_threshold': 0.8
            }
        }
    
    async def _execute_analysis_pipeline(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the main analysis pipeline steps"""
        
        self.workflow_logger.info("🔄 Executing main analysis pipeline")
        
        current_input = input_data
        
        # Step 1: Data Acquisition
        acquisition_result = await self._execute_step(
            'data_acquisition',
            BVBRCDataAcquisitionStep,
            current_input,
            self._get_step_config('data_acquisition')
        )
        current_input = acquisition_result
        
        # Step 2: Clustering
        clustering_result = await self._execute_step(
            'clustering',
            ClusteringStep,
            current_input,
            self._get_step_config('clustering')
        )
        
        # Step 3: PSSM Analysis
        pssm_result = await self._execute_step(
            'pssm_analysis',
            PSSMAnalysisStep,
            clustering_result,
            self._get_step_config('pssm_analysis')
        )
        
        self.workflow_logger.info("✅ Main analysis pipeline completed")
        
        return self.step_results
    
    async def _execute_result_management_pipeline(self, pipeline_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the result management pipeline"""
        
        self.workflow_logger.info("🔄 Executing result management pipeline")
        
        # Data Aggregation Step
        aggregation_result = await self._execute_step(
            'data_aggregation',
            DataAggregationStep,
            pipeline_results,
            self._get_step_config('data_aggregation')
        )
        
        # Result Collection Step
        collection_result = await self._execute_step(
            'result_collection',
            ResultCollectionStep,
            self.step_results,
            self._get_step_config('result_collection')
        )
        
        # Viral PSSM Generation Step
        pssm_generation_result = await self._execute_step(
            'viral_pssm_generation',
            ViralPSSMGenerationStep,
            self.step_results,
            self._get_step_config('viral_pssm_generation')
        )
        
        self.workflow_logger.info("✅ Result management pipeline completed")
        
        # Return the final comprehensive result
        return {
            'aggregated_data': aggregation_result.get('aggregated_data'),
            'output_files': collection_result.get('output_files'),
            'viral_pssm_json': pssm_generation_result.get('viral_pssm_json'),
            'pipeline_summary': {
                'total_execution_time': sum(
                    result.get('execution_time', 0) 
                    for result in self.step_results.values()
                ),
                'steps_completed': len(self.step_results),
                'success_rate': sum(1 for result in self.step_results.values() 
                                  if result.get('success', False)) / len(self.step_results)
            }
        }
    
    async def _execute_step(self, step_name: str, step_class: type, 
                           input_data: Dict[str, Any], 
                           step_config: StepConfig) -> Dict[str, Any]:
        """Execute a single step and store its result"""
        
        self.workflow_logger.info(f"🔄 Executing step: {step_name}")
        
        try:
            # Create step instance using mandatory from_config pattern
            step_instance = step_class.from_config(step_config, executor=self.executor)
            step_result = await step_instance.process(input_data)
            
            # Store result
            self.step_results[step_name] = step_result
            
            if step_result.get('success'):
                self.workflow_logger.info(f"✅ Step {step_name} completed successfully")
            else:
                self.workflow_logger.warning(f"⚠️ Step {step_name} completed with issues")
            
            return step_result
            
        except Exception as e:
            self.workflow_logger.error(f"❌ Step {step_name} failed: {e}")
            
            error_result = {
                'success': False,
                'error': str(e),
                'step_name': step_name,
                'execution_time': 0
            }
            
            self.step_results[step_name] = error_result
            return error_result
    
    def _get_step_config(self, step_name: str) -> StepConfig:
        """Get configuration for a specific step"""
        
        # Default step configuration
        default_config = {
            'name': step_name,
            'type': 'analysis',
            'config': {}
        }
        
        # Get step-specific configuration from workflow config
        if hasattr(self.workflow_config, 'steps') and self.workflow_config.steps:
            step_configs = self.workflow_config.steps
            if step_name in step_configs:
                step_specific_config = step_configs[step_name]
                default_config['config'].update(step_specific_config)
        
        # Add result management step configurations
        if step_name == 'data_aggregation':
            default_config['config'].update({
                'data_mappings': self._get_data_aggregation_mappings(),
                'output_format': {'include_metadata': True}
            })
        elif step_name == 'result_collection':
            default_config['config'].update({
                'output_config': {
                    'base_directory': 'data/viral_analysis',
                    'timestamp_subdirectory': True,
                    'generate_manifest': True
                },
                'file_types': self._get_file_type_configurations()
            })
        elif step_name == 'viral_pssm_generation':
            default_config['config'].update({
                'metadata': {
                    'workflow_name': 'AlphavirusWorkflow',
                    'version': '2.0.0',
                    'organism': 'Viral species',
                    'method': 'nanobrain_viral_protein_analysis'
                },
                'output_format': {'validate_structure': True}
            })
        
        return StepConfig(**default_config)
    
    def _get_data_aggregation_mappings(self) -> Dict[str, Any]:
        """Get data mapping configuration for the aggregation step"""
        
        return {
            'data_acquisition': {
                'source_fields': ['filtered_genomes', 'unique_proteins', 'protein_sequences'],
                'target_section': 'genome_data',
                'field_mappings': {}
            },
            'clustering': {
                'source_fields': ['clusters', 'clustering_analysis'],
                'target_section': 'analysis_data',
                'field_mappings': {}
            },
            'pssm_analysis': {
                'source_fields': ['pssm_matrices', 'viral_pssm_json'],
                'target_section': 'analysis_data',
                'field_mappings': {}
            }
        }
    
    def _get_file_type_configurations(self) -> Dict[str, Any]:
        """Get file type configuration for the result collection step"""
        
        return {
            'filtered_genomes': {
                'filename': 'viral_filtered_genomes.json',
                'format': 'json'
            },
            'unique_proteins': {
                'filename': 'viral_unique_proteins.fasta',
                'format': 'fasta'
            },
            'clusters': {
                'filename': 'viral_clusters.json',
                'format': 'json'
            },
            'pssm_matrices': {
                'filename': 'viral_pssm_matrices.json',
                'format': 'json'
            },
            'viral_pssm_json': {
                'filename': 'viral_pssm.json',
                'format': 'json'
            }
        }
    
    # Legacy method aliases for test compatibility
    async def _execute_main_pipeline(self, input_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Legacy method alias for test compatibility"""
        if input_data is None:
            input_data = await self._create_default_input()
        return await self._execute_analysis_pipeline(input_data)
    
    async def _execute_result_pipeline(self, pipeline_results: Dict[str, Any]) -> Dict[str, Any]:
        """Legacy method alias for test compatibility"""
        return await self._execute_result_management_pipeline(pipeline_results)
    
    def _get_enabled_steps(self, steps_config: Dict[str, Any]) -> List[str]:
        """Get list of enabled steps from configuration"""
        enabled_steps = []
        for step_name, step_config in steps_config.items():
            if step_config.get('enabled', True):
                enabled_steps.append(step_name)
        return enabled_steps
    
    def get_workflow_config(self) -> Dict[str, Any]:
        """Get workflow configuration for validation"""
        return {
            'virus_name': self.virus_name,
            'pipeline_config': self.pipeline_config,
            'error_handling_config': self.error_handling_config,
            'main_steps_config': self.main_steps_config,
            'result_steps_config': self.result_steps_config
        }
    
    async def process(self, input_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process method for Step interface compatibility"""
        result = await self.execute(input_data)
        # Ensure result always has success field
        if result is None:
            return {'success': True, 'message': 'Process completed'}
        return result 