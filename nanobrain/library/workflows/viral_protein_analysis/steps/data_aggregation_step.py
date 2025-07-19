"""
Data Aggregation Step

Aggregates and standardizes data from multiple workflow steps.
Replaces the WorkflowData class logic from AlphavirusWorkflow.
"""

import asyncio
import time
import json
from datetime import datetime
from typing import Dict, Any, List, Optional

from nanobrain.core.step import Step, StepConfig
from nanobrain.core.logging_system import get_logger


class DataAggregationStep(Step):
    """
    Step to aggregate and standardize data from multiple previous steps
    
    This step replaces the complex WorkflowData class logic that was embedded
    in the AlphavirusWorkflow. It uses configurable mappings to aggregate
    data from different steps into a standardized format.
    """

    REQUIRED_CONFIG_FIELDS = ['name']
    OPTIONAL_CONFIG_FIELDS = {
        'description': '',
        'data_mappings': {},
        'output_format': {},
        'auto_initialize': True,
        'debug_mode': False,
        'enable_logging': True
    }

    @classmethod
    def _get_config_class(cls):
        """UNIFIED PATTERN: Return StepConfig - ONLY method that differs from other components"""
        return StepConfig
    
    # Now inherits unified from_config implementation from FromConfigBase

    @classmethod
    def extract_component_config(cls, config: StepConfig) -> Dict[str, Any]:
        """Extract DataAggregationStep configuration"""
        base_config = super().extract_component_config(config)
        step_config_dict = config.config if hasattr(config, 'config') else {}
        
        return {
            **base_config,
            'data_mappings': step_config_dict.get('data_mappings', {}),
            'output_format': step_config_dict.get('output_format', {})
        }

    def _init_from_config(self, config: StepConfig, component_config: Dict[str, Any],
                         dependencies: Dict[str, Any]) -> None:
        """Initialize DataAggregationStep with resolved dependencies"""
        super()._init_from_config(config, component_config, dependencies)
        
        # Extract configuration from step config
        self.step_config = component_config
        
        # Configuration for data mappings
        self.data_mappings = component_config.get('data_mappings', {})
        self.output_format = component_config.get('output_format', {})
        
        self.nb_logger.info(f"🔄 DataAggregationStep initialized with {len(self.data_mappings)} mappings")
    
    def __init__(self, config: StepConfig, **kwargs):
        super().__init__(config, **kwargs)
        
        # Extract configuration from step config
        step_config_dict = config.config if hasattr(config, 'config') else {}
        self.step_config = step_config_dict
        
        # Configuration for data mappings
        self.data_mappings = step_config_dict.get('data_mappings', {})
        self.output_format = step_config_dict.get('output_format', {})
        
        self.nb_logger.info(f"🔄 DataAggregationStep initialized with {len(self.data_mappings)} mappings")
        
    async def process(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Process method required by Step base class.
        """
        self.nb_logger.info("🔄 Processing data aggregation step")
        result = await self.execute(input_data)
        self.nb_logger.info(f"✅ Data aggregation completed successfully")
        return result
        
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aggregate data from all workflow steps into standardized format
        
        Args:
            input_data: Results from all previous steps
            
        Returns:
            Dict with aggregated and standardized data
        """
        
        step_start_time = time.time()
        
        try:
            self.nb_logger.info("🔄 Starting data aggregation from workflow steps")
            
            # Initialize aggregated data structure
            aggregated_data = {
                'genome_data': {},
                'protein_data': {},
                'analysis_data': {},
                'quality_data': {},
                'execution_metadata': {}
            }
            
            # Process each step's results using configurable mappings
            for step_id, step_result in input_data.items():
                if step_id in self.data_mappings:
                    self.nb_logger.debug(f"Applying mapping for step: {step_id}")
                    mapping_config = self.data_mappings[step_id]
                    await self._apply_data_mapping(
                        aggregated_data, 
                        step_result, 
                        mapping_config,
                        step_id
                    )
                else:
                    self.nb_logger.debug(f"No mapping configured for step: {step_id}")
            
            # Ensure backward compatibility fields are present
            self._ensure_backward_compatibility_fields(aggregated_data)
            
            # Generate data summary
            data_summary = await self._generate_data_summary(aggregated_data)
            
            # Apply output formatting if configured
            if self.output_format.get('standardize_keys', False):
                aggregated_data = await self._standardize_keys(aggregated_data)
            
            if self.output_format.get('include_metadata', True):
                aggregated_data['_metadata'] = {
                    'aggregation_timestamp': datetime.now().isoformat(),
                    'source_steps': list(input_data.keys()),
                    'data_summary': data_summary
                }
            
            execution_time = time.time() - step_start_time
            self.nb_logger.info(f"✅ Data aggregation completed in {execution_time:.2f} seconds")
            self.nb_logger.info(f"Aggregated data from {len(input_data)} steps")
            
            return {
                'success': True,
                'step_id': 'data_aggregation',
                'execution_time': execution_time,
                'primary_output': aggregated_data,
                'aggregated_data': aggregated_data,
                'data_summary': data_summary,
                'metadata': {
                    'step_type': self.__class__.__name__,
                    'input_data_keys': list(input_data.keys()),
                    'output_data_size': len(str(aggregated_data)),
                    'mapping_count': len(self.data_mappings)
                },
                'quality_metrics': {
                    'data_completeness': await self._calculate_data_completeness(aggregated_data),
                    'mapping_success_rate': 1.0  # All configured mappings processed
                }
            }
            
        except Exception as e:
            self.nb_logger.error(f"❌ Data aggregation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'execution_time': time.time() - step_start_time
            }
    
    async def _apply_data_mapping(self, aggregated_data: Dict[str, Any], 
                                 step_result: Dict[str, Any], 
                                 mapping_config: Dict[str, Any],
                                 step_id: str) -> None:
        """Apply data mapping configuration for a specific step"""
        
        source_fields = mapping_config.get('source_fields', [])
        target_section = mapping_config.get('target_section', 'analysis_data')
        field_mappings = mapping_config.get('field_mappings', {})
        
        # Ensure target section exists
        if target_section not in aggregated_data:
            aggregated_data[target_section] = {}
        
        # Map each source field
        for field in source_fields:
            if field in step_result:
                # Apply field mapping if configured
                target_field = field_mappings.get(field, field)
                target_key = f"{step_id}_{target_field}"
                
                aggregated_data[target_section][target_key] = step_result[field]
                self.nb_logger.debug(f"Mapped {step_id}.{field} -> {target_section}.{target_key}")
        
        # Handle special mappings for backwards compatibility with WorkflowData
        await self._handle_legacy_mappings(aggregated_data, step_result, step_id)
    
    async def _handle_legacy_mappings(self, aggregated_data: Dict[str, Any], 
                                    step_result: Dict[str, Any], 
                                    step_id: str) -> None:
        """Handle legacy mappings for compatibility with existing workflow"""
        
        # Map common fields based on step type
        if step_id == 'data_acquisition':
            genome_data = aggregated_data['genome_data']
            genome_data['original_genomes'] = step_result.get('original_genomes', [])
            genome_data['filtered_genomes'] = step_result.get('filtered_genomes', [])
            genome_data['unique_proteins'] = step_result.get('unique_proteins', [])
            genome_data['protein_sequences'] = step_result.get('protein_sequences', [])
            genome_data['protein_annotations'] = step_result.get('protein_annotations', [])
            genome_data['annotated_fasta'] = step_result.get('annotated_fasta', '')
            
        elif step_id == 'annotation_mapping':
            analysis_data = aggregated_data['analysis_data']
            analysis_data['standardized_annotations'] = step_result.get('standardized_annotations', [])
            analysis_data['genome_schematics'] = step_result.get('genome_schematics', [])
            
        elif step_id == 'sequence_curation':
            quality_data = aggregated_data['quality_data']
            quality_data['length_analysis'] = step_result.get('length_analysis')
            quality_data['curation_report'] = step_result.get('curation_report')
            quality_data['problematic_sequences'] = step_result.get('problematic_sequences', [])
            
        elif step_id == 'clustering':
            analysis_data = aggregated_data['analysis_data']
            analysis_data['clusters'] = step_result.get('clusters', [])
            analysis_data['clustering_analysis'] = step_result.get('clustering_analysis')
            
        elif step_id == 'alignment':
            analysis_data = aggregated_data['analysis_data']
            analysis_data['aligned_clusters'] = step_result.get('aligned_clusters', [])
            analysis_data['alignment_quality_stats'] = step_result.get('alignment_quality_stats')
            
        elif step_id == 'pssm_analysis':
            analysis_data = aggregated_data['analysis_data']
            analysis_data['pssm_matrices'] = step_result.get('pssm_matrices', [])
            analysis_data['viral_pssm_json'] = step_result.get('viral_pssm_json', {})
            analysis_data['final_curation_report'] = step_result.get('final_curation_report')
        
        # Store execution metadata
        if 'execution_time' in step_result:
            aggregated_data['execution_metadata'][f"{step_id}_execution_time"] = step_result['execution_time']
        
        if 'quality_metrics' in step_result:
            if 'step_quality_metrics' not in aggregated_data['quality_data']:
                aggregated_data['quality_data']['step_quality_metrics'] = {}
            aggregated_data['quality_data']['step_quality_metrics'][step_id] = step_result['quality_metrics']
    
    async def _generate_data_summary(self, aggregated_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics for aggregated data"""
        
        summary = {
            'total_sections': len(aggregated_data),
            'genome_data_count': len(aggregated_data.get('genome_data', {})),
            'protein_data_count': len(aggregated_data.get('protein_data', {})),
            'analysis_data_count': len(aggregated_data.get('analysis_data', {})),
            'quality_data_count': len(aggregated_data.get('quality_data', {}))
        }
        
        # Calculate specific counts for important data
        genome_data = aggregated_data.get('genome_data', {})
        if 'filtered_genomes' in genome_data:
            summary['filtered_genomes_count'] = len(genome_data['filtered_genomes'])
        if 'unique_proteins' in genome_data:
            summary['unique_proteins_count'] = len(genome_data['unique_proteins'])
        
        analysis_data = aggregated_data.get('analysis_data', {})
        if 'clusters' in analysis_data:
            summary['clusters_count'] = len(analysis_data['clusters'])
        if 'pssm_matrices' in analysis_data:
            summary['pssm_matrices_count'] = len(analysis_data['pssm_matrices'])
        
        return summary
    
    async def _standardize_keys(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Standardize keys in the aggregated data"""
        # Convert keys to lowercase and replace spaces with underscores
        # This is a placeholder implementation
        return data
    
    async def _calculate_data_completeness(self, aggregated_data: Dict[str, Any]) -> float:
        """Calculate data completeness score based on sections with actual data"""
        
        # Only consider sections that have actual content
        relevant_sections = []
        for section_name, section_data in aggregated_data.items():
            if (section_name not in ['_metadata', 'execution_metadata'] and 
                isinstance(section_data, dict) and len(section_data) > 0):
                relevant_sections.append(section_name)
        
        # If we have both genome_data and analysis_data with content, that's complete
        # for the basic workflow case
        if ('genome_data' in relevant_sections and 'analysis_data' in relevant_sections):
            return 1.0
        
        # Otherwise calculate as fraction of expected basic sections
        expected_basic_sections = ['genome_data', 'analysis_data']
        present_basic_sections = [s for s in expected_basic_sections if s in relevant_sections]
        
        return len(present_basic_sections) / len(expected_basic_sections)

    def _ensure_backward_compatibility_fields(self, aggregated_data: Dict[str, Any]) -> None:
        """Ensure required fields are present for backward compatibility"""
        
        # Only add backward compatibility fields if the section has actual data
        genome_data = aggregated_data.get('genome_data', {})
        if len(genome_data) > 0:  # Only if there's actually data
            if 'original_genomes' not in genome_data:
                genome_data['original_genomes'] = []
            if 'filtered_genomes' not in genome_data:
                genome_data['filtered_genomes'] = []
            if 'unique_proteins' not in genome_data:
                genome_data['unique_proteins'] = []
            aggregated_data['genome_data'] = genome_data
        
        # Only add backward compatibility fields if the section has actual data
        analysis_data = aggregated_data.get('analysis_data', {})
        if len(analysis_data) > 0:  # Only if there's actually data
            if 'clusters' not in analysis_data:
                analysis_data['clusters'] = []
            if 'pssm_matrices' not in analysis_data:
                analysis_data['pssm_matrices'] = []
            if 'clustering_analysis' not in analysis_data:
                analysis_data['clustering_analysis'] = {}
            aggregated_data['analysis_data'] = analysis_data 