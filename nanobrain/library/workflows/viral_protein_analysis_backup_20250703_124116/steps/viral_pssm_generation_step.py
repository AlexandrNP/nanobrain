"""
Viral PSSM Generation Step

Generates the final viral_pssm.json output.
Replaces the _generate_viral_pssm_json method from AlphavirusWorkflow.
"""

import asyncio
import time
import json
from datetime import datetime
from typing import Dict, Any, List, Optional

from nanobrain.core.step import Step, StepConfig
from nanobrain.core.logging_system import get_logger


class ViralPSSMGenerationStep(Step):
    """
    Step to generate final viral_pssm.json output
    
    This step replaces the _generate_viral_pssm_json method that was embedded
    in the AlphavirusWorkflow. It generates the standardized viral_pssm.json
    format from aggregated workflow results.
    """

    REQUIRED_CONFIG_FIELDS = ['name']
    OPTIONAL_CONFIG_FIELDS = {
        'description': '',
        'metadata': {},
        'output_format': {},
        'auto_initialize': True,
        'debug_mode': False,
        'enable_logging': True
    }

    @classmethod
    def from_config(cls, config: StepConfig, **kwargs) -> 'ViralPSSMGenerationStep':
        """Create ViralPSSMGenerationStep from configuration using mandatory pattern"""
        # Use parent class from_config implementation
        return super().from_config(config, **kwargs)

    @classmethod
    def extract_component_config(cls, config: StepConfig) -> Dict[str, Any]:
        """Extract ViralPSSMGenerationStep configuration"""
        base_config = super().extract_component_config(config)
        step_config_dict = config.config if hasattr(config, 'config') else {}
        
        return {
            **base_config,
            'metadata': step_config_dict.get('metadata', {}),
            'output_format': step_config_dict.get('output_format', {}),
            'pssm_config': step_config_dict.get('pssm_config', {}),
            'output_config': step_config_dict.get('output_config', {}),
            'validation': step_config_dict.get('validation', {})
        }

    def _init_from_config(self, config: StepConfig, component_config: Dict[str, Any],
                         dependencies: Dict[str, Any]) -> None:
        """Initialize ViralPSSMGenerationStep with resolved dependencies"""
        super()._init_from_config(config, component_config, dependencies)
        
        # Extract configuration from step config
        self.step_config = component_config
        
        # Configuration for viral PSSM generation
        self.metadata_config = component_config.get('metadata', {})
        self.output_format = component_config.get('output_format', {})
        self.pssm_config = component_config.get('pssm_config', {})
        self.output_config = component_config.get('output_config', {})
        self.validation_config = component_config.get('validation', {})
        
        # Extract cluster_id_mapping from pssm_config (expected by tests)
        self.cluster_id_mapping = self.pssm_config.get('cluster_id_mapping', {})
        self.protein_functions = self.pssm_config.get('protein_functions', {})
        
        self.nb_logger.info(f"🧬 ViralPSSMGenerationStep initialized")
    
    def __init__(self, config: StepConfig, **kwargs):
        super().__init__(config, **kwargs)
        
        # Extract configuration from step config
        step_config_dict = config.config if hasattr(config, 'config') else {}
        self.step_config = step_config_dict
        
        # Configuration for viral PSSM generation
        self.metadata_config = step_config_dict.get('metadata', {})
        self.output_format = step_config_dict.get('output_format', {})
        
        self.nb_logger.info(f"🧬 ViralPSSMGenerationStep initialized")
        
    async def process(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Process method required by Step base class.
        """
        self.nb_logger.info("🔄 Processing viral PSSM generation step")
        result = await self.execute(input_data)
        self.nb_logger.info(f"✅ Viral PSSM generation completed successfully")
        return result
        
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate viral_pssm.json from all workflow results
        
        Args:
            input_data: Aggregated results from all previous steps
            
        Returns:
            Dict containing viral_pssm.json data
        """
        
        step_start_time = time.time()
        
        try:
            self.nb_logger.info("🧬 Starting viral PSSM JSON generation")
            
            # Extract aggregated data
            aggregated_data = self._extract_aggregated_data(input_data)
            
            # Generate viral PSSM structure
            viral_pssm_data = await self._create_viral_pssm_structure(aggregated_data, input_data)
            
            # Validate the generated structure
            validation_result = await self._validate_viral_pssm(viral_pssm_data)
            
            execution_time = time.time() - step_start_time
            self.nb_logger.info(f"✅ Viral PSSM generation completed in {execution_time:.2f} seconds")
            
            return {
                'success': True,
                'step_id': 'viral_pssm_generation',
                'execution_time': execution_time,
                'primary_output': viral_pssm_data,
                'viral_pssm_json': viral_pssm_data,
                'validation_result': validation_result,
                'generation_metadata': {
                    'step_type': self.__class__.__name__,
                    'input_data_keys': list(input_data.keys()),
                    'pssm_data_size': len(str(viral_pssm_data)),
                    'generated_at': datetime.now().isoformat()
                },
                'metadata': {
                    'step_type': self.__class__.__name__,
                    'input_data_keys': list(input_data.keys()),
                    'pssm_data_size': len(str(viral_pssm_data)),
                    'generated_at': datetime.now().isoformat()
                },
                'quality_metrics': {
                    'structure_completeness': validation_result.get('completeness_score', 0.0),
                    'data_quality_score': validation_result.get('quality_score', 0.0),
                    'pssm_generation_quality': validation_result.get('quality_score', 0.0),
                    'data_completeness': validation_result.get('completeness_score', 0.0),
                    'validation_results': self._generate_validation_results(aggregated_data)
                }
            }
            
        except Exception as e:
            self.nb_logger.error(f"❌ Viral PSSM generation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'execution_time': time.time() - step_start_time
            }
    
    def _extract_aggregated_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract aggregated data from input"""
        
        # Try to get from data_aggregation step first
        if 'data_aggregation' in input_data:
            return input_data['data_aggregation'].get('aggregated_data', {})
        
        # Fallback: construct aggregated data from individual steps
        aggregated_data = {
            'genome_data': {},
            'analysis_data': {},
            'quality_data': {},
            'execution_metadata': {}
        }
        
        # Extract data from known steps
        for step_id, step_result in input_data.items():
            if isinstance(step_result, dict) and step_result.get('success'):
                if step_id == 'data_acquisition':
                    aggregated_data['genome_data'].update({
                        'filtered_genomes': step_result.get('filtered_genomes', []),
                        'unique_proteins': step_result.get('unique_proteins', [])
                    })
                elif step_id == 'clustering':
                    aggregated_data['analysis_data']['clusters'] = step_result.get('clusters', [])
                elif step_id == 'pssm_analysis':
                    aggregated_data['analysis_data']['pssm_matrices'] = step_result.get('pssm_matrices', [])
        
        return aggregated_data
    
    async def _create_viral_pssm_structure(self, aggregated_data: Dict[str, Any], 
                                         input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create the viral_pssm.json structure"""
        
        # Calculate execution timing
        total_execution_time = sum(
            step_result.get('execution_time', 0) 
            for step_result in input_data.values() 
            if isinstance(step_result, dict)
        )
        
        # Generate workflow metadata
        workflow_metadata = await self._generate_workflow_metadata(aggregated_data, total_execution_time)
        
        # Calculate genome statistics
        genome_statistics = await self._calculate_genome_statistics(aggregated_data)
        
        # Compile analysis results
        analysis_results = await self._compile_analysis_results(aggregated_data)
        
        # Calculate quality metrics
        quality_metrics = await self._calculate_quality_metrics(aggregated_data, input_data)
        
        # Generate proteins array from clusters and PSSM data
        proteins_array = await self._generate_proteins_array(aggregated_data)
        
        # Generate statistics
        statistics = await self._generate_statistics(aggregated_data, proteins_array)
        
        # Enhance workflow metadata with counts
        workflow_metadata.update({
            'genome_count': len(aggregated_data.get('genome_data', {}).get('filtered_genomes', [])),
            'protein_count': len(proteins_array),
            'cluster_count': len(aggregated_data.get('analysis_data', {}).get('clusters', [])),
            'quality_metrics': {
                'overall_quality_score': aggregated_data.get('quality_data', {}).get('curation_report', {}).get('quality_score', 0.0)
            }
        })
        
        # Construct viral PSSM data
        viral_pssm_data = {
            "metadata": workflow_metadata,
            "workflow_metadata": workflow_metadata,  # Legacy compatibility
            "proteins": proteins_array,
            "statistics": statistics,
            "genome_statistics": genome_statistics,
            "analysis_results": analysis_results,
            "quality_metrics": quality_metrics,
            "analysis_summary": await self._generate_analysis_summary(aggregated_data)
        }
        
        # Add progress information if available
        progress_info = self._extract_progress_information(input_data)
        if progress_info:
            viral_pssm_data["progress_information"] = progress_info
        
        return viral_pssm_data
    
    async def _generate_workflow_metadata(self, aggregated_data: Dict[str, Any], 
                                        execution_time: float) -> Dict[str, Any]:
        """Generate workflow metadata section"""
        
        metadata = {
            "workflow_name": self.metadata_config.get('workflow_name', 'AlphavirusWorkflow'),
            "version": self.metadata_config.get('version', '4.2.0'),
            "created_at": datetime.now().isoformat(),
            "generated_at": datetime.now().isoformat(),  # For test compatibility
            "execution_time": execution_time,
            "timestamp": time.time(),
            "organism": self.metadata_config.get('organism', 'Alphavirus'),
            "method": self.metadata_config.get('method', 'nanobrain_alphavirus_analysis'),
            "framework": "NanoBrain",
            "framework_version": "2.0.0"
        }
        
        return metadata
    
    async def _calculate_genome_statistics(self, aggregated_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate genome statistics section"""
        
        genome_data = aggregated_data.get('genome_data', {})
        analysis_data = aggregated_data.get('analysis_data', {})
        
        statistics = {
            "total_genomes_acquired": len(genome_data.get('original_genomes', [])),
            "filtered_genomes": len(genome_data.get('filtered_genomes', [])),
            "unique_proteins": len(genome_data.get('unique_proteins', [])),
            "protein_clusters": len(analysis_data.get('clusters', [])),
            "pssm_matrices_generated": len(analysis_data.get('pssm_matrices', []))
        }
        
        return statistics
    
    async def _compile_analysis_results(self, aggregated_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compile analysis results section"""
        
        analysis_data = aggregated_data.get('analysis_data', {})
        quality_data = aggregated_data.get('quality_data', {})
        
        results = {
            "clusters": analysis_data.get('clusters', []),
            "pssm_matrices": analysis_data.get('pssm_matrices', []),
            "alignment_statistics": analysis_data.get('alignment_quality_stats'),
            "curation_report": quality_data.get('curation_report')
        }
        
        return results
    
    async def _calculate_quality_metrics(self, aggregated_data: Dict[str, Any], 
                                       input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate quality metrics section"""
        
        quality_data = aggregated_data.get('quality_data', {})
        step_metrics = quality_data.get('step_quality_metrics', {})
        
        metrics = {
            "overall_quality_score": 0.0,
            "data_completeness": 0.0,
            "analysis_confidence": 0.0,
            "step_success_rate": 0.0
        }
        
        # Calculate step success rate
        successful_steps = sum(1 for result in input_data.values() 
                             if isinstance(result, dict) and result.get('success'))
        total_steps = len(input_data)
        metrics["step_success_rate"] = successful_steps / total_steps if total_steps > 0 else 0.0
        
        return metrics
    
    async def _generate_analysis_summary(self, aggregated_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate analysis summary section"""
        
        genome_data = aggregated_data.get('genome_data', {})
        analysis_data = aggregated_data.get('analysis_data', {})
        execution_metadata = aggregated_data.get('execution_metadata', {})
        
        # Calculate total execution time from metadata
        total_execution_time = (
            execution_metadata.get('data_acquisition_execution_time', 0) + 
            execution_metadata.get('clustering_execution_time', 0) + 
            execution_metadata.get('pssm_analysis_execution_time', 0)
        )
        
        # Find highest conservation protein from PSSM matrices
        highest_conservation_protein = "unknown"
        highest_conservation_score = 0.0
        pssm_matrices = analysis_data.get('pssm_matrices', [])
        
        for pssm in pssm_matrices:
            conservation_score = pssm.get('metadata', {}).get('conservation_score', 0.0)
            if conservation_score > highest_conservation_score:
                highest_conservation_score = conservation_score
                cluster_id = pssm.get('cluster_id', '')
                highest_conservation_protein = self.cluster_id_mapping.get(cluster_id, cluster_id)
        
        summary = {
            "workflow_completion_status": "completed",
            "total_proteins_analyzed": len(genome_data.get('unique_proteins', [])),
            "clusters_generated": len(analysis_data.get('clusters', [])),
            "pssm_matrices_created": len(analysis_data.get('pssm_matrices', [])),
            "total_execution_time": total_execution_time,
            "key_findings": {
                "protein_families_identified": len(analysis_data.get('clusters', [])),
                "conservation_patterns": "Analyzed",
                "functional_annotations": "Generated",
                "highest_conservation_protein": highest_conservation_protein,
                "highest_conservation_score": highest_conservation_score
            },
            "data_sources": {
                "genomes_analyzed": len(genome_data.get('filtered_genomes', [])),
                "proteins_clustered": len(analysis_data.get('clusters', [])),
                "original_genomes": len(genome_data.get('original_genomes', [])),
                "total_sequences": aggregated_data.get('quality_data', {}).get('curation_report', {}).get('total_sequences', 0),
                "filtered_sequences": aggregated_data.get('quality_data', {}).get('curation_report', {}).get('filtered_sequences', 0)
            },
            "curation_statistics": aggregated_data.get('quality_data', {}).get('curation_report', {})
        }
        
        return summary
    
    def _extract_progress_information(self, input_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract progress information from input data"""
        
        progress_info = {}
        
        # Extract timing information
        step_timings = {}
        for step_id, step_result in input_data.items():
            if isinstance(step_result, dict) and 'execution_time' in step_result:
                step_timings[step_id] = step_result['execution_time']
        
        if step_timings:
            progress_info['step_execution_times'] = step_timings
            progress_info['total_workflow_time'] = sum(step_timings.values())
        
        return progress_info if progress_info else None
    
    async def _generate_proteins_array(self, aggregated_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate proteins array from clusters and PSSM matrices.
        Expected by tests.
        
        Args:
            aggregated_data: Aggregated data from previous steps
            
        Returns:
            List of protein dictionaries with PSSM information
        """
        proteins = []
        
        analysis_data = aggregated_data.get('analysis_data', {})
        clusters = analysis_data.get('clusters', [])
        pssm_matrices = analysis_data.get('pssm_matrices', [])
        
        # Create mapping of cluster_id to PSSM matrix
        pssm_by_cluster = {pssm['cluster_id']: pssm for pssm in pssm_matrices}
        
        # Process clusters if available
        for cluster in clusters:
            cluster_id = cluster.get('cluster_id', '')
            protein_type = cluster.get('protein_type', cluster_id)
            
            # Map cluster ID to protein name using configuration
            protein_name = self.cluster_id_mapping.get(cluster_id, protein_type)
            
            # Get PSSM matrix for this cluster
            pssm_data = pssm_by_cluster.get(cluster_id, {})
            pssm_matrix = pssm_data.get('matrix', [])
            pssm_metadata = pssm_data.get('metadata', {})
            
            protein_entry = {
                'id': protein_name,
                'cluster_id': cluster_id,
                'function': self._get_protein_function(protein_name),
                'pssm_matrix': pssm_matrix,
                'sequence_count': pssm_metadata.get('sequence_count', cluster.get('member_count', 0)),
                'conservation_score': pssm_metadata.get('conservation_score', 0.0),
                'alignment_length': pssm_metadata.get('alignment_length', 0),
                'representative_sequence': cluster.get('representative_sequence', '')
            }
            
            proteins.append(protein_entry)
        
        # Handle case where there are PSSM matrices but no cluster information
        # This can happen in test scenarios or when clustering data is missing
        if not clusters and pssm_matrices:
            for pssm_data in pssm_matrices:
                cluster_id = pssm_data.get('cluster_id', '')
                protein_name = self.cluster_id_mapping.get(cluster_id, cluster_id)
                pssm_matrix = pssm_data.get('matrix', [])
                pssm_metadata = pssm_data.get('metadata', {})
                
                protein_entry = {
                    'id': protein_name,
                    'cluster_id': cluster_id,
                    'function': self._get_protein_function(protein_name),
                    'pssm_matrix': pssm_matrix,
                    'sequence_count': pssm_metadata.get('sequence_count', 0),
                    'conservation_score': pssm_metadata.get('conservation_score', 0.0),
                    'alignment_length': pssm_metadata.get('alignment_length', 0),
                    'representative_sequence': ''
                }
                
                proteins.append(protein_entry)
        
        return proteins

    async def _generate_statistics(self, aggregated_data: Dict[str, Any], proteins_array: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate statistics section for viral PSSM.
        Expected by tests.
        
        Args:
            aggregated_data: Aggregated data from previous steps
            proteins_array: Generated proteins array
            
        Returns:
            Statistics dictionary
        """
        # Calculate conservation scores
        conservation_scores = [p.get('conservation_score', 0.0) for p in proteins_array]
        avg_conservation = sum(conservation_scores) / len(conservation_scores) if conservation_scores else 0.0
        
        # Calculate total sequences analyzed
        total_sequences = sum(p.get('sequence_count', 0) for p in proteins_array)
        
        statistics = {
            'total_proteins': len(proteins_array),
            'average_conservation_score': avg_conservation,
            'total_sequences_analyzed': total_sequences,
            'clusters_analyzed': len(aggregated_data.get('analysis_data', {}).get('clusters', [])),
            'pssm_matrices_generated': len(aggregated_data.get('analysis_data', {}).get('pssm_matrices', []))
        }
        
        return statistics
    
    async def _validate_viral_pssm(self, viral_pssm_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the generated viral PSSM structure"""
        
        validation_result = {
            'valid': True,
            'completeness_score': 0.0,
            'quality_score': 0.0,
            'validation_errors': [],
            'validation_warnings': []
        }
        
        try:
            # Check required sections
            required_sections = ['metadata', 'genome_statistics', 'analysis_results', 'quality_metrics']
            missing_sections = [section for section in required_sections if section not in viral_pssm_data]
            
            if missing_sections:
                validation_result['validation_errors'].extend([
                    f"Missing required section: {section}" for section in missing_sections
                ])
                validation_result['valid'] = False
            
            # Calculate completeness score
            completeness_score = (len(required_sections) - len(missing_sections)) / len(required_sections)
            validation_result['completeness_score'] = completeness_score
            
            # Calculate quality score (basic implementation)
            quality_score = 1.0 if validation_result['valid'] else 0.5
            validation_result['quality_score'] = quality_score
            
            return validation_result
            
        except Exception as e:
            validation_result['valid'] = False
            validation_result['validation_errors'].append(f"Validation exception: {e}")
            return validation_result

    def _get_protein_function(self, protein_name: str) -> str:
        """
        Get protein function from configuration mapping.
        Expected by tests.
        
        Args:
            protein_name: Name of the protein to look up
            
        Returns:
            Function description or 'Unknown function' if not found
        """
        return self.protein_functions.get(protein_name, 'Unknown function')
    
    def _calculate_average_conservation(self, pssm_matrices: List[Dict[str, Any]]) -> float:
        """
        Calculate average conservation score from PSSM matrices.
        Expected by tests.
        
        Args:
            pssm_matrices: List of PSSM matrix dictionaries with metadata
            
        Returns:
            Average conservation score
        """
        if not pssm_matrices:
            return 0.0
        
        conservation_scores = []
        for matrix in pssm_matrices:
            metadata = matrix.get('metadata', {})
            score = metadata.get('conservation_score', 0.0)
            if isinstance(score, (int, float)):
                conservation_scores.append(float(score))
        
        return sum(conservation_scores) / len(conservation_scores) if conservation_scores else 0.0

    def _generate_validation_results(self, aggregated_data: Dict[str, Any]) -> Dict[str, bool]:
        """
        Generate validation results for quality metrics.
        Expected by tests.
        
        Args:
            aggregated_data: Aggregated data from workflow steps
            
        Returns:
            Dictionary of validation check results
        """
        analysis_data = aggregated_data.get('analysis_data', {})
        clusters = analysis_data.get('clusters', [])
        pssm_matrices = analysis_data.get('pssm_matrices', [])
        
        # Check minimum clusters requirement from configuration
        min_clusters = self.validation_config.get('minimum_clusters', 1)
        minimum_clusters_check = len(clusters) >= min_clusters or len(pssm_matrices) >= min_clusters
        
        # Check PSSM size requirements
        max_pssm_size = self.validation_config.get('maximum_pssm_size', 10000)
        pssm_size_check = True
        for pssm in pssm_matrices:
            matrix = pssm.get('matrix', [])
            matrix_size = len(str(matrix))
            if matrix_size > max_pssm_size:
                pssm_size_check = False
                break
        
        return {
            'minimum_clusters_check': minimum_clusters_check,
            'pssm_size_check': pssm_size_check
        }

    async def _generate_metadata(self, aggregated_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate metadata for viral PSSM (for test mocking purposes).
        Expected by error handling tests.
        
        Args:
            aggregated_data: Aggregated data from workflow steps
            
        Returns:
            Metadata dictionary
        """
        return await self._generate_workflow_metadata(aggregated_data, 0.0) 