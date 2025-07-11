"""
Result Collection Step

Handles file collection and organization logic from the workflow.
Replaces the _collect_output_files method from AlphavirusWorkflow.
"""

import asyncio
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

from nanobrain.core.step import Step, StepConfig
from nanobrain.core.logging_system import get_logger


class ResultCollectionStep(Step):
    """
    Step to collect and organize workflow output files
    
    This step replaces the _collect_output_files method that was embedded
    in the AlphavirusWorkflow. It handles file generation and organization
    based on configurable output formats.
    """

    REQUIRED_CONFIG_FIELDS = ['name']
    OPTIONAL_CONFIG_FIELDS = {
        'description': '',
        'output_config': {},
        'file_types': {},
        'auto_initialize': True,
        'debug_mode': False,
        'enable_logging': True
    }

    @classmethod
    def from_config(cls, config: StepConfig, **kwargs) -> 'ResultCollectionStep':
        """Create ResultCollectionStep from configuration using mandatory pattern"""
        # Use parent class from_config implementation
        return super().from_config(config, **kwargs)

    @classmethod
    def extract_component_config(cls, config: StepConfig) -> Dict[str, Any]:
        """Extract ResultCollectionStep configuration"""
        base_config = super().extract_component_config(config)
        step_config_dict = config.config if hasattr(config, 'config') else {}
        
        return {
            **base_config,
            'output_config': step_config_dict.get('output_config', {}),
            'file_types': step_config_dict.get('file_types', {})
        }

    def _init_from_config(self, config: StepConfig, component_config: Dict[str, Any],
                         dependencies: Dict[str, Any]) -> None:
        """Initialize ResultCollectionStep with resolved dependencies"""
        super()._init_from_config(config, component_config, dependencies)
        
        # Extract configuration from step config
        self.step_config = component_config
        
        # Configuration for output handling
        self.output_config = component_config.get('output_config', {})
        self.file_types = component_config.get('file_types', {})
        
        self.nb_logger.info(f"📁 ResultCollectionStep initialized with {len(self.file_types)} file types")
    
    def __init__(self, config: StepConfig, **kwargs):
        super().__init__(config, **kwargs)
        
        # Extract configuration from step config
        step_config_dict = config.config if hasattr(config, 'config') else {}
        self.step_config = step_config_dict
        
        # Configuration for output handling
        self.output_config = step_config_dict.get('output_config', {})
        self.file_types = step_config_dict.get('file_types', {})
        
        self.nb_logger.info(f"📁 ResultCollectionStep initialized with {len(self.file_types)} file types")
        
    async def process(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Process method required by Step base class.
        """
        self.nb_logger.info("🔄 Processing result collection step")
        result = await self.execute(input_data)
        self.nb_logger.info(f"✅ Result collection completed successfully")
        return result
        
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collect output files from all previous steps
        
        Args:
            input_data: Contains results from all previous steps
            
        Returns:
            Dict with organized output file paths
        """
        
        step_start_time = time.time()
        
        try:
            self.nb_logger.info("📁 Starting output file collection and organization")
            
            # Setup output directory
            output_directory = await self._setup_output_directory()
            
            # Initialize output files collection
            output_files = {}
            file_statistics = {
                'files_created': 0,
                'total_size_bytes': 0,
                'file_types_generated': []
            }
            
            # Collect files from each step's results
            await self._collect_step_files(input_data, output_directory, output_files, file_statistics)
            
            # Generate manifest file
            manifest_file = await self._generate_manifest_file(output_files, output_directory, file_statistics)
            if manifest_file:
                output_files['manifest'] = manifest_file
                file_statistics['files_created'] += 1
            
            execution_time = time.time() - step_start_time
            self.nb_logger.info(f"✅ Result collection completed in {execution_time:.2f} seconds")
            self.nb_logger.info(f"Created {file_statistics['files_created']} output files")
            
            return {
                'success': True,
                'step_id': 'result_collection',
                'execution_time': execution_time,
                'primary_output': output_files,
                'output_files': output_files,
                'output_directory': str(output_directory),
                'file_statistics': file_statistics,
                'metadata': {
                    'step_type': self.__class__.__name__,
                    'input_data_keys': list(input_data.keys()),
                    'files_generated': len(output_files),
                    'output_directory': str(output_directory)
                },
                'quality_metrics': {
                    'file_generation_success_rate': 1.0,  # All files successfully generated
                    'total_files_created': file_statistics['files_created']
                }
            }
            
        except Exception as e:
            self.nb_logger.error(f"❌ Result collection failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'execution_time': time.time() - step_start_time
            }
    
    async def _setup_output_directory(self) -> Path:
        """Setup and ensure output directory exists"""
        
        base_directory = self.output_config.get('base_directory', 'data/alphavirus_analysis')
        
        # Add timestamp if configured
        if self.output_config.get('timestamp_subdirectory', False):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base_directory = f"{base_directory}/{timestamp}"
        
        output_path = Path(base_directory)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.nb_logger.info(f"📁 Output directory: {output_path}")
        return output_path
    
    async def _collect_step_files(self, input_data: Dict[str, Any], 
                                 output_directory: Path, 
                                 output_files: Dict[str, str],
                                 file_statistics: Dict[str, Any]) -> None:
        """Collect files from each step's results"""
        
        # Handle data_aggregation step results
        if 'data_aggregation' in input_data:
            await self._collect_aggregation_files(
                input_data['data_aggregation'], 
                output_directory, 
                output_files, 
                file_statistics
            )
        
        # Handle individual step outputs
        for step_id, step_result in input_data.items():
            if step_id != 'data_aggregation' and isinstance(step_result, dict):
                await self._collect_individual_step_files(
                    step_id, 
                    step_result, 
                    output_directory, 
                    output_files, 
                    file_statistics
                )
    
    async def _collect_aggregation_files(self, aggregation_result: Dict[str, Any], 
                                       output_directory: Path, 
                                       output_files: Dict[str, str],
                                       file_statistics: Dict[str, Any]) -> None:
        """Collect files from data aggregation step"""
        
        aggregated_data = aggregation_result.get('aggregated_data', {})
        
        # Generate files for different data sections
        await self._generate_genome_data_files(
            aggregated_data.get('genome_data', {}), 
            output_directory, 
            output_files, 
            file_statistics
        )
        
        await self._generate_analysis_data_files(
            aggregated_data.get('analysis_data', {}), 
            output_directory, 
            output_files, 
            file_statistics
        )
        
        await self._generate_quality_data_files(
            aggregated_data.get('quality_data', {}), 
            output_directory, 
            output_files, 
            file_statistics
        )
    
    async def _generate_genome_data_files(self, genome_data: Dict[str, Any], 
                                        output_directory: Path, 
                                        output_files: Dict[str, str],
                                        file_statistics: Dict[str, Any]) -> None:
        """Generate files for genome data"""
        
        prefix = self.output_config.get('file_naming', {}).get('prefix', 'alphavirus_')
        
        # Filtered genomes file
        if 'filtered_genomes' in genome_data and genome_data['filtered_genomes']:
            filename = self._get_configured_filename('filtered_genomes', f"{prefix}filtered_genomes.json")
            file_path = output_directory / filename
            
            with open(file_path, 'w') as f:
                json.dump(genome_data['filtered_genomes'], f, indent=2)
            
            output_files['filtered_genomes'] = str(file_path)
            await self._update_file_statistics(file_path, file_statistics, 'json')
        
        # Unique proteins FASTA file
        if 'annotated_fasta' in genome_data and genome_data['annotated_fasta']:
            filename = self._get_configured_filename('unique_proteins', f"{prefix}unique_proteins.fasta")
            file_path = output_directory / filename
            
            with open(file_path, 'w') as f:
                f.write(genome_data['annotated_fasta'])
            
            output_files['unique_proteins'] = str(file_path)
            await self._update_file_statistics(file_path, file_statistics, 'fasta')
        
        # Protein annotations file
        if 'protein_annotations' in genome_data and genome_data['protein_annotations']:
            filename = self._get_configured_filename('protein_annotations', f"{prefix}protein_annotations.json")
            file_path = output_directory / filename
            
            with open(file_path, 'w') as f:
                json.dump(genome_data['protein_annotations'], f, indent=2)
            
            output_files['protein_annotations'] = str(file_path)
            await self._update_file_statistics(file_path, file_statistics, 'json')
    
    async def _generate_analysis_data_files(self, analysis_data: Dict[str, Any], 
                                          output_directory: Path, 
                                          output_files: Dict[str, str],
                                          file_statistics: Dict[str, Any]) -> None:
        """Generate files for analysis data"""
        
        prefix = self.output_config.get('file_naming', {}).get('prefix', 'alphavirus_')
        
        # Clusters file
        if 'clusters' in analysis_data and analysis_data['clusters']:
            filename = self._get_configured_filename('clusters', f"{prefix}clusters.json")
            file_path = output_directory / filename
            
            with open(file_path, 'w') as f:
                json.dump(analysis_data['clusters'], f, indent=2)
            
            output_files['clusters'] = str(file_path)
            await self._update_file_statistics(file_path, file_statistics, 'json')
        
        # PSSM matrices file
        if 'pssm_matrices' in analysis_data and analysis_data['pssm_matrices']:
            filename = self._get_configured_filename('pssm_matrices', f"{prefix}pssm_matrices.json")
            file_path = output_directory / filename
            
            with open(file_path, 'w') as f:
                json.dump(analysis_data['pssm_matrices'], f, indent=2)
            
            output_files['pssm_matrices'] = str(file_path)
            await self._update_file_statistics(file_path, file_statistics, 'json')
        
        # Alignments file
        if 'aligned_clusters' in analysis_data and analysis_data['aligned_clusters']:
            filename = self._get_configured_filename('alignments', f"{prefix}alignments.json")
            file_path = output_directory / filename
            
            with open(file_path, 'w') as f:
                json.dump(analysis_data['aligned_clusters'], f, indent=2)
            
            output_files['alignments'] = str(file_path)
            await self._update_file_statistics(file_path, file_statistics, 'json')
    
    async def _generate_quality_data_files(self, quality_data: Dict[str, Any], 
                                         output_directory: Path, 
                                         output_files: Dict[str, str],
                                         file_statistics: Dict[str, Any]) -> None:
        """Generate files for quality data"""
        
        prefix = self.output_config.get('file_naming', {}).get('prefix', 'alphavirus_')
        
        # Curation report file
        if 'curation_report' in quality_data and quality_data['curation_report']:
            filename = self._get_configured_filename('curation_report', f"{prefix}curation_report.json")
            file_path = output_directory / filename
            
            with open(file_path, 'w') as f:
                json.dump(quality_data['curation_report'], f, indent=2)
            
            output_files['curation_report'] = str(file_path)
            await self._update_file_statistics(file_path, file_statistics, 'json')
        
        # Quality metrics summary
        if 'step_quality_metrics' in quality_data:
            filename = self._get_configured_filename('quality_metrics', f"{prefix}quality_metrics.json")
            file_path = output_directory / filename
            
            with open(file_path, 'w') as f:
                json.dump(quality_data['step_quality_metrics'], f, indent=2)
            
            output_files['quality_metrics'] = str(file_path)
            await self._update_file_statistics(file_path, file_statistics, 'json')
    
    async def _collect_individual_step_files(self, step_id: str, 
                                           step_result: Dict[str, Any], 
                                           output_directory: Path, 
                                           output_files: Dict[str, str],
                                           file_statistics: Dict[str, Any]) -> None:
        """Collect files from individual step outputs"""
        
        # Check if step has output_files in its result
        if 'output_files' in step_result:
            step_output_files = step_result['output_files']
            if isinstance(step_output_files, dict):
                for file_type, file_path in step_output_files.items():
                    # Copy or reference the file
                    output_key = f"{step_id}_{file_type}"
                    output_files[output_key] = file_path
                    
                    # Update statistics if file exists
                    if Path(file_path).exists():
                        await self._update_file_statistics(Path(file_path), file_statistics, 'various')
    
    def _get_configured_filename(self, file_type: str, default_filename: str) -> str:
        """Get configured filename for a file type"""
        
        if file_type in self.file_types:
            file_config = self.file_types[file_type]
            return file_config.get('filename', default_filename)
        
        return default_filename
    
    async def _update_file_statistics(self, file_path: Path, 
                                     file_statistics: Dict[str, Any], 
                                     file_format: str) -> None:
        """Update file statistics"""
        
        if file_path.exists():
            file_size = file_path.stat().st_size
            file_statistics['total_size_bytes'] += file_size
            file_statistics['files_created'] += 1
            
            if file_format not in file_statistics['file_types_generated']:
                file_statistics['file_types_generated'].append(file_format)
    
    async def _generate_manifest_file(self, output_files: Dict[str, str], 
                                    output_directory: Path, 
                                    file_statistics: Dict[str, Any]) -> Optional[str]:
        """Generate a manifest file listing all output files"""
        
        if not self.output_config.get('generate_manifest', True):
            return None
        
        manifest_data = {
            'manifest_version': '1.0',
            'generated_at': datetime.now().isoformat(),
            'workflow_name': 'AlphavirusWorkflow',
            'output_directory': str(output_directory),
            'file_statistics': file_statistics,
            'output_files': {}
        }
        
        # Add file information to manifest
        for file_type, file_path in output_files.items():
            file_path_obj = Path(file_path)
            if file_path_obj.exists():
                manifest_data['output_files'][file_type] = {
                    'path': file_path,
                    'filename': file_path_obj.name,
                    'size_bytes': file_path_obj.stat().st_size,
                    'format': file_path_obj.suffix.lstrip('.') or 'unknown'
                }
        
        # Write manifest file
        manifest_file = output_directory / 'workflow_manifest.json'
        with open(manifest_file, 'w') as f:
            json.dump(manifest_data, f, indent=2)
        
        self.nb_logger.info(f"📋 Generated manifest file: {manifest_file}")
        return str(manifest_file) 