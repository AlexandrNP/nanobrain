"""
Alphavirus Protein Analysis Workflow

Main orchestrator for the 14-step Alphavirus protein analysis workflow
with comprehensive progress reporting, YAML-configurable steps, and
proper NanoBrain framework integration.

This workflow integrates:
- BV-BRC data acquisition (Steps 1-7)
- Annotation mapping (Step 8)
- Sequence curation (Steps 9-11) 
- Clustering analysis (Step 12)
- Multiple sequence alignment (Step 13)
- PSSM analysis and reporting (Step 14)

Author: NanoBrain Development Team
Date: December 2024
Version: 4.2.0
"""

import asyncio
import time
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path
import json
import yaml

from nanobrain.core.workflow import Workflow, WorkflowConfig
from nanobrain.core.logging_system import get_logger

from .steps.bv_brc_data_acquisition_step import BVBRCDataAcquisitionStep
from .steps.annotation_mapping_step import AnnotationMappingStep
from .steps.sequence_curation_step import SequenceCurationStep
from .steps.clustering_step import ClusteringStep
from .steps.alignment_step import AlignmentStep
from .steps.pssm_analysis_step import PSSMAnalysisStep
from .config.workflow_config import AlphavirusWorkflowConfig


class WorkflowData:
    """Container for workflow data that passes between steps"""
    
    def __init__(self):
        # Step 1-7: BV-BRC Data Acquisition
        self.original_genomes = []
        self.filtered_genomes = []
        self.unique_proteins = []
        self.protein_sequences = []
        self.protein_annotations = []
        self.annotated_fasta = ""
        
        # Step 8: Annotation Mapping
        self.standardized_annotations = []
        self.genome_schematics = []
        
        # Step 9-11: Sequence Curation
        self.length_analysis = None
        self.curation_report = None
        self.problematic_sequences = []
        
        # Step 12: Clustering
        self.clusters = []
        self.clustering_analysis = None
        
        # Step 13: Alignment
        self.aligned_clusters = []
        self.alignment_quality_stats = None
        
        # Step 14: PSSM Analysis
        self.pssm_matrices = []
        self.final_curation_report = None
        
        # Metadata
        self.execution_start_time = time.time()
        self.step_timings = {}
        self.quality_metrics = {}
        
    def update_from_acquisition(self, result: Dict[str, Any]) -> None:
        """Update data from BV-BRC acquisition step"""
        self.original_genomes = result.get('original_genomes', [])
        self.filtered_genomes = result.get('filtered_genomes', [])
        self.unique_proteins = result.get('unique_proteins', [])
        self.protein_sequences = result.get('protein_sequences', [])
        self.protein_annotations = result.get('protein_annotations', [])
        self.annotated_fasta = result.get('annotated_fasta', '')
        
    def update_from_mapping(self, result: Dict[str, Any]) -> None:
        """Update data from annotation mapping step"""
        self.standardized_annotations = result.get('standardized_annotations', [])
        self.genome_schematics = result.get('genome_schematics', [])
        
    def update_from_curation(self, result: Dict[str, Any]) -> None:
        """Update data from sequence curation step"""
        self.length_analysis = result.get('length_analysis')
        self.curation_report = result.get('curation_report')
        self.problematic_sequences = result.get('problematic_sequences', [])
        
    def update_from_clustering(self, result: Dict[str, Any]) -> None:
        """Update data from clustering step"""
        self.clusters = result.get('clusters', [])
        self.clustering_analysis = result.get('clustering_analysis')
        
    def update_from_alignment(self, result: Dict[str, Any]) -> None:
        """Update data from alignment step"""
        self.aligned_clusters = result.get('aligned_clusters', [])
        self.alignment_quality_stats = result.get('alignment_quality_stats')
        
    def update_from_pssm(self, result: Dict[str, Any]) -> None:
        """Update data from PSSM analysis step"""
        self.pssm_matrices = result.get('pssm_matrices', [])
        self.final_curation_report = result.get('final_curation_report')


class WorkflowResult:
    """Container for final workflow results"""
    
    def __init__(self, success: bool, workflow_data: Optional[WorkflowData] = None,
                 error: Optional[str] = None, execution_time: Optional[float] = None,
                 output_files: Optional[Dict[str, str]] = None):
        self.success = success
        self.workflow_data = workflow_data
        self.error = error
        self.execution_time = execution_time
        self.output_files = output_files or {}
        self.viral_pssm_json = None


class AlphavirusWorkflow(Workflow):
    """
    Main orchestrator for Alphavirus protein analysis workflow
    
    Executes all 14 steps in sequence with proper error handling,
    checkpointing, progress tracking, and YAML-configurable steps
    following NanoBrain framework patterns.
    """
    
    def __init__(self, config_path: Optional[str] = None, session_id: Optional[str] = None, **kwargs):
        # Load YAML configuration
        if config_path is None:
            config_path = Path(__file__).parent / "config" / "AlphavirusWorkflow.yml"
        
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
        
        # Create WorkflowConfig from YAML
        workflow_config = WorkflowConfig(**yaml_config)
        
        # Initialize with session ID for progress tracking
        super().__init__(workflow_config, session_id=session_id, **kwargs)
        
        # Store YAML config for step initialization
        self.yaml_config = yaml_config
        
        # Initialize workflow data container
        self.workflow_data = WorkflowData()
        
        # Progress tracking callbacks
        self.progress_callbacks: List[Callable] = []
        
        # Initialize execution timing
        self.execution_start_time = None
        
        self.workflow_logger.info("🧬 Alphavirus Workflow initialized with progress reporting")
        
    async def execute_full_workflow(self, input_params: Optional[Dict[str, Any]] = None) -> WorkflowResult:
        """
        Execute complete 14-step Alphavirus analysis workflow with progress tracking
        
        Args:
            input_params: Optional input parameters for the workflow
            
        Returns:
            WorkflowResult: Complete workflow results with progress information
        """
        
        self.execution_start_time = time.time()
        
        # Set default input params
        if input_params is None:
            input_params = {
                "target_genus": "Alphavirus",
                "output_directory": self.yaml_config.get('resources', {}).get('temporary_directory', 'data/alphavirus_analysis')
            }
            
        try:
            self.workflow_logger.info("🧬 Starting Alphavirus protein analysis workflow with progress tracking")
            
            # Update progress: workflow started
            if self.progress_reporter:
                await self.progress_reporter.update_progress(
                    'workflow_init', 0, 'running',
                    message="Initializing Alphavirus analysis workflow"
                )
            
            # Execute the workflow using the parent class method
            result = await self.process(input_params)
            
            # Calculate execution time
            execution_time = time.time() - self.execution_start_time
            
            # Collect output files
            output_files = await self._collect_output_files(self.workflow_data)
            
            # Generate viral PSSM JSON
            viral_pssm_json = await self._generate_viral_pssm_json(self.workflow_data)
            
            # Create final result
            workflow_result = WorkflowResult(
                success=True,
                workflow_data=self.workflow_data,
                execution_time=execution_time,
                output_files=output_files
            )
            workflow_result.viral_pssm_json = viral_pssm_json
            
            # Update progress: workflow completed
            if self.progress_reporter:
                await self.progress_reporter.update_progress(
                    'workflow_complete', 100, 'completed',
                    message=f"Alphavirus analysis completed in {execution_time:.1f}s"
                )
            
            self.workflow_logger.info(f"✅ Alphavirus workflow completed successfully in {execution_time:.1f}s")
            
            return workflow_result
            
        except Exception as e:
            execution_time = time.time() - self.execution_start_time
            
            # Update progress: workflow failed
            if self.progress_reporter:
                await self.progress_reporter.update_progress(
                    'workflow_error', 0, 'failed',
                    error=str(e),
                    technical_details={
                        'execution_time': execution_time,
                        'error_type': type(e).__name__
                    }
                )
            
            self.workflow_logger.error(f"❌ Alphavirus workflow failed after {execution_time:.1f}s: {e}")
            
            return WorkflowResult(
                success=False,
                error=str(e),
                execution_time=execution_time
            )
    
    async def process(self, input_data: Dict[str, Any], **kwargs) -> Any:
        """
        Execute the alphavirus workflow with progress reporting.
        
        This method routes through the workflow steps based on YAML configuration
        and provides comprehensive progress tracking.
        """
        # Update progress: workflow processing started
        if self.progress_reporter:
            await self.progress_reporter.update_progress(
                'workflow_processing', 5, 'running',
                message="Starting alphavirus protein analysis steps"
            )
        
        # Execute the workflow using the parent class method
        result = await super().process(input_data, **kwargs)
        
        # Update progress: workflow processing completed
        if self.progress_reporter:
            await self.progress_reporter.update_progress(
                'workflow_processing', 95, 'completed',
                message="Alphavirus protein analysis steps completed"
            )
        
        return result
    
    async def _collect_output_files(self, workflow_data: WorkflowData) -> Dict[str, str]:
        """
        Collect and organize output files from the workflow execution
        
        Args:
            workflow_data: Container with all workflow results
            
        Returns:
            Dictionary mapping file types to file paths
        """
        
        output_files = {}
        
        try:
            # Get output directory from configuration
            output_dir = self.yaml_config.get('resources', {}).get('temporary_directory', 'data/alphavirus_analysis')
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Generate output files
            if workflow_data.filtered_genomes:
                genomes_file = output_path / "alphavirus_filtered_genomes.json"
                with open(genomes_file, 'w') as f:
                    json.dump(workflow_data.filtered_genomes, f, indent=2)
                output_files['filtered_genomes'] = str(genomes_file)
            
            if workflow_data.unique_proteins:
                proteins_file = output_path / "alphavirus_unique_proteins.fasta"
                with open(proteins_file, 'w') as f:
                    f.write(workflow_data.annotated_fasta)
                output_files['unique_proteins'] = str(proteins_file)
            
            if workflow_data.clusters:
                clusters_file = output_path / "alphavirus_clusters.json"
                with open(clusters_file, 'w') as f:
                    json.dump(workflow_data.clusters, f, indent=2)
                output_files['clusters'] = str(clusters_file)
            
            if workflow_data.pssm_matrices:
                pssm_file = output_path / "alphavirus_pssm_matrices.json"
                with open(pssm_file, 'w') as f:
                    json.dump(workflow_data.pssm_matrices, f, indent=2)
                output_files['pssm_matrices'] = str(pssm_file)
            
            if workflow_data.curation_report:
                report_file = output_path / "alphavirus_curation_report.json"
                with open(report_file, 'w') as f:
                    json.dump(workflow_data.curation_report, f, indent=2)
                output_files['curation_report'] = str(report_file)
            
            self.workflow_logger.info(f"📁 Output files collected: {len(output_files)} files")
            
        except Exception as e:
            self.workflow_logger.error(f"❌ Error collecting output files: {e}")
        
        return output_files
    
    async def _generate_viral_pssm_json(self, workflow_data: WorkflowData) -> Dict[str, Any]:
        """
        Generate the final viral PSSM JSON output for integration
        
        Args:
            workflow_data: Container with all workflow results
            
        Returns:
            Dictionary with viral PSSM data in expected format
        """
        
        try:
            viral_pssm_data = {
                "workflow_metadata": {
                    "workflow_name": "AlphavirusWorkflow",
                    "version": "4.2.0",
                    "execution_time": time.time() - self.execution_start_time,
                    "timestamp": time.time(),
                    "step_count": len(self.workflow_config.steps)
                },
                "genome_statistics": {
                    "total_genomes_acquired": len(workflow_data.original_genomes),
                    "filtered_genomes": len(workflow_data.filtered_genomes),
                    "unique_proteins": len(workflow_data.unique_proteins),
                    "protein_clusters": len(workflow_data.clusters)
                },
                "analysis_results": {
                    "clusters": workflow_data.clusters,
                    "pssm_matrices": workflow_data.pssm_matrices,
                    "alignment_statistics": workflow_data.alignment_quality_stats,
                    "curation_report": workflow_data.curation_report
                },
                "quality_metrics": workflow_data.quality_metrics,
                "step_timings": workflow_data.step_timings
            }
            
            # Add progress information if available
            if self.progress_reporter:
                progress_summary = self.get_progress_summary()
                if progress_summary:
                    viral_pssm_data["progress_information"] = progress_summary
            
            self.workflow_logger.info("📊 Viral PSSM JSON generated successfully")
            
            return viral_pssm_data
            
        except Exception as e:
            self.workflow_logger.error(f"❌ Error generating viral PSSM JSON: {e}")
            return {}
    
    def add_progress_callback(self, callback: Callable) -> None:
        """Add progress callback for workflow updates."""
        self.progress_callbacks.append(callback)
        if self.progress_reporter:
            self.progress_reporter.add_progress_callback(callback)
    
    def get_execution_time(self) -> float:
        """Get current execution time."""
        if self.execution_start_time:
            return time.time() - self.execution_start_time
        return 0.0
    
    async def get_workflow_status(self) -> Dict[str, Any]:
        """Get comprehensive workflow status including progress."""
        status = {
            'workflow_name': 'AlphavirusWorkflow',
            'execution_time': self.get_execution_time(),
            'completed_steps': len(self.completed_steps),
            'failed_steps': len(self.failed_steps),
            'total_steps': len(self.child_steps),
            'is_complete': self.is_workflow_complete
        }
        
        # Add progress information
        if self.progress_reporter:
            progress_summary = self.get_progress_summary()
            if progress_summary:
                status['progress'] = progress_summary
        
        return status
    
    async def restore_from_checkpoint(self, checkpoint_data: Dict[str, Any]) -> bool:
        """Restore workflow state from checkpoint."""
        try:
            # Restore workflow data
            if 'workflow_data' in checkpoint_data:
                self.workflow_data = WorkflowData()
                # Restore specific data fields
                for key, value in checkpoint_data['workflow_data'].items():
                    if hasattr(self.workflow_data, key):
                        setattr(self.workflow_data, key, value)
            
            # Restore execution state
            if 'execution_state' in checkpoint_data:
                state = checkpoint_data['execution_state']
                self.current_step_index = state.get('current_step_index', 0)
                self.completed_steps = set(state.get('completed_steps', []))
                self.failed_steps = set(state.get('failed_steps', []))
            
            self.workflow_logger.info("🔄 Workflow state restored from checkpoint")
            return True
            
        except Exception as e:
            self.workflow_logger.error(f"❌ Failed to restore from checkpoint: {e}")
            return False


# Factory function for creating workflow instances
async def create_alphavirus_workflow(config_path: Optional[str] = None,
                                   session_id: Optional[str] = None,
                                   **kwargs) -> AlphavirusWorkflow:
    """
    Factory function to create and initialize AlphavirusWorkflow.
    
    Args:
        config_path: Path to YAML configuration file
        session_id: Session ID for progress tracking
        **kwargs: Additional workflow parameters
        
    Returns:
        Initialized AlphavirusWorkflow instance
    """
    workflow = AlphavirusWorkflow(config_path=config_path, session_id=session_id, **kwargs)
    await workflow.initialize()
    return workflow


# Legacy compatibility function
def create_workflow_from_config(config: Optional[AlphavirusWorkflowConfig] = None) -> AlphavirusWorkflow:
    """
    Legacy compatibility function for creating workflow from old config format.
    
    Args:
        config: Legacy AlphavirusWorkflowConfig instance
        
    Returns:
        AlphavirusWorkflow instance (not initialized)
    """
    # Convert legacy config to new format if needed
    if config:
        # This would involve converting the old config format to YAML
        # For now, use default YAML configuration
        pass
    
    return AlphavirusWorkflow() 