"""
Alphavirus Protein Analysis Workflow

Main orchestrator for the 14-step Alphavirus protein analysis workflow
as defined in PHASE2_IMPLEMENTATION_PLAN.md.

This workflow integrates:
- BV-BRC data acquisition (Steps 1-7)
- Annotation mapping (Step 8)
- Sequence curation (Steps 9-11) 
- Clustering analysis (Step 12)
- Multiple sequence alignment (Step 13)
- PSSM analysis and reporting (Step 14)
"""

import asyncio
import time
from typing import Dict, Any, List, Optional
from pathlib import Path
import json

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


class AlphavirusWorkflow:
    """
    Main orchestrator for Alphavirus protein analysis workflow
    
    Executes all 14 steps in sequence with proper error handling,
    checkpointing, and progress tracking.
    
    Note: This is a standalone workflow that doesn't inherit from the core Workflow class
    to maintain compatibility with the existing AlphavirusWorkflowConfig structure.
    """
    
    def __init__(self, config: Optional[AlphavirusWorkflowConfig] = None):
        # Load default config if none provided
        if config is None:
            config_path = Path(__file__).parent / "config" / "AlphavirusWorkflow.yml"
            config = AlphavirusWorkflowConfig.from_file(str(config_path))
        self.config = config
        self.logger = get_logger("alphavirus_workflow")
        
        # Initialize workflow steps
        self.steps = self._initialize_workflow_steps()
        
        # Execution tracking
        self.current_step = 0
        self.execution_start_time = None
        self.progress_callback = None
        
    def _initialize_workflow_steps(self) -> Dict[str, Any]:
        """Initialize all workflow steps with proper configuration"""
        
        return {
            'data_acquisition': BVBRCDataAcquisitionStep(
                bvbrc_config=self.config.bvbrc,
                step_config=self.config.bvbrc.__dict__
            ),
            'annotation_mapping': AnnotationMappingStep(
                step_config=self.config.quality_control.__dict__
            ),
            'sequence_curation': SequenceCurationStep(
                step_config=self.config.quality_control.__dict__
            ),
            'clustering': ClusteringStep(
                mmseqs_config=self.config.clustering,
                step_config=self.config.clustering.__dict__
            ),
            'alignment': AlignmentStep(
                muscle_config=self.config.alignment,
                step_config=self.config.alignment.__dict__
            ),
            'pssm_analysis': PSSMAnalysisStep(
                pssm_config=self.config.quality_control,
                step_config=self.config.output.__dict__
            )
        }
        
    async def execute_full_workflow(self, input_params: Optional[Dict[str, Any]] = None) -> WorkflowResult:
        """
        Execute complete 14-step Alphavirus analysis workflow
        
        Args:
            input_params: Optional input parameters for the workflow
            
        Returns:
            WorkflowResult: Complete workflow results
        """
        
        self.execution_start_time = time.time()
        workflow_data = WorkflowData()
        
        # Set default input params
        if input_params is None:
            input_params = {
                "target_genus": "Alphavirus",
                "output_directory": self.config.output.base_directory
            }
            
        try:
            self.logger.info("🧬 Starting Alphavirus protein analysis workflow")
            await self._update_progress("Initializing workflow", 0)
            
            # Steps 1-7: BV-BRC Data Acquisition
            self.logger.info("📊 Starting BV-BRC data acquisition (Steps 1-7)")
            await self._update_progress("BV-BRC data acquisition", 10)
            
            step_start = time.time()
            acquisition_result = await self.steps['data_acquisition'].execute(input_params)
            workflow_data.update_from_acquisition(acquisition_result)
            workflow_data.step_timings['data_acquisition'] = time.time() - step_start
            
            self.logger.info(f"✅ Acquired {len(workflow_data.unique_proteins)} unique proteins from {len(workflow_data.filtered_genomes)} genomes")
            
            # Step 8: Annotation Mapping
            self.logger.info("🔍 Starting annotation mapping (Step 8)")
            await self._update_progress("Annotation mapping and ICTV integration", 25)
            
            step_start = time.time()
            mapping_result = await self.steps['annotation_mapping'].execute({
                'annotations': workflow_data.protein_annotations,
                'genome_data': workflow_data.filtered_genomes
            })
            workflow_data.update_from_mapping(mapping_result)
            workflow_data.step_timings['annotation_mapping'] = time.time() - step_start
            
            # Steps 9-11: Sequence Curation
            self.logger.info("🧹 Starting sequence curation (Steps 9-11)")
            await self._update_progress("Sequence curation and quality control", 40)
            
            step_start = time.time()
            curation_result = await self.steps['sequence_curation'].execute({
                'sequences': workflow_data.protein_sequences,
                'annotations': workflow_data.standardized_annotations
            })
            workflow_data.update_from_curation(curation_result)
            workflow_data.step_timings['sequence_curation'] = time.time() - step_start
            
            # Step 12: Clustering
            self.logger.info("🗂️ Starting MMseqs2 clustering (Step 12)")
            await self._update_progress("Sequence clustering with conservation focus", 55)
            
            step_start = time.time()
            clustering_result = await self.steps['clustering'].execute({
                'curated_sequences': workflow_data.protein_sequences,
                'curation_report': workflow_data.curation_report
            })
            workflow_data.update_from_clustering(clustering_result)
            workflow_data.step_timings['clustering'] = time.time() - step_start
            
            self.logger.info(f"✅ Generated {len(workflow_data.clusters)} protein clusters")
            
            # Step 13: Alignment
            self.logger.info("📏 Starting multiple sequence alignment (Step 13)")
            await self._update_progress("Multiple sequence alignment", 70)
            
            step_start = time.time()
            alignment_result = await self.steps['alignment'].execute({
                'clusters': workflow_data.clusters
            })
            workflow_data.update_from_alignment(alignment_result)
            workflow_data.step_timings['alignment'] = time.time() - step_start
            
            # Step 14: PSSM Analysis
            self.logger.info("🎯 Starting PSSM analysis and curation report (Step 14)")
            await self._update_progress("PSSM generation and final analysis", 85)
            
            step_start = time.time()
            pssm_result = await self.steps['pssm_analysis'].execute({
                'aligned_clusters': workflow_data.aligned_clusters,
                'workflow_data': workflow_data
            })
            workflow_data.update_from_pssm(pssm_result)
            workflow_data.step_timings['pssm_analysis'] = time.time() - step_start
            
            # Generate final results
            await self._update_progress("Generating final outputs", 95)
            
            execution_time = time.time() - self.execution_start_time
            output_files = await self._collect_output_files(workflow_data)
            
            # Generate Viral_PSSM.json format output
            viral_pssm_json = await self._generate_viral_pssm_json(workflow_data)
            
            final_result = WorkflowResult(
                success=True,
                workflow_data=workflow_data,
                execution_time=execution_time,
                output_files=output_files
            )
            final_result.viral_pssm_json = viral_pssm_json
            
            await self._update_progress("Workflow completed successfully", 100)
            self.logger.info(f"🎉 Alphavirus workflow completed successfully in {execution_time:.2f} seconds")
            
            return final_result
            
        except Exception as e:
            execution_time = time.time() - self.execution_start_time if self.execution_start_time else 0
            self.logger.error(f"❌ Workflow failed after {execution_time:.2f} seconds: {e}")
            
            return WorkflowResult(
                success=False,
                error=str(e),
                execution_time=execution_time,
                workflow_data=workflow_data  # Return partial data
            )
            
    async def _update_progress(self, message: str, percentage: int) -> None:
        """Update workflow progress"""
        self.logger.info(f"Progress: {percentage}% - {message}")
        
        if self.progress_callback:
            await self.progress_callback({
                'percentage': percentage,
                'message': message,
                'current_step': self.current_step,
                'total_steps': 14
            })
            
    async def _collect_output_files(self, workflow_data: WorkflowData) -> Dict[str, str]:
        """Collect and organize output files"""
        base_dir = Path(self.config.output.base_directory)
        base_dir.mkdir(parents=True, exist_ok=True)
        
        output_files = {}
        
        # Save filtered genomes
        genomes_file = base_dir / "alphavirus_filtered_genomes.json"
        with open(genomes_file, 'w') as f:
            json.dump([g.__dict__ if hasattr(g, '__dict__') else g for g in workflow_data.filtered_genomes], f, indent=2)
        output_files['filtered_genomes'] = str(genomes_file)
        
        # Save unique proteins FASTA
        if workflow_data.annotated_fasta:
            fasta_file = base_dir / "alphavirus_unique_proteins.fasta"
            with open(fasta_file, 'w') as f:
                f.write(workflow_data.annotated_fasta)
            output_files['unique_proteins_fasta'] = str(fasta_file)
        
        # Save clusters
        if workflow_data.clusters:
            clusters_file = base_dir / "alphavirus_clusters.json"
            with open(clusters_file, 'w') as f:
                json.dump(workflow_data.clusters, f, indent=2)
            output_files['clusters'] = str(clusters_file)
        
        # Save PSSM matrices
        if workflow_data.pssm_matrices:
            pssm_file = base_dir / "alphavirus_pssm_matrices.json"
            with open(pssm_file, 'w') as f:
                json.dump(workflow_data.pssm_matrices, f, indent=2)
            output_files['pssm_matrices'] = str(pssm_file)
        
        # Save final curation report
        if workflow_data.final_curation_report:
            report_file = base_dir / "alphavirus_curation_report.json"
            with open(report_file, 'w') as f:
                json.dump(workflow_data.final_curation_report, f, indent=2)
            output_files['curation_report'] = str(report_file)
        
        return output_files
        
    async def _generate_viral_pssm_json(self, workflow_data: WorkflowData) -> Dict[str, Any]:
        """
        Generate output in Viral_PSSM.json format
        
        Based on: https://github.com/jimdavis1/Viral_Annotation/blob/main/Viral_PSSM.json
        """
        
        viral_pssm_output = {
            "metadata": {
                "organism": "Alphavirus",
                "analysis_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "coordinate_system": "1-based",
                "method": "nanobrain_alphavirus_analysis",
                "version": "1.0.0",
                "data_source": "BV-BRC",
                "total_genomes_analyzed": len(workflow_data.filtered_genomes),
                "clustering_method": "MMseqs2",
                "alignment_method": "MUSCLE",
                "pssm_generation_method": "custom_nanobrain"
            },
            "proteins": [],
            "analysis_summary": {
                "total_proteins": len(workflow_data.unique_proteins),
                "clusters_generated": len(workflow_data.clusters),
                "pssm_matrices_created": len(workflow_data.pssm_matrices),
                "execution_time_seconds": sum(workflow_data.step_timings.values()),
                "step_timings": workflow_data.step_timings
            },
            "quality_metrics": workflow_data.quality_metrics
        }
        
        # Add protein entries
        for i, cluster in enumerate(workflow_data.clusters[:10]):  # Limit for demo
            protein_entry = {
                "id": f"alphavirus_cluster_{i+1}",
                "function": cluster.get('consensus_annotation', 'hypothetical protein'),
                "protein_class": cluster.get('protein_class', 'unknown'),
                "cluster_info": {
                    "member_count": cluster.get('member_count', 0),
                    "consensus_score": cluster.get('consensus_score', 0.0)
                },
                "confidence_metrics": {
                    "overall_confidence": cluster.get('overall_confidence', 0.0)
                }
            }
            viral_pssm_output["proteins"].append(protein_entry)
        
        return viral_pssm_output
        
    def set_progress_callback(self, callback) -> None:
        """Set callback function for progress updates"""
        self.progress_callback = callback
        
    def get_execution_time(self) -> float:
        """Get current execution time"""
        if self.execution_start_time:
            return time.time() - self.execution_start_time
        return 0.0 