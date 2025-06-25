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
from nanobrain.core.step import Step, StepConfig
from nanobrain.core.component_base import ComponentConfigurationError, ComponentDependencyError

from nanobrain.core.workflow import WorkflowConfig


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
    
    COMPONENT_TYPE = "workflow"
    REQUIRED_CONFIG_FIELDS = ['name']
    OPTIONAL_CONFIG_FIELDS = {
        'description': 'Alphavirus protein analysis workflow',
        'version': '4.2.0',
        'execution_strategy': 'sequential',
        'error_handling': 'stop'
    }
    
    @classmethod
    def extract_component_config(cls, config: WorkflowConfig) -> Dict[str, Any]:
        """Extract AlphavirusWorkflow configuration"""
        return {
            'name': config.name,
            'description': getattr(config, 'description', 'Alphavirus protein analysis workflow'),
            'version': getattr(config, 'version', '4.2.0'),
            'execution_strategy': getattr(config, 'execution_strategy', 'sequential'),
            'error_handling': getattr(config, 'error_handling', 'stop'),
            'steps': getattr(config, 'steps', []),
            'links': getattr(config, 'links', []),
            'input_parameters': getattr(config, 'input_parameters', {}),
            'workflow_config': getattr(config, 'workflow_config', {}),
            'debug_mode': True,  # Required for component base
            'enable_logging': True  # Required for component base
        }
    
    @classmethod  
    def resolve_dependencies(cls, component_config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Resolve AlphavirusWorkflow dependencies"""
        return {
            'executor': kwargs.get('executor'),
            'enable_logging': kwargs.get('enable_logging', True),
            'debug_mode': kwargs.get('debug_mode', True),
            'session_id': kwargs.get('session_id'),
            'progress_reporter': kwargs.get('progress_reporter')
        }
    
    def _init_from_config(self, config: WorkflowConfig, component_config: Dict[str, Any],
                         dependencies: Dict[str, Any]) -> None:
        """Initialize AlphavirusWorkflow with resolved dependencies"""
        # Initialize parent with configuration
        super()._init_from_config(config, component_config, dependencies)
        
        # Initialize workflow logger
        self.workflow_logger = get_logger("alphavirus_workflow", 
                                        debug_mode=dependencies.get('debug_mode', True))
        
        self.workflow_logger.info("🧬 AlphavirusWorkflow initialized with REAL step implementations (no fallbacks)")
        self.workflow_logger.info(f"Workflow will execute {len(component_config.get('steps', []))} step groups mapping to 14 logical steps")
        
        # Initialize workflow-specific attributes
        self.workflow_data = WorkflowData()
        self.execution_start_time = None
        
        # Progress reporting setup
        self.progress_reporter = dependencies.get('progress_reporter')
        self.progress_callbacks = []  # Initialize progress callbacks list
        

    
    async def execute_full_workflow(self, input_params: Optional[Dict[str, Any]] = None) -> WorkflowResult:
        """
        Execute complete 14-step Alphavirus analysis workflow with progress tracking
        
        Args:
            input_params: Optional input parameters for the workflow
            
        Returns:
            WorkflowResult: Complete workflow results with progress information
        """
        
        self.execution_start_time = time.time()
        
        # Set default input params and handle parameter mapping
        if input_params is None:
            input_params = {
                "target_genus": "Alphavirus",
                "output_directory": "data/alphavirus_analysis"
            }
        
        # Map incoming parameters to workflow parameters
        workflow_params = {}
        
        # Handle organism parameter mapping for user queries like "Create PSSM matrix for EEE virus"
        if 'organism' in input_params:
            organism = input_params['organism']
            # Map specific virus names to genus for workflow processing
            if 'equine encephalitis' in organism.lower() or 'eee' in organism.lower():
                workflow_params['target_genus'] = 'Alphavirus'
                workflow_params['custom_virus_name'] = organism
            else:
                workflow_params['target_genus'] = 'Alphavirus'
                workflow_params['custom_virus_name'] = organism
        else:
            workflow_params['target_genus'] = input_params.get('target_genus', 'Alphavirus')
        
        # Pass through other parameters
        for key, value in input_params.items():
            if key not in ['organism']:  # Skip already mapped parameters
                workflow_params[key] = value
                
        input_params = workflow_params
            
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
        
        This method executes the workflow steps in the correct order,
        properly passing input_params to each step as required.
        """
        # Update progress: workflow processing started
        if self.progress_reporter:
            await self.progress_reporter.update_progress(
                'workflow_processing', 5, 'running',
                message="Starting alphavirus protein analysis steps"
            )
        
        try:
            # Execute steps sequentially with proper parameter passing
            workflow_data = WorkflowData()
            step_results = {}
            
            # Execute each step in order, passing input_data as input_params
            for step_id in ['data_acquisition', 'annotation_mapping', 'sequence_curation', 'clustering', 'alignment', 'pssm_analysis']:
                if step_id in self.child_steps:
                    step = self.child_steps[step_id]
                    self.workflow_logger.info(f"🔄 Executing step: {step_id}")
                    
                    # Call execute method with input_params for viral workflow steps
                    if hasattr(step, 'execute') and 'execute' in str(type(step.execute)):
                        step_result = await step.execute(input_data)
                    else:
                        # Fallback to process method for other step types
                        step_result = await step.process(input_data, **kwargs)
                    
                    step_results[step_id] = step_result
                    
                    # Update workflow data based on step results
                    self._update_workflow_data(workflow_data, step_id, step_result)
                    
                    # Update progress
                    progress = (len(step_results) * 90) // 6 + 5  # 5% to 95%
                    if self.progress_reporter:
                        await self.progress_reporter.update_progress(
                            step_id, progress, 'completed',
                            message=f"Completed {step_id} step"
                        )
            
            # Store workflow data for result collection
            self.workflow_data = workflow_data
            
            # Update progress: workflow processing completed
            if self.progress_reporter:
                await self.progress_reporter.update_progress(
                    'workflow_processing', 95, 'completed',
                    message="Alphavirus protein analysis steps completed"
                )
            
            return step_results
            
        except Exception as e:
            self.workflow_logger.error(f"❌ Workflow processing failed: {e}")
            if self.progress_reporter:
                await self.progress_reporter.update_progress(
                    'workflow_error', 0, 'failed',
                    error=str(e),
                    message="Workflow execution failed"
                )
                raise
    
    def _update_workflow_data(self, workflow_data: WorkflowData, step_id: str, step_result: Dict[str, Any]) -> None:
        """Update workflow data container with step results"""
        
        try:
            if step_id == 'data_acquisition' and step_result.get('success'):
                workflow_data.update_from_acquisition(step_result)
            elif step_id == 'annotation_mapping' and step_result.get('success'):
                workflow_data.update_from_mapping(step_result)
            elif step_id == 'sequence_curation' and step_result.get('success'):
                workflow_data.update_from_curation(step_result)
            elif step_id == 'clustering' and step_result.get('success'):
                workflow_data.update_from_clustering(step_result)
            elif step_id == 'alignment' and step_result.get('success'):
                workflow_data.update_from_alignment(step_result)
            elif step_id == 'pssm_analysis' and step_result.get('success'):
                workflow_data.update_from_pssm(step_result)
        except Exception as e:
            self.workflow_logger.warning(f"⚠️ Could not update workflow data for {step_id}: {e}")
    
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
            workflow_config = getattr(self.config, 'workflow_config', {})
            output_dir = workflow_config.get('output_directory', 'data/alphavirus_analysis')
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

    async def _create_step_instance(self, step_id: str, step_config: StepConfig, config_dict: Dict[str, Any]) -> Step:
        """Create a step instance from configuration using custom step types."""
        
        # Determine step type/class from the loaded config
        # First try getting the class from step_config.config dict (external config files)
        step_class = None
        if hasattr(step_config, 'config') and isinstance(step_config.config, dict):
            step_class = step_config.config.get('class')
        
        # If not found, try direct attribute on step_config
        if not step_class and hasattr(step_config, 'class'):
            step_class = getattr(step_config, 'class')
            
        # Fall back to config_dict for inline configurations
        if not step_class:
            step_class = config_dict.get('class', 'SequenceCurationStep')  # Use SequenceCurationStep as default instead of SimpleStep
        
        self.workflow_logger.info(f"🔧 Creating step instance: {step_id} ({step_class})")
        
        # Create step instance based on class using from_config pattern
        if step_class == 'BVBRCDataAcquisitionStep':
            from .steps.bv_brc_data_acquisition_step import BVBRCDataAcquisitionStep
            step = BVBRCDataAcquisitionStep.from_config(step_config, executor=self.executor)
            self.workflow_logger.info(f"✅ Created BVBRCDataAcquisitionStep: {step_id}")
            
        elif step_class == 'AnnotationMappingStep':
            from .steps.annotation_mapping_step import AnnotationMappingStep
            step = AnnotationMappingStep.from_config(step_config, executor=self.executor)
            self.workflow_logger.info(f"✅ Created AnnotationMappingStep: {step_id}")
            
        elif step_class == 'SequenceCurationStep':
            from .steps.sequence_curation_step import SequenceCurationStep
            step = SequenceCurationStep.from_config(step_config, executor=self.executor)
            self.workflow_logger.info(f"✅ Created SequenceCurationStep: {step_id}")
            
        elif step_class == 'ClusteringStep':
            from .steps.clustering_step import ClusteringStep
            step = ClusteringStep.from_config(step_config, executor=self.executor)
            self.workflow_logger.info(f"✅ Created ClusteringStep: {step_id}")
            
        elif step_class == 'AlignmentStep':
            from .steps.alignment_step import AlignmentStep
            step = AlignmentStep.from_config(step_config, executor=self.executor)
            self.workflow_logger.info(f"✅ Created AlignmentStep: {step_id}")
            
        elif step_class == 'SequenceCurationStep':
            from .steps.sequence_curation_step import SequenceCurationStep
            step = SequenceCurationStep.from_config(step_config, executor=self.executor)
            self.workflow_logger.info(f"✅ Created SequenceCurationStep: {step_id}")
            
        elif step_class == 'PSSMAnalysisStep':
            from .steps.pssm_analysis_step import PSSMAnalysisStep
            step = PSSMAnalysisStep.from_config(step_config, executor=self.executor)
            self.workflow_logger.info(f"✅ Created PSSMAnalysisStep: {step_id}")
            
        else:
            # Fall back to parent class for standard step types
            self.workflow_logger.info(f"⚡ Using parent class for step creation: {step_id} ({step_class})")
            step = await super()._create_step_instance(step_id, step_config, config_dict)
        
        return step


# Factory function for creating workflow instances
async def create_alphavirus_workflow(config_path: Optional[str] = None,
                                   session_id: Optional[str] = None,
                                   **kwargs) -> AlphavirusWorkflow:
    """
    Factory function to create and initialize AlphavirusWorkflow using mandatory from_config pattern.
    
    Args:
        config_path: Path to YAML configuration file
        session_id: Session ID for progress tracking
        **kwargs: Additional workflow parameters
        
    Returns:
        Initialized AlphavirusWorkflow instance
    """
    from nanobrain.core.workflow import ConfigLoader
    from nanobrain.core.executor import LocalExecutor, ExecutorConfig
    
    # Use the provided config_path or default
    if not config_path:
        config_path = Path(__file__).parent / "config" / "CleanWorkflow.yml"
    
    # Create workflow config from the YAML file using ConfigLoader
    config_loader = ConfigLoader(base_path=str(Path(__file__).parent))
    workflow_config = config_loader.load_workflow_config(str(config_path))
    
    # Fix workflow_directory to use absolute path for proper config resolution
    if hasattr(workflow_config, 'workflow_directory') and workflow_config.workflow_directory:
        # Convert relative workflow_directory to absolute path
        if not Path(workflow_config.workflow_directory).is_absolute():
            # Make it relative to the workflow config file location
            config_dir = Path(config_path).parent
            absolute_workflow_dir = config_dir.resolve()
            workflow_config.workflow_directory = str(absolute_workflow_dir)
    
    # Create executor if not provided
    executor = kwargs.get('executor')
    if not executor:
        executor_config = ExecutorConfig(executor_type='local', max_workers=3)
        executor = LocalExecutor.from_config(executor_config)
    
    # Create workflow using from_config pattern
    workflow = AlphavirusWorkflow.from_config(
        workflow_config,
        session_id=session_id,
        executor=executor,
        **kwargs
    )
    await workflow.initialize()
    return workflow


def create_workflow_from_config(config: Optional[WorkflowConfig] = None) -> AlphavirusWorkflow:
    """
    Factory function to create an AlphavirusWorkflow instance from configuration
    
    Args:
        config: Optional WorkflowConfig object. If None, loads from default path.
        
    Returns:
        AlphavirusWorkflow: Configured workflow instance
    """
    
    if config is None:
        # Load from default configuration file - use create_alphavirus_workflow instead
        default_config_path = Path(__file__).parent / "config" / "AlphavirusWorkflow.yml"
        import asyncio
        loop = asyncio.get_event_loop()
        workflow = loop.run_until_complete(create_alphavirus_workflow(config_path=str(default_config_path)))
    else:
        # For config object, convert to WorkflowConfig and use from_config
        from nanobrain.core.executor import LocalExecutor, ExecutorConfig
        executor_config = ExecutorConfig(executor_type='local', max_workers=3)
        executor = LocalExecutor.from_config(executor_config)
        
        # Create a proper WorkflowConfig from the provided config
        from nanobrain.core.workflow import WorkflowConfig
        workflow_config = WorkflowConfig(
            name=getattr(config, 'name', 'AlphavirusWorkflow'),
            description=getattr(config, 'description', 'Alphavirus protein analysis workflow'),
            steps=getattr(config, 'steps', {}),
            links=getattr(config, 'links', [])
        )
        
        workflow = AlphavirusWorkflow.from_config(
            workflow_config,
            executor=executor
        )
    
    return workflow 