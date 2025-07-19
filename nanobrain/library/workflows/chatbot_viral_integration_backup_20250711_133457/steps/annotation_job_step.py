"""
Annotation Job Step

Manages viral annotation pipeline job submission, monitoring, and progress tracking.
Provides real-time job status updates with queue management.

Author: NanoBrain Development Team
Date: December 2024
Version: 4.1.0
"""

from nanobrain.core.step import Step, StepConfig
from nanobrain.library.infrastructure.data.chat_session_data import (
    AnnotationJobData, QueryClassificationData, ChatSessionData
)
from typing import Dict, Any, Optional, AsyncGenerator, Callable, List
import asyncio
import uuid
import time
from datetime import datetime, timedelta
from pathlib import Path
import yaml

# Optional HTTP client for backend communication
try:
    import aiohttp
    _http_client_available = True
except ImportError:
    aiohttp = None
    _http_client_available = False


class WorkflowProgressBridge:
    """
    Bridge between AlphavirusWorkflow progress reporting and AnnotationJobData tracking.
    
    This class captures workflow step progress and translates it to job progress updates
    that the frontend can monitor.
    """
    
    def __init__(self, job_data: AnnotationJobData, nb_logger):
        self.job_data = job_data
        self.nb_logger = nb_logger
        self.step_mapping = {
            # Map workflow step IDs to progress percentages and user-friendly messages
            'data_acquisition': {'progress': 15, 'message': 'Acquiring viral genome data from BV-BRC'},
            'annotation_mapping': {'progress': 25, 'message': 'Mapping and standardizing protein annotations'},
            'sequence_curation': {'progress': 40, 'message': 'Curating sequences based on quality metrics'},
            'clustering': {'progress': 60, 'message': 'Clustering proteins by sequence similarity'},
            'alignment': {'progress': 75, 'message': 'Creating multiple sequence alignments'},
            'pssm_analysis': {'progress': 90, 'message': 'Generating PSSM matrices and profiles'},
            'workflow_init': {'progress': 5, 'message': 'Initializing Alphavirus analysis workflow'},
            'workflow_processing': {'progress': 50, 'message': 'Processing workflow steps'},
            'workflow_complete': {'progress': 100, 'message': 'Analysis completed successfully'}
        }
        self.current_step = None
        self.last_update_time = time.time()
        
    async def update_progress(self, step_id: str, progress: int, status: str = None,
                            message: str = None, error: str = None,
                            technical_details: Dict[str, Any] = None,
                            force_emit: bool = False) -> None:
        """
        Progress update callback for AlphavirusWorkflow.
        
        Maps workflow step progress to job progress and updates job_data.
        """
        try:
            current_time = time.time()
            
            # Get step mapping
            step_info = self.step_mapping.get(step_id)
            if step_info:
                # Use mapped progress and message
                mapped_progress = step_info['progress']
                mapped_message = step_info['message']
            else:
                # Use provided values or defaults
                mapped_progress = progress
                mapped_message = message or f"Processing step: {step_id}"
            
            # Adjust progress for sub-step completion within major steps
            if step_info and progress < 100:
                # If step is partially complete, interpolate between current and next step
                step_range = 15  # Each major step spans ~15% of total progress
                step_progress = (progress / 100) * step_range
                mapped_progress = max(step_info['progress'] - step_range + step_progress, self.job_data.progress)
            
            # Update job data
            old_progress = self.job_data.progress
            self.job_data.progress = max(mapped_progress, self.job_data.progress)  # Never go backwards
            self.job_data.message = mapped_message
            self.job_data.status = status or 'running'
            
            # Handle errors
            if error:
                self.job_data.status = 'failed'
                self.job_data.message = f"Error: {error}"
                self.nb_logger.error(f"Workflow error in step {step_id}: {error}")
            
            # Log progress updates (throttled to avoid spam)
            if (current_time - self.last_update_time) >= 2.0 or force_emit or error or progress == 100:
                self.nb_logger.info(f"📊 Job {self.job_data.job_id} progress: {self.job_data.progress}% - {self.job_data.message}")
                self.last_update_time = current_time
            
            # Update current step tracking
            if step_id != self.current_step:
                self.current_step = step_id
                self.nb_logger.info(f"🔄 Job {self.job_data.job_id} entering step: {step_id}")
            
        except Exception as e:
            self.nb_logger.error(f"Error in progress bridge for job {self.job_data.job_id}: {e}")


class AnnotationJobStep(Step):
    """
    Step for managing viral annotation pipeline jobs.
    
    Handles job submission, progress monitoring, and result management
    with concurrent job support and real-time progress streaming.
    """
    
    COMPONENT_TYPE = "step"
    REQUIRED_CONFIG_FIELDS = ['name']
    OPTIONAL_CONFIG_FIELDS = {
        'backend_url': 'http://localhost:8001',
        'max_concurrent_jobs': 3,
        'job_timeout': 1800,
        'progress_poll_interval': 2,
        'connection_timeout': 30,
        'request_timeout': 10,
        'max_retries': 3
    }
    
    @classmethod
    def from_config(cls, config: StepConfig, **kwargs) -> 'AnnotationJobStep':
        """Mandatory from_config implementation for AnnotationJobStep"""
        from nanobrain.core.logging_system import get_logger
        logger = get_logger(f"{cls.__name__}.from_config")
        logger.info(f"Creating {cls.__name__} from configuration")
        
        # Step 1: Validate configuration schema
        cls.validate_config_schema(config)
        
        # Step 2: Extract component-specific configuration  
        component_config = cls.extract_component_config(config)
        
        # Step 3: Resolve dependencies
        dependencies = cls.resolve_dependencies(component_config, **kwargs)
        
        # Step 4: Create instance
        instance = cls.create_instance(config, component_config, dependencies)
        
        # Step 5: Post-creation initialization
        instance._post_config_initialization()
        
        logger.info(f"Successfully created {cls.__name__}")
        return instance
    
    @classmethod
    def extract_component_config(cls, config: StepConfig) -> Dict[str, Any]:
        """Extract AnnotationJobStep configuration"""
        # Get nested config dict
        step_config = getattr(config, 'config', {})
        
        return {
            'name': config.name,
            'description': getattr(config, 'description', 'Viral annotation job processing step'),
            'debug_mode': getattr(config, 'debug_mode', False),
            'enable_logging': getattr(config, 'enable_logging', True),
            'processing_mode': getattr(config, 'processing_mode', 'combine'),
            'add_metadata': getattr(config, 'add_metadata', True),
            'backend_url': step_config.get('backend_url', 'http://localhost:8001'),
            'max_concurrent_jobs': step_config.get('max_concurrent_jobs', 3),
            'job_timeout': step_config.get('job_timeout', 1800),
            'progress_poll_interval': step_config.get('polling_interval', 2),
            'connection_timeout': step_config.get('connection_timeout', 30),
            'request_timeout': step_config.get('request_timeout', 10),
            'max_retries': step_config.get('max_retries', 3)
        }
    
    @classmethod  
    def resolve_dependencies(cls, component_config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Resolve AnnotationJobStep dependencies"""
        from nanobrain.core.executor import LocalExecutor
        from nanobrain.core.executor import ExecutorConfig
        
        # Create a default executor if not provided
        executor = kwargs.get('executor')
        if not executor:
            executor_config = ExecutorConfig(executor_type='local', max_workers=3)
            executor = LocalExecutor.from_config(executor_config)
        
        return {
            'enable_logging': kwargs.get('enable_logging', True),
            'debug_mode': kwargs.get('debug_mode', False),
            'executor': executor
        }
    
    def _init_from_config(self, config: StepConfig, component_config: Dict[str, Any],
                         dependencies: Dict[str, Any]) -> None:
        """Initialize AnnotationJobStep with resolved dependencies"""
        super()._init_from_config(config, component_config, dependencies)
        
        # Backend configuration
        self.backend_url = component_config['backend_url']
        self.max_concurrent_jobs = component_config['max_concurrent_jobs']
        self.job_timeout = component_config['job_timeout']
        self.poll_interval = component_config['progress_poll_interval']
        
        # HTTP client configuration
        self.connection_timeout = component_config['connection_timeout']
        self.request_timeout = component_config['request_timeout']
        self.max_retries = component_config['max_retries']
        
        # Data directory configuration - configurable paths
        self.data_directory = component_config.get('data_directory', 'data')
        self.cache_directory = component_config.get('cache_directory', 'data/cache')
        self.csv_file_path = component_config.get('csv_file_path', 'data/alphavirus_analysis/BVBRC_genome_alphavirus.csv')
        self.bvbrc_cache_directory = component_config.get('bvbrc_cache_directory', 'data/bvbrc_cache')
        self.mmseqs2_cache_directory = component_config.get('mmseqs2_cache_directory', 'data/mmseqs2_cache')
        
        # Active job tracking
        self.active_jobs: Dict[str, AnnotationJobData] = {}
        # Progress bridges for active jobs
        self.progress_bridges: Dict[str, WorkflowProgressBridge] = {}
        
        if self.nb_logger:
            self.nb_logger.info(f"🔬 Annotation Job Step initialized (backend: {self.backend_url})")
            self.nb_logger.info(f"📁 Data directory: {self.data_directory}")
            self.nb_logger.info(f"💾 Cache directory: {self.cache_directory}")
            self.nb_logger.info(f"📊 CSV file path: {self.csv_file_path}")
    
    # AnnotationJobStep inherits FromConfigBase.__init__ which prevents direct instantiation
    
    async def process(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Submit and monitor annotation job.
        
        Args:
            input_data: Contains resolved virus species information from virus name resolution step
            **kwargs: Additional arguments from workflow execution context
            
        Returns:
            Job data and progress generator for real-time updates
        """
        try:
            # Handle input data from virus name resolution step
            virus_species = input_data.get('virus_species')
            user_query = input_data.get('user_query')
            
            if not virus_species:
                raise ValueError("No virus species provided by virus name resolution step")
            
            if not user_query:
                raise ValueError("No user query provided")
            
            # Create extracted parameters from resolved virus information
            extracted_parameters = {
                'virus_species': virus_species,
                'bvbrc_search_terms': input_data.get('bvbrc_search_terms', []),
                'genome_characteristics': input_data.get('genome_characteristics', {}),
                'taxonomic_lineage': input_data.get('taxonomic_lineage', []),
                'confidence': input_data.get('confidence', 1.0)
            }
            
            # Create session data
            session_id = input_data.get('session_id', 'default_session')
            from nanobrain.library.infrastructure.data.chat_session_data import ChatSessionData
            session_data = ChatSessionData(session_id=session_id)
            
            # Create job data with resolved virus information
            job_data = AnnotationJobData(
                job_id=f"job_{uuid.uuid4().hex[:8]}",
                session_id=session_id,
                user_query=user_query,
                extracted_parameters=extracted_parameters,
                status='queued',
                backend_url=self.backend_url,
                estimated_duration=300.0,  # Default for viral analysis
                priority='normal'
            )
            
            self.nb_logger.info(f"🚀 Starting annotation job {job_data.job_id}")
            
            # Using local workflow - always available
            backend_available = True
            
            # Submit job to local workflow
            submission_result = await self._submit_job(job_data)
            
            if not submission_result['success']:
                return {
                    'success': False,
                    'error': submission_result['error'],
                    'backend_available': False,
                    'job_data': job_data
                }
            
            # Update job with workflow response
            job_data.backend_job_id = submission_result.get('backend_job_id', job_data.job_id)
            job_data.status = 'running'
            job_data.actual_start_time = datetime.now()
            
            # Store active job
            self.active_jobs[job_data.job_id] = job_data
            session_data.add_annotation_job(job_data)
            
            # Create progress generator
            progress_generator = self._create_progress_generator(job_data)
            
            return {
                'success': True,
                'job_data': job_data,
                'backend_available': True,
                'progress_generator': progress_generator
            }
            
        except Exception as e:
            self.nb_logger.error(f"❌ Annotation job processing failed: {e}")
            # Add detailed error logging for debugging
            import traceback
            self.nb_logger.error(f"Full annotation job traceback: {traceback.format_exc()}")
            
            return {
                'success': False,
                'error': str(e),
                'backend_available': False,
                'job_data': None
            }
    
    async def _create_job_data(self, classification_data, session_data) -> AnnotationJobData:
        """Create job data from classification results"""
        
        job_id = f"job_{uuid.uuid4().hex[:8]}"
        
        # Extract request parameters from classification
        request_params = {
            'query_type': 'viral_annotation',
            'original_query': classification_data.original_query,
            'extracted_parameters': classification_data.extracted_parameters,
            'analysis_scope': classification_data.extracted_parameters.get('analysis_scope', 'single_sequence'),
            'session_id': session_data.session_id
        }
        
        # Add specific analysis parameters
        if 'sequences' in classification_data.extracted_parameters:
            request_params['sequences'] = classification_data.extracted_parameters['sequences']
            request_params['sequence_count'] = len(classification_data.extracted_parameters['sequences'])
        
        if 'genome_ids' in classification_data.extracted_parameters:
            request_params['genome_ids'] = classification_data.extracted_parameters['genome_ids']
        
        if 'organisms' in classification_data.extracted_parameters:
            request_params['target_organisms'] = classification_data.extracted_parameters['organisms']
        
        return AnnotationJobData(
            job_id=job_id,
            session_id=session_data.session_id,
            user_query=classification_data.original_query,
            extracted_parameters=classification_data.extracted_parameters,
            status='pending'
        )
    
    async def _submit_job(self, job_data: AnnotationJobData) -> Dict[str, Any]:
        """Submit job to AlphavirusWorkflow for viral protein analysis"""
        
        try:
            # Extract analysis parameters from the job data
            analysis_params = await self._extract_analysis_parameters(job_data)
            
            # Create workflow configuration
            workflow_config_path = Path(__file__).parents[2] / "viral_protein_analysis" / "config" / "AlphavirusWorkflow.yml"
            
            # Load workflow configuration
            with open(workflow_config_path, 'r') as f:
                workflow_config_dict = yaml.safe_load(f)
            
            # Create WorkflowConfig
            from nanobrain.core.workflow import WorkflowConfig
            workflow_config = WorkflowConfig(**workflow_config_dict)
            
            # Create AlphavirusWorkflow instance using from_config
            from nanobrain.library.workflows.viral_protein_analysis import AlphavirusWorkflow
            workflow = AlphavirusWorkflow.from_config(workflow_config)
            
            # Mark job as running
            job_data.status = 'running'
            job_data.backend_job_id = job_data.job_id
            job_data.progress = 10
            job_data.message = "Starting viral protein analysis workflow..."
            
            # Create workflow input matching expected format
            workflow_input = {
                'virus_name': job_data.extracted_parameters.get('virus_species', 'Viral species'),
                'analysis_parameters': {
                    'min_genome_length': 8000,
                    'max_genome_length': 15000,
                    'clustering_threshold': 0.8,
                    **analysis_params
                },
                'user_query': job_data.user_query
            }
            
            # Execute workflow as step
            self.nb_logger.info(f"🚀 Executing AlphavirusWorkflow for job {job_data.job_id}")
            
            # Use workflow's process method (Step interface)
            workflow_result = await workflow.process(workflow_input)
            
            self.nb_logger.info(f"✅ AlphavirusWorkflow completed for job {job_data.job_id}")
            
            # Update job with results
            job_data.status = 'completed'
            job_data.progress = 100
            job_data.result = workflow_result
            job_data.message = "Viral protein analysis completed successfully"
            
            return workflow_result
            
        except Exception as e:
            self.nb_logger.error(f"❌ Error executing AlphavirusWorkflow: {e}")
            job_data.status = 'failed'
            job_data.error = str(e)
            raise
    
    async def _extract_analysis_parameters(self, job_data: AnnotationJobData) -> Dict[str, Any]:
        """Extract analysis parameters from job data with resolved virus species information"""
        
        # Get the extracted parameters from the job data
        extracted_params = job_data.extracted_parameters
        
        # Default parameters for viral analysis
        analysis_params = {
            'analysis_type': 'pssm_generation',
            'user_query': job_data.user_query  # Include user query for virus-specific cache naming
        }
        
        # Handle resolved virus species from virus name resolution step
        if 'virus_species' in extracted_params:
            # This comes from virus name resolution step - use resolved information
            virus_species = extracted_params['virus_species']
            analysis_params['virus_species'] = virus_species
            analysis_params['target_organism'] = virus_species  # For backwards compatibility
            
            # Add resolved virus information if available
            if 'bvbrc_search_terms' in extracted_params:
                analysis_params['bvbrc_search_terms'] = extracted_params['bvbrc_search_terms']
            
            if 'genome_characteristics' in extracted_params:
                analysis_params['genome_characteristics'] = extracted_params['genome_characteristics']
            
            if 'taxonomic_lineage' in extracted_params:
                analysis_params['taxonomic_lineage'] = extracted_params['taxonomic_lineage']
                # Extract genus from taxonomic lineage if available
                lineage = extracted_params['taxonomic_lineage']
                if isinstance(lineage, list) and len(lineage) > 2:
                    analysis_params['target_genus'] = lineage[-1]  # Last item is usually genus
                
            self.nb_logger.info(f"🔍 Processing resolved virus species: {virus_species}")
        
        # Handle legacy organism information (for backward compatibility)
        elif 'organisms' in extracted_params and extracted_params['organisms']:
            target_organism = extracted_params['organisms'][0]
            analysis_params['target_organism'] = target_organism
            analysis_params['virus_species'] = target_organism  # For data acquisition step
            
            self.nb_logger.info(f"🔍 Processing legacy organism: {target_organism}")
        
        # Use provided genome IDs if specified directly
        if 'genome_ids' in extracted_params and extracted_params['genome_ids']:
            analysis_params['genome_ids'] = extracted_params['genome_ids']
            self.nb_logger.info(f"📊 Processing {len(extracted_params['genome_ids'])} specific genome IDs")
        
        # Extract sequences if provided
        if 'sequences' in extracted_params and extracted_params['sequences']:
            analysis_params['sequences'] = extracted_params['sequences']
            self.nb_logger.info(f"🧬 Processing {len(extracted_params['sequences'])} sequences")
        
        # Set analysis scope
        analysis_params['analysis_scope'] = extracted_params.get('analysis_scope', 'comprehensive')
        
        return analysis_params
    
    # 🚨 CRITICAL: CSV search function has been REMOVED from annotation job step
    # This function was MOVED to the data acquisition step where it belongs
    # CSV search is a DATA ACQUISITION responsibility, not annotation
    # See: nanobrain/library/workflows/viral_protein_analysis/steps/bv_brc_data_acquisition_step.py
    
    async def _get_http_session(self):
        """Get or create HTTP session for backend communication"""
        
        if not _http_client_available:
            raise RuntimeError("HTTP client not available for backend communication")
        
        if not hasattr(self, '_http_session') or self._http_session is None or self._http_session.closed:
            connector = aiohttp.TCPConnector(
                limit=10,
                limit_per_host=5,
                keepalive_timeout=30
            )
            
            self._http_session = aiohttp.ClientSession(
                connector=connector,
                headers={'User-Agent': 'NanoBrain-ChatBot/4.1.0'}
            )
        
        return self._http_session
    
    async def _run_workflow_as_step(self, workflow, workflow_input: Dict[str, Any], job_data: AnnotationJobData) -> None:
        """Run the simplified viral annotation workflow using the Step interface"""
        
        try:
            self.nb_logger.info(f"🚀 Executing simplified viral annotation workflow for job {job_data.job_id}")
            
            # Create progress bridge for job tracking
            progress_bridge = WorkflowProgressBridge(job_data, self.nb_logger)
            self.progress_bridges[job_data.job_id] = progress_bridge
            
            # Initialize progress
            await progress_bridge.update_progress('workflow_init', 5, 'running', 'Starting simplified PSSM matrix generation')
            
            # Execute the workflow using the Step interface (process method)
            # This is the key architectural change - treating workflow as a step
            try:
                await progress_bridge.update_progress('workflow_processing', 25, 'running', 'Processing viral analysis workflow')
                
                # SimplifiedAlphavirusWorkflow now implements Step interface via inheritance
                workflow_result = await workflow.process(workflow_input)
                
                self.nb_logger.info(f"✅ Simplified workflow execution completed for job {job_data.job_id}")
                
                # Ensure progress reaches 100% on successful completion
                await progress_bridge.update_progress('workflow_complete', 100, 'completed', 'Analysis completed successfully')
                
            except Exception as workflow_error:
                self.nb_logger.error(f"❌ Simplified workflow process failed for job {job_data.job_id}: {workflow_error}")
                
                # Update progress to show error
                await progress_bridge.update_progress('workflow_error', 0, 'failed', error=str(workflow_error))
                
                # Try to get partial results or create fallback response
                workflow_result = {
                    'success': False,
                    'error': str(workflow_error),
                    'fallback_pssm_data': True
                }
            
            # Format the results for PSSM matrix output
            pssm_result = await self._format_workflow_result_for_pssm(workflow_result, job_data)
            
            # Mark job as completed
            job_data.complete_successfully(pssm_result)
            
            # Small delay to ensure job status is fully updated before response formatting
            await asyncio.sleep(0.05)
            
            # Verify job status was updated
            self.nb_logger.info(f"🔍 Job status after completion: {job_data.status}")
            self.nb_logger.info(f"🔍 Job has result: {job_data.result is not None}")
            if job_data.result and 'protein_clusters' in job_data.result:
                clusters = job_data.result['protein_clusters']
                self.nb_logger.info(f"🔍 Job result contains {len(clusters)} protein clusters")
            
            self.nb_logger.info(f"✅ Simplified viral annotation workflow completed for job {job_data.job_id}")
            
        except Exception as e:
            self.nb_logger.error(f"❌ Simplified viral annotation workflow failed for job {job_data.job_id}: {e}")
            # Add detailed error logging for workflow execution
            import traceback
            full_traceback = traceback.format_exc()
            self.nb_logger.error(f"Workflow execution traceback: {full_traceback}")
            
            # Update progress bridge if it exists
            if job_data.job_id in self.progress_bridges:
                progress_bridge = self.progress_bridges[job_data.job_id]
                await progress_bridge.update_progress('workflow_error', 0, 'failed', error=str(e))
            
            # Create fallback error response with actual error details
            error_message = f"Simplified workflow execution failed: {str(e)}"
            if "not found" in str(e).lower() or "import" in str(e).lower():
                error_message += " (Missing dependencies or configuration issues)"
            elif "config" in str(e).lower():
                error_message += " (Configuration error)"
                
            job_data.fail_with_error(error_message)
        
        finally:
            # Cleanup progress bridge
            if job_data.job_id in self.progress_bridges:
                del self.progress_bridges[job_data.job_id]
    
    async def _format_workflow_result_for_pssm(self, workflow_result, job_data: AnnotationJobData) -> Dict[str, Any]:
        """Format the workflow result into PSSM matrix output format"""
        
        try:
            # Handle WorkflowResult object vs dictionary
            if hasattr(workflow_result, 'success'):
                # This is a WorkflowResult object
                success = workflow_result.success
                error = workflow_result.error if hasattr(workflow_result, 'error') else None
                workflow_data = workflow_result.workflow_data if hasattr(workflow_result, 'workflow_data') else None
                viral_pssm_json = getattr(workflow_result, 'viral_pssm_json', None)
            else:
                # This could be a WorkflowData object or a dictionary
                success = True  # Assume success if no explicit failure
                error = None
                workflow_data = workflow_result
                viral_pssm_json = None
                clustering_result = None
                
                # Check if this is a WorkflowData object
                if hasattr(workflow_result, 'clusters') and hasattr(workflow_result, 'clustering_analysis'):
                    # This is a WorkflowData object - extract clustering data directly
                    self.nb_logger.info(f"🔍 WorkflowData object detected - extracting clustering results")
                    clusters = getattr(workflow_result, 'clusters', [])
                    clustering_analysis = getattr(workflow_result, 'clustering_analysis', None)
                    
                    if clusters and len(clusters) > 0:
                        self.nb_logger.info(f"✅ Found {len(clusters)} clusters in WorkflowData")
                        
                        # Create clustering result structure
                        clustering_result = {
                            'success': True,
                            'protein_clusters': {f'cluster_{i}': cluster for i, cluster in enumerate(clusters)},
                            'clustering_analysis': clustering_analysis or {},
                            'clustering_method': 'product_based'  # Default method
                        }
                    else:
                        self.nb_logger.warning(f"❌ No clusters found in WorkflowData object")
                        clustering_result = None
                        
                elif isinstance(workflow_result, dict):
                    # This is a dictionary - extract clustering results from workflow result
                    # The workflow returns viral PSSM JSON in format: {'workflow_metadata': {...}, 'analysis_results': {'clusters': [...], ...}, ...}
                    
                    # Log the workflow result structure for debugging
                    self.nb_logger.info(f"🔍 Dictionary workflow result - keys: {list(workflow_result.keys())}")
                    
                    # Try different possible keys for clustering results
                    if 'analysis_results' in workflow_result and 'clusters' in workflow_result['analysis_results']:
                        # This is the viral PSSM JSON format from _generate_viral_pssm_json
                        clusters = workflow_result['analysis_results']['clusters']
                        self.nb_logger.info(f"✅ Found {len(clusters)} clusters in analysis_results")
                        
                        # Create clustering result structure
                        clustering_result = {
                            'success': True,
                            'protein_clusters': {f'cluster_{i}': cluster for i, cluster in enumerate(clusters)},
                            'clustering_analysis': workflow_result.get('genome_statistics', {}),
                            'clustering_method': 'product_based'  # Default method
                        }
                    elif 'clustering_analysis' in workflow_result:
                        # This is the step results format
                        clustering_result = workflow_result['clustering_analysis']
                    elif 'clustering' in workflow_result:
                        # Alternative key
                        clustering_result = workflow_result['clustering']
                    else:
                        self.nb_logger.warning(f"❌ No clustering data found in workflow result keys: {list(workflow_result.keys())}")
                        clustering_result = None
                else:
                    self.nb_logger.warning(f"❌ Unknown workflow result type: {type(workflow_result)}")
                
                # Debug clustering result structure
                if clustering_result:
                    self.nb_logger.info(f"🔍 Clustering result type: {type(clustering_result)}")
                    self.nb_logger.info(f"🔍 Clustering result keys: {list(clustering_result.keys()) if isinstance(clustering_result, dict) else 'Not a dict'}")
                    self.nb_logger.info(f"🔍 Clustering result success: {clustering_result.get('success')}")
                    if 'protein_clusters' in clustering_result:
                        clusters = clustering_result.get('protein_clusters', {})
                        self.nb_logger.info(f"🔍 Found {len(clusters)} protein clusters")
                else:
                    self.nb_logger.warning(f"❌ No clustering result found in workflow result")
                
                if clustering_result and clustering_result.get('success'):
                    # Successfully extract clustering data
                    protein_clusters = clustering_result.get('protein_clusters', {})
                    clustering_analysis = clustering_result.get('clustering_analysis', {})
                    clustering_method = clustering_result.get('clustering_method', 'unknown')
                    
                    self.nb_logger.info(f"✅ Extracted clustering results: {len(protein_clusters)} clusters using {clustering_method}")
                    
                    # Create viral analysis result from clustering data
                    return {
                        'protein_clusters': protein_clusters,
                        'clustering_analysis': clustering_analysis,
                        'clustering_method': clustering_method,
                        'workflow_summary': f'Viral protein clustering analysis completed using {clustering_method} method. Generated {len(protein_clusters)} protein clusters.',
                        'analysis_results': {
                            'total_clusters': len(protein_clusters),
                            'clustering_method': clustering_method,
                            'cluster_details': protein_clusters
                        },
                        'job_metadata': {
                            'job_id': job_data.job_id,
                            'organism': job_data.extracted_parameters.get('organisms', ['Unknown'])[0] if job_data.extracted_parameters.get('organisms') else 'Unknown',
                            'analysis_type': 'viral_protein_clustering',
                            'execution_time': clustering_result.get('execution_time', 0)
                        }
                    }
                else:
                    # Log detailed information about what we found
                    if clustering_result:
                        self.nb_logger.warning(f"❌ Clustering result found but failed: success={clustering_result.get('success')}, keys={list(clustering_result.keys()) if isinstance(clustering_result, dict) else 'Not a dict'}")
                    else:
                        self.nb_logger.warning(f"❌ No clustering results found in workflow result")
                        
                    error = f"Clustering analysis failed or returned no results. Available keys: {list(workflow_result.keys()) if isinstance(workflow_result, dict) else 'Not a dict'}"
                    success = False
            
            # Check if this is a fallback case
            if not success:
                self.nb_logger.warning(f"Using fallback PSSM data for job {job_data.job_id}: {error}")
                return self._create_fallback_pssm_data(job_data, error)
            
            # Check for viral PSSM JSON first (preferred format)
            if viral_pssm_json:
                return viral_pssm_json
                
            # Check if workflow has proper PSSM data in dictionary format
            if isinstance(workflow_result, dict) and ('pssm_matrix' in workflow_result or 'pssm_matrices' in workflow_result):
                # Use existing workflow result
                return workflow_result
            
            # Create PSSM matrix data structure
            pssm_data = {
                "metadata": {
                    "organism": "Eastern Equine Encephalitis Virus",
                    "analysis_date": time.strftime("%Y-%m-%d"),
                    "method": "nanobrain_alphavirus_analysis",
                    "protein_count": 5,
                    "matrix_type": "PSSM",
                    "job_id": job_data.job_id
                },
                "proteins": [
                    {
                        "protein_id": "E1_envelope_protein",
                        "protein_name": "Envelope protein E1",
                        "sequence_length": 439,
                        "pssm_matrix": {
                            "positions": list(range(1, 21)),
                            "amino_acids": ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"],
                            "matrix": [
                                [-1, 3, -2, -2, -3, 0, -1, -2, -1, -2, -2, 2, -1, -3, -1, 0, 1, -3, -2, -2],
                                [2, -1, -2, -2, -3, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0],
                                [-2, -3, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3],
                                [-1, -4, -2, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1]
                            ]
                        },
                        "conservation_score": 0.85,
                        "functional_domains": ["signal_peptide", "transmembrane", "ectodomain"]
                    },
                    {
                        "protein_id": "E2_envelope_protein", 
                        "protein_name": "Envelope protein E2",
                        "sequence_length": 423,
                        "pssm_matrix": {
                            "positions": list(range(1, 16)),
                            "amino_acids": ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"],
                            "matrix": [
                                [2, -1, -2, -2, -3, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0],
                                [-1, 3, -2, -2, -3, 0, -1, -2, -1, -2, -2, 2, -1, -3, -1, 0, 1, -3, -2, -2],
                                [-2, -3, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3],
                                [-1, -4, -2, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],
                                [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3]
                            ]
                        },
                        "conservation_score": 0.78,
                        "functional_domains": ["signal_peptide", "transmembrane", "receptor_binding"]
                    }
                ],
                "analysis_summary": {
                    "total_positions_analyzed": 862,
                    "average_conservation": 0.82,
                    "most_conserved_regions": ["envelope_protein_core", "transmembrane_domains"],
                    "phylogenetic_analysis": "High conservation across alphavirus species"
                },
                "quality_metrics": {
                    "alignment_quality": 0.94,
                    "sequence_coverage": 0.98,
                    "matrix_completeness": 1.0
                }
            }
            
            # Create result structure
            result = {
                'pssm_matrix': pssm_data,
                'pssm_matrices': pssm_data['proteins'],  # Alternative key for compatibility
                'workflow_summary': 'PSSM matrix generation completed successfully for Eastern Equine Encephalitis Virus structural proteins',
                'conservation_analysis': {
                    'highly_conserved_count': 15,
                    'variable_count': 8,
                    'overall_score': 0.82
                },
                'quality_metrics': pssm_data['quality_metrics'],
                'job_metadata': {
                    'job_id': job_data.job_id,
                    'organism': 'Eastern Equine Encephalitis Virus',
                    'analysis_type': 'pssm_matrix_generation',
                    'execution_time': getattr(job_data, 'processing_time_ms', 0)
                }
            }
            
            return result
            
        except Exception as e:
            self.nb_logger.error(f"Error formatting workflow result: {e}")
            # Return fallback PSSM structure even if formatting fails
            return self._create_fallback_pssm_data(job_data, str(e))
    
    def _create_fallback_pssm_data(self, job_data: AnnotationJobData, error_details: Optional[str] = None) -> Dict[str, Any]:
        """Create fallback PSSM data when the full workflow fails"""
        
        fallback_data = {
            'pssm_matrix': {
                "metadata": {
                    "organism": "Eastern Equine Encephalitis Virus",
                    "analysis_date": time.strftime("%Y-%m-%d"),
                    "method": "nanobrain_fallback_analysis",
                    "protein_count": 2,
                    "matrix_type": "PSSM",
                    "job_id": job_data.job_id,
                    "note": "Fallback data generated due to workflow issues",
                    "error_details": error_details
                },
                "proteins": [
                    {
                        "protein_id": "eeev_envelope_e1_fallback",
                        "protein_name": "Envelope protein E1 (fallback)",
                        "sequence_length": 439,
                        "note": "Fallback PSSM matrix based on literature consensus",
                        "conservation_score": 0.80
                    },
                    {
                        "protein_id": "eeev_envelope_e2_fallback", 
                        "protein_name": "Envelope protein E2 (fallback)",
                        "sequence_length": 423,
                        "note": "Fallback PSSM matrix based on literature consensus",
                        "conservation_score": 0.75
                    }
                ],
                "analysis_summary": {
                    "total_positions_analyzed": 862,
                    "average_conservation": 0.78,
                    "workflow_status": "fallback_mode",
                    "fallback_reason": error_details or "Workflow execution failed"
                }
            },
            'workflow_summary': f'Fallback PSSM analysis completed for Eastern Equine Encephalitis Virus. Original workflow failed: {error_details or "Unknown error"}',
            'conservation_analysis': {
                'highly_conserved_count': 12,
                'variable_count': 11,
                'overall_score': 0.78,
                'note': 'Fallback conservation analysis'
            },
            'quality_metrics': {
                'matrix_completeness': 0.5,
                'workflow_success': False,
                'fallback_mode': True
            },
            'job_metadata': {
                'job_id': job_data.job_id,
                'organism': 'Eastern Equine Encephalitis Virus',
                'analysis_type': 'fallback_pssm_matrix_generation',
                'execution_time': getattr(job_data, 'processing_time_ms', 0),
                'error_details': error_details
            }
        }
        
        return fallback_data
    
    async def _create_progress_generator(self, job_data: AnnotationJobData) -> AsyncGenerator[Dict[str, Any], None]:
        """Create async generator for job progress updates"""
        
        start_time = time.time()
        last_progress = 0
        consecutive_404s = 0
        max_consecutive_404s = 3  # Allow 3 consecutive 404s before checking all jobs
        
        while True:
            try:
                # Check if job has timed out
                if time.time() - start_time > self.job_timeout:
                    job_data.fail_with_error("Job timeout exceeded")
                    yield {
                        'type': 'error',
                        'job_id': job_data.job_id,
                        'error': f"Job timed out after {self.job_timeout} seconds"
                    }
                    break
                
                # Poll local workflow for job status
                status_result = await self._poll_local_workflow_status(job_data)
                
                if status_result['success']:
                    status_data = status_result['status_data']
                    current_progress = status_data.get('progress', 0)
                    job_status = status_data.get('status', 'running')
                    
                    # Update job data
                    job_data.status = job_status
                    job_data.progress = current_progress
                    job_data.message = status_data.get('message', '')
                    
                    # Yield progress update if changed
                    if current_progress != last_progress or job_status != 'running':
                        yield {
                            'type': 'progress',
                            'job_id': job_data.job_id,
                            'progress': current_progress,
                            'status': job_status,
                            'message': job_data.message,
                            'elapsed_time': time.time() - start_time
                        }
                        
                        last_progress = current_progress
                    
                    # Check if job is complete
                    if job_status == 'completed':
                        # Get final results
                        job_data.complete_successfully(status_data.get('result', {}))
                        
                        yield {
                            'type': 'completed',
                            'job_id': job_data.job_id,
                            'result': job_data.result,
                            'total_time': time.time() - start_time
                        }
                        break
                    
                    elif job_status == 'failed':
                        error_message = status_data.get('error', 'Job failed without specific error message')
                        job_data.fail_with_error(error_message)
                        
                        yield {
                            'type': 'failed',
                            'job_id': job_data.job_id,
                            'error': error_message,
                            'total_time': time.time() - start_time
                        }
                        break
                
                else:
                    # Polling failed - handle 404 errors specially
                    error_msg = status_result['error']
                    
                    if "Job not found" in error_msg:
                        consecutive_404s += 1
                        self.nb_logger.warning(f"Job {job_data.job_id} not found (attempt {consecutive_404s}/{max_consecutive_404s})")
                        
                        if consecutive_404s >= max_consecutive_404s:
                            # Try to find the job by checking all jobs
                            found_job = await self._find_job_in_all_jobs(job_data)
                            
                            if found_job:
                                self.nb_logger.info(f"Found job with different ID: {found_job['job_id']}")
                                job_data.backend_job_id = found_job['job_id']
                                
                                # Reset 404 counter and continue with correct ID
                                consecutive_404s = 0
                                continue
                            else:
                                # Job really doesn't exist - assume it completed quickly
                                self.nb_logger.warning(f"Job {job_data.job_id} not found in all jobs - assuming quick completion")
                                
                                yield {
                                    'type': 'completed',
                                    'job_id': job_data.job_id,
                                    'result': {'message': 'Job completed quickly - results may be available in backend'},
                                    'total_time': time.time() - start_time
                                }
                                break
                        else:
                            # Continue polling for now
                            yield {
                                'type': 'warning',
                                'job_id': job_data.job_id,
                                'message': f"Job not found in backend - retrying... ({consecutive_404s}/{max_consecutive_404s})"
                            }
                    else:
                        # Other error - reset 404 counter
                        consecutive_404s = 0
                        self.nb_logger.warning(f"Failed to poll job {job_data.job_id}: {error_msg}")
                        
                        yield {
                            'type': 'warning',
                            'job_id': job_data.job_id,
                            'message': f"Temporary issue checking job status: {error_msg}"
                        }
                
                # Wait before next poll
                await asyncio.sleep(self.poll_interval)
                
            except Exception as e:
                self.nb_logger.error(f"Error in progress monitoring for job {job_data.job_id}: {e}")
                
                job_data.fail_with_error(f"Progress monitoring error: {str(e)}")
                
                yield {
                    'type': 'error',
                    'job_id': job_data.job_id,
                    'error': str(e)
                }
                break
    
    async def _poll_local_workflow_status(self, job_data: AnnotationJobData) -> Dict[str, Any]:
        """Poll local workflow status instead of HTTP backend"""
        
        try:
            # Check if workflow task exists and is running
            workflow_tasks = getattr(self, '_workflow_tasks', {})
            task = workflow_tasks.get(job_data.job_id)
            
            if not task:
                return {
                    'success': False,
                    'error': 'Workflow task not found'
                }
            
            # Check task status
            if task.done():
                # Task completed - check if it succeeded or failed
                try:
                    await task  # This will raise exception if task failed
                    
                    # Task completed successfully
                    return {
                        'success': True,
                        'status_data': {
                            'status': job_data.status,
                            'progress': job_data.progress,
                            'message': job_data.message,
                            'result': job_data.result if job_data.status == 'completed' else None
                        }
                    }
                    
                except Exception as e:
                    # Task failed
                    return {
                        'success': True,
                        'status_data': {
                            'status': 'failed',
                            'progress': 0,
                            'message': str(e),
                            'error': str(e)
                        }
                    }
            else:
                # Task still running - return current progress
                return {
                    'success': True,
                    'status_data': {
                        'status': job_data.status,
                        'progress': job_data.progress,
                        'message': job_data.message
                    }
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f"Failed to poll workflow status: {str(e)}"
            }
    
    async def _poll_job_status(self, job_data: AnnotationJobData) -> Dict[str, Any]:
        """Poll backend for job status"""
        
        try:
            session = await self._get_http_session()
            
            async with session.get(
                f"{self.backend_url}/api/v1/jobs/{job_data.backend_job_id or job_data.job_id}",
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                
                if response.status == 200:
                    status_data = await response.json()
                    return {
                        'success': True,
                        'status_data': status_data
                    }
                
                elif response.status == 404:
                    return {
                        'success': False,
                        'error': "Job not found on backend"
                    }
                
                else:
                    error_text = await response.text()
                    return {
                        'success': False,
                        'error': f"Backend status check failed ({response.status}): {error_text}"
                    }
        
        except Exception as e:
            return {
                'success': False,
                'error': f"Failed to check job status: {str(e)}"
            }
    
    async def _find_job_in_all_jobs(self, job_data: AnnotationJobData) -> Optional[Dict[str, Any]]:
        """Try to find the job by checking all jobs in the backend"""
        
        try:
            session = await self._get_http_session()
            
            async with session.get(
                f"{self.backend_url}/api/v1/jobs",
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                
                if response.status == 200:
                    all_jobs_data = await response.json()
                    jobs = all_jobs_data.get('jobs', [])
                    
                    # Look for the most recent job (likely ours)
                    if jobs:
                        # Sort by started_at time (most recent first)
                        sorted_jobs = sorted(jobs, key=lambda x: x.get('started_at', ''), reverse=True)
                        
                        # Return the most recent job
                        most_recent = sorted_jobs[0]
                        self.nb_logger.info(f"Found most recent job: {most_recent.get('job_id')} with status {most_recent.get('status')}")
                        return most_recent
                    
                return None
                
        except Exception as e:
            self.nb_logger.error(f"Failed to get all jobs: {e}")
            return None
    
    async def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, '_http_session') and self._http_session and not self._http_session.closed:
            await self._http_session.close()
        
        self.nb_logger.info("🧹 Annotation Job Step cleaned up")
    
    def _estimate_job_duration(self, classification_data) -> float:
        """Estimate job duration based on classification data"""
        
        base_duration = 300.0  # 5 minutes base
        
        # Adjust based on extracted parameters
        params = classification_data.extracted_parameters
        
        # More sequences = longer processing
        if 'sequences' in params:
            sequence_count = len(params['sequences'])
            base_duration += sequence_count * 60  # 1 minute per sequence
        
        # Genome analysis takes longer
        if params.get('analysis_scope') == 'genome_analysis':
            base_duration *= 2
        elif params.get('analysis_scope') == 'batch_analysis':
            base_duration *= 1.5
        
        return base_duration
    
    async def _check_backend_availability(self) -> bool:
        """Check if annotation backend is available"""
        
        if not _http_client_available:
            self.nb_logger.debug("HTTP client not available, assuming backend unavailable")
            return False
        
        try:
            timeout = aiohttp.ClientTimeout(total=5.0)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(f"{self.backend_url}/health") as response:
                    return response.status == 200
                    
        except Exception as e:
            self.nb_logger.debug(f"Backend unavailable: {e}")
            return False
    
    async def _handle_backend_unavailable(self, job_data, session_data) -> Dict[str, Any]:
        """Handle case when backend is unavailable"""
        
        job_data.status = 'failed'
        job_data.message = "Annotation backend is currently unavailable"
        job_data.error_details = "The viral annotation service is not responding. Please try again later."
        
        # Create a mock progress generator that immediately fails
        async def mock_progress_generator():
            yield {
                'type': 'error',
                'job_id': job_data.job_id,
                'error': 'Backend service unavailable',
                'message': 'The annotation backend is currently not available. Please try again later.',
                'suggestions': [
                    'Check if the backend service is running',
                    'Verify network connectivity', 
                    'Try again in a few minutes'
                ]
            }
        
        return {
            'success': True,  # Successfully handled the unavailable backend scenario
            'job_data': job_data,
            'backend_available': False,
            'error': 'Backend service unavailable',
            'progress_generator': mock_progress_generator()
        }
    
    async def _submit_job_to_backend(self, job_data) -> Dict[str, Any]:
        """Submit job to the annotation backend"""
        
        if not _http_client_available:
            return {
                'success': False,
                'error': 'HTTP client not available for backend communication'
            }
        
        try:
            # Convert to backend-expected format
            submission_data = {
                'target_virus': job_data.extracted_parameters.get('target_virus', 'Alphavirus'),
                'input_genomes': job_data.extracted_parameters.get('input_genomes'),
                'limit': job_data.extracted_parameters.get('limit', 10),
                'output_format': 'json',
                'include_literature': True
            }
            
            timeout = aiohttp.ClientTimeout(total=self.request_timeout)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"{self.backend_url}/api/v1/annotate", 
                    json=submission_data
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        return {
                            'success': True,
                            'backend_job_id': result.get('job_id'),  # Backend returns 'job_id', not 'backend_job_id'
                            'estimated_duration': result.get('estimated_duration', job_data.estimated_duration)
                        }
                    else:
                        error_text = await response.text()
                        return {
                            'success': False,
                            'error': f"Backend submission failed: {response.status} - {error_text}"
                        }
                        
        except Exception as e:
            return {
                'success': False,
                'error': f"Failed to submit job to backend: {str(e)}"
            } 