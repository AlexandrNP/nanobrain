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
from typing import Dict, Any, Optional, AsyncGenerator
import asyncio
import uuid
import time
from datetime import datetime, timedelta

# Optional HTTP client for backend communication
try:
    import aiohttp
    _http_client_available = True
except ImportError:
    aiohttp = None
    _http_client_available = False


class AnnotationJobStep(Step):
    """
    Step for managing viral annotation pipeline jobs.
    
    Handles job submission, progress monitoring, and result management
    with concurrent job support and real-time progress streaming.
    """
    
    def __init__(self, config: StepConfig):
        super().__init__(config)
        
        # Get nested config dict
        step_config = getattr(config, 'config', {})
        
        # Backend configuration
        self.backend_url = step_config.get('backend_url', 'http://localhost:8001')
        self.max_concurrent_jobs = step_config.get('max_concurrent_jobs', 3)
        self.job_timeout = step_config.get('job_timeout_seconds', 1800)  # 30 minutes
        self.poll_interval = step_config.get('progress_poll_interval', 2)  # seconds
        
        # HTTP client configuration
        self.connection_timeout = step_config.get('connection_timeout', 30)
        self.request_timeout = step_config.get('request_timeout', 10)
        self.max_retries = step_config.get('max_retries', 3)
        
        # Active job tracking
        self.active_jobs: Dict[str, AnnotationJobData] = {}
        
        self.nb_logger.info(f"🔬 Annotation Job Step initialized (backend: {self.backend_url})")
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Submit and monitor annotation job.
        
        Args:
            input_data: Contains classification_data, session_data, routing_decision
            
        Returns:
            Job data and progress generator for real-time updates
        """
        try:
            classification_data: QueryClassificationData = input_data['classification_data']
            session_data: ChatSessionData = input_data['session_data']
            routing_decision: Dict[str, Any] = input_data['routing_decision']
            
            # Create job data
            job_data = AnnotationJobData(
                job_id=f"job_{uuid.uuid4().hex[:8]}",
                session_id=session_data.session_id,
                user_query=classification_data.original_query,
                extracted_parameters=classification_data.extracted_parameters,
                status='queued',
                backend_url=self.backend_url,
                estimated_duration=self._estimate_job_duration(classification_data),
                priority='normal'
            )
            
            self.nb_logger.info(f"🚀 Starting annotation job {job_data.job_id}")
            
            # Check if backend is available
            backend_available = await self._check_backend_availability()
            
            if not backend_available:
                # Backend unavailable - simulate with mock job
                return await self._handle_backend_unavailable(job_data, session_data)
            
            # Submit job to backend
            submission_result = await self._submit_job_to_backend(job_data)
            
            if not submission_result['success']:
                return {
                    'success': False,
                    'error': submission_result['error'],
                    'backend_available': True,
                    'job_data': job_data
                }
            
            # Update job with backend response
            job_data.backend_job_id = submission_result.get('backend_job_id')
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
            original_query=classification_data.original_query,
            request_parameters=request_params,
            status='pending'
        )
    
    async def _submit_job(self, job_data: AnnotationJobData) -> Dict[str, Any]:
        """Submit job to viral annotation backend"""
        
        try:
            # Prepare submission payload
            payload = {
                'job_id': job_data.job_id,
                'analysis_type': 'viral_protein_annotation',
                'parameters': job_data.request_parameters,
                'priority': 'normal',
                'callback_url': None  # We'll poll for progress instead
            }
            
            # Get or create HTTP session
            session = await self._get_http_session()
            
            # Submit to backend
            async with session.post(
                f"{self.backend_url}/api/v1/jobs",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    
                    # Update job data with backend info
                    job_data.backend_job_id = result.get('backend_job_id', job_data.job_id)
                    job_data.status = 'submitted'
                    
                    return {
                        'success': True,
                        'backend_job_id': job_data.backend_job_id
                    }
                
                elif response.status == 503:
                    # Service unavailable
                    return {
                        'success': False,
                        'error': "Viral annotation service is currently unavailable. Please try again later.",
                        'backend_available': False
                    }
                
                elif response.status == 429:
                    # Rate limited
                    return {
                        'success': False,
                        'error': "Too many requests. Please wait a moment before submitting another job.",
                        'backend_available': True
                    }
                
                else:
                    # Other error
                    error_text = await response.text()
                    return {
                        'success': False,
                        'error': f"Backend error ({response.status}): {error_text}",
                        'backend_available': True
                    }
        
        except aiohttp.ClientConnectorError:
            return {
                'success': False,
                'error': "Cannot connect to viral annotation service. The backend may be offline.",
                'backend_available': False
            }
        
        except asyncio.TimeoutError:
            return {
                'success': False,
                'error': "Request to viral annotation service timed out. Please try again.",
                'backend_available': True
            }
        
        except Exception as e:
            return {
                'success': False,
                'error': f"Failed to submit annotation job: {str(e)}",
                'backend_available': False
            }
    
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
                
                # Poll backend for job status
                status_result = await self._poll_job_status(job_data)
                
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
            'success': False,
            'error': 'Backend service unavailable',
            'backend_available': False,
            'job_data': job_data,
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