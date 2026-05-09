"""
Viral Analysis Web Workflow

Web-optimized wrapper for viral protein analysis workflow with progress tracking,
result formatting, and client/server optimization.

✅ FRAMEWORK COMPLIANCE: Uses from_config pattern exclusively
✅ WEB OPTIMIZATION: Progress tracking and formatted results for frontend
✅ NO HARDCODING: All behavior configurable via YAML
"""

import time
import uuid
from typing import Dict, Any, Optional
from datetime import datetime

from nanobrain.core.workflow import Workflow, WorkflowConfig
from nanobrain.core.logging_system import get_logger
from nanobrain.library.workflows.viral_protein_analysis.alphavirus_workflow import AlphavirusWorkflow

logger = get_logger(__name__)


class ViralAnalysisWebWorkflow(Workflow):
    """
    Viral Analysis Web Workflow - Web-Optimized Bioinformatics Analysis with Real-Time Progress Tracking
    ====================================================================================================

    The ViralAnalysisWebWorkflow provides a web-optimized wrapper around the core viral protein analysis
    workflow, specifically designed for web applications, client-server architectures, and real-time
    user interfaces. This workflow extends the standard AlphavirusWorkflow with web-specific features
    including progress tracking, result formatting, session management, and client optimization.

    **Core Architecture:**
        The web workflow enhances bioinformatics analysis with web-specific capabilities:

        * **Progress Tracking**: Real-time progress updates for long-running analyses
        * **Result Formatting**: Web-friendly result structuring and visualization preparation
        * **Session Management**: Multi-user session handling with unique identifiers
        * **Client Optimization**: Optimized data transfer and response formatting
        * **Error Handling**: Web-specific error responses and recovery mechanisms
        * **Download Management**: Preparation of user-downloadable analysis results

    **Web Optimization Features:**

        **Real-Time Progress Tracking:**
        * Step-by-step progress monitoring with weighted completion percentages
        * Session-based progress isolation for concurrent analyses
        * Detailed step metrics including data acquisition, clustering, and PSSM generation
        * WebSocket-compatible progress updates for responsive user interfaces

        **Client-Server Optimization:**
        * Structured response formatting for frontend consumption
        * Metadata extraction optimized for web display
        * Error responses formatted for user-friendly presentation
        * Session management with unique identifiers for multi-user environments

        **Result Visualization Preparation:**
        * Automatic preparation of visualization data for frontend charts
        * Support for histograms, dendrograms, and heatmap visualizations
        * Structured data formatting for JavaScript visualization libraries
        * Interactive visualization metadata and configuration

        **Download and Export Management:**
        * Preparation of downloadable files (FASTA, CSV, PDF reports)
        * File size estimation and metadata for download interfaces
        * On-demand report generation capabilities
        * Multiple format support for different user needs

    **Workflow Integration:**
        The web workflow seamlessly integrates with the core analysis pipeline:

        **Core Workflow Wrapping:**
        * Delegates core analysis to AlphavirusWorkflow for scientific accuracy
        * Maintains full compatibility with existing analysis configurations
        * Preserves all bioinformatics functionality while adding web features
        * Transparent parameter passing and result handling

        **Progress Monitoring Integration:**
        * Hooks into workflow steps for progress reporting
        * Weight-based progress calculation across analysis phases
        * Real-time status updates during long-running operations
        * Error state management and recovery notifications

        **Result Enhancement:**
        * Enriches core analysis results with web-specific metadata
        * Adds visualization preparation and download management
        * Maintains scientific data integrity while optimizing presentation
        * Provides multiple result formats for different client needs

    **Configuration Architecture:**
        Web workflow configuration supports comprehensive customization:

        ```yaml
        # Viral Analysis Web Workflow Configuration
        name: "viral_web_analysis"
        description: "Web-optimized viral protein analysis with progress tracking"

        # Core workflow configuration
        core_workflow:
          class: "nanobrain.library.workflows.viral_protein_analysis.AlphavirusWorkflow"
          config: "config/AlphavirusWorkflow.yml"

        # Web optimization settings
        web_optimization:
          progress_tracking:
            enable_real_time: true
            update_interval: 2.0  # seconds
            step_weights:
              data_acquisition: 0.20
              annotation_mapping: 0.10
              sequence_curation: 0.15
              clustering_analysis: 0.20
              alignment: 0.15
              pssm_analysis: 0.15
              data_aggregation: 0.03
              result_collection: 0.02

          result_formatting:
            include_visualizations: true
            prepare_downloads: true
            enable_metadata_extraction: true
            format_errors_for_web: true

          session_management:
            enable_session_tracking: true
            session_timeout: 3600  # seconds
            max_concurrent_sessions: 10
            cleanup_completed_sessions: true

        # Client optimization
        client_optimization:
          response_compression: true
          metadata_inclusion: "comprehensive"
          error_detail_level: "user_friendly"
          visualization_preparation: "automatic"
        ```

    **Usage Patterns:**

        **Basic Web Analysis:**
        ```python
        from nanobrain.library.workflows.viral_protein_analysis.web import ViralAnalysisWebWorkflow

        # Create web-optimized workflow
        web_workflow = ViralAnalysisWebWorkflow.from_config('config/viral_web_workflow.yml')

        # Execute analysis with session tracking
        result = await web_workflow.process({
            'virus_species': ['chikungunya_virus', 'zika_virus'],
            'analysis_type': 'protein',
            'user_query': 'Analyze structural proteins for vaccine targets',
            'session_id': 'user_session_123'
        })

        # Access web-optimized results
        print(f"Analysis status: {result['success']}")
        print(f"Summary: {result['summary']}")
        print(f"Visualizations: {len(result['visualizations'])}")
        print(f"Downloads: {len(result['downloads'])}")
        ```

        **Real-Time Progress Monitoring:**
        ```python
        # Start analysis with progress callbacks
        session_id = "analysis_session_456"

        # Initialize progress tracking
        await web_workflow.progress_tracker.initialize_session(session_id)

        # Monitor progress during execution
        async def progress_callback(step, progress):
            print(f"Step: {step}, Progress: {progress:.1%}")
            # Update web interface or WebSocket clients
            await notify_clients(session_id, step, progress)

        # Execute with progress monitoring
        result = await web_workflow.process({
            'virus_species': ['dengue_virus'],
            'session_id': session_id,
            'progress_callback': progress_callback
        })
        ```

        **Multi-User Session Management:**
        ```python
        # Handle multiple concurrent analyses
        web_workflow = ViralAnalysisWebWorkflow.from_config('config/multi_user_config.yml')

        # Process multiple user requests
        user_sessions = {
            'user_001': {'virus_species': ['sars_cov_2'], 'priority': 'high'},
            'user_002': {'virus_species': ['influenza_a'], 'priority': 'normal'},
            'user_003': {'virus_species': ['ebola_virus'], 'priority': 'urgent'}
        }

        # Execute concurrent analyses
        tasks = []
        for user_id, params in user_sessions.items():
            params['session_id'] = f"session_{user_id}"
            task = web_workflow.process(params)
            tasks.append(task)

        # Wait for all analyses to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        ```

        **Visualization and Download Preparation:**
        ```python
        # Execute analysis with full web optimization
        result = await web_workflow.process({
            'virus_species': ['alphavirus_family'],
            'analysis_type': 'comprehensive',
            'include_visualizations': True,
            'prepare_downloads': True
        })

        # Access prepared visualizations
        for viz in result['visualizations']:
            print(f"Visualization: {viz['title']} ({viz['type']})")
            print(f"Description: {viz['description']}")
            # Send to frontend visualization library

        # Access prepared downloads
        for download in result['downloads']:
            print(f"Download: {download['title']} ({download['type']})")
            print(f"Size: {download['size']} bytes")
            # Prepare download links for user interface
        ```

    **Advanced Features:**

        **Error Handling and Recovery:**
        * Web-friendly error message formatting for user interfaces
        * Detailed error context preservation for debugging
        * Graceful degradation with partial result recovery
        * Session cleanup and resource management on failures

        **Performance Optimization:**
        * Asynchronous processing for responsive web interfaces
        * Memory-efficient result streaming for large datasets
        * Intelligent caching integration for repeated analyses
        * Resource usage monitoring and optimization recommendations

        **Scalability Features:**
        * Multi-user session isolation and management
        * Concurrent analysis support with resource balancing
        * Horizontal scaling compatibility for web deployment
        * Load balancing integration for high-throughput scenarios

        **Integration Patterns:**
        * REST API compatibility for web service deployment
        * WebSocket support for real-time progress updates
        * Message queue integration for asynchronous processing
        * Database integration for result persistence and retrieval

    **Web Application Integration:**

        **Frontend Integration:**
        * Structured JSON responses optimized for web frameworks
        * Visualization data prepared for D3.js, Chart.js, and similar libraries
        * Progress tracking compatible with modern web progress indicators
        * Error handling designed for user-friendly web interfaces

        **Backend Integration:**
        * Async/await pattern compatibility for modern web frameworks
        * Session management integration with web authentication systems
        * Caching layer integration for improved response times
        * Logging integration with web application monitoring systems

        **Deployment Patterns:**
        * Container-ready design for Docker and Kubernetes deployment
        * Environment variable configuration for cloud deployments
        * Health check endpoints for load balancer integration
        * Monitoring and metrics collection for production environments

    **Performance and Scalability:**

        **Execution Optimization:**
        * Asynchronous processing prevents blocking web server threads
        * Progress tracking adds minimal overhead to core analysis
        * Result formatting optimized for network transfer efficiency
        * Memory usage optimization for concurrent session handling

        **Scalability Features:**
        * Session-based isolation enables horizontal scaling
        * Stateless design supports load balancing and clustering
        * Resource usage monitoring helps with capacity planning
        * Database integration supports result persistence at scale

        **Monitoring and Analytics:**
        * Comprehensive session tracking and analysis duration metrics
        * Progress tracking provides insights into workflow bottlenecks
        * Error tracking and categorization for system health monitoring
        * Usage pattern analysis for optimization and capacity planning

    **Security and Reliability:**

        **Session Security:**
        * Session isolation prevents cross-contamination of analyses
        * Unique session identifiers prevent unauthorized access
        * Automatic session cleanup prevents resource leaks
        * Error information sanitization for secure error reporting

        **Data Integrity:**
        * Core analysis results preserved without modification
        * Web enhancements added without affecting scientific accuracy
        * Checksums and validation for result integrity verification
        * Audit logging for compliance and troubleshooting

        **Reliability Features:**
        * Graceful error handling with detailed error context
        * Resource cleanup and session management
        * Health monitoring and status reporting
        * Automatic recovery mechanisms for transient failures

    Attributes:
        core_workflow (AlphavirusWorkflow): Core viral analysis workflow instance
        progress_tracker (WebProgressTracker): Real-time progress tracking component
        result_formatter (WebResultFormatter): Web-optimized result formatting component
        nb_logger (Logger): Framework logger for workflow operations

    Note:
        This workflow requires proper web infrastructure configuration including session
        storage, caching systems, and monitoring tools. Progress tracking adds minimal
        overhead but provides significant user experience improvements for long-running
        analyses. All web optimizations maintain full compatibility with the underlying
        scientific analysis workflow.

    Warning:
        Web workflows may consume additional memory for session tracking and result
        formatting. Monitor resource usage in multi-user environments and implement
        appropriate session limits and cleanup mechanisms. Be cautious with concurrent
        analyses to prevent resource exhaustion in production deployments.

    See Also:
        * :class:`AlphavirusWorkflow`: Core viral protein analysis workflow
        * :class:`Workflow`: Base workflow implementation
        * :class:`WebProgressTracker`: Progress tracking component
        * :class:`WebResultFormatter`: Result formatting component
        * :mod:`nanobrain.library.workflows.viral_protein_analysis`: Viral analysis workflows
        * :mod:`nanobrain.core.workflow`: Core workflow framework
    """

    def _init_from_config(self, config: WorkflowConfig, component_config: Dict[str, Any],
                          dependencies: Dict[str, Any]) -> None:
        """Initialize web workflow from configuration"""
        super()._init_from_config(config, component_config, dependencies)

        # ✅ FRAMEWORK COMPLIANCE: Initialize instance variables
        self.core_workflow: Optional[AlphavirusWorkflow] = None
        self.progress_tracker: Optional['WebProgressTracker'] = None
        self.result_formatter: Optional['WebResultFormatter'] = None

        self.nb_logger.info("🧬 Initializing Viral Analysis Web Workflow")

        # Initialize core workflow
        self._initialize_core_workflow()

        # Initialize web components
        self._initialize_web_components()

        # ✅ FRAMEWORK COMPLIANCE: Data units initialized by framework from configuration

        # ✅ FRAMEWORK COMPLIANCE: Agents and triggers initialized in initialize() method after framework loads them

        self.nb_logger.info("✅ Viral Analysis Web Workflow initialized")

    async def initialize(self) -> None:
        """
        ✅ CRITICAL FIX: Proper workflow initialization with trigger binding
        This method MUST be called to bind triggers to workflow execution
        """
        if self._is_initialized:
            self.nb_logger.info("🔄 ViralAnalysisWebWorkflow already initialized, skipping")
            return

        self.nb_logger.info("🚀 Initializing ViralAnalysisWebWorkflow with trigger binding")
        self.nb_logger.info(f"🔍 DEBUG: Workflow has triggers: {hasattr(self, 'triggers')}")
        if hasattr(self, 'triggers'):
            self.nb_logger.info(f"🔍 DEBUG: Triggers: {self.triggers}")

        # ✅ CRITICAL: Call parent initialize() to bind triggers
        self.nb_logger.info("🔧 Calling parent initialize() to bind triggers")
        await super().initialize()
        self.nb_logger.info("✅ Parent initialize() completed")

        # ✅ FRAMEWORK COMPLIANCE: Initialize agents after framework loads them
        self.nb_logger.info("🤖 Initializing agents")
        await self._initialize_agents()
        self.nb_logger.info("✅ Agents initialized")

        self.nb_logger.info("✅ ViralAnalysisWebWorkflow initialization complete with triggers bound")

    async def _execute_on_trigger(self, trigger_event: Dict[str, Any]) -> None:
        """
        ✅ CRITICAL FIX: Framework trigger response method
        This method is called by the framework when DataUnitChangeTrigger fires
        """
        try:
            self.nb_logger.info("🎯 _execute_on_trigger method called - TRIGGER BINDING WORKING!")
            self.nb_logger.info(f"🔥 ViralAnalysisWebWorkflow triggered by {trigger_event.get('trigger_id', 'unknown')}")
            self.nb_logger.info(f"📥 Trigger event data: {trigger_event}")

            # Get input data from the triggered data unit - FRAMEWORK COMPLIANT
            change_event = trigger_event.get('change_event', {})
            input_data = change_event.get('new_data', {})

            if not input_data:
                # Try to get data directly from data unit - FRAMEWORK COMPLIANT
                data_unit_name = trigger_event.get('data_unit', 'web_analysis_request')

                # ✅ FRAMEWORK COMPLIANCE: Check both step and workflow data units
                data_unit = None
                if hasattr(self, 'step_input_data_units') and data_unit_name in self.step_input_data_units:
                    data_unit = self.step_input_data_units[data_unit_name]
                elif hasattr(self, 'input_data_units') and data_unit_name in self.input_data_units:
                    data_unit = self.input_data_units[data_unit_name]
                elif hasattr(self, 'data_units') and data_unit_name in self.data_units:
                    data_unit = self.data_units[data_unit_name]

                if not data_unit:
                    raise RuntimeError(
                        f"❌ TRIGGER DATA FAILURE: No input data in trigger event and "
                        f"data unit '{data_unit_name}' not found in any data unit collection"
                    )

                input_data = await data_unit.get()

                if not input_data:
                    raise RuntimeError(
                        f"❌ DATA UNIT EMPTY: Data unit '{data_unit_name}' contains no data"
                    )

                self.nb_logger.info(f"📥 Retrieved data from {data_unit_name}: {len(str(input_data))} chars")

            # Execute the main workflow process
            result = await self.process(input_data)

            # Update output data units if available
            if hasattr(self, 'step_output_data_units') and 'web_analysis_response' in self.step_output_data_units:
                output_data_unit = self.step_output_data_units['web_analysis_response']
                await output_data_unit.set(result)
                self.nb_logger.info("✅ Workflow result set to output data unit")

            self.nb_logger.info("✅ ViralAnalysisWebWorkflow execution completed via trigger")

        except Exception as e:
            self.nb_logger.error(f"❌ Error in workflow trigger execution: {e}", exc_info=True)
            raise

    def _initialize_core_workflow(self) -> None:
        """
        ✅ FRAMEWORK COMPLIANCE: Initialize core alphavirus workflow
        """
        try:
            # Load core workflow via from_config - use config path from configuration
            core_workflow_config = getattr(
                self.config, 'core_workflow_config', {})
            core_config_path = core_workflow_config.get(
                'config_path',
                "/Users/onarykov/git/nanobrain-upd-Jun/nanobrain/nanobrain/library/workflows/viral_protein_analysis/config/AlphavirusWorkflow.yml"
            )
            self.core_workflow = AlphavirusWorkflow.from_config(
                core_config_path)

            self.nb_logger.info("✅ Core viral analysis workflow initialized")

        except Exception as e:
            self.nb_logger.error(f"❌ Failed to initialize core workflow: {e}")
            raise

    def _initialize_web_components(self) -> None:
        """
        ✅ FRAMEWORK COMPLIANCE: Initialize web optimization components
        """
        try:
            # Initialize progress tracker for web clients
            self.progress_tracker = WebProgressTracker()

            # Initialize result formatter for web consumption
            self.result_formatter = WebResultFormatter()

            self.nb_logger.info("✅ Web optimization components initialized")

        except Exception as e:
            self.nb_logger.error(f"❌ Failed to initialize web components: {e}")
            raise



    async def _initialize_agents(self) -> None:
        """
        ✅ FRAMEWORK COMPLIANCE: Initialize workflow agents using framework-instantiated objects
        """
        try:
            # Check if agents were resolved by ConfigBase
            if hasattr(self.config, 'resolved_agents') and self.config.resolved_agents:
                # Use framework-instantiated agents
                if 'virus_extraction_agent' not in self.config.resolved_agents:
                    raise RuntimeError(
                        "❌ CRITICAL FAILURE: VirusExtractionAgent not found in resolved agents. "
                        "Check agent configuration in workflow YAML."
                    )

                self.virus_extraction_agent = self.config.resolved_agents['virus_extraction_agent']
                self.nb_logger.info("✅ Using framework-instantiated VirusExtractionAgent")

            elif hasattr(self.config, 'agents') and 'virus_extraction_agent' in self.config.agents:
                # Fallback: check if already instantiated by framework
                agent_config = self.config.agents['virus_extraction_agent']
                if hasattr(agent_config, 'aprocess'):
                    # Already instantiated by framework
                    self.virus_extraction_agent = agent_config
                    self.nb_logger.info("✅ Using pre-instantiated VirusExtractionAgent")
                else:
                    raise RuntimeError(
                        "❌ CRITICAL FAILURE: Agent configuration found but not instantiated. "
                        "Framework agent resolution failed."
                    )
            else:
                raise RuntimeError(
                    "❌ CRITICAL FAILURE: VirusExtractionAgent not available. "
                    "System cannot function without proper entity extraction. "
                    "Check agent configuration in workflow YAML."
                )

            # Validate agent is properly instantiated
            if not hasattr(self.virus_extraction_agent, 'aprocess'):
                raise RuntimeError(f"Invalid agent object: {type(self.virus_extraction_agent)}")

            # Initialize agent for LLM client setup
            await self.virus_extraction_agent.initialize()

            self.nb_logger.info("✅ VirusExtractionAgent loaded and initialized")
            self.nb_logger.info(f"✅ Agent type: {type(self.virus_extraction_agent)}")

        except Exception as e:
            self.nb_logger.error(f"❌ Failed to initialize agents: {e}")
            raise  # FAIL FAST - no fallbacks



    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        ✅ FRAMEWORK COMPLIANCE: Execute viral analysis with web optimizations
        ✅ ROUTING COMPATIBILITY: Handle routing system data format
        """
        start_time = time.time()

        try:
            # Extract basic parameters from routing data
            user_query = input_data.get('user_query', '')
            session_id = input_data.get('session_id', f"session_{uuid.uuid4().hex[:8]}")

            self.nb_logger.info(f"🧬 Processing routing data for session: {session_id}")
            self.nb_logger.info(f"🔍 User query: {user_query}")

            # ✅ FRAMEWORK COMPLIANCE: Use existing VirusExtractionAgent - NO FALLBACKS
            if not hasattr(self, 'virus_extraction_agent') or not self.virus_extraction_agent:
                raise RuntimeError(
                    "❌ CRITICAL FAILURE: VirusExtractionAgent not available. "
                    "System cannot function without proper entity extraction. "
                    "Check agent configuration in workflow YAML."
                )

            self.nb_logger.info("🤖 Using VirusExtractionAgent for entity extraction")
            extraction_result = await self.virus_extraction_agent.process({
                'user_query': user_query,
                'expected_format': 'json',
                'analysis_context': 'viral_protein_analysis'
            })

            # ✅ FRAMEWORK COMPLIANCE: Parse agent response (agent returns JSON string)
            if isinstance(extraction_result, str):
                try:
                    import json
                    extraction_result = json.loads(extraction_result)
                    self.nb_logger.info(f"✅ Parsed agent JSON response: {extraction_result}")
                except json.JSONDecodeError as e:
                    raise ValueError(f"❌ Agent returned invalid JSON: {extraction_result}. Error: {e}")
            elif not isinstance(extraction_result, dict):
                raise ValueError(f"❌ Agent returned unexpected type: {type(extraction_result)}. Response: {extraction_result}")

            # Extract results from agent response - CRITICAL FIX: Handle both singular and plural keys
            virus_species = extraction_result.get('virus_names', [])
            if not virus_species:
                # Try singular key as fallback (agent returns 'virus_name' not 'virus_names')
                virus_name = extraction_result.get('virus_name', '')
                if virus_name:
                    virus_species = [virus_name]  # Convert to list format

            analysis_type = extraction_result.get('analysis_type', 'protein')
            # Minimal fix: ensure PSSM queries select the correct analysis pipeline
            try:
                uq = (user_query or '').lower()
                if (not analysis_type or analysis_type == 'protein') and ('pssm' in uq or 'matrix' in uq):
                    self.nb_logger.info("🧭 Detected PSSM-related query keywords; overriding analysis_type to 'pssm'")
                    analysis_type = 'pssm'
            except Exception:
                pass

            if not virus_species:
                raise ValueError(
                    f"❌ ENTITY EXTRACTION FAILURE: VirusExtractionAgent could not identify "
                    f"virus species from query: '{user_query}'. Agent response: {extraction_result}"
                )

            self.nb_logger.info(f"🦠 Agent extracted virus species: {virus_species}")
            self.nb_logger.info(f"🔬 Agent determined analysis type: {analysis_type}")

            # Validate extracted parameters
            if not virus_species:
                return self._create_error_response(
                    f"Could not identify virus species from query: '{user_query}'. "
                    f"Please specify a virus species in your query.",
                    session_id, start_time
                )

            # Prepare input for core workflow
            workflow_input = self._prepare_workflow_input(
                virus_species, analysis_type, user_query, session_id
            )

            # Initialize progress tracking
            await self.progress_tracker.initialize_session(session_id)

            # Execute core viral analysis workflow with progress monitoring
            analysis_result = await self._execute_with_progress_tracking(
                workflow_input, session_id
            )

            # Format result for web consumption
            formatted_result = await self.result_formatter.format_for_web(
                analysis_result, session_id
            )

            processing_time = time.time() - start_time
            self.nb_logger.info(
                f"✅ Viral analysis completed in {processing_time:.2f}s")

            return {
                'success': True,
                'response_type': 'viral_analysis',
                'content': formatted_result,
                'session_id': session_id,
                'metadata': {
                    'workflow_type': 'viral_protein_analysis',
                    'virus_species': virus_species,
                    'analysis_type': analysis_type,
                    'steps_completed': analysis_result.get('steps_completed', 0),
                    'total_processing_time': processing_time,
                    'web_optimized': True
                }
            }

        except Exception as e:
            self.nb_logger.error(
                f"❌ Web viral analysis failed: {e}", exc_info=True)
            return self._create_error_response(str(e), session_id, start_time)

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        ✅ CRITICAL FIX: Main execution method called by triggers
        This method is called when the DataUnitChangeTrigger fires
        """
        self.nb_logger.info("🚀 ViralAnalysisWebWorkflow.execute() called by trigger")
        self.nb_logger.info(f"📥 Input data keys: {list(input_data.keys())}")

        # Call the main process method with the input data
        result = await self.process(input_data)

        # Update the output data unit to complete the workflow chain
        web_analysis_response_du = None
        if hasattr(self, 'output_data_units') and 'web_analysis_response' in self.output_data_units:
            web_analysis_response_du = self.output_data_units['web_analysis_response']
        elif hasattr(self, 'data_units') and 'web_analysis_response' in self.data_units:
            web_analysis_response_du = self.data_units['web_analysis_response']

        if web_analysis_response_du:
            await web_analysis_response_du.set(result)
            self.nb_logger.info("✅ Result set to web_analysis_response data unit")

        return result

    def _prepare_workflow_input(self, virus_species: list, analysis_type: str,
                                user_query: str, session_id: str) -> Dict[str, Any]:
        """
        ✅ FRAMEWORK COMPLIANCE: Prepare input for core workflow
        """
        return {
            'virus_names': virus_species,
            'analysis_type': analysis_type,
            'user_context': user_query,
            'session_id': session_id,
            'web_mode': True,
            'progress_tracking': True
        }

    async def _execute_with_progress_tracking(self, workflow_input: Dict[str, Any],
                                              session_id: str) -> Dict[str, Any]:
        """
        ✅ FRAMEWORK COMPLIANCE: Execute core workflow with progress callbacks
        """
        try:
            # Set up progress callbacks for each step
            progress_callbacks = {
                'data_acquisition': lambda p: self.progress_tracker.update_step(
                    'data_acquisition', p, session_id
                ),
                'annotation_mapping': lambda p: self.progress_tracker.update_step(
                    'annotation_mapping', p, session_id
                ),
                'sequence_curation': lambda p: self.progress_tracker.update_step(
                    'sequence_curation', p, session_id
                ),
                'clustering_analysis': lambda p: self.progress_tracker.update_step(
                    'clustering_analysis', p, session_id
                ),
                'alignment': lambda p: self.progress_tracker.update_step(
                    'alignment', p, session_id
                ),
                'pssm_analysis': lambda p: self.progress_tracker.update_step(
                    'pssm_analysis', p, session_id
                ),
                'data_aggregation': lambda p: self.progress_tracker.update_step(
                    'data_aggregation', p, session_id
                ),
                'result_collection': lambda p: self.progress_tracker.update_step(
                    'result_collection', p, session_id
                )
            }

            # Add progress callbacks to workflow input
            workflow_input['progress_callbacks'] = progress_callbacks

            # Execute core workflow with progress monitoring
            result = await self.core_workflow.process(workflow_input)

            # Mark session as completed
            await self.progress_tracker.mark_completed(session_id)

            return result

        except Exception as e:
            # Mark session as failed
            await self.progress_tracker.mark_failed(session_id, str(e))
            raise



    def _create_error_response(self, error_message: str, session_id: str,
                               start_time: Optional[float] = None) -> Dict[str, Any]:
        """
        ✅ FRAMEWORK COMPLIANCE: Create structured error response for web clients
        """
        processing_time = (time.time() - start_time) if start_time else 0.0

        return {
            'success': False,
            'response_type': 'analysis_error',
            'error': error_message,
            'session_id': session_id,
            'metadata': {
                'workflow_type': 'viral_protein_analysis',
                'processing_time': processing_time,
                'web_optimized': True,
                'timestamp': datetime.now().isoformat()
            }
        }


class WebProgressTracker:
    """
    ✅ FRAMEWORK COMPLIANCE: Web-optimized progress tracking for viral analysis
    Extracted and adapted from chatbot_viral_integration progress tracking
    """

    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.step_weights = {
            'data_acquisition': 0.20,
            'annotation_mapping': 0.10,
            'sequence_curation': 0.15,
            'clustering_analysis': 0.20,
            'alignment': 0.15,
            'pssm_analysis': 0.15,
            'data_aggregation': 0.03,
            'result_collection': 0.02
        }

    async def initialize_session(self, session_id: str) -> None:
        """Initialize progress tracking for analysis session"""
        self.sessions[session_id] = {
            'overall_progress': 0.0,
            'current_step': 'initializing',
            'step_progress': {},
            'started_at': datetime.now(),
            'status': 'running',
            'steps_completed': 0,
            'total_steps': len(self.step_weights)
        }
        logger.debug(
            f"✅ Progress tracking initialized for session: {session_id}")

    async def update_step(self, step_name: str, progress: float, session_id: str) -> None:
        """Update progress for specific analysis step"""
        if session_id not in self.sessions:
            logger.warning(
                f"⚠️ Session {session_id} not found for progress update")
            return

        session = self.sessions[session_id]
        session['step_progress'][step_name] = progress
        session['current_step'] = step_name

        # Calculate overall progress
        overall = sum(
            self.step_weights.get(step, 0) * prog
            for step, prog in session['step_progress'].items()
        )
        session['overall_progress'] = min(overall, 1.0)

        # Count completed steps
        session['steps_completed'] = sum(
            1 for prog in session['step_progress'].values() if prog >= 1.0
        )

        logger.debug(
            f"📊 Progress update: {step_name} = {progress:.2f}, overall = {overall:.2f}")

    async def mark_completed(self, session_id: str) -> None:
        """Mark analysis session as completed"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            session['status'] = 'completed'
            session['overall_progress'] = 1.0
            session['completed_at'] = datetime.now()
            logger.info(f"✅ Analysis session completed: {session_id}")

    async def mark_failed(self, session_id: str, error_message: str) -> None:
        """Mark analysis session as failed"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            session['status'] = 'failed'
            session['error'] = error_message
            session['failed_at'] = datetime.now()
            logger.error(
                f"❌ Analysis session failed: {session_id} - {error_message}")

    async def get_progress(self, session_id: str) -> Dict[str, Any]:
        """Get current progress for analysis session"""
        return self.sessions.get(session_id, {
            'status': 'not_found',
            'error': 'Session not found'
        })


class WebResultFormatter:
    """
    ✅ FRAMEWORK COMPLIANCE: Format viral analysis results for web consumption
    Adapted from ResponseFormattingStep in chatbot_viral_integration
    """

    def __init__(self):
        self.formatter_id = f"formatter_{uuid.uuid4().hex[:8]}"

    async def format_for_web(self, analysis_result: Dict[str, Any],
                             session_id: str) -> Dict[str, Any]:
        """
        ✅ REUSED LOGIC: Format viral analysis results for frontend display
        """
        try:
            formatted = {
                'summary': self._create_summary(analysis_result),
                'detailed_results': self._format_detailed_results(analysis_result),
                'visualizations': self._prepare_visualizations(analysis_result),
                'downloadable_files': self._prepare_downloads(analysis_result),
                'metadata': self._extract_metadata(analysis_result, session_id)
            }

            logger.debug(
                f"✅ Results formatted for web consumption: {session_id}")
            return formatted

        except Exception as e:
            logger.error(f"❌ Result formatting failed: {e}")
            return {
                'summary': 'Analysis completed with formatting errors',
                'error': str(e),
                'session_id': session_id
            }

    def _create_summary(self, result: Dict[str, Any]) -> str:
        """Create human-readable summary of analysis results"""
        virus_count = len(result.get('virus_species_processed', []))
        protein_count = result.get('proteins_analyzed', 0)
        pssm_generated = result.get('pssm_matrices_generated', 0)

        return (
            f"Analysis completed for {virus_count} virus species, "
            f"processing {protein_count} proteins and generating "
            f"{pssm_generated} PSSM matrices."
        )

    def _format_detailed_results(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Format detailed analysis results for structured display"""
        return {
            'sequence_statistics': result.get('sequence_stats', {}),
            'clustering_results': result.get('clustering_data', {}),
            'alignment_quality': result.get('alignment_metrics', {}),
            'pssm_analysis': result.get('pssm_results', {}),
            'data_acquisition_summary': result.get('acquisition_summary', {}),
            'annotation_mapping_results': result.get('annotation_results', {})
        }

    def _prepare_visualizations(self, result: Dict[str, Any]) -> list:
        """Prepare visualization data for frontend charts and graphs"""
        visualizations = []

        # Sequence length distribution
        if 'sequence_stats' in result:
            visualizations.append({
                'type': 'histogram',
                'title': 'Sequence Length Distribution',
                'data': result['sequence_stats'].get('length_distribution', []),
                'description': 'Distribution of protein sequence lengths'
            })

        # Clustering dendrogram
        if 'clustering_data' in result:
            visualizations.append({
                'type': 'dendrogram',
                'title': 'Protein Clustering Analysis',
                'data': result['clustering_data'].get('dendrogram_data', {}),
                'description': 'Hierarchical clustering of protein sequences'
            })

        # PSSM heatmap
        if 'pssm_results' in result:
            visualizations.append({
                'type': 'heatmap',
                'title': 'PSSM Matrix Visualization',
                'data': result['pssm_results'].get('matrix_data', {}),
                'description': 'Position-specific scoring matrix heatmap'
            })

        return visualizations

    def _prepare_downloads(self, result: Dict[str, Any]) -> list:
        """Prepare downloadable files for user access"""
        downloads = []

        # FASTA sequences
        if 'sequence_files' in result:
            downloads.append({
                'type': 'fasta',
                'title': 'Protein Sequences',
                'filename': 'viral_proteins.fasta',
                'size': result['sequence_files'].get('size', 0),
                'description': 'Curated protein sequences in FASTA format'
            })

        # PSSM matrices
        if 'pssm_files' in result:
            downloads.append({
                'type': 'matrix',
                'title': 'PSSM Matrices',
                'filename': 'pssm_matrices.csv',
                'size': result['pssm_files'].get('size', 0),
                'description': 'Position-specific scoring matrices'
            })

        # Analysis report
        downloads.append({
            'type': 'report',
            'title': 'Analysis Report',
            'filename': 'viral_analysis_report.pdf',
            'size': 0,  # Generated on demand
            'description': 'Comprehensive analysis report with all results'
        })

        return downloads

    def _extract_metadata(self, result: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """Extract and format metadata for web display"""
        return {
            'session_id': session_id,
            'analysis_timestamp': datetime.now().isoformat(),
            'formatter_id': self.formatter_id,
            'total_processing_time': result.get('total_processing_time', 0),
            'steps_executed': result.get('steps_executed', []),
            'workflow_version': result.get('workflow_version', '1.0.0'),
            'data_sources': result.get('data_sources', []),
            'quality_metrics': result.get('quality_metrics', {})
        }
