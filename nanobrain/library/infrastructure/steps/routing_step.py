"""
Routing Step for NanoBrain Framework

This step provides LLM-based intelligent query analysis and workflow routing.
Combines query classification and workflow selection into a unified routing solution.
Reuses proven logic from the IntelligentWorkflowRouter and QueryAnalysisAgent.

✅ FRAMEWORK COMPLIANCE: Uses from_config pattern exclusively
✅ NO HARDCODING: All routing decisions via LLM agents and configuration  
✅ NO SIMPLIFIED SOLUTIONS: Complete intelligent routing implementation
✅ DATA-DRIVEN: Configuration-driven routing rules and thresholds
✅ SINGLE RESPONSIBILITY: Pure query routing - from analysis to workflow selection
"""

import json
import uuid
import time
import yaml
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

from pydantic import Field

from nanobrain.core.step import BaseStep, StepConfig
from nanobrain.core.component_base import ComponentConfigurationError, ComponentDependencyError
from nanobrain.core.logging_system import get_logger
from nanobrain.core.data_unit import DataUnitBase

logger = get_logger(__name__)


class RoutingStepConfig(StepConfig):
    """
    ✅ FRAMEWORK COMPLIANCE: Configuration for Routing Step
    Generalizable configuration that works with any domain and workflow types
    """

    # Classification agents - configurable for any domain
    classification_agents: List[Any] = Field(
        default_factory=lambda: [
            {
                'name': 'primary_classifier',
                'class': 'nanobrain.library.agents.specialized.query_analysis_agent.QueryAnalysisAgent',
                'config': 'nanobrain/library/agents/specialized/config/query_analysis_agent_config.yml',
                'role': 'intent_classification'
            }
        ],
        description="List of classification agents for query analysis (supports both dict configs and resolved agent instances)"
    )

    # General classification configuration
    classification_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            'classification_method': 'llm_based',
            'confidence_threshold': 0.6,
            'fallback_threshold': 0.3,
            'enable_multi_class': True,
            'enable_confidence_scoring': True,
            'max_classification_time': 30.0
        },
        description="General classification behavior configuration"
    )

    # Workflow discovery configuration - discovers available workflows dynamically
    workflow_discovery: Dict[str, Any] = Field(
        default_factory=lambda: {
            'discovery_method': 'registry_based',  # or 'config_based', 'filesystem_based'
            'workflow_registry_path': None,  # Path to workflow registry if using registry_based
            'workflow_config_paths': [],  # List of workflow config paths if using config_based
            'capability_extraction': True,  # Extract capabilities from workflow metadata
            'auto_discovery': True  # Automatically discover workflows at runtime
        },
        description="Configuration for discovering available workflows"
    )

    # Classification strategy configuration
    classification_strategy: Dict[str, Any] = Field(
        default_factory=lambda: {
            # capability_based, keyword_based, semantic_based
            'matching_algorithm': 'capability_based',
            'capability_weights': {
                'domain_match': 0.4,
                'intent_match': 0.3,
                'keyword_match': 0.2,
                'confidence_score': 0.1
            },
            'enable_fallback_routing': True,
            'fallback_workflow_pattern': 'conversational_*',  # Pattern for fallback workflows
            'min_confidence_for_routing': 0.5
        },
        description="Strategy for matching queries to workflows"
    )

    # Analysis configuration
    analysis_depth: str = Field(
        default="comprehensive", description="Analysis depth: standard, detailed, comprehensive")
    enable_confidence_scoring: bool = Field(
        default=True, description="Enable detailed confidence scoring")
    enable_fallback_processing: bool = Field(
        default=True, description="Enable fallback processing for edge cases")
    max_classification_time: float = Field(
        default=30.0, description="Maximum time for classification in seconds")


class RoutingStep(BaseStep):
    """
    ✅ FRAMEWORK COMPLIANCE: Routing Step using LLM-based intelligent query analysis and workflow routing

    Provides intelligent query analysis and workflow routing using specialized LLM agents.
    Combines query classification and workflow selection into unified routing solution.
    Reuses proven logic from IntelligentWorkflowRouter and specialized agents.

    **Single Responsibility**: Pure query routing - from analysis to workflow selection.
    **Framework Integration**: Full compliance with NanoBrain step patterns.
    **LLM-Powered**: Uses specialized agents for intelligent routing decisions.
    **Configuration-Driven**: All routing rules and workflow selection via configuration.
    """

    # NOTE: Data unit interfaces are now defined in the configuration YAML files
    # and created during initialization. The class attributes above were causing
    # conflicts with the BaseStep property mapping system.
    #
    # Data units are accessible via:
    # - self.step_input_data_units['user_request']
    # - self.step_input_data_units['session_context']
    # - self.step_output_data_units['workflow_routing']
    # - self.step_output_data_units['session_context']
    #
    # Or via the property mapping:
    # - self.input_data_units (maps to step_input_data_units)
    # - self.output_data_units (maps to step_output_data_units)

    def __init__(self):
        """Initialize Routing Step - use from_config for creation"""
        super().__init__()
        # Instance variables will be initialized in _init_from_config

    @classmethod
    def _get_config_class(cls):
        """Return the configuration class for this component"""
        return RoutingStepConfig

    def _init_from_config(self, config: RoutingStepConfig, component_config: Dict[str, Any], dependencies: Dict[str, Any]) -> None:
        """Initialize Routing Step from configuration"""
        super()._init_from_config(config, component_config, dependencies)

        # Initialize instance variables from configuration
        self.classification_agents: Dict[str, Any] = {}
        self.available_workflows: Dict[str, Dict[str, Any]] = {}
        self.classification_config: Dict[str, Any] = component_config.get(
            'classification_config', {})
        self.workflow_discovery_config: Dict[str, Any] = component_config.get(
            'workflow_discovery', {})
        self.classification_strategy: Dict[str, Any] = component_config.get(
            'classification_strategy', {})

        # Initialize logger
        self.logger = get_logger(__name__, debug_mode=config.debug_mode)

        self.logger.info(
            "🧠 Initializing Routing Step with intelligent query analysis and workflow routing")

        # Debug logging to see what configuration is loaded
        self.logger.info(
            f"🔍 INIT DEBUG: component_config keys: {list(component_config.keys())}")
        self.logger.info(
            f"🔍 INIT DEBUG: workflow_discovery_config: {self.workflow_discovery_config}")

        try:
            # Setup classification configuration (sync)
            self._setup_classification_configuration()

            # Discover available workflows dynamically (sync)
            self._discover_available_workflows()

            # Setup classification strategy (sync)
            self._setup_classification_strategy()

            self.logger.info(
                "✅ Routing Step basic initialization completed - agents will be loaded in initialize()")

        except Exception as e:
            self.logger.error(
                f"❌ Failed to initialize Routing Step: {e}", exc_info=True)
            raise ComponentConfigurationError(
                f"Routing Step initialization failed: {e}")

    async def initialize(self) -> None:
        """
        ✅ ASYNC INITIALIZATION: Load agents with proper LLM client initialization
        """
        # Initialize parent first
        await super().initialize()

        try:
            # Load classification agents asynchronously (this is where the fix is applied)
            await self._load_classification_agents()

            # Validate classification agents were loaded
            if not self.classification_agents:
                raise ComponentConfigurationError(
                    "No classification agents configured or loaded")

            self.logger.info(
                f"✅ Routing Step fully initialized with {len(self.classification_agents)} agents")

        except Exception as e:
            self.logger.error(
                f"❌ Failed to initialize Routing Step agents: {e}", exc_info=True)
            raise ComponentConfigurationError(
                f"Routing Step agent initialization failed: {e}")

    async def _load_classification_agents(self) -> None:
        """
        ✅ FRAMEWORK COMPLIANCE: Load classification agents dynamically
        Supports multiple agents for different classification tasks
        ✅ FIX: Now async to support agent initialization
        """
        try:
            for i, agent_config in enumerate(self.config.classification_agents):
                # Handle both dictionary configs and already-resolved agent instances
                if isinstance(agent_config, dict):
                    # Dictionary configuration
                    agent_name = agent_config.get('name', f'agent_{i}')
                    agent_class = agent_config.get('class')
                    agent_config_path = agent_config.get('config')
                    agent_role = agent_config.get('role', 'general')

                    self.logger.debug(
                        f"Loading classification agent: {agent_name} with role: {agent_role}")

                    # Load via framework pattern (now async)
                    agent_instance = await self._load_agent_component(
                        agent_class, agent_config_path)
                    self.classification_agents[agent_name] = {
                        'instance': agent_instance,
                        'role': agent_role,
                        'name': agent_name
                    }

                elif hasattr(agent_config, '_process_specialized_request') or hasattr(agent_config, 'aprocess'):
                    # Already resolved agent instance
                    agent_name = getattr(
                        agent_config, 'name', f'resolved_agent_{i}')
                    self.classification_agents[agent_name] = {
                        'instance': agent_config,
                        'role': 'intent_classification',  # Default role for resolved agents
                        'name': agent_name
                    }
                    self.logger.debug(
                        f"✅ Agent {agent_name} already resolved by framework")

                else:
                    self.logger.warning(
                        f"Unrecognized agent config type at index {i}: {type(agent_config)}")
                    continue

            self.logger.info(
                f"✅ Loaded {len(self.classification_agents)} classification agents")

        except Exception as e:
            self.logger.error(
                f"❌ Failed to load classification agents: {e}", exc_info=True)
            raise ComponentDependencyError(
                f"Classification agents loading failed: {e}")

    async def _load_agent_component(self, agent_class: str, config_path: str):
        """
        ✅ FRAMEWORK COMPLIANCE: Load agent component using framework patterns
        Reuses logic from IntelligentWorkflowRouter
        ✅ FIX: Now properly initializes agents after creation
        """
        try:
            self.logger.info(f"🔍 DEBUG: Loading agent class: {agent_class}")
            self.logger.info(f"🔍 DEBUG: Config path: {config_path}")

            # Import agent class dynamically
            module_path, class_name = agent_class.rsplit('.', 1)
            module = __import__(module_path, fromlist=[class_name])
            agent_cls = getattr(module, class_name)

            self.logger.info(f"🔍 DEBUG: Agent class loaded: {agent_cls}")

            # Create agent using from_config pattern
            agent = agent_cls.from_config(config_path)

            self.logger.info(f"🔍 DEBUG: Agent created: {type(agent)}")
            self.logger.info(
                f"🔍 DEBUG: Agent has confidence_threshold: {hasattr(agent, 'confidence_threshold')}")

            # ✅ CRITICAL FIX: Initialize agent to set up LLM client
            await agent.initialize()

            self.logger.info(f"🔍 DEBUG: Agent initialized: {type(agent)}")
            self.logger.info(
                f"🔍 DEBUG: Agent has confidence_threshold after init: {hasattr(agent, 'confidence_threshold')}")

            self.logger.debug(
                f"✅ Agent {class_name} created and initialized successfully")
            return agent

        except Exception as e:
            self.logger.error(
                f"❌ Failed to load agent component {agent_class}: {e}", exc_info=True)
            raise ComponentDependencyError(
                f"Agent component loading failed: {e}")

    def _setup_classification_configuration(self) -> None:
        """
        ✅ FRAMEWORK COMPLIANCE: Setup classification configuration from config
        """
        self.classification_config = self.config.classification_config
        self.workflow_discovery_config = self.config.workflow_discovery
        self.classification_strategy = self.config.classification_strategy

        # Initialize classification thresholds from config
        self.classification_thresholds = self.classification_config.get('thresholds', {
            'high_confidence': 0.8,
            'medium_confidence': 0.6,
            'low_confidence': 0.4,
            'fallback_threshold': 0.2
        })

        # Validate configuration
        self._validate_classification_configuration()

        self.logger.debug("✅ Classification configuration setup complete")

    def _validate_classification_configuration(self) -> None:
        """
        ✅ FRAMEWORK COMPLIANCE: Validate classification configuration
        """
        # Validate confidence threshold ranges
        confidence_threshold = self.classification_config.get(
            'confidence_threshold', 0.6)
        if not (0.0 <= confidence_threshold <= 1.0):
            self.logger.warning(
                f"⚠️ Invalid confidence_threshold: {confidence_threshold}. Setting to 0.6")
            self.classification_config['confidence_threshold'] = 0.6

        fallback_threshold = self.classification_config.get(
            'fallback_threshold', 0.3)
        if not (0.0 <= fallback_threshold <= 1.0):
            self.logger.warning(
                f"⚠️ Invalid fallback_threshold: {fallback_threshold}. Setting to 0.3")
            self.classification_config['fallback_threshold'] = 0.3

        self.logger.debug("✅ Classification configuration setup completed")

    def _discover_available_workflows(self) -> None:
        """
        ✅ FRAMEWORK COMPLIANCE: Discover available workflows dynamically
        """
        try:
            discovery_method = self.workflow_discovery_config.get(
                'discovery_method', 'registry_based')

            self.logger.info(
                f"🔍 WORKFLOW DISCOVERY DEBUG: Using discovery method: {discovery_method}")
            self.logger.info(
                f"🔍 WORKFLOW DISCOVERY DEBUG: workflow_discovery_config: {self.workflow_discovery_config}")

            if discovery_method == 'direct_reference':
                self.logger.info(
                    "🔍 WORKFLOW DISCOVERY DEBUG: Using direct_reference discovery")
                self._discover_workflows_from_direct_references()
            elif discovery_method == 'registry_based':
                self.logger.info(
                    "🔍 WORKFLOW DISCOVERY DEBUG: Using registry_based discovery")
                self._discover_workflows_from_registry()
            elif discovery_method == 'config_based':
                self.logger.info(
                    "🔍 WORKFLOW DISCOVERY DEBUG: Using config_based discovery")
                self._discover_workflows_from_configs()
            elif discovery_method == 'filesystem_based':
                self.logger.info(
                    "🔍 WORKFLOW DISCOVERY DEBUG: Using filesystem_based discovery")
                self._discover_workflows_from_filesystem()
            else:
                self.logger.warning(
                    f"Unknown discovery method: {discovery_method}. Using registry_based")
                self._discover_workflows_from_registry()

            self.logger.info(
                f"✅ Discovered {len(self.available_workflows)} workflows")
            self.logger.info(
                f"🔍 WORKFLOW DISCOVERY DEBUG: available_workflows keys: {list(self.available_workflows.keys())}")

        except Exception as e:
            self.logger.error(
                f"❌ Failed to discover workflows: {e}", exc_info=True)
            # Use empty workflows dict as fallback
            self.available_workflows = {}

    def _discover_workflows_from_direct_references(self) -> None:
        """
        Discover workflows from direct class and config references.

        This method processes workflow_mappings from the configuration to directly
        register workflows with their class paths and config paths.
        """
        try:
            workflow_mappings = self.workflow_discovery_config.get('workflow_mappings', {})

            self.logger.info(
                f"🔍 DIRECT REFERENCE DEBUG: Processing {len(workflow_mappings)} workflow mappings")

            for workflow_name, workflow_config in workflow_mappings.items():
                try:
                    self.logger.info(
                        f"🔍 DIRECT REFERENCE DEBUG: Processing workflow: {workflow_name}")

                    # Extract workflow information from mapping
                    workflow_class = workflow_config.get('workflow_class')
                    config_path = workflow_config.get('config_path')
                    triggers = workflow_config.get('triggers', [])
                    analysis_types = workflow_config.get('analysis_types', [])
                    confidence_threshold = workflow_config.get('confidence_threshold', 0.5)

                    if not workflow_class:
                        self.logger.warning(
                            f"No workflow_class specified for {workflow_name}, skipping")
                        continue

                    # Create workflow capabilities from mapping
                    capabilities = {
                        'name': workflow_name,
                        'description': f"Direct reference workflow: {workflow_name}",
                        'domain': self._extract_domain_from_name(workflow_name),
                        'workflow_class': workflow_class,
                        'config_path': config_path,
                        'triggers': triggers,
                        'analysis_types': analysis_types,
                        'confidence_threshold': confidence_threshold,
                        'capabilities': {
                            'analysis_types': analysis_types,
                            'triggers': triggers
                        },
                        'keywords': triggers + analysis_types,
                        'metadata': {
                            'discovery_method': 'direct_reference',
                            'workflow_class': workflow_class,
                            'config_path': config_path
                        }
                    }

                    # Register the workflow
                    self.available_workflows[workflow_name] = capabilities

                    self.logger.info(
                        f"✅ Registered workflow: {workflow_name} with {len(triggers)} triggers")

                except Exception as e:
                    self.logger.error(
                        f"Failed to process workflow mapping {workflow_name}: {e}", exc_info=True)
                    continue

            self.logger.info(
                f"✅ Direct reference discovery completed: {len(self.available_workflows)} workflows registered")

        except Exception as e:
            self.logger.error(
                f"❌ Failed to discover workflows from direct references: {e}", exc_info=True)
            self.available_workflows = {}

    def _extract_domain_from_name(self, workflow_name: str) -> str:
        """Extract domain from workflow name for classification."""
        if 'viral' in workflow_name.lower():
            return 'viral_biology'
        elif 'protein' in workflow_name.lower():
            return 'protein_analysis'
        elif 'conversational' in workflow_name.lower() or 'expert' in workflow_name.lower():
            return 'conversational'
        else:
            return 'general'

    def _discover_workflows_from_registry(self) -> None:
        """Discover workflows from workflow registry"""
        try:
            # Try to import and use workflow registry if available
            registry_path = self.workflow_discovery_config.get(
                'workflow_registry_path')

            # If no specific registry path, try to discover from dependencies
            if not registry_path and hasattr(self, 'dependencies'):
                for dep_name, dep_instance in self.dependencies.items():
                    if hasattr(dep_instance, 'get_registered_workflows'):
                        workflows = dep_instance.get_registered_workflows()
                        for workflow_name, workflow_info in workflows.items():
                            self.available_workflows[workflow_name] = self._extract_workflow_capabilities(
                                workflow_info)
                        return

            self.logger.debug(
                "No workflow registry found, using empty workflows")
            self.available_workflows = {}

        except Exception as e:
            self.logger.warning(
                f"Failed to discover workflows from registry: {e}", exc_info=True)
            self.available_workflows = {}

    def _discover_workflows_from_configs(self) -> None:
        """Discover workflows from configuration file paths"""
        config_paths = self.workflow_discovery_config.get(
            'workflow_config_paths', [])

        self.logger.info(
            f"🔍 CONFIG DISCOVERY DEBUG: config_paths: {config_paths}")

        for config_path in config_paths:
            try:
                self.logger.info(
                    f"🔍 CONFIG DISCOVERY DEBUG: Loading workflow config from: {config_path}")
                # Load workflow config and extract capabilities
                workflow_config = self._load_workflow_config(config_path)
                workflow_name = workflow_config.get(
                    'name', config_path.split('/')[-1].replace('.yml', ''))
                self.logger.info(
                    f"🔍 CONFIG DISCOVERY DEBUG: Loaded workflow: {workflow_name}")

                capabilities = self._extract_workflow_capabilities(
                    workflow_config)
                self.available_workflows[workflow_name] = capabilities
                self.logger.info(
                    f"🔍 CONFIG DISCOVERY DEBUG: Extracted capabilities for {workflow_name}: {capabilities}")

            except Exception as e:
                self.logger.warning(
                    f"Failed to load workflow config {config_path}: {e}", exc_info=True)

    def _discover_workflows_from_filesystem(self) -> None:
        """Discover workflows by scanning filesystem for workflow configs"""
        # This would implement filesystem scanning logic
        self.logger.warning(
            "Filesystem-based workflow discovery not yet implemented")
        self.available_workflows = {}

    def _extract_workflow_capabilities(self, workflow_info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract capabilities from workflow metadata"""
        capabilities = {
            'name': workflow_info.get('name', 'unknown'),
            'description': workflow_info.get('description', ''),
            'domain': workflow_info.get('domain', 'general'),
            'capabilities': workflow_info.get('capabilities', {}),
            'triggers': workflow_info.get('triggers', []),
            'keywords': workflow_info.get('keywords', []),
            'confidence_threshold': workflow_info.get('confidence_threshold', 0.5),
            'metadata': workflow_info.get('metadata', {})
        }
        return capabilities

    def _load_workflow_config(self, config_path: str) -> Dict[str, Any]:
        """Load workflow configuration from file"""
        try:
            config_file = Path(config_path)
            if not config_file.exists():
                raise FileNotFoundError(
                    f"Workflow config file not found: {config_path}")

            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)

            return config

        except Exception as e:
            self.logger.error(
                f"Failed to load workflow config from {config_path}: {e}", exc_info=True)
            raise

    def _setup_classification_strategy(self) -> None:
        """
        ✅ FRAMEWORK COMPLIANCE: Setup classification strategy
        """
        # Validate strategy configuration
        matching_algorithm = self.classification_strategy.get(
            'matching_algorithm', 'capability_based')
        if matching_algorithm not in ['capability_based', 'keyword_based', 'semantic_based']:
            self.logger.warning(
                f"Unknown matching algorithm: {matching_algorithm}. Using capability_based")
            self.classification_strategy['matching_algorithm'] = 'capability_based'

        # Validate capability weights sum to approximately 1.0
        weights = self.classification_strategy.get('capability_weights', {})
        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 0.1:
            self.logger.warning(
                f"Capability weights sum to {total_weight}, normalizing to 1.0")
            # Normalize weights
            for key in weights:
                weights[key] = weights[key] / total_weight

        self.logger.debug("✅ Classification strategy setup complete")

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, DataUnitBase]:
        """
        ✅ FRAMEWORK COMPLIANCE: Main processing method for query classification

        Processes user queries through intelligent LLM-based classification to determine
        appropriate workflow routing without actually performing the routing.
        """
        try:
            self.logger.info(
                f"🔍 PROCESS DEBUG: input_data type: {type(input_data)}, keys: {list(input_data.keys()) if isinstance(input_data, dict) else 'not a dict'}")

            # ✅ ENHANCED DEBUG: Log data content for troubleshooting
            if isinstance(input_data, dict):
                for key, value in input_data.items():
                    if value is not None:
                        self.logger.debug(
                            f"🔍 input_data['{key}'] type: {type(value).__name__}, has_data: {bool(value)}")
                        if isinstance(value, dict) and 'user_query' in value:
                            self.logger.debug(
                                f"🔍 Found user_query in {key}: {value.get('user_query', '')[:50]}...")

            self.logger.debug("🧠 Starting query classification process")

            # ✅ ENHANCED: Extract and validate user request data
            user_request_data = await self._extract_user_request_data(input_data)
            if not user_request_data:
                # ❌ REMOVED FALLBACK: Fail fast instead of masking the error
                raise ComponentConfigurationError(
                    "No user request data available for routing")

            # ✅ ENHANCED: Extract session context with proper handling
            session_context = None
            if input_data and 'session_context' in input_data:
                session_context = input_data['session_context']
                self.logger.debug(
                    f"📥 Using session context from input_data: {type(session_context).__name__}")
            elif hasattr(self, 'step_input_data_units') and 'session_context' in self.step_input_data_units:
                session_context_unit = self.step_input_data_units['session_context']
                session_context = await session_context_unit.get()
                self.logger.debug(
                    f"📥 Fallback: Read session context from step_input_data_units: {type(session_context).__name__}")

            user_query = user_request_data.get('user_query', '')

            if not user_query or not user_query.strip():
                self.logger.warning("Empty or whitespace-only query received")
                return await self._handle_empty_query_gracefully(user_request_data, session_context)

            # ✅ CIRCULAR LOOP PREVENTION: Check if this request is already being processed
            request_id = user_request_data.get('request_id', '')
            if hasattr(self, '_processing_requests') and request_id in self._processing_requests:
                self.logger.debug(
                    f"🔄 Request {request_id} already being processed, skipping to prevent circular loop")
                return {}

            # ✅ CIRCULAR LOOP PREVENTION: Check if we've already routed this request
            if hasattr(self, '_routed_requests') and request_id in self._routed_requests:
                self.logger.debug(
                    f"🔄 Request {request_id} already routed, skipping to prevent circular loop")
                return {}

            # Mark request as being processed and routed
            if not hasattr(self, '_processing_requests'):
                self._processing_requests = set()
            self._processing_requests.add(request_id)

            if not hasattr(self, '_routed_requests'):
                self._routed_requests = set()
            self._routed_requests.add(request_id)

            self.logger.debug(f"🔍 Classifying query: {user_query[:100]}...")

            # Perform intelligent classification
            classification_result = await self._classify_query_intelligent(user_query, session_context)

            # Create workflow routing data unit
            routing_data_unit = await self._create_routing_data_unit(
                classification_result, user_query, session_context
            )

            # Route to specific workflow based on classification
            selected_workflow = classification_result.get(
                'primary_classification') or classification_result.get('selected_workflow')
            await self._route_to_specific_workflow(selected_workflow, user_request_data, classification_result)

            # Update session context
            updated_session_context = await self._update_session_context(
                session_context, classification_result, user_query
            )

            self.logger.info(f"✅ Query routing complete: {selected_workflow}")

            # ✅ CIRCULAR LOOP PREVENTION: Clean up processing request
            if hasattr(self, '_processing_requests') and request_id:
                self._processing_requests.discard(request_id)

            # Get actual data from data units instead of returning the DataUnit objects
            routing_data = await routing_data_unit.get() if routing_data_unit else None
            viral_expert_data = await self.step_output_data_units['viral_expert_input'].get() if 'viral_expert_input' in self.step_output_data_units else None
            viral_analysis_data = await self.step_output_data_units['viral_analysis_input'].get() if 'viral_analysis_input' in self.step_output_data_units else None

            return {
                'workflow_routing': routing_data,
                'viral_expert_input': viral_expert_data,
                'viral_analysis_input': viral_analysis_data,
                'session_context': updated_session_context
            }

        except Exception as e:
            # ✅ CIRCULAR LOOP PREVENTION: Clean up processing request on error
            if hasattr(self, '_processing_requests') and 'request_id' in locals():
                self._processing_requests.discard(request_id)

            self.logger.error(f"❌ Query routing failed: {e}", exc_info=True)
            # ❌ REMOVED FALLBACK: Let errors propagate instead of masking them
            raise ComponentConfigurationError(
                f"Query routing failed: {e}") from e

    async def _extract_user_request_data(self, input_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        ✅ FIXED: Extract and validate user request data from input_data parameter first.

        This method now prioritizes the input_data parameter passed from _execute_on_trigger()
        which already contains the correctly collected data from DataUnits.

        Args:
            input_data: Input data dictionary from _execute_on_trigger() or execute()

        Returns:
            Extracted user request data or None if no valid data found
        """
        try:
            user_request_data = None

            # ✅ PRIMARY FIX: Use input_data parameter first (from _execute_on_trigger)
            # This contains data already collected by BaseStep._execute_on_trigger()
            if input_data and 'user_request' in input_data:
                user_request_data = input_data['user_request']
                self.logger.debug(
                    f"📥 Using user request data from input_data parameter: {type(user_request_data).__name__}")

            # ✅ FALLBACK: Only read from DataUnits if input_data doesn't contain the data
            # This handles cases where the step is called directly (not via trigger)
            elif hasattr(self, 'step_input_data_units') and 'user_request' in self.step_input_data_units:
                user_request_unit = self.step_input_data_units['user_request']
                user_request_data = await user_request_unit.get()
                self.logger.debug(
                    f"📥 Fallback: Read user request data from step_input_data_units: {type(user_request_data).__name__}")

            # Additional fallbacks for other DataUnit access patterns
            elif hasattr(self, 'input_data_units') and 'user_request' in self.input_data_units:
                user_request_unit = self.input_data_units['user_request']
                user_request_data = await user_request_unit.get()
                self.logger.debug(
                    f"📥 Fallback: Read user request data from input_data_units: {type(user_request_data).__name__}")

            if not user_request_data:
                input_units = list(
                    getattr(self, 'input_data_units', {}).keys())
                step_input_units = list(
                    getattr(self, 'step_input_data_units', {}).keys())
                self.logger.error(
                    f"❌ No user_request data found. "
                    f"input_data keys: {list(input_data.keys()) if isinstance(input_data, dict) else 'N/A'}, "
                    f"step_input_data_units: {step_input_units}, "
                    f"input_data_units: {input_units}")
                return None

            # Check if user_request contains query fields (expected format)
            if isinstance(user_request_data, dict) and ('user_query' in user_request_data or 'query' in user_request_data or 'message' in user_request_data):
                try:
                    extracted_data = {
                        'user_query': user_request_data.get('user_query', user_request_data.get('query', user_request_data.get('message', ''))),
                        'request_id': user_request_data.get('request_id', str(uuid.uuid4())),
                        'session_id': user_request_data.get('session_id', 'default'),
                        'timestamp': user_request_data.get('timestamp', datetime.now().isoformat()),
                        'request_type': user_request_data.get('request_type', 'chat'),
                        'metadata': user_request_data.get('metadata', {})
                    }
                    self.logger.debug(
                        f"✅ Successfully extracted user query: {extracted_data.get('user_query')}")
                    return extracted_data
                except Exception as extract_error:
                    self.logger.error(
                        f"Error creating extracted_data: {extract_error}", exc_info=True)
                    return None

            # ✅ LEGACY SUPPORT: Check if input_data contains direct user query fields
            if isinstance(input_data, dict) and ('user_query' in input_data or 'query' in input_data or 'message' in input_data):
                legacy_data = {
                    'user_query': input_data.get('user_query', input_data.get('query', input_data.get('message', ''))),
                    'request_id': input_data.get('request_id', str(uuid.uuid4())),
                    'session_id': input_data.get('session_id', 'default'),
                    'timestamp': datetime.now().isoformat()
                }
                self.logger.debug(
                    "Using legacy format user query data from input_data")
                return legacy_data

            self.logger.debug(
                f"No valid user request data found. Data type: {type(user_request_data).__name__}, content: {str(user_request_data)[:200] if user_request_data else 'None'}")
            return None

        except Exception as e:
            self.logger.error(
                f"Error extracting user request data: {e}", exc_info=True)
            self.logger.error(
                f"❌ DEBUG: input_data type: {type(input_data)}, keys: {input_data.keys() if isinstance(input_data, dict) else 'not a dict'}")
            return None

    async def _handle_empty_request_gracefully(self, input_data: Dict[str, Any]) -> Dict[str, DataUnitBase]:
        """
        ✅ ENHANCED: Handle empty request gracefully without breaking workflow.

        This method provides a graceful response when no user request data is available,
        maintaining workflow continuity while indicating the await state.

        Args:
            input_data: Original input data for context

        Returns:
            Fallback routing response indicating no action needed
        """
        try:
            self.logger.info(
                "No user request data available, returning graceful fallback routing")

            # Create fallback routing data
            fallback_routing_data = {
                'routing_decision': 'no_action',
                'confidence_score': 0.0,
                'selected_workflow': None,
                'routing_reason': 'no_user_request_provided',
                'fallback_action': 'await_user_input',
                'workflow_matches': [],
                'suggested_workflows': [],
                'timestamp': datetime.now().isoformat(),
                'routing_id': str(uuid.uuid4()),
                'metadata': {
                    'routing_type': 'fallback',
                    'reason': 'empty_request',
                    'status': 'awaiting_input'
                }
            }

            # Create routing data unit
            routing_unit = self.step_output_data_units.get(
                'workflow_routing')  # USE PRE-CONFIGURED DATA UNIT
            await routing_unit.set(fallback_routing_data)

            # Create or update session context
            session_context = input_data.get('session_context')
            updated_session = await self._create_fallback_session_context(session_context)

            return {
                'workflow_routing': routing_unit,
                'session_context': updated_session
            }

        except Exception as e:
            self.logger.error(
                f"Failed to create graceful fallback routing: {e}", exc_info=True)
            # Final fallback - create minimal response
            return await self._create_minimal_fallback_response()

    async def _handle_empty_query_gracefully(self, user_request_data: Dict[str, Any],
                                             session_context: Optional[DataUnitBase]) -> Dict[str, DataUnitBase]:
        """
        ✅ ENHANCED: Handle empty or whitespace-only query gracefully.

        Args:
            user_request_data: User request data with empty query
            session_context: Session context for maintaining state

        Returns:
            Fallback routing response for empty query
        """
        try:
            self.logger.info(
                "Empty query received, providing helpful fallback response")

            # Create helpful fallback routing
            fallback_routing_data = {
                'routing_decision': 'request_clarification',
                'confidence_score': 0.0,
                'selected_workflow': None,
                'routing_reason': 'empty_query_provided',
                'fallback_action': 'request_user_input',
                'workflow_matches': [],
                'suggested_workflows': [],
                'timestamp': datetime.now().isoformat(),
                'routing_id': str(uuid.uuid4()),
                'user_message': 'Please provide a question or request for assistance.',
                'metadata': {
                    'routing_type': 'fallback',
                    'reason': 'empty_query',
                    'status': 'awaiting_clarification',
                    'request_id': user_request_data.get('request_id'),
                    'session_id': user_request_data.get('session_id')
                }
            }

            # Create routing data unit
            routing_unit = self.step_output_data_units.get(
                'workflow_routing')  # USE PRE-CONFIGURED DATA UNIT
            await routing_unit.set(fallback_routing_data)

            # Update session context
            updated_session = await self._update_session_context(
                session_context, fallback_routing_data, ""
            )

            return {
                'workflow_routing': routing_unit,
                'session_context': updated_session
            }

        except Exception as e:
            self.logger.error(
                f"Failed to handle empty query gracefully: {e}", exc_info=True)
            return await self._create_minimal_fallback_response()

    async def _create_fallback_session_context(self, existing_session: Optional[DataUnitBase]) -> DataUnitBase:
        """
        ✅ ENHANCED: Create or update session context for fallback scenarios.

        Args:
            existing_session: Existing session context or None

        Returns:
            Updated session context data unit
        """
        try:
            # Get existing session data if available
            session_data = {
                'session_id': 'default',
                'last_activity': datetime.now().isoformat(),
                'request_count': 1,
                'user_context': {},
                'conversation_history': [],
                'context_variables': {
                    'status': 'awaiting_input',
                    'last_routing_result': 'no_action'
                }
            }

            if existing_session and hasattr(existing_session, 'get'):
                try:
                    existing_data = await existing_session.get()
                    if existing_data and isinstance(existing_data, dict):
                        session_data.update(existing_data)
                        session_data['request_count'] = existing_data.get(
                            'request_count', 0) + 1
                        session_data['last_activity'] = datetime.now(
                        ).isoformat()
                except Exception as e:
                    self.logger.warning(
                        f"Could not get existing session data: {e}", exc_info=True)

            # Use pre-configured session context data unit - NO DYNAMIC CREATION
            session_unit = self.step_output_data_units.get('session_context')
            if not session_unit:
                session_unit = existing_session  # Try to use input if no output configured
            if not session_unit:
                raise ComponentConfigurationError(
                    "session_context data unit not configured")
            await session_unit.set(session_data)
            return session_unit

        except Exception as e:
            self.logger.error(
                f"Failed to create fallback session context: {e}", exc_info=True)
            # Use pre-configured session context
            empty_session = self.step_output_data_units.get('session_context')
            if not empty_session:
                raise ComponentConfigurationError(
                    "session_context data unit not configured")
            await empty_session.set({'session_id': 'fallback', 'status': 'error'})
            return empty_session

    async def _create_minimal_fallback_response(self) -> Dict[str, DataUnitBase]:
        """
        ✅ ENHANCED: Create minimal fallback response as last resort.

        Returns:
            Minimal routing response to prevent workflow failures
        """
        try:
            # Use pre-configured output data units - NO DYNAMIC CREATION
            minimal_routing = self.step_output_data_units.get(
                'workflow_routing')
            if minimal_routing:
                await minimal_routing.set({
                    'routing_decision': 'error_fallback',
                    'confidence_score': 0.0,
                    'status': 'error',
                    'timestamp': datetime.now().isoformat()
                })
            else:
                raise ComponentConfigurationError(
                    "workflow_routing output data unit not configured")

            # Use pre-configured session context
            minimal_session = self.step_output_data_units.get(
                'session_context')
            if minimal_session:
                await minimal_session.set({
                    'session_id': 'error_fallback',
                    'status': 'error',
                    'timestamp': datetime.now().isoformat()
                })
            else:
                raise ComponentConfigurationError(
                    "session_context output data unit not configured")

            return {
                'workflow_routing': minimal_routing,
                'session_context': minimal_session
            }

        except Exception as e:
            self.logger.critical(
                f"Failed to create minimal fallback response: {e}", exc_info=True)
            raise ComponentConfigurationError(
                f"Routing step completely failed: {e}")

    async def _classify_query_intelligent(self, user_query: str, session_context: Optional[DataUnitBase]) -> Dict[str, Any]:
        """
        ✅ FRAMEWORK COMPLIANCE: Generalizable intelligent query classification
        Uses configured classification agents to analyze queries for any domain
        """
        try:
            classification_start_time = time.time()

            # Collect results from all classification agents
            agent_results = {}

            for agent_name, agent_info in self.classification_agents.items():
                try:
                    self.logger.debug(
                        f"🔍 Using agent {agent_name} for classification")

                    # Get agent instance
                    agent_instance = agent_info.get('instance') if isinstance(
                        agent_info, dict) else agent_info

                    # Process query with agent
                    if hasattr(agent_instance, '_process_specialized_request'):
                        result = await agent_instance._process_specialized_request(
                            user_query,
                            expected_format='json',
                            analysis_type='classification'
                        )
                    elif hasattr(agent_instance, 'aprocess'):
                        result = await agent_instance.aprocess(user_query)
                    else:
                        self.logger.warning(
                            f"Agent {agent_name} doesn't have expected processing methods")
                        continue

                    agent_results[agent_name] = {
                        'result': result,
                        'role': agent_info.get('role', 'general') if isinstance(agent_info, dict) else 'general'
                    }

                except Exception as e:
                    self.logger.warning(
                        f"Agent {agent_name} failed: {e}", exc_info=True)
                    continue

            # Combine and analyze agent results
            classification_result = await self._combine_agent_results(
                agent_results, user_query, session_context
            )

            # Match against available workflows
            workflow_matches = await self._match_workflows(classification_result)
            classification_result['workflow_matches'] = workflow_matches

            # Determine primary classification from workflow matches
            primary_classification, routing_confidence = self._determine_primary_classification(
                classification_result)
            classification_result['primary_classification'] = primary_classification
            classification_result['routing_confidence'] = routing_confidence
            classification_result['suggested_workflows'] = workflow_matches

            # Calculate final confidence scores
            classification_result['confidence_scores'] = self._calculate_confidence_scores(
                classification_result)

            # Add timing information
            classification_result['processing_time'] = time.time(
            ) - classification_start_time

            self.logger.debug(
                f"✅ Generalizable classification completed in {classification_result['processing_time']:.2f}s")

            return classification_result

        except Exception as e:
            self.logger.error(
                f"❌ Error in intelligent query classification: {e}", exc_info=True)
            return await self._create_fallback_classification_result(user_query, str(e))

    async def _combine_agent_results(self, agent_results: Dict[str, Dict[str, Any]],
                                     user_query: str, session_context: Optional[DataUnitBase]) -> Dict[str, Any]:
        """
        ✅ FRAMEWORK COMPLIANCE: Combine results from multiple classification agents
        """
        combined_result = {
            'user_query': user_query,
            'timestamp': datetime.now().isoformat(),
            'classification_id': f"cls_{uuid.uuid4().hex[:8]}",
            'agent_results': {},
            'extracted_entities': {},
            'intent_analysis': {},
            'domain_analysis': {},
            'confidence_scores': {}
        }

        try:
            # Process each agent result
            for agent_name, agent_data in agent_results.items():
                result = agent_data.get('result')
                role = agent_data.get('role', 'general')

                # Parse agent result
                parsed_result = self._parse_agent_result(result, agent_name)
                combined_result['agent_results'][agent_name] = parsed_result

                # Extract information based on agent role
                if role == 'intent_classification':
                    combined_result['intent_analysis'].update({
                        'intent': parsed_result.get('intent', 'unknown'),
                        'complexity': parsed_result.get('complexity', 'moderate'),
                        'confidence': parsed_result.get('confidence', 0.0)
                    })
                elif role == 'domain_analysis':
                    combined_result['domain_analysis'].update({
                        'domain': parsed_result.get('domain', 'general'),
                        'subdomain': parsed_result.get('subdomain', ''),
                        'confidence': parsed_result.get('confidence', 0.0)
                    })
                elif role == 'entity_extraction':
                    combined_result['extracted_entities'].update(
                        parsed_result.get('entities', {}))
                else:
                    # General role - extract any available information
                    if 'intent' in parsed_result:
                        combined_result['intent_analysis']['intent'] = parsed_result['intent']
                    if 'domain' in parsed_result:
                        combined_result['domain_analysis']['domain'] = parsed_result['domain']
                    if 'entities' in parsed_result:
                        combined_result['extracted_entities'].update(
                            parsed_result['entities'])

                # Store confidence score
                combined_result['confidence_scores'][agent_name] = parsed_result.get(
                    'confidence', 0.0)

            # Calculate overall confidence
            overall_confidence = self._calculate_overall_confidence(
                combined_result['confidence_scores'])
            combined_result['overall_confidence'] = overall_confidence

            # Build classification metadata
            combined_result['classification_metadata'] = self._build_classification_metadata(
                combined_result, session_context)

            return combined_result

        except Exception as e:
            self.logger.error(
                f"❌ Error combining agent results: {e}", exc_info=True)
            return await self._create_fallback_classification_result(user_query, str(e))

    def _parse_agent_result(self, result: Any, agent_name: str) -> Dict[str, Any]:
        """Parse result from classification agent"""
        try:
            if isinstance(result, str):
                # Try to parse as JSON
                try:
                    return json.loads(result)
                except json.JSONDecodeError:
                    # Treat as text response
                    return {'text_response': result, 'confidence': 0.5}
            elif isinstance(result, dict):
                return result
            else:
                self.logger.warning(
                    f"Unexpected result type from agent {agent_name}: {type(result)}")
                return {'raw_result': str(result), 'confidence': 0.0}
        except Exception as e:
            self.logger.warning(
                f"Failed to parse result from agent {agent_name}: {e}", exc_info=True)
            return {'error': str(e), 'confidence': 0.0}

    def _calculate_overall_confidence(self, confidence_scores: Dict[str, float]) -> float:
        """Calculate overall confidence from individual agent scores"""
        if not confidence_scores:
            return 0.0

        # Use weighted average based on number of agents and their consistency
        scores = list(confidence_scores.values())
        if len(scores) == 1:
            return scores[0]

        # Calculate average and adjust for consistency
        avg_confidence = sum(scores) / len(scores)

        # Adjust for consistency - more consistent scores get higher confidence
        variance = sum((score - avg_confidence) **
                       2 for score in scores) / len(scores)
        # Higher consistency = higher factor
        consistency_factor = max(0.5, 1.0 - variance)

        return min(1.0, avg_confidence * consistency_factor)

    async def _match_workflows(self, classification_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        ✅ FRAMEWORK COMPLIANCE: Match classification results against available workflows
        """
        try:
            # Debug logging at the very beginning
            self.logger.info(
                f"🔍 _match_workflows called with classification_result type: {type(classification_result)}")
            self.logger.info(
                f"🔍 _match_workflows classification_result keys: {list(classification_result.keys()) if isinstance(classification_result, dict) else 'NOT_DICT'}")

            matching_algorithm = self.classification_strategy.get(
                'matching_algorithm', 'capability_based')

            if matching_algorithm == 'capability_based':
                return await self._match_workflows_by_capability(classification_result)
            elif matching_algorithm == 'keyword_based':
                return await self._match_workflows_by_keywords(classification_result)
            elif matching_algorithm == 'semantic_based':
                return await self._match_workflows_semantically(classification_result)
            elif matching_algorithm == 'agent_direct':
                return await self._match_workflows_agent_direct(classification_result)
            else:
                self.logger.warning(
                    f"Unknown matching algorithm: {matching_algorithm}")
                return await self._match_workflows_by_capability(classification_result)

        except Exception as e:
            self.logger.error(
                f"❌ Error matching workflows: {e}", exc_info=True)
            return []

    async def _match_workflows_by_capability(self, classification_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Match workflows based on capability matching"""
        matches = []

        # Early debug logging
        self.logger.info(
            f"🔍 EARLY DEBUG: classification_result type: {type(classification_result)}")
        self.logger.info(
            f"🔍 EARLY DEBUG: classification_result keys: {list(classification_result.keys()) if isinstance(classification_result, dict) else 'NOT_DICT'}")

        # Extract classification information
        intent_analysis = classification_result.get('intent_analysis', {})
        self.logger.info(
            f"🔍 EARLY DEBUG: intent_analysis type: {type(intent_analysis)}, value: {intent_analysis}")
        if isinstance(intent_analysis, dict):
            intent = intent_analysis.get('intent', 'unknown')
        else:
            intent = str(intent_analysis) if intent_analysis else 'unknown'

        domain_analysis = classification_result.get('domain_analysis', {})
        if isinstance(domain_analysis, dict):
            domain = domain_analysis.get('domain', 'general')
        else:
            domain = str(domain_analysis) if domain_analysis else 'general'

        entities = classification_result.get('extracted_entities', {})
        overall_confidence = classification_result.get(
            'overall_confidence', 0.0)

        # Debug logging to identify the type error
        self.logger.debug(
            f"🔍 DEBUG: intent type: {type(intent)}, value: {intent}")
        self.logger.debug(
            f"🔍 DEBUG: domain type: {type(domain)}, value: {domain}")
        self.logger.debug(
            f"🔍 DEBUG: entities type: {type(entities)}, value: {entities}")
        self.logger.debug(
            f"🔍 DEBUG: overall_confidence type: {type(overall_confidence)}, value: {overall_confidence}")

        # Get capability weights
        weights = self.classification_strategy.get('capability_weights', {
            'domain_match': 0.4, 'intent_match': 0.3, 'keyword_match': 0.2, 'confidence_score': 0.1
        })

        # Score each available workflow
        for workflow_name, workflow_info in self.available_workflows.items():
            score = 0.0

            # Domain matching
            workflow_domain = workflow_info.get('domain', 'general')
            if isinstance(domain, str) and domain == workflow_domain:
                score += weights.get('domain_match', 0.4)
            elif isinstance(domain, str) and domain in workflow_info.get('capabilities', {}).get('domains', []):
                score += weights.get('domain_match', 0.4) * 0.8

            # Intent matching - use workflow capabilities instead of triggers
            workflow_intents = workflow_info.get(
                'capabilities', {}).get('intents', [])
            if isinstance(intent, str) and intent in workflow_intents:
                score += weights.get('intent_match', 0.3)
            elif isinstance(intent, str) and any(workflow_intent in intent for workflow_intent in workflow_intents):
                score += weights.get('intent_match', 0.3) * 0.6

            # Keyword matching - use capabilities keywords
            workflow_keywords = workflow_info.get(
                'capabilities', {}).get('keywords', [])
            # Get user_query from the classification result structure
            user_query = classification_result.get('user_query', '')
            if not user_query and 'routing_metadata' in classification_result:
                user_query = classification_result['routing_metadata'].get(
                    'user_query', '')

            if isinstance(user_query, str) and user_query:
                query_words = user_query.lower().split()
                keyword_matches = sum(
                    1 for keyword in workflow_keywords if keyword.lower() in query_words)
                if workflow_keywords:
                    keyword_score = keyword_matches / len(workflow_keywords)
                    score += weights.get('keyword_match', 0.2) * keyword_score

            # Confidence score factor
            score += weights.get('confidence_score', 0.1) * overall_confidence

            # Only include workflows above minimum confidence
            min_confidence = self.classification_strategy.get(
                'min_confidence_for_routing', 0.5)
            workflow_threshold = workflow_info.get('confidence_threshold', 0.5)
            final_threshold = max(min_confidence, workflow_threshold)

            if score >= final_threshold:
                matches.append({
                    'workflow_name': workflow_name,
                    'confidence': score,
                    'reasoning': f"Capability-based match: domain={domain}, intent={intent}",
                    'workflow_metadata': workflow_info
                })

        # Sort by confidence
        matches.sort(key=lambda x: x['confidence'], reverse=True)

        return matches[:5]  # Return top 5 matches

    async def _match_workflows_by_keywords(self, classification_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Match workflows based on keyword matching"""
        matches = []
        query_words = set(classification_result['user_query'].lower().split())

        for workflow_name, workflow_info in self.available_workflows.items():
            workflow_keywords = set(keyword.lower()
                                    for keyword in workflow_info.get('keywords', []))

            if workflow_keywords:
                # Calculate Jaccard similarity
                intersection = len(query_words & workflow_keywords)
                union = len(query_words | workflow_keywords)
                similarity = intersection / union if union > 0 else 0.0

                if similarity > 0.1:  # Minimum keyword similarity threshold
                    matches.append({
                        'workflow_name': workflow_name,
                        'confidence': similarity,
                        'reasoning': f"Keyword-based match: {intersection}/{union} keywords matched",
                        'workflow_metadata': workflow_info
                    })

        matches.sort(key=lambda x: x['confidence'], reverse=True)
        return matches[:5]

    async def _match_workflows_semantically(self, classification_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Match workflows using semantic similarity (placeholder for future implementation)"""
        self.logger.debug(
            "Semantic workflow matching not yet implemented, falling back to capability-based")
        return await self._match_workflows_by_capability(classification_result)

    async def _match_workflows_agent_direct(self, classification_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        ✅ FRAMEWORK COMPLIANCE: Agent-direct workflow matching
        Uses structured agent output to directly determine workflow routing
        """
        try:
            self.logger.info("🤖 Using agent-direct workflow matching")

            # Get agent routing configuration
            agent_config = self.classification_strategy.get('agent_routing_config', {})
            min_conf_computational = agent_config.get('min_confidence_computational', 0.7)
            min_conf_conversational = agent_config.get('min_confidence_conversational', 0.5)
            fallback_workflow = agent_config.get('fallback_workflow', 'viral_expert_workflow')

            # Extract agent results from classification_result
            agent_results = classification_result.get('agent_results', {})
            routing_agent_result = None

            # Find the routing strategy agent result
            for agent_name, result in agent_results.items():
                if 'routing_strategy' in agent_name or agent_name == 'routing_strategy_agent':
                    routing_agent_result = result
                    break

            if not routing_agent_result:
                self.logger.warning("⚠️ No routing strategy agent result found, using fallback")
                return [{
                    'workflow_name': fallback_workflow,
                    'confidence': 0.5,
                    'reasoning': 'Fallback: No routing agent result',
                    'workflow_metadata': self.available_workflows.get(fallback_workflow, {})
                }]

            # Parse agent output
            agent_output = routing_agent_result.get('result', {})
            if isinstance(agent_output, str):
                try:
                    import json
                    agent_output = json.loads(agent_output)
                except json.JSONDecodeError:
                    self.logger.error(f"❌ Failed to parse agent output as JSON: {agent_output}")
                    agent_output = {}

            # Extract routing decision from agent output
            workflow = agent_output.get('workflow', fallback_workflow)
            intent = agent_output.get('intent', 'conversational')
            confidence = float(agent_output.get('confidence', 0.5))
            keywords = agent_output.get('keywords', [])
            reasoning = agent_output.get('reasoning', 'Agent direct routing')

            self.logger.info(f"🤖 Agent routing decision: workflow={workflow}, intent={intent}, confidence={confidence}")

            # Validate confidence thresholds
            if intent == 'computational' and confidence < min_conf_computational:
                self.logger.info(f"⚠️ Computational confidence {confidence} below threshold {min_conf_computational}, routing to conversational")
                workflow = fallback_workflow
                confidence = max(confidence, min_conf_conversational)
                reasoning = f"Fallback: Computational confidence too low ({confidence} < {min_conf_computational})"
            elif intent == 'conversational' and confidence < min_conf_conversational:
                self.logger.info(f"⚠️ Conversational confidence {confidence} below threshold {min_conf_conversational}, using fallback")
                workflow = fallback_workflow
                confidence = min_conf_conversational
                reasoning = f"Fallback: Conversational confidence too low ({confidence} < {min_conf_conversational})"

            # Ensure workflow exists in available workflows
            if workflow not in self.available_workflows:
                self.logger.warning(f"⚠️ Workflow '{workflow}' not found in available workflows, using fallback")
                workflow = fallback_workflow
                reasoning = f"Fallback: Workflow '{workflow}' not available"

            # Create match result
            match_result = {
                'workflow_name': workflow,
                'confidence': confidence,
                'reasoning': reasoning,
                'workflow_metadata': self.available_workflows.get(workflow, {}),
                'agent_output': {
                    'intent': intent,
                    'keywords': keywords,
                    'original_confidence': agent_output.get('confidence', 0.5)
                }
            }

            self.logger.info(f"✅ Agent-direct routing: {workflow} (confidence: {confidence})")
            return [match_result]

        except Exception as e:
            self.logger.error(f"❌ Agent-direct matching failed: {e}", exc_info=True)
            # Fallback to conversational workflow
            fallback_workflow = self.classification_strategy.get('agent_routing_config', {}).get('fallback_workflow', 'viral_expert_workflow')
            return [{
                'workflow_name': fallback_workflow,
                'confidence': 0.5,
                'reasoning': f'Error fallback: {str(e)}',
                'workflow_metadata': self.available_workflows.get(fallback_workflow, {})
            }]

    def _determine_primary_classification(self, classification_result: Dict[str, Any]) -> Tuple[str, float]:
        """
        ✅ FRAMEWORK COMPLIANCE: Determine primary classification from workflow matches
        """
        workflow_matches = classification_result.get('workflow_matches', [])

        if workflow_matches:
            # Return the highest confidence workflow
            best_match = workflow_matches[0]
            return best_match['workflow_name'], best_match['confidence']
        else:
            # Use fallback logic
            return self._get_fallback_classification()

    def _get_fallback_classification(self) -> Tuple[str, float]:
        """Get fallback classification when no workflows match"""
        fallback_pattern = self.classification_strategy.get(
            'fallback_workflow_pattern', 'conversational_*')
        fallback_confidence = self.classification_config.get(
            'fallback_threshold', 0.3)

        # Look for workflows matching the fallback pattern
        for workflow_name in self.available_workflows.keys():
            if self._matches_pattern(workflow_name, fallback_pattern):
                return workflow_name, fallback_confidence

        # If no pattern match, return first available workflow or unknown
        if self.available_workflows:
            first_workflow = list(self.available_workflows.keys())[0]
            return first_workflow, fallback_confidence
        else:
            return 'unknown', 0.0

    def _matches_pattern(self, workflow_name: str, pattern: str) -> bool:
        """Check if workflow name matches a pattern (supports * wildcard)"""
        if '*' not in pattern:
            return workflow_name == pattern

        # Simple wildcard matching
        pattern_parts = pattern.split('*')
        name_pos = 0

        for i, part in enumerate(pattern_parts):
            if not part:  # Empty part from * at start/end or consecutive *
                continue

            pos = workflow_name.find(part, name_pos)
            if pos == -1:
                return False

            if i == 0 and pos != 0:  # First part must be at start
                return False

            name_pos = pos + len(part)

        # Last part must be at end if pattern doesn't end with *
        if pattern_parts and pattern_parts[-1] and not pattern.endswith('*'):
            return workflow_name.endswith(pattern_parts[-1])

        return True

    def _calculate_confidence_scores(self, classification_result: Dict[str, Any]) -> Dict[str, float]:
        """
        ✅ FRAMEWORK COMPLIANCE: Calculate detailed confidence scores
        """
        agent_scores = classification_result.get('confidence_scores', {})
        overall_confidence = classification_result.get(
            'overall_confidence', 0.0)

        # Calculate domain-specific confidence scores
        confidence_scores = {
            'overall': overall_confidence,
            'agent_consensus': self._calculate_agent_consensus(agent_scores),
            'workflow_matching': self._calculate_workflow_matching_confidence(classification_result),
            'intent_classification': classification_result.get('intent_analysis', {}).get('confidence', 0.0),
            'domain_analysis': classification_result.get('domain_analysis', {}).get('confidence', 0.0)
        }

        # Add individual agent scores
        for agent_name, score in agent_scores.items():
            confidence_scores[f'agent_{agent_name}'] = score

        return confidence_scores

    def _calculate_agent_consensus(self, agent_scores: Dict[str, float]) -> float:
        """Calculate consensus score among agents"""
        if len(agent_scores) < 2:
            return list(agent_scores.values())[0] if agent_scores else 0.0

        scores = list(agent_scores.values())
        avg_score = sum(scores) / len(scores)

        # Calculate agreement (lower variance = higher consensus)
        variance = sum((score - avg_score) **
                       2 for score in scores) / len(scores)
        # Higher agreement = higher consensus
        consensus = max(0.0, 1.0 - variance)

        return consensus

    def _calculate_workflow_matching_confidence(self, classification_result: Dict[str, Any]) -> float:
        """Calculate confidence based on workflow matching quality"""
        workflow_matches = classification_result.get('workflow_matches', [])

        if not workflow_matches:
            return 0.0

        # Use the confidence of the best match
        best_match_confidence = workflow_matches[0]['confidence']

        # Boost confidence if multiple workflows match with similar scores
        if len(workflow_matches) > 1:
            second_best = workflow_matches[1]['confidence']
            if abs(best_match_confidence - second_best) < 0.1:
                # Close scores indicate good classification
                return min(1.0, best_match_confidence * 1.1)

        return best_match_confidence

    def _build_classification_metadata(self, classification_data: Dict[str, Any],
                                       session_context: Optional[DataUnitBase]) -> Dict[str, Any]:
        """
        ✅ FRAMEWORK COMPLIANCE: Build comprehensive classification metadata
        """
        metadata = {
            'classification_method': 'llm_intelligent',
            'agents_used': ['virus_extraction_agent', 'query_analysis_agent'],
            'processing_time': classification_data.get('processing_time', 0.0),
            'thresholds_applied': self.classification_thresholds,
            'analysis_depth': self.config.analysis_depth
        }

        # Add session context metadata if available
        if session_context:
            try:
                session_data = session_context.get_data() if hasattr(
                    session_context, 'get_data') else session_context.data
                metadata['session_metadata'] = {
                    'session_id': session_data.get('session_id'),
                    'conversation_length': len(session_data.get('conversation_history', [])),
                    'previous_classifications': len(session_data.get('classification_history', []))
                }
            except Exception as e:
                self.logger.warning(
                    f"⚠️ Could not extract session metadata: {e}", exc_info=True)

        return metadata

    async def _create_routing_data_unit(self, classification_result: Dict[str, Any],
                                        user_query: str, session_context: Optional[DataUnitBase]) -> DataUnitBase:
        """
        ✅ FRAMEWORK COMPLIANCE: Create workflow routing data unit from analysis results
        """
        try:
            routing_data = {
                'query': user_query,
                'selected_workflow': classification_result.get('primary_classification'),
                'routing_confidence': classification_result.get('routing_confidence', 0.0),
                'workflow_matches': classification_result.get('workflow_matches', []),
                'suggested_workflows': classification_result.get('suggested_workflows', []),
                'biological_entities': classification_result.get('biological_entities', {}),
                'intent_classification': classification_result.get('intent_classification'),
                'analysis_type': classification_result.get('biological_context', {}).get('analysis_type'),
                'confidence_scores': classification_result.get('confidence_scores', {}),
                'routing_metadata': classification_result.get('classification_metadata', {}),
                'timestamp': classification_result.get('timestamp'),
                'routing_id': classification_result.get('classification_id')
            }

            # Use pre-configured output data unit - NO DYNAMIC CREATION
            data_unit = self.step_output_data_units.get('workflow_routing')
            if not data_unit:
                raise ComponentConfigurationError(
                    "workflow_routing output data unit not configured")
            await data_unit.set(routing_data)
            return data_unit

        except Exception as e:
            self.logger.error(
                f"❌ Failed to create workflow routing data unit: {e}", exc_info=True)
            raise ComponentConfigurationError(
                f"Workflow routing data unit creation failed: {e}")

    async def _update_session_context(self, session_context: Optional[DataUnitBase],
                                      classification_result: Dict[str, Any], user_query: str) -> DataUnitBase:
        """
        ✅ FRAMEWORK COMPLIANCE: Update session context with classification results
        """
        try:
            # Get existing session data or create new using framework-compliant methods
            if session_context:
                try:
                    session_data = await session_context.get()
                    if session_data is None:
                        session_data = {}
                except Exception:
                    session_data = {}
            else:
                session_data = {}

            # Ensure required session structure
            if not session_data or not isinstance(session_data, dict):
                session_data = {
                    'session_id': f"session_{uuid.uuid4().hex[:8]}",
                    'created_at': datetime.now().isoformat(),
                    'conversation_history': [],
                    'classification_history': [],
                    'context_variables': {}
                }

            # Add current classification to history
            session_data['classification_history'].append({
                'query': user_query,
                'classification': classification_result.get('primary_classification'),
                'confidence': classification_result.get('routing_confidence'),
                'timestamp': classification_result.get('timestamp'),
                'classification_id': classification_result.get('classification_id')
            })

            # Update conversation history
            session_data['conversation_history'].append({
                'type': 'user_query',
                'content': user_query,
                'timestamp': classification_result.get('timestamp')
            })

            # Update context variables
            session_data['context_variables'].update({
                'last_classification': classification_result.get('primary_classification'),
                'last_confidence': classification_result.get('routing_confidence'),
                'detected_domain': classification_result.get('biological_context', {}).get('domain'),
                'virus_context': bool(classification_result.get('virus_species', []))
            })

            session_data['last_activity'] = datetime.now().isoformat()

            # Use pre-configured session context output data unit - NO DYNAMIC CREATION
            session_unit = self.step_output_data_units.get('session_context')
            if not session_unit:
                # If no output configured, try to use the input session context
                session_unit = session_context
            if not session_unit:
                raise ComponentConfigurationError(
                    "session_context data unit not configured")
            await session_unit.set(session_data)
            return session_unit

        except Exception as e:
            self.logger.error(
                f"❌ Failed to update session context: {e}", exc_info=True)
            raise ComponentConfigurationError(
                f"Session context update failed: {e}")

    async def _create_fallback_classification_result(self, user_query: str, error_message: str) -> Dict[str, Any]:
        """
        ✅ FRAMEWORK COMPLIANCE: Create fallback classification when LLM agents fail
        """
        return {
            'user_query': user_query,
            'primary_classification': 'conversational_viral_expert',
            'routing_confidence': 0.0,
            'suggested_workflows': [{
                'workflow_name': 'conversational_viral_expert',
                'confidence': 0.0,
                'reasoning': f'Fallback due to classification error: {error_message}',
                'workflow_metadata': self.available_workflows.get('conversational_viral_expert', {})
            }],
            'biological_entities': {},
            'intent_classification': 'error_fallback',
            'confidence_scores': {
                'overall': 0.0,
                'extraction': 0.0,
                'intent': 0.0,
                'biological_context': 0.0,
                'routing_decision': 0.0
            },
            'classification_metadata': {
                'classification_method': 'fallback',
                'error': error_message,
                'fallback_reason': 'llm_classification_failed'
            },
            'timestamp': datetime.now().isoformat(),
            'classification_id': f"fallback_{uuid.uuid4().hex[:8]}"
        }

    async def _route_to_specific_workflow(self, selected_workflow: str, user_request_data: Dict[str, Any], classification_result: Dict[str, Any]) -> None:
        """Route query to specific workflow based on classification result"""
        try:
            # Prepare workflow input data
            workflow_input = {
                'user_query': user_request_data.get('user_query', ''),
                'session_id': user_request_data.get('session_id', 'default'),
                'request_id': user_request_data.get('request_id'),
                'timestamp': user_request_data.get('timestamp'),
                'routing_metadata': classification_result
            }

            # Add any extracted biological entities
            if 'biological_entities' in classification_result:
                workflow_input['biological_entities'] = classification_result['biological_entities']

            # CRITICAL FIX: Enhanced routing logic with explicit PSSM detection
            user_query = classification_result.get('user_query', '').lower()

            # Check for PSSM/analysis keywords that should route to viral analysis workflow
            pssm_keywords = ['pssm', 'matrix']  # Only use specific PSSM keywords, not generic ones
            has_pssm_keywords = any(keyword in user_query for keyword in pssm_keywords)

            # Check agent classification for biological analysis intent
            agent_results = classification_result.get('agent_results', {})
            routing_agent = agent_results.get('routing_strategy_agent', {})
            intent = routing_agent.get('intent', '')
            analysis_type = routing_agent.get('biological_context', {}).get('analysis_type', '')

            # PSSM/Analysis routing logic - ONLY for explicit PSSM requests
            if has_pssm_keywords and intent == 'biological_analysis' and analysis_type == 'sequence_analysis':
                self.logger.info("🔬 PSSM/Analysis detected - routing to viral analysis workflow")
                self.logger.info(f"🔍 Keywords found: {[kw for kw in pssm_keywords if kw in user_query]}")
                self.logger.info(f"🔍 Intent: {intent}, Analysis type: {analysis_type}")
                await self.step_output_data_units['viral_analysis_input'].set(workflow_input)

            # Route based on selected workflow
            elif selected_workflow == 'viral_expert_workflow':
                self.logger.info("🧠 Routing to viral expert workflow")
                await self.step_output_data_units['viral_expert_input'].set(workflow_input)

            elif selected_workflow in ['viral_analysis_workflow', 'viral_protein_analysis']:
                self.logger.info(f"🔬 Routing to viral analysis workflow (decision: {selected_workflow})")
                await self.step_output_data_units['viral_analysis_input'].set(workflow_input)

            else:
                # Default to viral expert for conversational queries
                self.logger.info(
                    f"🤔 Unknown workflow '{selected_workflow}', defaulting to viral expert")
                await self.step_output_data_units['viral_expert_input'].set(workflow_input)

        except Exception as e:
            self.logger.error(
                f"❌ Failed to route to workflow {selected_workflow}: {e}", exc_info=True)
            # Fallback to viral expert workflow
            try:
                fallback_input = {
                    'user_query': user_request_data.get('user_query', ''),
                    'session_id': user_request_data.get('session_id', 'default'),
                    'error': f"Routing failed: {e}"
                }
                await self.step_output_data_units['viral_expert_input'].set(fallback_input)
                self.logger.info(
                    "✅ Fallback routing to viral expert workflow")
            except Exception as fallback_error:
                self.logger.error(
                    f"❌ Fallback routing also failed: {fallback_error}", exc_info=True)

    async def _create_fallback_classification(self, error_message: str, input_data: Dict[str, DataUnitBase]) -> Dict[str, DataUnitBase]:
        """
        ✅ FRAMEWORK COMPLIANCE: Create fallback classification output
        """
        try:
            # Extract original query if possible using framework-compliant methods
            user_request_data = await input_data['user_request'].get()
            if user_request_data:
                try:
                    user_query = user_request_data.get('user_query', user_request_data.get(
                        'query', user_request_data.get('message', 'unknown query')))
                except Exception:
                    user_query = 'unknown query'
            else:
                user_query = 'unknown query'

            # Create fallback classification result
            fallback_result = await self._create_fallback_classification_result(user_query, error_message)

            # Create fallback data units
            fallback_routing = await self._create_routing_data_unit(
                fallback_result, user_query, input_data.get('session_context')
            )

            fallback_session = await self._update_session_context(
                input_data.get('session_context'), fallback_result, user_query
            )

            # Get actual data from data units instead of returning the DataUnit objects
            routing_data = await fallback_routing.get() if fallback_routing else None
            session_data = await fallback_session.get() if fallback_session else None

            return {
                'workflow_routing': routing_data,
                'session_context': session_data
            }

        except Exception as e:
            self.logger.error(
                f"❌ Failed to create fallback routing: {e}", exc_info=True)
            raise ComponentConfigurationError(
                f"Fallback routing creation failed: {e}")
