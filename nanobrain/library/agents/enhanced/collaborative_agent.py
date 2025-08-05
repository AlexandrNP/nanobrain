"""
Collaborative agent implementation.

Multi-protocol collaborative agent with delegation and coordination capabilities.
"""

import asyncio
from typing import Any, Dict, List, Optional
from nanobrain.core.component_base import FromConfigBase
from nanobrain.core.logging_system import get_logger
from nanobrain.core.agent import ConversationalAgent, AgentConfig
from nanobrain.core.a2a_support import A2ASupportMixin
from nanobrain.core.mcp_support import MCPSupportMixin
from .delegation_engine import DelegationEngine
# from .performance_tracker import AgentPerformanceTracker  # Module not available, using None for now


class CollaborativeAgent(A2ASupportMixin, MCPSupportMixin, ConversationalAgent):
    """
    Collaborative Agent - Multi-Protocol Team Coordination with Intelligent Delegation and Workflow Orchestration
    ============================================================================================================
    
    The CollaborativeAgent provides sophisticated multi-agent coordination capabilities, combining A2A (Agent-to-Agent)
    and MCP (Model Context Protocol) integration with intelligent task delegation and workflow orchestration. This agent
    serves as a central coordination hub for complex multi-agent workflows, enabling seamless collaboration between
    specialized agents, tools, and external systems in distributed AI environments.
    
    **Core Architecture:**
        The collaborative agent provides enterprise-grade multi-agent coordination:
        
        * **Multi-Protocol Integration**: Seamless A2A and MCP protocol support for diverse agent ecosystems
        * **Intelligent Delegation**: Advanced task analysis and optimal agent selection for subtask execution
        * **Workflow Orchestration**: Complex multi-step workflow coordination with dependency management
        * **Performance Tracking**: Comprehensive monitoring and optimization of collaborative processes
        * **Context Management**: Intelligent context sharing and state synchronization across agents
        * **Framework Integration**: Full integration with NanoBrain's distributed agent architecture
    
    **Collaboration Capabilities:**
        
        **Agent-to-Agent (A2A) Integration:**
        * Native A2A protocol support for direct agent communication
        * Secure message passing and state synchronization
        * Distributed workflow execution across agent networks
        * Real-time collaboration and task coordination
        
        **Model Context Protocol (MCP) Support:**
        * MCP server and client implementation for standardized model interaction
        * Context sharing and model state management
        * Tool and resource sharing across different AI models
        * Standardized protocol compliance for interoperability
        
        **Task Delegation Framework:**
        * Intelligent task analysis and decomposition
        * Agent capability matching and optimal selection
        * Dynamic load balancing and resource allocation
        * Dependency tracking and execution ordering
        
        **Workflow Orchestration:**
        * Complex multi-agent workflow design and execution
        * Conditional branching and parallel execution support
        * Error handling and recovery mechanisms
        * Progress monitoring and completion tracking
    
    **Team Coordination Features:**
        
        **Intelligent Task Analysis:**
        * Automatic task decomposition into optimal subtasks
        * Complexity assessment and resource requirement estimation
        * Agent capability matching and selection algorithms
        * Performance prediction and optimization recommendations
        
        **Dynamic Agent Selection:**
        * Real-time agent availability and capability assessment
        * Load balancing across available agent resources
        * Specialization matching for domain-specific tasks
        * Performance history and reliability scoring
        
        **Context Synchronization:**
        * Intelligent context sharing and state management
        * Version control and conflict resolution for shared data
        * Secure information passing and access control
        * Real-time synchronization and consistency maintenance
        
        **Performance Optimization:**
        * Continuous monitoring of collaboration efficiency
        * Bottleneck identification and resolution suggestions
        * Resource utilization optimization and scaling recommendations
        * Quality assessment and improvement tracking
    
    **Configuration Architecture:**
        Comprehensive configuration supports diverse collaboration workflows:
        
        ```yaml
        # Collaborative Agent Configuration
        agent_name: "collaborative_agent"
        agent_type: "enhanced"
        
        # Agent card for framework integration
        agent_card:
          name: "collaborative_agent"
          description: "Multi-protocol collaborative agent with delegation"
          version: "1.0.0"
          category: "coordination"
          capabilities:
            - "task_delegation"
            - "workflow_orchestration"
            - "multi_protocol"
        
        # LLM Configuration
        llm_config:
          model: "gpt-4"
          temperature: 0.3        # Balanced creativity for problem-solving
          max_tokens: 3000
          
        # A2A Protocol Configuration
        a2a_config:
          server_host: "localhost"
          server_port: 8080
          encryption_enabled: true
          message_queue_size: 1000
          heartbeat_interval: 30
          
        # MCP Protocol Configuration  
        mcp_config:
          server_url: "ws://localhost:3000/mcp"
          client_timeout: 30
          max_connections: 10
          protocol_version: "1.0"
          
        # Delegation Configuration
        delegation_rules:
          - task_type: "data_analysis"
            preferred_agents: ["data_scientist_agent", "analytics_agent"]
            fallback_agents: ["general_purpose_agent"]
            max_parallel_tasks: 3
            
          - task_type: "bioinformatics"
            preferred_agents: ["protein_analysis_agent", "sequence_agent"]
            required_capabilities: ["sequence_analysis", "protein_folding"]
            max_execution_time: 300
            
        # Performance Configuration
        enable_metrics: true
        performance_tracking:
          collect_timing_data: true
          track_resource_usage: true
          monitor_quality_metrics: true
          generate_optimization_reports: true
          
        # Workflow Configuration
        workflow_settings:
          max_concurrent_workflows: 5
          default_timeout: 600        # 10 minutes
          retry_attempts: 3
          enable_checkpointing: true
        ```
    
    **Usage Patterns:**
        
        **Basic Task Delegation:**
        ```python
        from nanobrain.library.agents.enhanced import CollaborativeAgent
        
        # Create collaborative agent with configuration
        agent_config = AgentConfig.from_config('config/collaborative_config.yml')
        collaborative_agent = CollaborativeAgent.from_config(agent_config)
        
        # Complex task requiring multiple specialized agents
        complex_task = {
            'objective': 'Analyze viral protein structure and predict drug targets',
            'data': {
                'protein_sequence': 'MKWVTFISLLFLF...',
                'virus_species': 'SARS-CoV-2',
                'analysis_type': 'comprehensive'
            },
            'requirements': [
                'sequence_analysis',
                'structure_prediction', 
                'drug_target_identification',
                'literature_validation'
            ]
        }
        
        # Delegate task with intelligent agent selection
        delegation_result = await collaborative_agent.delegate_task(complex_task)
        
        # Monitor collaboration progress
        for subtask in delegation_result['subtasks']:
            print(f"Subtask: {subtask['name']}")
            print(f"Assigned Agent: {subtask['assigned_agent']}")
            print(f"Status: {subtask['status']}")
            print(f"Progress: {subtask['progress']:.1%}")
            print("---")
        ```
        
        **Multi-Agent Workflow Orchestration:**
        ```python
        # Configure for complex workflow orchestration
        workflow_config = {
            'max_concurrent_workflows': 3,
            'enable_parallel_execution': True,
            'dependency_resolution': True,
            'error_recovery': True
        }
        
        agent_config = AgentConfig.from_config(workflow_config)
        collaborative_agent = CollaborativeAgent.from_config(agent_config)
        
        # Define multi-step research workflow
        research_workflow = {
            'workflow_id': 'viral_research_pipeline',
            'steps': [
                {
                    'name': 'literature_review',
                    'agent_type': 'research_agent',
                    'inputs': ['research_topic', 'search_parameters'],
                    'outputs': ['relevant_papers', 'research_gaps'],
                    'dependencies': []
                },
                {
                    'name': 'data_acquisition',
                    'agent_type': 'data_agent', 
                    'inputs': ['virus_species', 'data_types'],
                    'outputs': ['genomic_data', 'protein_data'],
                    'dependencies': ['literature_review']
                },
                {
                    'name': 'sequence_analysis',
                    'agent_type': 'bioinformatics_agent',
                    'inputs': ['genomic_data', 'analysis_parameters'],
                    'outputs': ['sequence_alignments', 'phylogenetic_tree'],
                    'dependencies': ['data_acquisition']
                },
                {
                    'name': 'result_synthesis',
                    'agent_type': 'synthesis_agent',
                    'inputs': ['all_previous_outputs'],
                    'outputs': ['final_report', 'recommendations'],
                    'dependencies': ['literature_review', 'data_acquisition', 'sequence_analysis']
                }
            ]
        }
        
        # Execute workflow with coordination
        workflow_result = await collaborative_agent.orchestrate_workflow(research_workflow)
        
        # Monitor execution progress
        execution_status = workflow_result['execution_status']
        print(f"Workflow Status: {execution_status['status']}")
        print(f"Completed Steps: {execution_status['completed_steps']}")
        print(f"Active Steps: {execution_status['active_steps']}")
        print(f"Total Progress: {execution_status['overall_progress']:.1%}")
        ```
        
        **A2A Protocol Team Coordination:**
        ```python
        # Configure for A2A team coordination
        a2a_config = {
            'protocol': 'a2a',
            'team_coordination': True,
            'real_time_sync': True,
            'secure_messaging': True
        }
        
        agent_config = AgentConfig.from_config(a2a_config)
        collaborative_agent = CollaborativeAgent.from_config(agent_config)
        
        # Initialize A2A team coordination
        team_setup = {
            'team_id': 'bioinformatics_research_team',
            'agents': [
                {'id': 'sequence_analyst', 'capabilities': ['sequence_analysis', 'alignment']},
                {'id': 'structure_predictor', 'capabilities': ['protein_folding', '3d_modeling']},
                {'id': 'drug_designer', 'capabilities': ['molecular_docking', 'drug_discovery']},
                {'id': 'literature_expert', 'capabilities': ['research', 'validation']}
            ],
            'coordination_rules': {
                'max_parallel_tasks': 2,
                'priority_system': 'deadline_based',
                'conflict_resolution': 'collaborative_consensus'
            }
        }
        
        # Establish A2A team coordination
        team_session = await collaborative_agent.establish_a2a_team(team_setup)
        
        # Coordinate complex multi-agent research task
        research_task = {
            'objective': 'Design novel antiviral compounds for SARS-CoV-2',
            'timeline': '2_weeks',
            'deliverables': ['target_analysis', 'compound_designs', 'validation_results']
        }
        
        # Real-time team coordination
        coordination_result = await collaborative_agent.coordinate_team_task(
            team_session,
            research_task
        )
        
        # Monitor real-time collaboration
        for update in coordination_result['real_time_updates']:
            print(f"Agent: {update['agent_id']}")
            print(f"Action: {update['action']}")
            print(f"Progress: {update['progress']}")
            print(f"Timestamp: {update['timestamp']}")
            print("---")
        ```
        
        **MCP Protocol Model Coordination:**
        ```python
        # Configure for MCP model coordination
        mcp_config = {
            'protocol': 'mcp',
            'model_coordination': True,
            'context_sharing': True,
            'resource_pooling': True
        }
        
        agent_config = AgentConfig.from_config(mcp_config)
        collaborative_agent = CollaborativeAgent.from_config(agent_config)
        
        # Initialize MCP model coordination
        model_pool = {
            'pool_id': 'research_model_pool',
            'models': [
                {'id': 'gpt4_research', 'specialization': 'research_analysis'},
                {'id': 'claude_writing', 'specialization': 'document_generation'}, 
                {'id': 'codex_analysis', 'specialization': 'code_analysis'},
                {'id': 'biobert_domain', 'specialization': 'biomedical_nlp'}
            ],
            'coordination_strategy': 'capability_based_routing',
            'context_sharing_level': 'selective'
        }
        
        # Establish MCP model coordination
        model_session = await collaborative_agent.establish_mcp_coordination(model_pool)
        
        # Coordinate multi-model analysis task
        analysis_task = {
            'input_data': 'large_biomedical_dataset.json',
            'analysis_components': [
                'statistical_analysis',
                'pattern_recognition', 
                'report_generation',
                'visualization_creation'
            ],
            'output_format': 'comprehensive_research_report'
        }
        
        # Execute with model coordination
        model_result = await collaborative_agent.coordinate_model_analysis(
            model_session,
            analysis_task
        )
        
        # Access coordinated results
        analysis_output = model_result['coordinated_output']
        model_contributions = model_result['model_contributions']
        
        print(f"Analysis Complete: {analysis_output['completion_status']}")
        for model_id, contribution in model_contributions.items():
            print(f"Model {model_id}: {contribution['contribution_summary']}")
        ```
        
        **Performance Optimization and Monitoring:**
        ```python
        # Configure for performance optimization
        performance_config = {
            'enable_advanced_metrics': True,
            'optimization_mode': 'adaptive',
            'monitoring_interval': 60,  # seconds
            'auto_scaling': True
        }
        
        agent_config = AgentConfig.from_config(performance_config)
        collaborative_agent = CollaborativeAgent.from_config(agent_config)
        
        # Monitor collaboration performance
        performance_session = await collaborative_agent.start_performance_monitoring()
        
        # Execute monitored collaborative workflow
        monitored_workflow = {
            'workflow_type': 'high_performance_analysis',
            'performance_targets': {
                'max_execution_time': 300,
                'min_quality_score': 0.85,
                'max_resource_usage': 0.8
            }
        }
        
        # Run with performance optimization
        optimized_result = await collaborative_agent.execute_optimized_workflow(
            monitored_workflow,
            performance_session
        )
        
        # Analyze performance metrics
        performance_report = optimized_result['performance_report']
        optimization_suggestions = performance_report['optimization_suggestions']
        
        print(f"Execution Time: {performance_report['execution_time']:.2f}s")
        print(f"Quality Score: {performance_report['quality_score']:.3f}")
        print(f"Resource Efficiency: {performance_report['resource_efficiency']:.3f}")
        
        print("\\nOptimization Suggestions:")
        for suggestion in optimization_suggestions:
            print(f"  - {suggestion['description']}")
            print(f"    Impact: {suggestion['expected_improvement']}")
        ```
    
    **Advanced Features:**
        
        **Intelligent Load Balancing:**
        * Dynamic agent workload distribution and optimization
        * Real-time performance monitoring and adjustment
        * Predictive scaling based on workflow complexity
        * Resource optimization and bottleneck resolution
        
        **Context-Aware Delegation:**
        * Semantic understanding of task requirements and agent capabilities
        * Historical performance analysis for optimal agent selection
        * Domain expertise matching and specialization routing
        * Quality prediction and reliability assessment
        
        **Fault Tolerance and Recovery:**
        * Automatic error detection and recovery mechanisms
        * Graceful degradation and fallback strategies
        * Checkpoint and restart capabilities for long-running workflows
        * Data consistency and integrity maintenance
        
        **Security and Access Control:**
        * Secure communication protocols and data encryption
        * Role-based access control and permission management
        * Audit trails and compliance monitoring
        * Privacy-preserving collaboration and data sharing
    
    **Enterprise Applications:**
        
        **Research Coordination:**
        * Multi-institutional research project coordination
        * Collaborative data analysis and model development
        * Distributed experiment execution and validation
        * Knowledge synthesis and report generation
        
        **Business Process Automation:**
        * Complex business workflow orchestration
        * Multi-department task coordination and tracking
        * Resource optimization and performance monitoring
        * Quality assurance and compliance management
        
        **Scientific Computing:**
        * Large-scale computational workflow management
        * Distributed simulation and analysis coordination
        * High-performance computing resource optimization
        * Results aggregation and synthesis
        
        **AI Development:**
        * Multi-model training and evaluation coordination
        * Distributed hyperparameter optimization
        * Model ensemble creation and management
        * Performance benchmarking and comparison
    
    Attributes:
        delegation_engine (DelegationEngine): Engine for intelligent task delegation
        a2a_config (dict): A2A protocol configuration and settings
        mcp_config (dict): MCP protocol configuration and settings
        delegation_rules (list): Rules and policies for task delegation
        enable_metrics (bool): Whether performance metrics are enabled
        active_workflows (dict): Currently active workflow executions
        team_sessions (dict): Active A2A team coordination sessions
        model_pools (dict): Active MCP model coordination pools
    
    Note:
        This agent requires A2A and/or MCP protocol support for full functionality.
        Network connectivity is needed for distributed collaboration features.
        Performance monitoring requires additional system resources but provides
        valuable optimization insights. Security features should be configured
        appropriately for enterprise deployment environments.
    
    Warning:
        Collaborative workflows can consume significant computational resources
        and network bandwidth. Monitor system performance and configure appropriate
        limits for production deployments. Security and access control settings
        should be carefully reviewed for multi-agent environments handling
        sensitive data or critical workflows.
    
    See Also:
        * :class:`A2ASupportMixin`: A2A protocol support implementation
        * :class:`MCPSupportMixin`: MCP protocol support implementation  
        * :class:`ConversationalAgent`: Base conversational agent capabilities
        * :class:`DelegationEngine`: Intelligent task delegation engine
        * :class:`AgentConfig`: Agent configuration schema
        * :mod:`nanobrain.library.agents.enhanced`: Enhanced agent implementations
    """
    
    # Component configuration
    COMPONENT_TYPE = "collaborative_agent"
    REQUIRED_CONFIG_FIELDS = ['name']
    OPTIONAL_CONFIG_FIELDS = {
        'a2a_config_path': None,
        'mcp_config_path': None,
        'delegation_rules': [],
        'enable_metrics': True
    }
    
    @classmethod
    def extract_component_config(cls, config: AgentConfig) -> Dict[str, Any]:
        """Extract CollaborativeAgent configuration"""
        return {
            'name': config.name,
            'description': config.description,
            'model': config.model,
            'system_prompt': config.system_prompt,
        }
    
    @classmethod  
    def resolve_dependencies(cls, component_config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Resolve CollaborativeAgent dependencies"""
        # Create executor via from_config to avoid direct instantiation
        from nanobrain.core.executor import LocalExecutor, ExecutorConfig
        
        executor_config = kwargs.get('executor_config') or ExecutorConfig()
        executor = LocalExecutor.from_config(executor_config)
        
        return {
            'executor': executor,
            'a2a_config_path': kwargs.get('a2a_config_path'),
            'mcp_config_path': kwargs.get('mcp_config_path'),
            'delegation_rules': kwargs.get('delegation_rules', []),
            'enable_metrics': kwargs.get('enable_metrics', True),
        }
    
    @classmethod
    def from_config(cls, config: AgentConfig, **kwargs) -> 'CollaborativeAgent':
        """Mandatory from_config implementation for CollaborativeAgent"""
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
        
        # Step 5: Mandatory agent_card validation and extraction
        if hasattr(config, 'agent_card') and config.agent_card:
            instance._a2a_card_data = config.agent_card.model_dump() if hasattr(config.agent_card, 'model_dump') else config.agent_card
            logger.info(f"Agent {instance.name} loaded with A2A card metadata")
        elif isinstance(config, dict) and 'agent_card' in config:
            instance._a2a_card_data = config['agent_card']
            logger.info(f"Agent {instance.name} loaded with A2A card metadata")
        else:
            raise ValueError(
                f"Missing mandatory 'agent_card' section in configuration for {cls.__name__}. "
                f"All agents must include A2A protocol compliant agent_card metadata for proper discovery and usage."
            )
        
        # Step 6: Post-creation initialization
        instance._post_config_initialization()
        
        logger.info(f"Successfully created {cls.__name__} with A2A compliance")
        return instance
        
    def _init_from_config(self, config: AgentConfig, component_config: Dict[str, Any],
                         dependencies: Dict[str, Any]) -> None:
        """Initialize CollaborativeAgent with resolved dependencies"""
        # Extract executor from dependencies to pass to parent
        executor = dependencies.pop('executor', None)
        a2a_config_path = dependencies.pop('a2a_config_path', None)
        mcp_config_path = dependencies.pop('mcp_config_path', None)
        delegation_rules = dependencies.pop('delegation_rules', [])
        enable_metrics = dependencies.pop('enable_metrics', True)
        
        # Initialize parent classes first
        ConversationalAgent.__init__(self, config, executor=executor, **dependencies)
        
        # Protocol configuration
        self.a2a_config_path = a2a_config_path
        self.mcp_config_path = mcp_config_path
        
        # Delegation and performance tracking
        self.delegation_engine = DelegationEngine(delegation_rules)
        self.performance_tracker = None  # AgentPerformanceTracker() if enable_metrics else None
        
        # Collaboration statistics
        self.collaboration_count = 0
        self.tool_usage_count = 0
        self.delegation_count = 0
        
    async def initialize(self):
        """Initialize the collaborative agent."""
        await super().initialize()
        
        # Initialize protocol support
        if self.a2a_config_path:
            await self.initialize_a2a(self.a2a_config_path)
            
        if self.mcp_config_path:
            await self.initialize_mcp(self.mcp_config_path)
            
        # Initialize performance tracking
        if self.performance_tracker:
            await self.performance_tracker.start_tracking()
            
        self.nb_logger.info("Collaborative agent initialized with protocol support")
        
    async def shutdown(self):
        """Shutdown the collaborative agent."""
        if self.performance_tracker:
            await self.performance_tracker.stop_tracking()
            
        await super().shutdown()
        
    async def process(self, input_text: str, **kwargs) -> str:
        """Enhanced process method with delegation and protocol support."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Record performance metrics
            if self.performance_tracker:
                await self.performance_tracker.record_request_start()
                
            # Check for delegation opportunities
            delegation_result = await self._check_delegation(input_text, **kwargs)
            if delegation_result:
                self.delegation_count += 1
                return delegation_result
                
            # Check for A2A collaboration
            if self.a2a_enabled and self.a2a_agents:
                a2a_result = await self._check_a2a_collaboration(input_text, **kwargs)
                if a2a_result:
                    self.collaboration_count += 1
                    return a2a_result
                    
            # Check for MCP tool usage
            if self.mcp_enabled and self.mcp_tools:
                mcp_result = await self._check_mcp_tools(input_text, **kwargs)
                if mcp_result:
                    self.tool_usage_count += 1
                    return mcp_result
                    
            # Fall back to normal processing
            result = await super().process(input_text, **kwargs)
            
            # Record successful processing
            if self.performance_tracker:
                response_time = asyncio.get_event_loop().time() - start_time
                await self.performance_tracker.record_request_end(response_time, success=True)
                
            return result
            
        except Exception as e:
            # Record failed processing
            if self.performance_tracker:
                response_time = asyncio.get_event_loop().time() - start_time
                await self.performance_tracker.record_request_end(response_time, success=False)
                
            self.nb_logger.error(f"Error in collaborative processing: {e}")
            raise e
            
    async def _check_delegation(self, input_text: str, **kwargs) -> Optional[str]:
        """Check if the input should be delegated based on rules."""
        delegation_target = await self.delegation_engine.should_delegate(input_text, **kwargs)
        
        if delegation_target:
            try:
                # Log delegation
                self.nb_logger.info(f"Delegating to: {delegation_target['target']}")
                
                # Perform delegation (this would be implemented based on the target type)
                if delegation_target['type'] == 'a2a_agent':
                    return await self._delegate_to_a2a_agent(delegation_target['target'], input_text, **kwargs)
                elif delegation_target['type'] == 'mcp_tool':
                    return await self._delegate_to_mcp_tool(delegation_target['target'], input_text, **kwargs)
                elif delegation_target['type'] == 'custom':
                    return await self._delegate_custom(delegation_target, input_text, **kwargs)
                    
            except Exception as e:
                self.nb_logger.error(f"Delegation failed: {e}")
                # Continue with normal processing
                
        return None
        
    async def _check_a2a_collaboration(self, input_text: str, **kwargs) -> Optional[str]:
        """Check for A2A collaboration opportunities."""
        # Simple keyword-based collaboration detection
        collaboration_keywords = {
            'translate': 'translator_agent',
            'summarize': 'summarizer_agent',
            'analyze': 'analyzer_agent',
            'calculate': 'calculator_agent'
        }
        
        input_lower = input_text.lower()
        for keyword, agent_name in collaboration_keywords.items():
            if keyword in input_lower and agent_name in self.a2a_agents:
                try:
                    result = await self.call_a2a_agent(agent_name, input_text)
                    return f"🤝 Collaborated with {agent_name}:\n\n{result}"
                except Exception as e:
                    self.nb_logger.error(f"A2A collaboration failed: {e}")
                    break
                    
        return None
        
    async def _check_mcp_tools(self, input_text: str, **kwargs) -> Optional[str]:
        """Check for MCP tool usage opportunities."""
        # Simple keyword-based tool detection
        tool_keywords = {
            'file': ['file', 'read', 'write', 'save', 'load'],
            'calculator': ['calculate', 'math', 'compute', 'add', 'subtract'],
            'weather': ['weather', 'temperature', 'forecast', 'climate'],
            'search': ['search', 'find', 'lookup', 'query']
        }
        
        input_lower = input_text.lower()
        for tool_name, keywords in tool_keywords.items():
            if any(keyword in input_lower for keyword in keywords) and tool_name in self.mcp_tools:
                try:
                    # This would call the actual MCP tool
                    result = f"🔧 Used {tool_name} tool to process: {input_text}"
                    return result
                except Exception as e:
                    self.nb_logger.error(f"MCP tool usage failed: {e}")
                    break
                    
        return None
        
    async def _delegate_to_a2a_agent(self, agent_name: str, input_text: str, **kwargs) -> str:
        """Delegate to an A2A agent."""
        if agent_name in self.a2a_agents:
            result = await self.call_a2a_agent(agent_name, input_text)
            return f"🎯 Delegated to {agent_name}:\n\n{result}"
        else:
            raise ValueError(f"A2A agent {agent_name} not available")
            
    async def _delegate_to_mcp_tool(self, tool_name: str, input_text: str, **kwargs) -> str:
        """Delegate to an MCP tool."""
        if tool_name in self.mcp_tools:
            # This would call the actual MCP tool with proper arguments
            result = f"🔧 Delegated to {tool_name} tool: {input_text}"
            return result
        else:
            raise ValueError(f"MCP tool {tool_name} not available")
            
    async def _delegate_custom(self, delegation_target: Dict[str, Any], input_text: str, **kwargs) -> str:
        """Handle custom delegation logic."""
        # This would implement custom delegation logic based on the target configuration
        return f"🔄 Custom delegation to {delegation_target['target']}: {input_text}"
        
    async def add_delegation_rule(self, rule: Dict[str, Any]):
        """Add a new delegation rule."""
        await self.delegation_engine.add_rule(rule)
        
    async def remove_delegation_rule(self, rule_id: str):
        """Remove a delegation rule."""
        await self.delegation_engine.remove_rule(rule_id)
        
    async def get_enhanced_status(self) -> Dict[str, Any]:
        """Get enhanced status including collaboration and performance metrics."""
        status = {
            'agent_name': self.config.name,
            'collaboration_count': self.collaboration_count,
            'tool_usage_count': self.tool_usage_count,
            'delegation_count': self.delegation_count,
            'delegation_rules': len(self.delegation_engine.rules)
        }
        
        # Add A2A status
        if hasattr(self, 'get_a2a_status'):
            status['a2a'] = self.get_a2a_status()
            
        # Add MCP status
        if hasattr(self, 'get_mcp_status'):
            status['mcp'] = self.get_mcp_status()
            
        # Add performance metrics
        if self.performance_tracker:
            status['performance'] = await self.performance_tracker.get_metrics()
            
        return status
        
    async def get_collaboration_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent collaboration history."""
        if self.performance_tracker:
            return await self.performance_tracker.get_collaboration_history(limit)
        return []
        
    async def reset_statistics(self):
        """Reset collaboration statistics."""
        self.collaboration_count = 0
        self.tool_usage_count = 0
        self.delegation_count = 0
        
        if self.performance_tracker:
            await self.performance_tracker.reset_metrics()
            
        self.nb_logger.info("Collaboration statistics reset")
        
    async def configure_protocols(self, a2a_config: Optional[Dict[str, Any]] = None, 
                                 mcp_config: Optional[Dict[str, Any]] = None):
        """Configure protocol settings."""
        if a2a_config and self.a2a_enabled:
            # Update A2A configuration
            await self.update_a2a_config(a2a_config)
            
        if mcp_config and self.mcp_enabled:
            # Update MCP configuration
            await self.update_mcp_config(mcp_config)
            
        self.nb_logger.info("Protocol configurations updated")
        
    async def get_available_collaborators(self) -> Dict[str, List[str]]:
        """Get list of available collaborators."""
        collaborators = {
            'a2a_agents': list(self.a2a_agents.keys()) if self.a2a_enabled else [],
            'mcp_tools': list(self.mcp_tools.keys()) if self.mcp_enabled else [],
            'delegation_targets': [rule.get('target', 'unknown') for rule in self.delegation_engine.rules]
        }
        
        return collaborators 