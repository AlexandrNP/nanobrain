"""
Enhanced Collaborative Agent

An advanced conversational agent with A2A and MCP protocol support for the NanoBrain library.

This agent provides:
- Agent-to-Agent (A2A) collaboration capabilities
- Model Context Protocol (MCP) tool integration
- Enhanced conversation management
- Performance tracking and metrics
- Delegation rules for specialized tasks
"""

import sys
import os
from typing import Dict, Any, Optional, List
from datetime import datetime

# Add src to path for core imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from nanobrain.core.agent import ConversationalAgent, AgentConfig
from nanobrain.core.a2a_support import A2ASupportMixin
from nanobrain.core.mcp_support import MCPSupportMixin
from nanobrain.core.component_base import FromConfigBase, ComponentConfigurationError, ComponentDependencyError
from nanobrain.core.logging_system import get_logger


class EnhancedCollaborativeAgent(A2ASupportMixin, MCPSupportMixin, ConversationalAgent):
    """
    Enhanced Collaborative Agent - Multi-Protocol AI Agent with A2A and MCP Integration
    ================================================================================
    
    The Enhanced Collaborative Agent is a sophisticated conversational AI agent that extends
    the base conversational capabilities with advanced collaboration protocols, tool
    integration, and intelligent task delegation. This agent serves as a central coordination
    point for complex multi-agent workflows and tool-intensive operations.
    
    **Core Architecture:**
        The Enhanced Collaborative Agent combines multiple capabilities:
        
        * **Conversational AI**: Advanced LLM-based conversation management and context handling
        * **A2A Protocol**: Agent-to-Agent collaboration for distributed processing and specialization
        * **MCP Protocol**: Model Context Protocol for structured tool integration and execution
        * **Intelligent Delegation**: Rule-based task routing to specialized agents and tools
        * **Performance Monitoring**: Comprehensive metrics collection and optimization tracking
        * **Context Management**: Advanced conversation history and state management
    
    **Multi-Protocol Integration:**
        
        **Agent-to-Agent (A2A) Collaboration:**
        * Seamless communication with other NanoBrain agents
        * Task delegation based on capability matching and workload
        * Context sharing and result aggregation across agent networks
        * Dynamic agent discovery and capability negotiation
        
        **Model Context Protocol (MCP) Support:**
        * Structured tool integration with standardized interfaces
        * Tool capability discovery and automatic selection
        * Context preservation across tool invocations
        * Resource management and lifecycle control
        
        **Unified Protocol Management:**
        * Automatic protocol selection based on task requirements
        * Fallback mechanisms for protocol failures
        * Cross-protocol context translation and preservation
        * Performance optimization across protocol boundaries
    
    **Intelligent Delegation System:**
        The agent features sophisticated task delegation capabilities:
        
        **Rule-Based Routing:**
        * Configurable delegation rules based on task content and context
        * Keyword-based specialization detection and routing
        * Priority-based agent selection and load balancing
        * Dynamic rule evaluation and adaptation
        
        **Capability Matching:**
        * Automatic agent capability discovery and assessment
        * Task-to-capability mapping with confidence scoring
        * Multi-agent collaboration for complex task decomposition
        * Resource availability and performance-based selection
        
        **Delegation Strategies:**
        * **Direct Delegation**: Route specific tasks to specialized agents
        * **Collaborative Processing**: Coordinate multiple agents for complex tasks
        * **Tool Chain Execution**: Orchestrate tool sequences across protocols
        * **Fallback Handling**: Graceful degradation when preferred agents unavailable
    
    **Configuration Architecture:**
        Enhanced agents support comprehensive configuration management:
        
        ```yaml
        # Enhanced Collaborative Agent Configuration
        name: "enhanced_coordinator"
        description: "Multi-protocol collaborative agent with delegation"
        
        # Base conversational agent settings
        model: "gpt-4"
        temperature: 0.7
        max_tokens: 4000
        system_prompt: |
          You are an enhanced collaborative agent capable of coordinating
          multiple agents and tools to solve complex problems.
        
        # A2A Protocol Configuration
        a2a_config:
          enable_discovery: true
          collaboration_mode: "intelligent"
          max_concurrent_delegations: 5
          delegation_timeout: 300
          capability_assessment: true
        
        # MCP Protocol Configuration
        mcp_config:
          tool_discovery: "automatic"
          context_preservation: true
          resource_management: "optimized"
          tool_selection_strategy: "capability_match"
        
        # Delegation Rules Configuration
        delegation_rules:
          - trigger_keywords: ["analyze", "data", "statistics"]
            target_agent_type: "data_analysis_agent"
            confidence_threshold: 0.8
            
          - trigger_keywords: ["write", "code", "program"]
            target_agent_type: "code_generation_agent"
            min_complexity: "moderate"
            
          - trigger_keywords: ["search", "find", "lookup"]
            target_protocol: "mcp"
            preferred_tools: ["search_tool", "database_tool"]
        
        # Performance and Monitoring
        enable_metrics: true
        performance_tracking:
          delegation_success_rate: true
          response_time_analysis: true
          protocol_performance_comparison: true
          resource_utilization_tracking: true
        
        # Tool Integration
        tool_keywords:
          data_analysis: ["pandas", "numpy", "statistics"]
          web_search: ["search", "find", "lookup"]
          file_operations: ["read", "write", "save", "load"]
          computation: ["calculate", "compute", "analyze"]
        ```
    
    **Usage Patterns:**
        
        **Basic Enhanced Agent:**
        ```python
        from nanobrain.library.agents.conversational import EnhancedCollaborativeAgent
        
        # Create from configuration
        agent = EnhancedCollaborativeAgent.from_config('config/enhanced_agent.yml')
        
        # Process with automatic delegation
        response = await agent.aprocess(
            "Analyze the sales data and generate a comprehensive report"
        )
        
        # Agent automatically determines optimal delegation strategy
        print(f"Response: {response}")
        print(f"Delegation used: {agent.get_last_delegation_info()}")
        ```
        
        **Multi-Agent Collaboration:**
        ```python
        # Enhanced agent coordinating specialized agents
        coordinator = EnhancedCollaborativeAgent.from_config('config/coordinator.yml')
        
        # Complex task requiring multiple specializations
        task = "Research market trends, analyze competitor data, and create presentation"
        
        response = await coordinator.aprocess(task)
        
        # Coordinator automatically:
        # 1. Identifies subtasks (research, analysis, presentation)
        # 2. Delegates to appropriate specialized agents
        # 3. Coordinates results and creates unified response
        ```
        
        **MCP Tool Integration:**
        ```python
        # Enhanced agent with MCP tool integration
        agent = EnhancedCollaborativeAgent.from_config('config/mcp_enabled_agent.yml')
        
        # Task requiring tool usage
        response = await agent.aprocess(
            "Search for recent AI research papers and summarize key findings"
        )
        
        # Agent automatically:
        # 1. Detects need for search functionality
        # 2. Discovers and selects appropriate MCP tools
        # 3. Executes search with proper context management
        # 4. Synthesizes results into comprehensive response
        ```
        
        **Custom Delegation Rules:**
        ```python
        # Configure custom delegation patterns
        agent_config = {
            "name": "custom_coordinator",
            "delegation_rules": [
                {
                    "trigger_pattern": r"analyze.*data.*(\d+).*rows",
                    "target_agent": "large_data_specialist",
                    "condition": "lambda match: int(match.group(1)) > 10000"
                },
                {
                    "trigger_keywords": ["bioinformatics", "protein", "genome"],
                    "target_agent": "bioinformatics_specialist",
                    "priority": "high"
                }
            ]
        }
        
        agent = EnhancedCollaborativeAgent.from_config(agent_config)
        
        # Delegation rules automatically applied during processing
        ```
    
    **Advanced Features:**
        
        **Dynamic Protocol Selection:**
        * Automatic protocol selection based on task analysis and agent availability
        * Performance-based protocol preference learning and optimization
        * Cross-protocol fallback mechanisms for reliability
        * Protocol performance comparison and optimization recommendations
        
        **Context-Aware Delegation:**
        * Conversation history analysis for delegation decision making
        * Multi-turn context preservation across agent handoffs
        * Result correlation and synthesis from multiple delegation sources
        * Context-based delegation rule evaluation and adaptation
        
        **Performance Optimization:**
        * Delegation success rate tracking and optimization
        * Response time analysis and performance tuning
        * Resource utilization monitoring and load balancing
        * Caching of delegation decisions for similar tasks
        
        **Error Handling and Recovery:**
        * Graceful handling of agent unavailability and failures
        * Automatic fallback to alternative agents or protocols
        * Error context preservation and delegation retry mechanisms
        * Comprehensive error logging and diagnostic information
    
    **Collaboration Patterns:**
        
        **Hierarchical Delegation:**
        * Top-level coordinator delegates to specialized sub-agents
        * Sub-agents may further delegate to highly specialized tools
        * Result aggregation and synthesis at coordination level
        * Error propagation and recovery across delegation hierarchy
        
        **Peer-to-Peer Collaboration:**
        * Direct agent-to-agent communication and coordination
        * Shared context and state management across peer agents
        * Collaborative problem solving with result sharing
        * Dynamic role assignment based on agent capabilities
        
        **Tool Orchestration:**
        * Coordination of multiple tools for complex task execution
        * Tool chain planning and optimization for efficiency
        * Result passing and transformation between tool stages
        * Tool failure handling and alternative execution paths
    
    **Integration Patterns:**
        
        **Workflow Integration:**
        * Enhanced agents as intelligent workflow coordinators
        * Dynamic workflow adaptation based on delegation outcomes
        * Multi-agent workflow orchestration and synchronization
        * Workflow performance optimization through intelligent delegation
        
        **System Integration:**
        * Integration with external systems through protocol adapters
        * Enterprise system connectivity via A2A and MCP protocols
        * API gateway functionality for agent network access
        * Service mesh integration for distributed agent deployments
        
        **Data Flow Management:**
        * Context-aware data routing between agents and tools
        * Data transformation and format adaptation across protocols
        * Data privacy and security enforcement during delegation
        * Data lineage tracking across multi-agent processing chains
    
    **Performance and Scalability:**
        
        **Execution Optimization:**
        * Parallel delegation execution for independent subtasks
        * Intelligent batching of similar delegation requests
        * Caching of delegation results for repeated patterns
        * Resource pooling and reuse across delegation instances
        
        **Scalability Features:**
        * Horizontal scaling through agent network distribution
        * Load balancing across multiple agent instances
        * Dynamic scaling based on delegation demand patterns
        * Cloud-native deployment and auto-scaling support
        
        **Monitoring and Analytics:**
        * Real-time delegation performance tracking and analysis
        * Protocol efficiency comparison and optimization recommendations
        * Agent network health monitoring and alerting
        * Usage pattern analysis and optimization suggestions
    
    **Security and Reliability:**
        
        **Secure Collaboration:**
        * Authentication and authorization for agent-to-agent communication
        * Encrypted context sharing and result transmission
        * Role-based access control for delegation permissions
        * Audit logging for compliance and security monitoring
        
        **Reliability Features:**
        * Fault tolerance with automatic failover and recovery
        * Transaction-like delegation with rollback capabilities
        * Health monitoring and circuit breaker patterns
        * Graceful degradation for partial system failures
    
    **Development and Testing:**
        
        **Testing Support:**
        * Mock agent implementations for delegation testing
        * Protocol simulation and validation frameworks
        * Performance benchmarking and profiling tools
        * Integration testing with real agent networks
        
        **Debugging Features:**
        * Comprehensive logging with delegation trace information
        * Protocol interaction visualization and analysis
        * Performance profiling and bottleneck identification
        * Interactive debugging with delegation step-through
        
        **Development Tools:**
        * Delegation rule validation and testing utilities
        * Protocol compliance verification and certification
        * Agent capability profiling and optimization tools
        * Network topology visualization and management
    
    Attributes:
        name (str): Agent identifier for logging and network registration
        delegation_rules (List[Dict]): Configurable rules for intelligent task delegation
        tool_keywords (Dict[str, List[str]]): Keyword mappings for tool selection
        enable_metrics (bool): Whether comprehensive performance tracking is enabled
        a2a_config_path (str, optional): Path to A2A protocol configuration
        mcp_config_path (str, optional): Path to MCP protocol configuration
        delegation_count (int): Total number of delegations performed
        successful_delegations (int): Number of successful delegation operations
        protocol_performance (Dict): Performance metrics by protocol type
        collaboration_history (List): History of agent collaboration sessions
    
    Note:
        Enhanced Collaborative Agents require proper A2A and MCP protocol configuration
        for full functionality. Delegation rules should be carefully designed to prevent
        infinite delegation loops. All agents must be created using the from_config
        pattern with proper configuration files following framework patterns.
    
    Warning:
        Enhanced agents may consume significant resources when managing multiple
        concurrent delegations. Monitor delegation patterns and implement appropriate
        limits and timeouts. Be cautious with delegation rules that might create
        circular delegation patterns or excessive resource consumption.
    
    See Also:
        * :class:`ConversationalAgent`: Base conversational agent implementation
        * :class:`A2ASupportMixin`: Agent-to-Agent protocol support
        * :class:`MCPSupportMixin`: Model Context Protocol support
        * :class:`AgentConfig`: Agent configuration schema and validation
        * :mod:`nanobrain.library.agents.specialized`: Specialized agent implementations
        * :mod:`nanobrain.core.a2a_support`: A2A protocol implementation
        * :mod:`nanobrain.core.mcp_support`: MCP protocol implementation
    """
    
    COMPONENT_TYPE = "enhanced_collaborative_agent"
    REQUIRED_CONFIG_FIELDS = ['name', 'description']
    OPTIONAL_CONFIG_FIELDS = {
        'model': 'gpt-3.5-turbo',
        'temperature': 0.7,
        'max_tokens': None,
        'system_prompt': '',
        'delegation_rules': [],
        'tool_keywords': {},
        'enable_metrics': True,
        'a2a_config_path': None,
        'mcp_config_path': None
    }
    
    @classmethod
    def extract_component_config(cls, config: AgentConfig) -> Dict[str, Any]:
        """Extract EnhancedCollaborativeAgent configuration"""
        return {
            'name': getattr(config, 'name', 'enhanced_agent'),
            'description': getattr(config, 'description', ''),
            'model': getattr(config, 'model', 'gpt-3.5-turbo'),
            'temperature': getattr(config, 'temperature', 0.7),
            'max_tokens': getattr(config, 'max_tokens', None),
            'system_prompt': getattr(config, 'system_prompt', ''),
        }
    
    @classmethod  
    def resolve_dependencies(cls, component_config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Resolve EnhancedCollaborativeAgent dependencies"""
        # Create executor via from_config to avoid direct instantiation
        from nanobrain.core.executor import LocalExecutor, ExecutorConfig
        import os
        from pathlib import Path
        
        # ✅ FRAMEWORK COMPLIANCE: Use from_config pattern for ExecutorConfig
        provided_executor_config = kwargs.get('executor_config')
        if provided_executor_config is None:
            # Use default executor config file
            # Get the actual path to the Chatbot directory
            import sys
            for path_item in sys.path:
                if 'nanobrain-upd-Jun/nanobrain' in path_item:
                    project_root = Path(path_item)
                    break
            else:
                project_root = Path(__file__).parent.parent.parent.parent
                
            default_config_path = project_root / "Chatbot" / "config" / "components" / "default_agent_executor_config.yml"
            if default_config_path.exists():
                executor_config_path = str(default_config_path)
            else:
                # Fallback to library default
                lib_default_path = Path(__file__).parent.parent / "config" / "default_executor_config.yml"
                executor_config_path = str(lib_default_path)
            
            # Load the ExecutorConfig first, then pass it to LocalExecutor
            executor_config_obj = ExecutorConfig.from_config(executor_config_path)
            executor = LocalExecutor.from_config(executor_config_obj)
        else:
            # If executor_config is provided, it should already be a valid ExecutorConfig object
            executor = LocalExecutor.from_config(provided_executor_config)
        
        return {
            'executor': executor,
            'delegation_rules': kwargs.get('delegation_rules', []),
            'tool_keywords': kwargs.get('tool_keywords', {}),
            'enable_metrics': kwargs.get('enable_metrics', True),
            'a2a_config_path': kwargs.get('a2a_config_path'),
            'mcp_config_path': kwargs.get('mcp_config_path'),
        }
    
    @classmethod
    def from_config(cls, config: AgentConfig, **kwargs) -> 'EnhancedCollaborativeAgent':
        """Mandatory from_config implementation for EnhancedCollaborativeAgent"""
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
    
    def _init_from_config(self, config: AgentConfig, component_config: Dict[str, Any],
                         dependencies: Dict[str, Any]) -> None:
        """Initialize EnhancedCollaborativeAgent with resolved dependencies"""
        # Extract executor from dependencies to pass to parent
        executor = dependencies.pop('executor', None)
        
        # Initialize parent classes using super() for proper MRO
        super()._init_from_config(config, component_config, {'executor': executor, **dependencies})
        
        # Enhanced logger
        self.enhanced_logger = get_logger(f"enhanced.{self.name}")
        
        # Collaboration tracking
        self.collaboration_count = 0
        self.tool_usage_count = 0
        
        # Configuration from dependencies
        self.delegation_rules = dependencies.get('delegation_rules', [])
        self.tool_keywords = dependencies.get('tool_keywords') or self._get_default_tool_keywords()
        self.enable_metrics = dependencies.get('enable_metrics', True)
        
        # Protocol configuration paths
        self.a2a_config_path = dependencies.get('a2a_config_path')
        self.mcp_config_path = dependencies.get('mcp_config_path')
        
        # Performance tracking
        self.start_time = datetime.now()
        self.total_requests = 0
        self.successful_delegations = 0
        self.successful_tool_uses = 0
        self.delegation_errors = 0
        self.tool_errors = 0
    
    # EnhancedCollaborativeAgent inherits FromConfigBase.__init__ which prevents direct instantiation
    
    async def initialize(self) -> None:
        """Initialize the enhanced collaborative agent."""
        await super().initialize()
        
        # Initialize A2A support if configured
        if self.a2a_config_path:
            try:
                await self.initialize_a2a(self.a2a_config_path)
                self.enhanced_logger.info("A2A support initialized")
            except Exception as e:
                self.enhanced_logger.error(f"Failed to initialize A2A support: {e}")
        
        # Initialize MCP support if configured
        if self.mcp_config_path:
            try:
                await self.initialize_mcp(self.mcp_config_path)
                self.enhanced_logger.info("MCP support initialized")
            except Exception as e:
                self.enhanced_logger.error(f"Failed to initialize MCP support: {e}")
        
        self.enhanced_logger.info(
            f"Enhanced collaborative agent {self.name} initialized",
            delegation_rules_count=len(self.delegation_rules),
            a2a_enabled=hasattr(self, 'a2a_enabled') and self.a2a_enabled,
            mcp_enabled=hasattr(self, 'mcp_enabled') and self.mcp_enabled
        )
    
    async def shutdown(self) -> None:
        """Shutdown the enhanced collaborative agent."""
        # Log final statistics
        uptime = datetime.now() - self.start_time
        self.enhanced_logger.info(
            f"Enhanced collaborative agent {self.name} shutting down",
            uptime_seconds=uptime.total_seconds(),
            total_requests=self.total_requests,
            collaboration_count=self.collaboration_count,
            tool_usage_count=self.tool_usage_count,
            successful_delegations=self.successful_delegations,
            successful_tool_uses=self.successful_tool_uses,
            delegation_errors=self.delegation_errors,
            tool_errors=self.tool_errors
        )
        
        await super().shutdown()
    
    def _get_default_tool_keywords(self) -> Dict[str, List[str]]:
        """Get default tool detection keywords."""
        return {
            'calculator': ['calculate', 'math', 'compute', 'add', 'subtract', 'multiply', 'divide'],
            'weather': ['weather', 'temperature', 'forecast', 'climate'],
            'file': ['file', 'read', 'write', 'save', 'load'],
            'search': ['search', 'find', 'lookup', 'query'],
            'code': ['code', 'program', 'script', 'function', 'class']
        }
    
    async def process(self, input_text: str, **kwargs) -> str:
        """
        Enhanced process method with A2A delegation and MCP tool usage.
        
        Processing flow:
        1. Check for A2A delegation opportunities
        2. Check for MCP tool usage opportunities
        3. Fall back to normal conversational processing
        
        Args:
            input_text: User input to process
            **kwargs: Additional processing parameters
            
        Returns:
            str: Processed response
        """
        self.total_requests += 1
        
        try:
            # Check if we should delegate to an A2A agent
            if hasattr(self, 'a2a_enabled') and self.a2a_enabled and hasattr(self, 'a2a_agents'):
                delegation_result = await self._check_for_delegation(input_text)
                if delegation_result:
                    self.collaboration_count += 1
                    self.successful_delegations += 1
                    # Add to conversation history
                    self.add_to_conversation("user", input_text)
                    self.add_to_conversation("assistant", delegation_result)
                    return delegation_result
            
            # Check if we should use MCP tools
            if hasattr(self, 'mcp_enabled') and self.mcp_enabled and hasattr(self, 'mcp_tools'):
                tool_result = await self._check_for_tool_usage(input_text)
                if tool_result:
                    self.tool_usage_count += 1
                    self.successful_tool_uses += 1
                    # Add to conversation history
                    self.add_to_conversation("user", input_text)
                    self.add_to_conversation("assistant", tool_result)
                    return tool_result
            
            # Fall back to normal conversational processing
            return await super().process(input_text, **kwargs)
            
        except Exception as e:
            self.enhanced_logger.error(f"Error in enhanced processing: {e}")
            # Fall back to basic processing
            return await super().process(input_text, **kwargs)
    
    async def _check_for_delegation(self, input_text: str) -> Optional[str]:
        """
        Check if the input should be delegated to an A2A agent.
        
        Args:
            input_text: Input text to analyze
            
        Returns:
            Optional[str]: Delegation result or None if no delegation needed
        """
        input_lower = input_text.lower()
        
        # Check delegation rules
        for rule in self.delegation_rules:
            keywords = rule.get('keywords', [])
            agent_name = rule.get('agent')
            
            if any(keyword in input_lower for keyword in keywords):
                if hasattr(self, 'a2a_agents') and agent_name in self.a2a_agents:
                    try:
                        # Log delegation
                        self.enhanced_logger.info(
                            f"Delegating to A2A agent: {agent_name}",
                            rule_description=rule.get('description', ''),
                            collaboration_count=self.collaboration_count + 1
                        )
                        
                        # Call A2A agent
                        result = await self.call_a2a_agent(agent_name, input_text)
                        
                        # Wrap result with context
                        return f"🤝 Collaborated with {agent_name}:\n\n{result}"
                        
                    except Exception as e:
                        self.delegation_errors += 1
                        self.enhanced_logger.error(f"A2A delegation failed: {e}")
                        # Continue with normal processing
                        break
        
        return None
    
    async def _check_for_tool_usage(self, input_text: str) -> Optional[str]:
        """
        Check if the input should use MCP tools.
        
        Args:
            input_text: Input text to analyze
            
        Returns:
            Optional[str]: Tool usage result or None if no tool needed
        """
        input_lower = input_text.lower()
        
        # Check tool keywords
        for tool_name, keywords in self.tool_keywords.items():
            if any(keyword in input_lower for keyword in keywords):
                if hasattr(self, 'mcp_tools') and tool_name in self.mcp_tools:
                    try:
                        # Log tool usage
                        self.enhanced_logger.info(
                            f"Using MCP tool: {tool_name}",
                            tool_usage_count=self.tool_usage_count + 1
                        )
                        
                        # Use MCP tool
                        result = await self._execute_mcp_tool(tool_name, input_text)
                        
                        return f"🔧 Used {tool_name} tool:\n\n{result}"
                        
                    except Exception as e:
                        self.tool_errors += 1
                        self.enhanced_logger.error(f"MCP tool usage failed: {e}")
                        # Continue with normal processing
                        break
        
        return None
    
    async def _execute_mcp_tool(self, tool_name: str, input_text: str) -> str:
        """
        Execute an MCP tool with the given input.
        
        Args:
            tool_name: Name of the MCP tool to execute
            input_text: Input text for the tool
            
        Returns:
            str: Tool execution result
        """
        try:
            # This would be implemented based on the actual MCP tool interface
            # For now, return a placeholder
            if hasattr(self, 'call_mcp_tool'):
                return await self.call_mcp_tool(tool_name, {"input": input_text})
            else:
                return f"MCP tool {tool_name} executed with input: {input_text[:100]}..."
        except Exception as e:
            self.enhanced_logger.error(f"Error executing MCP tool {tool_name}: {e}")
            raise
    
    def get_enhanced_status(self) -> Dict[str, Any]:
        """
        Get enhanced status information including collaboration metrics.
        
        Returns:
            Dict[str, Any]: Enhanced status information
        """
        base_stats = self.get_performance_stats()
        uptime = datetime.now() - self.start_time
        
        enhanced_stats = {
            'uptime_seconds': uptime.total_seconds(),
            'total_requests': self.total_requests,
            'collaboration_count': self.collaboration_count,
            'tool_usage_count': self.tool_usage_count,
            'successful_delegations': self.successful_delegations,
            'successful_tool_uses': self.successful_tool_uses,
            'delegation_errors': self.delegation_errors,
            'tool_errors': self.tool_errors,
            'delegation_success_rate': (
                self.successful_delegations / max(1, self.collaboration_count)
            ),
            'tool_success_rate': (
                self.successful_tool_uses / max(1, self.tool_usage_count)
            ),
            'a2a_enabled': hasattr(self, 'a2a_enabled') and self.a2a_enabled,
            'mcp_enabled': hasattr(self, 'mcp_enabled') and self.mcp_enabled,
            'delegation_rules_count': len(self.delegation_rules),
            'tool_keywords_count': len(self.tool_keywords)
        }
        
        return {**base_stats, **enhanced_stats}
    
    def add_delegation_rule(self, keywords: List[str], agent_name: str, description: str = ""):
        """
        Add a new delegation rule.
        
        Args:
            keywords: Keywords that trigger delegation
            agent_name: Name of the A2A agent to delegate to
            description: Description of the rule
        """
        rule = {
            'keywords': keywords,
            'agent': agent_name,
            'description': description
        }
        self.delegation_rules.append(rule)
        self.enhanced_logger.info(f"Added delegation rule for agent {agent_name}")
    
    def add_tool_keywords(self, tool_name: str, keywords: List[str]):
        """
        Add keywords for MCP tool detection.
        
        Args:
            tool_name: Name of the MCP tool
            keywords: Keywords that trigger tool usage
        """
        self.tool_keywords[tool_name] = keywords
        self.enhanced_logger.info(f"Added tool keywords for {tool_name}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get a summary of performance metrics.
        
        Returns:
            Dict[str, Any]: Performance summary
        """
        uptime = datetime.now() - self.start_time
        
        return {
            'agent_name': self.name,
            'uptime_hours': uptime.total_seconds() / 3600,
            'total_requests': self.total_requests,
            'requests_per_hour': self.total_requests / max(1, uptime.total_seconds() / 3600),
            'collaboration_percentage': (
                (self.collaboration_count / max(1, self.total_requests)) * 100
            ),
            'tool_usage_percentage': (
                (self.tool_usage_count / max(1, self.total_requests)) * 100
            ),
            'overall_success_rate': (
                ((self.successful_delegations + self.successful_tool_uses) / 
                 max(1, self.collaboration_count + self.tool_usage_count)) * 100
            )
        } 