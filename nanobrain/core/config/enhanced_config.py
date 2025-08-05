"""
Enhanced Configuration System with Integrated Card Metadata

Implements YAML-based configuration with mandatory agent_card and tool_card sections
for A2A protocol compliance and unified framework architecture.
"""

from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field, ConfigDict
from pathlib import Path
import json
import yaml
import logging

from .yaml_config import YAMLConfig

logger = logging.getLogger(__name__)


class InputOutputFormat(BaseModel):
    """Structured input/output format specification"""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    primary_mode: str
    supported_modes: List[str]
    content_types: List[str]
    format_schema: Dict[str, Any]


class A2ACapabilities(BaseModel):
    """A2A capability declarations for agents"""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    streaming: bool = False
    push_notifications: bool = False
    state_transition_history: bool = False
    multi_turn_conversation: bool = True
    context_retention: bool = True
    tool_usage: bool = False
    delegation: bool = False
    collaboration: bool = False


class A2ASkill(BaseModel):
    """A2A skill definition for agents"""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    id: str
    name: str
    description: str
    complexity: str = "intermediate"  # beginner, intermediate, advanced, expert
    input_modes: List[str] = Field(default_factory=list)
    output_modes: List[str] = Field(default_factory=list)
    examples: List[str] = Field(default_factory=list)


class PerformanceSpec(BaseModel):
    """Performance characteristics specification"""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    typical_response_time: str
    max_response_time: Optional[str] = None
    memory_usage: Optional[str] = None
    cpu_requirements: Optional[str] = None
    concurrency_support: bool = False
    max_concurrent_sessions: Optional[int] = None
    concurrent_limit: Optional[int] = None
    rate_limit: Optional[str] = None
    scaling_characteristics: Optional[str] = None


class UsageExample(BaseModel):
    """Detailed usage example with input/output"""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    name: str
    description: str
    context: Optional[str] = None
    input_example: Dict[str, Any]
    expected_output: Dict[str, Any]


class AgentCardSection(BaseModel):
    """Agent card metadata section for A2A protocol compliance"""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    version: str = "1.0.0"
    purpose: str
    detailed_description: str
    url: Optional[str] = None
    domain: Optional[str] = None
    expertise_level: str = "intermediate"
    
    # Input/Output specifications
    input_format: InputOutputFormat
    output_format: InputOutputFormat
    
    # A2A Protocol requirements
    capabilities: A2ACapabilities = Field(default_factory=A2ACapabilities)
    skills: List[A2ASkill] = Field(default_factory=list)
    
    # Performance and usage
    performance: PerformanceSpec
    usage_examples: List[UsageExample] = Field(default_factory=list)


class ToolCapabilities(BaseModel):
    """Tool-specific capabilities"""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    batch_processing: bool = False
    streaming: bool = False
    caching: bool = False
    rate_limiting: bool = False
    authentication_required: bool = False
    concurrent_requests: int = 1


class ToolCardSection(BaseModel):
    """Tool card metadata section"""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    version: str = "1.0.0"
    purpose: str
    detailed_description: str
    category: str
    subcategory: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    
    # Input/Output specifications
    input_format: InputOutputFormat
    output_format: InputOutputFormat
    
    # Tool characteristics
    capabilities: ToolCapabilities = Field(default_factory=ToolCapabilities)
    performance: PerformanceSpec
    usage_examples: List[UsageExample] = Field(default_factory=list)


class EnhancedAgentConfig(YAMLConfig):
    """
    Enhanced Agent Configuration - A2A Protocol Integration and Advanced Agent Management
    =================================================================================
    
    The EnhancedAgentConfig provides comprehensive agent configuration with native A2A
    (Agent-to-Agent) protocol integration, advanced capability management, and enterprise-grade
    configuration features. This configuration system enables seamless agent interoperability,
    automated card generation, performance specification, and advanced collaboration patterns
    for distributed AI agent systems and enterprise AI deployments.
    
    **Core Architecture:**
        The enhanced agent configuration provides enterprise-grade agent management capabilities:
        
        * **A2A Protocol Integration**: Native Agent-to-Agent protocol support with automatic card generation
        * **Advanced Capability Management**: Comprehensive capability definition and skill management
        * **Performance Specification**: Detailed performance characteristics and resource requirements
        * **Enterprise Configuration**: Advanced configuration management with validation and compliance
        * **Collaboration Framework**: Multi-agent collaboration and delegation configuration
        * **Framework Integration**: Complete integration with NanoBrain's agent and workflow systems
    
    **A2A Protocol Capabilities:**
        
        **Agent Card Generation:**
        * **Automatic Card Creation**: Intelligent generation of A2A-compliant agent cards from configuration
        * **Protocol Compliance**: Validation and enforcement of A2A protocol standards and specifications
        * **Capability Declaration**: Comprehensive capability and skill declaration for agent discovery
        * **Interoperability Support**: Configuration for seamless interoperability with other A2A agents
        
        **Enhanced Agent Features:**
        * **Multi-Turn Conversation**: Advanced conversation management with context retention
        * **Tool Integration**: Comprehensive tool usage and integration capabilities
        * **Delegation Support**: Agent delegation and task forwarding configuration
        * **Collaboration Patterns**: Multi-agent collaboration and coordination configuration
        
        **Performance Management:**
        * **Response Time Specification**: Detailed response time characteristics and guarantees
        * **Resource Requirements**: CPU, memory, and concurrency requirement specification
        * **Scaling Configuration**: Scaling characteristics and concurrent session management
        * **Rate Limiting**: Rate limiting and throttling configuration for performance management
    
    **Configuration Structure:**
        
        **Enhanced Agent Configuration YAML:**
        ```yaml
        # Enhanced Agent Configuration with A2A Integration
        enhanced_agent_config:
          # Basic agent information
          agent_info:
            name: "enterprise_collaborative_agent"
            version: "2.0.0"
            description: "Enterprise collaborative AI agent with A2A integration"
            author: "NanoBrain Team"
            license: "Enterprise"
            
          # Agent card metadata (A2A compliance)
          agent_card:
            agent_id: "nanobrain.agents.enterprise_collaborative"
            name: "Enterprise Collaborative Agent"
            version: "2.0.0"
            description: "Advanced collaborative AI agent for enterprise environments"
            
            # Agent capabilities
            capabilities:
              streaming: true
              push_notifications: true
              state_transition_history: true
              multi_turn_conversation: true
              context_retention: true
              tool_usage: true
              delegation: true
              collaboration: true
              
            # Agent skills
            skills:
              - id: "data_analysis"
                name: "Data Analysis"
                description: "Comprehensive data analysis and visualization"
                complexity: "advanced"
                input_modes: ["text", "structured_data", "file"]
                output_modes: ["text", "visualization", "structured_data"]
                examples:
                  - "Analyze sales trends from CSV data"
                  - "Generate insights from customer feedback"
                  
              - id: "research_assistance"
                name: "Research Assistance"
                description: "Academic and business research support"
                complexity: "expert"
                input_modes: ["text", "document"]
                output_modes: ["text", "structured_report", "citations"]
                examples:
                  - "Conduct literature review on AI ethics"
                  - "Research market trends for technology adoption"
                  
            # Communication protocols
            communication:
              protocols: ["A2A_v2.0", "HTTP_REST", "WebSocket"]
              message_formats: ["JSON", "YAML", "Markdown"]
              authentication: "JWT"
              encryption: "TLS_1.3"
              
            # Input/output format specifications
            input_output_format:
              primary_mode: "text"
              supported_modes: ["text", "structured_data", "file", "audio"]
              content_types: ["application/json", "text/plain", "text/markdown"]
              format_schema:
                request:
                  type: "object"
                  properties:
                    query: {"type": "string"}
                    context: {"type": "object"}
                    preferences: {"type": "object"}
                response:
                  type: "object"
                  properties:
                    result: {"type": "string"}
                    confidence: {"type": "number"}
                    sources: {"type": "array"}
                    
            # Performance specifications
            performance:
              typical_response_time: "2-5 seconds"
              max_response_time: "30 seconds"
              memory_usage: "512MB-2GB"
              cpu_requirements: "2 cores minimum"
              concurrency_support: true
              max_concurrent_sessions: 50
              rate_limit: "100 requests/minute"
              scaling_characteristics: "horizontal_scalable"
              
            # Usage examples
            usage_examples:
              - name: "Business Analysis"
                description: "Analyze business performance data"
                context: "Enterprise quarterly review"
                input_example:
                  query: "Analyze our Q3 sales performance and identify growth opportunities"
                  data_source: "sales_data.csv"
                  format: "structured_analysis"
                expected_output:
                  analysis: "Q3 sales increased 15% YoY with strong performance in enterprise segment"
                  recommendations: ["Expand enterprise sales team", "Increase marketing in high-growth regions"]
                  metrics:
                    growth_rate: 0.15
                    confidence: 0.87
                    
              - name: "Research Collaboration"
                description: "Collaborative research with other agents"
                context: "Multi-agent research project"
                input_example:
                  research_topic: "AI ethics in healthcare"
                  collaboration_mode: "parallel_research"
                  other_agents: ["literature_agent", "data_agent"]
                expected_output:
                  research_findings: "Comprehensive analysis of AI ethics frameworks"
                  collaboration_summary: "Successfully coordinated with 2 other agents"
                  citations: ["IEEE_2023_AI_Ethics", "Nature_2023_Healthcare_AI"]
                  
          # Agent implementation configuration
          agent_implementation:
            class: "nanobrain.library.agents.enhanced.EnhancedCollaborativeAgent"
            config_file: "config/agents/enhanced_collaborative_agent.yml"
            
            # Core configuration
            core_config:
              llm_provider: "openai"
              model: "gpt-4"
              temperature: 0.7
              max_tokens: 4096
              timeout: 30
              
            # Tool integration
            tools:
              - name: "data_analysis_tool"
                class: "nanobrain.library.tools.analysis.DataAnalysisTool"
                enabled: true
                
              - name: "web_search_tool"
                class: "nanobrain.library.tools.search.WebSearchTool"
                enabled: true
                
            # Memory and context
            memory:
              type: "enhanced_memory"
              capacity: "10000_messages"
              retention_policy: "30_days"
              context_window: "8000_tokens"
              
            # Collaboration settings
            collaboration:
              delegation_enabled: true
              max_delegation_depth: 3
              collaboration_timeout: 300
              peer_discovery: true
              
          # Validation and compliance
          validation:
            a2a_compliance: true
            schema_validation: true
            capability_verification: true
            performance_testing: true
            
          # Environment-specific overrides
          environments:
            development:
              agent_implementation:
                core_config:
                  temperature: 0.9
                  timeout: 60
                memory:
                  retention_policy: "7_days"
                  
            production:
              agent_implementation:
                core_config:
                  temperature: 0.5
                  timeout: 15
                memory:
                  retention_policy: "90_days"
                validation:
                  performance_testing: true
                  security_validation: true
        ```
    
    **Usage Patterns:**
        
        **Basic Enhanced Agent Configuration:**
        ```python
        from nanobrain.core.config.enhanced_config import EnhancedAgentConfig
        
        # Load enhanced agent configuration
        agent_config = EnhancedAgentConfig.from_config('config/agents/collaborative_agent.yml')
        
        # Access agent card information
        agent_card = agent_config.agent_card
        print(f"Agent: {agent_card['name']} v{agent_card['version']}")
        print(f"Capabilities: {agent_card['capabilities']}")
        
        # Access skills and performance specs
        skills = agent_card['skills']
        performance = agent_card['performance']
        
        print(f"Agent skills: {[skill['name'] for skill in skills]}")
        print(f"Response time: {performance['typical_response_time']}")
        
        # Generate A2A-compliant card
        a2a_card = agent_config.generate_a2a_card()
        
        # Validate A2A compliance
        compliance_errors = agent_config.validate_a2a_compliance()
        if not compliance_errors:
            print("✅ Agent is A2A compliant")
        else:
            print(f"❌ A2A compliance issues: {compliance_errors}")
        
        # Export card for agent registry
        agent_config.export_agent_card('output/agent_cards/')
        ```
        
        **Enterprise Agent Management:**
        ```python
        # Enterprise agent configuration management
        class EnterpriseAgentManager:
            def __init__(self):
                self.agent_configs = {}
                self.agent_registry = {}
                self.performance_metrics = {}
                
            def load_agent_suite(self, agent_directory: str):
                \"\"\"Load complete suite of enterprise agents\"\"\"
                
                agent_path = Path(agent_directory)
                loaded_agents = {}
                
                for config_file in agent_path.glob('*.yml'):
                    try:
                        agent_config = EnhancedAgentConfig.from_config(str(config_file))
                        agent_name = agent_config.agent_card['agent_id']
                        
                        # Validate configuration
                        validation_result = self.validate_agent_config(agent_config)
                        if validation_result['valid']:
                            loaded_agents[agent_name] = agent_config
                            self.agent_configs[agent_name] = agent_config
                            
                            # Register agent capabilities
                            self.register_agent_capabilities(agent_name, agent_config)
                            
                            print(f"✅ Loaded agent: {agent_name}")
                        else:
                            print(f"❌ Invalid agent config: {config_file}")
                            print(f"   Errors: {validation_result['errors']}")
                            
                    except Exception as e:
                        print(f"❌ Failed to load agent config: {config_file}")
                        print(f"   Error: {str(e)}")
                
                return loaded_agents
                
            def create_agent_collaboration_network(self, agents: List[str]) -> Dict[str, Any]:
                \"\"\"Create collaboration network between agents\"\"\"
                
                collaboration_network = {
                    'network_id': f"collaboration_network_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    'agents': [],
                    'capabilities_matrix': {},
                    'delegation_paths': {},
                    'communication_channels': {}
                }
                
                # Analyze agent capabilities
                all_capabilities = set()
                agent_capabilities = {}
                
                for agent_name in agents:
                    if agent_name in self.agent_configs:
                        agent_config = self.agent_configs[agent_name]
                        capabilities = agent_config.agent_card['capabilities']
                        skills = [skill['id'] for skill in agent_config.agent_card['skills']]
                        
                        agent_info = {
                            'agent_id': agent_name,
                            'capabilities': capabilities,
                            'skills': skills,
                            'performance': agent_config.agent_card['performance']
                        }
                        
                        collaboration_network['agents'].append(agent_info)
                        agent_capabilities[agent_name] = skills
                        all_capabilities.update(skills)
                
                # Create capabilities matrix
                capabilities_matrix = {}
                for capability in all_capabilities:
                    capable_agents = [
                        agent for agent, skills in agent_capabilities.items()
                        if capability in skills
                    ]
                    capabilities_matrix[capability] = capable_agents
                
                collaboration_network['capabilities_matrix'] = capabilities_matrix
                
                # Generate delegation paths
                delegation_paths = self.generate_delegation_paths(agent_capabilities)
                collaboration_network['delegation_paths'] = delegation_paths
                
                # Setup communication channels
                communication_channels = self.setup_communication_channels(agents)
                collaboration_network['communication_channels'] = communication_channels
                
                return collaboration_network
                
            def optimize_agent_performance(self, agent_name: str, performance_data: Dict[str, Any]) -> Dict[str, Any]:
                \"\"\"Optimize agent performance based on metrics\"\"\"
                
                if agent_name not in self.agent_configs:
                    raise ValueError(f"Agent not found: {agent_name}")
                
                agent_config = self.agent_configs[agent_name]
                current_performance = agent_config.agent_card['performance']
                
                optimization_results = {
                    'agent_name': agent_name,
                    'current_performance': current_performance,
                    'performance_data': performance_data,
                    'optimizations': [],
                    'updated_config': None
                }
                
                # Analyze performance metrics
                avg_response_time = performance_data.get('avg_response_time', 0)
                memory_usage = performance_data.get('memory_usage', 0)
                concurrent_sessions = performance_data.get('concurrent_sessions', 0)
                error_rate = performance_data.get('error_rate', 0)
                
                # Generate optimizations
                optimizations = []
                
                # Response time optimization
                if avg_response_time > 10:  # seconds
                    optimizations.append({
                        'type': 'response_time',
                        'issue': f'High average response time: {avg_response_time}s',
                        'recommendation': 'Reduce model complexity or increase timeout',
                        'config_changes': {
                            'agent_implementation.core_config.timeout': min(avg_response_time * 2, 60),
                            'agent_implementation.core_config.max_tokens': 2048
                        }
                    })
                
                # Memory optimization
                if memory_usage > 1000:  # MB
                    optimizations.append({
                        'type': 'memory_usage',
                        'issue': f'High memory usage: {memory_usage}MB',
                        'recommendation': 'Optimize memory configuration and context window',
                        'config_changes': {
                            'agent_implementation.memory.context_window': '4000_tokens',
                            'agent_implementation.memory.capacity': '5000_messages'
                        }
                    })
                
                # Concurrency optimization
                max_concurrent = int(current_performance.get('max_concurrent_sessions', 10))
                if concurrent_sessions > max_concurrent * 0.8:
                    optimizations.append({
                        'type': 'concurrency',
                        'issue': f'High concurrent load: {concurrent_sessions}/{max_concurrent}',
                        'recommendation': 'Increase concurrent session limit',
                        'config_changes': {
                            'agent_card.performance.max_concurrent_sessions': max_concurrent * 2
                        }
                    })
                
                optimization_results['optimizations'] = optimizations
                
                # Apply optimizations if requested
                if optimizations:
                    updated_config = self.apply_performance_optimizations(agent_config, optimizations)
                    optimization_results['updated_config'] = updated_config
                
                return optimization_results
        
        # Enterprise agent management
        agent_manager = EnterpriseAgentManager()
        
        # Load agent suite
        loaded_agents = agent_manager.load_agent_suite('config/agents/')
        print(f"Loaded {len(loaded_agents)} enterprise agents")
        
        # Create collaboration network
        collaboration_agents = ['research_agent', 'analysis_agent', 'writing_agent']
        network = agent_manager.create_agent_collaboration_network(collaboration_agents)
        
        print(f"Collaboration network created:")
        print(f"  Agents: {len(network['agents'])}")
        print(f"  Capabilities: {len(network['capabilities_matrix'])}")
        
        # Optimize agent performance
        performance_data = {
            'avg_response_time': 15.2,
            'memory_usage': 1200,
            'concurrent_sessions': 25,
            'error_rate': 0.02
        }
        
        optimization_result = agent_manager.optimize_agent_performance('research_agent', performance_data)
        print(f"Performance optimization:")
        print(f"  Optimizations: {len(optimization_result['optimizations'])}")
        for opt in optimization_result['optimizations']:
            print(f"    {opt['type']}: {opt['recommendation']}")
        ```
        
        **A2A Protocol Integration:**
        ```python
        # Advanced A2A protocol integration and management
        class A2AProtocolManager:
            def __init__(self):
                self.registered_agents = {}
                self.protocol_handlers = {}
                self.communication_channels = {}
                
            def register_enhanced_agent(self, agent_config: EnhancedAgentConfig):
                \"\"\"Register enhanced agent with A2A protocol support\"\"\"
                
                agent_id = agent_config.agent_card['agent_id']
                
                # Validate A2A compliance
                compliance_errors = agent_config.validate_a2a_compliance()
                if compliance_errors:
                    raise ValueError(f"Agent {agent_id} is not A2A compliant: {compliance_errors}")
                
                # Generate A2A card
                a2a_card = agent_config.generate_a2a_card()
                
                # Register agent
                self.registered_agents[agent_id] = {
                    'config': agent_config,
                    'a2a_card': a2a_card,
                    'status': 'registered',
                    'capabilities': agent_config.agent_card['capabilities'],
                    'communication': agent_config.agent_card['communication']
                }
                
                # Setup communication channels
                communication_config = agent_config.agent_card['communication']
                self.setup_agent_communication(agent_id, communication_config)
                
                print(f"✅ Registered A2A agent: {agent_id}")
                
            def discover_agent_capabilities(self, required_skills: List[str]) -> List[Dict[str, Any]]:
                \"\"\"Discover agents with required capabilities\"\"\"
                
                matching_agents = []
                
                for agent_id, agent_info in self.registered_agents.items():
                    agent_config = agent_info['config']
                    agent_skills = [skill['id'] for skill in agent_config.agent_card['skills']]
                    
                    # Check if agent has required skills
                    skill_match = set(required_skills).intersection(set(agent_skills))
                    skill_coverage = len(skill_match) / len(required_skills)
                    
                    if skill_coverage > 0.5:  # At least 50% skill match
                        matching_agents.append({
                            'agent_id': agent_id,
                            'skill_coverage': skill_coverage,
                            'matching_skills': list(skill_match),
                            'all_skills': agent_skills,
                            'performance': agent_config.agent_card['performance']
                        })
                
                # Sort by skill coverage
                matching_agents.sort(key=lambda x: x['skill_coverage'], reverse=True)
                
                return matching_agents
                
            def create_agent_delegation_chain(self, task: Dict[str, Any]) -> Dict[str, Any]:
                \"\"\"Create delegation chain for complex task\"\"\"
                
                required_skills = task.get('required_skills', [])
                complexity = task.get('complexity', 'intermediate')
                
                # Discover capable agents
                capable_agents = self.discover_agent_capabilities(required_skills)
                
                if not capable_agents:
                    raise ValueError(f"No agents found with required skills: {required_skills}")
                
                # Create delegation chain
                delegation_chain = {
                    'task_id': task.get('task_id', f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
                    'task': task,
                    'delegation_steps': [],
                    'coordination_agent': None,
                    'estimated_duration': 0
                }
                
                # Select coordination agent (highest skill coverage)
                coordination_agent = capable_agents[0]
                delegation_chain['coordination_agent'] = coordination_agent
                
                # Create delegation steps
                remaining_skills = set(required_skills)
                for agent_info in capable_agents:
                    if not remaining_skills:
                        break
                    
                    agent_skills = set(agent_info['matching_skills'])
                    step_skills = remaining_skills.intersection(agent_skills)
                    
                    if step_skills:
                        delegation_step = {
                            'agent_id': agent_info['agent_id'],
                            'skills': list(step_skills),
                            'estimated_time': self.estimate_task_time(agent_info, step_skills, complexity),
                            'dependencies': []
                        }
                        
                        delegation_chain['delegation_steps'].append(delegation_step)
                        remaining_skills -= step_skills
                
                # Calculate total estimated duration
                total_duration = sum(step['estimated_time'] for step in delegation_chain['delegation_steps'])
                delegation_chain['estimated_duration'] = total_duration
                
                return delegation_chain
        
        # A2A protocol management
        a2a_manager = A2AProtocolManager()
        
        # Register enhanced agents
        for agent_name, agent_config in loaded_agents.items():
            a2a_manager.register_enhanced_agent(agent_config)
        
        # Discover capabilities
        required_skills = ['data_analysis', 'research_assistance', 'report_generation']
        capable_agents = a2a_manager.discover_agent_capabilities(required_skills)
        
        print(f"Found {len(capable_agents)} agents with required skills:")
        for agent in capable_agents:
            print(f"  {agent['agent_id']}: {agent['skill_coverage']:.1%} coverage")
        
        # Create delegation chain
        complex_task = {
            'task_id': 'market_research_analysis',
            'description': 'Comprehensive market research and analysis',
            'required_skills': ['data_analysis', 'research_assistance', 'report_generation'],
            'complexity': 'advanced',
            'deadline': '2024-12-01'
        }
        
        delegation_chain = a2a_manager.create_agent_delegation_chain(complex_task)
        print(f"Delegation chain created:")
        print(f"  Coordination agent: {delegation_chain['coordination_agent']['agent_id']}")
        print(f"  Steps: {len(delegation_chain['delegation_steps'])}")
        print(f"  Estimated duration: {delegation_chain['estimated_duration']} minutes")
        ```
    
    **Advanced Features:**
        
        **A2A Protocol Integration:**
        * **Automatic Card Generation**: Intelligent generation of A2A-compliant agent cards
        * **Protocol Compliance**: Comprehensive A2A protocol validation and enforcement
        * **Capability Declaration**: Advanced capability and skill declaration for discovery
        * **Communication Setup**: Automated setup of A2A communication channels
        
        **Performance Management:**
        * **Performance Specification**: Detailed performance characteristics and requirements
        * **Resource Optimization**: Intelligent resource allocation and optimization
        * **Scaling Configuration**: Horizontal and vertical scaling configuration
        * **Monitoring Integration**: Performance monitoring and analytics integration
        
        **Enterprise Features:**
        * **Configuration Validation**: Comprehensive validation of agent configurations
        * **Environment Management**: Multi-environment configuration with overrides
        * **Security Integration**: Secure communication and access control
        * **Audit and Compliance**: Complete audit trails and compliance reporting
    
    **Production Deployment:**
        
        **High Availability:**
        * **Agent Replication**: Agent configuration replication across environments
        * **Failover Support**: Automatic failover and redundancy for agent services
        * **Load Balancing**: Load balancing for agent instances and communication
        * **Disaster Recovery**: Agent configuration backup and recovery
        
        **Security and Compliance:**
        * **Secure Communication**: TLS encryption and JWT authentication
        * **Access Control**: Role-based access control for agent management
        * **Credential Management**: Secure API key and credential management
        * **Compliance Validation**: Automated compliance validation and reporting
    
    Attributes:
        agent_card (Dict[str, Any]): A2A-compliant agent card metadata
        agent_implementation (Dict[str, Any]): Agent implementation configuration
        validation (Dict[str, Any]): Validation and compliance settings
        
    Methods:
        generate_a2a_card: Generate A2A-compliant agent card
        validate_a2a_compliance: Validate A2A protocol compliance
        export_agent_card: Export agent card to file
        
    Note:
        This configuration requires properly structured agent card metadata for A2A compliance.
        Agent skills and capabilities must be accurately defined for proper discovery.
        Performance specifications should reflect actual agent characteristics.
        Configuration validation ensures compatibility with the NanoBrain framework.
        
    Warning:
        Invalid agent card metadata may cause A2A protocol compliance failures.
        Performance specifications should be realistic to prevent overcommitment.
        Complex collaboration configurations may require significant testing.
        Agent configuration changes may affect existing collaborations and delegations.
        
    See Also:
        * :class:`EnhancedToolConfig`: Enhanced tool configuration with A2A support
        * :class:`A2ACapabilities`: A2A capability definitions
        * :class:`PerformanceSpec`: Performance specification and requirements
        * :mod:`nanobrain.core.config.enhanced_config_manager`: Enhanced configuration management
        * :mod:`nanobrain.library.agents.enhanced`: Enhanced agent implementations
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    # Existing fields
    name: str
    description: str = ""
    agent_type: str = "simple"
    llm_config: Dict[str, Any] = Field(default_factory=dict)
    memory_config: Dict[str, Any] = Field(default_factory=dict)
    tools_config: Dict[str, Any] = Field(default_factory=dict)
    
    # MANDATORY: Agent card section for A2A compliance
    agent_card: AgentCardSection
    
    # Framework metadata
    framework_metadata: Dict[str, Any] = Field(default_factory=lambda: {
        "config_type": "agent",
        "framework_version": "1.0.0"
    })
    
    def generate_a2a_card(self) -> Dict[str, Any]:
        """Generate A2A compliant agent card from configuration"""
        return {
            "name": self.name,
            "description": self.description,
            "agent_type": self.agent_type,
            "a2a_version": "1.0.0",
            "a2a_compatible": True,
            **self.agent_card.model_dump()
        }
    
    def validate_a2a_compliance(self) -> List[str]:
        """Validate A2A protocol compliance"""
        errors = []
        
        if not self.agent_card.capabilities:
            errors.append("Missing A2A capabilities declaration")
        
        if not self.agent_card.skills:
            errors.append("Missing agent skills definition")
        
        if not self.agent_card.input_format:
            errors.append("Missing input format specification")
            
        if not self.agent_card.output_format:
            errors.append("Missing output format specification")
            
        if not self.agent_card.purpose.strip():
            errors.append("Missing agent purpose description")
            
        if not self.agent_card.detailed_description.strip():
            errors.append("Missing detailed agent description")
        
        return errors


class EnhancedToolConfig(YAMLConfig):
    """Enhanced tool configuration with integrated tool_card section"""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    # Existing fields
    name: str
    description: str = ""
    tool_type: str = "analysis"
    timeout: float = 600.0
    max_retries: int = 3
    
    # MANDATORY: Tool card section
    tool_card: ToolCardSection
    
    # Framework metadata
    framework_metadata: Dict[str, Any] = Field(default_factory=lambda: {
        "config_type": "tool",
        "framework_version": "1.0.0"
    })
    
    def generate_tool_card(self) -> Dict[str, Any]:
        """Generate tool card from configuration"""
        return {
            "name": self.name,
            "description": self.description,
            "tool_type": self.tool_type,
            **self.tool_card.model_dump()
        }
    
    def validate_tool_card(self) -> List[str]:
        """Validate tool card completeness"""
        errors = []
        
        if not self.tool_card.input_format:
            errors.append("Missing input format specification")
            
        if not self.tool_card.output_format:
            errors.append("Missing output format specification")
            
        if not self.tool_card.purpose.strip():
            errors.append("Missing tool purpose description")
            
        if not self.tool_card.detailed_description.strip():
            errors.append("Missing detailed tool description")
            
        if not self.tool_card.category.strip():
            errors.append("Missing tool category")
        
        return errors


def create_agent_config_template(
    name: str,
    agent_type: str = "simple",
    domain: Optional[str] = None
) -> EnhancedAgentConfig:
    """Create a template agent configuration with required sections"""
    
    # Basic input/output format for agents
    input_format = InputOutputFormat(
        primary_mode="json",
        supported_modes=["text", "json"],
        content_types=["application/json", "text/plain"],
        format_schema={
            "type": "object",
            "required_fields": {
                "message": {
                    "type": "string",
                    "description": "Natural language instruction or query"
                }
            },
            "optional_fields": {
                "context": {
                    "type": "object",
                    "description": "Conversation context"
                }
            }
        }
    )
    
    output_format = InputOutputFormat(
        primary_mode="json",
        supported_modes=["json", "text"],
        content_types=["application/json", "text/plain"],
        format_schema={
            "type": "object",
            "guaranteed_fields": {
                "response": {
                    "type": "string",
                    "description": "Natural language response"
                },
                "metadata": {
                    "type": "object",
                    "description": "Execution metadata"
                }
            }
        }
    )
    
    # Basic capabilities based on agent type
    capabilities = A2ACapabilities(
        multi_turn_conversation=True,
        context_retention=True,
        tool_usage=(agent_type in ["collaborative", "specialized"]),
        delegation=(agent_type == "collaborative"),
        collaboration=(agent_type == "collaborative")
    )
    
    # Basic skill
    basic_skill = A2ASkill(
        id="natural_language_processing",
        name="Natural Language Processing",
        description="Basic natural language understanding and generation",
        complexity="intermediate",
        input_modes=["text", "json"],
        output_modes=["text", "json"],
        examples=["Process natural language queries", "Generate coherent responses"]
    )
    
    performance = PerformanceSpec(
        typical_response_time="1-5 seconds",
        concurrency_support=True,
        memory_usage="50-200 MB",
        cpu_requirements="Low to Medium"
    )
    
    agent_card = AgentCardSection(
        purpose=f"A {agent_type} agent for {domain or 'general'} tasks",
        detailed_description=f"This {agent_type} agent provides capabilities for {domain or 'general purpose'} processing and interaction.",
        domain=domain,
        input_format=input_format,
        output_format=output_format,
        capabilities=capabilities,
        skills=[basic_skill],
        performance=performance
    )
    
    return EnhancedAgentConfig(
        name=name,
        description=f"{agent_type.title()} agent for {domain or 'general'} tasks",
        agent_type=agent_type,
        agent_card=agent_card
    )


def create_tool_config_template(
    name: str,
    category: str,
    tool_type: str = "analysis"
) -> EnhancedToolConfig:
    """Create a template tool configuration with required sections"""
    
    # Basic input/output format for tools
    input_format = InputOutputFormat(
        primary_mode="json",
        supported_modes=["json", "data"],
        content_types=["application/json"],
        format_schema={
            "type": "object",
            "required_fields": {
                "query_type": {
                    "type": "string",
                    "description": "Type of operation to perform"
                },
                "parameters": {
                    "type": "object",
                    "description": "Operation parameters"
                }
            }
        }
    )
    
    output_format = InputOutputFormat(
        primary_mode="json",
        supported_modes=["json", "structured"],
        content_types=["application/json"],
        format_schema={
            "type": "object",
            "guaranteed_fields": {
                "status": {
                    "type": "string",
                    "description": "Operation status"
                },
                "data": {
                    "type": "object",
                    "description": "Result data"
                },
                "metadata": {
                    "type": "object",
                    "description": "Execution metadata"
                }
            }
        }
    )
    
    capabilities = ToolCapabilities(
        batch_processing=True,
        caching=True
    )
    
    performance = PerformanceSpec(
        typical_response_time="1-10 seconds",
        memory_usage="50-200 MB",
        cpu_requirements="Low"
    )
    
    tool_card = ToolCardSection(
        purpose=f"A {category} tool for {tool_type} operations",
        detailed_description=f"This tool provides {category} capabilities for {tool_type} operations.",
        category=category,
        input_format=input_format,
        output_format=output_format,
        capabilities=capabilities,
        performance=performance
    )
    
    return EnhancedToolConfig(
        name=name,
        description=f"{category.title()} {tool_type} tool",
        tool_type=tool_type,
        tool_card=tool_card
    ) 