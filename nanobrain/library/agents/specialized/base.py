"""
Base Agent Classes

Base classes for specialized agents in the NanoBrain framework.
"""

import asyncio
import time
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from abc import ABC, abstractmethod
from pathlib import Path

# Updated imports for nanobrain package structure
from nanobrain.core.component_base import FromConfigBase
from nanobrain.core.agent import Agent, SimpleAgent, ConversationalAgent, AgentConfig
from nanobrain.core.executor import LocalExecutor, ExecutorConfig
from nanobrain.core.logging_system import NanoBrainLogger, get_logger


class SpecializedAgentBase(ABC):
    """
    Base mixin for specialized agents that provides common specialized functionality.
    
    This class provides:
    - Specialized processing patterns
    - Domain-specific error handling
    - Performance tracking for specialized operations
    - Integration with core agent capabilities
    """
    
    def __init__(self, **kwargs):
        """Initialize specialized agent base."""
        super().__init__(**kwargs)
        
        # Specialized agent tracking
        self._specialized_operations_count = 0
        self._specialized_errors_count = 0
        self._domain_specific_metrics = {}
        
        # Get specialized logger
        if hasattr(self, 'name'):
            self.specialized_logger = get_logger(f"specialized.{self.name}")
        else:
            self.specialized_logger = get_logger("specialized.agent")
    
    async def initialize(self) -> None:
        """Initialize the specialized agent."""
        await super().initialize()
        await self._initialize_specialized_features()
        
        self.specialized_logger.info(f"Specialized agent {getattr(self, 'name', 'unknown')} initialized")
    
    async def shutdown(self) -> None:
        """Shutdown the specialized agent."""
        await self._shutdown_specialized_features()
        await super().shutdown()
        
        self.specialized_logger.info(
            f"Specialized agent {getattr(self, 'name', 'unknown')} shutdown",
            specialized_operations=self._specialized_operations_count,
            specialized_errors=self._specialized_errors_count
        )
    
    async def _initialize_specialized_features(self) -> None:
        """Initialize specialized features. Override in subclasses."""
        pass
    
    async def _shutdown_specialized_features(self) -> None:
        """Shutdown specialized features. Override in subclasses."""
        pass
    
    def _track_specialized_operation(self, operation_name: str, success: bool = True) -> None:
        """Track specialized operation metrics."""
        self._specialized_operations_count += 1
        
        if not success:
            self._specialized_errors_count += 1
        
        # Track domain-specific metrics
        if operation_name not in self._domain_specific_metrics:
            self._domain_specific_metrics[operation_name] = {'count': 0, 'errors': 0}
        
        self._domain_specific_metrics[operation_name]['count'] += 1
        if not success:
            self._domain_specific_metrics[operation_name]['errors'] += 1
    
    def get_specialized_performance_stats(self) -> Dict[str, Any]:
        """Get specialized performance statistics."""
        base_stats = {}
        if hasattr(super(), 'get_performance_stats'):
            base_stats = super().get_performance_stats()
        
        specialized_stats = {
            'specialized_operations_count': self._specialized_operations_count,
            'specialized_errors_count': self._specialized_errors_count,
            'specialized_error_rate': (
                self._specialized_errors_count / max(1, self._specialized_operations_count)
            ),
            'domain_specific_metrics': self._domain_specific_metrics.copy()
        }
        
        return {**base_stats, **specialized_stats}
    
    @abstractmethod
    async def _process_specialized_request(self, input_text: str, **kwargs) -> Optional[str]:
        """
        Process specialized requests that don't require LLM.
        
        Args:
            input_text: Input text to process
            **kwargs: Additional parameters
            
        Returns:
            Processed result if handled, None if should fall back to LLM
        """
        pass
    
    def _should_handle_specialized(self, input_text: str, **kwargs) -> bool:
        """
        Determine if this request should be handled by specialized logic.
        
        Args:
            input_text: Input text
            **kwargs: Additional parameters
            
        Returns:
            True if should be handled by specialized logic
        """
        return False


class BaseAgent(FromConfigBase, ABC):
    """
    Enhanced Base Agent with Universal Tool Loading
    
    Implements comprehensive tool loading via ConfigBase class+config patterns.
    All agent subclasses inherit this functionality automatically unless they
    need to override for additional complexity.
    
    ✅ FRAMEWORK COMPLIANCE:
    - Universal tool loading through ConfigBase._resolve_nested_objects()
    - Complete validation of agent and tool configurations
    - No manual tool instantiation or factory logic
    - Pure configuration-driven tool availability
    - Inherited by all agent types without code duplication
    """
    
    COMPONENT_TYPE = "base_agent"
    REQUIRED_CONFIG_FIELDS = ['name']
    OPTIONAL_CONFIG_FIELDS = {
        'description': '',
        'auto_initialize': True,
        'debug_mode': False,
        'enable_logging': True,
        'capabilities': []
    }
    
    @classmethod
    def _get_config_class(cls):
        """Return the appropriate config class for this agent type"""
        # Default implementation - subclasses can override
        return AgentConfig
    
    @classmethod
    def from_config(cls, config_path: Union[str, Path], **context) -> 'BaseAgent':
        """
        Universal agent loading with automatic tool resolution
        
        This base implementation handles all common agent functionality:
        - Configuration loading and validation via ConfigBase
        - Automatic tool instantiation through class+config patterns
        - Tool validation and agent integration
        - Capability validation against available tools
        
        Subclasses should only override this method if they need additional
        complexity beyond the standard agent + tools pattern.
        
        Args:
            config_path: Path to agent configuration file
            **context: Additional context
            
        Returns:
            Fully initialized agent with tools
            
        Example Configuration (works for all agent types):
        ```yaml
        name: "universal_agent"
        description: "Agent with automatic tool loading"
        class: "nanobrain.library.agents.specialized.conversational_specialized_agent.ConversationalSpecializedAgent"
        
        # Agent-specific configuration (varies by agent type)
        agent_config:
          model: "gpt-4"
          temperature: 0.3
          max_tokens: 1000
          system_prompt: "You are an expert agent..."
          
        # Universal tool loading (same pattern for all agents)
        tools:
          primary_tool:
            class: "nanobrain.library.tools.bioinformatics.bv_brc_tool.BVBRCTool"
            config: "config/tools/BVBRCTool.yml"
          
          analysis_tool:
            class: "nanobrain.library.tools.analysis.sequence_analyzer.SequenceAnalyzer"
            config:
              analysis_mode: "protein"
              enable_secondary_structure: true
          
        capabilities:
          - "data_retrieval"
          - "sequence_analysis"
        ```
        
        ✅ FRAMEWORK COMPLIANCE:
        - ConfigBase._resolve_nested_objects() automatically instantiates all tools
        - Tools validated through their respective ConfigBase schemas
        - No manual tool creation or factory dependencies
        - Complete configuration-driven tool availability
        - Inherited by all agent subclasses without modification
        """
        # Use ConfigBase for validation and automatic tool resolution
        agent_config = cls._get_config_class().from_config(config_path, **context)
        
        # ConfigBase._resolve_nested_objects() has already instantiated all tools
        # specified in the 'tools' section using their class+config patterns
        
        # Create agent instance
        agent = cls.__new__(cls)
        
        # Initialize with universal agent logic
        agent._initialize_from_config(agent_config, **context)
        agent._initialize_universal_tools(agent_config, **context)
        agent._initialize_agent_specifics(agent_config, **context)
        
        return agent
    
    def _initialize_from_config(self, config: AgentConfig, **context) -> None:
        """Initialize basic agent properties from validated configuration"""
        # Basic agent properties
        self.config = config
        self.name = config.name
        self.description = getattr(config, 'description', '')
        
        # Initialize logging
        self.logger = get_logger(f"agent.{self.name}")
        
        # Initialize capabilities
        self.capabilities = getattr(config, 'capabilities', [])
        
        # Initialize agent state
        self.is_active = False
        self.execution_history = []
        
        self.logger.info(f"✅ Initialized base agent: {self.name} with {len(self.capabilities)} capabilities")
    
    def _initialize_universal_tools(self, config: AgentConfig, **context) -> None:
        """
        Universal tool initialization for all agent types
        
        ConfigBase._resolve_nested_objects() has already created all tool instances
        from the 'tools' section. This method extracts and validates them.
        
        ✅ FRAMEWORK COMPLIANCE:
        - Tools are already instantiated via class+config patterns
        - No manual tool creation or factory logic
        - Complete validation through ConfigBase schemas
        - Tools immediately available for agent use
        - Works identically for all agent types
        """
        # Extract instantiated tools from resolved configuration
        tools_config = getattr(config, 'tools', {})
        self.tools = {}
        
        for tool_name, tool_instance in tools_config.items():
            # Validate that it's a proper tool instance
            if hasattr(tool_instance, 'execute') or hasattr(tool_instance, 'run'):
                self.tools[tool_name] = tool_instance
                
                # Bind tool to agent context if supported
                if hasattr(tool_instance, 'set_agent_context'):
                    tool_instance.set_agent_context(self)
                
                # Initialize tool-specific agent integration
                if hasattr(tool_instance, 'initialize_agent_integration'):
                    tool_instance.initialize_agent_integration(self)
                
                self.logger.info(f"✅ Loaded tool: {tool_name} ({tool_instance.__class__.__name__})")
            else:
                self.logger.warning(f"⚠️ Skipping invalid tool instance: {tool_name} (missing execute/run method)")
        
        # Validate tool requirements for capabilities
        self._validate_tool_capability_mapping()
        
        self.logger.info(f"✅ Initialized universal tools: {len(self.tools)} tools available")
    
    def _initialize_agent_specifics(self, config: AgentConfig, **context) -> None:
        """
        Initialize agent-specific configuration
        
        Base implementation does nothing. Subclasses override this method
        to handle their specific initialization requirements without needing
        to override the entire from_config method.
        
        Args:
            config: Validated agent configuration
            **context: Additional context
        """
        # Base implementation - subclasses can override for specific needs
        pass
    
    def _validate_tool_capability_mapping(self) -> None:
        """
        Validate that agent capabilities are supported by available tools
        
        Base implementation provides common capability validation.
        Subclasses can override to provide specific mappings.
        """
        # Default capability-tool mapping (can be overridden by subclasses)
        default_mapping = {
            'data_retrieval': ['bvbrc_tool', 'pubmed_client', 'database_tool'],
            'sequence_analysis': ['sequence_analyzer', 'alignment_tool', 'clustering_tool'],
            'literature_search': ['pubmed_client', 'literature_tool'],
            'protein_analysis': ['protein_analyzer', 'structure_tool', 'domain_tool'],
            'collaboration': ['communication_tool', 'shared_workspace']
        }
        
        # Get specific mapping from subclass if available
        capability_tool_mapping = getattr(self, '_capability_tool_mapping', default_mapping)
        
        missing_tools = []
        for capability in self.capabilities:
            required_tools = capability_tool_mapping.get(capability, [])
            available_tools = set(self.tools.keys())
            
            if required_tools and not any(tool in available_tools for tool in required_tools):
                missing_tools.append({
                    'capability': capability,
                    'required_tools': required_tools,
                    'available_tools': list(available_tools)
                })
        
        if missing_tools:
            self.logger.warning(f"⚠️ Some capabilities may be limited due to missing tools: {missing_tools}")
        else:
            self.logger.info("✅ All capabilities supported by available tools")
    
    # Universal tool access methods (inherited by all agent types)
    def get_tool(self, tool_name: str) -> Optional[Any]:
        """Get tool instance by name"""
        return self.tools.get(tool_name)
    
    def list_available_tools(self) -> List[str]:
        """Get list of available tool names"""
        return list(self.tools.keys())
    
    def execute_with_tool(self, tool_name: str, tool_params: Dict[str, Any]) -> Any:
        """Execute a specific tool with parameters"""
        tool = self.get_tool(tool_name)
        if not tool:
            raise ValueError(f"Tool '{tool_name}' not available. Available tools: {self.list_available_tools()}")
        
        try:
            if hasattr(tool, 'execute'):
                return tool.execute(**tool_params)
            elif hasattr(tool, 'run'):
                return tool.run(**tool_params)
            else:
                raise ValueError(f"Tool '{tool_name}' does not have execute() or run() method")
        except Exception as e:
            self.logger.error(f"❌ Tool execution failed: {tool_name} - {str(e)}")
            raise ValueError(f"Tool execution failed: {tool_name} - {str(e)}") from e


class SimpleSpecializedAgent(BaseAgent, SpecializedAgentBase, SimpleAgent):
    """
    Simple Specialized Agent - Inherits Universal Tool Loading
    
    Uses BaseAgent's universal tool loading mechanism. Only implements
    agent-specific functionality beyond the standard pattern.
    
    ✅ FRAMEWORK COMPLIANCE:
    - Inherits tool loading from BaseAgent automatically
    - Only overrides methods for specialized functionality
    - No duplicate tool loading or from_config logic
    - Pure configuration-driven through inheritance
    """
    
    # Component configuration
    COMPONENT_TYPE = "simple_specialized_agent"
    REQUIRED_CONFIG_FIELDS = ['name']
    OPTIONAL_CONFIG_FIELDS = {}
    
    @classmethod
    def _get_config_class(cls):
        """Return agent specific config class"""
        return AgentConfig
    
    @classmethod
    def extract_component_config(cls, config: AgentConfig) -> Dict[str, Any]:
        """Extract SimpleSpecializedAgent configuration"""
        # Handle both dictionary and object configurations - NO HARDCODED DEFAULTS
        if isinstance(config, dict):
            return {
                'name': config['name'],  # Required field - no default
                'description': config.get('description', ''),  # Optional field can default to empty
                'model': config['model'],  # Required field - no default  
                'system_prompt': config.get('system_prompt', ''),  # Optional field can default to empty
            }
        else:
            return {
                'name': config.name,
                'description': config.description,
                'model': config.model,
                'system_prompt': config.system_prompt,
            }
    
    @classmethod  
    def resolve_dependencies(cls, component_config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Resolve SimpleSpecializedAgent dependencies"""
        # Create executor via from_config to avoid direct instantiation
        from nanobrain.core.executor import LocalExecutor, ExecutorConfig
        
        executor_config = kwargs.get('executor_config') or ExecutorConfig()
        executor = LocalExecutor.from_config(executor_config)
        
        return {
            'executor': executor,
        }
    
    # Now inherits unified from_config implementation from FromConfigBase
        
    def _init_from_config(self, config: AgentConfig, component_config: Dict[str, Any],
                         dependencies: Dict[str, Any]) -> None:
        """Initialize SimpleSpecializedAgent with resolved dependencies"""
        # Extract executor from dependencies to pass to parent
        executor = dependencies.pop('executor', None)
        
        # Initialize parent classes first
        SimpleAgent.__init__(self, config, executor=executor, **dependencies)
        # SpecializedAgentBase doesn't need config, just call super() directly
        self._specialized_operations_count = 0
        self._specialized_errors_count = 0
        self._domain_specific_metrics = {}
        
        # Get specialized logger
        self.specialized_logger = get_logger(f"specialized.{self.name}")
    
    async def process(self, input_text: str, **kwargs) -> str:
        """
        Process input with specialized logic first, then fall back to LLM.
        
        Args:
            input_text: Input text to process
            **kwargs: Additional parameters
            
        Returns:
            Processed response
        """
        # Try specialized processing first
        if self._should_handle_specialized(input_text, **kwargs):
            try:
                specialized_result = await self._process_specialized_request(input_text, **kwargs)
                if specialized_result is not None:
                    self._track_specialized_operation("direct_processing", success=True)
                    return specialized_result
            except Exception as e:
                self._track_specialized_operation("direct_processing", success=False)
                self.specialized_logger.error(f"Specialized processing failed: {e}")
        
        # Fall back to parent LLM processing
        return await super().process(input_text, **kwargs)


class ConversationalSpecializedAgent(BaseAgent, SpecializedAgentBase, ConversationalAgent):
    """
    Conversational Specialized Agent - Inherits Universal Tool Loading
    
    Uses BaseAgent's universal tool loading mechanism. Only implements
    conversational and specialized functionality beyond the standard pattern.
    
    ✅ FRAMEWORK COMPLIANCE:
    - Inherits tool loading from BaseAgent automatically
    - Only overrides methods for conversational and specialized functionality
    - No duplicate tool loading or from_config logic
    - Pure configuration-driven through inheritance
    """
    
    # Component configuration
    COMPONENT_TYPE = "conversational_specialized_agent"
    REQUIRED_CONFIG_FIELDS = ['name']
    OPTIONAL_CONFIG_FIELDS = {}
    
    @classmethod
    def _get_config_class(cls):
        """Return conversational agent specific config class"""
        return AgentConfig
    
    def _initialize_agent_specifics(self, config: AgentConfig, **context) -> None:
        """
        Initialize conversational and specialized functionality
        
        This method is called by BaseAgent.from_config() after universal
        tool loading is complete. Only handles conversational and specialized logic.
        
        ✅ FRAMEWORK COMPLIANCE:
        - Called automatically by inherited BaseAgent.from_config()
        - Tools already loaded and available via self.tools
        - Only implements conversational and specialized initialization
        - No tool loading or validation logic needed
        """
        # Initialize conversational agent functionality
        if hasattr(ConversationalAgent, '__init__'):
            # Initialize conversation state
            self.conversation_history = []
            self.conversation_memory = {}
            self.max_conversation_length = getattr(config, 'max_conversation_length', 50)
            self.enable_memory = getattr(config, 'enable_memory', True)
        
        # Initialize specialized agent functionality  
        self._specialized_operations_count = 0
        self._specialized_errors_count = 0
        self._domain_specific_metrics = {}
        
        # Get specialized logger
        self.specialized_logger = get_logger(f"specialized.{self.name}")
        
        self.logger.info(f"✅ Initialized conversational and specialized functionality for agent: {self.name}")
    
    async def process(self, input_text: str, **kwargs) -> str:
        """
        Process input with specialized logic first, then fall back to conversational LLM.
        
        Args:
            input_text: Input text to process
            **kwargs: Additional parameters
            
        Returns:
            Processed response
        """
        # Try specialized processing first
        if self._should_handle_specialized(input_text, **kwargs):
            try:
                specialized_result = await self._process_specialized_request(input_text, **kwargs)
                if specialized_result is not None:
                    self._track_specialized_operation("direct_processing", success=True)
                    # Add to conversation history for context
                    if hasattr(self, 'add_to_conversation'):
                        self.add_to_conversation("user", input_text)
                        self.add_to_conversation("assistant", specialized_result)
                    return specialized_result
            except Exception as e:
                self._track_specialized_operation("direct_processing", success=False)
                self.specialized_logger.error(f"Specialized processing failed: {e}")
        
        # Fall back to conversational processing (using tools if available)
        if self.tools:
            # Determine if any tools should be used for this input
            relevant_tools = self._analyze_input_for_tool_usage(input_text, **kwargs)
            if relevant_tools:
                tool_results = {}
                for tool_name in relevant_tools:
                    try:
                        tool_result = self.execute_with_tool(tool_name, {'input': input_text, **kwargs})
                        tool_results[tool_name] = tool_result
                    except Exception as e:
                        self.logger.warning(f"Tool {tool_name} execution failed: {e}")
                
                # Enhance response with tool results if available
                if tool_results:
                    enhanced_input = f"Input: {input_text}\nTool Results: {tool_results}"
                    return await self._process_with_conversation_context(enhanced_input, **kwargs)
        
        # Standard conversational processing
        return await self._process_with_conversation_context(input_text, **kwargs)
    
    def _analyze_input_for_tool_usage(self, input_text: str, **kwargs) -> List[str]:
        """
        Analyze input to determine which tools should be used
        
        Args:
            input_text: Input text to analyze
            **kwargs: Additional parameters
            
        Returns:
            List of tool names that should be used
        """
        relevant_tools = []
        
        # Simple keyword-based tool selection (can be enhanced with NLP)
        tool_keywords = {
            'bvbrc_tool': ['virus', 'bacteria', 'genome', 'sequence', 'database'],
            'pubmed_client': ['research', 'paper', 'study', 'literature', 'publication'],
            'sequence_analyzer': ['analyze', 'protein', 'structure', 'domain', 'sequence']
        }
        
        input_lower = input_text.lower()
        
        for tool_name, keywords in tool_keywords.items():
            if tool_name in self.tools and any(keyword in input_lower for keyword in keywords):
                relevant_tools.append(tool_name)
        
        return relevant_tools
    
    async def _process_with_conversation_context(self, input_text: str, **kwargs) -> str:
        """
        Process input with conversation context
        
        This is a simplified implementation that can be enhanced
        with proper conversation management.
        """
        # Add to conversation history if conversation management is available
        if hasattr(self, 'conversation_history'):
            self.conversation_history.append({
                'role': 'user',
                'content': input_text,
                'timestamp': time.time()
            })
        
        # For now, return a simple response
        # In a full implementation, this would integrate with LLM processing
        response = f"Processed conversational input: {input_text}"
        
        # Add response to conversation history
        if hasattr(self, 'conversation_history'):
            self.conversation_history.append({
                'role': 'assistant',
                'content': response,
                'timestamp': time.time()
            })
        
        return response


def create_specialized_agent(
    agent_type: str,
    specialized_class: type,
    config: AgentConfig,
    **kwargs
) -> Agent:
    """
    Factory function to create specialized agents.
    
    Args:
        agent_type: Type of agent ('simple' or 'conversational')
        specialized_class: The specialized agent class
        config: Agent configuration
        **kwargs: Additional arguments
        
    Returns:
        Specialized agent instance
    """
    logger = get_logger("specialized.factory")
    logger.info(
        f"Creating specialized agent: {config.name}",
        agent_type=agent_type,
        specialized_class=specialized_class.__name__
    )
    
    if agent_type.lower() == "simple":
        # Create a simple specialized agent class dynamically
        class SimpleSpecialized(specialized_class, SimpleSpecializedAgent):
            pass
        return SimpleSpecialized(config, **kwargs)
    
    elif agent_type.lower() == "conversational":
        # Create a conversational specialized agent class dynamically
        class ConversationalSpecialized(specialized_class, ConversationalSpecializedAgent):
            pass
        return ConversationalSpecialized(config, **kwargs)
    
    else:
        raise ValueError(f"Unknown agent type: {agent_type}") 