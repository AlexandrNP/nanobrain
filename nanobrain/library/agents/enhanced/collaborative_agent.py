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


class CollaborativeAgent(FromConfigBase, A2ASupportMixin, MCPSupportMixin, ConversationalAgent):
    """Multi-protocol collaborative agent with delegation and coordination."""
    
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