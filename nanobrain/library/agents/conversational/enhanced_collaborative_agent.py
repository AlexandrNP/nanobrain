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
from nanobrain.core.logging_system import get_logger


class EnhancedCollaborativeAgent(A2ASupportMixin, MCPSupportMixin, ConversationalAgent):
    """
    Enhanced conversational agent with A2A and MCP protocol support.
    
    This agent can:
    - Use MCP tools for structured operations
    - Collaborate with A2A agents for specialized tasks
    - Maintain conversation context and history
    - Provide performance metrics and monitoring
    - Apply delegation rules for intelligent task routing
    
    Features:
    - Multi-protocol support (A2A, MCP)
    - Intelligent delegation based on configurable rules
    - Performance tracking and metrics collection
    - Enhanced error handling and fallback mechanisms
    - Extensible tool detection and usage patterns
    """
    
    def __init__(self, config: AgentConfig, **kwargs):
        """
        Initialize the Enhanced Collaborative Agent.
        
        Args:
            config: Agent configuration
            **kwargs: Additional configuration including:
                - delegation_rules: List of rules for A2A delegation
                - tool_keywords: Custom tool detection keywords
                - enable_metrics: Whether to track performance metrics
                - a2a_config_path: Path to A2A configuration
                - mcp_config_path: Path to MCP configuration
        """
        super().__init__(config, **kwargs)
        
        # Enhanced logger
        self.enhanced_logger = get_logger(f"enhanced.{self.name}")
        
        # Collaboration tracking
        self.collaboration_count = 0
        self.tool_usage_count = 0
        
        # Configuration
        self.delegation_rules = kwargs.get('delegation_rules', [])
        self.tool_keywords = kwargs.get('tool_keywords', self._get_default_tool_keywords())
        self.enable_metrics = kwargs.get('enable_metrics', True)
        
        # Protocol configuration paths
        self.a2a_config_path = kwargs.get('a2a_config_path')
        self.mcp_config_path = kwargs.get('mcp_config_path')
        
        # Performance tracking
        self.start_time = datetime.now()
        self.total_requests = 0
        self.successful_delegations = 0
        self.successful_tool_uses = 0
        self.delegation_errors = 0
        self.tool_errors = 0
    
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