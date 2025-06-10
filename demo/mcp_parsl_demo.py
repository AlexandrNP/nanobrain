#!/usr/bin/env python3
"""
MCP (Model Context Protocol) Integration Demo with Parsl

This demo shows how to integrate MCP support with NanoBrain agents in a Parsl workflow,
including YAML configuration support for easy server management.

Key features demonstrated:
- YAML-based MCP server configuration
- Agent configuration with MCP references
- Multiple MCP servers with different authentication methods
- Tool discovery and registration from MCP servers
- Parallel processing with MCP-enabled agents
- Error handling and fallback mechanisms
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the nanobrain src directory to the Python path
nanobrain_src = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(nanobrain_src))

from nanobrain.core.mcp_support import (
    MCPSupportMixin, 
    with_mcp_support, 
    MCPServerConfig, 
    MCPClientConfig,
    create_mcp_server_config,
    MCPConfigurationError
)
from nanobrain.core.agent import ConversationalAgent, AgentConfig
from nanobrain.core.logging_system import get_logger, set_debug_mode

# Enable debug logging
set_debug_mode(True)

class MCPEnabledConversationalAgent(MCPSupportMixin, ConversationalAgent):
    """
    Conversational agent with MCP support.
    
    This agent can connect to MCP servers and use their tools
    for enhanced functionality.
    """
    
    def __init__(self, config: AgentConfig, mcp_config_path: str = None):
        super().__init__(config)
        
        # Set MCP configuration path if provided
        if mcp_config_path:
            self.set_mcp_config_path(mcp_config_path)
    
    async def initialize(self):
        """Initialize the agent and MCP support."""
        print(f"Debug - MCPEnabledConversationalAgent.initialize() called")
        
        # Initialize base agent
        await super().initialize()
        print(f"Debug - Base agent initialized")
        
        # Initialize MCP support
        print(f"Debug - About to call initialize_mcp()")
        await self.initialize_mcp()
        print(f"Debug - initialize_mcp() completed")
        
        self.nb_logger.info(f"MCP-enabled agent {self.name} initialized",
                        mcp_servers=len(self.get_mcp_servers()),
                        mcp_tools=len(self.get_mcp_tools()))
    
    async def shutdown(self):
        """Shutdown the agent and MCP connections."""
        await self.shutdown_mcp()
        await super().shutdown()
    
    async def process_message(self, message: str) -> str:
        """Process a message, potentially using MCP tools."""
        try:
            # Get available MCP tools for context
            mcp_tools = self.get_mcp_tools()
            
            if mcp_tools:
                tool_context = f"\nAvailable MCP tools: {', '.join(mcp_tools)}"
                enhanced_message = message + tool_context
            else:
                enhanced_message = message
            
            # Process with base agent (which may use tools)
            response = await self.process(enhanced_message)
            
            return response
            
        except Exception as e:
            self.nb_logger.error(f"Error processing message with MCP agent: {e}")
            return f"I encountered an error while processing your message: {e}"


async def demo_yaml_configuration():
    """Demonstrate YAML-based MCP configuration."""
    print("\n" + "="*60)
    print("MCP YAML Configuration Demo")
    print("="*60)
    
    logger = get_logger("mcp.demo")
    
    # Path to configuration files
    config_dir = Path(__file__).parent.parent / "config"
    mcp_config_path = config_dir / "mcp_config.yaml"
    agent_config_path = config_dir / "agent_with_mcp.yaml"
    
    print(f"Using MCP config: {mcp_config_path}")
    print(f"Using agent config: {agent_config_path}")
    
    try:
        # Create agent configuration
        agent_config = AgentConfig(
            name="yaml_mcp_agent",
            description="Agent configured via YAML with MCP support",
            model="gpt-3.5-turbo",
            temperature=0.7
        )
        
        # Create MCP-enabled agent with YAML configuration
        agent = MCPEnabledConversationalAgent(
            config=agent_config,
            mcp_config_path=str(mcp_config_path)
        )
        
        # Initialize agent (this will load MCP config from YAML)
        print(f"\nDebug - About to initialize agent...")
        await agent.initialize()
        print(f"Debug - Agent initialization completed")
        
        # Debug: Check what happened during tool discovery
        print(f"\nDebug - MCP Client Status:")
        if agent.mcp_client:
            print(f"  Client initialized: {agent.mcp_client.is_initialized}")
            print(f"  Server tools: {agent.mcp_client.server_tools}")
            print(f"  Server sessions: {list(agent.mcp_client.server_sessions.keys())}")
            
            # Try to manually discover tools from demo_server
            if 'demo_server' in agent.mcp_client.servers:
                print(f"  Manually discovering tools from demo_server...")
                try:
                    tools = await agent.mcp_client.discover_tools('demo_server')
                    print(f"  Discovered tools: {[t.name for t in tools]}")
                except Exception as e:
                    print(f"  Tool discovery error: {e}")
            
            # Check get_all_tools
            try:
                all_tools = await agent.mcp_client.get_all_tools()
                print(f"  get_all_tools returned: {len(all_tools)} tools")
                for tool in all_tools:
                    print(f"    - {tool.name} from {tool.server_name}")
            except Exception as e:
                print(f"  get_all_tools error: {e}")
        
        print(f"\nDebug - Agent MCP Tools:")
        print(f"  MCP tools dict: {agent.mcp_tools}")
        print(f"  MCP tools list: {agent.get_mcp_tools()}")
        
        # Test MCP tools manually
        print(f"\nDebug - Testing MCP tools manually:")
        if hasattr(agent, 'mcp_tools') and agent.mcp_tools:
            for tool_name, tool in agent.mcp_tools.items():
                print(f"  Tool {tool_name}: {tool}")
                try:
                    if tool_name == 'calculator':
                        result = await tool.execute(expression='2+2')
                        print(f"    Manual test result: {result}")
                except Exception as e:
                    print(f"    Manual test error: {e}")
        else:
            print(f"  No MCP tools found in agent.mcp_tools")
            
            # Try to manually call the MCP client tools
            if agent.mcp_client and agent.mcp_client.server_tools:
                print(f"  But MCP client has server tools: {list(agent.mcp_client.server_tools.keys())}")
                for server_name, tools in agent.mcp_client.server_tools.items():
                    print(f"    Server {server_name} has {len(tools)} tools")
                    for tool_info in tools:
                        print(f"      - {tool_info.name}: {tool_info.description}")
                        
                        # Try to call the tool directly through the client
                        try:
                            if tool_info.name == 'calculator':
                                result = await agent.mcp_client.call_tool(server_name, tool_info.name, {'expression': '3+3'})
                                print(f"        Direct client call result: {result}")
                        except Exception as e:
                            print(f"        Direct client call error: {e}")
        
        # Try to manually register tools
        print(f"\nDebug - Manually testing tool registration:")
        if agent.mcp_client and agent.mcp_client.server_tools:
            for server_name, tools in agent.mcp_client.server_tools.items():
                for tool_info in tools:
                    print(f"  Attempting to register {tool_info.name}...")
                    print(f"    Before registration - mcp_tools: {agent.mcp_tools}")
                    print(f"    Before registration - id(mcp_tools): {id(agent.mcp_tools)}")
                    try:
                        await agent._register_mcp_tool(tool_info)
                        print(f"    Successfully registered {tool_info.name}")
                        print(f"    After registration - mcp_tools: {agent.mcp_tools}")
                        print(f"    After registration - id(mcp_tools): {id(agent.mcp_tools)}")
                    except Exception as e:
                        print(f"    Failed to register {tool_info.name}: {e}")
                        import traceback
                        print(f"    Traceback: {traceback.format_exc()}")
        
        print(f"\nDebug - After manual registration:")
        print(f"  MCP tools dict: {agent.mcp_tools}")
        print(f"  MCP tools list: {agent.get_mcp_tools()}")
        print(f"  hasattr(agent, 'mcp_tools'): {hasattr(agent, 'mcp_tools')}")
        print(f"  type(agent.mcp_tools): {type(agent.mcp_tools)}")
        print(f"  id(agent.mcp_tools): {id(agent.mcp_tools)}")
        
        # Display MCP status
        status = agent.get_mcp_status()
        print(f"\nMCP Status:")
        print(f"  Enabled: {status['enabled']}")
        print(f"  Client initialized: {status['client_initialized']}")
        print(f"  Config path: {status['config_path']}")
        print(f"  Total servers: {status['total_servers']}")
        print(f"  Total tools: {status['total_tools']}")
        
        print(f"\nConfigured MCP Servers:")
        for server_name, server_info in status['servers'].items():
            print(f"  - {server_name}:")
            print(f"    URL: {server_info['url']}")
            print(f"    Enabled: {server_info['enabled']}")
            print(f"    Auth: {server_info['auth_type']}")
            print(f"    Connected: {server_info['connected']}")
        
        print(f"\nAvailable MCP Tools:")
        for tool_name, tool_info in status['tools'].items():
            print(f"  - {tool_name}: {tool_info['description']}")
            print(f"    Server: {tool_info['server']}")
        
        # Test tool usage
        print(f"\nTesting MCP Tools:")
        
        # Test calculator tool
        if 'calculator' in status['tools']:
            try:
                result = await agent.call_mcp_tool('calculator', expression='2 + 3 * 4')
                print(f"Calculator result: {result}")
            except Exception as e:
                print(f"Calculator error: {e}")
        
        # Test weather tool
        if 'weather' in status['tools']:
            try:
                result = await agent.call_mcp_tool('weather', location='San Francisco')
                print(f"Weather result: {result}")
            except Exception as e:
                print(f"Weather error: {e}")
        
        # Test conversational interaction
        print(f"\nTesting Conversational Interaction:")
        test_messages = [
            "What's 15 * 7?",
            "What's the weather like in New York?",
            "Can you help me with both a calculation and weather check?"
        ]
        
        for message in test_messages:
            print(f"\nUser: {message}")
            try:
                response = await agent.process_message(message)
                print(f"Agent: {response}")
            except Exception as e:
                print(f"Error: {e}")
        
        # Shutdown
        await agent.shutdown()
        print(f"\nYAML configuration demo completed successfully!")
        
    except MCPConfigurationError as e:
        print(f"Configuration error: {e}")
        print("Make sure the YAML configuration files are properly formatted.")
    except Exception as e:
        logger.error(f"Demo error: {e}")
        print(f"Demo error: {e}")


async def demo_programmatic_configuration():
    """Demonstrate programmatic MCP configuration (original approach)."""
    print("\n" + "="*60)
    print("MCP Programmatic Configuration Demo")
    print("="*60)
    
    logger = get_logger("mcp.demo")
    
    try:
        # Create agent configuration
        agent_config = AgentConfig(
            name="programmatic_mcp_agent",
            description="Agent configured programmatically with MCP support",
            model="gpt-3.5-turbo",
            temperature=0.7
        )
        
        # Create MCP-enabled agent
        agent = MCPEnabledConversationalAgent(config=agent_config)
        
        # Add MCP servers programmatically
        demo_server = create_mcp_server_config(
            name="demo_server",
            url="mock://demo.example.com/mcp",
            description="Demo MCP server for testing"
        )
        
        test_server = create_mcp_server_config(
            name="test_server", 
            url="mock://test.example.com/mcp",
            description="Test MCP server with additional tools"
        )
        
        agent.add_mcp_server(demo_server)
        agent.add_mcp_server(test_server)
        
        # Initialize agent
        await agent.initialize()
        
        # Display status
        status = agent.get_mcp_status()
        print(f"Programmatic MCP Status:")
        print(f"  Total servers: {status['total_servers']}")
        print(f"  Total tools: {status['total_tools']}")
        
        # Test tools
        print(f"\nTesting programmatically configured tools:")
        if 'calculator' in agent.get_mcp_tools():
            result = await agent.call_mcp_tool('calculator', expression='10 + 5')
            print(f"Calculator: {result}")
        
        # Shutdown
        await agent.shutdown()
        print(f"Programmatic configuration demo completed!")
        
    except Exception as e:
        logger.error(f"Programmatic demo error: {e}")
        print(f"Programmatic demo error: {e}")


async def demo_mcp_with_decorator():
    """Demonstrate MCP support using the decorator pattern."""
    print("\n" + "="*60)
    print("MCP Decorator Pattern Demo")
    print("="*60)
    
    # Use decorator to add MCP support to existing agent class
    @with_mcp_support
    class DecoratedAgent(ConversationalAgent):
        async def process_message(self, message: str) -> str:
            # Custom processing with MCP tools available
            mcp_tools = self.get_mcp_tools() if hasattr(self, 'get_mcp_tools') else []
            
            if mcp_tools and any(tool in message.lower() for tool in ['calculate', 'weather', 'math']):
                enhanced_message = f"{message}\n\nNote: I have access to these tools: {', '.join(mcp_tools)}"
            else:
                enhanced_message = message
            
            return await self.process(enhanced_message)
    
    try:
        # Create agent configuration
        config = AgentConfig(
            name="decorated_agent",
            description="Agent with MCP support via decorator",
            model="gpt-3.5-turbo"
        )
        
        # Create decorated agent
        agent = DecoratedAgent(config)
        
        # Add MCP server
        server_config = create_mcp_server_config(
            name="decorator_server",
            url="mock://decorator.example.com/mcp",
            description="MCP server for decorator demo"
        )
        agent.add_mcp_server(server_config)
        
        # Initialize (MCP will be initialized automatically)
        await agent.initialize()
        
        print(f"Decorator pattern agent initialized with {len(agent.get_mcp_tools())} MCP tools")
        
        # Test interaction
        response = await agent.process_message("Can you help me calculate 7 * 8?")
        print(f"Response: {response}")
        
        # Shutdown
        await agent.shutdown()
        print(f"Decorator pattern demo completed!")
        
    except Exception as e:
        print(f"Decorator demo error: {e}")


async def demo_configuration_comparison():
    """Compare different MCP configuration approaches."""
    print("\n" + "="*60)
    print("MCP Configuration Approaches Comparison")
    print("="*60)
    
    approaches = [
        ("YAML Configuration", demo_yaml_configuration),
        ("Programmatic Configuration", demo_programmatic_configuration), 
        ("Decorator Pattern", demo_mcp_with_decorator)
    ]
    
    for approach_name, demo_func in approaches:
        print(f"\n--- {approach_name} ---")
        try:
            await demo_func()
        except Exception as e:
            print(f"Error in {approach_name}: {e}")
        
        print(f"--- End {approach_name} ---")


async def main():
    """Main demo function."""
    print("MCP (Model Context Protocol) Integration Demo")
    print("Demonstrating YAML configuration and multiple integration patterns")
    
    try:
        # Run configuration comparison demo
        await demo_configuration_comparison()
        
        print("\n" + "="*60)
        print("All MCP demos completed successfully!")
        print("="*60)
        
        print("\nKey takeaways:")
        print("1. YAML configuration provides clean, maintainable MCP setup")
        print("2. Agent configs can reference MCP configs for modularity")
        print("3. Multiple integration patterns support different use cases")
        print("4. MCP tools are automatically discovered and registered")
        print("5. Error handling ensures graceful fallbacks")
        
    except Exception as e:
        print(f"Demo failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
