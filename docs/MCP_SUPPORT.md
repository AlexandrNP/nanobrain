# MCP (Model Context Protocol) Support for NanoBrain

This document describes the MCP (Model Context Protocol) support in NanoBrain, which enables agents to connect to standardized tool servers and expand their capabilities through external services.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Configuration](#configuration)
   - [YAML Configuration](#yaml-configuration)
   - [Programmatic Configuration](#programmatic-configuration)
4. [Integration Patterns](#integration-patterns)
5. [Usage Examples](#usage-examples)
6. [API Reference](#api-reference)
7. [Error Handling](#error-handling)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)

## Overview

MCP (Model Context Protocol) is an open protocol that standardizes how applications provide context to LLMs. Think of MCP like a USB-C port for AI applications - it provides a standardized way to connect AI models to different data sources and tools.

NanoBrain's MCP support provides:

- **Standardized Tool Access**: Connect to any MCP-compliant server
- **Multiple Server Support**: Connect to multiple MCP servers simultaneously
- **Authentication Support**: Bearer tokens and OAuth authentication
- **Automatic Tool Discovery**: Automatically discover and register tools from servers
- **YAML Configuration**: Configure servers and clients through YAML files
- **Error Handling**: Graceful fallbacks when servers are unavailable
- **Caching**: Tool result caching for improved performance
- **Integration Patterns**: Mixin and decorator patterns for easy adoption

## Architecture

The MCP support is built around several key components:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   NanoBrain     │    │   MCP Support    │    │   MCP Servers   │
│     Agent       │◄──►│     Layer        │◄──►│   (External)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌──────────────────┐
                       │  Tool Registry   │
                       │   & Execution    │
                       └──────────────────┘
```

### Core Components

- **MCPClient**: Manages connections to MCP servers
- **MCPTool**: Wraps MCP server tools for NanoBrain compatibility
- **MCPSupportMixin**: Adds MCP capabilities to existing agents
- **MCPServerConfig**: Configuration for individual MCP servers
- **MCPClientConfig**: Configuration for MCP client behavior

## Configuration

### YAML Configuration

YAML configuration provides a clean, maintainable way to configure MCP servers and client behavior.

#### MCP Configuration File (`mcp_config.yaml`)

```yaml
# MCP (Model Context Protocol) Configuration
mcp:
  # Client configuration - controls overall MCP behavior
  client:
    default_timeout: 30.0
    default_max_retries: 3
    default_retry_delay: 1.0
    connection_pool_size: 10
    enable_tool_caching: true
    tool_cache_ttl: 300  # seconds
    auto_discover_tools: true
    fail_on_server_error: false
    log_tool_calls: true

  # MCP servers configuration
  servers:
    # Example server with no authentication
    - name: "demo_server"
      url: "mock://demo.example.com/mcp"
      description: "Demo MCP server for testing"
      auth_type: "none"
      enabled: true
      timeout: 30.0
      max_retries: 3
      retry_delay: 1.0
      capabilities:
        - "tools"

    # Example server with Bearer token authentication
    - name: "authenticated_server"
      url: "https://api.example.com/mcp"
      description: "Production MCP server with authentication"
      auth_type: "bearer"
      auth_token: "${MCP_AUTH_TOKEN}"  # Environment variable
      enabled: true
      timeout: 45.0
      max_retries: 5
      retry_delay: 2.0
      capabilities:
        - "tools"
        - "resources"

    # Example server with OAuth authentication
    - name: "oauth_server"
      url: "https://oauth.example.com/mcp"
      description: "MCP server with OAuth authentication"
      auth_type: "oauth"
      oauth_config:
        client_id: "${OAUTH_CLIENT_ID}"
        client_secret: "${OAUTH_CLIENT_SECRET}"
        scope: "mcp:tools mcp:resources"
        token_url: "https://oauth.example.com/token"
      enabled: false  # Disabled by default
      timeout: 60.0
      max_retries: 3
      retry_delay: 1.5
      capabilities:
        - "tools"
        - "resources"
        - "prompts"
```

#### Agent Configuration with MCP Reference (`agent_with_mcp.yaml`)

```yaml
# Example Agent Configuration with MCP Support
agent:
  name: "mcp_enabled_agent"
  description: "Conversational agent with MCP tool support"
  model: "gpt-4"
  temperature: 0.7
  max_tokens: 2000
  
  # Agent-specific settings
  system_prompt: |
    You are a helpful AI assistant with access to external tools through MCP.
    You can use various tools to help users with calculations, weather information, and other tasks.
    Always explain what tools you're using and why.

  # Tool configuration
  tools:
    enabled: true
    max_parallel_calls: 3
    timeout: 30.0
    
  # MCP configuration reference
  mcp:
    # Path to MCP configuration file (relative to this config file)
    config_path: "mcp_config.yaml"
    
    # Override specific settings for this agent
    client_overrides:
      auto_discover_tools: true
      fail_on_server_error: false
      log_tool_calls: true
    
    # Enable/disable specific servers for this agent
    server_overrides:
      demo_server:
        enabled: true
      authenticated_server:
        enabled: false  # Disable for this agent
      local_dev:
        enabled: true   # Enable for development
```

#### Loading YAML Configuration

```python
from core.mcp_support import MCPSupportMixin, MCPClient

class MyAgent(MCPSupportMixin, ConversationalAgent):
    def __init__(self, config, mcp_config_path=None):
        super().__init__(config)
        if mcp_config_path:
            self.set_mcp_config_path(mcp_config_path)

# Create agent with YAML configuration
agent = MyAgent(agent_config, mcp_config_path="config/mcp_config.yaml")
await agent.initialize()  # This will load MCP config from YAML

# Or load MCP client directly from YAML
mcp_client = MCPClient.from_yaml_config("config/mcp_config.yaml")
```

### Programmatic Configuration

You can also configure MCP servers programmatically:

```python
from core.mcp_support import (
    MCPSupportMixin, 
    create_mcp_server_config,
    MCPClientConfig
)

class MyAgent(MCPSupportMixin, ConversationalAgent):
    pass

agent = MyAgent(config)

# Add servers programmatically
server_config = create_mcp_server_config(
    name="my_server",
    url="https://api.example.com/mcp",
    description="My MCP server",
    auth_token="your_token_here"
)

agent.add_mcp_server(server_config)
await agent.initialize()
```

## Integration Patterns

### Mixin Pattern

The mixin pattern allows you to add MCP support to existing agent classes:

```python
from core.mcp_support import MCPSupportMixin
from core.agent import ConversationalAgent

class MCPEnabledAgent(MCPSupportMixin, ConversationalAgent):
    def __init__(self, config, mcp_config_path=None):
        super().__init__(config)
        if mcp_config_path:
            self.set_mcp_config_path(mcp_config_path)
    
    async def initialize(self):
        await super().initialize()
        await self.initialize_mcp()
    
    async def shutdown(self):
        await self.shutdown_mcp()
        await super().shutdown()

# Usage
agent = MCPEnabledAgent(config, mcp_config_path="mcp_config.yaml")
await agent.initialize()
```

### Decorator Pattern

The decorator pattern provides a quick way to add MCP support:

```python
from core.mcp_support import with_mcp_support

@with_mcp_support
class MyAgent(ConversationalAgent):
    pass

# MCP support is automatically added
agent = MyAgent(config)
agent.set_mcp_config_path("mcp_config.yaml")
await agent.initialize()  # MCP is initialized automatically
```

## Usage Examples

### Basic Setup with YAML Configuration

```python
import asyncio
from core.agent import ConversationalAgent, AgentConfig
from core.mcp_support import MCPSupportMixin

class MCPAgent(MCPSupportMixin, ConversationalAgent):
    def __init__(self, config, mcp_config_path=None):
        super().__init__(config)
        if mcp_config_path:
            self.set_mcp_config_path(mcp_config_path)

async def main():
    # Create agent configuration
    config = AgentConfig(
        name="mcp_agent",
        description="Agent with MCP support",
        model="gpt-3.5-turbo"
    )
    
    # Create agent with MCP configuration
    agent = MCPAgent(config, mcp_config_path="config/mcp_config.yaml")
    
    # Initialize agent (loads MCP config automatically)
    await agent.initialize()
    
    # Check available tools
    tools = agent.get_mcp_tools()
    print(f"Available MCP tools: {tools}")
    
    # Use a tool
    if 'calculator' in tools:
        result = await agent.call_mcp_tool('calculator', expression='2 + 3 * 4')
        print(f"Calculator result: {result}")
    
    # Process messages (tools are available automatically)
    response = await agent.process_message("What's 15 * 7?")
    print(f"Agent response: {response}")
    
    # Shutdown
    await agent.shutdown()

asyncio.run(main())
```

### Multiple Servers with Different Authentication

```python
# In mcp_config.yaml
mcp:
  servers:
    - name: "public_tools"
      url: "https://public.mcp-server.com"
      auth_type: "none"
      enabled: true
    
    - name: "premium_tools"
      url: "https://premium.mcp-server.com"
      auth_type: "bearer"
      auth_token: "${PREMIUM_TOKEN}"
      enabled: true
    
    - name: "oauth_tools"
      url: "https://oauth.mcp-server.com"
      auth_type: "oauth"
      oauth_config:
        client_id: "${OAUTH_CLIENT_ID}"
        client_secret: "${OAUTH_CLIENT_SECRET}"
        token_url: "https://oauth.mcp-server.com/token"
      enabled: true

# Usage remains the same
agent = MCPAgent(config, mcp_config_path="config/mcp_config.yaml")
await agent.initialize()

# All servers' tools are available
status = agent.get_mcp_status()
print(f"Connected to {status['total_servers']} servers")
print(f"Available tools: {status['total_tools']}")
```

### Environment Variable Support

MCP configuration supports environment variables for sensitive data:

```yaml
# In mcp_config.yaml
mcp:
  servers:
    - name: "secure_server"
      url: "https://api.example.com/mcp"
      auth_type: "bearer"
      auth_token: "${MCP_AUTH_TOKEN}"  # Will be replaced with env var value
```

```bash
# Set environment variable
export MCP_AUTH_TOKEN="your_secret_token_here"
```

### Configuration Path Resolution

MCP configuration files support flexible path resolution:

```python
# Absolute path
agent.set_mcp_config_path("/absolute/path/to/mcp_config.yaml")

# Relative to agent config file
agent.set_mcp_config_path("mcp_config.yaml")

# Relative to nanobrain config directory
agent.set_mcp_config_path("config/mcp_config.yaml")

# The system will try each location in order
```

## API Reference

### MCPClientConfig

Configuration class for MCP client behavior.

```python
@dataclass
class MCPClientConfig:
    default_timeout: float = 30.0
    default_max_retries: int = 3
    default_retry_delay: float = 1.0
    connection_pool_size: int = 10
    enable_tool_caching: bool = True
    tool_cache_ttl: int = 300  # seconds
    auto_discover_tools: bool = True
    fail_on_server_error: bool = False
    log_tool_calls: bool = True
```

### MCPServerConfig

Configuration class for individual MCP servers.

```python
@dataclass
class MCPServerConfig:
    name: str
    url: str
    description: str = ""
    auth_type: str = "none"  # "none", "bearer", "oauth"
    auth_token: Optional[str] = None
    oauth_config: Optional[Dict[str, Any]] = None
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0
    enabled: bool = True
    capabilities: List[str] = field(default_factory=lambda: ["tools"])
```

### MCPSupportMixin Methods

#### Configuration Methods

- `set_mcp_config_path(config_path: str)`: Set path to MCP configuration file
- `load_mcp_config_from_file(config_path: Optional[str] = None)`: Load configuration from YAML file
- `add_mcp_server(config: MCPServerConfig)`: Add MCP server programmatically
- `remove_mcp_server(server_name: str)`: Remove MCP server

#### Lifecycle Methods

- `initialize_mcp()`: Initialize MCP support
- `shutdown_mcp()`: Shutdown MCP connections

#### Tool Methods

- `call_mcp_tool(tool_name: str, **parameters) -> Any`: Call MCP tool directly
- `get_mcp_tools() -> List[str]`: Get list of available MCP tool names
- `refresh_mcp_tools(server_name: Optional[str] = None)`: Refresh tools from servers

#### Status Methods

- `get_mcp_servers() -> List[str]`: Get list of configured server names
- `get_mcp_status() -> Dict[str, Any]`: Get comprehensive MCP status information

### Utility Functions

```python
# Configuration loading
load_mcp_config_from_yaml(config_path: str) -> Dict[str, Any]
resolve_config_path(config_path: str, base_path: Optional[str] = None) -> str

# Server configuration helpers
create_mcp_server_config(name: str, url: str, description: str = "", 
                        auth_token: Optional[str] = None, **kwargs) -> MCPServerConfig

create_oauth_mcp_server_config(name: str, url: str, oauth_config: Dict[str, Any], 
                              description: str = "", **kwargs) -> MCPServerConfig

# Decorator
@with_mcp_support
def decorator_function(cls): ...
```

## Error Handling

The MCP support includes comprehensive error handling:

### Exception Types

- `MCPError`: Base exception for MCP-related errors
- `MCPConnectionError`: Server connection failures
- `MCPAuthenticationError`: Authentication failures
- `MCPToolExecutionError`: Tool execution failures
- `MCPConfigurationError`: Configuration file errors

### Error Handling Strategies

```python
try:
    agent = MCPAgent(config, mcp_config_path="mcp_config.yaml")
    await agent.initialize()
except MCPConfigurationError as e:
    print(f"Configuration error: {e}")
    # Handle configuration issues
except MCPConnectionError as e:
    print(f"Connection error: {e}")
    # Handle server connection issues
except MCPError as e:
    print(f"MCP error: {e}")
    # Handle general MCP errors

# Tool execution error handling
try:
    result = await agent.call_mcp_tool('calculator', expression='invalid')
except MCPToolExecutionError as e:
    print(f"Tool execution failed: {e}")
    # Handle tool execution failures
```

### Graceful Degradation

The MCP support is designed to degrade gracefully:

- If `aiohttp` is not available, mock tools are used for testing
- If servers are unavailable, the agent continues without those tools
- Configuration errors don't prevent agent initialization (unless `fail_on_server_error` is True)
- Tool execution errors are caught and reported without crashing the agent

## Best Practices

### Configuration Management

1. **Use YAML Configuration**: Prefer YAML configuration over programmatic setup for maintainability
2. **Environment Variables**: Use environment variables for sensitive data like tokens
3. **Relative Paths**: Use relative paths in configuration files for portability
4. **Server Grouping**: Group related servers and tools logically

### Security

1. **Token Management**: Never hardcode authentication tokens in configuration files
2. **HTTPS Only**: Use HTTPS URLs for production MCP servers
3. **Scope Limitation**: Limit OAuth scopes to minimum required permissions
4. **Regular Rotation**: Rotate authentication tokens regularly

### Performance

1. **Enable Caching**: Use tool result caching for frequently called tools
2. **Connection Pooling**: Configure appropriate connection pool sizes
3. **Timeout Settings**: Set reasonable timeouts for different server types
4. **Selective Enabling**: Only enable servers and tools that are actually needed

### Error Handling

1. **Graceful Degradation**: Design agents to work even when MCP servers are unavailable
2. **Retry Logic**: Configure appropriate retry settings for different server reliability levels
3. **Monitoring**: Log MCP operations for debugging and monitoring
4. **Fallback Strategies**: Implement fallback strategies for critical tools

### Development Workflow

1. **Local Development**: Use mock servers for local development and testing
2. **Environment Separation**: Use different configurations for development, staging, and production
3. **Testing**: Test with both real and mock MCP servers
4. **Documentation**: Document custom MCP server integrations

## Troubleshooting

### Common Issues

#### Configuration File Not Found

```
MCPConfigurationError: MCP configuration file not found: config/mcp_config.yaml
```

**Solution**: Check that the configuration file exists and the path is correct. The system tries multiple locations:
1. Absolute path (if provided)
2. Relative to base path (if provided)
3. Relative to current working directory
4. Relative to nanobrain config directory

#### Server Connection Failures

```
MCPConnectionError: Connection to server_name failed: Connection timeout
```

**Solutions**:
- Check server URL and availability
- Verify network connectivity
- Check authentication credentials
- Increase timeout settings
- Set `fail_on_server_error: false` for non-critical servers

#### Authentication Errors

```
MCPAuthenticationError: Authentication failed for server_name
```

**Solutions**:
- Verify authentication tokens are correct and not expired
- Check OAuth configuration and credentials
- Ensure environment variables are set correctly
- Test authentication outside of NanoBrain

#### Tool Execution Failures

```
MCPToolExecutionError: MCP tool calculator execution failed: Invalid parameters
```

**Solutions**:
- Check tool parameter requirements and formats
- Verify tool is available on the server
- Test tool execution directly with the MCP server
- Check server logs for detailed error information

### Debug Mode

Enable debug logging for detailed MCP operation information:

```python
from core.logging_system import set_debug_mode
set_debug_mode(True)

# Or configure logging level
import logging
logging.getLogger("mcp").setLevel(logging.DEBUG)
```

### Status Checking

Use the status methods to diagnose issues:

```python
status = agent.get_mcp_status()
print(f"MCP Status: {status}")

# Check specific aspects
if not status['client_initialized']:
    print("MCP client not initialized")

for server_name, server_info in status['servers'].items():
    if not server_info['connected']:
        print(f"Server {server_name} not connected")
    if not server_info['enabled']:
        print(f"Server {server_name} is disabled")
```

### Testing Configuration

Test your MCP configuration before deploying:

```python
# Test configuration loading
try:
    from core.mcp_support import load_mcp_config_from_yaml
    config = load_mcp_config_from_yaml("config/mcp_config.yaml")
    print("Configuration loaded successfully")
except Exception as e:
    print(f"Configuration error: {e}")

# Test server connections
client = MCPClient.from_yaml_config("config/mcp_config.yaml")
await client.initialize()

for server_name in client.servers:
    try:
        connected = await client.connect_to_server(server_name)
        print(f"Server {server_name}: {'Connected' if connected else 'Failed'}")
    except Exception as e:
        print(f"Server {server_name}: Error - {e}")
```

This comprehensive MCP support enables NanoBrain agents to seamlessly integrate with external tool servers while maintaining flexibility, security, and reliability. 