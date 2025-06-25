#!/usr/bin/env python3
"""
MCP (Model Context Protocol) Support for NanoBrain Agents

This module provides MCP support as a mixin that can be added to any agent to enable
standardized tool calling and context management following the MCP specification.

MCP provides a standardized way to connect AI models to different data sources and tools,
similar to how USB-C provides a standardized connection interface.

Key Features:
- MCP server connection and management
- Tool discovery and registration from MCP servers
- OAuth authentication support for MCP servers
- Multiple server support in a single agent
- Standardized tool calling through Messages API
- Error handling and fallback mechanisms
- YAML configuration support
"""

import asyncio
import json
import time
import logging
import yaml
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from urllib.parse import urljoin, urlparse
from datetime import datetime, timezone

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

# Import NanoBrain components
from nanobrain.core.logging_system import get_logger, OperationType
from nanobrain.core.tool import ToolBase, ToolConfig
from nanobrain.core.component_base import ComponentDependencyError


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server connection."""
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

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MCPServerConfig':
        """Create MCPServerConfig from dictionary."""
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert MCPServerConfig to dictionary."""
        return {
            'name': self.name,
            'url': self.url,
            'description': self.description,
            'auth_type': self.auth_type,
            'auth_token': self.auth_token,
            'oauth_config': self.oauth_config,
            'timeout': self.timeout,
            'max_retries': self.max_retries,
            'retry_delay': self.retry_delay,
            'enabled': self.enabled,
            'capabilities': self.capabilities
        }


@dataclass
class MCPClientConfig:
    """Configuration for MCP client behavior."""
    default_timeout: float = 30.0
    default_max_retries: int = 3
    default_retry_delay: float = 1.0
    connection_pool_size: int = 10
    enable_tool_caching: bool = True
    tool_cache_ttl: int = 300  # seconds
    auto_discover_tools: bool = True
    fail_on_server_error: bool = False
    log_tool_calls: bool = True
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MCPClientConfig':
        """Create MCPClientConfig from dictionary."""
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert MCPClientConfig to dictionary."""
        return {
            'default_timeout': self.default_timeout,
            'default_max_retries': self.default_max_retries,
            'default_retry_delay': self.default_retry_delay,
            'connection_pool_size': self.connection_pool_size,
            'enable_tool_caching': self.enable_tool_caching,
            'tool_cache_ttl': self.tool_cache_ttl,
            'auto_discover_tools': self.auto_discover_tools,
            'fail_on_server_error': self.fail_on_server_error,
            'log_tool_calls': self.log_tool_calls
        }


@dataclass
class MCPToolInfo:
    """Information about a tool from an MCP server."""
    name: str
    description: str
    schema: Dict[str, Any]
    server_name: str
    server_url: str


class MCPError(Exception):
    """Base exception for MCP-related errors."""
    pass


class MCPConnectionError(MCPError):
    """Exception raised when MCP server connection fails."""
    pass


class MCPAuthenticationError(MCPError):
    """Exception raised when MCP server authentication fails."""
    pass


class MCPToolExecutionError(MCPError):
    """Exception raised when MCP tool execution fails."""
    pass


class MCPConfigurationError(MCPError):
    """Exception raised when MCP configuration is invalid."""
    pass


def load_mcp_config_from_yaml(config_path: str) -> Dict[str, Any]:
    """
    Load MCP configuration from YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing MCP configuration
        
    Raises:
        MCPConfigurationError: If configuration file is invalid
    """
    try:
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise MCPConfigurationError(f"MCP configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if not isinstance(config, dict):
            raise MCPConfigurationError("MCP configuration must be a dictionary")
        
        # Validate required sections
        if 'mcp' not in config:
            raise MCPConfigurationError("Configuration must contain 'mcp' section")
        
        mcp_config = config['mcp']
        
        # Validate servers section
        if 'servers' in mcp_config and not isinstance(mcp_config['servers'], list):
            raise MCPConfigurationError("'servers' section must be a list")
        
        # Validate client section
        if 'client' in mcp_config and not isinstance(mcp_config['client'], dict):
            raise MCPConfigurationError("'client' section must be a dictionary")
        
        return mcp_config
        
    except yaml.YAMLError as e:
        raise MCPConfigurationError(f"Invalid YAML in MCP configuration: {e}")
    except Exception as e:
        raise MCPConfigurationError(f"Error loading MCP configuration: {e}")


def resolve_config_path(config_path: str, base_path: Optional[str] = None) -> str:
    """
    Resolve configuration file path, supporting relative paths.
    
    Args:
        config_path: Path to configuration file
        base_path: Base path for relative resolution
        
    Returns:
        Resolved absolute path
    """
    config_path = Path(config_path)
    
    if config_path.is_absolute():
        return str(config_path)
    
    # Try relative to base_path first
    if base_path:
        base_path = Path(base_path)
        if base_path.is_file():
            base_path = base_path.parent
        
        resolved_path = base_path / config_path
        if resolved_path.exists():
            return str(resolved_path.resolve())
    
    # Try relative to current working directory
    if config_path.exists():
        return str(config_path.resolve())
    
    # Try relative to nanobrain config directory
    nanobrain_config_dir = Path(__file__).parent.parent.parent / "config"
    resolved_path = nanobrain_config_dir / config_path
    if resolved_path.exists():
        return str(resolved_path.resolve())
    
    # Return original path if not found (will cause error later)
    return str(config_path)


class MCPTool(ToolBase):
    """
    Tool wrapper for MCP server tools.
    Enhanced with mandatory from_config pattern implementation.
    
    This class wraps tools from MCP servers to make them compatible with
    the NanoBrain tool system.
    """
    
    @classmethod
    def from_config(cls, config: ToolConfig, **kwargs) -> 'MCPTool':
        """Mandatory from_config implementation for MCPTool"""
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
    
    @classmethod
    def resolve_dependencies(cls, component_config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Resolve MCPTool dependencies"""
        tool_info = kwargs.get('tool_info')
        mcp_client = kwargs.get('mcp_client')
        
        if not tool_info:
            raise ComponentDependencyError("MCPTool requires 'tool_info' parameter")
        if not mcp_client:
            raise ComponentDependencyError("MCPTool requires 'mcp_client' parameter")
        
        base_deps = super().resolve_dependencies(component_config, **kwargs)
        return {
            **base_deps,
            'tool_info': tool_info,
            'mcp_client': mcp_client
        }
    
    def _init_from_config(self, config: ToolConfig, component_config: Dict[str, Any],
                         dependencies: Dict[str, Any]) -> None:
        """Initialize MCPTool with resolved dependencies"""
        super()._init_from_config(config, component_config, dependencies)
        self.tool_info = dependencies['tool_info']
        self.mcp_client = dependencies['mcp_client']
        self.schema = self.tool_info.schema
    
    # MCPTool inherits FromConfigBase.__init__ which prevents direct instantiation
        
    async def execute(self, **kwargs) -> Any:
        """Execute the MCP tool on the remote server."""
        if not self.is_initialized:
            await self.initialize()
        
        try:
            # Execute tool through MCP client
            result = await self.mcp_client.call_tool(
                server_name=self.tool_info.server_name,
                tool_name=self.tool_info.name,
                parameters=kwargs
            )
            
            await self._record_call(True)
            return result
            
        except Exception as e:
            await self._record_call(False)
            raise MCPToolExecutionError(f"MCP tool {self.name} execution failed: {e}")
    
    def get_schema(self) -> Dict[str, Any]:
        """Get the tool schema for LLM function calling."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.schema.get("parameters", {})
            }
        }


class MCPClient:
    """
    Client for connecting to and interacting with MCP servers.
    
    This client handles the low-level communication with MCP servers,
    including authentication, tool discovery, and tool execution.
    """
    
    def __init__(self, config: MCPClientConfig = None, logger: Optional[logging.Logger] = None):
        self.config = config or MCPClientConfig()
        self.logger = logger or get_logger("mcp.client")
        self.servers: Dict[str, MCPServerConfig] = {}
        self.server_sessions: Dict[str, Any] = {}  # aiohttp sessions or mock sessions
        self.server_tools: Dict[str, List[MCPToolInfo]] = {}
        self.tool_cache: Dict[str, Any] = {}  # Tool result cache
        self.is_initialized = False
        
        if not AIOHTTP_AVAILABLE:
            self.logger.warning("aiohttp not available, using mock MCP client")
    
    @classmethod
    def from_yaml_config(cls, config_path: str, base_path: Optional[str] = None, logger: Optional[logging.Logger] = None) -> 'MCPClient':
        """
        Create MCPClient from YAML configuration file.
        
        Args:
            config_path: Path to YAML configuration file
            base_path: Base path for relative path resolution
            logger: Optional logger instance
            
        Returns:
            Configured MCPClient instance
        """
        resolved_path = resolve_config_path(config_path, base_path)
        mcp_config = load_mcp_config_from_yaml(resolved_path)
        
        # Create client config
        client_config_data = mcp_config.get('client', {})
        client_config = MCPClientConfig.from_dict(client_config_data)
        
        # Create client
        client = cls(config=client_config, logger=logger)
        
        # Add servers
        servers_config = mcp_config.get('servers', [])
        for server_data in servers_config:
            try:
                server_config = MCPServerConfig.from_dict(server_data)
                client.add_server(server_config)
            except Exception as e:
                if logger:
                    logger.error(f"Failed to add server {server_data.get('name', 'unknown')}: {e}")
                if client_config.fail_on_server_error:
                    raise MCPConfigurationError(f"Failed to configure server: {e}")
        
        return client
    
    async def initialize(self):
        """Initialize the MCP client."""
        if self.is_initialized:
            return
        
        self.logger.info("Initializing MCP client", 
                        servers_count=len(self.servers),
                        auto_discover=self.config.auto_discover_tools)
        self.is_initialized = True
    
    async def shutdown(self):
        """Shutdown the MCP client and close all connections."""
        self.logger.info("Shutting down MCP client")
        
        # Close all server sessions
        if AIOHTTP_AVAILABLE:
            for session in self.server_sessions.values():
                if hasattr(session, 'close'):
                    await session.close()
        
        self.server_sessions.clear()
        self.tool_cache.clear()
        self.is_initialized = False
    
    def add_server(self, config: MCPServerConfig):
        """Add an MCP server configuration."""
        self.servers[config.name] = config
        self.logger.info(f"Added MCP server: {config.name}", 
                        server_url=config.url,
                        auth_type=config.auth_type,
                        enabled=config.enabled)
    
    def remove_server(self, server_name: str):
        """Remove an MCP server configuration."""
        if server_name in self.servers:
            del self.servers[server_name]
            if server_name in self.server_sessions:
                if AIOHTTP_AVAILABLE and hasattr(self.server_sessions[server_name], 'close'):
                    asyncio.create_task(self.server_sessions[server_name].close())
                del self.server_sessions[server_name]
            if server_name in self.server_tools:
                del self.server_tools[server_name]
            
            # Clear cache for this server
            cache_keys_to_remove = [key for key in self.tool_cache.keys() if key.startswith(f"{server_name}:")]
            for key in cache_keys_to_remove:
                del self.tool_cache[key]
            
            self.logger.info(f"Removed MCP server: {server_name}")
    
    async def connect_to_server(self, server_name: str) -> bool:
        """Connect to an MCP server and establish a session."""
        if server_name not in self.servers:
            raise MCPError(f"Server {server_name} not configured")
        
        config = self.servers[server_name]
        if not config.enabled:
            self.logger.warning(f"Server {server_name} is disabled")
            return False
        
        if not AIOHTTP_AVAILABLE or config.url.startswith("mock://"):
            # Mock connection for demo purposes
            self.server_sessions[server_name] = {"mock": True, "url": config.url}
            self.logger.info(f"Mock connected to MCP server: {server_name}")
            return True
        
        try:
            # Create session with authentication
            headers = {}
            if config.auth_type == "bearer" and config.auth_token:
                headers["Authorization"] = f"Bearer {config.auth_token}"
            
            timeout = aiohttp.ClientTimeout(total=config.timeout)
            session = aiohttp.ClientSession(
                headers=headers,
                timeout=timeout
            )
            
            # Test connection
            async with session.get(urljoin(config.url, "/health")) as response:
                if response.status == 200:
                    self.server_sessions[server_name] = session
                    self.logger.info(f"Connected to MCP server: {server_name}")
                    return True
                else:
                    await session.close()
                    raise MCPConnectionError(f"Server {server_name} health check failed: {response.status}")
        
        except Exception as e:
            self.logger.error(f"Failed to connect to MCP server {server_name}: {e}")
            if self.config.fail_on_server_error:
                raise MCPConnectionError(f"Connection to {server_name} failed: {e}")
            return False
    
    async def discover_tools(self, server_name: str) -> List[MCPToolInfo]:
        """Discover available tools from an MCP server."""
        if server_name not in self.server_sessions:
            await self.connect_to_server(server_name)
        
        config = self.servers[server_name]
        session = self.server_sessions[server_name]
        
        if not AIOHTTP_AVAILABLE or session.get("mock") or config.url.startswith("mock://"):
            # Mock tool discovery for demo purposes
            mock_tools = [
                MCPToolInfo(
                    name="calculator",
                    description="Perform mathematical calculations",
                    schema={
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "expression": {
                                    "type": "string",
                                    "description": "Mathematical expression to evaluate"
                                }
                            },
                            "required": ["expression"]
                        }
                    },
                    server_name=server_name,
                    server_url=config.url
                ),
                MCPToolInfo(
                    name="weather",
                    description="Get weather information for a location",
                    schema={
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "Location to get weather for"
                                }
                            },
                            "required": ["location"]
                        }
                    },
                    server_name=server_name,
                    server_url=config.url
                )
            ]
            
            self.server_tools[server_name] = mock_tools
            self.logger.info(f"Mock discovered {len(mock_tools)} tools from server {server_name}")
            return mock_tools
        
        try:
            # Request tools from server
            tools_url = urljoin(config.url, "/tools")
            async with session.get(tools_url) as response:
                if response.status == 200:
                    tools_data = await response.json()
                    
                    tools = []
                    for tool_data in tools_data.get("tools", []):
                        tool_info = MCPToolInfo(
                            name=tool_data["name"],
                            description=tool_data.get("description", ""),
                            schema=tool_data.get("schema", {}),
                            server_name=server_name,
                            server_url=config.url
                        )
                        tools.append(tool_info)
                    
                    self.server_tools[server_name] = tools
                    self.logger.info(f"Discovered {len(tools)} tools from server {server_name}")
                    return tools
                
                else:
                    raise MCPError(f"Tool discovery failed: {response.status}")
        
        except Exception as e:
            self.logger.error(f"Tool discovery failed for server {server_name}: {e}")
            if self.config.fail_on_server_error:
                raise
            return []
    
    async def call_tool(self, server_name: str, tool_name: str, parameters: Dict[str, Any]) -> Any:
        """Call a tool on an MCP server."""
        # Check cache first
        if self.config.enable_tool_caching:
            cache_key = f"{server_name}:{tool_name}:{hash(str(sorted(parameters.items())))}"
            if cache_key in self.tool_cache:
                cache_entry = self.tool_cache[cache_key]
                if time.time() - cache_entry['timestamp'] < self.config.tool_cache_ttl:
                    self.logger.debug(f"Using cached result for {tool_name}")
                    return cache_entry['result']
        
        if server_name not in self.server_sessions:
            await self.connect_to_server(server_name)
        
        config = self.servers[server_name]
        session = self.server_sessions[server_name]
        
        if not AIOHTTP_AVAILABLE or session.get("mock") or config.url.startswith("mock://"):
            # Mock tool execution for demo purposes
            if tool_name == "calculator":
                expression = parameters.get("expression", "1+1")
                try:
                    # Simple evaluation for demo (in real implementation, use safe evaluation)
                    result = eval(expression)
                    result = f"Result: {result}"
                except:
                    result = "Error: Invalid expression"
            elif tool_name == "weather":
                location = parameters.get("location", "Unknown")
                result = f"Weather in {location}: Sunny, 22°C"
            else:
                result = f"Mock result from {tool_name} with parameters {parameters}"
            
            # Cache result
            if self.config.enable_tool_caching:
                cache_key = f"{server_name}:{tool_name}:{hash(str(sorted(parameters.items())))}"
                self.tool_cache[cache_key] = {
                    'result': result,
                    'timestamp': time.time()
                }
            
            return result
        
        try:
            # Prepare tool call request
            tool_call_data = {
                "tool": tool_name,
                "parameters": parameters
            }
            
            # Call tool on server
            tools_url = urljoin(config.url, f"/tools/{tool_name}/call")
            async with session.post(tools_url, json=tool_call_data) as response:
                if response.status == 200:
                    result_data = await response.json()
                    result = result_data.get("result")
                    
                    # Cache result
                    if self.config.enable_tool_caching:
                        cache_key = f"{server_name}:{tool_name}:{hash(str(sorted(parameters.items())))}"
                        self.tool_cache[cache_key] = {
                            'result': result,
                            'timestamp': time.time()
                        }
                    
                    if self.config.log_tool_calls:
                        self.logger.debug(f"Tool call successful: {tool_name}", 
                                        server=server_name,
                                        parameters=parameters)
                    return result
                
                else:
                    error_text = await response.text()
                    raise MCPToolExecutionError(f"Tool call failed: {response.status} - {error_text}")
        
        except Exception as e:
            if self.config.log_tool_calls:
                self.logger.error(f"Tool call failed: {tool_name}", 
                                server=server_name,
                                error=str(e))
            raise
    
    async def get_all_tools(self) -> List[MCPToolInfo]:
        """Get all tools from all connected servers."""
        all_tools = []
        
        for server_name in self.servers:
            if self.servers[server_name].enabled:
                try:
                    if server_name not in self.server_tools and self.config.auto_discover_tools:
                        await self.discover_tools(server_name)
                    
                    all_tools.extend(self.server_tools.get(server_name, []))
                
                except Exception as e:
                    self.logger.error(f"Failed to get tools from server {server_name}: {e}")
                    if self.config.fail_on_server_error:
                        raise
        
        return all_tools


class MCPSupportMixin:
    """
    Mixin class that adds MCP (Model Context Protocol) support to agents.
    
    This mixin can be added to any agent class to provide:
    - Connection to multiple MCP servers
    - Automatic tool discovery and registration
    - Standardized tool calling through MCP
    - OAuth authentication support
    - Error handling and fallback mechanisms
    - YAML configuration support
    
    Usage:
        class MyAgent(MCPSupportMixin, ConversationalAgent):
            pass
        
        agent = MyAgent(config)
        await agent.add_mcp_server(server_config)
        await agent.initialize_mcp()
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize MCP components
        self.mcp_client = None
        self.mcp_tools: Dict[str, MCPTool] = {}
        self.mcp_enabled = True
        self.mcp_config_path = None
        
        # MCP-specific logging
        self.mcp_logger = get_logger("mcp.agent")
        
        # Ensure compatibility with different agent logger attributes
        if not hasattr(self, 'logger') and hasattr(self, 'nb_logger'):
            self.logger = self.nb_logger
    
    def set_mcp_config_path(self, config_path: str):
        """Set the path to MCP configuration file."""
        self.mcp_config_path = config_path
        self.mcp_logger.info(f"Set MCP config path: {config_path}")
    
    async def load_mcp_config_from_file(self, config_path: Optional[str] = None):
        """Load MCP configuration from YAML file."""
        if config_path:
            self.mcp_config_path = config_path
        
        if not self.mcp_config_path:
            self.mcp_logger.warning("No MCP config path set, skipping file configuration")
            return
        
        try:
            # Determine base path for relative resolution
            base_path = None
            if hasattr(self, 'config') and hasattr(self.config, 'tools_config_path'):
                base_path = self.config.tools_config_path
            
            self.mcp_client = MCPClient.from_yaml_config(
                self.mcp_config_path, 
                base_path=base_path,
                logger=self.mcp_logger
            )
            
            self.mcp_logger.info(f"Loaded MCP configuration from {self.mcp_config_path}",
                               servers_count=len(self.mcp_client.servers))
            
        except Exception as e:
            self.mcp_logger.error(f"Failed to load MCP configuration: {e}")
            # Create default client if loading fails
            self.mcp_client = MCPClient(logger=self.mcp_logger)
            raise MCPConfigurationError(f"Failed to load MCP configuration: {e}")
    
    async def initialize_mcp(self):
        """Initialize MCP support for the agent."""
        self.mcp_logger.info(f"Starting MCP initialization for agent {getattr(self, 'name', 'unknown')}")
        
        if not self.mcp_enabled:
            self.mcp_logger.warning("MCP is disabled, skipping initialization")
            return
        
        try:
            # Initialize client if not already done
            if self.mcp_client is None:
                self.mcp_logger.debug("Creating new MCP client")
                if self.mcp_config_path:
                    await self.load_mcp_config_from_file()
                else:
                    self.mcp_client = MCPClient(logger=self.mcp_logger)
            
            self.mcp_logger.debug("Initializing MCP client")
            await self.mcp_client.initialize()
            
            # Discover and register tools from all servers
            self.mcp_logger.debug("Starting tool discovery and registration")
            await self._discover_and_register_mcp_tools()
            
            self.mcp_logger.info(f"MCP support initialized for agent {getattr(self, 'name', 'unknown')}", 
                               servers_count=len(self.mcp_client.servers),
                               tools_count=len(self.mcp_tools))
        
        except Exception as e:
            self.mcp_logger.error(f"Failed to initialize MCP support: {e}")
            import traceback
            self.mcp_logger.error(f"MCP initialization traceback: {traceback.format_exc()}")
            raise
    
    async def shutdown_mcp(self):
        """Shutdown MCP support and close connections."""
        if hasattr(self, 'mcp_client') and self.mcp_client:
            await self.mcp_client.shutdown()
        
        self.mcp_logger.info("MCP support shutdown complete")
    
    def add_mcp_server(self, config: MCPServerConfig):
        """Add an MCP server to the agent."""
        if self.mcp_client is None:
            self.mcp_client = MCPClient(logger=self.mcp_logger)
        
        self.mcp_client.add_server(config)
        self.mcp_logger.info(f"Added MCP server to agent: {config.name}")
    
    def remove_mcp_server(self, server_name: str):
        """Remove an MCP server from the agent."""
        if not self.mcp_client:
            return
        
        # Remove tools from this server
        tools_to_remove = [name for name, tool in self.mcp_tools.items() 
                          if tool.tool_info.server_name == server_name]
        
        for tool_name in tools_to_remove:
            self._unregister_mcp_tool(tool_name)
        
        self.mcp_client.remove_server(server_name)
        self.mcp_logger.info(f"Removed MCP server from agent: {server_name}")
    
    async def _discover_and_register_mcp_tools(self):
        """Discover tools from all MCP servers and register them with the agent."""
        if not self.mcp_client:
            return
        
        try:
            # First ensure all enabled servers are connected and have discovered tools
            for server_name, server_config in self.mcp_client.servers.items():
                if server_config.enabled:
                    # Connect to server if not already connected
                    if server_name not in self.mcp_client.server_sessions:
                        await self.mcp_client.connect_to_server(server_name)
                    
                    # Discover tools if not already discovered
                    if server_name not in self.mcp_client.server_tools:
                        await self.mcp_client.discover_tools(server_name)
            
            # Now get all tools and register them
            all_tools = await self.mcp_client.get_all_tools()
            
            self.mcp_logger.debug(f"Found {len(all_tools)} tools to register")
            
            for tool_info in all_tools:
                await self._register_mcp_tool(tool_info)
            
            self.mcp_logger.info(f"Registered {len(all_tools)} MCP tools")
        
        except Exception as e:
            self.mcp_logger.error(f"Failed to discover and register MCP tools: {e}")
            # Don't raise - allow agent to continue without MCP tools
    
    async def _register_mcp_tool(self, tool_info: MCPToolInfo):
        """Register an MCP tool with the agent."""
        print(f"DEBUG: _register_mcp_tool called for {tool_info.name}")
        try:
            print(f"DEBUG: About to create tool config for {tool_info.name}")
            self.mcp_logger.debug(f"Attempting to register MCP tool: {tool_info.name}")
            
            # Create MCP tool wrapper
            from .tool import ToolType
            tool_config = ToolConfig(
                name=tool_info.name,
                description=tool_info.description,
                tool_type=ToolType.EXTERNAL
            )
            
            print(f"DEBUG: Created tool config, about to create MCPTool for {tool_info.name}")
            mcp_tool = MCPTool.from_config(tool_config, tool_info=tool_info, mcp_client=self.mcp_client)
            print(f"DEBUG: Created MCPTool, about to initialize for {tool_info.name}")
            await mcp_tool.initialize()
            
            print(f"DEBUG: Initialized MCPTool for {tool_info.name}")
            self.mcp_logger.debug(f"Created and initialized MCP tool: {tool_info.name}")
            
            # Register with agent's tool registry
            if hasattr(self, 'register_tool'):
                print(f"DEBUG: Agent has register_tool method, registering {tool_info.name}")
                self.register_tool(mcp_tool)
                self.mcp_logger.debug(f"Registered tool with agent registry: {tool_info.name}")
            else:
                print(f"DEBUG: Agent does not have register_tool method")
                self.mcp_logger.warning(f"Agent does not have register_tool method")
            
            # Store in MCP tools registry
            print(f"DEBUG: About to store {tool_info.name} in mcp_tools dict")
            print(f"DEBUG: mcp_tools before assignment: {self.mcp_tools}")
            print(f"DEBUG: id(mcp_tools) before assignment: {id(self.mcp_tools)}")
            self.mcp_tools[tool_info.name] = mcp_tool
            print(f"DEBUG: mcp_tools after assignment: {self.mcp_tools}")
            print(f"DEBUG: id(mcp_tools) after assignment: {id(self.mcp_tools)}")
            
            print(f"DEBUG: Successfully completed registration for {tool_info.name}")
            self.mcp_logger.info(f"Successfully registered MCP tool: {tool_info.name}", 
                                server=tool_info.server_name)
        
        except Exception as e:
            print(f"DEBUG: Exception in _register_mcp_tool for {tool_info.name}: {e}")
            self.mcp_logger.error(f"Failed to register MCP tool {tool_info.name}: {e}")
            import traceback
            traceback.print_exc()
            self.mcp_logger.error(f"Registration traceback: {traceback.format_exc()}")
    
    def _unregister_mcp_tool(self, tool_name: str):
        """Unregister an MCP tool from the agent."""
        if tool_name in self.mcp_tools:
            # Remove from agent's tool registry if possible
            if hasattr(self, 'tool_registry') and hasattr(self.tool_registry, 'unregister'):
                self.tool_registry.unregister(tool_name)
            
            # Remove from MCP tools registry
            del self.mcp_tools[tool_name]
            
            self.mcp_logger.debug(f"Unregistered MCP tool: {tool_name}")
    
    async def call_mcp_tool(self, tool_name: str, **parameters) -> Any:
        """Call an MCP tool directly."""
        if tool_name not in self.mcp_tools:
            raise MCPError(f"MCP tool {tool_name} not found")
        
        tool = self.mcp_tools[tool_name]
        return await tool.execute(**parameters)
    
    def get_mcp_tools(self) -> List[str]:
        """Get list of available MCP tool names."""
        return list(self.mcp_tools.keys())
    
    def get_mcp_servers(self) -> List[str]:
        """Get list of configured MCP server names."""
        if not self.mcp_client:
            return []
        return list(self.mcp_client.servers.keys())
    
    async def refresh_mcp_tools(self, server_name: Optional[str] = None):
        """Refresh tools from MCP servers."""
        if not self.mcp_client:
            return
        
        if server_name:
            # Refresh tools from specific server
            if server_name in self.mcp_client.servers:
                # Remove existing tools from this server
                tools_to_remove = [name for name, tool in self.mcp_tools.items() 
                                 if tool.tool_info.server_name == server_name]
                
                for tool_name in tools_to_remove:
                    self._unregister_mcp_tool(tool_name)
                
                # Discover and register new tools
                tools = await self.mcp_client.discover_tools(server_name)
                for tool_info in tools:
                    await self._register_mcp_tool(tool_info)
                
                self.mcp_logger.info(f"Refreshed tools from server {server_name}")
        else:
            # Refresh tools from all servers
            await self._discover_and_register_mcp_tools()
            self.mcp_logger.info("Refreshed tools from all MCP servers")
    
    def get_mcp_status(self) -> Dict[str, Any]:
        """Get status information about MCP support."""
        if not self.mcp_client:
            return {
                "enabled": self.mcp_enabled,
                "aiohttp_available": AIOHTTP_AVAILABLE,
                "client_initialized": False,
                "config_path": self.mcp_config_path,
                "servers": {},
                "tools": {},
                "total_servers": 0,
                "total_tools": 0
            }
        
        return {
            "enabled": self.mcp_enabled,
            "aiohttp_available": AIOHTTP_AVAILABLE,
            "client_initialized": self.mcp_client.is_initialized,
            "config_path": self.mcp_config_path,
            "client_config": self.mcp_client.config.to_dict(),
            "servers": {
                name: {
                    "url": config.url,
                    "enabled": config.enabled,
                    "auth_type": config.auth_type,
                    "connected": name in self.mcp_client.server_sessions
                }
                for name, config in self.mcp_client.servers.items()
            },
            "tools": {
                name: {
                    "server": tool.tool_info.server_name,
                    "description": tool.description
                }
                for name, tool in self.mcp_tools.items()
            },
            "total_servers": len(self.mcp_client.servers),
            "total_tools": len(self.mcp_tools)
        }


# Decorator for adding MCP support to existing agent classes
def with_mcp_support(cls):
    """
    Decorator to add MCP support to an existing agent class.
    
    Usage:
        @with_mcp_support
        class MyAgent(ConversationalAgent):
            pass
    """
    
    class MCPEnabledAgent(MCPSupportMixin, cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
        
        async def initialize(self):
            # Call parent initialize
            await super().initialize()
            
            # Initialize MCP support
            await self.initialize_mcp()
        
        async def shutdown(self):
            # Shutdown MCP support
            await self.shutdown_mcp()
            
            # Call parent shutdown
            await super().shutdown()
    
    # Preserve original class metadata
    MCPEnabledAgent.__name__ = cls.__name__
    MCPEnabledAgent.__qualname__ = cls.__qualname__
    MCPEnabledAgent.__module__ = cls.__module__
    
    return MCPEnabledAgent


# Utility functions for MCP server configuration
def create_mcp_server_config(
    name: str,
    url: str,
    description: str = "",
    auth_token: Optional[str] = None,
    **kwargs
) -> MCPServerConfig:
    """Create an MCP server configuration with common defaults."""
    auth_type = "bearer" if auth_token else "none"
    
    return MCPServerConfig(
        name=name,
        url=url,
        description=description,
        auth_type=auth_type,
        auth_token=auth_token,
        **kwargs
    )


def create_oauth_mcp_server_config(
    name: str,
    url: str,
    oauth_config: Dict[str, Any],
    description: str = "",
    **kwargs
) -> MCPServerConfig:
    """Create an MCP server configuration with OAuth authentication."""
    return MCPServerConfig(
        name=name,
        url=url,
        description=description,
        auth_type="oauth",
        oauth_config=oauth_config,
        **kwargs
    )


# Example usage and testing functions
async def test_mcp_support():
    """Test function demonstrating MCP support usage."""
    from nanobrain.core.agent import ConversationalAgent, AgentConfig
    
    # Create an agent with MCP support
    @with_mcp_support
    class TestAgent(ConversationalAgent):
        pass
    
    # Configure agent
    config = AgentConfig(
        name="mcp_test_agent",
        description="Test agent with MCP support",
        model="gpt-3.5-turbo"
    )
    
    agent = TestAgent(config)
    
    # Add MCP server
    server_config = create_mcp_server_config(
        name="test_server",
        url="https://api.example.com/mcp",
        description="Test MCP server"
    )
    
    agent.add_mcp_server(server_config)
    
    # Initialize agent (this will also initialize MCP)
    await agent.initialize()
    
    # Check MCP status
    status = agent.get_mcp_status()
    print(f"MCP Status: {status}")
    
    # Use MCP tools
    mcp_tools = agent.get_mcp_tools()
    print(f"Available MCP tools: {mcp_tools}")
    
    # Shutdown
    await agent.shutdown()


if __name__ == "__main__":
    # Run test
    asyncio.run(test_mcp_support()) 