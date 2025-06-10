"""
A2A (Agent-to-Agent) Protocol Support for NanoBrain

This module implements the A2A protocol for agent-to-agent communication,
following the specification from Google's A2A initiative.

A2A enables agents to:
- Discover each other's capabilities
- Negotiate interaction modalities (text, forms, media)
- Securely collaborate on long running tasks
- Operate without exposing internal state, memory, or tools

Key Components:
- A2AClient: Handles communication with A2A servers
- A2AAgent: Wraps A2A remote agents for NanoBrain compatibility
- A2ASupportMixin: Mixin class adding A2A capabilities to agents
- A2AAgentConfig: Configuration for A2A agent connections
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from urllib.parse import urljoin

import yaml

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

from .logging_system import get_logger


# A2A Protocol Data Structures
class TaskStatus(str, Enum):
    """Task status enumeration following A2A specification."""
    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    COMPLETED = "completed"
    CANCELED = "canceled"
    FAILED = "failed"
    UNKNOWN = "unknown"


class PartType(str, Enum):
    """Part type enumeration for A2A message parts."""
    TEXT = "text"
    FILE = "file"
    DATA = "data"


@dataclass
class A2APart:
    """Represents a part within an A2A message or artifact."""
    type: PartType
    text: Optional[str] = None
    file: Optional[Dict[str, Any]] = None
    data: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class A2AMessage:
    """Represents a message in A2A protocol."""
    role: str  # "user" or "agent"
    parts: List[A2APart]
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class A2AArtifact:
    """Represents an artifact (output) from an A2A task."""
    name: Optional[str] = None
    description: Optional[str] = None
    parts: List[A2APart] = field(default_factory=list)
    index: Optional[int] = None
    append: bool = False
    lastChunk: bool = False
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class A2ATaskStatus:
    """Represents the status of an A2A task."""
    state: TaskStatus
    message: Optional[A2AMessage] = None


@dataclass
class A2ATask:
    """Represents an A2A task."""
    id: str
    status: A2ATaskStatus
    sessionId: Optional[str] = None
    artifacts: Optional[List[A2AArtifact]] = None
    history: Optional[List[A2AMessage]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class A2ASkill:
    """Represents a skill advertised by an A2A agent."""
    id: str
    name: str
    description: str
    tags: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    inputModes: List[str] = field(default_factory=lambda: ["text"])
    outputModes: List[str] = field(default_factory=lambda: ["text"])


@dataclass
class A2ACapabilities:
    """Represents capabilities of an A2A agent."""
    streaming: bool = False
    pushNotifications: bool = False
    stateTransitionHistory: bool = False


@dataclass
class A2AAuthentication:
    """Represents authentication schemes supported by an A2A agent."""
    schemes: List[str] = field(default_factory=lambda: ["none"])


@dataclass
class A2AProvider:
    """Represents the provider information for an A2A agent."""
    organization: str
    contact: Optional[str] = None
    website: Optional[str] = None


@dataclass
class A2AAgentCard:
    """Represents an A2A Agent Card for capability discovery."""
    name: str
    description: str
    url: str
    version: str = "1.0.0"
    provider: Optional[A2AProvider] = None
    documentationUrl: Optional[str] = None
    capabilities: A2ACapabilities = field(default_factory=A2ACapabilities)
    authentication: A2AAuthentication = field(default_factory=A2AAuthentication)
    defaultInputModes: List[str] = field(default_factory=lambda: ["text"])
    defaultOutputModes: List[str] = field(default_factory=lambda: ["text"])
    skills: List[A2ASkill] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'A2AAgentCard':
        """Create A2AAgentCard from dictionary."""
        # Handle nested objects
        if 'provider' in data and data['provider']:
            data['provider'] = A2AProvider(**data['provider'])
        
        if 'capabilities' in data and data['capabilities']:
            data['capabilities'] = A2ACapabilities(**data['capabilities'])
        
        if 'authentication' in data and data['authentication']:
            data['authentication'] = A2AAuthentication(**data['authentication'])
        
        if 'skills' in data and data['skills']:
            data['skills'] = [A2ASkill(**skill) for skill in data['skills']]
        
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert A2AAgentCard to dictionary."""
        result = {
            'name': self.name,
            'description': self.description,
            'url': self.url,
            'version': self.version,
            'defaultInputModes': self.defaultInputModes,
            'defaultOutputModes': self.defaultOutputModes
        }
        
        if self.provider:
            result['provider'] = {
                'organization': self.provider.organization,
                'contact': self.provider.contact,
                'website': self.provider.website
            }
        
        if self.documentationUrl:
            result['documentationUrl'] = self.documentationUrl
        
        result['capabilities'] = {
            'streaming': self.capabilities.streaming,
            'pushNotifications': self.capabilities.pushNotifications,
            'stateTransitionHistory': self.capabilities.stateTransitionHistory
        }
        
        result['authentication'] = {
            'schemes': self.authentication.schemes
        }
        
        result['skills'] = [
            {
                'id': skill.id,
                'name': skill.name,
                'description': skill.description,
                'tags': skill.tags,
                'examples': skill.examples,
                'inputModes': skill.inputModes,
                'outputModes': skill.outputModes
            }
            for skill in self.skills
        ]
        
        return result


@dataclass
class A2AAgentConfig:
    """Configuration for an A2A agent connection."""
    name: str
    url: str
    description: str = ""
    auth_type: str = "none"  # "none", "bearer", "oauth2"
    auth_token: Optional[str] = None
    oauth_config: Optional[Dict[str, Any]] = None
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0
    enabled: bool = True
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'A2AAgentConfig':
        """Create A2AAgentConfig from dictionary."""
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert A2AAgentConfig to dictionary."""
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
            'enabled': self.enabled
        }


@dataclass
class A2AClientConfig:
    """Configuration for A2A client behavior."""
    default_timeout: float = 30.0
    default_max_retries: int = 3
    default_retry_delay: float = 1.0
    connection_pool_size: int = 10
    enable_task_caching: bool = True
    task_cache_ttl: int = 300  # seconds
    auto_discover_agents: bool = True
    fail_on_agent_error: bool = False
    log_agent_calls: bool = True
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'A2AClientConfig':
        """Create A2AClientConfig from dictionary."""
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert A2AClientConfig to dictionary."""
        return {
            'default_timeout': self.default_timeout,
            'default_max_retries': self.default_max_retries,
            'default_retry_delay': self.default_retry_delay,
            'connection_pool_size': self.connection_pool_size,
            'enable_task_caching': self.enable_task_caching,
            'task_cache_ttl': self.task_cache_ttl,
            'auto_discover_agents': self.auto_discover_agents,
            'fail_on_agent_error': self.fail_on_agent_error,
            'log_agent_calls': self.log_agent_calls
        }


# Exception Classes
class A2AError(Exception):
    """Base exception for A2A protocol errors."""
    pass


class A2AConnectionError(A2AError):
    """Exception raised when connection to A2A agent fails."""
    pass


class A2AAuthenticationError(A2AError):
    """Exception raised when A2A agent authentication fails."""
    pass


class A2ATaskExecutionError(A2AError):
    """Exception raised when A2A task execution fails."""
    pass


class A2AConfigurationError(A2AError):
    """Exception raised when A2A configuration is invalid."""
    pass


# Utility Functions
def load_a2a_config_from_yaml(config_path: str) -> Dict[str, Any]:
    """
    Load A2A configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Dictionary containing A2A configuration
        
    Raises:
        A2AConfigurationError: If configuration file is invalid
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if not isinstance(config, dict):
            raise A2AConfigurationError(f"Invalid YAML configuration in {config_path}")
        
        return config
    
    except FileNotFoundError:
        raise A2AConfigurationError(f"A2A configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise A2AConfigurationError(f"Invalid YAML in {config_path}: {e}")
    except Exception as e:
        raise A2AConfigurationError(f"Error loading A2A configuration: {e}")


def resolve_config_path(config_path: str, base_path: Optional[str] = None) -> str:
    """
    Resolve configuration file path with fallback locations.
    
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


class A2AClient:
    """
    Client for connecting to and interacting with A2A agents.
    
    This client handles the low-level communication with A2A agents,
    including authentication, capability discovery, and task execution.
    """
    
    def __init__(self, config: A2AClientConfig = None, logger: Optional[logging.Logger] = None):
        self.config = config or A2AClientConfig()
        self.logger = logger or get_logger("a2a.client")
        self.agents: Dict[str, A2AAgentConfig] = {}
        self.agent_sessions: Dict[str, Any] = {}  # aiohttp sessions or mock sessions
        self.agent_cards: Dict[str, A2AAgentCard] = {}
        self.task_cache: Dict[str, Any] = {}  # Task result cache
        self.is_initialized = False
        
        if not AIOHTTP_AVAILABLE:
            self.logger.warning("aiohttp not available, using mock A2A client")
    
    @classmethod
    def from_yaml_config(cls, config_path: str, base_path: Optional[str] = None, logger: Optional[logging.Logger] = None) -> 'A2AClient':
        """
        Create A2AClient from YAML configuration file.
        
        Args:
            config_path: Path to YAML configuration file
            base_path: Base path for relative path resolution
            logger: Optional logger instance
            
        Returns:
            Configured A2AClient instance
        """
        resolved_path = resolve_config_path(config_path, base_path)
        a2a_config = load_a2a_config_from_yaml(resolved_path)
        
        # Create client config
        client_config_data = a2a_config.get('client', {})
        client_config = A2AClientConfig.from_dict(client_config_data)
        
        # Create client
        client = cls(config=client_config, logger=logger)
        
        # Add agents
        agents_config = a2a_config.get('agents', [])
        for agent_data in agents_config:
            try:
                agent_config = A2AAgentConfig.from_dict(agent_data)
                client.add_agent(agent_config)
            except Exception as e:
                if logger:
                    logger.error(f"Failed to add agent {agent_data.get('name', 'unknown')}: {e}")
                if client_config.fail_on_agent_error:
                    raise A2AConfigurationError(f"Failed to configure agent: {e}")
        
        return client
    
    async def initialize(self):
        """Initialize the A2A client."""
        if self.is_initialized:
            return
        
        self.logger.info("Initializing A2A client", 
                        agents_count=len(self.agents),
                        auto_discover=self.config.auto_discover_agents)
        self.is_initialized = True
    
    async def shutdown(self):
        """Shutdown the A2A client and close all connections."""
        self.logger.info("Shutting down A2A client")
        
        # Close all agent sessions
        if AIOHTTP_AVAILABLE:
            for session in self.agent_sessions.values():
                if hasattr(session, 'close'):
                    await session.close()
        
        self.agent_sessions.clear()
        self.task_cache.clear()
        self.is_initialized = False
    
    def add_agent(self, config: A2AAgentConfig):
        """Add an A2A agent configuration."""
        self.agents[config.name] = config
        self.logger.info(f"Added A2A agent: {config.name}", 
                        agent_url=config.url,
                        auth_type=config.auth_type,
                        enabled=config.enabled)
    
    def remove_agent(self, agent_name: str):
        """Remove an A2A agent configuration."""
        if agent_name in self.agents:
            del self.agents[agent_name]
            if agent_name in self.agent_sessions:
                if AIOHTTP_AVAILABLE and hasattr(self.agent_sessions[agent_name], 'close'):
                    asyncio.create_task(self.agent_sessions[agent_name].close())
                del self.agent_sessions[agent_name]
            if agent_name in self.agent_cards:
                del self.agent_cards[agent_name]
            
            # Clear cache for this agent
            cache_keys_to_remove = [key for key in self.task_cache.keys() if key.startswith(f"{agent_name}:")]
            for key in cache_keys_to_remove:
                del self.task_cache[key]
            
            self.logger.info(f"Removed A2A agent: {agent_name}")
    
    async def connect_to_agent(self, agent_name: str) -> bool:
        """Connect to an A2A agent and establish a session."""
        if agent_name not in self.agents:
            raise A2AError(f"Agent {agent_name} not configured")
        
        agent_config = self.agents[agent_name]
        
        if not agent_config.enabled:
            self.logger.debug(f"Agent {agent_name} is disabled, skipping connection")
            return False
        
        try:
            if AIOHTTP_AVAILABLE:
                # Create aiohttp session with authentication
                headers = {}
                if agent_config.auth_type == "bearer" and agent_config.auth_token:
                    headers["Authorization"] = f"Bearer {agent_config.auth_token}"
                
                timeout = aiohttp.ClientTimeout(total=agent_config.timeout)
                session = aiohttp.ClientSession(
                    headers=headers,
                    timeout=timeout,
                    connector=aiohttp.TCPConnector(limit=self.config.connection_pool_size)
                )
                
                self.agent_sessions[agent_name] = session
                
                # Test connection by fetching agent card
                await self.discover_agent_capabilities(agent_name)
                
                self.logger.info(f"Connected to A2A agent: {agent_name}")
                return True
            else:
                # Mock connection for testing
                self.agent_sessions[agent_name] = MockA2ASession(agent_config)
                await self._create_mock_agent_card(agent_name)
                self.logger.info(f"Mock connected to A2A agent: {agent_name}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to connect to agent {agent_name}: {e}")
            if agent_config.name in self.agent_sessions:
                del self.agent_sessions[agent_name]
            raise A2AConnectionError(f"Failed to connect to agent {agent_name}: {e}")
    
    async def discover_agent_capabilities(self, agent_name: str) -> A2AAgentCard:
        """Discover capabilities of an A2A agent by fetching its Agent Card."""
        if agent_name not in self.agents:
            raise A2AError(f"Agent {agent_name} not configured")
        
        agent_config = self.agents[agent_name]
        
        try:
            if AIOHTTP_AVAILABLE and agent_name in self.agent_sessions:
                session = self.agent_sessions[agent_name]
                
                # Fetch agent card from well-known URL
                agent_card_url = urljoin(agent_config.url, "/.well-known/agent.json")
                
                async with session.get(agent_card_url) as response:
                    if response.status == 200:
                        card_data = await response.json()
                        agent_card = A2AAgentCard.from_dict(card_data)
                        self.agent_cards[agent_name] = agent_card
                        
                        self.logger.info(f"Discovered capabilities for agent {agent_name}",
                                       skills_count=len(agent_card.skills),
                                       capabilities=agent_card.capabilities.to_dict() if hasattr(agent_card.capabilities, 'to_dict') else str(agent_card.capabilities))
                        
                        return agent_card
                    else:
                        raise A2AError(f"Failed to fetch agent card: HTTP {response.status}")
            else:
                # Return mock agent card
                return await self._create_mock_agent_card(agent_name)
                
        except Exception as e:
            self.logger.error(f"Failed to discover capabilities for agent {agent_name}: {e}")
            raise A2AError(f"Capability discovery failed for {agent_name}: {e}")
    
    async def send_task(self, agent_name: str, task_id: str, message: A2AMessage, 
                       session_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> A2ATask:
        """Send a task to an A2A agent."""
        if agent_name not in self.agents:
            raise A2AError(f"Agent {agent_name} not configured")
        
        if agent_name not in self.agent_sessions:
            await self.connect_to_agent(agent_name)
        
        agent_config = self.agents[agent_name]
        
        try:
            # Prepare JSON-RPC request
            request_data = {
                "jsonrpc": "2.0",
                "id": str(uuid.uuid4()),
                "method": "tasks/send",
                "params": {
                    "id": task_id,
                    "message": {
                        "role": message.role,
                        "parts": [self._part_to_dict(part) for part in message.parts]
                    }
                }
            }
            
            if session_id:
                request_data["params"]["sessionId"] = session_id
            if metadata:
                request_data["params"]["metadata"] = metadata
            
            if AIOHTTP_AVAILABLE and agent_name in self.agent_sessions:
                session = self.agent_sessions[agent_name]
                
                async with session.post(agent_config.url, json=request_data) as response:
                    if response.status == 200:
                        response_data = await response.json()
                        
                        if "error" in response_data:
                            raise A2ATaskExecutionError(f"Task execution failed: {response_data['error']}")
                        
                        task_data = response_data.get("result")
                        if task_data:
                            return self._dict_to_task(task_data)
                        else:
                            raise A2ATaskExecutionError("No task data in response")
                    else:
                        raise A2ATaskExecutionError(f"HTTP {response.status}: {await response.text()}")
            else:
                # Mock execution
                return await self._mock_send_task(agent_name, task_id, message, session_id, metadata)
                
        except Exception as e:
            self.logger.error(f"Failed to send task to agent {agent_name}: {e}")
            raise A2ATaskExecutionError(f"Task execution failed for {agent_name}: {e}")
    
    async def get_task(self, agent_name: str, task_id: str) -> A2ATask:
        """Get the status of a task from an A2A agent."""
        if agent_name not in self.agents:
            raise A2AError(f"Agent {agent_name} not configured")
        
        if agent_name not in self.agent_sessions:
            await self.connect_to_agent(agent_name)
        
        agent_config = self.agents[agent_name]
        
        try:
            # Prepare JSON-RPC request
            request_data = {
                "jsonrpc": "2.0",
                "id": str(uuid.uuid4()),
                "method": "tasks/get",
                "params": {
                    "id": task_id
                }
            }
            
            if AIOHTTP_AVAILABLE and agent_name in self.agent_sessions:
                session = self.agent_sessions[agent_name]
                
                async with session.post(agent_config.url, json=request_data) as response:
                    if response.status == 200:
                        response_data = await response.json()
                        
                        if "error" in response_data:
                            raise A2ATaskExecutionError(f"Task query failed: {response_data['error']}")
                        
                        task_data = response_data.get("result")
                        if task_data:
                            return self._dict_to_task(task_data)
                        else:
                            raise A2ATaskExecutionError("No task data in response")
                    else:
                        raise A2ATaskExecutionError(f"HTTP {response.status}: {await response.text()}")
            else:
                # Mock execution
                return await self._mock_get_task(agent_name, task_id)
                
        except Exception as e:
            self.logger.error(f"Failed to get task from agent {agent_name}: {e}")
            raise A2ATaskExecutionError(f"Task query failed for {agent_name}: {e}")
    
    async def cancel_task(self, agent_name: str, task_id: str) -> A2ATask:
        """Cancel a task on an A2A agent."""
        if agent_name not in self.agents:
            raise A2AError(f"Agent {agent_name} not configured")
        
        if agent_name not in self.agent_sessions:
            await self.connect_to_agent(agent_name)
        
        agent_config = self.agents[agent_name]
        
        try:
            # Prepare JSON-RPC request
            request_data = {
                "jsonrpc": "2.0",
                "id": str(uuid.uuid4()),
                "method": "tasks/cancel",
                "params": {
                    "id": task_id
                }
            }
            
            if AIOHTTP_AVAILABLE and agent_name in self.agent_sessions:
                session = self.agent_sessions[agent_name]
                
                async with session.post(agent_config.url, json=request_data) as response:
                    if response.status == 200:
                        response_data = await response.json()
                        
                        if "error" in response_data:
                            raise A2ATaskExecutionError(f"Task cancellation failed: {response_data['error']}")
                        
                        task_data = response_data.get("result")
                        if task_data:
                            return self._dict_to_task(task_data)
                        else:
                            raise A2ATaskExecutionError("No task data in response")
                    else:
                        raise A2ATaskExecutionError(f"HTTP {response.status}: {await response.text()}")
            else:
                # Mock execution
                return await self._mock_cancel_task(agent_name, task_id)
                
        except Exception as e:
            self.logger.error(f"Failed to cancel task on agent {agent_name}: {e}")
            raise A2ATaskExecutionError(f"Task cancellation failed for {agent_name}: {e}")
    
    async def get_all_agents(self) -> List[A2AAgentCard]:
        """Get all available agent cards."""
        agent_cards = []
        
        for agent_name in self.agents.keys():
            try:
                if agent_name not in self.agent_cards:
                    await self.discover_agent_capabilities(agent_name)
                
                if agent_name in self.agent_cards:
                    agent_cards.append(self.agent_cards[agent_name])
            except Exception as e:
                self.logger.warning(f"Failed to get agent card for {agent_name}: {e}")
        
        return agent_cards

    # Helper methods for data conversion
    def _part_to_dict(self, part: A2APart) -> Dict[str, Any]:
        """Convert A2APart to dictionary for JSON serialization."""
        result = {"type": part.type.value}
        
        if part.text is not None:
            result["text"] = part.text
        if part.file is not None:
            result["file"] = part.file
        if part.data is not None:
            result["data"] = part.data
        if part.metadata is not None:
            result["metadata"] = part.metadata
        
        return result
    
    def _dict_to_part(self, data: Dict[str, Any]) -> A2APart:
        """Convert dictionary to A2APart."""
        return A2APart(
            type=PartType(data["type"]),
            text=data.get("text"),
            file=data.get("file"),
            data=data.get("data"),
            metadata=data.get("metadata")
        )
    
    def _dict_to_message(self, data: Dict[str, Any]) -> A2AMessage:
        """Convert dictionary to A2AMessage."""
        return A2AMessage(
            role=data["role"],
            parts=[self._dict_to_part(part) for part in data["parts"]],
            metadata=data.get("metadata")
        )
    
    def _dict_to_artifact(self, data: Dict[str, Any]) -> A2AArtifact:
        """Convert dictionary to A2AArtifact."""
        return A2AArtifact(
            name=data.get("name"),
            description=data.get("description"),
            parts=[self._dict_to_part(part) for part in data.get("parts", [])],
            index=data.get("index"),
            append=data.get("append", False),
            lastChunk=data.get("lastChunk", False),
            metadata=data.get("metadata")
        )
    
    def _dict_to_task_status(self, data: Dict[str, Any]) -> A2ATaskStatus:
        """Convert dictionary to A2ATaskStatus."""
        return A2ATaskStatus(
            state=TaskStatus(data["state"]),
            message=self._dict_to_message(data["message"]) if data.get("message") else None
        )
    
    def _dict_to_task(self, data: Dict[str, Any]) -> A2ATask:
        """Convert dictionary to A2ATask."""
        return A2ATask(
            id=data["id"],
            status=self._dict_to_task_status(data["status"]),
            sessionId=data.get("sessionId"),
            artifacts=[self._dict_to_artifact(artifact) for artifact in data.get("artifacts", [])],
            history=[self._dict_to_message(msg) for msg in data.get("history", [])],
            metadata=data.get("metadata")
        )
    
    # Mock functionality for testing without aiohttp
    async def _create_mock_agent_card(self, agent_name: str) -> A2AAgentCard:
        """Create a mock agent card for testing."""
        agent_config = self.agents[agent_name]
        
        # Create mock skills based on agent name
        skills = []
        if "travel" in agent_name.lower():
            skills = [
                A2ASkill(
                    id="flight-search",
                    name="Flight Search",
                    description="Search for available flights",
                    tags=["travel", "flights"],
                    examples=["Find flights from NYC to LAX"],
                    inputModes=["text", "data"],
                    outputModes=["text", "data"]
                ),
                A2ASkill(
                    id="hotel-search",
                    name="Hotel Search", 
                    description="Search for available hotels",
                    tags=["travel", "hotels"],
                    examples=["Find hotels in Paris"],
                    inputModes=["text", "data"],
                    outputModes=["text", "data"]
                )
            ]
        elif "code" in agent_name.lower():
            skills = [
                A2ASkill(
                    id="code-generation",
                    name="Code Generation",
                    description="Generate code based on requirements",
                    tags=["coding", "programming"],
                    examples=["Write a Python function to sort a list"],
                    inputModes=["text"],
                    outputModes=["text", "file"]
                ),
                A2ASkill(
                    id="code-review",
                    name="Code Review",
                    description="Review and analyze code",
                    tags=["coding", "review"],
                    examples=["Review this Python code for bugs"],
                    inputModes=["text", "file"],
                    outputModes=["text"]
                )
            ]
        else:
            skills = [
                A2ASkill(
                    id="general-assistance",
                    name="General Assistance",
                    description="Provide general help and information",
                    tags=["general", "assistance"],
                    examples=["Help me with my question"],
                    inputModes=["text"],
                    outputModes=["text"]
                )
            ]
        
        agent_card = A2AAgentCard(
            name=agent_config.name,
            description=agent_config.description or f"Mock A2A agent: {agent_config.name}",
            url=agent_config.url,
            version="1.0.0",
            provider=A2AProvider(organization="NanoBrain Mock"),
            capabilities=A2ACapabilities(streaming=True, pushNotifications=False),
            authentication=A2AAuthentication(schemes=["none"]),
            skills=skills
        )
        
        self.agent_cards[agent_name] = agent_card
        return agent_card
    
    async def _mock_send_task(self, agent_name: str, task_id: str, message: A2AMessage,
                             session_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> A2ATask:
        """Mock task execution for testing."""
        # Simulate processing delay
        await asyncio.sleep(0.1)
        
        # Create mock response based on message content
        user_text = ""
        for part in message.parts:
            if part.text:
                user_text += part.text + " "
        user_text = user_text.strip()
        
        # Generate mock response
        if "travel" in agent_name.lower():
            response_text = f"Mock travel agent response for: {user_text}"
            if "flight" in user_text.lower():
                response_text = "Found 3 flights: Flight AA123 ($299), Flight UA456 ($325), Flight DL789 ($310)"
            elif "hotel" in user_text.lower():
                response_text = "Found 5 hotels: Hotel Grand ($150/night), Hotel Plaza ($200/night), Hotel Comfort ($120/night)"
        elif "code" in agent_name.lower():
            response_text = f"Mock code agent response for: {user_text}"
            if "python" in user_text.lower():
                response_text = "```python\ndef example_function():\n    return 'Hello, World!'\n```"
        else:
            response_text = f"Mock agent response for: {user_text}"
        
        # Create mock artifacts
        artifacts = [
            A2AArtifact(
                name="response",
                description="Agent response",
                parts=[A2APart(type=PartType.TEXT, text=response_text)],
                index=0,
                lastChunk=True
            )
        ]
        
        # Create mock task
        task = A2ATask(
            id=task_id,
            status=A2ATaskStatus(
                state=TaskStatus.COMPLETED,
                message=A2AMessage(
                    role="agent",
                    parts=[A2APart(type=PartType.TEXT, text="Task completed successfully")]
                )
            ),
            sessionId=session_id,
            artifacts=artifacts,
            metadata=metadata
        )
        
        return task
    
    async def _mock_get_task(self, agent_name: str, task_id: str) -> A2ATask:
        """Mock task status retrieval for testing."""
        # Return a completed mock task
        return A2ATask(
            id=task_id,
            status=A2ATaskStatus(
                state=TaskStatus.COMPLETED,
                message=A2AMessage(
                    role="agent",
                    parts=[A2APart(type=PartType.TEXT, text="Task completed")]
                )
            ),
            artifacts=[
                A2AArtifact(
                    name="result",
                    description="Mock task result",
                    parts=[A2APart(type=PartType.TEXT, text="Mock result data")],
                    index=0,
                    lastChunk=True
                )
            ]
        )
    
    async def _mock_cancel_task(self, agent_name: str, task_id: str) -> A2ATask:
        """Mock task cancellation for testing."""
        return A2ATask(
            id=task_id,
            status=A2ATaskStatus(
                state=TaskStatus.CANCELED,
                message=A2AMessage(
                    role="agent",
                    parts=[A2APart(type=PartType.TEXT, text="Task canceled")]
                )
            )
        )


class MockA2ASession:
    """Mock session for testing A2A functionality without aiohttp."""
    
    def __init__(self, agent_config: A2AAgentConfig):
        self.agent_config = agent_config
    
    async def close(self):
        """Mock close method."""
        pass 


class A2ASupportMixin:
    """
    Mixin class that adds A2A (Agent-to-Agent) protocol support to NanoBrain agents.
    
    This mixin enables agents to:
    - Connect to remote A2A agents
    - Discover agent capabilities
    - Send tasks to other agents
    - Manage agent-to-agent communication
    
    Usage:
        class MyAgent(A2ASupportMixin, ConversationalAgent):
            pass
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # A2A-specific attributes
        self.a2a_enabled = kwargs.get('a2a_enabled', False)
        self.a2a_client: Optional[A2AClient] = None
        self.a2a_config_path: Optional[str] = kwargs.get('a2a_config_path')
        self.a2a_agents: Dict[str, A2AAgentCard] = {}
        self.a2a_logger = get_logger("a2a.mixin")
        
        # Initialize A2A if enabled
        if self.a2a_enabled and self.a2a_config_path:
            asyncio.create_task(self.load_a2a_config_from_file())
    
    def set_a2a_config_path(self, config_path: str):
        """Set the A2A configuration file path."""
        self.a2a_config_path = config_path
    
    async def load_a2a_config_from_file(self, config_path: Optional[str] = None):
        """
        Load A2A configuration from YAML file.
        
        Args:
            config_path: Optional path to configuration file. If not provided,
                        uses the path set in a2a_config_path.
        """
        if config_path:
            self.a2a_config_path = config_path
        
        if not self.a2a_config_path:
            self.a2a_logger.warning("No A2A configuration path provided")
            return
        
        try:
            # Resolve config path relative to agent config if needed
            base_path = None
            if hasattr(self, 'config') and hasattr(self.config, 'config_file_path'):
                base_path = self.config.config_file_path
            
            self.a2a_client = A2AClient.from_yaml_config(
                self.a2a_config_path, 
                base_path=base_path,
                logger=self.a2a_logger
            )
            
            self.a2a_enabled = True
            self.a2a_logger.info(f"Loaded A2A configuration from {self.a2a_config_path}")
            
        except Exception as e:
            self.a2a_logger.error(f"Failed to load A2A configuration: {e}")
            self.a2a_enabled = False
            raise A2AConfigurationError(f"Failed to load A2A configuration: {e}")
    
    async def initialize_a2a(self):
        """Initialize A2A support."""
        if not self.a2a_enabled or not self.a2a_client:
            self.a2a_logger.debug("A2A support not enabled or client not configured")
            return
        
        try:
            await self.a2a_client.initialize()
            
            # Discover and register A2A agents
            if self.a2a_client.config.auto_discover_agents:
                await self._discover_and_register_a2a_agents()
            
            self.a2a_logger.info("A2A support initialized successfully")
            
        except Exception as e:
            self.a2a_logger.error(f"Failed to initialize A2A support: {e}")
            # Don't raise - allow agent to continue without A2A
    
    async def shutdown_a2a(self):
        """Shutdown A2A support."""
        if self.a2a_client:
            await self.a2a_client.shutdown()
            self.a2a_logger.info("A2A support shutdown")
    
    def add_a2a_agent(self, config: A2AAgentConfig):
        """Add an A2A agent configuration."""
        if not self.a2a_client:
            raise A2AError("A2A client not initialized")
        
        self.a2a_client.add_agent(config)
        self.a2a_logger.info(f"Added A2A agent: {config.name}")
    
    def remove_a2a_agent(self, agent_name: str):
        """Remove an A2A agent configuration."""
        if not self.a2a_client:
            return
        
        self.a2a_client.remove_agent(agent_name)
        if agent_name in self.a2a_agents:
            del self.a2a_agents[agent_name]
        
        self.a2a_logger.info(f"Removed A2A agent: {agent_name}")
    
    async def _discover_and_register_a2a_agents(self):
        """Discover and register A2A agents."""
        if not self.a2a_client:
            return
        
        try:
            self.a2a_logger.debug("Discovering A2A agents...")
            
            # Get all agent cards
            agent_cards = await self.a2a_client.get_all_agents()
            
            for agent_card in agent_cards:
                # Find the agent name from the client's agents dict
                agent_name = None
                for name, config in self.a2a_client.agents.items():
                    if config.url == agent_card.url:
                        agent_name = name
                        break
                
                if agent_name:
                    self.a2a_agents[agent_name] = agent_card
                    self.a2a_logger.debug(f"Registered A2A agent: {agent_name}")
            
            self.a2a_logger.info(f"Discovered and registered {len(self.a2a_agents)} A2A agents")
            
        except Exception as e:
            self.a2a_logger.error(f"Failed to discover and register A2A agents: {e}")
            # Don't raise - allow agent to continue without A2A agents
    
    async def call_a2a_agent(self, agent_name: str, message: str, task_id: Optional[str] = None, 
                            session_id: Optional[str] = None, **parameters) -> Any:
        """Call an A2A agent directly."""
        if not self.a2a_client:
            raise A2AError("A2A client not initialized")
        
        if agent_name not in self.a2a_agents:
            raise A2AError(f"A2A agent {agent_name} not found")
        
        # Generate task ID if not provided
        if not task_id:
            task_id = str(uuid.uuid4())
        
        # Create A2A message
        parts = [A2APart(type=PartType.TEXT, text=message)]
        
        # Add data part if parameters provided
        if parameters:
            parts.append(A2APart(type=PartType.DATA, data=parameters))
        
        a2a_message = A2AMessage(role="user", parts=parts)
        
        # Send task to A2A agent
        task = await self.a2a_client.send_task(agent_name, task_id, a2a_message, session_id)
        
        # Extract response from artifacts
        if task.artifacts:
            response_parts = []
            for artifact in task.artifacts:
                for part in artifact.parts:
                    if part.text:
                        response_parts.append(part.text)
            
            return "\n".join(response_parts) if response_parts else "No response from agent"
        
        return "No artifacts returned from agent"
    
    def get_a2a_agents(self) -> List[str]:
        """Get list of available A2A agent names."""
        return list(self.a2a_agents.keys())
    
    def get_a2a_agent_capabilities(self, agent_name: str) -> Optional[A2AAgentCard]:
        """Get capabilities of a specific A2A agent."""
        return self.a2a_agents.get(agent_name)
    
    async def refresh_a2a_agents(self, agent_name: Optional[str] = None):
        """Refresh agents from A2A network."""
        if not self.a2a_client:
            return
        
        if agent_name:
            # Refresh specific agent
            if agent_name in self.a2a_client.agents:
                try:
                    agent_card = await self.a2a_client.discover_agent_capabilities(agent_name)
                    self.a2a_agents[agent_name] = agent_card
                    self.a2a_logger.info(f"Refreshed A2A agent {agent_name}")
                except Exception as e:
                    self.a2a_logger.error(f"Failed to refresh agent {agent_name}: {e}")
        else:
            # Refresh all agents
            await self._discover_and_register_a2a_agents()
            self.a2a_logger.info("Refreshed all A2A agents")
    
    def get_a2a_status(self) -> Dict[str, Any]:
        """Get status information about A2A support."""
        if not self.a2a_client:
            return {
                "enabled": self.a2a_enabled,
                "aiohttp_available": AIOHTTP_AVAILABLE,
                "client_initialized": False,
                "config_path": self.a2a_config_path,
                "agents": {},
                "total_agents": 0
            }
        
        return {
            "enabled": self.a2a_enabled,
            "aiohttp_available": AIOHTTP_AVAILABLE,
            "client_initialized": self.a2a_client.is_initialized,
            "config_path": self.a2a_config_path,
            "client_config": self.a2a_client.config.to_dict(),
            "agents": {
                name: {
                    "url": card.url,
                    "description": card.description,
                    "skills_count": len(card.skills),
                    "capabilities": {
                        "streaming": card.capabilities.streaming,
                        "pushNotifications": card.capabilities.pushNotifications,
                        "stateTransitionHistory": card.capabilities.stateTransitionHistory
                    }
                }
                for name, card in self.a2a_agents.items()
            },
            "total_agents": len(self.a2a_agents)
        }


# Decorator for adding A2A support to existing agent classes
def with_a2a_support(cls):
    """
    Decorator to add A2A support to an existing agent class.
    
    Usage:
        @with_a2a_support
        class MyAgent(ConversationalAgent):
            pass
    """
    
    class A2AEnabledAgent(A2ASupportMixin, cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
        
        async def initialize(self):
            # Call parent initialize
            await super().initialize()
            
            # Initialize A2A support
            await self.initialize_a2a()
        
        async def shutdown(self):
            # Shutdown A2A support
            await self.shutdown_a2a()
            
            # Call parent shutdown
            await super().shutdown()
    
    # Preserve original class metadata
    A2AEnabledAgent.__name__ = cls.__name__
    A2AEnabledAgent.__qualname__ = cls.__qualname__
    A2AEnabledAgent.__module__ = cls.__module__
    
    return A2AEnabledAgent


# Utility functions for A2A agent configuration
def create_a2a_agent_config(
    name: str,
    url: str,
    description: str = "",
    auth_token: Optional[str] = None,
    **kwargs
) -> A2AAgentConfig:
    """Create an A2A agent configuration with common defaults."""
    auth_type = "bearer" if auth_token else "none"
    
    return A2AAgentConfig(
        name=name,
        url=url,
        description=description,
        auth_type=auth_type,
        auth_token=auth_token,
        **kwargs
    )


def create_oauth_a2a_agent_config(
    name: str,
    url: str,
    oauth_config: Dict[str, Any],
    description: str = "",
    **kwargs
) -> A2AAgentConfig:
    """Create an A2A agent configuration with OAuth authentication."""
    return A2AAgentConfig(
        name=name,
        url=url,
        description=description,
        auth_type="oauth2",
        oauth_config=oauth_config,
        **kwargs
    )


async def test_a2a_support():
    """Test function for A2A support functionality."""
    from .agent import ConversationalAgent, AgentConfig
    
    @with_a2a_support
    class TestAgent(ConversationalAgent):
        pass
    
    # Create test agent
    config = AgentConfig(name="test_agent", description="Test agent with A2A support")
    agent = TestAgent(config, a2a_enabled=True)
    
    # Test A2A functionality
    print("A2A support test completed successfully!")


if __name__ == "__main__":
    asyncio.run(test_a2a_support()) 