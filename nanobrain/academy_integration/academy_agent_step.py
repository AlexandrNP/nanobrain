"""
AcademyAgentStep: Nanobrain Step for EXISTING Academy Agent Integration

CRITICAL ARCHITECTURE:
- Connects to EXISTING Academy Redis/ProxyStore infrastructure (does NOT create new managers)
- Discovers EXISTING Academy agents using transport.discover()
- Gets handles to EXISTING agents by their AgentId
- Invokes actions on those remote agents via handles

This module implements AcademyAgentStep, which extends BaseStep to enable
communication with EXISTING Academy agents running on distributed resources.
"""

import asyncio
# Removed logging import - using self.nb_logger instead
import uuid
from typing import Any, Dict, Optional
from pydantic import Field

# Import Nanobrain components
from nanobrain.core.step import BaseStep, StepConfig

# Import Academy components - REAL ONES ONLY
from academy.handle import Handle
from academy.identifier import AgentId
from academy.exchange import (
    RedisExchangeFactory,
    ProxyStoreExchangeFactory,
    LocalExchangeFactory
)


class AcademyAgentStepConfig(StepConfig):
    """Configuration for connecting to EXISTING Academy agents."""

    # Exchange configuration for connecting to EXISTING Academy infrastructure
    exchange_type: str = Field(
        description="Type of Academy exchange: 'redis', 'proxystore', or 'local'"
    )

    # Redis configuration (for connecting to existing Redis exchange)
    redis_host: Optional[str] = Field(
        default="localhost",
        description="Redis server hostname where Academy agents are running"
    )
    redis_port: Optional[int] = Field(
        default=6379,
        description="Redis server port where Academy agents are running"
    )
    redis_kwargs: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional Redis connection parameters"
    )

    # ProxyStore configuration (for connecting to existing ProxyStore exchange)
    proxystore_config: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="ProxyStore configuration parameters"
    )

    # Agent discovery configuration
    agent_id: Optional[str] = Field(
        default=None,
        description="Specific Academy agent ID to connect to (UUID string)"
    )
    agent_class: Optional[str] = Field(
        default=None,
        description="Full import path of Academy agent class for discovery (e.g., 'my_agents.MyAgent')"
    )
    allow_subclasses: bool = Field(
        default=True,
        description="Allow subclasses when discovering agents by class"
    )
    action_name: str = Field(
        default="process",
        description="Action method to invoke on the Academy agent"
    )

    # Communication settings
    timeout_seconds: float = Field(
        default=30.0,
        description="Timeout for Academy agent communication"
    )
    retry_attempts: int = Field(
        default=3,
        description="Number of retry attempts for failed communications"
    )

    # Data transformation
    input_key: Optional[str] = Field(
        default=None,
        description="Key to extract from input_data for Academy agent"
    )
    output_key: Optional[str] = Field(
        default=None,
        description="Key to store Academy agent result in output"
    )


class AcademyAgentStep(BaseStep):
    """
    Nanobrain Step for communicating with EXISTING remote Academy agents.

    This step connects to existing Academy infrastructure and invokes actions
    on already-running Academy agents. It does NOT create new agents or managers.

    **CORRECT ARCHITECTURE:**
    - Connects to EXISTING Academy Redis/ProxyStore infrastructure
    - Discovers EXISTING Academy agents using transport.discover()
    - Gets handles to EXISTING agents by their AgentId
    - Invokes actions on those remote agents via handles
    """

    @classmethod
    def _get_config_class(cls):
        """Return AcademyAgentStepConfig for proper configuration loading"""
        return AcademyAgentStepConfig

    def _init_from_config(self, config: AcademyAgentStepConfig, component_config: Dict[str, Any], dependencies: Dict[str, Any]) -> None:
        """Initialize AcademyAgentStep with proper NanoBrain framework sequence"""
        # Call parent initialization first (required by NanoBrain framework)
        super()._init_from_config(config, component_config, dependencies)

        # Store configuration
        self.config = config

        # Academy components - NO MANAGERS, ONLY CLIENTS AND HANDLES
        self._exchange_client = None
        self._agent_handle = None

        # Connection state
        self._is_connected = False

    async def _connect_to_existing_academy(self) -> None:
        """Connect to EXISTING Academy infrastructure and get agent handle"""
        if self._is_connected:
            return

        try:
            # Check if exchange factory is injected (for testing)
            if hasattr(self, '_exchange_factory'):
                self.nb_logger.info("Using injected exchange factory for testing")
                exchange_factory = self._exchange_factory
            # Create exchange factory to connect to EXISTING infrastructure
            elif self.config.exchange_type == "redis":
                exchange_factory = RedisExchangeFactory(
                    hostname=self.config.redis_host,
                    port=self.config.redis_port,
                    **self.config.redis_kwargs
                )
            elif self.config.exchange_type == "proxystore":
                # ProxyStore requires a base exchange factory
                base_config = self.config.proxystore_config.get("base", {})
                base_exchange_type = base_config.get("exchange_type", "local")

                if base_exchange_type == "local":
                    base_factory = LocalExchangeFactory()
                elif base_exchange_type == "redis":
                    base_factory = RedisExchangeFactory(
                        hostname=base_config.get("redis_host", "localhost"),
                        port=base_config.get("redis_port", 6379),
                        **base_config.get("redis_kwargs", {})
                    )
                else:
                    raise ValueError(f"Unsupported ProxyStore base exchange type: {base_exchange_type}")

                exchange_factory = ProxyStoreExchangeFactory(
                    base=base_factory,
                    store=self.config.proxystore_config.get("store"),
                    should_proxy=self.config.proxystore_config.get("should_proxy", lambda x: True),
                    resolve_async=self.config.proxystore_config.get("resolve_async", False)
                )
            elif self.config.exchange_type == "local":
                exchange_factory = LocalExchangeFactory()
            else:
                raise ValueError(f"Unsupported exchange type: {self.config.exchange_type}")

            # Create USER exchange client to connect to existing infrastructure
            # This is the correct way - we are a USER connecting to existing agents
            self._exchange_client = await exchange_factory.create_user_client()

            # Get handle to existing Academy agent
            if self.config.agent_id:
                # Connect to specific agent by ID
                agent_id = AgentId(uid=uuid.UUID(self.config.agent_id))
                self._agent_handle = Handle(agent_id)
            else:
                # Discover existing agent by class
                await self._discover_existing_agent()

            if not self._agent_handle:
                raise RuntimeError("Failed to get handle to existing Academy agent")

            # Test connection to the agent (skip for testing with injected agent)
            if not hasattr(self, '_test_agent_id'):
                await self._test_agent_connection()
            else:
                self.nb_logger.info("Skipping connection test for injected test agent")

            self._is_connected = True
            self.nb_logger.info(f"Connected to existing Academy agent via {self.config.exchange_type}")

        except Exception as e:
            self.nb_logger.error(f"Failed to connect to existing Academy infrastructure: {e}")
            raise

    async def _discover_existing_agent(self) -> None:
        """Discover existing Academy agent by class using exchange client discover()"""

        # Check if test agent_id is injected (for testing only)
        if hasattr(self, '_test_agent_id'):
            self._agent_handle = Handle(self._test_agent_id)
            self.nb_logger.info(f"Using injected test agent: {self._test_agent_id}")
            return

        if not self.config.agent_class:
            raise ValueError("agent_class required for agent discovery")

        try:
            # Import the agent class dynamically
            module_path, class_name = self.config.agent_class.rsplit('.', 1)
            module = __import__(module_path, fromlist=[class_name])
            agent_class = getattr(module, class_name)

            # Use exchange client's discover method to find existing agents
            agent_ids = await self._exchange_client.discover(
                agent_class,
                allow_subclasses=self.config.allow_subclasses
            )

            if not agent_ids:
                raise RuntimeError(f"No existing agents found for class {self.config.agent_class}")

            # Use the first discovered agent
            agent_id = agent_ids[0]
            # Create handle to communicate with the existing agent
            self._agent_handle = Handle(agent_id)

            self.nb_logger.info(f"Discovered existing Academy agent: {agent_id}")

        except Exception as e:
            self.nb_logger.error(f"Failed to discover existing Academy agent: {e}")
            raise

    async def _test_agent_connection(self) -> None:
        """Test connection to the existing Academy agent"""
        try:
            # Check if the agent's mailbox is active using exchange client
            status = await self._exchange_client.status(self._agent_handle.agent_id)
            if status.name != "ACTIVE":
                raise ConnectionError(f"Academy agent mailbox status is {status.name}, not ACTIVE")

            self.nb_logger.debug("Successfully verified existing Academy agent is active")

        except Exception as e:
            self.nb_logger.error(f"Failed to verify existing Academy agent: {e}")
            raise ConnectionError(f"Cannot reach existing Academy agent: {e}")

    async def _invoke_with_retry(self, agent_input: Any) -> Any:
        """Invoke Academy agent action with retry logic"""
        last_exception = None

        for attempt in range(self.config.retry_attempts):
            try:
                # Use the exchange client context for Handle operations
                async with self._exchange_client:
                    # Get the action method from the handle
                    action_method = getattr(self._agent_handle, self.config.action_name)

                    # Invoke the action with timeout
                    result = await asyncio.wait_for(
                        action_method(agent_input),
                        timeout=self.config.timeout_seconds
                    )

                self.nb_logger.debug(f"Academy agent action succeeded on attempt {attempt + 1}")
                return result

            except asyncio.TimeoutError as e:
                last_exception = e
                self.nb_logger.warning(f"Academy agent timeout on attempt {attempt + 1}")

            except Exception as e:
                last_exception = e
                self.nb_logger.warning(f"Academy agent error on attempt {attempt + 1}: {e}")

            # Wait before retry (exponential backoff)
            if attempt < self.config.retry_attempts - 1:
                await asyncio.sleep(0.5 * (2 ** attempt))

        # All retries failed
        self.nb_logger.error(f"Academy agent action failed after {self.config.retry_attempts} attempts")
        raise last_exception

    def _prepare_input(self, input_data: Dict[str, Any]) -> Any:
        """Prepare input data for Academy agent"""
        if self.config.input_key:
            return input_data.get(self.config.input_key, input_data)
        return input_data

    def _extract_result(self, academy_result: Any) -> Any:
        """Extract and format result from Academy agent"""
        if self.config.output_key:
            return {self.config.output_key: academy_result}
        return academy_result

    async def process(self, input_data: Dict[str, Any], **kwargs) -> Any:
        """
        Process input data by invoking EXISTING Academy agent action.

        Args:
            input_data: Input data from Nanobrain workflow
            **kwargs: Additional keyword arguments

        Returns:
            Result from existing Academy agent action
        """
        try:
            # Connect to existing Academy infrastructure
            await self._connect_to_existing_academy()

            # Prepare input for Academy agent
            agent_input = self._prepare_input(input_data)

            # Invoke existing Academy agent with retry logic
            result = await self._invoke_with_retry(agent_input)

            # Extract and format result
            output = self._extract_result(result)

            self.nb_logger.info("Academy agent step completed successfully")

            # Return data in format expected by Nanobrain output data units
            # The key must match the output data unit name in the config
            return {
                "output_data": output
            }

        except Exception as e:
            self.nb_logger.error(f"Academy agent step failed: {e}")
            raise

    async def cleanup(self) -> None:
        """Cleanup Academy resources"""
        try:
            if self._exchange_client:
                await self._exchange_client.close()
                self._exchange_client = None

            self._agent_handle = None
            self._is_connected = False

            self.nb_logger.info("Academy agent step cleanup completed")

        except Exception as e:
            self.nb_logger.error(f"Academy agent step cleanup failed: {e}")



