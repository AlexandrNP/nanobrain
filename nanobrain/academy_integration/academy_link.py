"""
AcademyLink: Remote Communication Bridge for Academy Integration

This module implements AcademyLink, which extends Nanobrain's LinkBase to enable
communication with remote Academy agents. It handles protocol translation between
Nanobrain's data flow and Academy's action/message system.
"""

import asyncio
import logging
from typing import Any, Dict, Optional

# Import real Nanobrain components
try:
    from nanobrain.core.link import LinkBase
    NANOBRAIN_AVAILABLE = True
except ImportError:
    # Fallback for testing without full Nanobrain
    NANOBRAIN_AVAILABLE = False
    class LinkBase:
        """Mock LinkBase for testing"""
        COMPONENT_TYPE = "link"
        REQUIRED_CONFIG_FIELDS = ['link_type']
        OPTIONAL_CONFIG_FIELDS = {}

        def __init__(self, *args, **kwargs):
            raise RuntimeError("Use from_config instead")

        @classmethod
        def resolve_dependencies(cls, component_config, **kwargs):
            return {}

        def _post_init_setup(self, **kwargs):
            self.name = "test_link"
            self.enable_logging = True
            self._is_active = False
            self._transfer_count = 0
            self._error_count = 0

# Import REAL Academy components (no more fake bullshit)
from academy.handle import Handle
from academy.manager import Manager
ACADEMY_AVAILABLE = True


class AcademyLink(LinkBase):
    """
    Link for communicating with remote Academy agents.
    
    This link enables Nanobrain components to send data to and receive data from
    Academy agents running on distributed resources. It handles the protocol
    translation between Nanobrain's data flow and Academy's action/message system.
    
    **Architecture:**
    - Extends LinkBase for proper Nanobrain integration
    - Communicates with remote Academy agents via handles
    - Translates Nanobrain data to Academy action parameters
    - Handles async communication with timeout and retry logic
    """
    
    COMPONENT_TYPE = "academy_link"
    REQUIRED_CONFIG_FIELDS = ['link_type', 'academy_agent_handle', 'action_name']
    OPTIONAL_CONFIG_FIELDS = {
        'timeout_seconds': 30,
        'retry_attempts': 3,
        'buffer_size': 100,
        'enable_logging': True,
        'source': None,
        'target': None,
        # ProxyStore configuration for cross-node communication
        'proxystore_enabled': False,
        'proxystore_store_dir': '/tmp/proxystore_academy',
        'proxystore_store_name': 'academy-cross-node',
        'proxystore_connector_type': 'file',
        'proxystore_clear_store': False,
        'proxystore_resolve_async': True
    }

    def __init__(self, *args, **kwargs):
        """Prevent direct instantiation - use from_config instead"""
        raise RuntimeError(
            "Direct instantiation of AcademyLink is prohibited. "
            "Use AcademyLink.from_config() as per framework requirements."
        )

    @classmethod
    def extract_component_config(cls, config) -> Dict[str, Any]:
        """Extract AcademyLink configuration including Academy-specific fields"""
        # Get base configuration
        base_config = super().extract_component_config(config)

        # Add Academy-specific fields
        academy_config = {
            'academy_agent_handle': getattr(config, 'academy_agent_handle', None),
            'action_name': getattr(config, 'action_name', None),
            'timeout_seconds': getattr(config, 'timeout_seconds', 30),
            'retry_attempts': getattr(config, 'retry_attempts', 3),
            'enable_logging': getattr(config, 'enable_logging', True),
            # ProxyStore configuration
            'proxystore_enabled': getattr(config, 'proxystore_enabled', False),
            'proxystore_store_dir': getattr(config, 'proxystore_store_dir', '/tmp/proxystore_academy'),
            'proxystore_store_name': getattr(config, 'proxystore_store_name', 'academy-cross-node'),
            'proxystore_connector_type': getattr(config, 'proxystore_connector_type', 'file'),
            'proxystore_clear_store': getattr(config, 'proxystore_clear_store', False),
            'proxystore_resolve_async': getattr(config, 'proxystore_resolve_async', True)
        }

        # Merge configurations
        base_config.update(academy_config)
        return base_config

    @classmethod
    def resolve_dependencies(cls, component_config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Resolve Academy agent handle and connection dependencies"""

        # Get Academy manager and agent handle
        academy_manager = kwargs.get('academy_manager')
        proxystore_enabled = component_config.get('proxystore_enabled', False)

        if not academy_manager and not proxystore_enabled:
            raise ValueError("AcademyLink requires academy_manager in dependencies or proxystore_enabled=True")

        agent_handle_name = component_config.get('academy_agent_handle')
        if not agent_handle_name:
            # Check if it's in kwargs instead
            agent_handle_name = kwargs.get('academy_agent_handle')
            if not agent_handle_name:
                raise ValueError(f"AcademyLink requires academy_agent_handle configuration. Got: {component_config}")

        # Get the actual handle from the manager
        agent_handle = academy_manager.get_handle(agent_handle_name)
        if not agent_handle:
            raise ValueError(f"Academy agent handle '{agent_handle_name}' not found in manager")

        # Get base dependencies from parent class (includes source/target)
        base_dependencies = super().resolve_dependencies(component_config, **kwargs)

        # Add Academy-specific dependencies
        academy_dependencies = {
            'academy_manager': academy_manager,
            'agent_handle': agent_handle,
            'action_name': component_config.get('action_name') or kwargs.get('action_name', 'default_action'),
            'timeout_seconds': component_config.get('timeout_seconds', 30),
            'retry_attempts': component_config.get('retry_attempts', 3),
            # ProxyStore configuration
            'proxystore_enabled': proxystore_enabled,
            'proxystore_store_dir': component_config.get('proxystore_store_dir', '/tmp/proxystore_academy'),
            'proxystore_store_name': component_config.get('proxystore_store_name', 'academy-cross-node'),
            'proxystore_connector_type': component_config.get('proxystore_connector_type', 'file'),
            'proxystore_clear_store': component_config.get('proxystore_clear_store', False),
            'proxystore_resolve_async': component_config.get('proxystore_resolve_async', True)
        }

        # Merge base and Academy dependencies
        base_dependencies.update(academy_dependencies)
        return base_dependencies

    def _init_from_config(self, config, component_config: Dict[str, Any], dependencies: Dict[str, Any]) -> None:
        """Initialize AcademyLink with resolved dependencies"""
        # Call parent _init_from_config first - this sets up self.name properly
        super()._init_from_config(config, component_config, dependencies)

        # Store Academy-specific dependencies
        self.academy_manager = dependencies.get('academy_manager')
        self.agent_handle = dependencies.get('agent_handle')
        self.action_name = dependencies.get('action_name', 'default_action')
        self.timeout_seconds = dependencies.get('timeout_seconds', 30)
        self.retry_attempts = dependencies.get('retry_attempts', 3)

        # Store ProxyStore configuration
        self.proxystore_enabled = dependencies.get('proxystore_enabled', False)
        self.proxystore_store_dir = dependencies.get('proxystore_store_dir', '/tmp/proxystore_academy')
        self.proxystore_store_name = dependencies.get('proxystore_store_name', 'academy-cross-node')
        self.proxystore_connector_type = dependencies.get('proxystore_connector_type', 'file')
        self.proxystore_clear_store = dependencies.get('proxystore_clear_store', False)
        self.proxystore_resolve_async = dependencies.get('proxystore_resolve_async', True)

        # Initialize connection state
        self._is_connected = False
        self._connection_attempts = 0

        # Setup logging AFTER parent initialization so self.name is properly set
        # Use config name as fallback if source->target name isn't available
        link_name = getattr(self, 'name', None) or getattr(config, 'name', 'academy_link')
        self.logger = logging.getLogger(f"AcademyLink.{link_name}")
        self.enable_logging = dependencies.get('enable_logging', True)

        if self.enable_logging and self.logger:
            self.logger.info(f"AcademyLink {link_name} initialized for action '{self.action_name}'")
            if self.proxystore_enabled:
                self.logger.info(f"ProxyStore enabled: {self.proxystore_connector_type} connector at {self.proxystore_store_dir}")

    async def start(self) -> None:
        """Start the Academy link and establish connection."""
        try:
            # Create ProxyStore-enabled Academy Manager if needed
            if self.proxystore_enabled and not self.academy_manager:
                await self._create_proxystore_manager()

            # Test connection to remote Academy agent
            await self._test_connection()
            self._is_active = True
            self._is_connected = True

            link_name = getattr(self, 'name', 'academy_link')
            if self.enable_logging:
                self.logger.info(f"AcademyLink {link_name} started successfully")

        except Exception as e:
            link_name = getattr(self, 'name', 'academy_link')
            self.logger.error(f"Failed to start AcademyLink {link_name}: {e}")
            raise

    async def stop(self) -> None:
        """Stop the Academy link."""
        self._is_active = False
        self._is_connected = False

        link_name = getattr(self, 'name', 'academy_link')
        if self.enable_logging:
            self.logger.info(f"AcademyLink {link_name} stopped")

    async def transfer(self, data: Any) -> Any:
        """Transfer data to remote Academy agent via action call and forward result to target"""
        link_name = getattr(self, 'name', 'academy_link')
        if not self._is_active or not self._is_connected:
            raise RuntimeError(f"AcademyLink {link_name} is not active or connected")

        for attempt in range(self.retry_attempts):
            try:
                # Call the specified action on the remote Academy agent
                result = await asyncio.wait_for(
                    self._call_remote_action(data),
                    timeout=self.timeout_seconds
                )

                # Transfer result to target data unit if target is specified
                if hasattr(self, 'target') and self.target is not None:
                    if hasattr(self.target, 'set') and callable(getattr(self.target, 'set')):
                        # Target is a DataUnit - set the result
                        await self.target.set(result)
                        if self.enable_logging:
                            self.logger.info(f"AcademyLink {link_name} transferred result to target data unit")
                    else:
                        if self.enable_logging:
                            self.logger.warning(f"AcademyLink {link_name} target does not support set() method")

                # Record successful transfer
                await self._record_transfer(True)

                if self.enable_logging:
                    self.logger.debug(f"AcademyLink {link_name} transfer successful (attempt {attempt + 1})")

                return result

            except asyncio.TimeoutError:
                self.logger.warning(f"AcademyLink {link_name} timeout on attempt {attempt + 1}")
                if attempt == self.retry_attempts - 1:
                    await self._record_transfer(False)
                    raise

            except Exception as e:
                self.logger.error(f"AcademyLink {link_name} transfer failed on attempt {attempt + 1}: {e}")
                if attempt == self.retry_attempts - 1:
                    await self._record_transfer(False)
                    raise

                # Wait before retry
                await asyncio.sleep(0.5 * (attempt + 1))

    async def _call_remote_action(self, data: Any) -> Any:
        """Call the remote Academy agent action with data"""
        # Get the action method from the handle
        action_method = getattr(self.agent_handle, self.action_name)
        
        # Call the remote action with the data
        return await action_method(data)

    async def _test_connection(self) -> None:
        """Test connection to remote Academy agent"""
        try:
            # Try to call a simple action or ping
            # For now, just verify the handle exists and has the required action
            if not hasattr(self.agent_handle, self.action_name):
                raise ValueError(f"Remote agent does not have action '{self.action_name}'")
                
            self._connection_attempts += 1
            
        except Exception as e:
            self._connection_attempts += 1
            raise ConnectionError(f"Failed to connect to Academy agent: {e}")

    async def _record_transfer(self, success: bool) -> None:
        """Record transfer statistics"""
        if success:
            self._transfer_count += 1
        else:
            self._error_count += 1

    async def _create_proxystore_manager(self) -> None:
        """Create ProxyStore-enabled Academy Manager for cross-node communication"""
        try:
            # Import DistributedAcademyManager
            from .distributed_manager import DistributedAcademyManager

            # Create ProxyStore-enabled Academy Manager
            self.academy_manager = await DistributedAcademyManager.create_proxystore_manager(
                store_dir=self.proxystore_store_dir,
                store_name=self.proxystore_store_name,
                connector_type=self.proxystore_connector_type,
                clear_store=self.proxystore_clear_store,
                resolve_async=self.proxystore_resolve_async
            )

            link_name = getattr(self, 'name', 'academy_link')
            if self.enable_logging:
                self.logger.info(f"Created ProxyStore Academy Manager for {link_name}")

        except Exception as e:
            link_name = getattr(self, 'name', 'academy_link')
            self.logger.error(f"Failed to create ProxyStore Academy Manager for {link_name}: {e}")
            raise
