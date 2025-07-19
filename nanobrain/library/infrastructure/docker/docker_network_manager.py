"""
Docker Network Manager for NanoBrain Framework

This module provides Docker network management capabilities for container
networking, isolation, and service discovery.

"""

from typing import Dict, List, Optional, Any, Union, ClassVar
from dataclasses import dataclass, field

import docker
from docker.errors import DockerException, APIError

from .container_config import DockerComponentConfig, DockerComponentBase
from nanobrain.core.logging_system import get_logger
from nanobrain.core.component_base import ComponentDependencyError


@dataclass
class NetworkConfig:
    """Docker network configuration"""
    name: str
    driver: str = "bridge"
    subnet: Optional[str] = None
    gateway: Optional[str] = None
    ip_range: Optional[str] = None
    enable_ipv6: bool = False
    internal: bool = False
    attachable: bool = True
    labels: Dict[str, str] = field(default_factory=dict)
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DockerNetworkManagerConfig(DockerComponentConfig):
    """Configuration for Docker Network Manager"""
    component_name: str = "docker_network_manager"
    nanobrain_network_enabled: bool = True
    default_network_driver: str = "bridge"
    default_subnet: str = "172.20.0.0/16"
    default_gateway: str = "172.20.0.1"


class DockerNetworkManager(DockerComponentBase):
    """
    Docker network management for container networking and isolation.
    Enhanced with mandatory from_config pattern implementation.
    
    Provides capabilities for:
    - Network creation and management
    - Container network attachment/detachment
    - Network isolation and security
    - Service discovery
    """
    
    # Component configuration schema
    COMPONENT_TYPE: ClassVar[str] = "docker_network_manager"
    REQUIRED_CONFIG_FIELDS: ClassVar[List[str]] = []
    OPTIONAL_CONFIG_FIELDS: ClassVar[Dict[str, Any]] = {
        'docker_client': None,
        'nanobrain_network_enabled': True,
        'default_network_driver': 'bridge'
    }
    
    def __init__(self, *args, **kwargs):
        """Prevent direct instantiation - use from_config instead"""
        raise RuntimeError(
            f"Direct instantiation of {self.__class__.__name__} is prohibited. "
            f"ALL framework components must use {self.__class__.__name__}.from_config() "
            f"as per mandatory framework requirements."
        )
    
    @classmethod
    def extract_component_config(cls, config: Any) -> Dict[str, Any]:
        """Extract component-specific configuration"""
        if isinstance(config, DockerNetworkManagerConfig):
            return {
                'component_name': config.component_name,
                'enabled': config.enabled,
                'docker_client': config.docker_client,
                'nanobrain_network_enabled': config.nanobrain_network_enabled,
                'default_network_driver': config.default_network_driver,
                'default_subnet': config.default_subnet,
                'default_gateway': config.default_gateway
            }
        elif isinstance(config, dict):
            return config
        else:
            return {}
    
    @classmethod
    def resolve_dependencies(cls, component_config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Resolve network manager dependencies"""
        try:
            docker_client = component_config.get('docker_client')
            if not docker_client:
                docker_client = docker.from_env()
            
            # Test Docker connection
            docker_client.ping()
            
            return {
                'docker_client': docker_client,
                **kwargs
            }
        except Exception as e:
            raise ComponentDependencyError(f"Failed to connect to Docker: {e}")
    
    def _init_from_config(self, config: Any, component_config: Dict[str, Any],
                         dependencies: Dict[str, Any]) -> None:
        """Initialize DockerNetworkManager from configuration"""
        super()._init_from_config(config, component_config, dependencies)
        
        # Docker client
        self.client = dependencies['docker_client']
        
        # Network configuration
        nanobrain_network_enabled = component_config.get('nanobrain_network_enabled', True)
        
        # Default NanoBrain network configuration
        if nanobrain_network_enabled:
            self.nanobrain_network_config = NetworkConfig(
                name="nanobrain-network",
                driver=component_config.get('default_network_driver', 'bridge'),
                subnet=component_config.get('default_subnet', '172.20.0.0/16'),
                gateway=component_config.get('default_gateway', '172.20.0.1'),
                labels={
                    "nanobrain.framework": "true",
                    "nanobrain.network.type": "framework"
                }
            )
        else:
            self.nanobrain_network_config = None
    
    async def create_network(self, config: NetworkConfig) -> bool:
        """
        Create a Docker network.
        
        Args:
            config: Network configuration
            
        Returns:
            bool: True if network created successfully
        """
        try:
            self.logger.info(f"🌐 Creating Docker network: {config.name}")
            
            # Check if network already exists
            if await self._network_exists(config.name):
                self.logger.info(f"Network {config.name} already exists")
                return True
            
            # Prepare network options
            ipam_config = {}
            if config.subnet or config.gateway or config.ip_range:
                ipam_config = {
                    "Driver": "default",
                    "Config": [{
                        "Subnet": config.subnet,
                        "Gateway": config.gateway,
                        "IPRange": config.ip_range
                    }]
                }
            
            # Create network
            network = self.client.networks.create(
                name=config.name,
                driver=config.driver,
                options=config.options,
                ipam=ipam_config if ipam_config else None,
                check_duplicate=True,
                internal=config.internal,
                labels=config.labels,
                enable_ipv6=config.enable_ipv6,
                attachable=config.attachable
            )
            
            self.logger.info(f"✅ Network {config.name} created successfully")
            return True
            
        except DockerException as e:
            self.logger.error(f"❌ Failed to create network {config.name}: {e}")
            return False
    
    async def remove_network(self, name: str, force: bool = False) -> bool:
        """
        Remove a Docker network.
        
        Args:
            name: Network name
            force: Force removal even if containers are connected
            
        Returns:
            bool: True if network removed successfully
        """
        try:
            network = self.client.networks.get(name)
            
            # Disconnect all containers if force is True
            if force:
                await self._disconnect_all_containers(name)
            
            network.remove()
            self.logger.info(f"✅ Network {name} removed successfully")
            return True
            
        except docker.errors.NotFound:
            self.logger.info(f"Network {name} not found (already removed)")
            return True
        except DockerException as e:
            self.logger.error(f"❌ Failed to remove network {name}: {e}")
            return False
    
    async def connect_container(self, network_name: str, container_name: str, 
                              aliases: Optional[List[str]] = None,
                              ipv4_address: Optional[str] = None,
                              ipv6_address: Optional[str] = None) -> bool:
        """
        Connect a container to a network.
        
        Args:
            network_name: Network name
            container_name: Container name
            aliases: Network aliases for the container
            ipv4_address: Static IPv4 address
            ipv6_address: Static IPv6 address
            
        Returns:
            bool: True if container connected successfully
        """
        try:
            network = self.client.networks.get(network_name)
            container = self.client.containers.get(container_name)
            
            # Prepare network configuration
            config = {}
            if aliases:
                config["Aliases"] = aliases
            if ipv4_address:
                config["IPAMConfig"] = {"IPv4Address": ipv4_address}
            if ipv6_address:
                if "IPAMConfig" not in config:
                    config["IPAMConfig"] = {}
                config["IPAMConfig"]["IPv6Address"] = ipv6_address
            
            network.connect(container, **config)
            
            self.logger.info(f"✅ Connected container {container_name} to network {network_name}")
            return True
            
        except docker.errors.NotFound as e:
            self.logger.error(f"❌ Network or container not found: {e}")
            return False
        except DockerException as e:
            self.logger.error(f"❌ Failed to connect container to network: {e}")
            return False
    
    async def disconnect_container(self, network_name: str, container_name: str, 
                                 force: bool = False) -> bool:
        """
        Disconnect a container from a network.
        
        Args:
            network_name: Network name
            container_name: Container name
            force: Force disconnection
            
        Returns:
            bool: True if container disconnected successfully
        """
        try:
            network = self.client.networks.get(network_name)
            container = self.client.containers.get(container_name)
            
            network.disconnect(container, force=force)
            
            self.logger.info(f"✅ Disconnected container {container_name} from network {network_name}")
            return True
            
        except docker.errors.NotFound as e:
            self.logger.error(f"❌ Network or container not found: {e}")
            return False
        except DockerException as e:
            self.logger.error(f"❌ Failed to disconnect container from network: {e}")
            return False
    
    async def list_networks(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        List Docker networks.
        
        Args:
            filters: Optional filters to apply
            
        Returns:
            List of network information
        """
        try:
            networks = self.client.networks.list(filters=filters or {})
            
            network_list = []
            for network in networks:
                network_info = {
                    "id": network.id[:12],
                    "name": network.name,
                    "driver": network.attrs.get("Driver"),
                    "scope": network.attrs.get("Scope"),
                    "created": network.attrs.get("Created"),
                    "internal": network.attrs.get("Internal", False),
                    "attachable": network.attrs.get("Attachable", False),
                    "labels": network.attrs.get("Labels", {}),
                    "containers": list(network.attrs.get("Containers", {}).keys()),
                    "subnet": self._extract_subnet(network.attrs),
                    "gateway": self._extract_gateway(network.attrs)
                }
                network_list.append(network_info)
            
            return network_list
            
        except DockerException as e:
            self.logger.error(f"❌ Failed to list networks: {e}")
            return []
    
    async def get_network_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific network.
        
        Args:
            name: Network name
            
        Returns:
            Network information or None if not found
        """
        try:
            network = self.client.networks.get(name)
            
            # Get connected containers with their details
            connected_containers = []
            containers = network.attrs.get("Containers", {})
            for container_id, container_info in containers.items():
                try:
                    container = self.client.containers.get(container_id)
                    connected_containers.append({
                        "id": container_id[:12],
                        "name": container.name,
                        "ipv4_address": container_info.get("IPv4Address", "").split("/")[0],
                        "ipv6_address": container_info.get("IPv6Address", "").split("/")[0],
                        "mac_address": container_info.get("MacAddress"),
                        "endpoint_id": container_info.get("EndpointID", "")[:12]
                    })
                except docker.errors.NotFound:
                    # Container no longer exists
                    continue
            
            network_info = {
                "id": network.id,
                "name": network.name,
                "driver": network.attrs.get("Driver"),
                "scope": network.attrs.get("Scope"),
                "created": network.attrs.get("Created"),
                "internal": network.attrs.get("Internal", False),
                "attachable": network.attrs.get("Attachable", False),
                "enable_ipv6": network.attrs.get("EnableIPv6", False),
                "labels": network.attrs.get("Labels", {}),
                "options": network.attrs.get("Options", {}),
                "ipam": network.attrs.get("IPAM", {}),
                "connected_containers": connected_containers,
                "subnet": self._extract_subnet(network.attrs),
                "gateway": self._extract_gateway(network.attrs)
            }
            
            return network_info
            
        except docker.errors.NotFound:
            self.logger.warning(f"Network {name} not found")
            return None
        except DockerException as e:
            self.logger.error(f"❌ Failed to get network info for {name}: {e}")
            return None
    
    async def create_nanobrain_network(self) -> bool:
        """Create the default NanoBrain framework network"""
        return await self.create_network(self.nanobrain_network_config)
    
    async def cleanup_unused_networks(self) -> Dict[str, Any]:
        """
        Clean up unused networks (not connected to any containers).
        
        Returns:
            Cleanup statistics
        """
        try:
            self.logger.info("🧹 Cleaning up unused Docker networks...")
            
            # Get all networks
            networks = self.client.networks.list()
            
            removed_networks = []
            protected_networks = {"bridge", "host", "none"}
            
            for network in networks:
                # Skip protected networks
                if network.name in protected_networks:
                    continue
                
                # Skip networks with connected containers
                containers = network.attrs.get("Containers", {})
                if containers:
                    continue
                
                # Skip NanoBrain networks that should be preserved
                labels = network.attrs.get("Labels", {})
                if labels.get("nanobrain.framework") == "true":
                    continue
                
                try:
                    network.remove()
                    removed_networks.append({
                        "id": network.id[:12],
                        "name": network.name,
                        "driver": network.attrs.get("Driver")
                    })
                    self.logger.debug(f"Removed unused network: {network.name}")
                except DockerException as e:
                    self.logger.warning(f"Failed to remove network {network.name}: {e}")
            
            cleanup_stats = {
                "networks_removed": len(removed_networks),
                "removed_networks": removed_networks
            }
            
            self.logger.info(f"✅ Network cleanup completed: {cleanup_stats}")
            return cleanup_stats
            
        except DockerException as e:
            self.logger.error(f"❌ Network cleanup failed: {e}")
            return {"error": str(e)}
    
    async def inspect_container_networks(self, container_name: str) -> Dict[str, Any]:
        """
        Get network information for a specific container.
        
        Args:
            container_name: Container name
            
        Returns:
            Container network information
        """
        try:
            container = self.client.containers.get(container_name)
            container.reload()
            
            network_settings = container.attrs.get("NetworkSettings", {})
            networks = network_settings.get("Networks", {})
            
            container_networks = {}
            for network_name, network_info in networks.items():
                container_networks[network_name] = {
                    "network_id": network_info.get("NetworkID", "")[:12],
                    "endpoint_id": network_info.get("EndpointID", "")[:12],
                    "gateway": network_info.get("Gateway"),
                    "ip_address": network_info.get("IPAddress"),
                    "ip_prefix_len": network_info.get("IPPrefixLen"),
                    "ipv6_gateway": network_info.get("IPv6Gateway"),
                    "global_ipv6_address": network_info.get("GlobalIPv6Address"),
                    "mac_address": network_info.get("MacAddress"),
                    "aliases": network_info.get("Aliases", [])
                }
            
            return {
                "container_name": container_name,
                "container_id": container.id[:12],
                "networks": container_networks,
                "ports": container.ports
            }
            
        except docker.errors.NotFound:
            self.logger.error(f"Container {container_name} not found")
            return {"error": "Container not found"}
        except DockerException as e:
            self.logger.error(f"❌ Failed to inspect container networks: {e}")
            return {"error": str(e)}
    
    async def _network_exists(self, name: str) -> bool:
        """Check if a network exists"""
        try:
            self.client.networks.get(name)
            return True
        except docker.errors.NotFound:
            return False
    
    async def _disconnect_all_containers(self, network_name: str):
        """Disconnect all containers from a network"""
        try:
            network = self.client.networks.get(network_name)
            containers = network.attrs.get("Containers", {})
            
            for container_id in containers.keys():
                try:
                    container = self.client.containers.get(container_id)
                    network.disconnect(container, force=True)
                    self.logger.debug(f"Forcefully disconnected {container.name} from {network_name}")
                except Exception as e:
                    self.logger.warning(f"Failed to disconnect container {container_id}: {e}")
        except Exception as e:
            self.logger.error(f"Failed to disconnect all containers from {network_name}: {e}")
    
    def _extract_subnet(self, network_attrs: Dict[str, Any]) -> Optional[str]:
        """Extract subnet from network attributes"""
        ipam = network_attrs.get("IPAM", {})
        config = ipam.get("Config", [])
        if config and len(config) > 0:
            return config[0].get("Subnet")
        return None
    
    def _extract_gateway(self, network_attrs: Dict[str, Any]) -> Optional[str]:
        """Extract gateway from network attributes"""
        ipam = network_attrs.get("IPAM", {})
        config = ipam.get("Config", [])
        if config and len(config) > 0:
            return config[0].get("Gateway")
        return None 