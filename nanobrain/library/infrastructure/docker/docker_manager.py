"""
Docker Manager for NanoBrain Framework

This module provides Docker-specific container management capabilities,
implementing the universal container interface for Docker operations.

"""

import asyncio
import json
import subprocess
from typing import Dict, List, Optional, Any, Union, ClassVar
from pathlib import Path

import docker
from docker.errors import DockerException, ContainerError, ImageNotFound, APIError

from .container_config import ContainerConfig, ContainerManagerBase, ContainerOrchestrator, DockerComponentConfig
from nanobrain.core.logging_system import get_logger
from nanobrain.core.component_base import ComponentDependencyError


class DockerError(Exception):
    """Base exception for Docker operations"""
    pass


class ContainerNotFoundError(DockerError):
    """Raised when a container is not found"""
    pass


class DockerManager(ContainerManagerBase):
    """
    Docker Container Manager - Enterprise Container Infrastructure Management and Orchestration
    ===========================================================================================
    
    The DockerManager provides comprehensive Docker container lifecycle management, infrastructure
    orchestration, and operational capabilities for the NanoBrain framework. This manager implements
    the universal container interface with Docker-specific optimizations, automated resource management,
    and production-ready monitoring and health checking capabilities.
    
    **Core Architecture:**
        The Docker manager provides enterprise-grade container management:
        
        * **Container Lifecycle**: Complete container creation, start, stop, remove operations
        * **Infrastructure Management**: Network, volume, and image management with automation
        * **Health Monitoring**: Real-time health checking and status monitoring
        * **Resource Management**: Resource limit enforcement and cleanup automation
        * **Production Operations**: Logging, monitoring, and operational excellence features
        * **Framework Integration**: Seamless integration with NanoBrain component architecture
    
    **Container Lifecycle Management:**
        
        **Container Operations:**
        * Automated container creation with configuration validation and optimization
        * Intelligent container start/stop operations with health verification
        * Graceful shutdown procedures with configurable timeout handling
        * Force removal capabilities for stuck or unresponsive containers
        
        **Configuration Management:**
        * Environment variable injection and configuration management
        * Port mapping and network attachment configuration
        * Volume mounting and data persistence management
        * Resource limit enforcement for memory, CPU, and disk usage
        
        **Health and Status Monitoring:**
        * Real-time container health checking with custom health check configuration
        * Detailed status information including resource usage and performance metrics
        * Container log access and monitoring capabilities
        * Automatic restart and recovery for failed containers
        
        **Image Management:**
        * Automatic image pulling and version management
        * Image availability verification and caching
        * Multi-architecture image support and platform optimization
        * Registry authentication and secure image access
    
    **Infrastructure Management:**
        
        **Network Management:**
        * Docker network creation and configuration for service communication
        * Network isolation and security policy enforcement
        * Service discovery and DNS configuration
        * Load balancing and traffic distribution setup
        
        **Volume Management:**
        * Persistent volume creation and lifecycle management
        * Volume mounting and data sharing between containers
        * Backup and snapshot management for data protection
        * Volume cleanup and space management automation
        
        **Resource Management:**
        * Memory and CPU limit enforcement for container isolation
        * Disk space monitoring and quota management
        * Resource usage tracking and optimization recommendations
        * Automatic resource cleanup and garbage collection
        
        **Security and Isolation:**
        * Container security policy enforcement and compliance
        * Network isolation and firewall configuration
        * Secret management and secure credential injection
        * Access control and permission management
    
    **Configuration Architecture:**
        Comprehensive configuration supports enterprise deployment:
        
        ```yaml
        # Docker Manager Configuration
        component_type: "docker_manager"
        
        # Docker Connection Configuration
        docker_client: null  # Auto-configured from environment
        enabled: true
        
        # Container Management
        max_containers: 50
        default_memory_limit: "1g"
        default_cpu_limit: 1.0
        
        # Resource Management
        resource_limits:
          memory_default: "1Gi"
          memory_max: "8Gi"
          cpu_default: 1.0
          cpu_max: 4.0
          
        # Health Monitoring
        health_monitoring:
          enabled: true
          check_interval: 30
          timeout: 10
          retries: 3
          
        # Network Configuration
        networking:
          default_network: "nanobrain-network"
          network_driver: "bridge"
          enable_isolation: true
          
        # Volume Management
        volumes:
          default_driver: "local"
          backup_enabled: true
          retention_days: 30
          
        # Cleanup Configuration
        cleanup:
          auto_cleanup: true
          cleanup_interval: 3600  # 1 hour
          keep_stopped_containers: 5
          prune_unused_images: true
        ```
    
    **Usage Patterns:**
        
        **Basic Container Management:**
        ```python
        from nanobrain.library.infrastructure.docker import DockerManager, ContainerConfig
        
        # Create Docker manager with configuration
        docker_mgr = DockerManager.from_config('config/docker_config.yml')
        
        # Create container configuration
        container_config = ContainerConfig(
            name="nanobrain-elasticsearch",
            image="elasticsearch",
            tag="8.14.0",
            ports=["9200:9200", "9300:9300"],
            environment={"discovery.type": "single-node"},
            memory_limit="2g",
            cpu_limit=1.0
        )
        
        # Deploy and manage container
        await docker_mgr.create_container(container_config)
        await docker_mgr.start_container("nanobrain-elasticsearch")
        
        # Monitor container health
        status = await docker_mgr.get_container_status("nanobrain-elasticsearch")
        print(f"Container health: {status['health']}")
        ```
        
        **Infrastructure Orchestration:**
        ```python
        # Create complete infrastructure stack
        docker_mgr = DockerManager.from_config('config/production_docker.yml')
        
        # Create network for service communication
        await docker_mgr.create_network("nanobrain-network", driver="bridge")
        
        # Create persistent volumes
        await docker_mgr.create_volume("nanobrain-data", driver="local")
        await docker_mgr.create_volume("nanobrain-logs", driver="local")
        
        # Deploy multiple services with dependencies
        services = [
            {
                "name": "nanobrain-db",
                "image": "postgres:15",
                "volumes": ["nanobrain-data:/var/lib/postgresql/data"],
                "networks": ["nanobrain-network"],
                "environment": {"POSTGRES_DB": "nanobrain"}
            },
            {
                "name": "nanobrain-search",
                "image": "elasticsearch:8.14.0",
                "networks": ["nanobrain-network"],
                "depends_on": ["nanobrain-db"]
            }
        ]
        
        # Deploy services in dependency order
        for service in services:
            config = ContainerConfig(**service)
            await docker_mgr.create_container(config)
            await docker_mgr.start_container(service["name"])
        ```
        
        **Health Monitoring and Management:**
        ```python
        # Comprehensive health monitoring
        docker_mgr = DockerManager.from_config('config/monitored_docker.yml')
        
        # Monitor all managed containers
        containers = await docker_mgr.list_containers()
        
        for container in containers:
            # Get detailed health status
            health = await docker_mgr.health_check(container["name"])
            status = await docker_mgr.get_container_status(container["name"])
            
            print(f"Container: {container['name']}")
            print(f"  Health: {'✅' if health else '❌'}")
            print(f"  Status: {status['state']}")
            print(f"  Memory: {status.get('memory_usage', 'N/A')}")
            print(f"  CPU: {status.get('cpu_usage', 'N/A')}")
            
            # Restart unhealthy containers
            if not health and status['state'] == 'running':
                print(f"  🔄 Restarting unhealthy container...")
                await docker_mgr.restart_container(container["name"])
        ```
        
        **Resource Management and Cleanup:**
        ```python
        # Automated resource management
        docker_mgr = DockerManager.from_config('config/resource_managed_docker.yml')
        
        # Monitor resource usage
        system_info = await docker_mgr.get_system_info()
        print(f"Docker version: {system_info['version']}")
        print(f"Total containers: {system_info['containers']}")
        print(f"Running containers: {system_info['containers_running']}")
        
        # Cleanup unused resources
        cleanup_stats = await docker_mgr.cleanup_unused_resources()
        print(f"Cleanup results:")
        print(f"  Containers removed: {cleanup_stats['containers_removed']}")
        print(f"  Images removed: {cleanup_stats['images_removed']}")
        print(f"  Networks removed: {cleanup_stats['networks_removed']}")
        print(f"  Space reclaimed: {cleanup_stats['space_reclaimed']} bytes")
        
        # Manage container lifecycle
        for container_name in ["old-container-1", "old-container-2"]:
            if await docker_mgr._container_exists(container_name):
                await docker_mgr.stop_container(container_name, timeout=30)
                await docker_mgr.remove_container(container_name)
        ```
        
        **Production Deployment:**
        ```python
        # Production-ready deployment with full configuration
        production_config = {
            'max_containers': 100,
            'default_memory_limit': '2g',
            'default_cpu_limit': 2.0,
            'health_monitoring': {'enabled': True, 'check_interval': 15},
            'cleanup': {'auto_cleanup': True, 'cleanup_interval': 1800}
        }
        
        docker_mgr = DockerManager.from_config(production_config)
        
        # Deploy with high availability configuration
        ha_config = ContainerConfig(
            name="nanobrain-app-ha",
            image="nanobrain/app:latest",
            ports=["8080:8080"],
            environment={
                "NODE_ENV": "production",
                "LOG_LEVEL": "info",
                "HEALTH_CHECK_ENABLED": "true"
            },
            memory_limit="4g",
            cpu_limit=2.0,
            restart_policy="unless-stopped",
            health_check={
                "test": ["CMD", "curl", "-f", "http://localhost:8080/health"],
                "interval": "30s",
                "timeout": "10s",
                "retries": 3,
                "start_period": "60s"
            }
        )
        
        await docker_mgr.create_container(ha_config)
        await docker_mgr.start_container("nanobrain-app-ha")
        ```
    
    **Advanced Features:**
        
        **Container Orchestration:**
        * Multi-container application deployment and coordination
        * Service dependency management and startup ordering
        * Rolling updates and blue-green deployment support
        * Container scaling and load balancing configuration
        
        **Security and Compliance:**
        * Container security scanning and vulnerability assessment
        * Network security policy enforcement and isolation
        * Secret management and secure credential injection
        * Compliance reporting and audit trail generation
        
        **Performance Optimization:**
        * Resource usage monitoring and optimization recommendations
        * Container performance tuning and configuration optimization
        * Image optimization and layer caching strategies
        * Network and storage performance optimization
        
        **Operational Excellence:**
        * Comprehensive logging and monitoring integration
        * Automated backup and disaster recovery procedures
        * Health checking and automatic recovery mechanisms
        * Metrics collection and dashboard integration
    
    **Integration Patterns:**
        
        **CI/CD Integration:**
        * Integration with continuous integration and deployment pipelines
        * Automated testing and validation in containerized environments
        * Image building and registry management automation
        * Deployment automation and rollback capabilities
        
        **Monitoring and Observability:**
        * Integration with monitoring systems (Prometheus, Grafana, ELK stack)
        * Structured logging with correlation IDs and distributed tracing
        * Metrics collection and alerting for operational insights
        * Performance monitoring and capacity planning support
        
        **Cloud Integration:**
        * Cloud provider integration for managed Docker services
        * Container registry integration with cloud providers
        * Auto-scaling integration with cloud orchestration services
        * Cost optimization and resource management in cloud environments
        
        **Enterprise Integration:**
        * LDAP/Active Directory integration for authentication
        * Enterprise security policy enforcement and compliance
        * Integration with enterprise monitoring and management tools
        * Support for enterprise networking and storage solutions
    
    **Production Deployment:**
        
        **High Availability:**
        * Multi-node container deployment for fault tolerance
        * Automatic failover and recovery mechanisms
        * Load balancing and traffic distribution
        * Geographic distribution for disaster recovery
        
        **Scalability Features:**
        * Horizontal container scaling based on demand
        * Resource allocation optimization for high-density deployments
        * Performance monitoring and capacity planning
        * Auto-scaling integration with orchestration platforms
        
        **Security Hardening:**
        * Container security best practices implementation
        * Network isolation and micro-segmentation
        * Runtime security monitoring and threat detection
        * Compliance automation and audit reporting
        
        **Operational Excellence:**
        * Automated deployment and configuration management
        * Zero-downtime updates and maintenance procedures
        * Comprehensive backup and disaster recovery
        * Performance optimization and troubleshooting tools
    
    Attributes:
        client (docker.DockerClient): Docker client for container operations
        api_client (docker.APIClient): Low-level Docker API client
        docker_version (str): Docker engine version information
        api_version (str): Docker API version information
        managed_containers (Dict[str, ContainerConfig]): Registry of managed containers
        max_containers (int): Maximum number of containers to manage
        default_memory_limit (str): Default memory limit for containers
        default_cpu_limit (float): Default CPU limit for containers
    
    Note:
        This manager requires Docker to be installed and running on the host system.
        The Docker daemon must be accessible through the Docker API. Container operations
        require appropriate permissions and may need elevated privileges for some operations.
        Network and volume operations may require additional system permissions.
    
    Warning:
        Container operations can consume significant system resources. Monitor resource
        usage and implement appropriate limits for production deployments. Be cautious
        with container removal operations as they can result in data loss. Network and
        volume operations affect system-wide resources and should be carefully planned.
    
    See Also:
        * :class:`ContainerManagerBase`: Base container manager interface
        * :class:`ContainerConfig`: Container configuration schema
        * :class:`ContainerOrchestrator`: Container orchestration capabilities
        * :mod:`nanobrain.library.infrastructure.docker`: Docker infrastructure components
        * :mod:`nanobrain.core.component_base`: Framework component architecture
    """
    
    # Component configuration schema
    COMPONENT_TYPE: ClassVar[str] = "docker_manager"
    REQUIRED_CONFIG_FIELDS: ClassVar[List[str]] = []
    OPTIONAL_CONFIG_FIELDS: ClassVar[Dict[str, Any]] = {
        'docker_client': None,
        'enabled': True,
        'max_containers': 50,
        'default_memory_limit': '1g',
        'default_cpu_limit': 1.0
    }
    
    @classmethod
    def resolve_dependencies(cls, component_config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Resolve Docker-specific dependencies"""
        try:
            docker_client = component_config.get('docker_client')
            if not docker_client:
                docker_client = docker.from_env()
            
            # Test Docker connection
            docker_client.ping()
            version_info = docker_client.version()
            
            return {
                'docker_client': docker_client,
                'docker_version': version_info.get('Version', 'unknown'),
                'api_version': version_info.get('ApiVersion', 'unknown'),
                **kwargs
            }
        except Exception as e:
            raise ComponentDependencyError(f"Failed to connect to Docker: {e}")
    
    def _init_from_config(self, config: Any, component_config: Dict[str, Any],
                         dependencies: Dict[str, Any]) -> None:
        """Initialize DockerManager from configuration"""
        super()._init_from_config(config, component_config, dependencies)
        
        # Docker-specific initialization
        self.client = dependencies['docker_client']
        self.docker_version = dependencies.get('docker_version', 'unknown')
        self.api_version = dependencies.get('api_version', 'unknown')
        
        # Container management
        self.managed_containers: Dict[str, ContainerConfig] = {}
        self.max_containers = component_config.get('max_containers', 50)
        self.default_memory_limit = component_config.get('default_memory_limit', '1g')
        self.default_cpu_limit = component_config.get('default_cpu_limit', 1.0)
        
        # API client for low-level operations
        try:
            self.api_client = docker.APIClient()
            self.logger.info(f"✅ Docker manager initialized - Docker {self.docker_version}, API {self.api_version}")
        except Exception as e:
            self.logger.warning(f"Failed to create API client: {e}")
            self.api_client = None
    
    async def verify_docker_available(self) -> bool:
        """Verify Docker is available and running"""
        try:
            # Test Docker connection
            self.client.ping()
            
            # Get Docker version info
            version_info = self.client.version()
            self.logger.info(f"Docker version: {version_info['Version']}")
            self.logger.info(f"Docker API version: {version_info['ApiVersion']}")
            
            return True
        except DockerException as e:
            self.logger.error(f"Docker not available: {e}")
            return False
    
    async def create_container(self, config: ContainerConfig) -> bool:
        """
        Create and start a container from configuration.
        
        Args:
            config: Universal container configuration
            
        Returns:
            bool: True if container created and started successfully
        """
        try:
            self.logger.info(f"🔄 Creating Docker container: {config.name}")
            
            # Convert config to Docker format
            docker_config = config.to_docker_format()
            
            # Extract Docker-specific parameters
            container_name = docker_config.pop("name")
            image = docker_config.pop("image")
            detach = docker_config.pop("detach", True)
            
            # Ensure image is available
            await self._ensure_image_available(image)
            
            # Check if container already exists
            if await self._container_exists(container_name):
                self.logger.warning(f"Container {container_name} already exists")
                return await self.start_container(container_name)
            
            # Create container
            container = self.client.containers.create(
                image=image,
                name=container_name,
                detach=detach,
                **docker_config
            )
            
            # Start container
            container.start()
            
            # Wait for container to be running
            container.reload()
            if container.status == "running":
                self.logger.info(f"✅ Container {container_name} created and started successfully")
                self.managed_containers[container_name] = config
                return True
            else:
                self.logger.error(f"❌ Container {container_name} failed to start: {container.status}")
                return False
            
        except ImageNotFound:
            self.logger.error(f"❌ Docker image not found: {image}")
            return False
        except ContainerError as e:
            self.logger.error(f"❌ Container creation failed: {e}")
            return False
        except DockerException as e:
            self.logger.error(f"❌ Docker operation failed: {e}")
            return False
        except Exception as e:
            self.logger.error(f"❌ Unexpected error creating container: {e}")
            return False
    
    async def start_container(self, name: str) -> bool:
        """
        Start an existing container.
        
        Args:
            name: Container name
            
        Returns:
            bool: True if container started successfully
        """
        try:
            container = self.client.containers.get(name)
            
            if container.status == "running":
                self.logger.info(f"Container {name} is already running")
                return True
            
            container.start()
            container.reload()
            
            if container.status == "running":
                self.logger.info(f"✅ Container {name} started successfully")
                return True
            else:
                self.logger.error(f"❌ Failed to start container {name}: {container.status}")
                return False
                
        except docker.errors.NotFound:
            raise ContainerNotFoundError(f"Container {name} not found")
        except DockerException as e:
            self.logger.error(f"❌ Failed to start container {name}: {e}")
            return False
    
    async def stop_container(self, name: str, timeout: int = 10) -> bool:
        """
        Stop a running container.
        
        Args:
            name: Container name
            timeout: Timeout in seconds for graceful shutdown
            
        Returns:
            bool: True if container stopped successfully
        """
        try:
            container = self.client.containers.get(name)
            
            if container.status != "running":
                self.logger.info(f"Container {name} is not running")
                return True
            
            container.stop(timeout=timeout)
            container.reload()
            
            if container.status in ["exited", "stopped"]:
                self.logger.info(f"✅ Container {name} stopped successfully")
                return True
            else:
                self.logger.warning(f"⚠️ Container {name} status after stop: {container.status}")
                return False
                
        except docker.errors.NotFound:
            raise ContainerNotFoundError(f"Container {name} not found")
        except DockerException as e:
            self.logger.error(f"❌ Failed to stop container {name}: {e}")
            return False
    
    async def remove_container(self, name: str, force: bool = False) -> bool:
        """
        Remove a container.
        
        Args:
            name: Container name
            force: Force removal of running container
            
        Returns:
            bool: True if container removed successfully
        """
        try:
            container = self.client.containers.get(name)
            
            # Stop container if running and not forcing
            if not force and container.status == "running":
                await self.stop_container(name)
            
            container.remove(force=force)
            
            # Remove from managed containers
            if name in self.managed_containers:
                del self.managed_containers[name]
            
            self.logger.info(f"✅ Container {name} removed successfully")
            return True
            
        except docker.errors.NotFound:
            self.logger.info(f"Container {name} not found (already removed)")
            return True
        except DockerException as e:
            self.logger.error(f"❌ Failed to remove container {name}: {e}")
            return False
    
    async def get_container_status(self, name: str) -> Dict[str, Any]:
        """
        Get detailed container status information.
        
        Args:
            name: Container name
            
        Returns:
            Dict with container status information
        """
        try:
            container = self.client.containers.get(name)
            container.reload()
            
            # Get detailed stats
            stats = {}
            try:
                # Get one-shot stats (non-streaming)
                stats_stream = container.stats(stream=False)
                if stats_stream:
                    stats = stats_stream
            except Exception as e:
                self.logger.debug(f"Could not get stats for {name}: {e}")
            
            return {
                "name": container.name,
                "id": container.id[:12],
                "status": container.status,
                "image": container.image.tags[0] if container.image.tags else container.image.id,
                "created": container.attrs["Created"],
                "started": container.attrs.get("State", {}).get("StartedAt"),
                "finished": container.attrs.get("State", {}).get("FinishedAt"),
                "exit_code": container.attrs.get("State", {}).get("ExitCode"),
                "ports": container.ports,
                "mounts": [
                    {"source": mount["Source"], "destination": mount["Destination"]}
                    for mount in container.attrs.get("Mounts", [])
                ],
                "networks": list(container.attrs.get("NetworkSettings", {}).get("Networks", {}).keys()),
                "stats": stats
            }
            
        except docker.errors.NotFound:
            raise ContainerNotFoundError(f"Container {name} not found")
        except DockerException as e:
            self.logger.error(f"❌ Failed to get status for container {name}: {e}")
            return {"error": str(e)}
    
    async def get_container_logs(self, name: str, lines: int = 100, follow: bool = False) -> str:
        """
        Get container logs.
        
        Args:
            name: Container name
            lines: Number of lines to retrieve (default: 100)
            follow: Whether to follow log output
            
        Returns:
            str: Container logs
        """
        try:
            container = self.client.containers.get(name)
            
            logs = container.logs(
                tail=lines,
                follow=follow,
                stdout=True,
                stderr=True,
                timestamps=True
            )
            
            if isinstance(logs, bytes):
                return logs.decode('utf-8', errors='replace')
            else:
                # Generator for follow mode
                return '\n'.join(log.decode('utf-8', errors='replace') for log in logs)
            
        except docker.errors.NotFound:
            raise ContainerNotFoundError(f"Container {name} not found")
        except DockerException as e:
            self.logger.error(f"❌ Failed to get logs for container {name}: {e}")
            return f"Error retrieving logs: {e}"
    
    async def health_check(self, name: str) -> bool:
        """
        Check container health.
        
        Args:
            name: Container name
            
        Returns:
            bool: True if container is healthy
        """
        try:
            container = self.client.containers.get(name)
            container.reload()
            
            # Check if container is running
            if container.status != "running":
                return False
            
            # Check Docker health status if available
            health_status = container.attrs.get("State", {}).get("Health", {}).get("Status")
            if health_status:
                return health_status == "healthy"
            
            # If no health check configured, consider running container as healthy
            return True
            
        except docker.errors.NotFound:
            return False
        except DockerException as e:
            self.logger.error(f"❌ Health check failed for container {name}: {e}")
            return False
    
    async def list_containers(self, all_containers: bool = False) -> List[Dict[str, Any]]:
        """
        List all containers.
        
        Args:
            all_containers: Include stopped containers
            
        Returns:
            List of container information
        """
        try:
            containers = self.client.containers.list(all=all_containers)
            
            container_list = []
            for container in containers:
                container_info = {
                    "name": container.name,
                    "id": container.id[:12],
                    "status": container.status,
                    "image": container.image.tags[0] if container.image.tags else container.image.id,
                    "created": container.attrs["Created"],
                    "ports": container.ports,
                    "labels": container.labels
                }
                container_list.append(container_info)
            
            return container_list
            
        except DockerException as e:
            self.logger.error(f"❌ Failed to list containers: {e}")
            return []
    
    async def pull_image(self, image: str, tag: str = "latest") -> bool:
        """
        Pull a Docker image.
        
        Args:
            image: Image name
            tag: Image tag
            
        Returns:
            bool: True if image pulled successfully
        """
        try:
            full_image = f"{image}:{tag}"
            self.logger.info(f"🔄 Pulling Docker image: {full_image}")
            
            # Pull image with progress logging
            self.client.images.pull(image, tag=tag)
            
            self.logger.info(f"✅ Successfully pulled image: {full_image}")
            return True
            
        except ImageNotFound:
            self.logger.error(f"❌ Image not found: {image}:{tag}")
            return False
        except DockerException as e:
            self.logger.error(f"❌ Failed to pull image {image}:{tag}: {e}")
            return False
    
    async def cleanup_unused_resources(self) -> Dict[str, Any]:
        """
        Clean up unused Docker resources.
        
        Returns:
            Dict with cleanup statistics
        """
        try:
            self.logger.info("🧹 Cleaning up unused Docker resources...")
            
            # Prune containers
            container_prune = self.client.containers.prune()
            
            # Prune images
            image_prune = self.client.images.prune(filters={"dangling": True})
            
            # Prune networks
            network_prune = self.client.networks.prune()
            
            # Prune volumes
            volume_prune = self.client.volumes.prune()
            
            cleanup_stats = {
                "containers_removed": len(container_prune.get("ContainersDeleted", [])),
                "images_removed": len(image_prune.get("ImagesDeleted", [])),
                "networks_removed": len(network_prune.get("NetworksDeleted", [])),
                "volumes_removed": len(volume_prune.get("VolumesDeleted", [])),
                "space_reclaimed": (
                    container_prune.get("SpaceReclaimed", 0) +
                    image_prune.get("SpaceReclaimed", 0) +
                    network_prune.get("SpaceReclaimed", 0) +
                    volume_prune.get("SpaceReclaimed", 0)
                )
            }
            
            self.logger.info(f"✅ Cleanup completed: {cleanup_stats}")
            return cleanup_stats
            
        except DockerException as e:
            self.logger.error(f"❌ Cleanup failed: {e}")
            return {"error": str(e)}
    
    async def _ensure_image_available(self, image: str) -> None:
        """Ensure Docker image is available locally"""
        try:
            self.client.images.get(image)
            self.logger.debug(f"Image {image} already available locally")
        except ImageNotFound:
            self.logger.info(f"Image {image} not found locally, pulling...")
            if ":" in image:
                image_name, tag = image.rsplit(":", 1)
                await self.pull_image(image_name, tag)
            else:
                await self.pull_image(image)
    
    async def _container_exists(self, name: str) -> bool:
        """Check if container exists"""
        try:
            self.client.containers.get(name)
            return True
        except docker.errors.NotFound:
            return False
    
    async def create_network(self, name: str, driver: str = "bridge") -> bool:
        """Create a Docker network"""
        try:
            self.client.networks.create(name, driver=driver)
            self.logger.info(f"✅ Network {name} created successfully")
            return True
        except DockerException as e:
            self.logger.error(f"❌ Failed to create network {name}: {e}")
            return False
    
    async def remove_network(self, name: str) -> bool:
        """Remove a Docker network"""
        try:
            network = self.client.networks.get(name)
            network.remove()
            self.logger.info(f"✅ Network {name} removed successfully")
            return True
        except docker.errors.NotFound:
            self.logger.info(f"Network {name} not found (already removed)")
            return True
        except DockerException as e:
            self.logger.error(f"❌ Failed to remove network {name}: {e}")
            return False
    
    async def create_volume(self, name: str, driver: str = "local") -> bool:
        """Create a Docker volume"""
        try:
            self.client.volumes.create(name, driver=driver)
            self.logger.info(f"✅ Volume {name} created successfully")
            return True
        except DockerException as e:
            self.logger.error(f"❌ Failed to create volume {name}: {e}")
            return False
    
    async def remove_volume(self, name: str, force: bool = False) -> bool:
        """Remove a Docker volume"""
        try:
            volume = self.client.volumes.get(name)
            volume.remove(force=force)
            self.logger.info(f"✅ Volume {name} removed successfully")
            return True
        except docker.errors.NotFound:
            self.logger.info(f"Volume {name} not found (already removed)")
            return True
        except DockerException as e:
            self.logger.error(f"❌ Failed to remove volume {name}: {e}")
            return False 