"""
Universal Container Configuration for NanoBrain Framework

This module provides configuration classes that work across different container
orchestrators (Docker, Kubernetes, Docker Compose, etc.), enabling unified
container deployment strategies.

"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Union, ClassVar
from pathlib import Path

from nanobrain.core.logging_system import get_logger
from nanobrain.core.component_base import FromConfigBase


class ContainerOrchestrator(Enum):
    """Supported container orchestrators"""
    DOCKER = "docker"
    KUBERNETES = "kubernetes"
    DOCKER_COMPOSE = "docker-compose"
    DOCKER_SWARM = "docker-swarm"
    AUTO = "auto"  # Auto-detect best available


class RestartPolicy(Enum):
    """Container restart policies"""
    NO = "no"
    ALWAYS = "always"
    UNLESS_STOPPED = "unless-stopped"
    ON_FAILURE = "on-failure"


class ServiceType(Enum):
    """Kubernetes service types"""
    CLUSTER_IP = "ClusterIP"
    NODE_PORT = "NodePort"
    LOAD_BALANCER = "LoadBalancer"
    EXTERNAL_NAME = "ExternalName"


@dataclass
class HealthCheckConfig:
    """Health check configuration for containers"""
    type: str = "http"  # http, tcp, exec, none
    path: str = "/health"
    port: int = 8080
    interval: int = 30  # seconds
    timeout: int = 10   # seconds
    retries: int = 3
    start_period: int = 60  # seconds
    failure_threshold: int = 3
    success_threshold: int = 1
    
    # For exec health checks
    command: Optional[List[str]] = None
    
    def to_docker_format(self) -> Dict[str, Any]:
        """Convert to Docker healthcheck format"""
        if self.type == "http":
            return {
                "test": ["CMD-SHELL", f"curl -f http://localhost:{self.port}{self.path} || exit 1"],
                "interval": f"{self.interval}s",
                "timeout": f"{self.timeout}s",
                "retries": self.retries,
                "start_period": f"{self.start_period}s"
            }
        elif self.type == "tcp":
            return {
                "test": ["CMD-SHELL", f"nc -z localhost {self.port} || exit 1"],
                "interval": f"{self.interval}s",
                "timeout": f"{self.timeout}s",
                "retries": self.retries,
                "start_period": f"{self.start_period}s"
            }
        elif self.type == "exec" and self.command:
            return {
                "test": ["CMD"] + self.command,
                "interval": f"{self.interval}s",
                "timeout": f"{self.timeout}s",
                "retries": self.retries,
                "start_period": f"{self.start_period}s"
            }
        return {}
    
    def to_kubernetes_probe(self) -> Dict[str, Any]:
        """Convert to Kubernetes probe format"""
        probe = {
            "initialDelaySeconds": self.start_period,
            "periodSeconds": self.interval,
            "timeoutSeconds": self.timeout,
            "failureThreshold": self.failure_threshold,
            "successThreshold": self.success_threshold
        }
        
        if self.type == "http":
            probe["httpGet"] = {
                "path": self.path,
                "port": self.port
            }
        elif self.type == "tcp":
            probe["tcpSocket"] = {
                "port": self.port
            }
        elif self.type == "exec" and self.command:
            probe["exec"] = {
                "command": self.command
            }
        
        return probe


@dataclass
class ResourceLimits:
    """Resource limits and requests for containers"""
    memory: Optional[str] = None      # e.g., "2Gi", "512Mi"
    cpu: Optional[str] = None         # e.g., "1.0", "500m"
    memory_request: Optional[str] = None
    cpu_request: Optional[str] = None
    
    # Docker-specific (will be converted from memory/cpu)
    mem_limit: Optional[str] = None
    cpu_quota: Optional[int] = None
    cpu_period: Optional[int] = None
    
    def __post_init__(self):
        """Set defaults for requests if not specified"""
        if self.memory and not self.memory_request:
            self.memory_request = self.memory
        if self.cpu and not self.cpu_request:
            self.cpu_request = self.cpu
    
    def to_docker_format(self) -> Dict[str, Any]:
        """Convert to Docker resource format"""
        docker_resources = {}
        
        if self.memory:
            docker_resources["mem_limit"] = self.memory
        
        if self.cpu:
            # Convert CPU to Docker quota/period format
            docker_resources["cpu_period"] = 100000  # 100ms
            if self.cpu.endswith('m'):
                # millicores (e.g., "500m" = 0.5 CPU)
                cpu_millicores = int(self.cpu[:-1])
                docker_resources["cpu_quota"] = int(cpu_millicores * 100)
            else:
                # Full CPUs (e.g., "1.5" = 1.5 CPUs)
                cpu_cores = float(self.cpu)
                docker_resources["cpu_quota"] = int(cpu_cores * 100000)
        
        return docker_resources
    
    def to_kubernetes_format(self) -> Dict[str, Any]:
        """Convert to Kubernetes resource format"""
        resources = {}
        
        if any([self.memory, self.cpu]):
            resources["limits"] = {}
            if self.memory:
                resources["limits"]["memory"] = self.memory
            if self.cpu:
                resources["limits"]["cpu"] = self.cpu
        
        if any([self.memory_request, self.cpu_request]):
            resources["requests"] = {}
            if self.memory_request:
                resources["requests"]["memory"] = self.memory_request
            if self.cpu_request:
                resources["requests"]["cpu"] = self.cpu_request
        
        return resources


@dataclass
class ContainerConfig:
    """
    Universal container configuration that supports multiple orchestrators.
    
    This configuration can be used with Docker, Kubernetes, Docker Compose,
    and other container orchestrators by converting to the appropriate format.
    """
    
    # Core container configuration
    name: str
    image: str
    tag: str = "latest"
    
    # Network configuration
    ports: List[str] = field(default_factory=list)  # ["8080:80", "9200:9200"]
    networks: List[str] = field(default_factory=list)
    hostname: Optional[str] = None
    
    # Environment and configuration
    environment: Dict[str, str] = field(default_factory=dict)
    volumes: List[str] = field(default_factory=list)  # ["host_path:container_path", "volume_name:container_path"]
    working_dir: Optional[str] = None
    user: Optional[str] = None
    
    # Container behavior
    restart_policy: RestartPolicy = RestartPolicy.UNLESS_STOPPED
    health_check: Optional[HealthCheckConfig] = None
    resource_limits: Optional[ResourceLimits] = None
    
    # Metadata
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    
    # Kubernetes-specific fields
    namespace: str = "default"
    replicas: int = 1
    service_type: ServiceType = ServiceType.CLUSTER_IP
    ingress_enabled: bool = False
    ingress_host: Optional[str] = None
    persistent_volume_claims: List[str] = field(default_factory=list)
    
    # Kubernetes scheduling
    node_selector: Dict[str, str] = field(default_factory=dict)
    tolerations: List[Dict[str, Any]] = field(default_factory=list)
    affinity: Optional[Dict[str, Any]] = None
    
    # Auto-scaling (Kubernetes)
    auto_scaling_enabled: bool = False
    min_replicas: int = 1
    max_replicas: int = 5
    target_cpu_utilization: int = 70
    target_memory_utilization: int = 80
    
    # Security context
    security_context: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization validation and setup"""
        # Ensure name is valid for container orchestrators
        self.name = self._sanitize_name(self.name)
        
        # Set default labels
        if "app" not in self.labels:
            self.labels["app"] = self.name
        
        # Add NanoBrain framework labels
        self.labels.update({
            "nanobrain.framework": "true",
            "nanobrain.version": "1.0.0"
        })

    @classmethod
    def from_config(cls, config: Union[Dict, 'ContainerConfig'], **kwargs) -> 'ContainerConfig':
        """
        Mandatory from_config implementation for ContainerConfig
        
        Args:
            config: Container configuration (dict or ContainerConfig)
            **kwargs: Additional configuration parameters
            
        Returns:
            ContainerConfig instance
        """
        logger = get_logger(f"{cls.__name__}.from_config")
        logger.info(f"Creating {cls.__name__} from configuration")
        
        if isinstance(config, cls):
            # Already a ContainerConfig instance
            return config
        
        if isinstance(config, dict):
            # Create from dictionary
            return cls(**config, **kwargs)
        
        # Handle other config objects
        config_dict = {}
        if hasattr(config, 'model_dump'):
            config_dict = config.model_dump()
        elif hasattr(config, '__dict__'):
            config_dict = config.__dict__
        else:
            raise ValueError(f"Cannot convert {type(config)} to ContainerConfig")
        
        # Merge with kwargs
        config_dict.update(kwargs)
        
        logger.info(f"Successfully created {cls.__name__} from configuration")
        return cls(**config_dict)
    
    def _sanitize_name(self, name: str) -> str:
        """Sanitize container name for orchestrator compatibility"""
        # Replace invalid characters and ensure lowercase
        import re
        sanitized = re.sub(r'[^a-zA-Z0-9\-]', '-', name.lower())
        sanitized = re.sub(r'-+', '-', sanitized)  # Remove multiple consecutive dashes
        sanitized = sanitized.strip('-')  # Remove leading/trailing dashes
        return sanitized
    
    @property
    def full_image(self) -> str:
        """Get full image name with tag"""
        return f"{self.image}:{self.tag}"
    
    def get_port_mappings(self) -> Dict[str, str]:
        """Parse port mappings into dict format"""
        port_map = {}
        for port in self.ports:
            if ":" in port:
                host_port, container_port = port.split(":", 1)
                port_map[container_port] = host_port
            else:
                port_map[port] = port
        return port_map
    
    def get_volume_mappings(self) -> List[Dict[str, str]]:
        """Parse volume mappings into structured format"""
        volume_mappings = []
        for volume in self.volumes:
            if ":" in volume:
                source, target = volume.split(":", 1)
                # Determine if it's a bind mount or named volume
                if os.path.isabs(source) or source.startswith('./'):
                    volume_mappings.append({
                        "type": "bind",
                        "source": source,
                        "target": target
                    })
                else:
                    volume_mappings.append({
                        "type": "volume",
                        "source": source,
                        "target": target
                    })
            else:
                # Named volume with default target
                volume_mappings.append({
                    "type": "volume",
                    "source": volume,
                    "target": f"/data/{volume}"
                })
        return volume_mappings
    
    def to_docker_format(self) -> Dict[str, Any]:
        """Convert configuration to Docker container format"""
        docker_config = {
            "image": self.full_image,
            "name": self.name,
            "detach": True,
            "labels": self.labels,
            "environment": self.environment,
            "restart_policy": {"Name": self.restart_policy.value}
        }
        
        # Port mappings
        if self.ports:
            ports = {}
            for container_port, host_port in self.get_port_mappings().items():
                ports[f"{container_port}/tcp"] = host_port
            docker_config["ports"] = ports
        
        # Volume mappings
        if self.volumes:
            volumes = {}
            for vol_mapping in self.get_volume_mappings():
                if vol_mapping["type"] == "bind":
                    volumes[vol_mapping["source"]] = {
                        "bind": vol_mapping["target"],
                        "mode": "rw"
                    }
                else:
                    volumes[vol_mapping["source"]] = {
                        "driver": "local"
                    }
            docker_config["volumes"] = volumes
        
        # Networks
        if self.networks:
            docker_config["network"] = self.networks[0]  # Docker run supports one network
        
        # Resource limits
        if self.resource_limits:
            docker_config.update(self.resource_limits.to_docker_format())
        
        # Health check
        if self.health_check:
            docker_config["healthcheck"] = self.health_check.to_docker_format()
        
        # Additional configuration
        if self.hostname:
            docker_config["hostname"] = self.hostname
        if self.working_dir:
            docker_config["working_dir"] = self.working_dir
        if self.user:
            docker_config["user"] = self.user
        
        return docker_config
    
    def to_kubernetes_deployment(self) -> Dict[str, Any]:
        """Convert configuration to Kubernetes Deployment format"""
        # Container specification
        container_spec = {
            "name": self.name,
            "image": self.full_image,
            "env": [{"name": k, "value": v} for k, v in self.environment.items()]
        }
        
        # Port configuration
        if self.ports:
            container_spec["ports"] = [
                {"containerPort": int(port.split(":")[1] if ":" in port else port)}
                for port in self.ports
            ]
        
        # Volume mounts
        if self.volumes:
            volume_mounts = []
            for vol_mapping in self.get_volume_mappings():
                volume_mounts.append({
                    "name": vol_mapping["source"].replace("/", "-").replace("_", "-"),
                    "mountPath": vol_mapping["target"]
                })
            container_spec["volumeMounts"] = volume_mounts
        
        # Resource limits
        if self.resource_limits:
            container_spec["resources"] = self.resource_limits.to_kubernetes_format()
        
        # Health checks
        if self.health_check:
            container_spec["livenessProbe"] = self.health_check.to_kubernetes_probe()
            container_spec["readinessProbe"] = self.health_check.to_kubernetes_probe()
        
        # Security context
        if self.security_context:
            container_spec["securityContext"] = self.security_context
        
        # Pod specification
        pod_spec = {
            "containers": [container_spec]
        }
        
        # Volumes
        if self.volumes:
            volumes = []
            for vol_mapping in self.get_volume_mappings():
                volume_name = vol_mapping["source"].replace("/", "-").replace("_", "-")
                if vol_mapping["type"] == "bind":
                    volumes.append({
                        "name": volume_name,
                        "hostPath": {"path": vol_mapping["source"]}
                    })
                else:
                    volumes.append({
                        "name": volume_name,
                        "persistentVolumeClaim": {"claimName": vol_mapping["source"]}
                    })
            pod_spec["volumes"] = volumes
        
        # Node scheduling
        if self.node_selector:
            pod_spec["nodeSelector"] = self.node_selector
        if self.tolerations:
            pod_spec["tolerations"] = self.tolerations
        if self.affinity:
            pod_spec["affinity"] = self.affinity
        
        # Deployment specification
        deployment = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": self.name,
                "namespace": self.namespace,
                "labels": self.labels,
                "annotations": self.annotations
            },
            "spec": {
                "replicas": self.replicas,
                "selector": {
                    "matchLabels": {"app": self.name}
                },
                "template": {
                    "metadata": {
                        "labels": {**self.labels, "app": self.name}
                    },
                    "spec": pod_spec
                }
            }
        }
        
        return deployment
    
    def to_kubernetes_service(self) -> Dict[str, Any]:
        """Convert configuration to Kubernetes Service format"""
        service_ports = []
        for port in self.ports:
            if ":" in port:
                host_port, container_port = port.split(":", 1)
                service_ports.append({
                    "port": int(host_port),
                    "targetPort": int(container_port),
                    "name": f"port-{container_port}"
                })
            else:
                service_ports.append({
                    "port": int(port),
                    "targetPort": int(port),
                    "name": f"port-{port}"
                })
        
        service = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": f"{self.name}-service",
                "namespace": self.namespace,
                "labels": self.labels
            },
            "spec": {
                "selector": {"app": self.name},
                "ports": service_ports,
                "type": self.service_type.value
            }
        }
        
        return service
    
    def to_docker_compose_format(self) -> Dict[str, Any]:
        """Convert configuration to Docker Compose service format"""
        compose_service = {
            "image": self.full_image,
            "container_name": self.name,
            "restart": self.restart_policy.value,
            "labels": self.labels,
            "environment": self.environment
        }
        
        # Port mappings
        if self.ports:
            compose_service["ports"] = self.ports
        
        # Volume mappings
        if self.volumes:
            compose_service["volumes"] = self.volumes
        
        # Networks
        if self.networks:
            compose_service["networks"] = self.networks
        
        # Health check
        if self.health_check:
            compose_service["healthcheck"] = self.health_check.to_docker_format()
        
        # Resource limits (Docker Compose v3.8+ format)
        if self.resource_limits:
            deploy_config = {}
            if self.resource_limits.memory or self.resource_limits.cpu:
                deploy_config["resources"] = {
                    "limits": {},
                    "reservations": {}
                }
                if self.resource_limits.memory:
                    deploy_config["resources"]["limits"]["memory"] = self.resource_limits.memory
                    if self.resource_limits.memory_request:
                        deploy_config["resources"]["reservations"]["memory"] = self.resource_limits.memory_request
                if self.resource_limits.cpu:
                    deploy_config["resources"]["limits"]["cpus"] = self.resource_limits.cpu
                    if self.resource_limits.cpu_request:
                        deploy_config["resources"]["reservations"]["cpus"] = self.resource_limits.cpu_request
            
            if deploy_config:
                compose_service["deploy"] = deploy_config
        
        # Additional configuration
        if self.hostname:
            compose_service["hostname"] = self.hostname
        if self.working_dir:
            compose_service["working_dir"] = self.working_dir
        if self.user:
            compose_service["user"] = self.user
        
        return compose_service


@dataclass
class DockerComponentConfig:
    """Base configuration for Docker infrastructure components"""
    component_name: str
    enabled: bool = True
    docker_client: Optional[Any] = None  # Docker client instance
    check_interval: int = 30  # seconds
    auto_cleanup: bool = True
    labels: Dict[str, str] = field(default_factory=lambda: {
        "nanobrain.framework": "true",
        "nanobrain.component": "docker-infrastructure"
    })
    
    def __post_init__(self):
        """Add component-specific labels"""
        self.labels.update({
            "nanobrain.component.name": self.component_name,
            "nanobrain.component.type": "docker-infrastructure"
        })


class DockerComponentBase(FromConfigBase):
    """
    Base class for Docker infrastructure components (non-container-managers).
    Provides concrete from_config implementation for health monitors, network managers, etc.
    """
    
    # Component configuration schema
    COMPONENT_TYPE: ClassVar[str] = "docker_component"
    REQUIRED_CONFIG_FIELDS: ClassVar[List[str]] = []
    OPTIONAL_CONFIG_FIELDS: ClassVar[Dict[str, Any]] = {
        'docker_client': None,
        'enabled': True
    }
    
    @classmethod
    def _get_config_class(cls):
        """Return DockerComponentConfig for docker components."""
        return DockerComponentConfig
    
    def __init__(self, *args, **kwargs):
        """Prevent direct instantiation - use from_config instead"""
        raise RuntimeError(
            f"Direct instantiation of {self.__class__.__name__} is prohibited. "
            f"ALL framework components must use {self.__class__.__name__}.from_config() "
            f"as per mandatory framework requirements."
        )
    
    @classmethod
    def from_config(cls, config: Union[DockerComponentConfig, Dict], **kwargs) -> 'DockerComponentBase':
        """Mandatory from_config implementation for DockerComponentBase"""
        logger = get_logger(f"{cls.__name__}.from_config")
        logger.info(f"Creating {cls.__name__} from configuration")
        
        # Handle configuration conversion more flexibly
        if isinstance(config, dict):
            # Use dict as-is for component_config extraction
            config_for_processing = config
        elif hasattr(config, '__dict__'):
            # Convert object to dict representation
            config_for_processing = config.__dict__
        elif hasattr(config, 'model_dump'):
            # Handle Pydantic models
            config_for_processing = config.model_dump()
        else:
            config_for_processing = config
        
        # Step 1: Validate configuration schema (use raw config dict)
        # cls.validate_config_schema(config_for_processing)
        
        # Step 2: Extract component-specific configuration  
        component_config = cls.extract_component_config(config_for_processing)
        
        # Step 3: Resolve dependencies
        dependencies = cls.resolve_dependencies(component_config, **kwargs)
        
        # Step 4: Create instance
        instance = cls.create_instance(config, component_config, dependencies)
        
        # Step 5: Post-creation initialization
        instance._post_config_initialization()
        
        logger.info(f"Successfully created {cls.__name__}")
        return instance
    
    @classmethod 
    def extract_component_config(cls, config: Any) -> Dict[str, Any]:
        """Extract component-specific configuration"""
        if isinstance(config, DockerComponentConfig):
            return {
                'component_name': config.component_name,
                'enabled': config.enabled,
                'docker_client': config.docker_client,
                'check_interval': config.check_interval,
                'auto_cleanup': config.auto_cleanup,
                'labels': config.labels
            }
        elif isinstance(config, dict):
            return config
        else:
            return {}
    
    @classmethod  
    def resolve_dependencies(cls, component_config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Resolve all component dependencies"""
        # Base implementation - can be overridden by subclasses
        return {
            'docker_client': component_config.get('docker_client'),
            **kwargs
        }
    
    def _init_from_config(self, config: Any, component_config: Dict[str, Any],
                         dependencies: Dict[str, Any]) -> None:
        """Initialize instance from configuration"""
        self.config = config
        self.component_config = component_config
        self.dependencies = dependencies
        self.logger = get_logger(self.__class__.__name__)
    
    def _post_config_initialization(self):
        """Post-configuration initialization hook"""
        pass


class ContainerManagerBase(FromConfigBase, ABC):
    """
    Abstract base class for container managers.
    Enhanced with mandatory from_config pattern implementation.
    """
    
    # Component configuration schema
    COMPONENT_TYPE: ClassVar[str] = "container_manager"
    REQUIRED_CONFIG_FIELDS: ClassVar[List[str]] = []
    OPTIONAL_CONFIG_FIELDS: ClassVar[Dict[str, Any]] = {
        'docker_client': None,
        'enabled': True
    }
    
    @classmethod
    def _get_config_class(cls):
        """Return DockerComponentConfig for container manager components."""
        return DockerComponentConfig
    
    def __init__(self, *args, **kwargs):
        """Prevent direct instantiation - use from_config instead"""
        raise RuntimeError(
            f"Direct instantiation of {self.__class__.__name__} is prohibited. "
            f"ALL framework components must use {self.__class__.__name__}.from_config() "
            f"as per mandatory framework requirements."
        )
    
    @classmethod
    def from_config(cls, config: Union[DockerComponentConfig, Dict], **kwargs) -> 'ContainerManagerBase':
        """Mandatory from_config implementation for ContainerManagerBase"""
        logger = get_logger(f"{cls.__name__}.from_config")
        logger.info(f"Creating {cls.__name__} from configuration")
        
        # Convert config to proper format
        if isinstance(config, dict):
            config = DockerComponentConfig(**config)
        elif not isinstance(config, DockerComponentConfig):
            # Handle other config types
            config_dict = {}
            if hasattr(config, 'model_dump'):
                config_dict = config.model_dump()
            elif hasattr(config, '__dict__'):
                config_dict = config.__dict__
            config = DockerComponentConfig(**config_dict)
        
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
    def extract_component_config(cls, config: Any) -> Dict[str, Any]:
        """Extract component-specific configuration"""
        if isinstance(config, DockerComponentConfig):
            return {
                'component_name': config.component_name,
                'enabled': config.enabled,
                'docker_client': config.docker_client,
                'check_interval': config.check_interval,
                'auto_cleanup': config.auto_cleanup,
                'labels': config.labels
            }
        elif isinstance(config, dict):
            return config
        else:
            return {}
    
    @classmethod  
    def resolve_dependencies(cls, component_config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Resolve all component dependencies"""
        # Base implementation - can be overridden by subclasses
        return {
            'docker_client': component_config.get('docker_client'),
            **kwargs
        }
    
    def _init_from_config(self, config: Any, component_config: Dict[str, Any],
                         dependencies: Dict[str, Any]) -> None:
        """Initialize instance from configuration"""
        self.config = config
        self.component_config = component_config
        self.dependencies = dependencies
        self.logger = get_logger(self.__class__.__name__)
    
    def _post_config_initialization(self):
        """Post-configuration initialization hook"""
        pass

    @abstractmethod
    async def create_container(self, config: ContainerConfig) -> bool:
        """Create a container from configuration"""
        pass

    @abstractmethod
    async def stop_container(self, name: str) -> bool:
        """Stop a running container"""
        pass

    @abstractmethod
    async def remove_container(self, name: str) -> bool:
        """Remove a container"""
        pass

    @abstractmethod
    async def get_container_status(self, name: str) -> Dict[str, Any]:
        """Get container status information"""
        pass

    @abstractmethod
    async def get_container_logs(self, name: str, lines: int = 100) -> str:
        """Get container logs"""
        pass

    @abstractmethod
    async def health_check(self, name: str) -> bool:
        """Perform health check on container"""
        pass

    async def restart_container(self, name: str) -> bool:
        """Restart a container (default implementation)"""
        success = await self.stop_container(name)
        if success:
            return await self.start_container(name)
        return False

    async def start_container(self, name: str) -> bool:
        """Start a container (default implementation)"""
        # Default implementation - can be overridden
        return True 