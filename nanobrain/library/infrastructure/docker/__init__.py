"""
Docker Infrastructure Module for NanoBrain Framework

This module provides Docker container management capabilities for tools and services
within the NanoBrain framework, enabling:

- Container lifecycle management (create, start, stop, remove)
- Health monitoring and verification
- Volume and network management
- Resource constraints and limits
- Production-ready container configuration
- Integration with NanoBrain logging and monitoring

ALL COMPONENTS IN THIS MODULE FOLLOW THE MANDATORY FROM_CONFIG PATTERN:
- Use ClassName.from_config(config) instead of ClassName(args)
- Direct instantiation via __init__ is prohibited
- Provides unified configuration management and dependency injection

Components:
- DockerManager: Core Docker operations and container management
- ContainerConfig: Configuration classes for Docker containers
- DockerHealthMonitor: Health checking and monitoring for containers
- DockerNetworkManager: Network isolation and management
- DockerVolumeManager: Persistent volume management

Configuration Classes:
- DockerComponentConfig: Base configuration for all Docker components
- DockerHealthMonitorConfig: Health monitor specific configuration
- DockerNetworkManagerConfig: Network manager specific configuration
- DockerVolumeManagerConfig: Volume manager specific configuration

Example Usage:
    # Create Docker manager
    docker_config = DockerComponentConfig(component_name="my_docker_manager")
    docker_manager = DockerManager.from_config(docker_config)
    
    # Create health monitor
    health_config = DockerHealthMonitorConfig(
        docker_manager=docker_manager,
        check_interval=60
    )
    health_monitor = DockerHealthMonitor.from_config(health_config)
    
    # Create container configuration
    container_config = ContainerConfig.from_config({
        "name": "elasticsearch",
        "image": "elasticsearch",
        "tag": "8.14.0",
        "ports": ["9200:9200"],
        "environment": {"discovery.type": "single-node"}
    })
"""

from .container_config import (
    ContainerConfig, 
    ContainerOrchestrator,
    DockerComponentConfig,
    DockerComponentBase,
    HealthCheckConfig,
    ResourceLimits,
    RestartPolicy,
    ServiceType
)
from .docker_manager import DockerManager
from .docker_health_monitor import (
    DockerHealthMonitor, 
    DockerHealthMonitorConfig,
    HealthStatus,
    HealthCheckResult,
    ResourceMetrics
)
from .docker_network_manager import (
    DockerNetworkManager,
    DockerNetworkManagerConfig,
    NetworkConfig
)
from .docker_volume_manager import (
    DockerVolumeManager,
    DockerVolumeManagerConfig,
    VolumeConfig
)

__all__ = [
    # Core classes (MANDATORY: Use .from_config() for instantiation)
    'DockerManager',
    'DockerHealthMonitor',
    'DockerNetworkManager',
    'DockerVolumeManager',
    
    # Configuration classes
    'ContainerConfig',
    'DockerComponentConfig',
    'DockerComponentBase',
    'DockerHealthMonitorConfig',
    'DockerNetworkManagerConfig',
    'DockerVolumeManagerConfig',
    'NetworkConfig',
    'VolumeConfig',
    'HealthCheckConfig',
    'ResourceLimits',
    
    # Enums and data classes
    'ContainerOrchestrator',
    'RestartPolicy',
    'ServiceType',
    'HealthStatus',
    'HealthCheckResult',
    'ResourceMetrics'
]

__version__ = "1.0.0"

# Framework compliance notice
def _check_from_config_usage():
    """
    Verify that components are being used with from_config pattern.
    This function can be called during testing to ensure compliance.
    """
    import warnings
    warnings.warn(
        "IMPORTANT: All Docker infrastructure components must use the from_config pattern. "
        "Direct instantiation via __init__ is prohibited. "
        "Use ClassName.from_config(config) instead of ClassName(args).",
        UserWarning,
        stacklevel=2
    )

# Make compliance check available
__all__.append('_check_from_config_usage') 