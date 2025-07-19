# Docker Infrastructure Configuration Files

This directory contains default YAML configuration files for all Docker infrastructure components in the NanoBrain framework. These configurations enable proper instantiation of Docker components using the mandatory `from_config` pattern.

## Overview

All Docker infrastructure components in NanoBrain follow the mandatory `from_config` pattern, which means:

- **Direct instantiation is prohibited**: Use `ClassName.from_config(config)` instead of `ClassName(args)`
- **Unified configuration management**: All components use YAML configuration files
- **Framework compliance**: All components include mandatory tool card metadata
- **Dependency injection**: Automatic resolution of component dependencies

## Configuration Files

### 1. DockerManager.yml

**Purpose**: Core Docker container management for the NanoBrain framework

**Usage**:
```python
import yaml
from nanobrain.library.infrastructure.docker import DockerManager

# Load configuration
with open('DockerManager.yml', 'r') as f:
    config = yaml.safe_load(f)

# Create DockerManager instance
docker_manager = DockerManager.from_config(config)
```

**Key Features**:
- Container lifecycle management
- Image management and pulling
- Resource monitoring and limits
- Health checking integration
- Security configuration

### 2. DockerHealthMonitor.yml

**Purpose**: Comprehensive health monitoring for Docker containers

**Usage**:
```python
import yaml
from nanobrain.library.infrastructure.docker import DockerHealthMonitor, DockerManager

# Create DockerManager first (required dependency)
docker_manager = DockerManager.from_config(docker_manager_config)

# Load health monitor configuration
with open('DockerHealthMonitor.yml', 'r') as f:
    config = yaml.safe_load(f)

# Add required dependency
config['docker_manager'] = docker_manager

# Create DockerHealthMonitor instance
health_monitor = DockerHealthMonitor.from_config(config)
```

**Key Features**:
- HTTP, TCP, and exec health checks
- Resource usage monitoring
- Alert notifications
- Health history tracking
- Configurable thresholds

**Required Dependencies**:
- `docker_manager`: DockerManager instance

### 3. DockerNetworkManager.yml

**Purpose**: Docker network management for container networking and isolation

**Usage**:
```python
import yaml
from nanobrain.library.infrastructure.docker import DockerNetworkManager

# Load configuration
with open('DockerNetworkManager.yml', 'r') as f:
    config = yaml.safe_load(f)

# Create DockerNetworkManager instance
network_manager = DockerNetworkManager.from_config(config)
```

**Key Features**:
- Network creation and management
- Container network attachment/detachment
- Network isolation and security
- Service discovery
- Load balancing support

### 4. DockerVolumeManager.yml

**Purpose**: Docker volume management for persistent storage and data operations

**Usage**:
```python
import yaml
from nanobrain.library.infrastructure.docker import DockerVolumeManager

# Load configuration
with open('DockerVolumeManager.yml', 'r') as f:
    config = yaml.safe_load(f)

# Create DockerVolumeManager instance
volume_manager = DockerVolumeManager.from_config(config)
```

**Key Features**:
- Volume creation and management
- Data backup and restore
- Volume sharing between containers
- Cleanup and maintenance
- Storage monitoring

### 5. ContainerConfig.yml

**Purpose**: Universal container configuration supporting multiple orchestrators

**Usage**:
```python
import yaml
from nanobrain.library.infrastructure.docker import ContainerConfig

# Load configuration
with open('ContainerConfig.yml', 'r') as f:
    config = yaml.safe_load(f)

# Create ContainerConfig instance
container_config = ContainerConfig.from_config(config)

# Use with Docker
docker_config = container_config.to_docker_format()

# Use with Kubernetes
k8s_deployment = container_config.to_kubernetes_deployment()
k8s_service = container_config.to_kubernetes_service()

# Use with Docker Compose
compose_service = container_config.to_docker_compose_format()
```

**Key Features**:
- Multi-orchestrator support (Docker, Kubernetes, Docker Compose)
- Resource limits and health checks
- Security configuration
- Environment-specific settings

## Complete Integration Example

Here's a complete example showing how to use all Docker infrastructure components together:

```python
import yaml
from nanobrain.library.infrastructure.docker import (
    DockerManager,
    DockerHealthMonitor,
    DockerNetworkManager,
    DockerVolumeManager,
    ContainerConfig
)

def load_config(filename):
    """Load YAML configuration file"""
    with open(f"nanobrain/library/config/defaults/docker/{filename}", 'r') as f:
        return yaml.safe_load(f)

# Step 1: Create DockerManager
docker_manager_config = load_config("DockerManager.yml")
docker_manager = DockerManager.from_config(docker_manager_config)

# Step 2: Create DockerNetworkManager
network_config = load_config("DockerNetworkManager.yml")
network_manager = DockerNetworkManager.from_config(network_config)

# Step 3: Create DockerVolumeManager
volume_config = load_config("DockerVolumeManager.yml")
volume_manager = DockerVolumeManager.from_config(volume_config)

# Step 4: Create DockerHealthMonitor with dependency
health_config = load_config("DockerHealthMonitor.yml")
health_config['docker_manager'] = docker_manager
health_monitor = DockerHealthMonitor.from_config(health_config)

# Step 5: Create ContainerConfig
container_config_data = load_config("ContainerConfig.yml")
container_config = ContainerConfig.from_config(container_config_data)

# All components are now ready for use
print("✅ All Docker infrastructure components created successfully!")
```

## Configuration Customization

### Environment-Specific Configurations

All configuration files support environment-specific settings:

```yaml
environment:
  type: "production"  # development, staging, production
  debug_logging: false
  preserve_containers_on_error: false
```

### Framework Integration

All components include framework integration settings:

```yaml
framework_integration:
  use_nanobrain_logging: true
  generate_a2a_cards: false
  use_nanobrain_monitoring: true
```

### Security Configuration

Security settings are included in all relevant components:

```yaml
security:
  prefer_non_root: true
  apply_security_context: true
  drop_capabilities: ["ALL"]
  add_capabilities: []
```

## Tool Card Compliance

All configuration files include mandatory tool card metadata for framework compliance:

```yaml
tool_card:
  name: "ComponentName"
  description: "Component description"
  version: "1.0.0"
  category: "infrastructure"
  capabilities: ["capability1", "capability2"]
  requirements: ["requirement1"]
  supported_platforms: ["linux", "darwin", "windows"]
  configuration_schema:
    type: "ConfigClassName"
    required_fields: ["field1"]
    optional_fields: ["field2", "field3"]
  usage_examples:
    - description: "Example usage"
      config:
        component_name: "example"
```

## Validation and Testing

Use the provided test suite to validate configuration files:

```bash
# Run configuration validation tests
cd tests
python -m pytest test_docker_yaml_configs.py -v
```

The test suite validates:
- YAML file validity and structure
- Required field presence
- Tool card compliance
- Configuration schema validation
- Framework integration metadata
- Usage example validity

## Best Practices

1. **Always use from_config pattern**: Never instantiate Docker components directly
2. **Validate configurations**: Use the test suite to validate custom configurations
3. **Environment-specific configs**: Create separate configuration files for different environments
4. **Dependency management**: Ensure required dependencies are properly configured
5. **Security settings**: Review and customize security settings for your environment
6. **Resource limits**: Set appropriate resource limits for production deployments
7. **Monitoring**: Enable health monitoring and alerting for production systems

## Troubleshooting

### Common Issues

1. **Missing Docker daemon**: Ensure Docker is running and accessible
2. **Permission errors**: Check Docker daemon permissions and user groups
3. **Network conflicts**: Verify network subnet configurations don't conflict
4. **Volume permissions**: Ensure proper file system permissions for volume mounts
5. **Resource constraints**: Check available system resources for containers

### Configuration Validation

If you encounter configuration errors, run the validation tests:

```bash
python -m pytest test_docker_yaml_configs.py::TestDockerYAMLConfigurations::test_yaml_files_valid -v
```

### Debug Mode

Enable debug logging in configurations for troubleshooting:

```yaml
environment:
  debug_logging: true
```

## Contributing

When adding new Docker infrastructure components:

1. Create a corresponding YAML configuration file
2. Include mandatory tool card metadata
3. Add comprehensive usage examples
4. Update this README with component documentation
5. Add validation tests to the test suite
6. Ensure from_config pattern compliance

## Support

For issues with Docker infrastructure configurations:

1. Check the validation test results
2. Review the component-specific documentation
3. Verify Docker daemon connectivity
4. Check system resource availability
5. Review NanoBrain framework logs 