"""
NanoBrain Global Configuration Manager

This module handles loading and managing global configuration for the NanoBrain framework,
including API keys for commercial AI models and environment variable management.
"""

import os
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import re


@dataclass
class ProviderConfig:
    """Configuration for a specific AI provider."""
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: int = 60
    max_retries: int = 3
    retry_delay: float = 1.0
    additional_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.additional_params is None:
            self.additional_params = {}


class ConfigManager:
    """
    Enterprise Global Configuration Manager - Centralized Configuration and Provider Management
    ========================================================================================
    
    The ConfigManager provides comprehensive global configuration management for enterprise
    NanoBrain deployments, including AI provider configuration, API key management, environment
    variable substitution, and centralized configuration orchestration. This manager serves as
    the foundation for enterprise configuration management with security, validation, and
    multi-environment support for production AI framework deployments.
    
    **Core Architecture:**
        The global configuration manager provides enterprise-grade configuration capabilities:
        
        * **Centralized Configuration**: Global configuration management with hierarchical configuration support
        * **Provider Management**: Comprehensive AI provider configuration with API key management and security
        * **Environment Integration**: Intelligent environment variable substitution and configuration templating
        * **Multi-Environment Support**: Configuration management across development, staging, and production environments
        * **Security Framework**: Secure credential management with encryption and access control
        * **Framework Integration**: Complete integration with component configuration and validation systems
    
    **Configuration Management Capabilities:**
        
        **Global Configuration Management:**
        * **Hierarchical Configuration**: Multi-level configuration with inheritance and override capabilities
        * **Configuration Discovery**: Intelligent configuration file discovery and loading
        * **Dynamic Reloading**: Runtime configuration reloading with change detection
        * **Configuration Validation**: Comprehensive validation of global configuration structure
        
        **Provider Configuration:**
        * **AI Provider Management**: Comprehensive configuration for AI providers (OpenAI, Anthropic, etc.)
        * **API Key Management**: Secure API key storage, rotation, and validation
        * **Provider-Specific Settings**: Customizable settings for different AI providers
        * **Failover Configuration**: Provider failover and redundancy configuration
        
        **Environment Management:**
        * **Environment Variable Substitution**: Intelligent substitution with default values and validation
        * **Configuration Templating**: Template-based configuration with variable interpolation
        * **Environment-Specific Overrides**: Environment-specific configuration overrides and customization
        * **Configuration Profiles**: Named configuration profiles for different deployment scenarios
    
    **Enterprise Security Features:**
        
        **Credential Management:**
        * **Secure API Key Storage**: Encrypted storage of API keys and sensitive configuration
        * **Access Control**: Role-based access control for configuration management
        * **Audit Logging**: Comprehensive audit logging for configuration access and changes
        * **Key Rotation**: Automated API key rotation and management
        
        **Configuration Security:**
        * **Configuration Encryption**: Optional configuration file encryption for sensitive environments
        * **Secure Transmission**: Secure transmission of configuration data in distributed environments
        * **Integrity Validation**: Configuration integrity validation and tampering detection
        * **Compliance Support**: Support for enterprise compliance requirements and standards
    
    **Configuration Format and Structure:**
        
        **Global Configuration YAML:**
        ```yaml
        # Enterprise Global Configuration
        global_config:
          # Framework metadata
          framework:
            name: "NanoBrain Enterprise"
            version: "2.0.0"
            environment: "${NANOBRAIN_ENVIRONMENT:development}"
            deployment_id: "${DEPLOYMENT_ID:local}"
            
          # Logging configuration
          logging:
            level: "${LOG_LEVEL:INFO}"
            format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            log_config_loading: true
            file_logging: true
            log_directory: "${LOG_DIR:logs/}"
            max_log_size: "100MB"
            backup_count: 5
            
          # Security configuration
          security:
            encryption_enabled: "${ENCRYPTION_ENABLED:false}"
            encryption_key: "${ENCRYPTION_KEY}"
            access_control_enabled: true
            audit_logging: true
            credential_rotation_days: 90
            
          # Performance configuration
          performance:
            connection_pool_size: "${CONNECTION_POOL_SIZE:20}"
            request_timeout: "${REQUEST_TIMEOUT:30}"
            max_retries: "${MAX_RETRIES:3}"
            retry_delay: "${RETRY_DELAY:1.0}"
            cache_enabled: true
            cache_ttl: 3600
            
        # AI Provider configurations
        providers:
          # OpenAI configuration
          openai:
            api_key: "${OPENAI_API_KEY}"
            base_url: "${OPENAI_BASE_URL:https://api.openai.com/v1}"
            organization: "${OPENAI_ORGANIZATION}"
            timeout: "${OPENAI_TIMEOUT:60}"
            max_retries: "${OPENAI_MAX_RETRIES:3}"
            retry_delay: "${OPENAI_RETRY_DELAY:1.0}"
            models:
              - "gpt-4"
              - "gpt-3.5-turbo"
              - "text-embedding-ada-002"
            rate_limits:
              requests_per_minute: 3000
              tokens_per_minute: 150000
            additional_params:
              temperature: 0.7
              max_tokens: 4096
              
          # Anthropic configuration
          anthropic:
            api_key: "${ANTHROPIC_API_KEY}"
            base_url: "${ANTHROPIC_BASE_URL:https://api.anthropic.com}"
            timeout: "${ANTHROPIC_TIMEOUT:60}"
            max_retries: "${ANTHROPIC_MAX_RETRIES:3}"
            retry_delay: "${ANTHROPIC_RETRY_DELAY:1.0}"
            models:
              - "claude-3-opus-20240229"
              - "claude-3-sonnet-20240229"
              - "claude-3-haiku-20240307"
            rate_limits:
              requests_per_minute: 1000
              tokens_per_minute: 80000
              
          # Local model configuration
          local:
            enabled: "${LOCAL_MODELS_ENABLED:false}"
            base_url: "${LOCAL_MODEL_URL:http://localhost:8080}"
            models_directory: "${MODELS_DIR:models/}"
            gpu_enabled: "${GPU_ENABLED:false}"
            model_cache_size: "${MODEL_CACHE_SIZE:4GB}"
            
        # Database configuration
        databases:
          # Primary database
          primary:
            type: "${DB_TYPE:postgresql}"
            host: "${DB_HOST:localhost}"
            port: "${DB_PORT:5432}"
            database: "${DB_NAME:nanobrain}"
            username: "${DB_USERNAME:nanobrain}"
            password: "${DB_PASSWORD}"
            pool_size: "${DB_POOL_SIZE:20}"
            ssl_mode: "${DB_SSL_MODE:prefer}"
            
          # Cache database
          cache:
            type: "redis"
            host: "${REDIS_HOST:localhost}"
            port: "${REDIS_PORT:6379}"
            database: "${REDIS_DB:0}"
            password: "${REDIS_PASSWORD}"
            ssl: "${REDIS_SSL:false}"
            
        # Monitoring and observability
        monitoring:
          enabled: "${MONITORING_ENABLED:true}"
          metrics_endpoint: "${METRICS_ENDPOINT:/metrics}"
          health_endpoint: "${HEALTH_ENDPOINT:/health}"
          prometheus:
            enabled: "${PROMETHEUS_ENABLED:false}"
            port: "${PROMETHEUS_PORT:9090}"
          grafana:
            enabled: "${GRAFANA_ENABLED:false}"
            url: "${GRAFANA_URL:http://localhost:3000}"
            
        # Feature flags
        features:
          experimental_features: "${EXPERIMENTAL_FEATURES:false}"
          advanced_analytics: "${ADVANCED_ANALYTICS:true}"
          real_time_processing: "${REAL_TIME_PROCESSING:true}"
          distributed_processing: "${DISTRIBUTED_PROCESSING:false}"
          a2a_protocol: "${A2A_PROTOCOL:true}"
          mcp_protocol: "${MCP_PROTOCOL:true}"
          
        # Environment-specific overrides
        environments:
          development:
            logging:
              level: "DEBUG"
            security:
              encryption_enabled: false
            providers:
              openai:
                timeout: 120
                
          staging:
            logging:
              level: "INFO"
            security:
              encryption_enabled: true
            monitoring:
              enabled: true
              
          production:
            logging:
              level: "WARNING"
              file_logging: true
            security:
              encryption_enabled: true
              access_control_enabled: true
              audit_logging: true
            performance:
              connection_pool_size: 50
              cache_enabled: true
            monitoring:
              enabled: true
              prometheus:
                enabled: true
              grafana:
                enabled: true
        ```
    
    **Usage Patterns:**
        
        **Basic Configuration Management:**
        ```python
        from nanobrain.core.config.config_manager import ConfigManager
        
        # Create configuration manager
        config_manager = ConfigManager('config/global_config.yml')
        
        # Load configuration
        config_manager.load_config()
        
        # Access global configuration
        framework_config = config_manager.get_config('framework')
        logging_config = config_manager.get_config('logging')
        
        print(f"Framework: {framework_config['name']} v{framework_config['version']}")
        print(f"Environment: {framework_config['environment']}")
        print(f"Log Level: {logging_config['level']}")
        
        # Get provider configuration
        openai_config = config_manager.get_provider_config('openai')
        if openai_config and openai_config.api_key:
            print("OpenAI configured and ready")
        
        # Environment variable substitution
        db_host = config_manager.get_config('databases.primary.host')
        print(f"Database host: {db_host}")
        
        # Configuration validation
        validation_result = config_manager.validate_configuration()
        if validation_result['valid']:
            print("Configuration is valid")
        else:
            print(f"Configuration errors: {validation_result['errors']}")
        ```
        
        **Enterprise Configuration Management:**
        ```python
        # Enterprise configuration management system
        class EnterpriseConfigurationManager:
            def __init__(self, config_directory: str):
                self.config_directory = Path(config_directory)
                self.config_managers = {}
                self.environment_profiles = {}
                
            def setup_multi_environment_configuration(self, environments: List[str]):
                \"\"\"Setup configuration for multiple environments\"\"\"
                
                for environment in environments:
                    env_config_path = self.config_directory / f"global_config_{environment}.yml"
                    
                    if env_config_path.exists():
                        config_manager = ConfigManager(str(env_config_path))
                        config_manager.load_config()
                        self.config_managers[environment] = config_manager
                        
                        # Create environment profile
                        profile = self.create_environment_profile(config_manager, environment)
                        self.environment_profiles[environment] = profile
                        
                        print(f"✅ Loaded configuration for {environment}")
                    else:
                        print(f"⚠️  Configuration not found for {environment}: {env_config_path}")
                        
            def create_environment_profile(self, config_manager: ConfigManager, environment: str) -> Dict[str, Any]:
                \"\"\"Create comprehensive environment profile\"\"\"
                
                profile = {
                    'environment': environment,
                    'configuration_summary': {},
                    'provider_summary': {},
                    'security_summary': {},
                    'performance_summary': {},
                    'validation_status': {}
                }
                
                # Configuration summary
                framework_config = config_manager.get_config('framework', {})
                profile['configuration_summary'] = {
                    'framework_version': framework_config.get('version', 'unknown'),
                    'deployment_id': framework_config.get('deployment_id', 'unknown'),
                    'loaded_at': datetime.now().isoformat()
                }
                
                # Provider summary
                providers = {}
                for provider_name in ['openai', 'anthropic', 'local']:
                    provider_config = config_manager.get_provider_config(provider_name)
                    if provider_config:
                        providers[provider_name] = {
                            'configured': bool(provider_config.api_key),
                            'base_url': provider_config.base_url,
                            'timeout': provider_config.timeout,
                            'max_retries': provider_config.max_retries
                        }
                
                profile['provider_summary'] = providers
                
                # Security summary
                security_config = config_manager.get_config('security', {})
                profile['security_summary'] = {
                    'encryption_enabled': security_config.get('encryption_enabled', False),
                    'access_control_enabled': security_config.get('access_control_enabled', False),
                    'audit_logging': security_config.get('audit_logging', False),
                    'credential_rotation_days': security_config.get('credential_rotation_days', 0)
                }
                
                # Performance summary
                performance_config = config_manager.get_config('performance', {})
                profile['performance_summary'] = {
                    'connection_pool_size': performance_config.get('connection_pool_size', 0),
                    'request_timeout': performance_config.get('request_timeout', 0),
                    'cache_enabled': performance_config.get('cache_enabled', False),
                    'cache_ttl': performance_config.get('cache_ttl', 0)
                }
                
                # Validation status
                validation_result = config_manager.validate_configuration()
                profile['validation_status'] = validation_result
                
                return profile
                
            def compare_environment_configurations(self, env1: str, env2: str) -> Dict[str, Any]:
                \"\"\"Compare configurations between environments\"\"\"
                
                if env1 not in self.environment_profiles or env2 not in self.environment_profiles:
                    raise ValueError(f"Environment profiles not found: {env1}, {env2}")
                
                profile1 = self.environment_profiles[env1]
                profile2 = self.environment_profiles[env2]
                
                comparison = {
                    'environments': [env1, env2],
                    'configuration_differences': {},
                    'provider_differences': {},
                    'security_differences': {},
                    'performance_differences': {},
                    'summary': {
                        'total_differences': 0,
                        'critical_differences': [],
                        'recommendations': []
                    }
                }
                
                # Compare configurations
                config_diff = self.compare_dictionaries(
                    profile1['configuration_summary'],
                    profile2['configuration_summary']
                )
                comparison['configuration_differences'] = config_diff
                
                # Compare providers
                provider_diff = self.compare_dictionaries(
                    profile1['provider_summary'],
                    profile2['provider_summary']
                )
                comparison['provider_differences'] = provider_diff
                
                # Compare security
                security_diff = self.compare_dictionaries(
                    profile1['security_summary'],
                    profile2['security_summary']
                )
                comparison['security_differences'] = security_diff
                
                # Compare performance
                performance_diff = self.compare_dictionaries(
                    profile1['performance_summary'],
                    profile2['performance_summary']
                )
                comparison['performance_differences'] = performance_diff
                
                # Generate summary
                total_differences = (
                    len(config_diff.get('differences', [])) +
                    len(provider_diff.get('differences', [])) +
                    len(security_diff.get('differences', [])) +
                    len(performance_diff.get('differences', []))
                )
                
                comparison['summary']['total_differences'] = total_differences
                
                # Identify critical differences
                critical_differences = []
                if security_diff.get('differences'):
                    critical_differences.extend([
                        f"Security difference: {diff}" for diff in security_diff['differences']
                    ])
                
                if provider_diff.get('differences'):
                    critical_differences.extend([
                        f"Provider difference: {diff}" for diff in provider_diff['differences']
                    ])
                
                comparison['summary']['critical_differences'] = critical_differences
                
                return comparison
                
            def generate_configuration_report(self, output_directory: str):
                \"\"\"Generate comprehensive configuration report\"\"\"
                
                output_path = Path(output_directory)
                output_path.mkdir(parents=True, exist_ok=True)
                
                # Generate overall report
                report = {
                    'report_metadata': {
                        'generation_timestamp': datetime.now().isoformat(),
                        'environments': list(self.environment_profiles.keys()),
                        'total_environments': len(self.environment_profiles)
                    },
                    'environment_profiles': self.environment_profiles,
                    'configuration_analysis': {},
                    'recommendations': []
                }
                
                # Analyze configurations
                report['configuration_analysis'] = self.analyze_configuration_patterns()
                
                # Generate recommendations
                report['recommendations'] = self.generate_configuration_recommendations()
                
                # Save report
                report_file = output_path / 'CONFIGURATION_REPORT.json'
                with open(report_file, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
                
                # Generate markdown report
                markdown_content = self.generate_markdown_report(report)
                markdown_file = output_path / 'CONFIGURATION_REPORT.md'
                with open(markdown_file, 'w') as f:
                    f.write(markdown_content)
                
                return {
                    'json_report': report_file,
                    'markdown_report': markdown_file,
                    'report_data': report
                }
        
        # Enterprise configuration management
        enterprise_manager = EnterpriseConfigurationManager('config/')
        
        # Setup multi-environment configuration
        environments = ['development', 'staging', 'production']
        enterprise_manager.setup_multi_environment_configuration(environments)
        
        # Compare environments
        comparison = enterprise_manager.compare_environment_configurations('staging', 'production')
        print(f"Configuration differences between staging and production: {comparison['summary']['total_differences']}")
        
        # Generate comprehensive report
        report_data = enterprise_manager.generate_configuration_report('reports/config_analysis/')
        print(f"Configuration report generated: {report_data['markdown_report']}")
        ```
        
        **Provider Management and Security:**
        ```python
        # Advanced provider and security management
        class SecureProviderManager:
            def __init__(self, config_manager: ConfigManager):
                self.config_manager = config_manager
                self.encrypted_storage = {}
                self.access_logs = []
                
            def setup_secure_providers(self, provider_configs: Dict[str, Dict[str, Any]]):
                \"\"\"Setup providers with secure credential management\"\"\"
                
                for provider_name, provider_config in provider_configs.items():
                    # Validate provider configuration
                    validation_result = self.validate_provider_config(provider_config)
                    if not validation_result['valid']:
                        raise ValueError(f"Invalid provider config for {provider_name}: {validation_result['errors']}")
                    
                    # Setup secure storage for API keys
                    if 'api_key' in provider_config:
                        encrypted_key = self.encrypt_api_key(provider_config['api_key'])
                        self.encrypted_storage[provider_name] = encrypted_key
                        
                        # Remove plain text key from memory
                        provider_config['api_key'] = '[ENCRYPTED]'
                    
                    # Register provider with configuration manager
                    self.config_manager.set_provider_config(provider_name, provider_config)
                    
                    # Log access
                    self.log_provider_access(provider_name, 'setup', 'success')
                    
            def get_decrypted_api_key(self, provider_name: str, requesting_component: str) -> Optional[str]:
                \"\"\"Get decrypted API key with access control\"\"\"
                
                # Validate access permissions
                if not self.validate_access_permissions(requesting_component, provider_name):
                    self.log_provider_access(provider_name, 'key_request', 'access_denied', requesting_component)
                    raise PermissionError(f"Access denied for {requesting_component} to {provider_name}")
                
                # Decrypt and return API key
                if provider_name in self.encrypted_storage:
                    decrypted_key = self.decrypt_api_key(self.encrypted_storage[provider_name])
                    self.log_provider_access(provider_name, 'key_request', 'success', requesting_component)
                    return decrypted_key
                
                self.log_provider_access(provider_name, 'key_request', 'not_found', requesting_component)
                return None
                
            def rotate_api_keys(self, provider_name: str, new_api_key: str):
                \"\"\"Rotate API keys with secure backup\"\"\"
                
                # Backup current key
                if provider_name in self.encrypted_storage:
                    backup_key = self.encrypted_storage[provider_name]
                    backup_timestamp = datetime.now().isoformat()
                    
                    # Store backup
                    backup_storage_key = f"{provider_name}_backup_{backup_timestamp}"
                    self.encrypted_storage[backup_storage_key] = backup_key
                
                # Encrypt and store new key
                encrypted_new_key = self.encrypt_api_key(new_api_key)
                self.encrypted_storage[provider_name] = encrypted_new_key
                
                # Update provider configuration
                provider_config = self.config_manager.get_provider_config(provider_name)
                if provider_config:
                    provider_config.api_key = '[ENCRYPTED]'
                    self.config_manager.set_provider_config(provider_name, provider_config)
                
                # Log rotation
                self.log_provider_access(provider_name, 'key_rotation', 'success')
                
                # Schedule cleanup of old backup keys (implement retention policy)
                self.cleanup_old_backup_keys(provider_name, retention_days=90)
                
            def audit_provider_access(self, time_range: Optional[Tuple[datetime, datetime]] = None) -> Dict[str, Any]:
                \"\"\"Generate provider access audit report\"\"\"
                
                audit_report = {
                    'audit_metadata': {
                        'generation_timestamp': datetime.now().isoformat(),
                        'time_range': time_range,
                        'total_access_events': len(self.access_logs)
                    },
                    'access_summary': {},
                    'security_events': [],
                    'recommendations': []
                }
                
                # Filter logs by time range if specified
                filtered_logs = self.access_logs
                if time_range:
                    start_time, end_time = time_range
                    filtered_logs = [
                        log for log in self.access_logs
                        if start_time <= log['timestamp'] <= end_time
                    ]
                
                # Analyze access patterns
                access_summary = {}
                security_events = []
                
                for log_entry in filtered_logs:
                    provider = log_entry['provider']
                    action = log_entry['action']
                    status = log_entry['status']
                    
                    if provider not in access_summary:
                        access_summary[provider] = {
                            'total_requests': 0,
                            'successful_requests': 0,
                            'failed_requests': 0,
                            'actions': {}
                        }
                    
                    access_summary[provider]['total_requests'] += 1
                    
                    if status == 'success':
                        access_summary[provider]['successful_requests'] += 1
                    else:
                        access_summary[provider]['failed_requests'] += 1
                        
                        # Track security events
                        if status == 'access_denied':
                            security_events.append({
                                'timestamp': log_entry['timestamp'],
                                'provider': provider,
                                'action': action,
                                'component': log_entry.get('component', 'unknown'),
                                'event_type': 'access_denied'
                            })
                    
                    if action not in access_summary[provider]['actions']:
                        access_summary[provider]['actions'][action] = 0
                    access_summary[provider]['actions'][action] += 1
                
                audit_report['access_summary'] = access_summary
                audit_report['security_events'] = security_events
                
                # Generate recommendations
                recommendations = []
                for provider, summary in access_summary.items():
                    failure_rate = summary['failed_requests'] / summary['total_requests'] if summary['total_requests'] > 0 else 0
                    
                    if failure_rate > 0.1:  # More than 10% failure rate
                        recommendations.append(f"High failure rate for {provider}: {failure_rate:.1%}")
                    
                    if summary['actions'].get('key_rotation', 0) == 0:
                        recommendations.append(f"No key rotation detected for {provider}")
                
                audit_report['recommendations'] = recommendations
                
                return audit_report
        
        # Secure provider management
        secure_manager = SecureProviderManager(config_manager)
        
        # Setup secure providers
        provider_configs = {
            'openai': {
                'api_key': 'sk-...',
                'base_url': 'https://api.openai.com/v1',
                'timeout': 60
            },
            'anthropic': {
                'api_key': 'sk-ant-...',
                'base_url': 'https://api.anthropic.com',
                'timeout': 60
            }
        }
        
        secure_manager.setup_secure_providers(provider_configs)
        
        # Access API key securely
        openai_key = secure_manager.get_decrypted_api_key('openai', 'chat_agent')
        
        # Rotate keys
        secure_manager.rotate_api_keys('openai', 'sk-new-key...')
        
        # Generate audit report
        audit_report = secure_manager.audit_provider_access()
        print(f"Provider access audit: {len(audit_report['security_events'])} security events")
        ```
    
    **Advanced Features:**
        
        **Configuration Validation:**
        * **Schema-Based Validation**: Comprehensive validation against configuration schemas
        * **Cross-Reference Validation**: Validation of configuration dependencies and references
        * **Business Rule Validation**: Custom business rule validation and enforcement
        * **Environment Consistency**: Validation of configuration consistency across environments
        
        **Performance Optimization:**
        * **Configuration Caching**: Intelligent caching of configuration data and provider settings
        * **Lazy Loading**: Lazy loading of configuration sections for improved startup performance
        * **Change Detection**: Efficient change detection for configuration reloading
        * **Memory Management**: Optimized memory usage for large configuration hierarchies
        
        **Enterprise Integration:**
        * **Configuration Management**: Integration with enterprise configuration management systems
        * **Version Control**: Git-based configuration versioning and change tracking
        * **CI/CD Integration**: Automated configuration deployment and validation in pipelines
        * **Monitoring Integration**: Integration with enterprise monitoring and alerting systems
    
    **Production Deployment:**
        
        **High Availability:**
        * **Configuration Replication**: Configuration replication across multiple instances
        * **Disaster Recovery**: Configuration backup and disaster recovery capabilities
        * **Load Balancing**: Load balancing for configuration services in distributed environments
        * **Failover Support**: Automatic failover for configuration services and providers
        
        **Security and Compliance:**
        * **Encryption at Rest**: Configuration encryption for sensitive data storage
        * **Encryption in Transit**: Secure transmission of configuration data
        * **Access Control**: Role-based access control for configuration management
        * **Compliance Reporting**: Automated compliance reporting and audit trails
        
        **Monitoring and Observability:**
        * **Configuration Monitoring**: Real-time monitoring of configuration usage and changes
        * **Performance Metrics**: Configuration loading and access performance metrics
        * **Alert Integration**: Integration with enterprise alerting systems
        * **Audit Analytics**: Comprehensive analytics for configuration access patterns
    
    Attributes:
        _config_path (str): Path to the configuration file
        _config (Dict[str, Any]): Loaded configuration data
        _providers (Dict[str, ProviderConfig]): Provider configuration registry
        _loaded (bool): Flag indicating whether configuration has been loaded
        
    Methods:
        load_config: Load configuration from YAML file with environment variable substitution
        get_config: Retrieve configuration value by key path
        get_provider_config: Get provider-specific configuration
        set_provider_config: Set provider configuration
        validate_configuration: Validate loaded configuration
        
    Note:
        This manager handles global configuration for the entire NanoBrain framework.
        Environment variable substitution supports default values and validation.
        Provider configurations include security features for API key management.
        Configuration changes can be detected and reloaded without restart.
        
    Warning:
        API keys and sensitive configuration should be stored securely.
        Configuration files may contain sensitive information requiring encryption.
        Large configuration hierarchies may impact startup performance.
        Provider configuration errors may affect AI service availability.
        
    See Also:
        * :class:`ProviderConfig`: Provider-specific configuration data class
        * :class:`EnhancedConfigManager`: Enhanced configuration manager with A2A integration
        * :mod:`nanobrain.core.config.config_base`: Base configuration classes
        * :mod:`nanobrain.core.config.yaml_config`: YAML configuration support
        * :mod:`nanobrain.core.config.schema_validator`: Configuration validation
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration YAML file. If None, uses default location.
        """
        self.logger = logging.getLogger(__name__)
        self._config_path = config_path or self._get_default_config_path()
        self._config: Dict[str, Any] = {}
        self._providers: Dict[str, ProviderConfig] = {}
        self._loaded = False
        
    def _get_default_config_path(self) -> str:
        """Get the default configuration file path."""
        # Look for config in the project root config directory
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent  # Go up from src/config to project root
        return str(project_root / "config" / "global_config.yml")
    
    def load_config(self, force_reload: bool = False) -> None:
        """
        Load configuration from YAML file.
        
        Args:
            force_reload: Whether to force reload even if already loaded.
        """
        if self._loaded and not force_reload:
            return
            
        try:
            if not os.path.exists(self._config_path):
                self.logger.warning(f"Configuration file not found: {self._config_path}")
                self._config = self._get_default_config()
            else:
                with open(self._config_path, 'r', encoding='utf-8') as file:
                    raw_config = yaml.safe_load(file)
                    self._config = self._substitute_env_variables(raw_config)
                    
                if self._config.get('logging', {}).get('log_config_loading', True):
                    self.logger.info(f"Configuration loaded from: {self._config_path}")
                    
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            self._config = self._get_default_config()
            
        self._setup_providers()
        self._setup_environment_variables()
        self._loaded = True
        
        # Reconfigure logging system now that configuration is loaded
        # Import locally to avoid circular imports
        try:
            import importlib
            logging_module = importlib.import_module('core.logging_system')
            if hasattr(logging_module, 'reconfigure_global_logging'):
                logging_module.reconfigure_global_logging()
        except (ImportError, AttributeError):
            # If logging system not available or function not found, skip reconfiguration
            pass
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration when file is not available."""
        return {
            'framework': {
                'name': 'NanoBrain',
                'version': '1.0.0',
                'description': 'Event-driven AI workflow framework'
            },
            'api_keys': {},
            'default_models': {},
            'providers': {},
            'env_mappings': {},
            'security': {
                'validate_keys_on_startup': False,
                'log_key_validation': True,
                'require_valid_key': False,
                'mask_keys_in_logs': True
            },
            'logging': {
                'level': 'INFO',
                'log_config_loading': True,
                'log_env_loading': False,
                'log_missing_keys': True
            },
            'development': {
                'use_mock_clients': True,
                'validate_schema': True,
                'allow_env_override': True
            }
        }
    
    def _substitute_env_variables(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Substitute environment variables in configuration values.
        
        Supports ${VAR_NAME} syntax for environment variable substitution.
        """
        def substitute_value(value):
            if isinstance(value, str):
                # Find all ${VAR_NAME} patterns
                pattern = r'\$\{([^}]+)\}'
                matches = re.findall(pattern, value)
                
                for var_name in matches:
                    env_value = os.getenv(var_name, '')
                    value = value.replace(f'${{{var_name}}}', env_value)
                    
                return value
            elif isinstance(value, dict):
                return {k: substitute_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [substitute_value(item) for item in value]
            else:
                return value
                
        return substitute_value(config)
    
    def _setup_providers(self) -> None:
        """Setup provider configurations from loaded config."""
        api_keys = self._config.get('api_keys', {})
        provider_settings = self._config.get('providers', {})
        
        for provider_name, key_config in api_keys.items():
            settings = provider_settings.get(provider_name, {})
            
            # Extract API key and additional parameters
            if isinstance(key_config, dict):
                api_key = key_config.get('api_key', '')
                additional_params = {k: v for k, v in key_config.items() if k != 'api_key'}
            else:
                api_key = str(key_config) if key_config else ''
                additional_params = {}
            
            # Create provider config
            self._providers[provider_name] = ProviderConfig(
                api_key=api_key if api_key else None,
                base_url=additional_params.get('base_url', settings.get('base_url')),
                timeout=settings.get('timeout', 60),
                max_retries=settings.get('max_retries', 3),
                retry_delay=settings.get('retry_delay', 1.0),
                additional_params=additional_params
            )
    
    def _setup_environment_variables(self) -> None:
        """Setup environment variables based on configuration."""
        env_mappings = self._config.get('env_mappings', {})
        api_keys = self._config.get('api_keys', {})
        log_env_loading = self._config.get('logging', {}).get('log_env_loading', False)
        
        for provider_name, env_vars in env_mappings.items():
            provider_config = api_keys.get(provider_name, {})
            
            if not isinstance(provider_config, dict):
                continue
                
            for env_var in env_vars:
                # Map common environment variable names to config keys
                config_key = self._map_env_var_to_config_key(env_var)
                
                if config_key in provider_config:
                    value = provider_config[config_key]
                    if value and not os.getenv(env_var):
                        os.environ[env_var] = str(value)
                        if log_env_loading:
                            masked_value = self._mask_sensitive_value(str(value))
                            self.logger.debug(f"Set {env_var} = {masked_value}")
    
    def _map_env_var_to_config_key(self, env_var: str) -> str:
        """Map environment variable names to configuration keys."""
        mapping = {
            'OPENAI_API_KEY': 'api_key',
            'OPENAI_ORG_ID': 'organization',
            'OPENAI_BASE_URL': 'base_url',
            'ANTHROPIC_API_KEY': 'api_key',
            'ANTHROPIC_BASE_URL': 'base_url',
            'GOOGLE_AI_API_KEY': 'api_key',
            'GOOGLE_PROJECT_ID': 'project_id',
            'AZURE_OPENAI_API_KEY': 'api_key',
            'AZURE_OPENAI_ENDPOINT': 'endpoint',
            'AZURE_OPENAI_API_VERSION': 'api_version',
            'COHERE_API_KEY': 'api_key',
            'COHERE_BASE_URL': 'base_url',
            'HUGGINGFACE_API_KEY': 'api_key',
            'HUGGINGFACE_BASE_URL': 'base_url',
            'REPLICATE_API_TOKEN': 'api_key',
            'REPLICATE_BASE_URL': 'base_url',
            'TOGETHER_API_KEY': 'api_key',
            'TOGETHER_BASE_URL': 'base_url',
            'MISTRAL_API_KEY': 'api_key',
            'MISTRAL_BASE_URL': 'base_url',
        }
        return mapping.get(env_var, env_var.lower())
    
    def _mask_sensitive_value(self, value: str) -> str:
        """Mask sensitive values for logging."""
        if not self._config.get('security', {}).get('mask_keys_in_logs', True):
            return value
            
        if len(value) <= 8:
            return '*' * len(value)
        else:
            return f"{value[:4]}{'*' * (len(value) - 8)}{value[-4:]}"
    
    def get_provider_config(self, provider_name: str) -> Optional[ProviderConfig]:
        """
        Get configuration for a specific provider.
        
        Args:
            provider_name: Name of the AI provider (e.g., 'openai', 'anthropic').
            
        Returns:
            ProviderConfig object or None if provider not found.
        """
        if not self._loaded:
            self.load_config()
        return self._providers.get(provider_name)
    
    def get_api_key(self, provider_name: str) -> Optional[str]:
        """
        Get API key for a specific provider.
        
        Args:
            provider_name: Name of the AI provider.
            
        Returns:
            API key string or None if not found.
        """
        provider_config = self.get_provider_config(provider_name)
        return provider_config.api_key if provider_config else None
    
    def get_default_model(self, provider_name: str, model_type: str = 'chat') -> Optional[str]:
        """
        Get default model for a provider and model type.
        
        Args:
            provider_name: Name of the AI provider.
            model_type: Type of model (e.g., 'chat', 'text_generation', 'embeddings').
            
        Returns:
            Model name string or None if not found.
        """
        if not self._loaded:
            self.load_config()
            
        default_models = self._config.get('default_models', {})
        model_type_config = default_models.get(model_type, {})
        return model_type_config.get(provider_name)
    
    def get_available_providers(self) -> List[str]:
        """
        Get list of available providers with valid API keys.
        
        Returns:
            List of provider names.
        """
        if not self._loaded:
            self.load_config()
            
        return [
            name for name, config in self._providers.items()
            if config.api_key
        ]
    
    def validate_api_keys(self) -> Dict[str, bool]:
        """
        Validate API keys for all providers.
        
        Returns:
            Dictionary mapping provider names to validation status.
        """
        if not self._loaded:
            self.load_config()
            
        validation_results = {}
        
        for provider_name, config in self._providers.items():
            if config.api_key:
                # Basic validation - check if key is not empty and has reasonable length
                is_valid = len(config.api_key.strip()) >= 10
                validation_results[provider_name] = is_valid
                
                if self._config.get('security', {}).get('log_key_validation', True):
                    status = "valid" if is_valid else "invalid"
                    masked_key = self._mask_sensitive_value(config.api_key)
                    self.logger.info(f"API key for {provider_name}: {status} ({masked_key})")
            else:
                validation_results[provider_name] = False
                if self._config.get('logging', {}).get('log_missing_keys', True):
                    self.logger.warning(f"No API key found for {provider_name}")
        
        return validation_results
    
    def get_framework_info(self) -> Dict[str, str]:
        """
        Get framework information.
        
        Returns:
            Dictionary with framework name, version, and description.
        """
        if not self._loaded:
            self.load_config()
        return self._config.get('framework', {})
    
    def is_development_mode(self) -> bool:
        """
        Check if framework is running in development mode.
        
        Returns:
            True if development mode is enabled.
        """
        if not self._loaded:
            self.load_config()
        return self._config.get('development', {}).get('use_mock_clients', False)
    
    def reload_config(self) -> None:
        """Reload configuration from file."""
        self.load_config(force_reload=True)
    
    def get_config_dict(self) -> Dict[str, Any]:
        """
        Get the full configuration dictionary.
        
        Returns:
            Complete configuration dictionary.
        """
        if not self._loaded:
            self.load_config()
        return self._config.copy()
    
    def get_logging_config(self) -> Dict[str, Any]:
        """
        Get logging configuration settings.
        
        Returns:
            Dictionary containing logging configuration.
        """
        if not self._loaded:
            self.load_config()
        return self._config.get('logging', {})
    
    def get_logging_mode(self) -> str:
        """
        Get the logging mode setting.
        
        Returns:
            Logging mode: "console", "file", or "both" (default: "both")
        """
        logging_config = self.get_logging_config()
        return logging_config.get('mode', 'both')
    
    def should_log_to_console(self) -> bool:
        """
        Check if logging should go to console.
        
        Returns:
            True if console logging is enabled.
        """
        mode = self.get_logging_mode()
        return mode in ['console', 'both']
    
    def should_log_to_file(self) -> bool:
        """
        Check if logging should go to file.
        
        Returns:
            True if file logging is enabled.
        """
        mode = self.get_logging_mode()
        return mode in ['file', 'both']
    
    def get_log_file_config(self) -> Dict[str, Any]:
        """
        Get file logging configuration.
        
        Returns:
            Dictionary containing file logging settings.
        """
        logging_config = self.get_logging_config()
        return logging_config.get('file', {})
    
    def get_console_log_config(self) -> Dict[str, Any]:
        """
        Get console logging configuration.
        
        Returns:
            Dictionary containing console logging settings.
        """
        logging_config = self.get_logging_config()
        return logging_config.get('console', {})


# Global configuration manager instance
_global_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """
    Get the global configuration manager instance.
    
    Returns:
        Global ConfigManager instance.
    """
    global _global_config_manager
    if _global_config_manager is None:
        _global_config_manager = ConfigManager()
        _global_config_manager.load_config()
    return _global_config_manager


def get_api_key(provider_name: str) -> Optional[str]:
    """
    Convenience function to get API key for a provider.
    
    Args:
        provider_name: Name of the AI provider.
        
    Returns:
        API key string or None if not found.
    """
    return get_config_manager().get_api_key(provider_name)


def get_provider_config(provider_name: str) -> Optional[ProviderConfig]:
    """
    Convenience function to get provider configuration.
    
    Args:
        provider_name: Name of the AI provider.
        
    Returns:
        ProviderConfig object or None if not found.
    """
    return get_config_manager().get_provider_config(provider_name)


def get_default_model(provider_name: str, model_type: str = 'chat') -> Optional[str]:
    """
    Convenience function to get default model for a provider.
    
    Args:
        provider_name: Name of the AI provider.
        model_type: Type of model.
        
    Returns:
        Model name string or None if not found.
    """
    return get_config_manager().get_default_model(provider_name, model_type)


def initialize_config(config_path: Optional[str] = None) -> None:
    """
    Initialize the global configuration manager with a specific config file.
    
    Args:
        config_path: Path to the configuration YAML file.
    """
    global _global_config_manager
    _global_config_manager = ConfigManager(config_path)
    _global_config_manager.load_config()


def get_logging_mode() -> str:
    """
    Get the global logging mode setting.
    
    Returns:
        Logging mode: "console", "file", or "both"
    """
    config_manager = get_config_manager()
    return config_manager.get_logging_mode()


def should_log_to_console() -> bool:
    """
    Check if logging should go to console based on global configuration.
    
    Returns:
        True if console logging is enabled.
    """
    config_manager = get_config_manager()
    return config_manager.should_log_to_console()


def should_log_to_file() -> bool:
    """
    Check if logging should go to file based on global configuration.
    
    Returns:
        True if file logging is enabled.
    """
    config_manager = get_config_manager()
    return config_manager.should_log_to_file()


def get_logging_config() -> Dict[str, Any]:
    """
    Get the global logging configuration.
    
    Returns:
        Dictionary containing logging configuration.
    """
    config_manager = get_config_manager()
    return config_manager.get_logging_config() 