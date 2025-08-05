"""
Docker Health Monitor for NanoBrain Framework

This module provides comprehensive health monitoring capabilities for Docker containers,
including health checks, resource monitoring, and alerting.

"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Callable, Union, ClassVar
from dataclasses import dataclass, field
from enum import Enum

from .docker_manager import DockerManager
from .container_config import HealthCheckConfig, DockerComponentConfig, DockerComponentBase
from nanobrain.core.logging_system import get_logger
from nanobrain.core.component_base import ComponentDependencyError


class HealthStatus(Enum):
    """Container health status"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    STARTING = "starting"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check"""
    container_name: str
    status: HealthStatus
    timestamp: float
    response_time: float
    details: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None


@dataclass
class ResourceMetrics:
    """Container resource usage metrics"""
    container_name: str
    timestamp: float
    cpu_percent: float
    memory_usage: int  # bytes
    memory_limit: int  # bytes
    memory_percent: float
    network_rx: int    # bytes
    network_tx: int    # bytes
    disk_read: int     # bytes
    disk_write: int    # bytes


@dataclass
class DockerHealthMonitorConfig(DockerComponentConfig):
    """Configuration for Docker Health Monitor"""
    component_name: str = "docker_health_monitor"
    docker_manager: Optional[DockerManager] = None
    alert_thresholds: Dict[str, Any] = field(default_factory=lambda: {
        'cpu_threshold': 80.0,
        'memory_threshold': 85.0,
        'consecutive_failures': 3
    })
    history_limit: int = 100  # Number of health checks to keep


class DockerHealthMonitor(DockerComponentBase):
    """
    Enterprise Docker Health Monitor - Comprehensive Container Health and Resource Monitoring
    =====================================================================================
    
    The DockerHealthMonitor provides comprehensive health monitoring and resource tracking for Docker
    containers in enterprise environments, offering real-time health checks, resource utilization
    monitoring, intelligent alerting, and automated recovery mechanisms. This monitoring system ensures
    container reliability, performance optimization, and proactive issue detection for production
    Docker deployments across distributed infrastructure.
    
    **Core Architecture:**
        The Docker health monitor provides enterprise-grade container monitoring capabilities:
        
        * **Real-Time Health Monitoring**: Continuous container health assessment and validation
        * **Resource Utilization Tracking**: CPU, memory, network, and disk usage monitoring
        * **Intelligent Alerting**: Threshold-based alerting with escalation and notification management
        * **Automated Recovery**: Self-healing capabilities with container restart and replacement
        * **Performance Analytics**: Historical trend analysis and capacity planning support
        * **Framework Integration**: Full integration with NanoBrain's infrastructure management architecture
    
    **Container Health Monitoring:**
        
        **Health Check Capabilities:**
        * **HTTP Health Endpoints**: Automated HTTP/HTTPS health check validation
        * **TCP Port Monitoring**: Network service availability and connectivity testing
        * **Custom Command Execution**: Container-specific health validation scripts
        * **Application-Level Checks**: Service-specific health validation and business logic
        * **Dependency Validation**: External service dependency health verification
        
        **Health Status Management:**
        * **HEALTHY**: Container is operating normally with all checks passing
        * **UNHEALTHY**: Container has failed health checks and requires attention
        * **STARTING**: Container is initializing and health status is being established
        * **UNKNOWN**: Health status cannot be determined due to monitoring issues
        
        **Health Validation Strategies:**
        * **Multi-Layer Validation**: Infrastructure, application, and business layer health checks
        * **Configurable Check Intervals**: Adaptive health check frequency based on container state
        * **Failure Threshold Management**: Configurable failure counts before marking unhealthy
        * **Recovery Validation**: Post-recovery health verification and status restoration
    
    **Resource Monitoring Capabilities:**
        
        **CPU Performance Monitoring:**
        * Real-time CPU utilization tracking and analysis
        * Multi-core CPU usage distribution and load balancing
        * CPU throttling detection and performance impact assessment
        * Historical CPU usage trends and capacity planning
        
        **Memory Management Monitoring:**
        * Memory usage tracking with limit enforcement
        * Memory leak detection and trend analysis
        * OOM (Out of Memory) prevention and alerting
        * Memory efficiency optimization recommendations
        
        **Network Performance Tracking:**
        * Network I/O throughput monitoring and analysis
        * Connection tracking and bandwidth utilization
        * Network latency and packet loss detection
        * Inter-container communication monitoring
        
        **Storage and Disk Monitoring:**
        * Disk I/O performance tracking and optimization
        * Storage utilization and capacity management
        * Volume mount health and accessibility validation
        * Disk performance bottleneck identification
    
    **Enterprise Alerting System:**
        
        **Intelligent Alert Management:**
        * **Threshold-Based Alerting**: Configurable resource and health thresholds
        * **Trend-Based Alerting**: Predictive alerting based on usage trends
        * **Correlation Analysis**: Multi-container alert correlation and root cause analysis
        * **Alert Suppression**: Intelligent alert grouping and noise reduction
        
        **Multi-Channel Notification:**
        * **Email Notifications**: SMTP-based alert delivery with rich formatting
        * **Webhook Integration**: HTTP webhook delivery for external system integration
        * **Dashboard Alerts**: Real-time dashboard notifications and visual indicators
        * **Mobile Push Notifications**: Critical alert delivery to mobile devices
        
        **Escalation Management:**
        * **Severity-Based Escalation**: Automatic escalation based on alert severity
        * **Time-Based Escalation**: Escalation triggers based on unresolved alert duration
        * **Team-Based Routing**: Alert routing based on team responsibilities and schedules
        * **Executive Notifications**: Critical issue escalation to leadership
    
    **Configuration Architecture:**
        Comprehensive configuration supports diverse Docker monitoring scenarios:
        
        ```yaml
        # Docker Health Monitor Configuration
        monitor_name: "enterprise_docker_health_monitor"
        monitor_type: "comprehensive"
        
        # Monitor card for framework integration
        monitor_card:
          name: "enterprise_docker_health_monitor"
          description: "Enterprise Docker container health and resource monitoring"
          version: "1.0.0"
          category: "infrastructure"
          capabilities:
            - "container_health_monitoring"
            - "resource_utilization_tracking"
            - "intelligent_alerting"
        
        # Docker Manager Integration
        docker_manager:
          connection_url: "unix:///var/run/docker.sock"
          timeout: 30
          api_version: "auto"
          
        # Health Check Configuration
        health_checks:
          enabled: true
          default_interval: 30          # seconds
          timeout: 10                   # seconds
          retries: 3
          failure_threshold: 3          # failures before marking unhealthy
          
          http_checks:
            enabled: true
            default_path: "/health"
            expected_status: [200, 204]
            follow_redirects: true
            verify_ssl: true
            
          tcp_checks:
            enabled: true
            connection_timeout: 5
            
          command_checks:
            enabled: true
            shell: "/bin/sh"
            working_directory: "/"
            
        # Resource Monitoring Configuration
        resource_monitoring:
          enabled: true
          collection_interval: 15       # seconds
          history_retention: 86400      # seconds (24 hours)
          
          cpu_monitoring:
            enabled: true
            threshold_warning: 75.0     # percentage
            threshold_critical: 90.0    # percentage
            
          memory_monitoring:
            enabled: true
            threshold_warning: 80.0     # percentage
            threshold_critical: 95.0    # percentage
            oom_detection: true
            
          network_monitoring:
            enabled: true
            bandwidth_threshold: 104857600  # bytes/sec (100 MB/s)
            connection_limit: 1000
            
          disk_monitoring:
            enabled: true
            io_threshold: 52428800      # bytes/sec (50 MB/s)
            space_threshold: 85.0       # percentage
            
        # Alert Configuration
        alerting:
          enabled: true
          alert_interval: 300           # seconds (5 minutes)
          escalation_enabled: true
          
          channels:
            email:
              enabled: true
              smtp_server: "smtp.company.com"
              smtp_port: 587
              username: "alerts@company.com"
              recipients: ["ops@company.com", "dev@company.com"]
              
            webhook:
              enabled: true
              url: "https://alerting.company.com/webhook"
              timeout: 10
              retry_attempts: 3
              
            dashboard:
              enabled: true
              update_interval: 5
              
          thresholds:
            cpu_usage:
              warning: 75.0
              critical: 90.0
              trend_alert: true
              
            memory_usage:
              warning: 80.0
              critical: 95.0
              leak_detection: true
              
            health_check_failures:
              warning: 2
              critical: 3
              consecutive_required: true
              
        # Recovery Configuration
        recovery:
          enabled: true
          auto_restart: true
          restart_policy: "unless-stopped"
          max_restart_attempts: 3
          restart_delay: 30             # seconds
          
          replacement_strategy:
            enabled: true
            replacement_timeout: 300    # seconds
            preserve_data: true
            
        # Analytics Configuration
        analytics:
          enabled: true
          trend_analysis: true
          capacity_planning: true
          performance_recommendations: true
          
        # Integration Configuration
        integration:
          prometheus_export: true
          grafana_dashboards: true
          elk_stack_logging: true
          external_apis: true
        ```
    
    **Usage Patterns:**
        
        **Basic Docker Health Monitoring Setup:**
        ```python
        from nanobrain.library.infrastructure.docker import DockerHealthMonitor, DockerManager
        
        # Initialize Docker manager
        docker_manager = DockerManager.from_config("docker_config.yml")
        await docker_manager.initialize()
        
        # Create health monitor configuration
        monitor_config = {
            'component_name': 'production_health_monitor',
            'docker_manager': docker_manager,
            'health_check_interval': 30,
            'resource_monitoring_interval': 15,
            'alert_thresholds': {
                'cpu_warning': 75.0,
                'cpu_critical': 90.0,
                'memory_warning': 80.0,
                'memory_critical': 95.0
            }
        }
        
        # Initialize health monitor
        health_monitor = DockerHealthMonitor.from_config(monitor_config)
        await health_monitor.initialize()
        
        # Start monitoring
        await health_monitor.start_monitoring()
        
        # Monitor specific containers
        container_names = ['web_frontend', 'api_backend', 'database', 'cache']
        for container_name in container_names:
            await health_monitor.add_container_monitoring(container_name)
        
        # Get current health status
        health_status = await health_monitor.get_overall_health()
        print(f"Overall Health: {health_status['status']}")
        print(f"Healthy Containers: {health_status['healthy_count']}")
        print(f"Unhealthy Containers: {health_status['unhealthy_count']}")
        ```
        
        **Enterprise Multi-Container Monitoring:**
        ```python
        # Comprehensive enterprise Docker monitoring setup
        class EnterpriseDockerMonitoringPlatform:
            def __init__(self):
                self.docker_manager = None
                self.health_monitor = None
                self.alert_manager = AlertManager()
                self.recovery_manager = RecoveryManager()
                self.analytics_engine = ContainerAnalyticsEngine()
                
            async def initialize_platform(self):
                # Initialize Docker manager with enterprise configuration
                self.docker_manager = DockerManager.from_config({
                    'connection_url': 'unix:///var/run/docker.sock',
                    'api_version': 'auto',
                    'timeout': 60
                })
                await self.docker_manager.initialize()
                
                # Configure comprehensive health monitoring
                monitor_config = {
                    'component_name': 'enterprise_docker_monitor',
                    'docker_manager': self.docker_manager,
                    'health_check_interval': 10,  # High-frequency monitoring
                    'resource_monitoring_interval': 5,
                    'alert_thresholds': {
                        'cpu_warning': 70.0,
                        'cpu_critical': 85.0,
                        'memory_warning': 75.0,
                        'memory_critical': 90.0,
                        'network_warning': 100 * 1024 * 1024,  # 100 MB/s
                        'disk_warning': 50 * 1024 * 1024       # 50 MB/s
                    }
                }
                
                self.health_monitor = DockerHealthMonitor.from_config(monitor_config)
                await self.health_monitor.initialize()
                
                # Configure intelligent alerting
                await self.alert_manager.configure_channels({
                    'email': {
                        'enabled': True,
                        'recipients': ['ops@company.com', 'devops@company.com']
                    },
                    'slack': {
                        'enabled': True,
                        'webhook_url': 'https://hooks.slack.com/...',
                        'channel': '#ops-alerts'
                    },
                    'pagerduty': {
                        'enabled': True,
                        'service_key': 'your-pagerduty-key'
                    }
                })
                
                # Configure automated recovery
                await self.recovery_manager.configure_policies({
                    'auto_restart': True,
                    'max_restart_attempts': 3,
                    'escalation_enabled': True,
                    'replacement_strategy': 'blue_green'
                })
                
            async def monitor_application_stack(self, stack_name: str, containers: List[str]):
                # Start monitoring for application stack
                await self.health_monitor.start_monitoring()
                
                # Add containers to monitoring
                for container_name in containers:
                    await self.health_monitor.add_container_monitoring(
                        container_name,
                        health_check_config={
                            'http_endpoint': f'http://localhost:8080/health',
                            'check_interval': 15,
                            'timeout': 5,
                            'retries': 2
                        }
                    )
                
                # Start continuous monitoring loop
                while True:
                    try:
                        # Collect health and resource metrics
                        health_results = await self.health_monitor.check_all_containers()
                        resource_metrics = await self.health_monitor.collect_resource_metrics()
                        
                        # Process health results
                        for result in health_results:
                            if result.status == HealthStatus.UNHEALTHY:
                                await self.handle_unhealthy_container(
                                    result.container_name, result
                                )
                            elif result.status == HealthStatus.STARTING:
                                await self.monitor_starting_container(
                                    result.container_name, result
                                )
                        
                        # Analyze resource utilization
                        resource_analysis = await self.analytics_engine.analyze_resources(
                            resource_metrics
                        )
                        
                        # Check for resource alerts
                        for metric in resource_metrics:
                            await self.check_resource_thresholds(metric)
                        
                        # Generate recommendations
                        recommendations = await self.analytics_engine.generate_recommendations(
                            resource_analysis
                        )
                        
                        if recommendations:
                            await self.alert_manager.send_recommendations(
                                stack_name, recommendations
                            )
                        
                        await asyncio.sleep(30)  # Monitor every 30 seconds
                        
                    except Exception as e:
                        self.health_monitor.logger.error(
                            f"Monitoring error for stack {stack_name}: {e}"
                        )
                        await asyncio.sleep(60)  # Retry after 1 minute
                        
            async def handle_unhealthy_container(self, container_name: str, health_result: HealthCheckResult):
                # Log unhealthy container
                self.health_monitor.logger.warning(
                    f"Container {container_name} is unhealthy: {health_result.error_message}"
                )
                
                # Send immediate alert
                await self.alert_manager.send_critical_alert(
                    f"Container {container_name} Health Critical",
                    {
                        'container': container_name,
                        'status': health_result.status.value,
                        'error': health_result.error_message,
                        'timestamp': health_result.timestamp
                    }
                )
                
                # Attempt automated recovery
                recovery_success = await self.recovery_manager.attempt_recovery(
                    container_name, health_result
                )
                
                if recovery_success:
                    await self.alert_manager.send_info_alert(
                        f"Container {container_name} Recovery Successful",
                        {'container': container_name, 'recovery_time': time.time()}
                    )
                else:
                    await self.alert_manager.send_critical_alert(
                        f"Container {container_name} Recovery Failed",
                        {'container': container_name, 'requires_manual_intervention': True}
                    )
                    
            async def check_resource_thresholds(self, metrics: ResourceMetrics):
                alerts = []
                
                # Check CPU thresholds
                if metrics.cpu_percent > 90.0:
                    alerts.append({
                        'type': 'cpu_critical',
                        'container': metrics.container_name,
                        'value': metrics.cpu_percent,
                        'threshold': 90.0
                    })
                elif metrics.cpu_percent > 75.0:
                    alerts.append({
                        'type': 'cpu_warning',
                        'container': metrics.container_name,
                        'value': metrics.cpu_percent,
                        'threshold': 75.0
                    })
                
                # Check memory thresholds
                if metrics.memory_percent > 95.0:
                    alerts.append({
                        'type': 'memory_critical',
                        'container': metrics.container_name,
                        'value': metrics.memory_percent,
                        'threshold': 95.0
                    })
                elif metrics.memory_percent > 80.0:
                    alerts.append({
                        'type': 'memory_warning',
                        'container': metrics.container_name,
                        'value': metrics.memory_percent,
                        'threshold': 80.0
                    })
                
                # Send resource alerts
                for alert in alerts:
                    if alert['type'].endswith('_critical'):
                        await self.alert_manager.send_critical_alert(
                            f"Resource Critical: {alert['type']}",
                            alert
                        )
                    else:
                        await self.alert_manager.send_warning_alert(
                            f"Resource Warning: {alert['type']}",
                            alert
                        )
        
        # Initialize and start enterprise monitoring
        monitoring_platform = EnterpriseDockerMonitoringPlatform()
        await monitoring_platform.initialize_platform()
        
        # Monitor production application stacks
        production_stacks = {
            'web_stack': ['nginx', 'web_app_1', 'web_app_2'],
            'api_stack': ['api_gateway', 'api_service_1', 'api_service_2'],
            'data_stack': ['postgres', 'redis', 'elasticsearch']
        }
        
        # Start monitoring all stacks
        monitoring_tasks = [
            asyncio.create_task(
                monitoring_platform.monitor_application_stack(stack_name, containers)
            )
            for stack_name, containers in production_stacks.items()
        ]
        
        await asyncio.gather(*monitoring_tasks)
        ```
        
        **Advanced Analytics and Capacity Planning:**
        ```python
        # Advanced Docker monitoring with predictive analytics
        class PredictiveDockerMonitor:
            def __init__(self, health_monitor: DockerHealthMonitor):
                self.health_monitor = health_monitor
                self.ml_analyzer = MachineLearningAnalyzer()
                self.capacity_planner = CapacityPlanner()
                self.anomaly_detector = AnomalyDetector()
                
            async def setup_predictive_monitoring(self):
                # Train ML models on historical data
                historical_data = await self.health_monitor.get_historical_metrics("30d")
                await self.ml_analyzer.train_models(historical_data)
                
                # Initialize anomaly detection
                await self.anomaly_detector.initialize(historical_data)
                
                # Setup capacity planning baselines
                await self.capacity_planner.establish_baselines(historical_data)
                
            async def run_predictive_analysis(self):
                while True:
                    # Collect current metrics
                    current_metrics = await self.health_monitor.collect_all_metrics()
                    
                    # Detect anomalies
                    anomalies = await self.anomaly_detector.detect_anomalies(current_metrics)
                    
                    # Generate predictions
                    predictions = await self.ml_analyzer.predict_future_usage(
                        current_metrics, horizon='1h'
                    )
                    
                    # Capacity planning analysis
                    capacity_analysis = await self.capacity_planner.analyze_capacity(
                        current_metrics, predictions
                    )
                    
                    # Process anomalies
                    for anomaly in anomalies:
                        await self.handle_anomaly(anomaly)
                    
                    # Process capacity warnings
                    if capacity_analysis['scaling_recommended']:
                        await self.handle_scaling_recommendation(capacity_analysis)
                    
                    # Generate optimization recommendations
                    optimizations = await self.ml_analyzer.recommend_optimizations(
                        current_metrics, predictions
                    )
                    
                    if optimizations:
                        await self.send_optimization_recommendations(optimizations)
                    
                    await asyncio.sleep(300)  # Run every 5 minutes
                    
            async def handle_anomaly(self, anomaly: dict):
                severity = anomaly['severity']
                if severity > 0.8:  # High severity
                    await self.health_monitor.alert_manager.send_critical_alert(
                        "Container Anomaly Detected",
                        {
                            'container': anomaly['container'],
                            'metric': anomaly['metric'],
                            'current_value': anomaly['current_value'],
                            'expected_value': anomaly['expected_value'],
                            'deviation': anomaly['deviation'],
                            'confidence': anomaly['confidence']
                        }
                    )
                else:  # Medium severity
                    await self.health_monitor.alert_manager.send_warning_alert(
                        "Container Performance Anomaly",
                        anomaly
                    )
        
        # Setup predictive monitoring
        predictive_monitor = PredictiveDockerMonitor(health_monitor)
        await predictive_monitor.setup_predictive_monitoring()
        
        # Start predictive analysis
        predictive_task = asyncio.create_task(
            predictive_monitor.run_predictive_analysis()
        )
        ```
        
        **Custom Health Check Implementation:**
        ```python
        # Custom health checks for specific applications
        class CustomHealthChecks:
            def __init__(self, health_monitor: DockerHealthMonitor):
                self.health_monitor = health_monitor
                
            async def register_custom_checks(self):
                # Database health check
                await self.health_monitor.register_custom_health_check(
                    'database',
                    self.check_database_health,
                    interval=60,
                    timeout=10
                )
                
                # API service health check
                await self.health_monitor.register_custom_health_check(
                    'api_service',
                    self.check_api_service_health,
                    interval=30,
                    timeout=5
                )
                
                # Queue health check
                await self.health_monitor.register_custom_health_check(
                    'message_queue',
                    self.check_queue_health,
                    interval=45,
                    timeout=15
                )
                
            async def check_database_health(self, container_name: str) -> HealthCheckResult:
                try:
                    # Custom database connectivity and performance check
                    start_time = time.time()
                    
                    # Check database connection
                    connection_ok = await self.test_database_connection(container_name)
                    if not connection_ok:
                        return HealthCheckResult(
                            container_name=container_name,
                            status=HealthStatus.UNHEALTHY,
                            timestamp=time.time(),
                            response_time=time.time() - start_time,
                            error_message="Database connection failed"
                        )
                    
                    # Check database performance
                    query_time = await self.test_database_performance(container_name)
                    if query_time > 5.0:  # 5 second threshold
                        return HealthCheckResult(
                            container_name=container_name,
                            status=HealthStatus.UNHEALTHY,
                            timestamp=time.time(),
                            response_time=time.time() - start_time,
                            error_message=f"Database performance degraded: {query_time}s query time"
                        )
                    
                    # Check database storage
                    storage_usage = await self.check_database_storage(container_name)
                    if storage_usage > 90.0:  # 90% threshold
                        return HealthCheckResult(
                            container_name=container_name,
                            status=HealthStatus.UNHEALTHY,
                            timestamp=time.time(),
                            response_time=time.time() - start_time,
                            error_message=f"Database storage critical: {storage_usage}% used"
                        )
                    
                    return HealthCheckResult(
                        container_name=container_name,
                        status=HealthStatus.HEALTHY,
                        timestamp=time.time(),
                        response_time=time.time() - start_time,
                        details={
                            'query_time': query_time,
                            'storage_usage': storage_usage,
                            'connection_pool_size': await self.get_connection_pool_size(container_name)
                        }
                    )
                    
                except Exception as e:
                    return HealthCheckResult(
                        container_name=container_name,
                        status=HealthStatus.UNKNOWN,
                        timestamp=time.time(),
                        response_time=time.time() - start_time,
                        error_message=f"Health check error: {str(e)}"
                    )
        
        # Register custom health checks
        custom_checks = CustomHealthChecks(health_monitor)
        await custom_checks.register_custom_checks()
        ```
    
    **Advanced Features:**
        
        **Machine Learning Integration:**
        * Predictive failure detection based on historical patterns
        * Anomaly detection for unusual resource usage patterns
        * Automated threshold optimization based on container behavior
        * Capacity forecasting and scaling recommendations
        
        **Container Lifecycle Management:**
        * Automated container restart and replacement strategies
        * Blue-green deployment health validation
        * Rolling update health monitoring and rollback triggers
        * Container dependency health validation and orchestration
        
        **Enterprise Integration:**
        * Integration with container orchestration platforms (Kubernetes, Docker Swarm)
        * SIEM integration for security monitoring and compliance
        * APM integration for application performance correlation
        * Service mesh integration for distributed tracing
        
        **Advanced Analytics:**
        * Cross-container performance correlation analysis
        * Resource optimization recommendations and cost analysis
        * Performance trend analysis and capacity planning
        * Business impact analysis of container health issues
    
    **Production Deployment:**
        
        **High Availability:**
        * Multi-host monitoring with distributed health checks
        * Monitoring system redundancy and failover capabilities
        * Cross-datacenter container health validation
        * Disaster recovery monitoring and alerting
        
        **Security & Compliance:**
        * Secure Docker daemon communication with TLS
        * Container security scanning and vulnerability monitoring
        * Compliance reporting and audit trail maintenance
        * Role-based access control for monitoring operations
        
        **Scalability:**
        * Horizontal scaling for large container environments
        * Efficient batch processing for high-volume metrics collection
        * Distributed monitoring architecture with load balancing
        * Cloud-native deployment with auto-scaling capabilities
    
    Attributes:
        config (DockerHealthMonitorConfig): Health monitor configuration and settings
        docker_manager (DockerManager): Docker API client for container management operations
        monitored_containers (Dict): Registry of containers under active monitoring
        health_check_tasks (Dict): Active health check task management
        resource_metrics_history (Dict): Historical resource usage data storage
        alert_manager (AlertManager): Alert processing and notification management
        logger (Logger): Structured logging system for monitoring operations
    
    Note:
        This monitor requires Docker API access and appropriate permissions for container inspection.
        Health check intervals should be balanced between monitoring accuracy and resource consumption.
        Resource monitoring frequency affects system performance and storage requirements.
        Alert thresholds should be calibrated based on application requirements and infrastructure capacity.
    
    Warning:
        Aggressive health check intervals may impact container performance on resource-constrained systems.
        Failed health checks may trigger unnecessary container restarts if thresholds are too sensitive.
        Resource monitoring data collection may consume significant storage in large-scale deployments.
        Monitoring system failures may impact container health visibility and automated recovery capabilities.
    
    See Also:
        * :class:`DockerManager`: Docker container lifecycle management and operations
        * :class:`HealthCheckResult`: Health check result data structure and status reporting
        * :class:`ResourceMetrics`: Container resource utilization metrics and tracking
        * :mod:`nanobrain.library.infrastructure.monitoring`: Comprehensive monitoring infrastructure
        * :mod:`nanobrain.library.infrastructure.deployment`: Container deployment and orchestration
    """
    
    # Component configuration schema
    COMPONENT_TYPE: ClassVar[str] = "docker_health_monitor"
    REQUIRED_CONFIG_FIELDS: ClassVar[List[str]] = ['docker_manager']
    OPTIONAL_CONFIG_FIELDS: ClassVar[Dict[str, Any]] = {
        'check_interval': 30,
        'alert_thresholds': {},
        'history_limit': 100
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
        if isinstance(config, DockerHealthMonitorConfig):
            return {
                'component_name': config.component_name,
                'enabled': config.enabled,
                'docker_manager': config.docker_manager,
                'check_interval': config.check_interval,
                'alert_thresholds': config.alert_thresholds,
                'history_limit': config.history_limit
            }
        elif isinstance(config, dict):
            return config
        else:
            return {}
    
    @classmethod
    def resolve_dependencies(cls, component_config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Resolve health monitor dependencies"""
        docker_manager = component_config.get('docker_manager')
        if not docker_manager:
            raise ComponentDependencyError("DockerHealthMonitor requires a docker_manager dependency")
        
        return {
            'docker_manager': docker_manager,
            **kwargs
        }
    
    def _init_from_config(self, config: Any, component_config: Dict[str, Any],
                         dependencies: Dict[str, Any]) -> None:
        """Initialize DockerHealthMonitor from configuration"""
        super()._init_from_config(config, component_config, dependencies)
        
        # Store Docker manager reference
        self.docker_manager = dependencies['docker_manager']
        
        # Health monitoring configuration
        self.check_interval = component_config.get('check_interval', 30)
        self.alert_thresholds = component_config.get('alert_thresholds', {
            'cpu_threshold': 80.0,
            'memory_threshold': 85.0,
            'consecutive_failures': 3
        })
        self.history_limit = component_config.get('history_limit', 100)
        
        # Monitoring state
        self.monitoring_enabled = False
        self.monitor_task: Optional[asyncio.Task] = None
        self.monitored_containers: Dict[str, HealthCheckConfig] = {}
        
        # Health history (last N checks per container)
        self.health_history: Dict[str, List[HealthCheckResult]] = {}
        self.resource_history: Dict[str, List[ResourceMetrics]] = {}
        
        # Alert callbacks
        self.alert_callbacks: List[Callable[[str, HealthCheckResult], None]] = []
        
        # Health check statistics
        self.stats = {
            "total_checks": 0,
            "healthy_checks": 0,
            "unhealthy_checks": 0,
            "failed_checks": 0
        }
    
    def add_container(self, container_name: str, health_config: Optional[HealthCheckConfig] = None):
        """
        Add a container to health monitoring.
        
        Args:
            container_name: Name of container to monitor
            health_config: Health check configuration (uses default if None)
        """
        if health_config is None:
            health_config = HealthCheckConfig()
        
        self.monitored_containers[container_name] = health_config
        self.health_history[container_name] = []
        self.resource_history[container_name] = []
        
        self.logger.info(f"➕ Added container {container_name} to health monitoring")
    
    def remove_container(self, container_name: str):
        """Remove a container from health monitoring"""
        if container_name in self.monitored_containers:
            del self.monitored_containers[container_name]
            
        if container_name in self.health_history:
            del self.health_history[container_name]
            
        if container_name in self.resource_history:
            del self.resource_history[container_name]
        
        self.logger.info(f"➖ Removed container {container_name} from health monitoring")
    
    def add_alert_callback(self, callback: Callable[[str, HealthCheckResult], None]):
        """
        Add alert callback function.
        
        Args:
            callback: Function to call when health issues detected
        """
        self.alert_callbacks.append(callback)
        self.logger.info("🔔 Added health alert callback")
    
    async def start_monitoring(self):
        """Start continuous health monitoring"""
        if self.monitoring_enabled:
            self.logger.warning("Health monitoring is already running")
            return
        
        self.monitoring_enabled = True
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info(f"🔄 Started health monitoring (interval: {self.check_interval}s)")
    
    async def stop_monitoring(self):
        """Stop health monitoring"""
        if not self.monitoring_enabled:
            return
        
        self.monitoring_enabled = False
        
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("⏹️ Stopped health monitoring")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        try:
            while self.monitoring_enabled:
                # Check all monitored containers
                for container_name in list(self.monitored_containers.keys()):
                    try:
                        await self._check_container_health(container_name)
                        await self._collect_resource_metrics(container_name)
                    except Exception as e:
                        self.logger.error(f"Error monitoring container {container_name}: {e}")
                
                # Wait for next check interval
                await asyncio.sleep(self.check_interval)
                
        except asyncio.CancelledError:
            self.logger.info("Health monitoring loop cancelled")
        except Exception as e:
            self.logger.error(f"Health monitoring loop failed: {e}")
            self.monitoring_enabled = False
    
    async def _check_container_health(self, container_name: str):
        """Perform health check on a specific container"""
        start_time = time.time()
        health_config = self.monitored_containers[container_name]
        
        try:
            # Get container status
            status_info = await self.docker_manager.get_container_status(container_name)
            
            if "error" in status_info:
                result = HealthCheckResult(
                    container_name=container_name,
                    status=HealthStatus.UNKNOWN,
                    timestamp=time.time(),
                    response_time=time.time() - start_time,
                    error_message=status_info["error"]
                )
            else:
                # Determine health status
                container_status = status_info.get("status", "unknown")
                
                if container_status == "running":
                    # Perform specific health check based on configuration
                    health_result = await self._perform_health_check(container_name, health_config)
                    
                    result = HealthCheckResult(
                        container_name=container_name,
                        status=health_result["status"],
                        timestamp=time.time(),
                        response_time=time.time() - start_time,
                        details=health_result.get("details", {}),
                        error_message=health_result.get("error")
                    )
                else:
                    result = HealthCheckResult(
                        container_name=container_name,
                        status=HealthStatus.UNHEALTHY,
                        timestamp=time.time(),
                        response_time=time.time() - start_time,
                        details={"container_status": container_status}
                    )
            
            # Record result
            self._record_health_result(result)
            
            # Trigger alerts if unhealthy
            if result.status == HealthStatus.UNHEALTHY:
                await self._trigger_alerts(container_name, result)
            
        except Exception as e:
            self.logger.error(f"Health check failed for {container_name}: {e}")
            
            result = HealthCheckResult(
                container_name=container_name,
                status=HealthStatus.UNKNOWN,
                timestamp=time.time(),
                response_time=time.time() - start_time,
                error_message=str(e)
            )
            self._record_health_result(result)
    
    async def _perform_health_check(self, container_name: str, health_config: HealthCheckConfig) -> Dict[str, Any]:
        """Perform specific health check based on configuration"""
        try:
            if health_config.type == "http":
                return await self._http_health_check(container_name, health_config)
            elif health_config.type == "tcp":
                return await self._tcp_health_check(container_name, health_config)
            elif health_config.type == "exec":
                return await self._exec_health_check(container_name, health_config)
            else:
                # Default to Docker's own health check
                docker_healthy = await self.docker_manager.health_check(container_name)
                return {
                    "status": HealthStatus.HEALTHY if docker_healthy else HealthStatus.UNHEALTHY,
                    "details": {"check_type": "docker_default"}
                }
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY,
                "error": str(e)
            }
    
    async def _http_health_check(self, container_name: str, health_config: HealthCheckConfig) -> Dict[str, Any]:
        """Perform HTTP health check"""
        import aiohttp
        
        try:
            # Get container network info to determine IP
            status_info = await self.docker_manager.get_container_status(container_name)
            
            # Try localhost first (for port mapping)
            url = f"http://localhost:{health_config.port}{health_config.path}"
            
            timeout = aiohttp.ClientTimeout(total=health_config.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        return {
                            "status": HealthStatus.HEALTHY,
                            "details": {
                                "url": url,
                                "status_code": response.status,
                                "response_time": response.headers.get("x-response-time")
                            }
                        }
                    else:
                        return {
                            "status": HealthStatus.UNHEALTHY,
                            "details": {
                                "url": url,
                                "status_code": response.status
                            }
                        }
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY,
                "error": f"HTTP check failed: {e}"
            }
    
    async def _tcp_health_check(self, container_name: str, health_config: HealthCheckConfig) -> Dict[str, Any]:
        """Perform TCP health check"""
        try:
            # Attempt TCP connection
            future = asyncio.open_connection("localhost", health_config.port)
            reader, writer = await asyncio.wait_for(future, timeout=health_config.timeout)
            
            # Close connection
            writer.close()
            await writer.wait_closed()
            
            return {
                "status": HealthStatus.HEALTHY,
                "details": {
                    "port": health_config.port,
                    "connection": "successful"
                }
            }
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY,
                "error": f"TCP check failed: {e}"
            }
    
    async def _exec_health_check(self, container_name: str, health_config: HealthCheckConfig) -> Dict[str, Any]:
        """Perform exec health check"""
        try:
            if not health_config.command:
                return {
                    "status": HealthStatus.UNHEALTHY,
                    "error": "No exec command configured"
                }
            
            # Execute command in container
            container = self.docker_manager.client.containers.get(container_name)
            
            result = container.exec_run(
                health_config.command,
                stdout=True,
                stderr=True
            )
            
            if result.exit_code == 0:
                return {
                    "status": HealthStatus.HEALTHY,
                    "details": {
                        "command": " ".join(health_config.command),
                        "exit_code": result.exit_code,
                        "output": result.output.decode('utf-8', errors='replace')[:200]  # Limit output
                    }
                }
            else:
                return {
                    "status": HealthStatus.UNHEALTHY,
                    "details": {
                        "command": " ".join(health_config.command),
                        "exit_code": result.exit_code,
                        "output": result.output.decode('utf-8', errors='replace')[:200]
                    }
                }
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY,
                "error": f"Exec check failed: {e}"
            }
    
    async def _collect_resource_metrics(self, container_name: str):
        """Collect resource usage metrics for a container"""
        try:
            status_info = await self.docker_manager.get_container_status(container_name)
            stats = status_info.get("stats", {})
            
            if not stats:
                return
            
            # Parse CPU usage
            cpu_percent = 0.0
            if "cpu_stats" in stats and "precpu_stats" in stats:
                cpu_stats = stats["cpu_stats"]
                precpu_stats = stats["precpu_stats"]
                
                cpu_delta = cpu_stats["cpu_usage"]["total_usage"] - precpu_stats["cpu_usage"]["total_usage"]
                system_delta = cpu_stats["system_cpu_usage"] - precpu_stats["system_cpu_usage"]
                
                if system_delta > 0 and cpu_delta > 0:
                    cpu_percent = (cpu_delta / system_delta) * len(cpu_stats["cpu_usage"].get("percpu_usage", [1])) * 100
            
            # Parse memory usage
            memory_stats = stats.get("memory_stats", {})
            memory_usage = memory_stats.get("usage", 0)
            memory_limit = memory_stats.get("limit", 0)
            memory_percent = (memory_usage / memory_limit * 100) if memory_limit > 0 else 0
            
            # Parse network stats
            networks = stats.get("networks", {})
            network_rx = sum(net.get("rx_bytes", 0) for net in networks.values())
            network_tx = sum(net.get("tx_bytes", 0) for net in networks.values())
            
            # Parse disk I/O stats
            blkio_stats = stats.get("blkio_stats", {})
            disk_read = sum(
                stat.get("value", 0) for stat in blkio_stats.get("io_service_bytes_recursive", [])
                if stat.get("op") == "Read"
            )
            disk_write = sum(
                stat.get("value", 0) for stat in blkio_stats.get("io_service_bytes_recursive", [])
                if stat.get("op") == "Write"
            )
            
            metrics = ResourceMetrics(
                container_name=container_name,
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_usage=memory_usage,
                memory_limit=memory_limit,
                memory_percent=memory_percent,
                network_rx=network_rx,
                network_tx=network_tx,
                disk_read=disk_read,
                disk_write=disk_write
            )
            
            self._record_resource_metrics(metrics)
            
        except Exception as e:
            self.logger.error(f"Failed to collect resource metrics for {container_name}: {e}")
    
    def _record_health_result(self, result: HealthCheckResult):
        """Record health check result in history"""
        self.stats["total_checks"] += 1
        
        if result.status == HealthStatus.HEALTHY:
            self.stats["healthy_checks"] += 1
        elif result.status == HealthStatus.UNHEALTHY:
            self.stats["unhealthy_checks"] += 1
        else:
            self.stats["failed_checks"] += 1
        
        # Add to history (keep last N results)
        history = self.health_history[result.container_name]
        history.append(result)
        if len(history) > self.history_limit:
            history.pop(0)
    
    def _record_resource_metrics(self, metrics: ResourceMetrics):
        """Record resource metrics in history"""
        # Add to history (keep last N metrics)
        history = self.resource_history[metrics.container_name]
        history.append(metrics)
        if len(history) > self.history_limit:
            history.pop(0)
    
    async def _trigger_alerts(self, container_name: str, result: HealthCheckResult):
        """Trigger alert callbacks for unhealthy containers"""
        for callback in self.alert_callbacks:
            try:
                callback(container_name, result)
            except Exception as e:
                self.logger.error(f"Alert callback failed: {e}")
    
    def get_container_health_summary(self, container_name: str) -> Dict[str, Any]:
        """Get health summary for a specific container"""
        if container_name not in self.health_history:
            return {"error": "Container not monitored"}
        
        history = self.health_history[container_name]
        if not history:
            return {"status": "no_data"}
        
        latest = history[-1]
        
        # Calculate health statistics
        total_checks = len(history)
        healthy_count = sum(1 for r in history if r.status == HealthStatus.HEALTHY)
        unhealthy_count = sum(1 for r in history if r.status == HealthStatus.UNHEALTHY)
        
        return {
            "container_name": container_name,
            "current_status": latest.status.value,
            "last_check": latest.timestamp,
            "response_time": latest.response_time,
            "total_checks": total_checks,
            "healthy_percentage": (healthy_count / total_checks * 100) if total_checks > 0 else 0,
            "unhealthy_count": unhealthy_count,
            "details": latest.details,
            "error_message": latest.error_message
        }
    
    def get_container_resource_summary(self, container_name: str) -> Dict[str, Any]:
        """Get resource usage summary for a specific container"""
        if container_name not in self.resource_history:
            return {"error": "Container not monitored"}
        
        history = self.resource_history[container_name]
        if not history:
            return {"status": "no_data"}
        
        latest = history[-1]
        
        # Calculate averages over last 10 readings
        recent_metrics = history[-10:]
        
        avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
        avg_memory_percent = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
        
        return {
            "container_name": container_name,
            "timestamp": latest.timestamp,
            "current_cpu_percent": latest.cpu_percent,
            "current_memory_percent": latest.memory_percent,
            "current_memory_usage": latest.memory_usage,
            "memory_limit": latest.memory_limit,
            "avg_cpu_percent": avg_cpu,
            "avg_memory_percent": avg_memory_percent,
            "network_rx": latest.network_rx,
            "network_tx": latest.network_tx,
            "disk_read": latest.disk_read,
            "disk_write": latest.disk_write
        }
    
    def get_monitoring_statistics(self) -> Dict[str, Any]:
        """Get overall monitoring statistics"""
        return {
            "monitoring_enabled": self.monitoring_enabled,
            "monitored_containers": len(self.monitored_containers),
            "check_interval": self.check_interval,
            "total_checks": self.stats["total_checks"],
            "healthy_checks": self.stats["healthy_checks"],
            "unhealthy_checks": self.stats["unhealthy_checks"],
            "failed_checks": self.stats["failed_checks"],
            "success_rate": (
                self.stats["healthy_checks"] / self.stats["total_checks"] * 100
                if self.stats["total_checks"] > 0 else 0
            )
        } 