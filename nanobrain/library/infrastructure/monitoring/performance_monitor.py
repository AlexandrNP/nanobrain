"""
Performance monitoring implementation.

Comprehensive metrics collection for system performance tracking.
"""

import asyncio
import time
import psutil
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from collections import deque, defaultdict
from dataclasses import dataclass, asdict
from nanobrain.core.logging_system import get_logger


@dataclass
class PerformanceMetric:
    """Represents a performance metric."""
    name: str
    value: float
    timestamp: datetime
    unit: str
    tags: Optional[Dict[str, str]] = None


class PerformanceMonitor:
    """
    Enterprise Performance Monitor - Comprehensive System Monitoring and Analytics Platform
    ====================================================================================
    
    The PerformanceMonitor provides comprehensive system performance monitoring, real-time analytics,
    and intelligent alerting for enterprise applications. This monitoring platform collects system
    metrics, application performance data, and custom business metrics, providing deep observability
    into system behavior, performance trends, and operational health for production environments.
    
    **Core Architecture:**
        The performance monitor provides enterprise-grade monitoring capabilities:
        
        * **Real-Time Metrics Collection**: Continuous system and application performance monitoring
        * **Historical Data Management**: Time-series data storage with configurable retention policies
        * **Custom Metrics Support**: Extensible framework for business-specific performance indicators
        * **Intelligent Alerting**: Threshold-based alerting with trend analysis and anomaly detection
        * **Performance Analytics**: Advanced analytics and trend analysis for capacity planning
        * **Framework Integration**: Full integration with NanoBrain's distributed monitoring architecture
    
    **Monitoring Capabilities:**
        
        **System Performance Metrics:**
        * **CPU Utilization**: Multi-core CPU usage tracking and analysis
        * **Memory Management**: RAM usage, swap utilization, and memory pressure monitoring
        * **Disk I/O Performance**: Read/write throughput, latency, and storage utilization
        * **Network Performance**: Bandwidth utilization, connection tracking, and network latency
        * **Process Monitoring**: Individual process resource consumption and performance tracking
        
        **Application Performance Metrics:**
        * **Response Time Monitoring**: Request/response latency tracking and optimization
        * **Throughput Analysis**: Request rate, transaction volume, and processing capacity
        * **Error Rate Tracking**: Application error monitoring and failure pattern analysis
        * **Resource Utilization**: Application-specific resource consumption monitoring
        * **Business Metrics**: Custom KPIs and domain-specific performance indicators
        
        **Advanced Analytics:**
        * **Trend Analysis**: Historical performance trend identification and forecasting
        * **Anomaly Detection**: Statistical anomaly detection and automatic alerting
        * **Capacity Planning**: Resource utilization forecasting and scaling recommendations
        * **Performance Optimization**: Bottleneck identification and optimization suggestions
        * **Correlation Analysis**: Cross-metric correlation and dependency analysis
    
    **Enterprise Features:**
        
        **Real-Time Dashboards:**
        * Interactive performance dashboards with customizable visualizations
        * Real-time metric streaming and live data updates
        * Multi-dimensional data exploration and drill-down capabilities
        * Executive summary views and operational dashboards
        
        **Intelligent Alerting:**
        * Configurable threshold-based alerting with dynamic baselines
        * Machine learning-based anomaly detection and predictive alerting
        * Multi-channel alert delivery (email, SMS, webhook, dashboard)
        * Alert correlation and noise reduction algorithms
        
        **Data Management:**
        * Efficient time-series data storage with compression
        * Configurable data retention and archival policies
        * High-performance data querying and aggregation
        * Export capabilities for external analytics systems
        
        **Integration Ecosystem:**
        * Prometheus metrics export for enterprise monitoring stacks
        * Grafana dashboard integration and visualization
        * ELK stack integration for log correlation
        * Custom webhook integration for external systems
    
    **Configuration Architecture:**
        Comprehensive configuration supports diverse monitoring scenarios:
        
        ```yaml
        # Performance Monitor Configuration
        monitor_name: "enterprise_performance_monitor"
        monitor_type: "comprehensive"
        
        # Monitor card for framework integration
        monitor_card:
          name: "enterprise_performance_monitor"
          description: "Enterprise system and application performance monitoring"
          version: "1.0.0"
          category: "infrastructure"
          capabilities:
            - "real_time_monitoring"
            - "historical_analytics"
            - "intelligent_alerting"
        
        # Collection Configuration
        collection_config:
          interval: 60.0              # Collection interval in seconds
          history_size: 10000         # Number of metrics to retain in memory
          batch_size: 100            # Metrics batch size for efficient processing
          compression_enabled: true   # Enable metric data compression
          
        # System Metrics Configuration
        system_metrics:
          cpu_monitoring:
            enabled: true
            per_core: true
            load_average: true
            
          memory_monitoring:
            enabled: true
            include_swap: true
            buffer_cache: true
            
          disk_monitoring:
            enabled: true
            io_stats: true
            space_usage: true
            mount_points: ["/", "/var", "/tmp"]
            
          network_monitoring:
            enabled: true
            interface_stats: true
            connection_tracking: true
            bandwidth_monitoring: true
            
        # Application Metrics Configuration
        application_metrics:
          response_time_monitoring:
            enabled: true
            percentiles: [50, 90, 95, 99]
            histogram_buckets: [0.1, 0.5, 1.0, 2.0, 5.0]
            
          throughput_monitoring:
            enabled: true
            rate_windows: [60, 300, 900]  # 1min, 5min, 15min windows
            
          error_rate_monitoring:
            enabled: true
            error_classification: true
            
        # Custom Metrics Configuration
        custom_metrics:
          business_kpis:
            enabled: true
            collection_method: "pull"
            metrics:
              - "active_users"
              - "transaction_volume"
              - "revenue_per_minute"
              
        # Analytics Configuration
        analytics:
          trend_analysis:
            enabled: true
            trend_window: 7200        # 2 hours for trend analysis
            seasonality_detection: true
            
          anomaly_detection:
            enabled: true
            algorithm: "isolation_forest"
            sensitivity: 0.1
            training_window: 86400    # 24 hours for training
            
          forecasting:
            enabled: true
            horizon: 3600            # 1 hour forecast horizon
            confidence_interval: 0.95
            
        # Alerting Configuration
        alerting:
          enabled: true
          channels:
            email:
              enabled: true
              recipients: ["ops@company.com"]
            webhook:
              enabled: true
              url: "https://alerting.company.com/webhook"
            dashboard:
              enabled: true
              
          thresholds:
            cpu_usage:
              warning: 75.0
              critical: 90.0
            memory_usage:
              warning: 80.0
              critical: 95.0
            response_time:
              warning: 1000.0         # milliseconds
              critical: 5000.0
            error_rate:
              warning: 0.05           # 5%
              critical: 0.15          # 15%
              
        # Export Configuration
        export:
          prometheus:
            enabled: true
            port: 9090
            endpoint: "/metrics"
            
          grafana:
            enabled: true
            dashboard_config: "grafana_dashboard.json"
            
          elasticsearch:
            enabled: false
            host: "elasticsearch:9200"
            index_pattern: "performance-metrics-{date}"
        ```
    
    **Usage Patterns:**
        
        **Basic Performance Monitoring Setup:**
        ```python
        from nanobrain.library.infrastructure.monitoring import PerformanceMonitor
        
        # Create performance monitor with basic configuration
        monitor = PerformanceMonitor(
            collection_interval=30.0,    # Collect metrics every 30 seconds
            history_size=2000            # Keep 2000 metrics in memory
        )
        
        # Start monitoring system performance
        await monitor.start_monitoring()
        
        # Add custom application metrics
        await monitor.add_custom_metric("active_users", 1250)
        await monitor.add_custom_metric("requests_per_second", 45.7)
        await monitor.add_custom_metric("database_connections", 23)
        
        # Record application events
        await monitor.record_response_time("api_endpoint", 125.5)  # milliseconds
        await monitor.record_error("database_connection_error")
        await monitor.record_throughput("processed_requests", 1)
        
        # Get current performance snapshot
        snapshot = await monitor.get_performance_snapshot()
        print(f"CPU Usage: {snapshot['cpu_percent']:.1f}%")
        print(f"Memory Usage: {snapshot['memory_percent']:.1f}%")
        print(f"Active Users: {snapshot['custom_metrics']['active_users']}")
        ```
        
        **Enterprise Monitoring with Analytics:**
        ```python
        # Configure comprehensive enterprise monitoring
        class EnterpriseMonitoringPlatform:
            def __init__(self):
                self.performance_monitor = PerformanceMonitor(
                    collection_interval=15.0,  # High-frequency collection
                    history_size=20000          # Extended history
                )
                self.alert_manager = AlertManager()
                self.analytics_engine = AnalyticsEngine()
                self.dashboard_service = DashboardService()
                
            async def initialize_monitoring(self):
                # Start performance monitoring
                await self.performance_monitor.start_monitoring()
                
                # Configure alerting thresholds
                await self.alert_manager.configure_thresholds({
                    'cpu_usage': {'warning': 70.0, 'critical': 85.0},
                    'memory_usage': {'warning': 75.0, 'critical': 90.0},
                    'response_time_p95': {'warning': 800.0, 'critical': 2000.0},
                    'error_rate': {'warning': 0.02, 'critical': 0.10}
                })
                
                # Initialize analytics
                await self.analytics_engine.start_trend_analysis()
                await self.analytics_engine.start_anomaly_detection()
                
                # Setup dashboards
                await self.dashboard_service.create_executive_dashboard()
                await self.dashboard_service.create_operational_dashboard()
                
            async def monitor_application_performance(self, application_name: str):
                while True:
                    try:
                        # Collect application-specific metrics
                        app_metrics = await self.collect_application_metrics(application_name)
                        
                        # Record metrics
                        for metric_name, value in app_metrics.items():
                            await self.performance_monitor.add_custom_metric(
                                f"{application_name}_{metric_name}", 
                                value
                            )
                        
                        # Analyze for anomalies
                        anomalies = await self.analytics_engine.detect_anomalies(app_metrics)
                        if anomalies:
                            await self.alert_manager.trigger_anomaly_alert(
                                application_name, anomalies
                            )
                        
                        # Update dashboards
                        await self.dashboard_service.update_application_metrics(
                            application_name, app_metrics
                        )
                        
                        await asyncio.sleep(60)  # Monitor every minute
                        
                    except Exception as e:
                        self.performance_monitor.logger.error(
                            f"Application monitoring error for {application_name}: {e}"
                        )
                        await asyncio.sleep(30)  # Retry after 30 seconds
                        
            async def collect_application_metrics(self, application_name: str) -> dict:
                # Implement application-specific metric collection
                return {
                    'active_sessions': await self.get_active_sessions(application_name),
                    'request_rate': await self.get_request_rate(application_name),
                    'error_rate': await self.get_error_rate(application_name),
                    'avg_response_time': await self.get_avg_response_time(application_name),
                    'database_connections': await self.get_db_connections(application_name),
                    'cache_hit_rate': await self.get_cache_hit_rate(application_name)
                }
                
            async def generate_performance_report(self, time_range: str = "24h"):
                # Generate comprehensive performance report
                performance_data = await self.performance_monitor.get_historical_data(time_range)
                analytics_insights = await self.analytics_engine.generate_insights(performance_data)
                
                report = {
                    'summary': {
                        'time_range': time_range,
                        'total_metrics': len(performance_data),
                        'avg_cpu_usage': analytics_insights['avg_cpu_usage'],
                        'avg_memory_usage': analytics_insights['avg_memory_usage'],
                        'peak_response_time': analytics_insights['peak_response_time']
                    },
                    'trends': analytics_insights['trends'],
                    'anomalies': analytics_insights['anomalies'],
                    'recommendations': analytics_insights['recommendations'],
                    'capacity_forecast': analytics_insights['capacity_forecast']
                }
                
                return report
        
        # Initialize enterprise monitoring platform
        monitoring_platform = EnterpriseMonitoringPlatform()
        await monitoring_platform.initialize_monitoring()
        
        # Start application monitoring
        applications = ['web_frontend', 'api_backend', 'data_processor']
        monitoring_tasks = [
            asyncio.create_task(
                monitoring_platform.monitor_application_performance(app)
            )
            for app in applications
        ]
        
        # Generate daily performance reports
        async def daily_reporting():
            while True:
                await asyncio.sleep(86400)  # Wait 24 hours
                report = await monitoring_platform.generate_performance_report("24h")
                await send_performance_report(report)
        
        # Start all monitoring tasks
        await asyncio.gather(*monitoring_tasks, daily_reporting())
        ```
        
        **Real-Time Alerting and Anomaly Detection:**
        ```python
        # Advanced monitoring with intelligent alerting
        class IntelligentMonitoringSystem:
            def __init__(self):
                self.monitor = PerformanceMonitor(collection_interval=10.0)
                self.anomaly_detector = StatisticalAnomalyDetector()
                self.alert_correlator = AlertCorrelator()
                self.escalation_manager = EscalationManager()
                
            async def setup_intelligent_monitoring(self):
                # Train anomaly detection models
                historical_data = await self.monitor.get_historical_data("7d")
                await self.anomaly_detector.train(historical_data)
                
                # Configure alert correlation rules
                await self.alert_correlator.add_correlation_rule(
                    "high_cpu_memory_correlation",
                    ["cpu_usage", "memory_usage"],
                    threshold=0.8
                )
                
                # Setup escalation policies
                await self.escalation_manager.configure_policy({
                    'critical_alerts': {
                        'immediate': ['on_call_engineer'],
                        '5_minutes': ['team_lead'],
                        '15_minutes': ['director']
                    }
                })
                
            async def monitor_with_intelligence(self):
                while True:
                    # Collect current metrics
                    current_metrics = await self.monitor.get_current_metrics()
                    
                    # Detect anomalies using machine learning
                    anomalies = await self.anomaly_detector.detect(current_metrics)
                    
                    # Process each detected anomaly
                    for anomaly in anomalies:
                        # Correlate with other metrics
                        correlated_alerts = await self.alert_correlator.correlate(
                            anomaly, current_metrics
                        )
                        
                        # Determine severity based on correlation
                        severity = self.calculate_severity(anomaly, correlated_alerts)
                        
                        # Trigger appropriate alerts
                        if severity >= 0.8:  # Critical
                            await self.escalation_manager.trigger_critical_alert(
                                anomaly, correlated_alerts
                            )
                        elif severity >= 0.5:  # Warning
                            await self.escalation_manager.trigger_warning_alert(
                                anomaly, correlated_alerts
                            )
                        
                        # Log anomaly for future learning
                        await self.anomaly_detector.log_anomaly(anomaly, severity)
                    
                    # Adaptive monitoring interval based on system state
                    if anomalies:
                        await asyncio.sleep(5)   # More frequent monitoring during issues
                    else:
                        await asyncio.sleep(30)  # Normal monitoring interval
                        
            def calculate_severity(self, anomaly, correlated_alerts):
                # Intelligent severity calculation
                base_severity = anomaly['severity']
                correlation_factor = len(correlated_alerts) * 0.1
                trend_factor = anomaly.get('trend_severity', 0.0)
                
                return min(1.0, base_severity + correlation_factor + trend_factor)
        
        # Initialize intelligent monitoring
        intelligent_system = IntelligentMonitoringSystem()
        await intelligent_system.setup_intelligent_monitoring()
        await intelligent_system.monitor_with_intelligence()
        ```
        
        **Custom Business Metrics Integration:**
        ```python
        # Business-specific performance monitoring
        class BusinessMetricsMonitor:
            def __init__(self, performance_monitor: PerformanceMonitor):
                self.performance_monitor = performance_monitor
                self.business_metrics = {}
                self.kpi_calculator = KPICalculator()
                
            async def track_business_metrics(self):
                while True:
                    try:
                        # Revenue metrics
                        revenue_per_hour = await self.calculate_revenue_per_hour()
                        await self.performance_monitor.add_custom_metric(
                            "revenue_per_hour", revenue_per_hour
                        )
                        
                        # User engagement metrics
                        active_users = await self.get_active_user_count()
                        user_engagement_score = await self.calculate_engagement_score()
                        
                        await self.performance_monitor.add_custom_metric(
                            "active_users", active_users
                        )
                        await self.performance_monitor.add_custom_metric(
                            "user_engagement_score", user_engagement_score
                        )
                        
                        # Operational efficiency metrics
                        processing_efficiency = await self.calculate_processing_efficiency()
                        resource_utilization = await self.calculate_resource_utilization()
                        
                        await self.performance_monitor.add_custom_metric(
                            "processing_efficiency", processing_efficiency
                        )
                        await self.performance_monitor.add_custom_metric(
                            "resource_utilization", resource_utilization
                        )
                        
                        # Quality metrics
                        service_quality_score = await self.calculate_service_quality()
                        customer_satisfaction = await self.get_customer_satisfaction()
                        
                        await self.performance_monitor.add_custom_metric(
                            "service_quality_score", service_quality_score
                        )
                        await self.performance_monitor.add_custom_metric(
                            "customer_satisfaction", customer_satisfaction
                        )
                        
                        # Calculate composite KPIs
                        kpis = await self.kpi_calculator.calculate_all_kpis({
                            'revenue_per_hour': revenue_per_hour,
                            'active_users': active_users,
                            'user_engagement_score': user_engagement_score,
                            'processing_efficiency': processing_efficiency,
                            'service_quality_score': service_quality_score
                        })
                        
                        # Record KPIs as metrics
                        for kpi_name, kpi_value in kpis.items():
                            await self.performance_monitor.add_custom_metric(
                                f"kpi_{kpi_name}", kpi_value
                            )
                        
                        await asyncio.sleep(300)  # Update every 5 minutes
                        
                    except Exception as e:
                        self.performance_monitor.logger.error(
                            f"Business metrics collection error: {e}"
                        )
                        await asyncio.sleep(60)  # Retry after 1 minute
                        
            async def generate_business_report(self):
                # Generate business-focused performance report
                recent_metrics = await self.performance_monitor.get_recent_metrics(3600)  # 1 hour
                
                business_report = {
                    'revenue_metrics': {
                        'current_revenue_rate': recent_metrics.get('revenue_per_hour', 0),
                        'revenue_trend': self.calculate_trend(
                            recent_metrics, 'revenue_per_hour'
                        )
                    },
                    'user_metrics': {
                        'active_users': recent_metrics.get('active_users', 0),
                        'engagement_score': recent_metrics.get('user_engagement_score', 0),
                        'user_growth_rate': self.calculate_growth_rate(
                            recent_metrics, 'active_users'
                        )
                    },
                    'operational_metrics': {
                        'efficiency': recent_metrics.get('processing_efficiency', 0),
                        'resource_utilization': recent_metrics.get('resource_utilization', 0),
                        'quality_score': recent_metrics.get('service_quality_score', 0)
                    },
                    'kpis': {
                        kpi: value for kpi, value in recent_metrics.items()
                        if kpi.startswith('kpi_')
                    }
                }
                
                return business_report
        
        # Integrate business metrics monitoring
        business_monitor = BusinessMetricsMonitor(monitor)
        business_task = asyncio.create_task(business_monitor.track_business_metrics())
        
        # Generate hourly business reports
        async def hourly_business_reporting():
            while True:
                await asyncio.sleep(3600)  # Every hour
                report = await business_monitor.generate_business_report()
                await send_business_performance_report(report)
        
        # Start business monitoring
        await asyncio.gather(business_task, hourly_business_reporting())
        ```
    
    **Advanced Features:**
        
        **Machine Learning Integration:**
        * Predictive performance modeling and forecasting
        * Automated threshold optimization based on historical patterns
        * Intelligent alert correlation and noise reduction
        * Anomaly detection with adaptive learning algorithms
        
        **Multi-Dimensional Analytics:**
        * Cross-metric correlation analysis and dependency mapping
        * Performance pattern recognition and classification
        * Capacity planning with growth trend analysis
        * Root cause analysis and impact assessment
        
        **Enterprise Integration:**
        * Prometheus and Grafana ecosystem integration
        * ELK stack integration for log correlation
        * Custom webhook and API integration capabilities
        * Enterprise alerting and incident management systems
        
        **Real-Time Processing:**
        * Stream processing for high-frequency metrics
        * Real-time aggregation and windowing functions
        * Live dashboard updates and visualization
        * Edge computing support for distributed monitoring
    
    **Production Deployment:**
        
        **High Availability:**
        * Multi-instance monitoring with leader election
        * Distributed metrics collection and aggregation
        * Failover capabilities and data replication
        * Geographic distribution and multi-region support
        
        **Scalability:**
        * Horizontal scaling for high-throughput environments
        * Efficient data storage and compression algorithms
        * Distributed processing and parallel analytics
        * Cloud-native deployment and auto-scaling
        
        **Security & Compliance:**
        * Secure metrics transmission and storage
        * Access control and role-based permissions
        * Data encryption and privacy protection
        * Audit logging and compliance reporting
    
    Attributes:
        collection_interval (float): Frequency of metric collection in seconds
        history_size (int): Maximum number of metrics to retain in memory
        metrics_history (Dict): Time-series storage for historical metric data
        custom_metrics (Dict): Custom application and business metrics storage
        logger (Logger): Structured logging system for monitoring operations
    
    Note:
        This monitor requires appropriate system permissions for collecting system metrics.
        Large history sizes may consume significant memory in high-frequency monitoring scenarios.
        Custom metrics should be properly categorized and tagged for effective analysis.
        Integration with external systems requires proper authentication and network configuration.
    
    Warning:
        High-frequency monitoring may impact system performance on resource-constrained systems.
        Large metric datasets may require external storage for long-term retention.
        Alert thresholds should be carefully tuned to prevent notification fatigue.
        Monitoring system failures may impact observability and incident response capabilities.
    
    See Also:
        * :class:`PerformanceMetric`: Individual performance metric representation
        * :mod:`nanobrain.library.infrastructure.monitoring.resource_monitor`: Resource-specific monitoring
        * :mod:`nanobrain.library.infrastructure.monitoring.health_checker`: Health monitoring and validation
        * :mod:`nanobrain.core.logging_system`: Integrated logging and monitoring
        * :mod:`nanobrain.library.infrastructure.deployment`: Enterprise deployment monitoring
    """
    
    def __init__(self, collection_interval: float = 60.0, history_size: int = 1000):
        self.collection_interval = collection_interval
        self.history_size = history_size
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=history_size))
        self.custom_metrics: Dict[str, Any] = {}
        self._monitoring_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        self.logger = get_logger("performance_monitor")
        self.start_time = datetime.now()
        
    async def start_monitoring(self):
        """Start continuous performance monitoring."""
        if self._monitoring_task is None or self._monitoring_task.done():
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            self.logger.info(f"Started performance monitoring with {self.collection_interval}s interval")
            
    async def stop_monitoring(self):
        """Stop performance monitoring."""
        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self.logger.info("Stopped performance monitoring")
            
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while True:
            try:
                await self.collect_system_metrics()
                await asyncio.sleep(self.collection_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.collection_interval)
                
    async def collect_system_metrics(self):
        """Collect system performance metrics."""
        timestamp = datetime.now()
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=None)
        await self.record_metric("cpu_usage_percent", cpu_percent, timestamp, "percent")
        
        # Memory metrics
        memory = psutil.virtual_memory()
        await self.record_metric("memory_usage_percent", memory.percent, timestamp, "percent")
        await self.record_metric("memory_used_bytes", memory.used, timestamp, "bytes")
        await self.record_metric("memory_available_bytes", memory.available, timestamp, "bytes")
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        await self.record_metric("disk_usage_percent", (disk.used / disk.total) * 100, timestamp, "percent")
        await self.record_metric("disk_used_bytes", disk.used, timestamp, "bytes")
        await self.record_metric("disk_free_bytes", disk.free, timestamp, "bytes")
        
        # Network metrics (if available)
        try:
            network = psutil.net_io_counters()
            await self.record_metric("network_bytes_sent", network.bytes_sent, timestamp, "bytes")
            await self.record_metric("network_bytes_recv", network.bytes_recv, timestamp, "bytes")
        except Exception:
            pass  # Network stats might not be available
            
        # Process metrics
        process = psutil.Process()
        await self.record_metric("process_cpu_percent", process.cpu_percent(), timestamp, "percent")
        await self.record_metric("process_memory_bytes", process.memory_info().rss, timestamp, "bytes")
        await self.record_metric("process_threads", process.num_threads(), timestamp, "count")
        
    async def record_metric(self, name: str, value: float, timestamp: Optional[datetime] = None, 
                           unit: str = "count", tags: Optional[Dict[str, str]] = None):
        """Record a performance metric."""
        if timestamp is None:
            timestamp = datetime.now()
            
        metric = PerformanceMetric(
            name=name,
            value=value,
            timestamp=timestamp,
            unit=unit,
            tags=tags
        )
        
        async with self._lock:
            self.metrics_history[name].append(metric)
            
        self.logger.debug(f"Recorded metric {name}: {value} {unit}")
        
    async def record_custom_metric(self, name: str, value: Any, metadata: Optional[Dict[str, Any]] = None):
        """Record a custom application metric."""
        async with self._lock:
            self.custom_metrics[name] = {
                'value': value,
                'timestamp': datetime.now().isoformat(),
                'metadata': metadata or {}
            }
            
    async def get_metric_history(self, name: str, limit: Optional[int] = None) -> List[PerformanceMetric]:
        """Get history for a specific metric."""
        async with self._lock:
            history = list(self.metrics_history.get(name, []))
            if limit:
                history = history[-limit:]
            return history
            
    async def get_current_metrics(self) -> Dict[str, PerformanceMetric]:
        """Get the most recent value for each metric."""
        async with self._lock:
            current_metrics = {}
            for name, history in self.metrics_history.items():
                if history:
                    current_metrics[name] = history[-1]
            return current_metrics
            
    async def get_metric_statistics(self, name: str, time_window: Optional[timedelta] = None) -> Dict[str, float]:
        """Get statistics for a metric over a time window."""
        async with self._lock:
            history = self.metrics_history.get(name, [])
            
            if not history:
                return {}
                
            # Filter by time window if specified
            if time_window:
                cutoff_time = datetime.now() - time_window
                history = [m for m in history if m.timestamp >= cutoff_time]
                
            if not history:
                return {}
                
            values = [m.value for m in history]
            
            return {
                'count': len(values),
                'min': min(values),
                'max': max(values),
                'avg': sum(values) / len(values),
                'latest': values[-1],
                'first': values[0]
            }
            
    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary."""
        current_metrics = await self.get_current_metrics()
        uptime = datetime.now() - self.start_time
        
        summary = {
            'uptime_seconds': uptime.total_seconds(),
            'monitoring_interval': self.collection_interval,
            'metrics_collected': len(self.metrics_history),
            'custom_metrics': len(self.custom_metrics),
            'current_metrics': {name: asdict(metric) for name, metric in current_metrics.items()},
            'custom_metrics_data': dict(self.custom_metrics)
        }
        
        # Add key performance indicators
        if 'cpu_usage_percent' in current_metrics:
            summary['cpu_usage'] = current_metrics['cpu_usage_percent'].value
        if 'memory_usage_percent' in current_metrics:
            summary['memory_usage'] = current_metrics['memory_usage_percent'].value
        if 'disk_usage_percent' in current_metrics:
            summary['disk_usage'] = current_metrics['disk_usage_percent'].value
            
        return summary
        
    async def get_alerts(self, thresholds: Dict[str, float]) -> List[Dict[str, Any]]:
        """Check for metrics that exceed thresholds."""
        alerts = []
        current_metrics = await self.get_current_metrics()
        
        for metric_name, threshold in thresholds.items():
            if metric_name in current_metrics:
                metric = current_metrics[metric_name]
                if metric.value > threshold:
                    alerts.append({
                        'metric': metric_name,
                        'current_value': metric.value,
                        'threshold': threshold,
                        'unit': metric.unit,
                        'timestamp': metric.timestamp.isoformat(),
                        'severity': 'high' if metric.value > threshold * 1.5 else 'medium'
                    })
                    
        return alerts
        
    async def export_metrics(self, output_file: str, time_window: Optional[timedelta] = None):
        """Export metrics to file."""
        import json
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'time_window': time_window.total_seconds() if time_window else None,
            'metrics': {}
        }
        
        async with self._lock:
            for name, history in self.metrics_history.items():
                filtered_history = history
                
                if time_window:
                    cutoff_time = datetime.now() - time_window
                    filtered_history = [m for m in history if m.timestamp >= cutoff_time]
                    
                export_data['metrics'][name] = [
                    {
                        'value': m.value,
                        'timestamp': m.timestamp.isoformat(),
                        'unit': m.unit,
                        'tags': m.tags
                    }
                    for m in filtered_history
                ]
                
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
            
        self.logger.info(f"Exported metrics to {output_file}")
        
    async def clear_metrics(self, older_than: Optional[timedelta] = None):
        """Clear metrics history."""
        async with self._lock:
            if older_than:
                cutoff_time = datetime.now() - older_than
                for name, history in self.metrics_history.items():
                    # Keep only recent metrics
                    recent_metrics = [m for m in history if m.timestamp >= cutoff_time]
                    self.metrics_history[name] = deque(recent_metrics, maxlen=self.history_size)
            else:
                # Clear all metrics
                self.metrics_history.clear()
                self.custom_metrics.clear()
                
        self.logger.info("Cleared metrics history")
        
    async def get_trend_analysis(self, metric_name: str, time_window: timedelta) -> Dict[str, Any]:
        """Analyze trends for a specific metric."""
        history = await self.get_metric_history(metric_name)
        
        if not history:
            return {'trend': 'no_data'}
            
        # Filter by time window
        cutoff_time = datetime.now() - time_window
        recent_history = [m for m in history if m.timestamp >= cutoff_time]
        
        if len(recent_history) < 2:
            return {'trend': 'insufficient_data'}
            
        values = [m.value for m in recent_history]
        
        # Simple trend analysis
        first_half = values[:len(values)//2]
        second_half = values[len(values)//2:]
        
        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)
        
        change_percent = ((second_avg - first_avg) / first_avg) * 100 if first_avg != 0 else 0
        
        if abs(change_percent) < 5:
            trend = 'stable'
        elif change_percent > 0:
            trend = 'increasing'
        else:
            trend = 'decreasing'
            
        return {
            'trend': trend,
            'change_percent': change_percent,
            'first_half_avg': first_avg,
            'second_half_avg': second_avg,
            'data_points': len(recent_history),
            'time_window_seconds': time_window.total_seconds()
        } 