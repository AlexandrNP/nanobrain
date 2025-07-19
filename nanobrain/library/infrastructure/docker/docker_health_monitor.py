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
    Comprehensive health monitoring for Docker containers.
    Enhanced with mandatory from_config pattern implementation.
    
    Provides:
    - Periodic health checks
    - Resource usage monitoring
    - Alert notifications
    - Health history tracking
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