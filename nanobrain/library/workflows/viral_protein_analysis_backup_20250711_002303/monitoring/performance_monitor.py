"""
Performance Monitor for NanoBrain Viral Protein Analysis Workflow

This module provides comprehensive performance monitoring, metrics collection,
and optimization for the production EEEV workflow.
"""

import asyncio
import time
import psutil
import gc
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

from nanobrain.core.logging_system import get_logger


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Workflow metrics
    workflow_duration: float = 0.0
    workflow_status: str = "unknown"
    step_durations: Dict[str, float] = field(default_factory=dict)
    
    # System metrics
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    memory_usage_percent: float = 0.0
    disk_usage_percent: float = 0.0
    
    # Application metrics
    cache_hit_rate: float = 0.0
    api_response_times: Dict[str, float] = field(default_factory=dict)
    error_count: int = 0
    active_connections: int = 0
    
    # EEEV-specific metrics
    proteins_analyzed: int = 0
    boundaries_detected: int = 0
    literature_references_found: int = 0
    clustering_efficiency: float = 0.0


@dataclass
class PerformanceThresholds:
    """Performance threshold configuration"""
    
    # Execution time thresholds (seconds)
    max_workflow_duration: float = 3600  # 1 hour
    max_step_duration: float = 600       # 10 minutes
    max_api_response_time: float = 30    # 30 seconds
    
    # Resource thresholds
    max_memory_usage_percent: float = 85.0
    max_cpu_usage_percent: float = 80.0
    max_disk_usage_percent: float = 90.0
    
    # Quality thresholds
    min_cache_hit_rate: float = 0.70
    max_error_rate: float = 0.05
    min_boundary_detection_rate: float = 0.80


class PerformanceMonitor:
    """
    Comprehensive performance monitor for viral protein analysis workflow.
    
    Features:
    - Real-time metrics collection
    - Performance threshold monitoring
    - Resource optimization suggestions
    - Automated performance alerts
    - Integration with Prometheus metrics
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = get_logger("performance_monitor")
        self.thresholds = PerformanceThresholds()
        
        # Metrics storage
        self.current_metrics = PerformanceMetrics()
        self.metrics_history: List[PerformanceMetrics] = []
        self.max_history_size = 1000
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_interval = 10  # seconds
        self.workflow_start_time = None
        self.step_start_times: Dict[str, float] = {}
        
        # Performance optimization
        self.optimization_enabled = self.config.get("optimization_enabled", True)
        self.auto_gc_enabled = self.config.get("auto_gc_enabled", True)
        
    async def start_monitoring(self) -> None:
        """Start continuous performance monitoring"""
        if self.monitoring_active:
            self.logger.warning("Monitoring already active")
            return
            
        self.monitoring_active = True
        self.logger.info("Starting performance monitoring")
        
        # Start monitoring task
        asyncio.create_task(self._monitoring_loop())
        
    async def stop_monitoring(self) -> None:
        """Stop performance monitoring"""
        self.monitoring_active = False
        self.logger.info("Stopping performance monitoring")
        
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect metrics
                await self._collect_metrics()
                
                # Check thresholds
                await self._check_thresholds()
                
                # Optimize if enabled
                if self.optimization_enabled:
                    await self._optimize_performance()
                    
                # Store metrics
                self._store_metrics()
                
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.monitoring_interval)
                
    async def _collect_metrics(self) -> None:
        """Collect comprehensive performance metrics"""
        try:
            # System metrics
            self.current_metrics.cpu_usage_percent = psutil.cpu_percent(interval=1)
            
            memory_info = psutil.virtual_memory()
            self.current_metrics.memory_usage_mb = memory_info.used / (1024 * 1024)
            self.current_metrics.memory_usage_percent = memory_info.percent
            
            disk_info = psutil.disk_usage('/')
            self.current_metrics.disk_usage_percent = (disk_info.used / disk_info.total) * 100
            
            # Network connections
            self.current_metrics.active_connections = len(psutil.net_connections())
            
            self.logger.debug(f"Collected system metrics: "
                            f"CPU={self.current_metrics.cpu_usage_percent:.1f}%, "
                            f"Memory={self.current_metrics.memory_usage_percent:.1f}%, "
                            f"Disk={self.current_metrics.disk_usage_percent:.1f}%")
                            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
            
    async def _check_thresholds(self) -> None:
        """Check performance thresholds and trigger alerts"""
        alerts = []
        
        # Memory threshold
        if self.current_metrics.memory_usage_percent > self.thresholds.max_memory_usage_percent:
            alerts.append({
                "type": "memory_high",
                "severity": "warning",
                "value": self.current_metrics.memory_usage_percent,
                "threshold": self.thresholds.max_memory_usage_percent,
                "message": f"Memory usage {self.current_metrics.memory_usage_percent:.1f}% exceeds threshold"
            })
            
        # CPU threshold
        if self.current_metrics.cpu_usage_percent > self.thresholds.max_cpu_usage_percent:
            alerts.append({
                "type": "cpu_high",
                "severity": "warning", 
                "value": self.current_metrics.cpu_usage_percent,
                "threshold": self.thresholds.max_cpu_usage_percent,
                "message": f"CPU usage {self.current_metrics.cpu_usage_percent:.1f}% exceeds threshold"
            })
            
        # Disk threshold
        if self.current_metrics.disk_usage_percent > self.thresholds.max_disk_usage_percent:
            alerts.append({
                "type": "disk_high",
                "severity": "critical",
                "value": self.current_metrics.disk_usage_percent,
                "threshold": self.thresholds.max_disk_usage_percent,
                "message": f"Disk usage {self.current_metrics.disk_usage_percent:.1f}% exceeds threshold"
            })
            
        # Cache performance
        if (self.current_metrics.cache_hit_rate < self.thresholds.min_cache_hit_rate 
            and self.current_metrics.cache_hit_rate > 0):
            alerts.append({
                "type": "cache_performance",
                "severity": "warning",
                "value": self.current_metrics.cache_hit_rate,
                "threshold": self.thresholds.min_cache_hit_rate,
                "message": f"Cache hit rate {self.current_metrics.cache_hit_rate:.2f} below threshold"
            })
            
        # Send alerts
        for alert in alerts:
            await self._send_alert(alert)
            
    async def _optimize_performance(self) -> None:
        """Perform automatic performance optimizations"""
        try:
            # Memory optimization
            if self.current_metrics.memory_usage_percent > 75.0:
                if self.auto_gc_enabled:
                    # Force garbage collection
                    collected = gc.collect()
                    self.logger.info(f"Garbage collection freed {collected} objects")
                    
                # Clear old metrics history
                if len(self.metrics_history) > self.max_history_size:
                    self.metrics_history = self.metrics_history[-self.max_history_size//2:]
                    self.logger.info("Cleared old metrics history")
                    
        except Exception as e:
            self.logger.error(f"Error during performance optimization: {e}")
            
    def _store_metrics(self) -> None:
        """Store current metrics in history"""
        # Create a copy of current metrics
        metrics_copy = PerformanceMetrics(
            timestamp=datetime.now(),
            workflow_duration=self.current_metrics.workflow_duration,
            workflow_status=self.current_metrics.workflow_status,
            step_durations=self.current_metrics.step_durations.copy(),
            cpu_usage_percent=self.current_metrics.cpu_usage_percent,
            memory_usage_mb=self.current_metrics.memory_usage_mb,
            memory_usage_percent=self.current_metrics.memory_usage_percent,
            disk_usage_percent=self.current_metrics.disk_usage_percent,
            cache_hit_rate=self.current_metrics.cache_hit_rate,
            api_response_times=self.current_metrics.api_response_times.copy(),
            error_count=self.current_metrics.error_count,
            active_connections=self.current_metrics.active_connections,
            proteins_analyzed=self.current_metrics.proteins_analyzed,
            boundaries_detected=self.current_metrics.boundaries_detected,
            literature_references_found=self.current_metrics.literature_references_found,
            clustering_efficiency=self.current_metrics.clustering_efficiency
        )
        
        self.metrics_history.append(metrics_copy)
        
        # Trim history if too large
        if len(self.metrics_history) > self.max_history_size:
            self.metrics_history = self.metrics_history[-self.max_history_size//2:]
            
    async def _send_alert(self, alert: Dict[str, Any]) -> None:
        """Send performance alert"""
        self.logger.warning(f"Performance alert: {alert['message']}")
        
        # Here you could integrate with external alerting systems
        # like Slack, email, PagerDuty, etc.
        
    def start_workflow_timing(self) -> None:
        """Start timing a workflow execution"""
        self.workflow_start_time = time.time()
        self.current_metrics.workflow_status = "running"
        self.logger.info("Started workflow timing")
        
    def end_workflow_timing(self, status: str = "completed") -> float:
        """End workflow timing and return duration"""
        if self.workflow_start_time is None:
            self.logger.warning("Workflow timing not started")
            return 0.0
            
        duration = time.time() - self.workflow_start_time
        self.current_metrics.workflow_duration = duration
        self.current_metrics.workflow_status = status
        
        self.logger.info(f"Workflow completed in {duration:.2f} seconds with status: {status}")
        return duration
        
    def start_step_timing(self, step_name: str) -> None:
        """Start timing a workflow step"""
        self.step_start_times[step_name] = time.time()
        
    def end_step_timing(self, step_name: str) -> float:
        """End step timing and return duration"""
        if step_name not in self.step_start_times:
            self.logger.warning(f"Step timing not started for: {step_name}")
            return 0.0
            
        duration = time.time() - self.step_start_times[step_name]
        self.current_metrics.step_durations[step_name] = duration
        
        self.logger.debug(f"Step '{step_name}' completed in {duration:.2f} seconds")
        return duration
        
    def update_cache_metrics(self, hits: int, misses: int) -> None:
        """Update cache performance metrics"""
        total = hits + misses
        if total > 0:
            self.current_metrics.cache_hit_rate = hits / total
            
    def update_eeev_metrics(self, proteins: int = 0, boundaries: int = 0, 
                           references: int = 0, clustering_eff: float = 0.0) -> None:
        """Update EEEV-specific metrics"""
        if proteins > 0:
            self.current_metrics.proteins_analyzed = proteins
        if boundaries > 0:
            self.current_metrics.boundaries_detected = boundaries
        if references > 0:
            self.current_metrics.literature_references_found = references
        if clustering_eff > 0:
            self.current_metrics.clustering_efficiency = clustering_eff
            
    def record_api_response_time(self, api_name: str, response_time: float) -> None:
        """Record API response time"""
        self.current_metrics.api_response_times[api_name] = response_time
        
        # Check if response time exceeds threshold
        if response_time > self.thresholds.max_api_response_time:
            self.logger.warning(f"API {api_name} response time {response_time:.2f}s exceeds threshold")
            
    def increment_error_count(self) -> None:
        """Increment error counter"""
        self.current_metrics.error_count += 1
        
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        if not self.metrics_history:
            return {"status": "no_data"}
            
        recent_metrics = self.metrics_history[-10:]  # Last 10 measurements
        
        return {
            "current_status": {
                "workflow_status": self.current_metrics.workflow_status,
                "workflow_duration": self.current_metrics.workflow_duration,
                "cpu_usage": self.current_metrics.cpu_usage_percent,
                "memory_usage": self.current_metrics.memory_usage_percent,
                "disk_usage": self.current_metrics.disk_usage_percent,
                "cache_hit_rate": self.current_metrics.cache_hit_rate,
                "error_count": self.current_metrics.error_count
            },
            "eeev_analysis": {
                "proteins_analyzed": self.current_metrics.proteins_analyzed,
                "boundaries_detected": self.current_metrics.boundaries_detected,
                "literature_references": self.current_metrics.literature_references_found,
                "clustering_efficiency": self.current_metrics.clustering_efficiency
            },
            "performance_trends": {
                "avg_cpu_usage": sum(m.cpu_usage_percent for m in recent_metrics) / len(recent_metrics),
                "avg_memory_usage": sum(m.memory_usage_percent for m in recent_metrics) / len(recent_metrics),
                "avg_cache_hit_rate": sum(m.cache_hit_rate for m in recent_metrics if m.cache_hit_rate > 0) / 
                                    max(1, len([m for m in recent_metrics if m.cache_hit_rate > 0]))
            },
            "step_performance": self.current_metrics.step_durations,
            "api_performance": self.current_metrics.api_response_times,
            "threshold_violations": self._get_threshold_violations(),
            "optimization_suggestions": self._get_optimization_suggestions()
        }
        
    def _get_threshold_violations(self) -> List[str]:
        """Get current threshold violations"""
        violations = []
        
        if self.current_metrics.memory_usage_percent > self.thresholds.max_memory_usage_percent:
            violations.append(f"Memory usage: {self.current_metrics.memory_usage_percent:.1f}%")
            
        if self.current_metrics.cpu_usage_percent > self.thresholds.max_cpu_usage_percent:
            violations.append(f"CPU usage: {self.current_metrics.cpu_usage_percent:.1f}%")
            
        if self.current_metrics.disk_usage_percent > self.thresholds.max_disk_usage_percent:
            violations.append(f"Disk usage: {self.current_metrics.disk_usage_percent:.1f}%")
            
        if (self.current_metrics.cache_hit_rate < self.thresholds.min_cache_hit_rate 
            and self.current_metrics.cache_hit_rate > 0):
            violations.append(f"Cache hit rate: {self.current_metrics.cache_hit_rate:.2f}")
            
        return violations
        
    def _get_optimization_suggestions(self) -> List[str]:
        """Get performance optimization suggestions"""
        suggestions = []
        
        if self.current_metrics.memory_usage_percent > 80:
            suggestions.append("Consider reducing batch sizes or enabling more aggressive garbage collection")
            
        if self.current_metrics.cache_hit_rate < 0.7 and self.current_metrics.cache_hit_rate > 0:
            suggestions.append("Cache hit rate is low - consider increasing cache size or improving cache strategy")
            
        if any(t > 30 for t in self.current_metrics.api_response_times.values()):
            suggestions.append("Some API calls are slow - consider implementing retry logic or fallback strategies")
            
        if self.current_metrics.clustering_efficiency < 0.8:
            suggestions.append("Clustering efficiency is low - consider adjusting MMseqs2 parameters")
            
        return suggestions
        
    async def export_metrics_to_prometheus(self) -> str:
        """Export metrics in Prometheus format"""
        metrics_lines = []
        
        # Current metrics
        metrics_lines.append(f"nanobrain_cpu_usage_percent {self.current_metrics.cpu_usage_percent}")
        metrics_lines.append(f"nanobrain_memory_usage_percent {self.current_metrics.memory_usage_percent}")
        metrics_lines.append(f"nanobrain_disk_usage_percent {self.current_metrics.disk_usage_percent}")
        metrics_lines.append(f"nanobrain_cache_hit_rate {self.current_metrics.cache_hit_rate}")
        metrics_lines.append(f"nanobrain_error_count {self.current_metrics.error_count}")
        metrics_lines.append(f"nanobrain_workflow_duration_seconds {self.current_metrics.workflow_duration}")
        
        # EEEV-specific metrics
        metrics_lines.append(f"nanobrain_proteins_analyzed {self.current_metrics.proteins_analyzed}")
        metrics_lines.append(f"nanobrain_boundaries_detected {self.current_metrics.boundaries_detected}")
        metrics_lines.append(f"nanobrain_literature_references_found {self.current_metrics.literature_references_found}")
        metrics_lines.append(f"nanobrain_clustering_efficiency {self.current_metrics.clustering_efficiency}")
        
        # API response times
        for api_name, response_time in self.current_metrics.api_response_times.items():
            metrics_lines.append(f'nanobrain_api_response_time_seconds{{api="{api_name}"}} {response_time}')
            
        # Step durations
        for step_name, duration in self.current_metrics.step_durations.items():
            metrics_lines.append(f'nanobrain_step_duration_seconds{{step="{step_name}"}} {duration}')
            
        return "\n".join(metrics_lines)


# Global performance monitor instance
performance_monitor = PerformanceMonitor() 