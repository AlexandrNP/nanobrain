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
    """Comprehensive metrics collection for system performance tracking."""
    
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