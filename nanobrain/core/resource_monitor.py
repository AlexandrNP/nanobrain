"""
Resource Monitoring System for NanoBrain Viral Protein Workflow

Provides comprehensive system resource monitoring with:
- Disk space monitoring with configurable thresholds
- Memory usage tracking
- Automatic workflow pausing on resource constraints
- User notification system for resource issues
- Integration with NanoBrain logging system
"""

import asyncio
import logging
import os
import time
from typing import Any, Dict, Optional, Callable, List
from dataclasses import dataclass
from pydantic import BaseModel, Field

from .logging_system import get_logger


@dataclass
class DiskSpaceInfo:
    """Information about disk space availability."""
    free_bytes: int
    free_gb: float
    total_bytes: int
    total_gb: float
    used_bytes: int
    used_gb: float
    usage_percent: float
    warning_triggered: bool
    critical_triggered: bool
    path: str


class ResourceMonitorConfig(BaseModel):
    """Configuration for resource monitoring."""
    # Disk space thresholds
    disk_warning_gb: float = 1.0  # Warning at 1GB remaining
    disk_critical_gb: float = 0.5  # Critical at 500MB remaining
    
    # Monitoring intervals
    monitoring_interval_seconds: float = 30.0  # Check every 30 seconds
    
    # Notification settings
    enable_notifications: bool = True
    notification_cooldown_seconds: float = 300.0  # 5 minutes between same notifications
    
    # Monitoring paths
    monitoring_paths: List[str] = Field(default_factory=lambda: ["."])
    
    # Auto-pause settings
    enable_auto_pause: bool = True
    pause_on_critical_disk: bool = True


class ResourceNotification:
    """Resource monitoring notification."""
    def __init__(self, notification_type: str, severity: str, message: str, 
                 resource_info: Dict[str, Any], timestamp: float = None):
        self.notification_type = notification_type
        self.severity = severity  # "warning", "critical"
        self.message = message
        self.resource_info = resource_info
        self.timestamp = timestamp or time.time()


class ResourceMonitor:
    """
    Monitor system resources during workflow execution.
    
    Features:
    - Disk space monitoring with configurable thresholds
    - Automatic workflow pausing on resource constraints
    - User notification system for resource issues
    - Integration with NanoBrain logging system
    """
    
    def __init__(self, config: ResourceMonitorConfig = None):
        self.config = config or ResourceMonitorConfig()
        self.logger = get_logger("resource_monitor")
        
        # State management
        self.monitoring_enabled = False
        self.monitoring_task: Optional[asyncio.Task] = None
        self.workflow_instance = None
        
        # Notification management
        self.notification_callbacks: List[Callable] = []
        self.last_notifications: Dict[str, float] = {}
        
        # Resource state tracking
        self.current_disk_info: Dict[str, DiskSpaceInfo] = {}
        self.workflow_paused = False
        self.pause_reason = None
        
    async def start_monitoring(self, workflow_instance=None) -> None:
        """Start resource monitoring for a workflow instance."""
        if self.monitoring_enabled:
            self.logger.warning("Resource monitoring already started")
            return
            
        self.workflow_instance = workflow_instance
        self.monitoring_enabled = True
        
        # Start monitoring task
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        self.logger.info("Resource monitoring started")
    
    async def stop_monitoring(self) -> None:
        """Stop resource monitoring."""
        self.monitoring_enabled = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
            
        self.logger.info("Resource monitoring stopped")
    
    def add_notification_callback(self, callback: Callable[[ResourceNotification], None]) -> None:
        """Add a callback for resource notifications."""
        self.notification_callbacks.append(callback)
        
    async def check_disk_space(self, path: str = ".") -> DiskSpaceInfo:
        """Check available disk space at specified path."""
        try:
            # Get disk usage statistics
            statvfs = os.statvfs(path)
            
            # Calculate space in bytes
            total_bytes = statvfs.f_frsize * statvfs.f_blocks
            free_bytes = statvfs.f_frsize * statvfs.f_bavail
            used_bytes = total_bytes - free_bytes
            
            # Convert to GB
            total_gb = total_bytes / (1024**3)
            free_gb = free_bytes / (1024**3)
            used_gb = used_bytes / (1024**3)
            usage_percent = (used_bytes / total_bytes) * 100 if total_bytes > 0 else 0
            
            # Check thresholds
            warning_triggered = free_gb < self.config.disk_warning_gb
            critical_triggered = free_gb < self.config.disk_critical_gb
            
            return DiskSpaceInfo(
                free_bytes=free_bytes,
                free_gb=free_gb,
                total_bytes=total_bytes,
                total_gb=total_gb,
                used_bytes=used_bytes,
                used_gb=used_gb,
                usage_percent=usage_percent,
                warning_triggered=warning_triggered,
                critical_triggered=critical_triggered,
                path=path
            )
            
        except Exception as e:
            self.logger.error(f"Failed to check disk space for {path}: {e}")
            # Return safe defaults
            return DiskSpaceInfo(
                free_bytes=0, free_gb=0.0, total_bytes=0, total_gb=0.0,
                used_bytes=0, used_gb=0.0, usage_percent=100.0,
                warning_triggered=True, critical_triggered=True, path=path
            )
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.monitoring_enabled:
            try:
                # Check all monitoring paths
                all_disk_info = {}
                critical_disk_detected = False
                warning_disk_detected = False
                
                for path in self.config.monitoring_paths:
                    disk_info = await self.check_disk_space(path)
                    all_disk_info[path] = disk_info
                    
                    if disk_info.critical_triggered:
                        critical_disk_detected = True
                    elif disk_info.warning_triggered:
                        warning_disk_detected = True
                
                self.current_disk_info = all_disk_info
                
                # Handle critical conditions
                if critical_disk_detected and self.config.pause_on_critical_disk:
                    await self._handle_critical_disk_space()
                elif warning_disk_detected:
                    await self._handle_disk_space_warning()
                
                await asyncio.sleep(self.config.monitoring_interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.config.monitoring_interval_seconds)
    
    async def _handle_critical_disk_space(self) -> None:
        """Handle critical disk space condition."""
        if self.workflow_paused and self.pause_reason == "critical_disk_space":
            return  # Already handled
            
        # Find the most critical path
        critical_paths = []
        for path, disk_info in self.current_disk_info.items():
            if disk_info.critical_triggered:
                critical_paths.append((path, disk_info.free_gb))
        
        if not critical_paths:
            return
            
        # Get the path with least free space
        most_critical_path, free_gb = min(critical_paths, key=lambda x: x[1])
        
        critical_message = (
            f"🛑 CRITICAL DISK SPACE - WORKFLOW PAUSED 🛑\n"
            f"Path: {most_critical_path}\n"
            f"Available space: {free_gb:.2f} GB\n"
            f"Threshold: {self.config.disk_critical_gb} GB\n"
            f"Workflow has been paused to prevent data loss.\n"
            f"Please free up disk space and resume manually."
        )
        
        self.logger.critical(critical_message)
        
        # Pause workflow if possible
        if self.workflow_instance and hasattr(self.workflow_instance, 'pause_workflow'):
            await self.workflow_instance.pause_workflow("critical_disk_space")
            self.workflow_paused = True
            self.pause_reason = "critical_disk_space"
        
        # Send notification
        await self._send_notification(ResourceNotification(
            notification_type="critical_disk_space",
            severity="critical",
            message=critical_message,
            resource_info={
                'path': most_critical_path,
                'free_gb': free_gb,
                'threshold_gb': self.config.disk_critical_gb
            }
        ))
    
    async def _handle_disk_space_warning(self) -> None:
        """Handle disk space warning condition."""
        notification_key = "disk_space_warning"
        
        # Check cooldown
        if not self._should_send_notification(notification_key):
            return
            
        # Find warning paths
        warning_paths = []
        for path, disk_info in self.current_disk_info.items():
            if disk_info.warning_triggered and not disk_info.critical_triggered:
                warning_paths.append((path, disk_info.free_gb))
        
        if not warning_paths:
            return
            
        # Get the path with least free space
        warning_path, free_gb = min(warning_paths, key=lambda x: x[1])
        
        warning_message = (
            f"⚠️ LOW DISK SPACE WARNING ⚠️\n"
            f"Path: {warning_path}\n"
            f"Available space: {free_gb:.2f} GB\n"
            f"Warning threshold: {self.config.disk_warning_gb} GB\n"
            f"Critical threshold: {self.config.disk_critical_gb} GB\n"
            f"Workflow will pause if space drops below critical threshold.\n"
            f"Consider freeing up disk space to continue safely."
        )
        
        self.logger.warning(warning_message)
        
        # Send notification
        await self._send_notification(ResourceNotification(
            notification_type="disk_space_warning",
            severity="warning",
            message=warning_message,
            resource_info={
                'path': warning_path,
                'free_gb': free_gb,
                'warning_threshold_gb': self.config.disk_warning_gb,
                'critical_threshold_gb': self.config.disk_critical_gb
            }
        ))
        
        # Update cooldown
        self.last_notifications[notification_key] = time.time()
    
    def _should_send_notification(self, notification_key: str) -> bool:
        """Check if notification should be sent based on cooldown."""
        if not self.config.enable_notifications:
            return False
            
        last_sent = self.last_notifications.get(notification_key, 0)
        return (time.time() - last_sent) > self.config.notification_cooldown_seconds
    
    async def _send_notification(self, notification: ResourceNotification) -> None:
        """Send notification to all registered callbacks."""
        if not self.config.enable_notifications:
            return
            
        for callback in self.notification_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(notification)
                else:
                    callback(notification)
            except Exception as e:
                self.logger.error(f"Error in notification callback: {e}")
    
    def get_resource_summary(self) -> Dict[str, Any]:
        """Get current resource usage summary."""
        disk_summary = {}
        for path, disk_info in self.current_disk_info.items():
            disk_summary[path] = {
                'free_gb': disk_info.free_gb,
                'total_gb': disk_info.total_gb,
                'usage_percent': disk_info.usage_percent,
                'warning_triggered': disk_info.warning_triggered,
                'critical_triggered': disk_info.critical_triggered
            }
        
        return {
            'disk_usage': disk_summary,
            'workflow_paused': self.workflow_paused,
            'pause_reason': self.pause_reason,
            'monitoring_enabled': self.monitoring_enabled
        } 