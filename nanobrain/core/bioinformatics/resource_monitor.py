"""
Resource Monitoring System for Viral Protein Analysis Workflow
Phase 3 Implementation - Resource Management and Workflow Control

Provides comprehensive resource monitoring with configurable thresholds,
automatic workflow pausing, and user notification system.
"""

import os
import time
import psutil
import asyncio
import shutil
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
from dataclasses import dataclass
from nanobrain.core.logger import get_logger

@dataclass
class DiskSpaceInfo:
    """Disk space information."""
    free_bytes: int
    free_gb: float
    total_gb: float
    used_gb: float
    usage_percentage: float
    warning_triggered: bool
    critical_triggered: bool

@dataclass
class MemoryInfo:
    """Memory usage information."""
    total_mb: float
    used_mb: float
    available_mb: float
    usage_percentage: float
    warning_triggered: bool
    critical_triggered: bool

@dataclass
class ResourceMonitorConfig:
    """Configuration for resource monitoring."""
    disk_warning_gb: float = 1.0  # 1GB warning threshold
    disk_critical_gb: float = 0.5  # 500MB critical threshold
    memory_warning_percentage: float = 85.0  # 85% memory warning
    memory_critical_percentage: float = 95.0  # 95% memory critical
    monitoring_interval_seconds: int = 30
    cleanup_enabled: bool = True
    user_notification_enabled: bool = True

class ResourceMonitor:
    """
    Monitor system resources during workflow execution.
    
    Features:
    - Disk space monitoring with configurable thresholds
    - Memory usage tracking
    - Automatic workflow pausing on resource constraints
    - User notification system for resource issues
    - Background cleanup operations
    """
    
    def __init__(self, config: Optional[ResourceMonitorConfig] = None):
        """
        Initialize ResourceMonitor.
        
        Args:
            config: Resource monitoring configuration
        """
        self.config = config or ResourceMonitorConfig()
        self.logger = get_logger("resource_monitor")
        
        # Monitoring state
        self.monitoring_enabled = True
        self.monitoring_task = None
        self.last_notification_time = {}
        
        # Resource thresholds
        self.disk_warning_threshold = self.config.disk_warning_gb
        self.disk_critical_threshold = self.config.disk_critical_gb
        
        # Workflow reference for pausing
        self.workflow_instance = None
        
        # User notification callback
        self.user_notification_callback = None
        
        self.logger.info(f"ResourceMonitor initialized with {self.disk_warning_threshold}GB warning threshold")
        
    async def start_monitoring(self, workflow_instance=None) -> None:
        """
        Start background resource monitoring.
        
        Args:
            workflow_instance: Workflow instance to monitor and control
        """
        if self.monitoring_task is not None:
            self.logger.warning("Resource monitoring already started")
            return
        
        self.workflow_instance = workflow_instance
        self.monitoring_enabled = True
        
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info("Started background resource monitoring")
        
    async def stop_monitoring(self) -> None:
        """Stop background resource monitoring."""
        self.monitoring_enabled = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
            self.monitoring_task = None
        
        self.logger.info("Stopped resource monitoring")
        
    async def check_disk_space(self, path: str = ".") -> DiskSpaceInfo:
        """
        Check available disk space at specified path.
        
        Args:
            path: Path to check disk space for
            
        Returns:
            DiskSpaceInfo with current disk space status
        """
        try:
            # Get disk usage statistics
            usage = shutil.disk_usage(path)
            
            free_bytes = usage.free
            total_bytes = usage.total
            used_bytes = usage.used
            
            # Convert to GB for easier handling
            free_gb = free_bytes / (1024**3)
            total_gb = total_bytes / (1024**3)
            used_gb = used_bytes / (1024**3)
            usage_percentage = (used_bytes / total_bytes) * 100
            
            # Check thresholds
            warning_triggered = free_gb < self.disk_warning_threshold
            critical_triggered = free_gb < self.disk_critical_threshold
            
            disk_info = DiskSpaceInfo(
                free_bytes=free_bytes,
                free_gb=free_gb,
                total_gb=total_gb,
                used_gb=used_gb,
                usage_percentage=usage_percentage,
                warning_triggered=warning_triggered,
                critical_triggered=critical_triggered
            )
            
            self.logger.debug(f"Disk space check: {free_gb:.2f}GB free ({usage_percentage:.1f}% used)")
            return disk_info
            
        except Exception as e:
            self.logger.error(f"Failed to check disk space: {e}")
            # Return safe defaults
            return DiskSpaceInfo(
                free_bytes=0,
                free_gb=0.0,
                total_gb=0.0,
                used_gb=0.0,
                usage_percentage=100.0,
                warning_triggered=True,
                critical_triggered=True
            )
    
    async def check_memory_usage(self) -> MemoryInfo:
        """
        Check current memory usage.
        
        Returns:
            MemoryInfo with current memory status
        """
        try:
            # Get memory statistics
            memory = psutil.virtual_memory()
            
            total_mb = memory.total / (1024**2)
            used_mb = memory.used / (1024**2)
            available_mb = memory.available / (1024**2)
            usage_percentage = memory.percent
            
            # Check thresholds
            warning_triggered = usage_percentage > self.config.memory_warning_percentage
            critical_triggered = usage_percentage > self.config.memory_critical_percentage
            
            memory_info = MemoryInfo(
                total_mb=total_mb,
                used_mb=used_mb,
                available_mb=available_mb,
                usage_percentage=usage_percentage,
                warning_triggered=warning_triggered,
                critical_triggered=critical_triggered
            )
            
            self.logger.debug(f"Memory usage: {usage_percentage:.1f}% ({used_mb:.1f}MB used)")
            return memory_info
            
        except Exception as e:
            self.logger.error(f"Failed to check memory usage: {e}")
            # Return safe defaults
            return MemoryInfo(
                total_mb=0.0,
                used_mb=0.0,
                available_mb=0.0,
                usage_percentage=100.0,
                warning_triggered=True,
                critical_triggered=True
            )
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop running in background."""
        self.logger.info("Resource monitoring loop started")
        
        while self.monitoring_enabled:
            try:
                # Check disk space
                working_dir = "."
                if self.workflow_instance and hasattr(self.workflow_instance, 'working_directory'):
                    working_dir = self.workflow_instance.working_directory
                
                disk_info = await self.check_disk_space(working_dir)
                
                # Check memory usage
                memory_info = await self.check_memory_usage()
                
                # Handle critical conditions
                if disk_info.critical_triggered:
                    await self._handle_critical_disk_space(disk_info)
                elif disk_info.warning_triggered:
                    await self._handle_disk_space_warning(disk_info)
                
                if memory_info.critical_triggered:
                    await self._handle_critical_memory(memory_info)
                elif memory_info.warning_triggered:
                    await self._handle_memory_warning(memory_info)
                
                # Log resource status periodically
                await self._log_resource_status(disk_info, memory_info)
                
                # Wait before next check
                await asyncio.sleep(self.config.monitoring_interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.config.monitoring_interval_seconds)
        
        self.logger.info("Resource monitoring loop stopped")
    
    async def _handle_disk_space_warning(self, disk_info: DiskSpaceInfo) -> None:
        """Handle disk space warning (1GB threshold)."""
        notification_key = "disk_warning"
        
        # Throttle notifications (max once per 5 minutes)
        if self._should_send_notification(notification_key, 300):
            warning_message = (
                f"⚠️ LOW DISK SPACE WARNING ⚠️\n"
                f"Available space: {disk_info.free_gb:.2f} GB\n"
                f"Used: {disk_info.usage_percentage:.1f}%\n"
                f"Workflow will pause if space drops below {self.disk_critical_threshold:.2f} GB\n"
                f"Consider freeing up disk space to continue safely."
            )
            
            self.logger.warning(warning_message)
            await self._send_user_notification(warning_message, "warning")
    
    async def _handle_critical_disk_space(self, disk_info: DiskSpaceInfo) -> None:
        """Handle critical disk space (500MB threshold) - pause workflow."""
        notification_key = "disk_critical"
        
        if self._should_send_notification(notification_key, 60):  # Max once per minute
            critical_message = (
                f"🛑 CRITICAL DISK SPACE - WORKFLOW PAUSED 🛑\n"
                f"Available space: {disk_info.free_gb:.2f} GB\n"
                f"Used: {disk_info.usage_percentage:.1f}%\n"
                f"Workflow has been paused to prevent data loss.\n"
                f"Please free up disk space and resume manually."
            )
            
            self.logger.critical(critical_message)
            
            # Pause workflow if available
            if self.workflow_instance and hasattr(self.workflow_instance, 'pause_workflow'):
                await self.workflow_instance.pause_workflow("critical_disk_space")
            
            await self._send_user_notification(critical_message, "critical")
            
            # Trigger cleanup if enabled
            if self.config.cleanup_enabled:
                await self._emergency_cleanup()
    
    async def _handle_memory_warning(self, memory_info: MemoryInfo) -> None:
        """Handle memory usage warning."""
        notification_key = "memory_warning"
        
        if self._should_send_notification(notification_key, 300):  # Max once per 5 minutes
            warning_message = (
                f"⚠️ HIGH MEMORY USAGE WARNING ⚠️\n"
                f"Memory usage: {memory_info.usage_percentage:.1f}%\n"
                f"Used: {memory_info.used_mb:.1f} MB / {memory_info.total_mb:.1f} MB\n"
                f"Available: {memory_info.available_mb:.1f} MB\n"
                f"Consider closing other applications."
            )
            
            self.logger.warning(warning_message)
            await self._send_user_notification(warning_message, "warning")
    
    async def _handle_critical_memory(self, memory_info: MemoryInfo) -> None:
        """Handle critical memory usage."""
        notification_key = "memory_critical"
        
        if self._should_send_notification(notification_key, 60):  # Max once per minute
            critical_message = (
                f"🛑 CRITICAL MEMORY USAGE 🛑\n"
                f"Memory usage: {memory_info.usage_percentage:.1f}%\n"
                f"Available: {memory_info.available_mb:.1f} MB\n"
                f"System may become unstable. Consider closing applications."
            )
            
            self.logger.critical(critical_message)
            await self._send_user_notification(critical_message, "critical")
    
    async def _emergency_cleanup(self) -> None:
        """Perform emergency cleanup operations."""
        self.logger.info("Starting emergency cleanup operations")
        
        cleanup_actions = []
        
        try:
            # Clean temporary files
            temp_dirs = ["data/cache/temp", "data/temp", "tmp"]
            for temp_dir in temp_dirs:
                temp_path = Path(temp_dir)
                if temp_path.exists():
                    cleanup_actions.append(f"Cleaned {temp_dir}")
                    for file_path in temp_path.glob("*"):
                        try:
                            if file_path.is_file():
                                file_path.unlink()
                            elif file_path.is_dir():
                                shutil.rmtree(file_path)
                        except Exception as e:
                            self.logger.warning(f"Failed to remove {file_path}: {e}")
            
            # Clean old cache files (older than 7 days)
            cache_dir = Path("data/cache")
            if cache_dir.exists():
                current_time = time.time()
                week_ago = current_time - (7 * 24 * 3600)  # 7 days
                
                cleaned_files = 0
                for cache_file in cache_dir.rglob("*.cache*"):
                    try:
                        if cache_file.stat().st_mtime < week_ago:
                            cache_file.unlink()
                            cleaned_files += 1
                    except Exception as e:
                        self.logger.warning(f"Failed to remove old cache file {cache_file}: {e}")
                
                if cleaned_files > 0:
                    cleanup_actions.append(f"Removed {cleaned_files} old cache files")
            
            # Clean log files (keep only recent ones)
            log_dirs = ["logs", "output/logs"]
            for log_dir in log_dirs:
                log_path = Path(log_dir)
                if log_path.exists():
                    cleaned_logs = 0
                    for log_file in log_path.glob("*.log"):
                        try:
                            if log_file.stat().st_size > 100 * 1024 * 1024:  # >100MB
                                log_file.unlink()
                                cleaned_logs += 1
                        except Exception as e:
                            self.logger.warning(f"Failed to remove large log file {log_file}: {e}")
                    
                    if cleaned_logs > 0:
                        cleanup_actions.append(f"Removed {cleaned_logs} large log files")
            
            if cleanup_actions:
                cleanup_message = "Emergency cleanup completed:\n" + "\n".join(cleanup_actions)
                self.logger.info(cleanup_message)
                await self._send_user_notification(cleanup_message, "info")
            else:
                self.logger.info("Emergency cleanup found no files to remove")
                
        except Exception as e:
            self.logger.error(f"Emergency cleanup failed: {e}")
    
    def _should_send_notification(self, notification_key: str, min_interval_seconds: int) -> bool:
        """Check if notification should be sent based on throttling."""
        current_time = time.time()
        last_time = self.last_notification_time.get(notification_key, 0)
        
        if current_time - last_time >= min_interval_seconds:
            self.last_notification_time[notification_key] = current_time
            return True
        
        return False
    
    async def _send_user_notification(self, message: str, severity: str) -> None:
        """Send notification to user."""
        if not self.config.user_notification_enabled:
            return
        
        notification = {
            "type": "resource_alert",
            "message": message,
            "severity": severity,
            "timestamp": time.time()
        }
        
        # Use callback if available
        if self.user_notification_callback:
            try:
                await self.user_notification_callback(notification)
            except Exception as e:
                self.logger.error(f"Failed to send user notification: {e}")
        
        # Also log to workflow if available
        if self.workflow_instance and hasattr(self.workflow_instance, 'notify_user'):
            try:
                await self.workflow_instance.notify_user(message, severity=severity)
            except Exception as e:
                self.logger.error(f"Failed to notify workflow: {e}")
    
    async def _log_resource_status(self, disk_info: DiskSpaceInfo, memory_info: MemoryInfo) -> None:
        """Log resource status periodically."""
        # Log every 10 monitoring cycles (5 minutes with 30s interval)
        if not hasattr(self, '_status_log_counter'):
            self._status_log_counter = 0
        
        self._status_log_counter += 1
        
        if self._status_log_counter >= 10:
            self._status_log_counter = 0
            
            status_message = (
                f"Resource Status: Disk: {disk_info.free_gb:.2f}GB free "
                f"({disk_info.usage_percentage:.1f}% used), "
                f"Memory: {memory_info.usage_percentage:.1f}% used"
            )
            
            self.logger.info(status_message)
    
    def set_user_notification_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Set callback function for user notifications."""
        self.user_notification_callback = callback
        self.logger.debug("User notification callback set")
    
    def get_resource_summary(self) -> Dict[str, Any]:
        """Get current resource summary."""
        return {
            "monitoring_enabled": self.monitoring_enabled,
            "disk_warning_threshold_gb": self.disk_warning_threshold,
            "disk_critical_threshold_gb": self.disk_critical_threshold,
            "memory_warning_threshold_percent": self.config.memory_warning_percentage,
            "memory_critical_threshold_percent": self.config.memory_critical_percentage,
            "monitoring_interval_seconds": self.config.monitoring_interval_seconds,
            "cleanup_enabled": self.config.cleanup_enabled
        } 