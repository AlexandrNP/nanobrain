"""
Docker Volume Manager for NanoBrain Framework

This module provides Docker volume management capabilities for persistent
storage, data sharing, and backup/restore operations.

"""

import os
import shutil
import tarfile
from typing import Dict, List, Optional, Any, Union, ClassVar
from dataclasses import dataclass, field
from pathlib import Path

import docker
from docker.errors import DockerException, APIError

from .container_config import DockerComponentConfig, DockerComponentBase
from nanobrain.core.logging_system import get_logger
from nanobrain.core.component_base import ComponentDependencyError


@dataclass
class VolumeConfig:
    """Docker volume configuration"""
    name: str
    driver: str = "local"
    driver_opts: Dict[str, str] = field(default_factory=dict)
    labels: Dict[str, str] = field(default_factory=dict)
    backup_enabled: bool = False
    backup_schedule: Optional[str] = None  # Cron format
    retention_days: int = 30


@dataclass
class DockerVolumeManagerConfig(DockerComponentConfig):
    """Configuration for Docker Volume Manager"""
    component_name: str = "docker_volume_manager"
    backup_dir: Optional[str] = None  # Custom backup directory path
    auto_backup_enabled: bool = True
    default_retention_days: int = 30


class DockerVolumeManager(DockerComponentBase):
    """
    Docker volume management for persistent storage and data operations.
    Enhanced with mandatory from_config pattern implementation.
    
    Provides capabilities for:
    - Volume creation and management
    - Data backup and restore
    - Volume sharing between containers
    - Cleanup and maintenance
    """
    
    # Component configuration schema
    COMPONENT_TYPE: ClassVar[str] = "docker_volume_manager"
    REQUIRED_CONFIG_FIELDS: ClassVar[List[str]] = []
    OPTIONAL_CONFIG_FIELDS: ClassVar[Dict[str, Any]] = {
        'docker_client': None,
        'backup_dir': None,
        'auto_backup_enabled': True,
        'default_retention_days': 30
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
        if isinstance(config, DockerVolumeManagerConfig):
            return {
                'component_name': config.component_name,
                'enabled': config.enabled,
                'docker_client': config.docker_client,
                'backup_dir': config.backup_dir,
                'auto_backup_enabled': config.auto_backup_enabled,
                'default_retention_days': config.default_retention_days
            }
        elif isinstance(config, dict):
            return config
        else:
            return {}
    
    @classmethod
    def resolve_dependencies(cls, component_config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Resolve volume manager dependencies"""
        try:
            docker_client = component_config.get('docker_client')
            if not docker_client:
                docker_client = docker.from_env()
            
            # Test Docker connection
            docker_client.ping()
            
            return {
                'docker_client': docker_client,
                **kwargs
            }
        except Exception as e:
            raise ComponentDependencyError(f"Failed to connect to Docker: {e}")
    
    def _init_from_config(self, config: Any, component_config: Dict[str, Any],
                         dependencies: Dict[str, Any]) -> None:
        """Initialize DockerVolumeManager from configuration"""
        super()._init_from_config(config, component_config, dependencies)
        
        # Docker client
        self.client = dependencies['docker_client']
        
        # Backup directory for volume data
        backup_dir = component_config.get('backup_dir')
        if backup_dir:
            self.backup_dir = Path(backup_dir)
        else:
            self.backup_dir = Path.home() / ".nanobrain" / "volume_backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    async def create_volume(self, config: VolumeConfig) -> bool:
        """
        Create a Docker volume.
        
        Args:
            config: Volume configuration
            
        Returns:
            bool: True if volume created successfully
        """
        try:
            self.logger.info(f"💾 Creating Docker volume: {config.name}")
            
            # Check if volume already exists
            if await self._volume_exists(config.name):
                self.logger.info(f"Volume {config.name} already exists")
                return True
            
            # Add NanoBrain framework labels
            labels = {
                "nanobrain.framework": "true",
                "nanobrain.volume.created": "true",
                **config.labels
            }
            
            if config.backup_enabled:
                labels["nanobrain.volume.backup_enabled"] = "true"
                if config.backup_schedule:
                    labels["nanobrain.volume.backup_schedule"] = config.backup_schedule
                labels["nanobrain.volume.retention_days"] = str(config.retention_days)
            
            # Create volume
            volume = self.client.volumes.create(
                name=config.name,
                driver=config.driver,
                driver_opts=config.driver_opts,
                labels=labels
            )
            
            self.logger.info(f"✅ Volume {config.name} created successfully")
            return True
            
        except DockerException as e:
            self.logger.error(f"❌ Failed to create volume {config.name}: {e}")
            return False
    
    async def remove_volume(self, name: str, force: bool = False) -> bool:
        """
        Remove a Docker volume.
        
        Args:
            name: Volume name
            force: Force removal even if volume is in use
            
        Returns:
            bool: True if volume removed successfully
        """
        try:
            volume = self.client.volumes.get(name)
            
            # Create backup before removal if backup is enabled
            labels = volume.attrs.get("Labels", {})
            if labels.get("nanobrain.volume.backup_enabled") == "true":
                self.logger.info(f"Creating backup before removing volume {name}")
                await self.backup_volume(name)
            
            volume.remove(force=force)
            self.logger.info(f"✅ Volume {name} removed successfully")
            return True
            
        except docker.errors.NotFound:
            self.logger.info(f"Volume {name} not found (already removed)")
            return True
        except DockerException as e:
            self.logger.error(f"❌ Failed to remove volume {name}: {e}")
            return False
    
    async def backup_volume(self, volume_name: str, backup_name: Optional[str] = None) -> Optional[str]:
        """
        Create a backup of a Docker volume.
        
        Args:
            volume_name: Name of volume to backup
            backup_name: Optional custom backup name
            
        Returns:
            str: Path to backup file if successful, None otherwise
        """
        try:
            if not await self._volume_exists(volume_name):
                self.logger.error(f"Volume {volume_name} not found")
                return None
            
            # Generate backup filename
            if not backup_name:
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_name = f"{volume_name}_{timestamp}.tar.gz"
            
            backup_path = self.backup_dir / backup_name
            
            self.logger.info(f"📦 Creating backup of volume {volume_name} to {backup_path}")
            
            # Create a temporary container to access volume data
            temp_container_name = f"nanobrain-backup-{volume_name}"
            
            try:
                # Create temporary container with volume mounted
                container = self.client.containers.create(
                    image="alpine:latest",
                    name=temp_container_name,
                    volumes={volume_name: {"bind": "/data", "mode": "ro"}},
                    command=["sleep", "3600"],  # Keep container alive
                    detach=True,
                    remove=True  # Auto-remove when stopped
                )
                
                container.start()
                
                # Create tar archive of volume data
                archive, _ = container.get_archive("/data")
                
                # Write archive to backup file
                with open(backup_path, "wb") as backup_file:
                    for chunk in archive:
                        backup_file.write(chunk)
                
                # Stop and remove temporary container
                container.stop()
                
                # Verify backup file
                if backup_path.exists() and backup_path.stat().st_size > 0:
                    self.logger.info(f"✅ Volume backup created: {backup_path}")
                    return str(backup_path)
                else:
                    self.logger.error(f"❌ Backup file is empty or not created: {backup_path}")
                    return None
                
            except Exception as e:
                # Cleanup temporary container if it exists
                try:
                    temp_container = self.client.containers.get(temp_container_name)
                    temp_container.stop()
                except:
                    pass
                raise e
            
        except DockerException as e:
            self.logger.error(f"❌ Failed to backup volume {volume_name}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"❌ Unexpected error during volume backup: {e}")
            return None
    
    async def restore_volume(self, volume_name: str, backup_path: str, 
                           overwrite: bool = False) -> bool:
        """
        Restore a Docker volume from backup.
        
        Args:
            volume_name: Name of volume to restore to
            backup_path: Path to backup file
            overwrite: Whether to overwrite existing volume
            
        Returns:
            bool: True if restore successful
        """
        try:
            backup_file = Path(backup_path)
            if not backup_file.exists():
                self.logger.error(f"Backup file not found: {backup_path}")
                return False
            
            # Check if volume exists
            volume_exists = await self._volume_exists(volume_name)
            
            if volume_exists and not overwrite:
                self.logger.error(f"Volume {volume_name} already exists. Use overwrite=True to replace.")
                return False
            
            if volume_exists and overwrite:
                self.logger.info(f"Removing existing volume {volume_name}")
                await self.remove_volume(volume_name, force=True)
            
            # Create new volume
            volume_config = VolumeConfig(name=volume_name)
            if not await self.create_volume(volume_config):
                return False
            
            self.logger.info(f"📦 Restoring volume {volume_name} from {backup_path}")
            
            # Create temporary container to restore data
            temp_container_name = f"nanobrain-restore-{volume_name}"
            
            try:
                # Create temporary container with volume mounted
                container = self.client.containers.create(
                    image="alpine:latest",
                    name=temp_container_name,
                    volumes={volume_name: {"bind": "/data", "mode": "rw"}},
                    command=["sleep", "3600"],
                    detach=True,
                    remove=True
                )
                
                container.start()
                
                # Extract backup archive to volume
                with open(backup_path, "rb") as backup_file:
                    container.put_archive("/", backup_file.read())
                
                # Stop and remove temporary container
                container.stop()
                
                self.logger.info(f"✅ Volume {volume_name} restored successfully")
                return True
                
            except Exception as e:
                # Cleanup temporary container if it exists
                try:
                    temp_container = self.client.containers.get(temp_container_name)
                    temp_container.stop()
                except:
                    pass
                raise e
            
        except DockerException as e:
            self.logger.error(f"❌ Failed to restore volume {volume_name}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"❌ Unexpected error during volume restore: {e}")
            return False
    
    async def list_volumes(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        List Docker volumes.
        
        Args:
            filters: Optional filters to apply
            
        Returns:
            List of volume information
        """
        try:
            volumes = self.client.volumes.list(filters=filters or {})
            
            volume_list = []
            for volume in volumes:
                # Get usage information
                usage_info = await self._get_volume_usage(volume.name)
                
                volume_info = {
                    "name": volume.name,
                    "driver": volume.attrs.get("Driver"),
                    "mountpoint": volume.attrs.get("Mountpoint"),
                    "created": volume.attrs.get("CreatedAt"),
                    "labels": volume.attrs.get("Labels", {}),
                    "options": volume.attrs.get("Options", {}),
                    "scope": volume.attrs.get("Scope"),
                    "usage": usage_info,
                    "backup_enabled": volume.attrs.get("Labels", {}).get("nanobrain.volume.backup_enabled") == "true"
                }
                volume_list.append(volume_info)
            
            return volume_list
            
        except DockerException as e:
            self.logger.error(f"❌ Failed to list volumes: {e}")
            return []
    
    async def get_volume_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific volume.
        
        Args:
            name: Volume name
            
        Returns:
            Volume information or None if not found
        """
        try:
            volume = self.client.volumes.get(name)
            
            # Get usage information
            usage_info = await self._get_volume_usage(name)
            
            # Get containers using this volume
            using_containers = await self._get_containers_using_volume(name)
            
            volume_info = {
                "name": volume.name,
                "driver": volume.attrs.get("Driver"),
                "mountpoint": volume.attrs.get("Mountpoint"),
                "created": volume.attrs.get("CreatedAt"),
                "labels": volume.attrs.get("Labels", {}),
                "options": volume.attrs.get("Options", {}),
                "scope": volume.attrs.get("Scope"),
                "usage": usage_info,
                "using_containers": using_containers,
                "backup_enabled": volume.attrs.get("Labels", {}).get("nanobrain.volume.backup_enabled") == "true",
                "backup_schedule": volume.attrs.get("Labels", {}).get("nanobrain.volume.backup_schedule"),
                "retention_days": volume.attrs.get("Labels", {}).get("nanobrain.volume.retention_days")
            }
            
            return volume_info
            
        except docker.errors.NotFound:
            self.logger.warning(f"Volume {name} not found")
            return None
        except DockerException as e:
            self.logger.error(f"❌ Failed to get volume info for {name}: {e}")
            return None
    
    async def cleanup_unused_volumes(self) -> Dict[str, Any]:
        """
        Clean up unused volumes (not mounted to any containers).
        
        Returns:
            Cleanup statistics
        """
        try:
            self.logger.info("🧹 Cleaning up unused Docker volumes...")
            
            # Use Docker's built-in prune function
            prune_result = self.client.volumes.prune()
            
            removed_volumes = prune_result.get("VolumesDeleted", [])
            space_reclaimed = prune_result.get("SpaceReclaimed", 0)
            
            # Also clean up old backups if retention is configured
            backup_cleanup_stats = await self._cleanup_old_backups()
            
            cleanup_stats = {
                "volumes_removed": len(removed_volumes) if removed_volumes else 0,
                "space_reclaimed": space_reclaimed,
                "removed_volumes": removed_volumes or [],
                "backup_cleanup": backup_cleanup_stats
            }
            
            self.logger.info(f"✅ Volume cleanup completed: {cleanup_stats}")
            return cleanup_stats
            
        except DockerException as e:
            self.logger.error(f"❌ Volume cleanup failed: {e}")
            return {"error": str(e)}
    
    async def list_backups(self, volume_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List available volume backups.
        
        Args:
            volume_name: Optional volume name to filter backups
            
        Returns:
            List of backup information
        """
        try:
            backups = []
            
            for backup_file in self.backup_dir.glob("*.tar.gz"):
                # Parse backup filename to extract volume name and timestamp
                filename = backup_file.stem.replace(".tar", "")
                parts = filename.split("_")
                
                if len(parts) >= 2:
                    backup_volume_name = "_".join(parts[:-1])
                    timestamp_str = parts[-1]
                    
                    # Filter by volume name if specified
                    if volume_name and backup_volume_name != volume_name:
                        continue
                    
                    backup_info = {
                        "volume_name": backup_volume_name,
                        "backup_file": str(backup_file),
                        "filename": backup_file.name,
                        "timestamp": timestamp_str,
                        "size": backup_file.stat().st_size,
                        "created": backup_file.stat().st_mtime
                    }
                    backups.append(backup_info)
            
            # Sort by creation time (newest first)
            backups.sort(key=lambda x: x["created"], reverse=True)
            
            return backups
            
        except Exception as e:
            self.logger.error(f"❌ Failed to list backups: {e}")
            return []
    
    async def _volume_exists(self, name: str) -> bool:
        """Check if a volume exists"""
        try:
            self.client.volumes.get(name)
            return True
        except docker.errors.NotFound:
            return False
    
    async def _get_volume_usage(self, volume_name: str) -> Dict[str, Any]:
        """Get volume usage information"""
        try:
            volume = self.client.volumes.get(volume_name)
            mountpoint = volume.attrs.get("Mountpoint")
            
            if mountpoint and os.path.exists(mountpoint):
                # Get directory size
                total_size = 0
                file_count = 0
                
                for dirpath, dirnames, filenames in os.walk(mountpoint):
                    for filename in filenames:
                        filepath = os.path.join(dirpath, filename)
                        try:
                            total_size += os.path.getsize(filepath)
                            file_count += 1
                        except (OSError, IOError):
                            continue
                
                return {
                    "size_bytes": total_size,
                    "size_mb": round(total_size / (1024 * 1024), 2),
                    "file_count": file_count,
                    "mountpoint": mountpoint
                }
            else:
                return {
                    "size_bytes": 0,
                    "size_mb": 0,
                    "file_count": 0,
                    "mountpoint": mountpoint,
                    "note": "Mountpoint not accessible"
                }
                
        except Exception as e:
            return {
                "error": str(e),
                "size_bytes": 0,
                "size_mb": 0,
                "file_count": 0
            }
    
    async def _get_containers_using_volume(self, volume_name: str) -> List[Dict[str, Any]]:
        """Get list of containers using a specific volume"""
        try:
            containers = self.client.containers.list(all=True)
            using_containers = []
            
            for container in containers:
                mounts = container.attrs.get("Mounts", [])
                for mount in mounts:
                    if mount.get("Name") == volume_name or mount.get("Source") == volume_name:
                        using_containers.append({
                            "container_id": container.id[:12],
                            "container_name": container.name,
                            "status": container.status,
                            "mount_destination": mount.get("Destination"),
                            "mount_mode": mount.get("Mode", "rw")
                        })
                        break
            
            return using_containers
            
        except Exception as e:
            self.logger.error(f"Failed to get containers using volume {volume_name}: {e}")
            return []
    
    async def _cleanup_old_backups(self) -> Dict[str, Any]:
        """Clean up old volume backups based on retention policy"""
        try:
            import datetime
            
            removed_backups = []
            total_space_freed = 0
            
            # Get all volumes with backup retention configured
            volumes = self.client.volumes.list()
            
            for volume in volumes:
                labels = volume.attrs.get("Labels", {})
                if labels.get("nanobrain.volume.backup_enabled") != "true":
                    continue
                
                try:
                    retention_days = int(labels.get("nanobrain.volume.retention_days", "30"))
                except ValueError:
                    retention_days = 30
                
                # Find backups for this volume
                volume_backups = await self.list_backups(volume.name)
                
                # Remove backups older than retention period
                cutoff_time = datetime.datetime.now().timestamp() - (retention_days * 24 * 3600)
                
                for backup in volume_backups:
                    if backup["created"] < cutoff_time:
                        backup_path = Path(backup["backup_file"])
                        if backup_path.exists():
                            file_size = backup_path.stat().st_size
                            backup_path.unlink()
                            
                            removed_backups.append({
                                "volume_name": backup["volume_name"],
                                "filename": backup["filename"],
                                "size": file_size
                            })
                            total_space_freed += file_size
            
            return {
                "removed_backups": len(removed_backups),
                "space_freed": total_space_freed,
                "backup_details": removed_backups
            }
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old backups: {e}")
            return {"error": str(e)} 