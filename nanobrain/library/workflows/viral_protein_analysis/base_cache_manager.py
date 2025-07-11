"""
Base Cache Manager with Generic Functionality

Consolidates generic cache operations previously in tool-specific managers.
Provides standard caching interface for all tools and steps.
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union
from abc import ABC, abstractmethod
import shutil
import time

from nanobrain.core.logging_system import get_logger


class BaseCacheManager(ABC):
    """
    Enhanced base cache manager with generic functionality
    
    Consolidates generic cache operations from tool-specific managers.
    All cache managers should inherit from this class.
    """
    
    def __init__(self, cache_base_dir: str = "data/cache", cache_ttl_hours: int = 24):
        """Initialize base cache manager with generic functionality"""
        self.cache_base_dir = Path(cache_base_dir)
        self.cache_ttl_hours = cache_ttl_hours
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        self.logger = get_logger(f"cache_manager.{self.__class__.__name__}")
        
        # Ensure cache directory exists
        self.cache_base_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics tracking
        self.cache_hits = 0
        self.cache_misses = 0
        
        self.logger.info(f"Initialized cache manager with base directory: {self.cache_base_dir}")
    
    def _generate_cache_key(self, **kwargs) -> str:
        """Generate cache key from parameters"""
        # Sort parameters for consistent key generation
        sorted_params = sorted(kwargs.items())
        param_string = "_".join(f"{k}={v}" for k, v in sorted_params)
        
        # Generate hash for long parameter strings
        if len(param_string) > 100:
            hash_object = hashlib.sha256(param_string.encode())
            return hash_object.hexdigest()[:32]
        
        return param_string.replace(" ", "_").replace("/", "_").lower()
    
    def _get_cache_path(self, cache_key: str, file_extension: str = ".json") -> Path:
        """Get cache file path for given cache key"""
        cache_filename = f"{cache_key}{file_extension}"
        return self.cache_base_dir / cache_filename
    
    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if cache file is valid (exists and not expired)"""
        if not cache_path.exists():
            return False
        
        # Check if cache has expired
        cache_mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        if datetime.now() - cache_mtime > self.cache_ttl:
            self.logger.debug(f"Cache expired: {cache_path}")
            return False
        
        return True
    
    def _load_from_cache(self, cache_path: Path) -> Optional[Dict[str, Any]]:
        """Load data from cache file"""
        try:
            if cache_path.suffix == '.json':
                with open(cache_path, 'r') as f:
                    data = json.load(f)
            else:
                # For non-JSON files, return path info
                data = {
                    'file_path': str(cache_path),
                    'file_exists': cache_path.exists(),
                    'file_size': cache_path.stat().st_size if cache_path.exists() else 0
                }
            
            self.logger.debug(f"Loaded from cache: {cache_path}")
            return data
        except Exception as e:
            self.logger.warning(f"Failed to load cache {cache_path}: {e}")
            return None
    
    def _save_to_cache(self, cache_path: Path, data: Union[Dict[str, Any], str, bytes]) -> bool:
        """Save data to cache file"""
        try:
            # Ensure parent directory exists
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            
            if isinstance(data, dict):
                with open(cache_path, 'w') as f:
                    json.dump(data, f, indent=2)
            elif isinstance(data, str):
                with open(cache_path, 'w') as f:
                    f.write(data)
            elif isinstance(data, bytes):
                with open(cache_path, 'wb') as f:
                    f.write(data)
            else:
                # For other types, try JSON serialization
                with open(cache_path, 'w') as f:
                    json.dump(data, f, indent=2)
            
            self.logger.debug(f"Saved to cache: {cache_path}")
            return True
        except Exception as e:
            self.logger.warning(f"Failed to save cache {cache_path}: {e}")
            return False
    
    def get_cached_result(self, cache_key: str, file_extension: str = ".json") -> Optional[Dict[str, Any]]:
        """Generic method to get cached result"""
        cache_path = self._get_cache_path(cache_key, file_extension)
        
        if not self._is_cache_valid(cache_path):
            self.cache_misses += 1
            return None
        
        self.cache_hits += 1
        return self._load_from_cache(cache_path)
    
    def store_result(self, cache_key: str, data: Union[Dict[str, Any], str, bytes], 
                    file_extension: str = ".json") -> bool:
        """Generic method to store result in cache"""
        cache_path = self._get_cache_path(cache_key, file_extension)
        return self._save_to_cache(cache_path, data)
    
    def get_cached_file(self, cache_key: str, source_file: Path, 
                       target_extension: Optional[str] = None) -> Optional[Path]:
        """Get cached file path if valid"""
        if target_extension is None:
            target_extension = source_file.suffix
        
        cache_path = self._get_cache_path(cache_key, target_extension)
        
        if self._is_cache_valid(cache_path):
            self.cache_hits += 1
            return cache_path
        
        self.cache_misses += 1
        return None
    
    def cache_file(self, cache_key: str, source_file: Path, 
                  target_extension: Optional[str] = None) -> Optional[Path]:
        """Cache a file and return cached path"""
        if not source_file.exists():
            self.logger.warning(f"Source file does not exist: {source_file}")
            return None
        
        if target_extension is None:
            target_extension = source_file.suffix
        
        cache_path = self._get_cache_path(cache_key, target_extension)
        
        try:
            # Copy file to cache
            shutil.copy2(source_file, cache_path)
            self.logger.debug(f"Cached file: {source_file} -> {cache_path}")
            return cache_path
        except Exception as e:
            self.logger.warning(f"Failed to cache file {source_file}: {e}")
            return None
    
    def clear_cache(self, cache_key: Optional[str] = None) -> bool:
        """Clear cache (specific key or all cache)"""
        try:
            if cache_key:
                # Clear specific cache entry
                for ext in ['.json', '.fasta', '.tsv', '.txt', '.csv']:
                    cache_path = self._get_cache_path(cache_key, ext)
                    if cache_path.exists():
                        cache_path.unlink()
                        self.logger.info(f"Cleared cache: {cache_key}{ext}")
            else:
                # Clear all cache files
                if self.cache_base_dir.exists():
                    shutil.rmtree(self.cache_base_dir)
                    self.cache_base_dir.mkdir(parents=True, exist_ok=True)
                    self.logger.info("Cleared all cache")
            
            return True
        except Exception as e:
            self.logger.warning(f"Failed to clear cache: {e}")
            return False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self.cache_base_dir.exists():
            return {
                "total_files": 0, 
                "total_size": 0, 
                "cache_directory": str(self.cache_base_dir),
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "hit_rate": 0.0
            }
        
        total_files = 0
        total_size = 0
        
        for cache_file in self.cache_base_dir.rglob("*"):
            if cache_file.is_file():
                total_files += 1
                total_size += cache_file.stat().st_size
        
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "total_files": total_files,
            "total_size_mb": total_size / (1024 * 1024),
            "cache_directory": str(self.cache_base_dir),
            "cache_ttl_hours": self.cache_ttl_hours,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate
        }
    
    @abstractmethod
    def get_cache_identifier(self, **kwargs) -> str:
        """Get cache identifier for specific cache manager implementation"""
        pass 