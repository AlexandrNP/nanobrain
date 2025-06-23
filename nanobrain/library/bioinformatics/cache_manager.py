"""
Multi-tier Caching System for Viral Protein Analysis Workflow
Phase 3 Implementation - Aggressive Caching with PubMed Deduplication

Provides comprehensive caching with memory and disk tiers, service-specific
strategies, and aggressive PubMed literature caching.
"""

import os
import json
import time
import pickle
import hashlib
import asyncio
import gzip
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from collections import OrderedDict
import yaml
from cachetools import TTLCache
from nanobrain.core.logging_system import get_logger

class CacheManager:
    """
    Multi-tier caching system with aggressive PubMed caching.
    
    Features:
    - Memory cache (TTL-based with LRU eviction)
    - Disk cache (compressed with indexing)
    - Service-specific caching strategies
    - Automatic cache warming for EEEV queries
    - Deduplication for similar literature searches
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize CacheManager with configuration.
        
        Args:
            config_path: Path to cache configuration file
        """
        self.logger = get_logger("cache_manager")
        
        # Load configuration
        if config_path is None:
            config_path = "nanobrain/library/workflows/viral_protein_analysis/config/cache_config.yml"
        
        self.config = self._load_config(config_path)
        
        # Initialize memory cache
        memory_config = self.config.get("cache_config", {}).get("storage", {}).get("memory_cache", {})
        max_size = memory_config.get("max_size_mb", 512) * 1024 * 1024  # Convert to bytes
        ttl_seconds = memory_config.get("ttl_hours", 24) * 3600
        
        self.memory_cache = TTLCache(maxsize=1000, ttl=ttl_seconds)
        
        # Initialize disk cache
        disk_config = self.config.get("cache_config", {}).get("storage", {}).get("disk_cache", {})
        self.disk_cache_dir = Path(disk_config.get("directory", "data/cache"))
        self.disk_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "disk_hits": 0,
            "memory_hits": 0,
            "writes": 0,
            "evictions": 0
        }
        
        # Initialize background tasks
        self._background_tasks = []
        
        self.logger.info("CacheManager initialized with multi-tier caching")
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load cache configuration from YAML file."""
        try:
            config_file = Path(config_path)
            if not config_file.exists():
                self.logger.warning(f"Config file not found: {config_path}, using defaults")
                return self._get_default_config()
                
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                
            self.logger.debug(f"Loaded cache configuration from {config_path}")
            return config
            
        except Exception as e:
            self.logger.error(f"Failed to load cache config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default cache configuration."""
        return {
            "cache_config": {
                "storage": {
                    "memory_cache": {"max_size_mb": 512, "ttl_hours": 24},
                    "disk_cache": {"directory": "data/cache", "max_size_gb": 5, "ttl_days": 7}
                },
                "strategies": {
                    "pubmed_references": {
                        "aggressive_caching": True,
                        "cache_duration_days": 30,
                        "deduplicate_similar": True,
                        "similarity_threshold": 0.95
                    },
                    "bvbrc_data": {
                        "cache_duration_hours": 168,
                        "compress_large_responses": True
                    }
                }
            }
        }
    
    async def get_cached_response(self, cache_key: str, service: str) -> Optional[Any]:
        """
        Get cached response with service-specific strategies.
        
        Args:
            cache_key: Unique cache key
            service: Service name for strategy selection
            
        Returns:
            Cached data if available and valid, None otherwise
        """
        # Check memory cache first
        if cache_key in self.memory_cache:
            self.stats["hits"] += 1
            self.stats["memory_hits"] += 1
            self.logger.debug(f"Memory cache hit for {cache_key}")
            return self.memory_cache[cache_key]
        
        # Check disk cache
        disk_result = await self._get_from_disk_cache(cache_key)
        if disk_result and self._is_cache_valid(disk_result, service):
            # Promote to memory cache
            self.memory_cache[cache_key] = disk_result["data"]
            self.stats["hits"] += 1
            self.stats["disk_hits"] += 1
            self.logger.debug(f"Disk cache hit for {cache_key}")
            return disk_result["data"]
        
        # Cache miss
        self.stats["misses"] += 1
        self.logger.debug(f"Cache miss for {cache_key}")
        return None
    
    async def cache_response(self, cache_key: str, data: Any, service: str, **metadata) -> None:
        """
        Cache response with service-specific TTL.
        
        Args:
            cache_key: Unique cache key
            data: Data to cache
            service: Service name for strategy selection
            **metadata: Additional metadata to store
        """
        try:
            # Store in memory cache
            self.memory_cache[cache_key] = data
            
            # Store in disk cache with metadata
            cache_entry = {
                "data": data,
                "service": service,
                "timestamp": time.time(),
                "ttl": self._get_ttl_for_service(service),
                "metadata": metadata
            }
            
            await self._store_to_disk_cache(cache_key, cache_entry)
            
            self.stats["writes"] += 1
            self.logger.debug(f"Cached response for {cache_key} (service: {service})")
            
        except Exception as e:
            self.logger.error(f"Failed to cache response for {cache_key}: {e}")
    
    async def warm_eeev_cache(self) -> None:
        """Pre-warm cache with common EEEV queries."""
        eeev_config = self.config.get("cache_config", {}).get("eeev_preload", {})
        
        if not eeev_config.get("enabled", True):
            return
        
        self.logger.info("Starting EEEV cache warming")
        
        organisms = eeev_config.get("organisms", [])
        protein_types = eeev_config.get("protein_types", [])
        literature_terms = eeev_config.get("literature_terms", [])
        
        warming_tasks = []
        
        # Warm common search combinations
        for organism in organisms:
            for protein_type in protein_types:
                cache_key = f"pubmed_{organism}_{protein_type}_boundary"
                if not await self.get_cached_response(cache_key, "pubmed_references"):
                    # Create placeholder for future warming
                    placeholder_data = {
                        "search_terms": f"{organism} {protein_type}",
                        "pre_warmed": True,
                        "results": []
                    }
                    warming_tasks.append(
                        self.cache_response(cache_key, placeholder_data, "pubmed_references")
                    )
        
        # Execute warming tasks
        if warming_tasks:
            await asyncio.gather(*warming_tasks, return_exceptions=True)
            self.logger.info(f"Pre-warmed {len(warming_tasks)} EEEV cache entries")
    
    async def deduplicate_literature_searches(self, search_terms: List[str], service: str = "pubmed_references") -> List[str]:
        """
        Deduplicate similar literature searches based on similarity threshold.
        
        Args:
            search_terms: List of search terms to deduplicate
            service: Service for similarity threshold lookup
            
        Returns:
            Deduplicated list of search terms
        """
        strategy_config = self.config.get("cache_config", {}).get("strategies", {}).get(service, {})
        similarity_threshold = strategy_config.get("similarity_threshold", 0.95)
        
        if not strategy_config.get("deduplicate_similar", False):
            return search_terms
        
        deduplicated = []
        
        for term in search_terms:
            is_similar = False
            
            for existing_term in deduplicated:
                similarity = self._calculate_text_similarity(term, existing_term)
                if similarity >= similarity_threshold:
                    is_similar = True
                    self.logger.debug(f"Skipping similar term: '{term}' (similar to '{existing_term}', similarity: {similarity:.3f})")
                    break
            
            if not is_similar:
                deduplicated.append(term)
        
        self.logger.info(f"Deduplicated {len(search_terms)} → {len(deduplicated)} search terms")
        return deduplicated
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings using Jaccard similarity."""
        set1 = set(text1.lower().split())
        set2 = set(text2.lower().split())
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    async def _get_from_disk_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve entry from disk cache."""
        try:
            cache_file = self.disk_cache_dir / f"{cache_key}.cache"
            
            if not cache_file.exists():
                return None
            
            # Check if compressed
            if cache_file.with_suffix('.cache.gz').exists():
                cache_file = cache_file.with_suffix('.cache.gz')
                with gzip.open(cache_file, 'rb') as f:
                    data = pickle.load(f)
            else:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to read from disk cache: {e}")
            return None
    
    async def _store_to_disk_cache(self, cache_key: str, data: Dict[str, Any]) -> None:
        """Store entry to disk cache with optional compression."""
        try:
            cache_file = self.disk_cache_dir / f"{cache_key}.cache"
            
            # Check if compression is enabled for large responses
            should_compress = self._should_compress_response(data)
            
            if should_compress:
                cache_file = cache_file.with_suffix('.cache.gz')
                with gzip.open(cache_file, 'wb') as f:
                    pickle.dump(data, f)
            else:
                with open(cache_file, 'wb') as f:
                    pickle.dump(data, f)
            
        except Exception as e:
            self.logger.error(f"Failed to write to disk cache: {e}")
    
    def _should_compress_response(self, data: Dict[str, Any]) -> bool:
        """Determine if response should be compressed."""
        try:
            # Estimate size
            serialized = pickle.dumps(data)
            size_mb = len(serialized) / (1024 * 1024)
            
            # Compress if larger than 1MB
            return size_mb > 1.0
            
        except Exception:
            return False
    
    def _is_cache_valid(self, cache_entry: Dict[str, Any], service: str) -> bool:
        """Check if cache entry is still valid."""
        try:
            timestamp = cache_entry.get("timestamp", 0)
            ttl = cache_entry.get("ttl", 3600)  # Default 1 hour
            current_time = time.time()
            
            return (current_time - timestamp) < ttl
            
        except Exception:
            return False
    
    def _get_ttl_for_service(self, service: str) -> int:
        """Get TTL in seconds for specific service."""
        strategy_config = self.config.get("cache_config", {}).get("strategies", {}).get(service, {})
        
        # Check for service-specific TTL configuration
        if "cache_duration_days" in strategy_config:
            return strategy_config["cache_duration_days"] * 24 * 3600
        elif "cache_duration_hours" in strategy_config:
            return strategy_config["cache_duration_hours"] * 3600
        else:
            return 24 * 3600  # Default 24 hours
    
    async def cleanup_expired_entries(self) -> int:
        """Clean up expired cache entries from disk."""
        cleaned_count = 0
        
        try:
            for cache_file in self.disk_cache_dir.glob("*.cache*"):
                try:
                    # Load and check TTL
                    if cache_file.suffix == '.gz':
                        with gzip.open(cache_file, 'rb') as f:
                            data = pickle.load(f)
                    else:
                        with open(cache_file, 'rb') as f:
                            data = pickle.load(f)
                    
                    if not self._is_cache_valid(data, data.get("service", "unknown")):
                        cache_file.unlink()
                        cleaned_count += 1
                        
                except Exception as e:
                    self.logger.warning(f"Failed to process cache file {cache_file}: {e}")
                    # Remove corrupted files
                    try:
                        cache_file.unlink()
                        cleaned_count += 1
                    except Exception:
                        pass
            
            if cleaned_count > 0:
                self.logger.info(f"Cleaned up {cleaned_count} expired cache entries")
            
        except Exception as e:
            self.logger.error(f"Cache cleanup failed: {e}")
        
        return cleaned_count
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_ratio = self.stats["hits"] / total_requests if total_requests > 0 else 0.0
        
        return {
            "hit_ratio": hit_ratio,
            "total_requests": total_requests,
            "memory_hits": self.stats["memory_hits"],
            "disk_hits": self.stats["disk_hits"],
            "misses": self.stats["misses"],
            "writes": self.stats["writes"],
            "evictions": self.stats["evictions"],
            "memory_cache_size": len(self.memory_cache),
            "disk_cache_files": len(list(self.disk_cache_dir.glob("*.cache*")))
        }
    
    async def invalidate_service_cache(self, service: str) -> int:
        """Invalidate all cache entries for a specific service."""
        invalidated_count = 0
        
        # Clear from memory cache (scan all entries)
        keys_to_remove = []
        for key in self.memory_cache:
            # This is a simplification - in practice, we'd need to track service per key
            keys_to_remove.append(key)
        
        for key in keys_to_remove:
            if key in self.memory_cache:
                del self.memory_cache[key]
                invalidated_count += 1
        
        # Clear from disk cache
        for cache_file in self.disk_cache_dir.glob("*.cache*"):
            try:
                if cache_file.suffix == '.gz':
                    with gzip.open(cache_file, 'rb') as f:
                        data = pickle.load(f)
                else:
                    with open(cache_file, 'rb') as f:
                        data = pickle.load(f)
                
                if data.get("service") == service:
                    cache_file.unlink()
                    invalidated_count += 1
                    
            except Exception as e:
                self.logger.warning(f"Failed to check cache file {cache_file}: {e}")
        
        self.logger.info(f"Invalidated {invalidated_count} cache entries for service {service}")
        return invalidated_count 