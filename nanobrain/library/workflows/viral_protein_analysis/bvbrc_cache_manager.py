"""
BV-BRC Cache Manager

Intelligent caching system for BV-BRC command pipeline results.
Caches intermediate files and pipeline results to avoid reprocessing.

Features:
- JSON-based cache index with metadata
- Configurable cache expiration
- Intelligent cache key generation
- Cache size monitoring and cleanup
"""

import json
import hashlib
import time
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from contextlib import contextmanager

from nanobrain.core.logging_system import get_logger
from .bvbrc_command_pipeline import PipelineResult, PipelineFiles


@dataclass
class CacheEntry:
    """Single cache entry metadata"""
    cache_key: str
    taxon_id: str
    created_at: str
    last_accessed: str
    expires_at: str
    file_paths: Dict[str, str]  # step_name -> file_path
    metrics: Dict[str, Any]
    file_sizes: Dict[str, int]  # file_path -> size_bytes


@dataclass
class CacheStats:
    """Cache statistics"""
    total_entries: int
    total_size_mb: float
    cache_hits: int
    cache_misses: int
    expired_entries: int
    hit_rate: float


class BVBRCCacheManager:
    """
    Manages caching of BV-BRC pipeline results and intermediate files.
    
    Cache Structure:
    cache_dir/
    ├── index.json              # Cache metadata index
    ├── taxon_<taxon_id>/
    │   ├── <cache_key>/
    │   │   ├── genomes.tsv
    │   │   ├── features.id_md5
    │   │   ├── unique.md5
    │   │   ├── sequences.fasta
    │   │   └── metadata.json
    """
    
    def __init__(self, 
                 cache_dir: str = "data/bvbrc_cache",
                 max_cache_size_mb: int = 5000,  # 5GB default
                 default_expiry_hours: int = 168):  # 1 week default
        """
        Initialize cache manager.
        
        Args:
            cache_dir: Directory for cache storage
            max_cache_size_mb: Maximum cache size in MB
            default_expiry_hours: Default cache expiration in hours
        """
        self.cache_dir = Path(cache_dir)
        self.max_cache_size_mb = max_cache_size_mb
        self.default_expiry_hours = default_expiry_hours
        self.index_file = self.cache_dir / "index.json"
        
        self.logger = get_logger("bvbrc_cache_manager")
        
        # Initialize cache directory and index
        self._initialize_cache()
        
        # Statistics
        self.cache_hits = 0
        self.cache_misses = 0
    
    def _initialize_cache(self) -> None:
        """Initialize cache directory and load index"""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.index_file.exists():
            self._create_empty_index()
        
        self.logger.info(f"📦 Cache initialized at {self.cache_dir}")
    
    def _create_empty_index(self) -> None:
        """Create empty cache index"""
        empty_index = {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "entries": {}
        }
        with open(self.index_file, 'w') as f:
            json.dump(empty_index, f, indent=2)
    
    def _load_index(self) -> Dict[str, Any]:
        """Load cache index from file"""
        try:
            with open(self.index_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.logger.warning(f"Could not load cache index, creating new one: {e}")
            self._create_empty_index()
            return self._load_index()
    
    def _save_index(self, index_data: Dict[str, Any]) -> None:
        """Save cache index to file"""
        try:
            with open(self.index_file, 'w') as f:
                json.dump(index_data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save cache index: {e}")
    
    def _generate_cache_key(self, taxon_id: str, 
                           additional_params: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate unique cache key for taxon and parameters.
        
        Args:
            taxon_id: Taxon ID
            additional_params: Additional parameters affecting cache validity
            
        Returns:
            Unique cache key string
        """
        # Base key components
        key_components = [f"taxon_{taxon_id}"]
        
        # Add additional parameters if provided
        if additional_params:
            for key, value in sorted(additional_params.items()):
                key_components.append(f"{key}_{value}")
        
        # Add timestamp for uniqueness (optional - could be removed for better caching)
        # key_components.append(f"timestamp_{int(time.time())}")
        
        # Create hash of key components
        key_string = "_".join(key_components)
        cache_key = hashlib.md5(key_string.encode()).hexdigest()[:12]
        
        return f"{taxon_id}_{cache_key}"
    
    async def get_cached_result(self, taxon_id: str, 
                               additional_params: Optional[Dict[str, Any]] = None) -> Optional[PipelineResult]:
        """
        Retrieve cached pipeline result if available and valid.
        
        Args:
            taxon_id: Taxon ID to look up
            additional_params: Additional parameters for cache key generation
            
        Returns:
            Cached PipelineResult if found and valid, None otherwise
        """
        cache_key = self._generate_cache_key(taxon_id, additional_params)
        
        index_data = self._load_index()
        entries = index_data.get("entries", {})
        
        if cache_key not in entries:
            self.cache_misses += 1
            self.logger.debug(f"Cache miss: {cache_key}")
            return None
        
        entry_data = entries[cache_key]
        entry = CacheEntry(**entry_data)
        
        # Check if entry has expired
        expires_at = datetime.fromisoformat(entry.expires_at)
        if datetime.now() > expires_at:
            self.logger.info(f"Cache entry expired: {cache_key}")
            await self._remove_cache_entry(cache_key)
            self.cache_misses += 1
            return None
        
        # Verify all cached files exist
        missing_files = []
        for step_name, file_path in entry.file_paths.items():
            if not Path(file_path).exists():
                missing_files.append(file_path)
        
        if missing_files:
            self.logger.warning(f"Cache entry {cache_key} has missing files: {missing_files}")
            await self._remove_cache_entry(cache_key)
            self.cache_misses += 1
            return None
        
        # Update last accessed time
        entry.last_accessed = datetime.now().isoformat()
        entries[cache_key] = asdict(entry)
        index_data["entries"] = entries
        self._save_index(index_data)
        
        # Create PipelineResult from cached data
        cache_dir = self.cache_dir / f"taxon_{taxon_id}" / cache_key
        files = PipelineFiles(
            working_dir=cache_dir,
            genomes_tsv=Path(entry.file_paths["genomes_tsv"]),
            features_id_md5=Path(entry.file_paths["features_id_md5"]),
            unique_md5=Path(entry.file_paths["unique_md5"]),
            sequences_fasta=Path(entry.file_paths["sequences_fasta"])
        )
        
        cached_result = PipelineResult(
            success=True,
            taxon_id=taxon_id,
            execution_time=0.0,  # From cache
            files=files,
            commands_executed=[],  # Not stored in cache
            genome_count=entry.metrics.get("genome_count", 0),
            feature_count=entry.metrics.get("feature_count", 0),
            unique_md5_count=entry.metrics.get("unique_md5_count", 0),
            sequence_count=entry.metrics.get("sequence_count", 0)
        )
        
        self.cache_hits += 1
        self.logger.info(f"✅ Cache hit: {cache_key} (taxon {taxon_id})")
        
        return cached_result
    
    async def store_result(self, result: PipelineResult, 
                          additional_params: Optional[Dict[str, Any]] = None,
                          expiry_hours: Optional[int] = None) -> str:
        """
        Store pipeline result in cache.
        
        Args:
            result: PipelineResult to cache
            additional_params: Additional parameters for cache key generation
            expiry_hours: Custom expiry time in hours
            
        Returns:
            Cache key for the stored result
        """
        if not result.success or not result.files:
            self.logger.warning("Cannot cache failed pipeline result")
            return ""
        
        cache_key = self._generate_cache_key(result.taxon_id, additional_params)
        expiry_hours = expiry_hours or self.default_expiry_hours
        
        # Create cache directory for this entry
        cache_entry_dir = self.cache_dir / f"taxon_{result.taxon_id}" / cache_key
        cache_entry_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Copy files to cache directory with standardized names
            cached_files = {}
            file_sizes = {}
            
            if result.files.genomes_tsv.exists():
                cached_genomes = cache_entry_dir / "genomes.tsv"
                shutil.copy2(result.files.genomes_tsv, cached_genomes)
                cached_files["genomes_tsv"] = str(cached_genomes)
                file_sizes[str(cached_genomes)] = cached_genomes.stat().st_size
            
            if result.files.features_id_md5.exists():
                cached_features = cache_entry_dir / "features.id_md5"
                shutil.copy2(result.files.features_id_md5, cached_features)
                cached_files["features_id_md5"] = str(cached_features)
                file_sizes[str(cached_features)] = cached_features.stat().st_size
            
            if result.files.unique_md5.exists():
                cached_unique = cache_entry_dir / "unique.md5"
                shutil.copy2(result.files.unique_md5, cached_unique)
                cached_files["unique_md5"] = str(cached_unique)
                file_sizes[str(cached_unique)] = cached_unique.stat().st_size
            
            if result.files.sequences_fasta.exists():
                cached_sequences = cache_entry_dir / "sequences.fasta"
                shutil.copy2(result.files.sequences_fasta, cached_sequences)
                cached_files["sequences_fasta"] = str(cached_sequences)
                file_sizes[str(cached_sequences)] = cached_sequences.stat().st_size
            
            # Create cache entry
            now = datetime.now()
            expires_at = now + timedelta(hours=expiry_hours)
            
            entry = CacheEntry(
                cache_key=cache_key,
                taxon_id=result.taxon_id,
                created_at=now.isoformat(),
                last_accessed=now.isoformat(),
                expires_at=expires_at.isoformat(),
                file_paths=cached_files,
                metrics={
                    "genome_count": result.genome_count,
                    "feature_count": result.feature_count,
                    "unique_md5_count": result.unique_md5_count,
                    "sequence_count": result.sequence_count,
                    "execution_time": result.execution_time
                },
                file_sizes=file_sizes
            )
            
            # Update index
            index_data = self._load_index()
            index_data["entries"][cache_key] = asdict(entry)
            self._save_index(index_data)
            
            # Check cache size and cleanup if needed
            await self._cleanup_if_needed()
            
            self.logger.info(f"📦 Cached result: {cache_key} (taxon {result.taxon_id})")
            return cache_key
            
        except Exception as e:
            self.logger.error(f"Failed to cache result for taxon {result.taxon_id}: {e}")
            # Cleanup partial cache entry
            if cache_entry_dir.exists():
                shutil.rmtree(cache_entry_dir, ignore_errors=True)
            return ""
    
    async def _remove_cache_entry(self, cache_key: str) -> None:
        """Remove cache entry and its files"""
        index_data = self._load_index()
        entries = index_data.get("entries", {})
        
        if cache_key in entries:
            entry = CacheEntry(**entries[cache_key])
            
            # Remove files
            for file_path in entry.file_paths.values():
                try:
                    Path(file_path).unlink(missing_ok=True)
                except Exception as e:
                    self.logger.warning(f"Could not remove cached file {file_path}: {e}")
            
            # Remove cache directory
            cache_entry_dir = self.cache_dir / f"taxon_{entry.taxon_id}" / cache_key
            if cache_entry_dir.exists():
                shutil.rmtree(cache_entry_dir, ignore_errors=True)
            
            # Remove from index
            del entries[cache_key]
            index_data["entries"] = entries
            self._save_index(index_data)
            
            self.logger.info(f"🗑️ Removed cache entry: {cache_key}")
    
    async def _cleanup_if_needed(self) -> None:
        """Cleanup cache if it exceeds size limits"""
        current_size_mb = await self._calculate_cache_size_mb()
        
        if current_size_mb > self.max_cache_size_mb:
            self.logger.info(f"Cache size ({current_size_mb:.1f}MB) exceeds limit ({self.max_cache_size_mb}MB), cleaning up")
            await self._cleanup_old_entries()
    
    async def _calculate_cache_size_mb(self) -> float:
        """Calculate total cache size in MB"""
        total_size = 0
        index_data = self._load_index()
        
        for entry_data in index_data.get("entries", {}).values():
            entry = CacheEntry(**entry_data)
            total_size += sum(entry.file_sizes.values())
        
        return total_size / (1024 * 1024)  # Convert to MB
    
    async def _cleanup_old_entries(self, target_size_mb: Optional[float] = None) -> None:
        """Remove old cache entries to free space"""
        target_size_mb = target_size_mb or (self.max_cache_size_mb * 0.8)  # Clean to 80% of limit
        
        index_data = self._load_index()
        entries = index_data.get("entries", {})
        
        # Sort entries by last accessed time (oldest first)
        entry_items = [(k, CacheEntry(**v)) for k, v in entries.items()]
        entry_items.sort(key=lambda x: x[1].last_accessed)
        
        current_size_mb = await self._calculate_cache_size_mb()
        removed_count = 0
        
        for cache_key, entry in entry_items:
            if current_size_mb <= target_size_mb:
                break
            
            # Calculate entry size
            entry_size_mb = sum(entry.file_sizes.values()) / (1024 * 1024)
            
            # Remove entry
            await self._remove_cache_entry(cache_key)
            current_size_mb -= entry_size_mb
            removed_count += 1
        
        if removed_count > 0:
            self.logger.info(f"🧹 Cleaned up {removed_count} old cache entries")
    
    async def get_cache_stats(self) -> CacheStats:
        """Get cache statistics"""
        index_data = self._load_index()
        entries = index_data.get("entries", {})
        
        total_entries = len(entries)
        total_size_mb = await self._calculate_cache_size_mb()
        
        # Count expired entries
        now = datetime.now()
        expired_count = 0
        for entry_data in entries.values():
            entry = CacheEntry(**entry_data)
            expires_at = datetime.fromisoformat(entry.expires_at)
            if now > expires_at:
                expired_count += 1
        
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        return CacheStats(
            total_entries=total_entries,
            total_size_mb=total_size_mb,
            cache_hits=self.cache_hits,
            cache_misses=self.cache_misses,
            expired_entries=expired_count,
            hit_rate=hit_rate
        )
    
    async def clear_expired_entries(self) -> int:
        """Remove all expired cache entries"""
        index_data = self._load_index()
        entries = index_data.get("entries", {})
        
        now = datetime.now()
        expired_keys = []
        
        for cache_key, entry_data in entries.items():
            entry = CacheEntry(**entry_data)
            expires_at = datetime.fromisoformat(entry.expires_at)
            if now > expires_at:
                expired_keys.append(cache_key)
        
        for cache_key in expired_keys:
            await self._remove_cache_entry(cache_key)
        
        if expired_keys:
            self.logger.info(f"🧹 Removed {len(expired_keys)} expired cache entries")
        
        return len(expired_keys)
    
    async def clear_all_cache(self) -> None:
        """Clear entire cache"""
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
        self._initialize_cache()
        self.cache_hits = 0
        self.cache_misses = 0
        self.logger.info("🧹 Cleared entire cache")
    
    @contextmanager
    def cache_context(self, taxon_id: str, additional_params: Optional[Dict[str, Any]] = None):
        """
        Context manager for cache operations.
        
        Usage:
            with cache_manager.cache_context("11020") as cache:
                if cache.result:
                    # Use cached result
                    return cache.result
                else:
                    # Execute pipeline and cache result
                    result = await pipeline.execute(taxon_id)
                    cache.store(result)
                    return result
        """
        class CacheContext:
            def __init__(self, manager, taxon_id, additional_params):
                self.manager = manager
                self.taxon_id = taxon_id
                self.additional_params = additional_params
                self.result = None
            
            async def load(self):
                self.result = await self.manager.get_cached_result(
                    self.taxon_id, self.additional_params
                )
            
            async def store(self, result: PipelineResult):
                await self.manager.store_result(
                    result, self.additional_params
                )
        
        context = CacheContext(self, taxon_id, additional_params)
        try:
            yield context
        finally:
            pass 