"""
Virus-Specific Cache Manager

Enhanced to inherit from BaseCacheManager and focus on virus-specific operations.
Includes functionality previously in BVBRCCacheManager for virus data.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import shutil
import hashlib

from .base_cache_manager import BaseCacheManager
from nanobrain.core.logging_system import get_logger


@dataclass
class VirusCacheEntry:
    """Virus-specific cache entry metadata"""
    virus_species: str
    data_type: str
    created_at: str
    file_paths: Dict[str, str]
    metrics: Dict[str, Any]


class VirusSpecificCacheManager(BaseCacheManager):
    """
    Virus-specific cache manager with enhanced functionality
    
    Inherits generic caching from BaseCacheManager and adds:
    - Virus-specific directory structure
    - BV-BRC data caching (migrated from BVBRCCacheManager)
    - Virus name standardization
    - Multi-file virus data management
    """
    
    def __init__(self, cache_base_dir: str = "data/cache/virus_specific", cache_ttl_hours: int = 24):
        """Initialize virus-specific cache manager"""
        super().__init__(cache_base_dir, cache_ttl_hours)
        self.logger = get_logger("cache_manager.virus_specific")
        
        # Index for virus cache entries
        self.index_file = self.cache_base_dir / "virus_index.json"
        self._load_index()
    
    def _load_index(self) -> Dict[str, Any]:
        """Load virus cache index"""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r') as f:
                    self.index = json.load(f)
            except:
                self.index = {"entries": {}}
        else:
            self.index = {"entries": {}}
        return self.index
    
    def _save_index(self) -> None:
        """Save virus cache index"""
        try:
            self.index_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.index_file, 'w') as f:
                json.dump(self.index, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to save index: {e}")
    
    def get_cache_identifier(self, virus_species: str, data_type: str = "general", **kwargs) -> str:
        """Get cache identifier for virus-specific data"""
        return self._generate_cache_key(
            virus_species=virus_species,
            data_type=data_type,
            **kwargs
        )
    
    def get_virus_cache_directory(self, virus_species: str) -> Path:
        """Get cache directory for specific virus"""
        normalized_virus = virus_species.replace(" ", "_").replace("-", "_").lower()
        virus_cache_dir = self.cache_base_dir / normalized_virus
        virus_cache_dir.mkdir(parents=True, exist_ok=True)
        return virus_cache_dir
    
    def get_cached_virus_data(self, virus_species: str, data_type: str = "general", 
                             **kwargs) -> Optional[Dict[str, Any]]:
        """Get cached data for specific virus"""
        cache_key = self.get_cache_identifier(virus_species=virus_species, data_type=data_type, **kwargs)
        virus_dir = self.get_virus_cache_directory(virus_species)
        
        # Check for multi-file virus data (BV-BRC pattern)
        if data_type == "proteins":
            return self._get_cached_protein_data(virus_species, virus_dir)
        
        # Default single-file cache
        return self.get_cached_result(cache_key)
    
    def _get_cached_protein_data(self, virus_species: str, virus_dir: Path) -> Optional[Dict[str, Any]]:
        """Get cached protein data (migrated from BVBRCCacheManager)"""
        # Expected files for protein data
        expected_files = {
            'genomes_tsv': virus_dir / f"{virus_species}_genomes.tsv",
            'proteins_tsv': virus_dir / f"{virus_species}_proteins.tsv",
            'proteins_fasta': virus_dir / f"{virus_species}_proteins.fasta",
            'annotations_json': virus_dir / f"{virus_species}_annotations.json"
        }
        
        # Check if all files exist and are valid
        all_valid = all(self._is_cache_valid(path) for path in expected_files.values())
        
        if not all_valid:
            return None
        
        # Return paths to cached files
        self.cache_hits += 1
        return {
            'cached': True,
            'file_paths': {k: str(v) for k, v in expected_files.items()},
            'virus_species': virus_species,
            'cache_time': datetime.now().isoformat()
        }
    
    def store_virus_data(self, virus_species: str, data: Dict[str, Any], 
                        data_type: str = "general", **kwargs) -> bool:
        """Store data for specific virus"""
        virus_dir = self.get_virus_cache_directory(virus_species)
        
        # Handle multi-file protein data (BV-BRC pattern)
        if data_type == "proteins" and 'file_paths' in data:
            return self._store_protein_data(virus_species, data, virus_dir)
        
        # Default single-file storage
        cache_key = self.get_cache_identifier(virus_species=virus_species, data_type=data_type, **kwargs)
        return self.store_result(cache_key, data)
    
    def _store_protein_data(self, virus_species: str, data: Dict[str, Any], 
                           virus_dir: Path) -> bool:
        """Store protein data files (migrated from BVBRCCacheManager)"""
        try:
            # Copy source files to cache directory
            file_mappings = {
                'genomes_tsv': f"{virus_species}_genomes.tsv",
                'proteins_tsv': f"{virus_species}_proteins.tsv", 
                'proteins_fasta': f"{virus_species}_proteins.fasta",
                'annotations_json': f"{virus_species}_annotations.json"
            }
            
            cached_files = {}
            for key, filename in file_mappings.items():
                if key in data.get('file_paths', {}):
                    source_path = Path(data['file_paths'][key])
                    if source_path.exists():
                        target_path = virus_dir / filename
                        shutil.copy2(source_path, target_path)
                        cached_files[key] = str(target_path)
            
            # Update index
            entry = VirusCacheEntry(
                virus_species=virus_species,
                data_type="proteins",
                created_at=datetime.now().isoformat(),
                file_paths=cached_files,
                metrics=data.get('metrics', {})
            )
            
            self.index['entries'][virus_species] = asdict(entry)
            self._save_index()
            
            self.logger.info(f"Cached protein data for {virus_species}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cache protein data for {virus_species}: {e}")
            return False
    
    # Enhanced BV-BRC specific methods (migrated from BVBRCCacheManager)
    def _generate_bvbrc_cache_key(self, taxon_id: str, 
                                 additional_params: Optional[Dict[str, Any]] = None) -> str:
        """Generate cache key for BV-BRC data (migrated functionality)"""
        # Base key components
        key_components = [f"taxon_{taxon_id}"]
        
        # Add additional parameters if provided
        if additional_params:
            for key, value in sorted(additional_params.items()):
                key_components.append(f"{key}_{value}")
        
        # Create hash of key components
        key_string = "_".join(key_components)
        cache_key = hashlib.md5(key_string.encode()).hexdigest()[:12]
        
        return f"{taxon_id}_{cache_key}"
    
    async def get_cached_bvbrc_result(self, taxon_id: str, 
                                     additional_params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Get cached BV-BRC pipeline result (migrated from BVBRCCacheManager)"""
        cache_key = self._generate_bvbrc_cache_key(taxon_id, additional_params)
        
        # Check if entry exists in index
        if cache_key not in self.index.get("entries", {}):
            self.cache_misses += 1
            self.logger.debug(f"BV-BRC cache miss: {cache_key}")
            return None
        
        entry_data = self.index["entries"][cache_key]
        entry = VirusCacheEntry(**entry_data)
        
        # Verify all cached files exist
        missing_files = []
        for step_name, file_path in entry.file_paths.items():
            if not Path(file_path).exists():
                missing_files.append(file_path)
        
        if missing_files:
            self.logger.warning(f"BV-BRC cache entry {cache_key} has missing files: {missing_files}")
            await self._remove_bvbrc_cache_entry(cache_key)
            self.cache_misses += 1
            return None
        
        self.cache_hits += 1
        self.logger.info(f"✅ BV-BRC cache hit: {cache_key} (taxon {taxon_id})")
        
        return {
            'success': True,
            'taxon_id': taxon_id,
            'execution_time': 0.0,  # From cache
            'file_paths': entry.file_paths,
            'metrics': entry.metrics,
            'cached': True
        }
    
    async def store_bvbrc_result(self, taxon_id: str, result_data: Dict[str, Any], 
                                additional_params: Optional[Dict[str, Any]] = None) -> str:
        """Store BV-BRC pipeline result (migrated from BVBRCCacheManager)"""
        cache_key = self._generate_bvbrc_cache_key(taxon_id, additional_params)
        
        # Create cache directory for this entry
        cache_entry_dir = self.cache_base_dir / f"taxon_{taxon_id}" / cache_key
        cache_entry_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Copy files to cache directory with standardized names
            cached_files = {}
            
            if 'file_paths' in result_data:
                for file_key, source_path in result_data['file_paths'].items():
                    if Path(source_path).exists():
                        # Standardized cache file names
                        cache_filename = f"{file_key}.{Path(source_path).suffix.lstrip('.')}"
                        target_path = cache_entry_dir / cache_filename
                        shutil.copy2(source_path, target_path)
                        cached_files[file_key] = str(target_path)
            
            # Create cache entry
            entry = VirusCacheEntry(
                virus_species=f"taxon_{taxon_id}",
                data_type="bvbrc_pipeline",
                created_at=datetime.now().isoformat(),
                file_paths=cached_files,
                metrics=result_data.get('metrics', {})
            )
            
            # Update index
            self.index['entries'][cache_key] = asdict(entry)
            self._save_index()
            
            self.logger.info(f"📦 Cached BV-BRC result: {cache_key} (taxon {taxon_id})")
            return cache_key
            
        except Exception as e:
            self.logger.error(f"Failed to cache BV-BRC result for taxon {taxon_id}: {e}")
            # Cleanup partial cache entry
            if cache_entry_dir.exists():
                shutil.rmtree(cache_entry_dir, ignore_errors=True)
            return ""
    
    async def _remove_bvbrc_cache_entry(self, cache_key: str) -> None:
        """Remove BV-BRC cache entry and its files"""
        if cache_key in self.index.get("entries", {}):
            entry_data = self.index["entries"][cache_key]
            entry = VirusCacheEntry(**entry_data)
            
            # Remove files
            for file_path in entry.file_paths.values():
                try:
                    Path(file_path).unlink(missing_ok=True)
                except Exception as e:
                    self.logger.warning(f"Could not remove cached file {file_path}: {e}")
            
            # Remove cache directory
            cache_entry_dir = Path(file_path).parent if entry.file_paths else None
            if cache_entry_dir and cache_entry_dir.exists():
                shutil.rmtree(cache_entry_dir, ignore_errors=True)
            
            # Remove from index
            del self.index["entries"][cache_key]
            self._save_index()
            
            self.logger.info(f"🗑️ Removed BV-BRC cache entry: {cache_key}")
    
    # Existing virus-specific functionality (enhanced)
    async def get_ictv_standards(self, virus_species: str) -> Optional[Dict[str, Any]]:
        """Get cached ICTV standards for virus species"""
        virus_cache_dir = self.get_virus_cache_directory(virus_species)
        ictv_file = virus_cache_dir / "ictv_standards.json"
        
        if not self._is_cache_valid(ictv_file):
            self.logger.debug(f"ICTV standards cache miss for {virus_species}")
            return None
            
        try:
            with open(ictv_file, 'r') as f:
                data = json.load(f)
                
            # Update last accessed time
            ictv_file.touch()
            
            self.logger.info(f"✅ ICTV standards cache hit for {virus_species}")
            return data
            
        except (json.JSONDecodeError, FileNotFoundError) as e:
            self.logger.warning(f"Failed to load ICTV standards cache: {e}")
            return None
    
    async def store_ictv_standards(self, virus_species: str, ictv_data: Dict[str, Any]) -> str:
        """Store ICTV standards in cache"""
        virus_cache_dir = self.get_virus_cache_directory(virus_species)
        ictv_file = virus_cache_dir / "ictv_standards.json"
        
        # Add metadata
        cache_entry = {
            "virus_species": virus_species,
            "cached_at": datetime.now().isoformat(),
            "cache_type": "ictv_standards",
            "data": ictv_data
        }
        
        try:
            with open(ictv_file, 'w') as f:
                json.dump(cache_entry, f, indent=2)
                
            self.logger.info(f"💾 ICTV standards cached for {virus_species}")
            return str(ictv_file)
            
        except Exception as e:
            self.logger.error(f"Failed to cache ICTV standards: {e}")
            raise
    
    async def get_synonym_groups(self, virus_species: str) -> Optional[Dict[str, List[Tuple[str, float]]]]:
        """Get cached synonym groups for virus species"""
        virus_cache_dir = self.get_virus_cache_directory(virus_species)
        synonym_file = virus_cache_dir / "synonym_groups.json"
        
        if not self._is_cache_valid(synonym_file):
            self.logger.debug(f"Synonym groups cache miss for {virus_species}")
            return None
            
        try:
            with open(synonym_file, 'r') as f:
                data = json.load(f)
                
            # Update last accessed time
            synonym_file.touch()
            
            # Convert tuples back from JSON lists
            synonym_groups = {}
            for canonical, synonym_list in data.get("data", {}).get("synonym_groups", {}).items():
                synonym_groups[canonical] = [(name, float(score)) for name, score in synonym_list]
            
            self.logger.info(f"✅ Synonym groups cache hit for {virus_species}")
            return synonym_groups
            
        except (json.JSONDecodeError, FileNotFoundError) as e:
            self.logger.warning(f"Failed to load synonym groups cache: {e}")
            return None
    
    async def store_synonym_groups(self, virus_species: str, 
                                  synonym_groups: Dict[str, List[Tuple[str, float]]]) -> str:
        """Store synonym groups in cache"""
        virus_cache_dir = self.get_virus_cache_directory(virus_species)
        synonym_file = virus_cache_dir / "synonym_groups.json"
        
        # Convert tuples to JSON-serializable format
        serializable_groups = {}
        for canonical, synonym_list in synonym_groups.items():
            serializable_groups[canonical] = [[name, score] for name, score in synonym_list]
        
        # Add metadata
        cache_entry = {
            "virus_species": virus_species,
            "cached_at": datetime.now().isoformat(),
            "cache_type": "synonym_groups",
            "stats": {
                "num_canonical_groups": len(synonym_groups),
                "total_synonyms": sum(len(synonyms) for synonyms in synonym_groups.values())
            },
            "data": {
                "synonym_groups": serializable_groups
            }
        }
        
        try:
            with open(synonym_file, 'w') as f:
                json.dump(cache_entry, f, indent=2)
                
            self.logger.info(f"💾 Synonym groups cached for {virus_species}")
            return str(synonym_file)
            
        except Exception as e:
            self.logger.error(f"Failed to cache synonym groups: {e}")
            raise
    
    def clear_virus_cache(self, virus_species: str) -> bool:
        """Clear cache for specific virus"""
        try:
            virus_cache_dir = self.get_virus_cache_directory(virus_species)
            if virus_cache_dir.exists():
                shutil.rmtree(virus_cache_dir)
                self.logger.info(f"Cleared cache for virus: {virus_species}")
            
            # Remove from index
            if virus_species in self.index.get('entries', {}):
                del self.index['entries'][virus_species]
                self._save_index()
            
            return True
        except Exception as e:
            self.logger.warning(f"Failed to clear virus cache for {virus_species}: {e}")
            return False
    
    def get_virus_cache_stats(self, virus_species: str) -> Dict[str, Any]:
        """Get cache statistics for specific virus"""
        virus_cache_dir = self.get_virus_cache_directory(virus_species)
        
        if not virus_cache_dir.exists():
            return {"virus_species": virus_species, "total_files": 0, "total_size": 0}
        
        total_files = 0
        total_size = 0
        
        for cache_file in virus_cache_dir.rglob("*"):
            if cache_file.is_file():
                total_files += 1
                total_size += cache_file.stat().st_size
        
        return {
            "virus_species": virus_species,
            "total_files": total_files,
            "total_size_mb": total_size / (1024 * 1024),
            "cache_directory": str(virus_cache_dir)
        } 