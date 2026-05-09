"""
CSV Search Optimizer for NanoBrain Framework

This module provides search optimization capabilities including query analysis,
result caching, and performance optimization for CSV fuzzy search operations.
"""

import hashlib
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging
import re

logger = logging.getLogger(__name__)


@dataclass
class QueryAnalysis:
    """Analysis results for a search query"""
    original_query: str
    normalized_query: str
    query_type: str  # 'exact', 'fuzzy', 'partial', 'range'
    suggested_fields: List[str]
    estimated_complexity: int  # 1-10 scale
    optimization_hints: List[str]
    cache_key: str


@dataclass
class SearchCache:
    """Cache entry for search results"""
    query_hash: str
    results: List[Dict[str, Any]]
    timestamp: datetime
    hit_count: int = 0
    field_names: List[str] = field(default_factory=list)
    index_name: str = ""


class CSVSearchOptimizer:
    """
    Search optimizer for CSV data with caching, query analysis,
    and performance optimization features.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize search optimizer with configuration"""
        self.config = config or {}
        
        # Cache configuration
        self.cache_enabled = self.config.get('cache_enabled', True)
        self.cache_ttl = self.config.get('cache_ttl', 3600)  # 1 hour
        self.max_cache_size = self.config.get('max_cache_size', 1000)
        self.cache_hit_threshold = self.config.get('cache_hit_threshold', 2)
        
        # Performance configuration
        self.max_results_per_field = self.config.get('max_results_per_field', 100)
        self.query_timeout = self.config.get('query_timeout', 30)
        self.enable_query_optimization = self.config.get('enable_query_optimization', True)
        
        # Query analysis configuration
        self.min_query_length = self.config.get('min_query_length', 2)
        self.max_query_length = self.config.get('max_query_length', 1000)
        
        # Internal cache storage
        self._cache: Dict[str, SearchCache] = {}
        self._cache_stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_queries': 0
        }
        
        logger.info("✅ CSV Search Optimizer initialized")
    
    def analyze_query(self, query: str, available_fields: List[str]) -> QueryAnalysis:
        """Analyze search query and provide optimization recommendations"""
        try:
            # Normalize query
            normalized_query = self._normalize_query(query)
            
            # Determine query type
            query_type = self._determine_query_type(normalized_query)
            
            # Suggest relevant fields
            suggested_fields = self._suggest_search_fields(normalized_query, available_fields)
            
            # Estimate complexity
            complexity = self._estimate_query_complexity(normalized_query, suggested_fields)
            
            # Generate optimization hints
            hints = self._generate_optimization_hints(normalized_query, query_type, complexity)
            
            # Generate cache key
            cache_key = self._generate_cache_key(normalized_query, suggested_fields)
            
            analysis = QueryAnalysis(
                original_query=query,
                normalized_query=normalized_query,
                query_type=query_type,
                suggested_fields=suggested_fields,
                estimated_complexity=complexity,
                optimization_hints=hints,
                cache_key=cache_key
            )
            
            logger.info(f"🔍 Query analyzed: type={query_type}, complexity={complexity}, fields={len(suggested_fields)}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Query analysis failed: {e}")
            # Return basic analysis
            return QueryAnalysis(
                original_query=query,
                normalized_query=query.strip(),
                query_type="fuzzy",
                suggested_fields=available_fields[:5],  # Limit to first 5 fields
                estimated_complexity=5,
                optimization_hints=["Use basic fuzzy search"],
                cache_key=hashlib.md5(query.encode()).hexdigest()
            )
    
    def get_cached_results(self, cache_key: str) -> Optional[List[Dict[str, Any]]]:
        """Retrieve cached search results if available and valid"""
        if not self.cache_enabled:
            return None
        
        try:
            if cache_key in self._cache:
                cache_entry = self._cache[cache_key]
                
                # Check if cache entry is still valid
                if self._is_cache_valid(cache_entry):
                    cache_entry.hit_count += 1
                    self._cache_stats['hits'] += 1
                    logger.debug(f"📋 Cache hit for key: {cache_key[:8]}...")
                    return cache_entry.results
                else:
                    # Remove expired entry
                    del self._cache[cache_key]
                    logger.debug(f"🗑️ Expired cache entry removed: {cache_key[:8]}...")
            
            self._cache_stats['misses'] += 1
            return None
            
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
            return None
    
    def cache_results(self, cache_key: str, results: List[Dict[str, Any]], 
                     field_names: List[str], index_name: str):
        """Cache search results for future use"""
        if not self.cache_enabled:
            return
        
        try:
            # Check cache size limit
            if len(self._cache) >= self.max_cache_size:
                self._evict_cache_entries()
            
            # Create cache entry
            cache_entry = SearchCache(
                query_hash=cache_key,
                results=results[:self.max_results_per_field],  # Limit cached results
                timestamp=datetime.now(),
                field_names=field_names,
                index_name=index_name
            )
            
            self._cache[cache_key] = cache_entry
            logger.debug(f"💾 Results cached for key: {cache_key[:8]}...")
            
        except Exception as e:
            logger.warning(f"Result caching failed: {e}")
    
    def optimize_search_fields(self, query: str, available_fields: List[str], 
                             field_types: Dict[str, str]) -> List[str]:
        """Optimize field selection for search based on query and field types"""
        try:
            optimized_fields = []
            query_lower = query.lower()
            
            # Priority scoring for fields
            field_scores = {}
            
            for field in available_fields:
                score = 0
                field_lower = field.lower()
                field_type = field_types.get(field, 'text')
                
                # Base score by field type
                if field_type == 'text':
                    score += 10
                elif field_type in ['integer', 'float']:
                    # Check if query looks numeric
                    if self._is_numeric_query(query):
                        score += 15
                    else:
                        score += 2
                elif field_type == 'date':
                    if self._is_date_query(query):
                        score += 15
                    else:
                        score += 1
                
                # Boost score for important field names
                if any(keyword in field_lower for keyword in ['name', 'title', 'description']):
                    score += 8
                elif any(keyword in field_lower for keyword in ['id', 'code', 'sku']):
                    score += 6
                
                # Boost if query contains field name
                if field_lower in query_lower or any(word in field_lower for word in query_lower.split()):
                    score += 5
                
                field_scores[field] = score
            
            # Sort fields by score and select top candidates
            sorted_fields = sorted(field_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Select fields with score > threshold or top N fields
            threshold = 5
            for field, score in sorted_fields:
                if score >= threshold or len(optimized_fields) < 3:
                    optimized_fields.append(field)
                
                # Limit to reasonable number of fields
                if len(optimized_fields) >= 10:
                    break
            
            logger.info(f"🎯 Optimized search fields: {len(optimized_fields)} selected from {len(available_fields)}")
            
            return optimized_fields
            
        except Exception as e:
            logger.error(f"❌ Field optimization failed: {e}")
            return available_fields[:5]  # Fallback to first 5 fields
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query for consistent processing"""
        if not query:
            return ""
        
        # Basic normalization
        normalized = query.strip().lower()
        
        # Remove extra whitespace
        normalized = ' '.join(normalized.split())
        
        # Remove special characters for fuzzy matching (but keep some context)
        normalized = re.sub(r'[^\w\s\-\.]', ' ', normalized)
        normalized = ' '.join(normalized.split())
        
        return normalized
    
    def _determine_query_type(self, query: str) -> str:
        """Determine the type of search query"""
        if not query:
            return "empty"
        
        # Check for range queries
        if any(op in query for op in ['>', '<', '>=', '<=', 'between', 'range']):
            return "range"
        
        # Check for exact match indicators
        if query.startswith('"') and query.endswith('"'):
            return "exact"
        
        # Check for numeric queries
        if self._is_numeric_query(query):
            return "numeric"
        
        # Check for date queries
        if self._is_date_query(query):
            return "date"
        
        # Check for partial match indicators
        if '*' in query or '%' in query:
            return "partial"
        
        # Default to fuzzy
        return "fuzzy"
    
    def _suggest_search_fields(self, query: str, available_fields: List[str]) -> List[str]:
        """Suggest relevant fields for searching based on query content"""
        suggested = []
        query_words = query.split()
        
        # Score fields based on relevance
        field_scores = {}
        
        for field in available_fields:
            score = 0
            field_lower = field.lower()
            
            # Check if query words match field name
            for word in query_words:
                if word in field_lower:
                    score += 10
            
            # Boost important field types
            if any(keyword in field_lower for keyword in ['name', 'title', 'description', 'content']):
                score += 5
            elif any(keyword in field_lower for keyword in ['id', 'code', 'sku', 'key']):
                score += 3
            
            field_scores[field] = score
        
        # Sort by score and select top fields
        sorted_fields = sorted(field_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Include high-scoring fields and ensure we have at least some fields
        for field, score in sorted_fields:
            if score > 0 or len(suggested) < 3:
                suggested.append(field)
            
            if len(suggested) >= 8:  # Reasonable limit
                break
        
        return suggested if suggested else available_fields[:5]
    
    def _estimate_query_complexity(self, query: str, fields: List[str]) -> int:
        """Estimate query complexity on a scale of 1-10"""
        complexity = 1
        
        # Base complexity from query length
        if len(query) > 50:
            complexity += 2
        elif len(query) > 20:
            complexity += 1
        
        # Complexity from number of words
        word_count = len(query.split())
        if word_count > 5:
            complexity += 2
        elif word_count > 2:
            complexity += 1
        
        # Complexity from number of fields
        if len(fields) > 10:
            complexity += 3
        elif len(fields) > 5:
            complexity += 2
        elif len(fields) > 2:
            complexity += 1
        
        # Special characters increase complexity
        if any(char in query for char in ['*', '%', '?', '(', ')', '[', ']']):
            complexity += 1
        
        return min(10, complexity)
    
    def _generate_optimization_hints(self, query: str, query_type: str, complexity: int) -> List[str]:
        """Generate optimization hints for the query"""
        hints = []
        
        if complexity > 7:
            hints.append("High complexity query - consider limiting search fields")
        
        if len(query.split()) > 5:
            hints.append("Multi-word query - token-based matching recommended")
        
        if query_type == "fuzzy" and len(query) < 3:
            hints.append("Short query - consider exact matching for better performance")
        
        if query_type == "range":
            hints.append("Range query detected - use numeric field optimization")
        
        if any(char.isupper() for char in query):
            hints.append("Mixed case query - case-insensitive matching recommended")
        
        return hints
    
    def _generate_cache_key(self, query: str, fields: List[str]) -> str:
        """Generate cache key for query and field combination"""
        key_data = f"{query}|{','.join(sorted(fields))}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _is_cache_valid(self, cache_entry: SearchCache) -> bool:
        """Check if cache entry is still valid"""
        age = datetime.now() - cache_entry.timestamp
        return age.total_seconds() < self.cache_ttl
    
    def _evict_cache_entries(self):
        """Evict old or least-used cache entries"""
        try:
            # Remove expired entries first
            expired_keys = []
            for key, entry in self._cache.items():
                if not self._is_cache_valid(entry):
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self._cache[key]
                self._cache_stats['evictions'] += 1
            
            # If still over limit, remove least-used entries
            if len(self._cache) >= self.max_cache_size:
                # Sort by hit count (ascending) and age (oldest first)
                sorted_entries = sorted(
                    self._cache.items(),
                    key=lambda x: (x[1].hit_count, x[1].timestamp)
                )
                
                # Remove oldest, least-used entries
                entries_to_remove = len(self._cache) - self.max_cache_size + 10  # Remove extra for buffer
                for i in range(min(entries_to_remove, len(sorted_entries))):
                    key = sorted_entries[i][0]
                    del self._cache[key]
                    self._cache_stats['evictions'] += 1
            
            logger.debug(f"🗑️ Cache eviction completed: {len(expired_keys)} expired, {self._cache_stats['evictions']} total evictions")
            
        except Exception as e:
            logger.warning(f"Cache eviction failed: {e}")
    
    def _is_numeric_query(self, query: str) -> bool:
        """Check if query appears to be numeric"""
        # Remove common numeric separators
        clean_query = query.replace(',', '').replace('.', '').replace('-', '')
        return clean_query.replace(' ', '').isdigit()
    
    def _is_date_query(self, query: str) -> bool:
        """Check if query appears to be a date"""
        date_patterns = [
            r'\d{4}[-/]\d{1,2}[-/]\d{1,2}',  # YYYY-MM-DD or YYYY/MM/DD
            r'\d{1,2}[-/]\d{1,2}[-/]\d{4}',  # MM-DD-YYYY or MM/DD/YYYY
            r'\d{1,2}[-/]\d{1,2}[-/]\d{2}',  # MM-DD-YY or MM/DD/YY
        ]
        
        return any(re.search(pattern, query) for pattern in date_patterns)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = self._cache_stats['hits'] + self._cache_stats['misses']
        hit_rate = (self._cache_stats['hits'] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'cache_enabled': self.cache_enabled,
            'cache_size': len(self._cache),
            'max_cache_size': self.max_cache_size,
            'hit_rate_percent': round(hit_rate, 2),
            'total_hits': self._cache_stats['hits'],
            'total_misses': self._cache_stats['misses'],
            'total_evictions': self._cache_stats['evictions'],
            'total_requests': total_requests
        }


# Factory function following NanoBrain patterns
def from_config(config: Dict[str, Any]) -> CSVSearchOptimizer:
    """Create CSVSearchOptimizer from configuration dictionary"""
    return CSVSearchOptimizer(config)
