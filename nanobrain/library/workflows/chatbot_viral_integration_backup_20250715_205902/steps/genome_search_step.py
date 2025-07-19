"""
Genome Search Step for Chatbot Viral Integration

This step provides specialized virus genome search capabilities using Elasticsearch,
replacing the CSV-based fuzzy matching with high-performance search infrastructure.

"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field

from nanobrain.core.step import Step, StepConfig
from nanobrain.core.data_unit import DataUnit
from nanobrain.core.logging_system import get_logger
from nanobrain.library.tools.search.elasticsearch_tool import ElasticsearchTool, ElasticsearchConfig


@dataclass
class GenomeSearchConfig(StepConfig):
    """Configuration for genome search step"""
    step_name: str = "genome_search"
    description: str = "High-performance virus genome search using Elasticsearch"
    
    # Elasticsearch configuration
    elasticsearch_config: Dict[str, Any] = field(default_factory=lambda: {
        "host": "localhost",
        "port": 9200,
        "auto_install_enabled": True,
        "virus_genome_index": "virus-genomes"
    })
    
    # Search configuration
    confidence_threshold: float = 0.7
    max_results: int = 20
    enable_fuzzy_search: bool = True
    fallback_to_csv: bool = True  # Fallback to CSV search if Elasticsearch fails
    
    # CSV fallback configuration (for compatibility)
    csv_file_path: str = "data/BVBRC_genome_alphavirus.csv"
    
    # Performance settings
    search_timeout: int = 30
    cache_results: bool = True
    cache_ttl: int = 3600  # 1 hour


@dataclass
class GenomeSearchResult:
    """Result of genome search operation"""
    query: str
    matches: List[Dict[str, Any]]
    total_found: int
    search_time: float
    confidence_scores: List[float]
    search_method: str  # "elasticsearch" or "csv_fallback"
    cached: bool = False


class GenomeSearchStep(Step):
    """
    Specialized step for virus genome search using Elasticsearch.
    
    Provides high-performance search capabilities for virus name resolution
    with fallback to CSV-based search for compatibility.
    """
    
    @classmethod
    def from_config(cls, config: Union[StepConfig, GenomeSearchConfig, Dict], **kwargs) -> 'GenomeSearchStep':
        """Mandatory from_config implementation for GenomeSearchStep"""
        logger = get_logger(f"{cls.__name__}.from_config")
        logger.info(f"Creating {cls.__name__} from configuration")
        
        # Convert any input to GenomeSearchConfig
        if isinstance(config, GenomeSearchConfig):
            # Already specific config, use as-is
            pass
        else:
            # Convert StepConfig, dict, or any other input to GenomeSearchConfig
            if hasattr(config, 'model_dump'):
                config_dict = config.model_dump()
            elif isinstance(config, dict):
                config_dict = config
            else:
                # Handle object with attributes
                config_dict = {}
                for attr in ['step_name', 'description', 'timeout', 'retry_attempts']:
                    if hasattr(config, attr):
                        config_dict[attr] = getattr(config, attr)
            
            # Filter config_dict to only include fields that GenomeSearchConfig accepts
            genome_search_compatible_fields = {
                # Core fields from GenomeSearchConfig
                'step_name', 'description', 'elasticsearch_config', 'confidence_threshold',
                'max_results', 'enable_fuzzy_search', 'fallback_to_csv', 'csv_file_path',
                'search_timeout', 'cache_results', 'cache_ttl',
                # Core fields from StepConfig
                'timeout', 'retry_attempts', 'parallel_execution', 'resource_requirements',
                'dependencies', 'outputs', 'metadata'
            }
            
            filtered_config_dict = {k: v for k, v in config_dict.items() 
                                   if k in genome_search_compatible_fields}
            
            logger.debug(f"Filtered config keys: {list(filtered_config_dict.keys())}")
            
            # Create GenomeSearchConfig from the filtered data
            config = GenomeSearchConfig(**filtered_config_dict)
        
        # Create instance
        instance = cls(config, **kwargs)
        
        logger.info(f"Successfully created {cls.__name__} with mandatory from_config pattern")
        return instance
    
    def __init__(self, config: GenomeSearchConfig, **kwargs):
        """Initialize genome search step"""
        if config is None:
            config = GenomeSearchConfig()
        
        # Ensure step_name is set consistently
        if not hasattr(config, 'step_name') or not config.step_name:
            config.step_name = "genome_search"
        
        # Initialize parent class
        super().__init__(config, **kwargs)
        
        # Store genome search specific configuration
        self.genome_config = config
        self.name = config.step_name
        self.logger = get_logger(f"genome_search_step_{self.name}")
        
        # Elasticsearch tool instance
        self.elasticsearch_tool: Optional[ElasticsearchTool] = None
        
        # Search cache
        self.search_cache: Dict[str, GenomeSearchResult] = {}
        
        # Statistics
        self.total_searches = 0
        self.elasticsearch_searches = 0
        self.csv_fallback_searches = 0
        self.cache_hits = 0
    
    async def initialize_step(self):
        """Initialize the genome search step"""
        try:
            self.logger.info("🔄 Initializing genome search step...")
            
            # Initialize Elasticsearch tool
            await self._initialize_elasticsearch_tool()
            
            # Pre-populate genome index if needed
            await self._ensure_genome_index_populated()
            
            self.logger.info("✅ Genome search step initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Genome search step initialization failed: {e}")
            
            # If Elasticsearch fails and fallback is enabled, continue with CSV mode
            if self.genome_config.fallback_to_csv:
                self.logger.warning("⚠️ Continuing with CSV fallback mode only")
            else:
                raise
    
    async def _initialize_elasticsearch_tool(self):
        """Initialize Elasticsearch tool for genome search"""
        try:
            # Create Elasticsearch configuration
            es_config = ElasticsearchConfig(**self.genome_config.elasticsearch_config)
            
            # Create Elasticsearch tool instance
            self.elasticsearch_tool = await ElasticsearchTool.from_config(es_config)
            
            # Initialize the tool (this handles auto-installation if enabled)
            await self.elasticsearch_tool.initialize_tool()
            
            self.logger.info("✅ Elasticsearch tool initialized for genome search")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Elasticsearch tool: {e}")
            
            if self.genome_config.fallback_to_csv:
                self.logger.warning("⚠️ Elasticsearch unavailable, will use CSV fallback")
                self.elasticsearch_tool = None
            else:
                raise
    
    async def _ensure_genome_index_populated(self):
        """Ensure the virus genome index is populated with data"""
        if not self.elasticsearch_tool:
            return
        
        try:
            # Check if index exists and has data
            index_name = self.genome_config.elasticsearch_config.get("virus_genome_index", "virus-genomes")
            
            # Simple check: try to get index stats
            search_result = await self.elasticsearch_tool.search_documents(
                query="*",
                index=index_name,
                max_results=1
            )
            
            if search_result["total_hits"] == 0:
                self.logger.info("📊 Virus genome index is empty, populating from CSV...")
                await self._populate_genome_index_from_csv()
            else:
                self.logger.info(f"✅ Virus genome index contains {search_result['total_hits']} documents")
                
        except Exception as e:
            self.logger.warning(f"⚠️ Could not verify genome index population: {e}")
    
    async def _populate_genome_index_from_csv(self):
        """Populate Elasticsearch index from CSV data"""
        try:
            import pandas as pd
            from pathlib import Path
            
            csv_path = Path(self.genome_config.csv_file_path)
            if not csv_path.exists():
                self.logger.warning(f"CSV file not found: {csv_path}")
                return
            
            self.logger.info(f"📊 Loading genome data from {csv_path}")
            
            # Load CSV data
            df = pd.read_csv(csv_path)
            
            # Process and index documents
            index_name = self.genome_config.elasticsearch_config.get("virus_genome_index", "virus-genomes")
            indexed_count = 0
            
            for _, row in df.iterrows():
                try:
                    # Create document for Elasticsearch
                    doc = {
                        "genome_id": str(row.get("genome_id", "")),
                        "genome_name": str(row.get("genome_name", "")),
                        "organism_name": str(row.get("organism_name", "")),
                        "strain": str(row.get("strain", "")),
                        "genome_length": int(row.get("genome_length", 0)) if pd.notna(row.get("genome_length")) else 0,
                        "taxon_id": str(row.get("taxon_id", "")),
                        "taxon_lineage": str(row.get("taxon_lineage_names", "")),
                        "host": str(row.get("host", "")),
                        "isolation_country": str(row.get("isolation_country", "")),
                        "genome_status": str(row.get("genome_status", "")),
                        "genome_type": str(row.get("genome_type", "")),
                        "description": str(row.get("product_name", "")),
                        
                        # Add synonyms and abbreviations based on genome name
                        "synonyms": self._extract_synonyms(str(row.get("genome_name", ""))),
                        "abbreviations": self._extract_abbreviations(str(row.get("genome_name", "")))
                    }
                    
                    # Index document
                    doc_id = doc["genome_id"] or f"genome_{indexed_count}"
                    success = await self.elasticsearch_tool.index_document(
                        index=index_name,
                        doc_id=doc_id,
                        document=doc
                    )
                    
                    if success:
                        indexed_count += 1
                        
                        # Log progress every 1000 documents
                        if indexed_count % 1000 == 0:
                            self.logger.info(f"📊 Indexed {indexed_count} documents...")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to index genome {row.get('genome_id', 'unknown')}: {e}")
                    continue
            
            self.logger.info(f"✅ Successfully indexed {indexed_count} genome documents")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to populate genome index from CSV: {e}")
    
    def _extract_synonyms(self, genome_name: str) -> List[str]:
        """Extract synonyms from genome name"""
        synonyms = []
        
        # Common virus synonyms
        synonym_map = {
            "Chikungunya": ["CHIKV", "Chikungunya virus"],
            "Eastern equine encephalitis": ["EEEV", "Eastern equine encephalitis virus"],
            "Western equine encephalitis": ["WEEV", "Western equine encephalitis virus"],
            "Venezuelan equine encephalitis": ["VEEV", "Venezuelan equine encephalitis virus"],
            "Sindbis": ["SINV", "Sindbis virus"],
            "Semliki Forest": ["SFV", "Semliki Forest virus"],
            "Ross River": ["RRV", "Ross River virus"],
            "Barmah Forest": ["BFV", "Barmah Forest virus"],
            "Mayaro": ["MAYV", "Mayaro virus"],
            "Una": ["UNAV", "Una virus"]
        }
        
        for key, values in synonym_map.items():
            if key.lower() in genome_name.lower():
                synonyms.extend(values)
        
        return synonyms
    
    def _extract_abbreviations(self, genome_name: str) -> List[str]:
        """Extract abbreviations from genome name"""
        abbreviations = []
        
        # Extract potential abbreviations (uppercase sequences)
        import re
        potential_abbrevs = re.findall(r'\b[A-Z]{2,}\b', genome_name)
        abbreviations.extend(potential_abbrevs)
        
        # Add common patterns
        if "virus" in genome_name.lower():
            # Extract first letters of words before "virus"
            words = genome_name.lower().split()
            if "virus" in words:
                virus_index = words.index("virus")
                if virus_index > 0:
                    abbrev = "".join(word[0].upper() for word in words[:virus_index] if word.isalpha())
                    if len(abbrev) >= 2:
                        abbreviations.append(abbrev)
                        abbreviations.append(abbrev + "V")  # Add "V" for virus
        
        return list(set(abbreviations))  # Remove duplicates
    
    async def execute_step(self, input_data: DataUnit) -> DataUnit:
        """Execute genome search step"""
        try:
            self.logger.info("🔍 Executing genome search step...")
            
            # Extract virus name from input data
            virus_name = self._extract_virus_name_from_input(input_data)
            
            if not virus_name:
                raise ValueError("No virus name found in input data")
            
            self.logger.info(f"🔍 Searching for virus: '{virus_name}'")
            
            # Perform genome search
            search_result = await self.search_virus_genomes(virus_name)
            
            # Create output data unit
            output_data = DataUnit(
                content={
                    "search_query": virus_name,
                    "search_result": search_result,
                    "genome_matches": search_result.matches,
                    "search_metadata": {
                        "total_found": search_result.total_found,
                        "search_time": search_result.search_time,
                        "search_method": search_result.search_method,
                        "confidence_scores": search_result.confidence_scores,
                        "cached": search_result.cached
                    }
                },
                metadata={
                    "step_name": self.name,
                    "execution_time": search_result.search_time,
                    "search_successful": search_result.total_found > 0
                }
            )
            
            self.logger.info(f"✅ Found {search_result.total_found} genome matches in {search_result.search_time:.2f}s")
            
            return output_data
            
        except Exception as e:
            self.logger.error(f"❌ Genome search step execution failed: {e}")
            
            # Return error result
            return DataUnit(
                content={
                    "search_query": getattr(self, '_last_virus_name', 'unknown'),
                    "error": str(e),
                    "genome_matches": []
                },
                metadata={
                    "step_name": self.name,
                    "execution_failed": True,
                    "error_message": str(e)
                }
            )
    
    def _extract_virus_name_from_input(self, input_data: DataUnit) -> str:
        """Extract virus name from input data unit"""
        try:
            # Try different possible keys for virus name
            content = input_data.content
            
            # Direct virus name
            if "virus_name" in content:
                return str(content["virus_name"])
            
            # Query or search term
            if "query" in content:
                return str(content["query"])
            
            # User message (extract virus name from natural language)
            if "user_message" in content:
                return self._extract_virus_from_message(str(content["user_message"]))
            
            # Raw content as string
            if isinstance(content, str):
                return content
            
            # Default: try to find any string value
            for key, value in content.items():
                if isinstance(value, str) and len(value.strip()) > 0:
                    return value.strip()
            
            raise ValueError("No virus name found in input data")
            
        except Exception as e:
            self.logger.error(f"Failed to extract virus name from input: {e}")
            raise
    
    def _extract_virus_from_message(self, message: str) -> str:
        """Extract virus name from natural language message"""
        # Simple extraction patterns for common virus names
        import re
        
        # Common patterns
        patterns = [
            r'\b(CHIKV|Chikungunya)\b',
            r'\b(EEEV|Eastern equine encephalitis)\b',
            r'\b(WEEV|Western equine encephalitis)\b', 
            r'\b(VEEV|Venezuelan equine encephalitis)\b',
            r'\b(EEE virus|Eastern equine encephalitis virus)\b',
            r'\b(\w+\s*virus)\b',
            r'\b([A-Z]{2,4}V?)\b'  # Abbreviations
        ]
        
        for pattern in patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                return match.group(1)
        
        # If no pattern matches, return the whole message for fuzzy search
        return message.strip()
    
    async def search_virus_genomes(self, virus_name: str) -> GenomeSearchResult:
        """Search for virus genomes using Elasticsearch or CSV fallback"""
        self._last_virus_name = virus_name
        self.total_searches += 1
        
        # Check cache first
        if self.genome_config.cache_results:
            cached_result = self._get_cached_result(virus_name)
            if cached_result:
                self.cache_hits += 1
                self.logger.debug(f"🎯 Cache hit for virus: {virus_name}")
                cached_result.cached = True
                return cached_result
        
        start_time = time.time()
        
        # Try Elasticsearch search first
        if self.elasticsearch_tool:
            try:
                result = await self._elasticsearch_search(virus_name)
                search_time = time.time() - start_time
                
                result.search_time = search_time
                result.search_method = "elasticsearch"
                
                # Cache result
                if self.genome_config.cache_results:
                    self._cache_result(virus_name, result)
                
                self.elasticsearch_searches += 1
                return result
                
            except Exception as e:
                self.logger.warning(f"⚠️ Elasticsearch search failed: {e}")
                
                if not self.genome_config.fallback_to_csv:
                    raise
        
        # Fallback to CSV search
        if self.genome_config.fallback_to_csv:
            try:
                result = await self._csv_fallback_search(virus_name)
                search_time = time.time() - start_time
                
                result.search_time = search_time
                result.search_method = "csv_fallback"
                
                # Cache result
                if self.genome_config.cache_results:
                    self._cache_result(virus_name, result)
                
                self.csv_fallback_searches += 1
                return result
                
            except Exception as e:
                self.logger.error(f"❌ CSV fallback search also failed: {e}")
                raise
        
        # If we get here, both methods failed
        raise Exception("All search methods failed")
    
    async def _elasticsearch_search(self, virus_name: str) -> GenomeSearchResult:
        """Perform Elasticsearch-based genome search"""
        try:
            # Use the specialized virus genome search
            results = await self.elasticsearch_tool.search_virus_genomes(
                virus_name,
                confidence_threshold=self.genome_config.confidence_threshold
            )
            
            # Limit results
            limited_results = results[:self.genome_config.max_results]
            
            # Extract confidence scores
            confidence_scores = [result.get("confidence", 0.0) for result in limited_results]
            
            return GenomeSearchResult(
                query=virus_name,
                matches=limited_results,
                total_found=len(results),
                search_time=0.0,  # Will be set by caller
                confidence_scores=confidence_scores,
                search_method="elasticsearch"
            )
            
        except Exception as e:
            self.logger.error(f"Elasticsearch search failed for '{virus_name}': {e}")
            raise
    
    async def _csv_fallback_search(self, virus_name: str) -> GenomeSearchResult:
        """Perform CSV-based fallback search"""
        try:
            # Use legacy CSV-based fuzzy search
            from fuzzywuzzy import fuzz, process
            import pandas as pd
            from pathlib import Path
            
            csv_path = Path(self.genome_config.csv_file_path)
            if not csv_path.exists():
                raise FileNotFoundError(f"CSV file not found: {csv_path}")
            
            # Load CSV data
            df = pd.read_csv(csv_path)
            
            # Prepare search data
            genome_names = df["genome_name"].fillna("").astype(str).tolist()
            
            # Perform fuzzy search
            matches = process.extract(
                virus_name,
                genome_names,
                scorer=fuzz.partial_ratio,
                limit=self.genome_config.max_results
            )
            
            # Filter by confidence threshold
            confidence_threshold_percent = self.genome_config.confidence_threshold * 100
            filtered_matches = [
                (name, score) for name, score in matches
                if score >= confidence_threshold_percent
            ]
            
            # Convert to result format
            results = []
            confidence_scores = []
            
            for genome_name, score in filtered_matches:
                # Find corresponding row in dataframe
                row = df[df["genome_name"] == genome_name].iloc[0]
                
                result = {
                    "genome_id": str(row.get("genome_id", "")),
                    "genome_name": str(row.get("genome_name", "")),
                    "organism_name": str(row.get("organism_name", "")),
                    "genome_length": int(row.get("genome_length", 0)) if pd.notna(row.get("genome_length")) else 0,
                    "taxon_id": str(row.get("taxon_id", "")),
                    "confidence": score / 100.0,  # Convert to 0-1 scale
                    "search_score": score
                }
                
                results.append(result)
                confidence_scores.append(score / 100.0)
            
            return GenomeSearchResult(
                query=virus_name,
                matches=results,
                total_found=len(results),
                search_time=0.0,  # Will be set by caller
                confidence_scores=confidence_scores,
                search_method="csv_fallback"
            )
            
        except Exception as e:
            self.logger.error(f"CSV fallback search failed for '{virus_name}': {e}")
            raise
    
    def _get_cached_result(self, virus_name: str) -> Optional[GenomeSearchResult]:
        """Get cached search result"""
        cache_key = virus_name.lower().strip()
        
        if cache_key in self.search_cache:
            cached_result = self.search_cache[cache_key]
            
            # Check if cache is still valid
            cache_age = time.time() - getattr(cached_result, '_cache_time', 0)
            if cache_age < self.genome_config.cache_ttl:
                return cached_result
            else:
                # Remove expired cache entry
                del self.search_cache[cache_key]
        
        return None
    
    def _cache_result(self, virus_name: str, result: GenomeSearchResult):
        """Cache search result"""
        cache_key = virus_name.lower().strip()
        result._cache_time = time.time()
        self.search_cache[cache_key] = result
        
        # Limit cache size (keep last 100 searches)
        if len(self.search_cache) > 100:
            # Remove oldest entries
            oldest_key = min(self.search_cache.keys(), key=lambda k: getattr(self.search_cache[k], '_cache_time', 0))
            del self.search_cache[oldest_key]
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Get search statistics"""
        return {
            "total_searches": self.total_searches,
            "elasticsearch_searches": self.elasticsearch_searches,
            "csv_fallback_searches": self.csv_fallback_searches,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": (self.cache_hits / self.total_searches * 100) if self.total_searches > 0 else 0,
            "cached_entries": len(self.search_cache),
            "elasticsearch_available": self.elasticsearch_tool is not None
        } 