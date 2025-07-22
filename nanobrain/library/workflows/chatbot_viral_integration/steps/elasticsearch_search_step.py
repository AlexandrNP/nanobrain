#!/usr/bin/env python3
"""
Elasticsearch Search Step for Chatbot Viral Integration

This step performs searches against indexed viral data to enhance chatbot
responses with relevant context. It integrates with the Elasticsearch MCP
server to provide powerful search capabilities including:

- Full-text search across protein sequences and metadata
- Semantic search using vector embeddings
- Contextual search based on chat history
- Aggregation queries for analytics
- Real-time search during conversation flow

Key Features:
- MCP client integration for standardized access
- Multiple search strategies (keyword, semantic, hybrid)
- Context-aware result ranking
- Search result enrichment with related data
- Performance optimization with caching
- Error handling and fallback mechanisms

Integration Points:
- Receives user queries from chat workflow
- Searches indexed protein and analysis data
- Returns enriched context for response generation
- Supports follow-up questions and clarifications

Usage:
    # Configure in workflow YAML
    - name: "elasticsearch_search"
      class: "ElasticsearchSearchStep"
      config:
        mcp_server_url: "http://elasticsearch-mcp:9202"
        search_indices: ["viral_proteins", "analysis_results"]
        enable_semantic_search: true
"""

import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field

# NanoBrain imports
from nanobrain.core.step import BaseStep, StepConfig
from nanobrain.core.data_unit import DataUnit
from nanobrain.core.mcp_support import MCPClient, MCPClientConfig, MCPServerConfig
from nanobrain.core.logging_system import get_logger


@dataclass
class ElasticsearchSearchConfig(StepConfig):
    """Configuration for Elasticsearch search step"""
    
    # Step identification
    step_name: str = "elasticsearch_search"
    
    # MCP server configuration
    mcp_server_url: str = "http://localhost:9202"
    mcp_server_name: str = "elasticsearch"
    connection_timeout: float = 30.0
    request_timeout: float = 30.0
    
    # Search configuration
    search_indices: List[str] = field(default_factory=lambda: [
        "viral_proteins", "analysis_results", "chat_interactions"
    ])
    
    # Search strategies
    enable_semantic_search: bool = True
    enable_hybrid_search: bool = True
    semantic_search_weight: float = 0.7
    keyword_search_weight: float = 0.3
    
    # Search parameters
    max_results: int = 10
    min_score_threshold: float = 0.1
    context_window_size: int = 5
    
    # Embedding configuration
    embedding_field: str = "embedding"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Result enrichment
    include_related_proteins: bool = True
    include_analysis_context: bool = True
    include_interaction_history: bool = True
    
    # Performance optimization
    enable_caching: bool = True
    cache_ttl: int = 300  # 5 minutes
    
    # Tool card (mandatory)
    tool_card: Dict[str, Any] = field(default_factory=lambda: {
        "name": "elasticsearch_search_step",
        "description": "Searches viral data in Elasticsearch to enhance chatbot responses",
        "version": "1.0.0",
        "category": "search_enhancement",
        "capabilities": ["search", "context_enrichment", "mcp_integration"]
    })


class ElasticsearchSearchStep(BaseStep):
    """
    Step to search viral protein data and analysis results in Elasticsearch
    to enhance chatbot responses with relevant context.
    """
    
    @classmethod
    def from_config(cls, config: Union[ElasticsearchSearchConfig, Dict], **kwargs) -> 'ElasticsearchSearchStep':
        """Mandatory from_config implementation"""
        logger = get_logger(f"{cls.__name__}.from_config")
        logger.info(f"Creating {cls.__name__} from configuration")
        
        # ✅ FRAMEWORK COMPLIANCE: Convert to ElasticsearchSearchConfig using from_config
        if isinstance(config, dict):
            config = ElasticsearchSearchConfig.from_config(config)
        elif not isinstance(config, ElasticsearchSearchConfig):
            # Convert other config types
            config_dict = config.model_dump() if hasattr(config, 'model_dump') else {}
            config = ElasticsearchSearchConfig.from_config(config_dict)
        
        # Validate tool card
        if not hasattr(config, 'tool_card') or not config.tool_card:
            raise ValueError(f"Missing mandatory 'tool_card' in configuration for {cls.__name__}")
        
        instance = cls(config, **kwargs)
        instance._tool_card_data = config.tool_card
        
        logger.info(f"Successfully created {cls.__name__} with search capabilities")
        return instance
    
    def __init__(self, config: ElasticsearchSearchConfig, **kwargs):
        """Initialize Elasticsearch search step"""
        super().__init__(config, **kwargs)
        
        self.config = config
        self.logger = get_logger(f"elasticsearch_search_step")
        
        # MCP client for Elasticsearch communication
        self.mcp_client: Optional[MCPClient] = None
        
        # Embedding model (lazy loaded)
        self.embedding_model = None
        
        # Search cache
        self.search_cache: Dict[str, Tuple[Any, datetime]] = {}
        
        # Statistics
        self.searches_performed = 0
        self.cache_hits = 0
        self.search_errors = 0
    
    async def initialize(self):
        """Initialize the step and MCP client"""
        await super().initialize()
        
        try:
            # ✅ FRAMEWORK COMPLIANCE: Initialize MCP client using from_config
            mcp_client_config = MCPClientConfig.from_config({
                'default_timeout': self.config.connection_timeout,
                'default_max_retries': 3,
                'default_retry_delay': 1.0
            })
            
            self.mcp_client = MCPClient(mcp_client_config, logger=self.logger)
            await self.mcp_client.initialize()
            
            # ✅ FRAMEWORK COMPLIANCE: Add Elasticsearch MCP server using from_config
            elasticsearch_server_config = MCPServerConfig.from_config({
                'name': self.config.mcp_server_name,
                'url': self.config.mcp_server_url,
                'description': "Elasticsearch MCP server for viral protein search",
                'timeout': self.config.request_timeout
            })
            
            self.mcp_client.add_server(elasticsearch_server_config)
            
            # Connect to server
            connected = await self.mcp_client.connect_to_server(self.config.mcp_server_name)
            if not connected:
                raise RuntimeError(f"Failed to connect to Elasticsearch MCP server at {self.config.mcp_server_url}")
            
            # Initialize embedding model if needed
            if self.config.enable_semantic_search:
                await self._initialize_embedding_model()
            
            self.logger.info("✅ Elasticsearch search step initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Elasticsearch search step: {e}")
            raise
    
    async def _initialize_embedding_model(self):
        """Initialize embedding model for semantic search"""
        try:
            # Try to import sentence-transformers
            from sentence_transformers import SentenceTransformer
            
            self.embedding_model = SentenceTransformer(self.config.embedding_model)
            self.logger.info(f"✅ Loaded embedding model: {self.config.embedding_model}")
            
        except ImportError:
            self.logger.warning("⚠️ sentence-transformers not available, disabling semantic search")
            self.config.enable_semantic_search = False
        except Exception as e:
            self.logger.error(f"❌ Failed to load embedding model: {e}")
            self.config.enable_semantic_search = False
    
    async def execute(self, data_unit: DataUnit) -> DataUnit:
        """Execute the search step"""
        self.logger.info("🔍 Starting Elasticsearch search...")
        
        try:
            # Ensure step is initialized
            await self.ensure_initialized()
            
            # Extract search query from data unit
            search_query = self._extract_search_query(data_unit)
            if not search_query:
                self.logger.warning("⚠️ No search query found in data unit")
                return data_unit
            
            # Perform search
            search_results = await self._perform_comprehensive_search(search_query, data_unit)
            
            # Enrich results with context
            enriched_results = await self._enrich_search_results(search_results, data_unit)
            
            # Add search results to data unit
            data_unit.search_results = enriched_results
            data_unit.search_metadata = {
                "query": search_query,
                "total_results": len(enriched_results.get("hits", [])),
                "search_strategy": self._get_search_strategy(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "indices_searched": self.config.search_indices
            }
            
            self.searches_performed += 1
            
            self.logger.info(f"✅ Search completed: {len(enriched_results.get('hits', []))} results found")
            
            return data_unit
            
        except Exception as e:
            self.logger.error(f"❌ Elasticsearch search failed: {e}")
            self.search_errors += 1
            
            # Add error information to data unit
            data_unit.search_results = {"error": str(e), "hits": []}
            data_unit.search_metadata = {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            # Don't fail the entire workflow - continue with empty results
            self.logger.warning("⚠️ Continuing workflow with empty search results")
            return data_unit
    
    def _extract_search_query(self, data_unit: DataUnit) -> Optional[str]:
        """Extract search query from data unit"""
        # Try different possible locations for the search query
        query_fields = [
            'user_query', 'query', 'search_query', 'question', 
            'input_text', 'message', 'chat_message'
        ]
        
        for field in query_fields:
            if hasattr(data_unit, field):
                query = getattr(data_unit, field)
                if query and isinstance(query, str):
                    return query.strip()
        
        # Try to extract from nested structures
        if hasattr(data_unit, 'chat_context') and isinstance(data_unit.chat_context, dict):
            for field in query_fields:
                if field in data_unit.chat_context:
                    query = data_unit.chat_context[field]
                    if query and isinstance(query, str):
                        return query.strip()
        
        return None
    
    async def _perform_comprehensive_search(self, query: str, data_unit: DataUnit) -> Dict[str, Any]:
        """Perform comprehensive search using multiple strategies"""
        # Check cache first
        if self.config.enable_caching:
            cached_result = self._get_cached_result(query)
            if cached_result:
                self.cache_hits += 1
                self.logger.debug("✅ Cache hit for search query")
                return cached_result
        
        search_results = {"hits": [], "total": 0, "max_score": 0.0}
        
        try:
            if self.config.enable_hybrid_search and self.config.enable_semantic_search:
                # Perform hybrid search (keyword + semantic)
                search_results = await self._hybrid_search(query, data_unit)
            elif self.config.enable_semantic_search:
                # Perform semantic search only
                search_results = await self._semantic_search(query, data_unit)
            else:
                # Perform keyword search only
                search_results = await self._keyword_search(query, data_unit)
            
            # Cache the results
            if self.config.enable_caching:
                self._cache_result(query, search_results)
            
            return search_results
            
        except Exception as e:
            self.logger.error(f"❌ Comprehensive search failed: {e}")
            raise
    
    async def _keyword_search(self, query: str, data_unit: DataUnit) -> Dict[str, Any]:
        """Perform keyword-based search"""
        self.logger.debug("🔍 Performing keyword search...")
        
        all_hits = []
        
        for index in self.config.search_indices:
            try:
                # Build keyword search query
                search_query = {
                    "bool": {
                        "should": [
                            {"match": {"sequence": {"query": query, "boost": 2.0}}},
                            {"match": {"function": {"query": query, "boost": 1.5}}},
                            {"match": {"organism": {"query": query, "boost": 1.2}}},
                            {"match": {"protein_id": {"query": query, "boost": 3.0}}},
                            {"multi_match": {
                                "query": query,
                                "fields": ["*"],
                                "type": "best_fields"
                            }}
                        ],
                        "minimum_should_match": 1
                    }
                }
                
                result = await self.mcp_client.call_tool(
                    self.config.mcp_server_name,
                    "search",
                    {
                        "index": index,
                        "query": search_query,
                        "size": self.config.max_results,
                        "from": 0
                    }
                )
                
                if result.get("search_completed"):
                    hits = result.get("hits", [])
                    for hit in hits:
                        hit["index"] = index
                        hit["search_type"] = "keyword"
                    all_hits.extend(hits)
                
            except Exception as e:
                self.logger.warning(f"⚠️ Keyword search failed for index {index}: {e}")
                continue
        
        # Sort by score and limit results
        all_hits.sort(key=lambda x: x.get("score", 0), reverse=True)
        all_hits = all_hits[:self.config.max_results]
        
        return {
            "hits": all_hits,
            "total": len(all_hits),
            "max_score": all_hits[0]["score"] if all_hits else 0.0
        }
    
    async def _semantic_search(self, query: str, data_unit: DataUnit) -> Dict[str, Any]:
        """Perform semantic search using embeddings"""
        if not self.embedding_model:
            self.logger.warning("⚠️ Embedding model not available, falling back to keyword search")
            return await self._keyword_search(query, data_unit)
        
        self.logger.debug("🧠 Performing semantic search...")
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query).tolist()
            
            all_hits = []
            
            for index in self.config.search_indices:
                try:
                    result = await self.mcp_client.call_tool(
                        self.config.mcp_server_name,
                        "semantic_search",
                        {
                            "index": index,
                            "query_vector": query_embedding,
                            "embedding_field": self.config.embedding_field,
                            "size": self.config.max_results
                        }
                    )
                    
                    if result.get("semantic_search_completed"):
                        hits = result.get("hits", [])
                        for hit in hits:
                            hit["index"] = index
                            hit["search_type"] = "semantic"
                        all_hits.extend(hits)
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Semantic search failed for index {index}: {e}")
                    continue
            
            # Sort by score and limit results
            all_hits.sort(key=lambda x: x.get("score", 0), reverse=True)
            all_hits = all_hits[:self.config.max_results]
            
            return {
                "hits": all_hits,
                "total": len(all_hits),
                "max_score": all_hits[0]["score"] if all_hits else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"❌ Semantic search failed: {e}")
            # Fallback to keyword search
            return await self._keyword_search(query, data_unit)
    
    async def _hybrid_search(self, query: str, data_unit: DataUnit) -> Dict[str, Any]:
        """Perform hybrid search combining keyword and semantic results"""
        self.logger.debug("🔀 Performing hybrid search...")
        
        try:
            # Perform both searches in parallel
            keyword_task = asyncio.create_task(self._keyword_search(query, data_unit))
            semantic_task = asyncio.create_task(self._semantic_search(query, data_unit))
            
            keyword_results, semantic_results = await asyncio.gather(
                keyword_task, semantic_task, return_exceptions=True
            )
            
            # Handle exceptions
            if isinstance(keyword_results, Exception):
                self.logger.warning(f"⚠️ Keyword search failed: {keyword_results}")
                keyword_results = {"hits": [], "total": 0, "max_score": 0.0}
            
            if isinstance(semantic_results, Exception):
                self.logger.warning(f"⚠️ Semantic search failed: {semantic_results}")
                semantic_results = {"hits": [], "total": 0, "max_score": 0.0}
            
            # Combine and rank results using RRF (Reciprocal Rank Fusion)
            combined_hits = self._combine_search_results(
                keyword_results.get("hits", []),
                semantic_results.get("hits", [])
            )
            
            return {
                "hits": combined_hits,
                "total": len(combined_hits),
                "max_score": combined_hits[0]["score"] if combined_hits else 0.0,
                "search_strategy": "hybrid"
            }
            
        except Exception as e:
            self.logger.error(f"❌ Hybrid search failed: {e}")
            # Fallback to keyword search
            return await self._keyword_search(query, data_unit)
    
    def _combine_search_results(self, keyword_hits: List[Dict], semantic_hits: List[Dict]) -> List[Dict]:
        """Combine keyword and semantic search results using RRF"""
        # Create document ID to hit mapping
        doc_map = {}
        
        # Process keyword hits
        for rank, hit in enumerate(keyword_hits):
            doc_id = hit.get("id", "")
            if doc_id not in doc_map:
                doc_map[doc_id] = hit.copy()
                doc_map[doc_id]["keyword_rank"] = rank + 1
                doc_map[doc_id]["keyword_score"] = hit.get("score", 0)
                doc_map[doc_id]["search_types"] = ["keyword"]
            else:
                doc_map[doc_id]["keyword_rank"] = rank + 1
                doc_map[doc_id]["keyword_score"] = hit.get("score", 0)
                doc_map[doc_id]["search_types"].append("keyword")
        
        # Process semantic hits
        for rank, hit in enumerate(semantic_hits):
            doc_id = hit.get("id", "")
            if doc_id not in doc_map:
                doc_map[doc_id] = hit.copy()
                doc_map[doc_id]["semantic_rank"] = rank + 1
                doc_map[doc_id]["semantic_score"] = hit.get("score", 0)
                doc_map[doc_id]["search_types"] = ["semantic"]
            else:
                doc_map[doc_id]["semantic_rank"] = rank + 1
                doc_map[doc_id]["semantic_score"] = hit.get("score", 0)
                if "semantic" not in doc_map[doc_id]["search_types"]:
                    doc_map[doc_id]["search_types"].append("semantic")
        
        # Calculate RRF scores
        for doc_id, hit in doc_map.items():
            rrf_score = 0
            
            # Keyword contribution
            if "keyword_rank" in hit:
                rrf_score += self.config.keyword_search_weight / (60 + hit["keyword_rank"])
            
            # Semantic contribution
            if "semantic_rank" in hit:
                rrf_score += self.config.semantic_search_weight / (60 + hit["semantic_rank"])
            
            hit["rrf_score"] = rrf_score
            hit["score"] = rrf_score  # Use RRF score as the main score
        
        # Sort by RRF score and limit results
        combined_hits = list(doc_map.values())
        combined_hits.sort(key=lambda x: x.get("rrf_score", 0), reverse=True)
        
        # Filter by minimum score threshold
        if self.config.min_score_threshold > 0:
            combined_hits = [
                hit for hit in combined_hits 
                if hit.get("rrf_score", 0) >= self.config.min_score_threshold
            ]
        
        return combined_hits[:self.config.max_results]
    
    async def _enrich_search_results(self, search_results: Dict[str, Any], data_unit: DataUnit) -> Dict[str, Any]:
        """Enrich search results with additional context"""
        enriched_hits = []
        
        for hit in search_results.get("hits", []):
            enriched_hit = hit.copy()
            
            try:
                # Add related proteins if enabled
                if self.config.include_related_proteins:
                    related_proteins = await self._find_related_proteins(hit)
                    if related_proteins:
                        enriched_hit["related_proteins"] = related_proteins
                
                # Add analysis context if enabled
                if self.config.include_analysis_context:
                    analysis_context = await self._find_analysis_context(hit)
                    if analysis_context:
                        enriched_hit["analysis_context"] = analysis_context
                
                # Add interaction history if enabled
                if self.config.include_interaction_history:
                    interaction_history = await self._find_interaction_history(hit, data_unit)
                    if interaction_history:
                        enriched_hit["interaction_history"] = interaction_history
                
                enriched_hits.append(enriched_hit)
                
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to enrich search result {hit.get('id', 'unknown')}: {e}")
                enriched_hits.append(hit)  # Use original hit without enrichment
        
        return {
            **search_results,
            "hits": enriched_hits,
            "enriched": True
        }
    
    async def _find_related_proteins(self, hit: Dict) -> List[Dict]:
        """Find proteins related to the current hit"""
        try:
            source = hit.get("source", {})
            organism = source.get("organism")
            function = source.get("function")
            
            if not organism and not function:
                return []
            
            # Build query for related proteins
            related_query = {
                "bool": {
                    "should": [],
                    "must_not": [
                        {"term": {"protein_id": hit.get("id", "")}}
                    ]
                }
            }
            
            if organism:
                related_query["bool"]["should"].append(
                    {"match": {"organism": organism}}
                )
            
            if function:
                related_query["bool"]["should"].append(
                    {"match": {"function": function}}
                )
            
            if not related_query["bool"]["should"]:
                return []
            
            result = await self.mcp_client.call_tool(
                self.config.mcp_server_name,
                "search",
                {
                    "index": "viral_proteins",
                    "query": related_query,
                    "size": 3  # Limit to 3 related proteins
                }
            )
            
            if result.get("search_completed"):
                return result.get("hits", [])[:3]
            
        except Exception as e:
            self.logger.debug(f"Failed to find related proteins: {e}")
        
        return []
    
    async def _find_analysis_context(self, hit: Dict) -> List[Dict]:
        """Find analysis results related to the current hit"""
        try:
            protein_id = hit.get("id", "")
            if not protein_id:
                return []
            
            # Search for analysis results containing this protein
            analysis_query = {
                "bool": {
                    "should": [
                        {"term": {"protein_ids": protein_id}},
                        {"nested": {
                            "path": "results.proteins",
                            "query": {"term": {"results.proteins.protein_id": protein_id}}
                        }}
                    ]
                }
            }
            
            result = await self.mcp_client.call_tool(
                self.config.mcp_server_name,
                "search",
                {
                    "index": "analysis_results",
                    "query": analysis_query,
                    "size": 2  # Limit to 2 analysis results
                }
            )
            
            if result.get("search_completed"):
                return result.get("hits", [])[:2]
            
        except Exception as e:
            self.logger.debug(f"Failed to find analysis context: {e}")
        
        return []
    
    async def _find_interaction_history(self, hit: Dict, data_unit: DataUnit) -> List[Dict]:
        """Find relevant chat interaction history"""
        try:
            # Get recent interactions mentioning similar topics
            protein_id = hit.get("id", "")
            source = hit.get("source", {})
            organism = source.get("organism", "")
            
            if not protein_id and not organism:
                return []
            
            # Build query for relevant interactions
            interaction_query = {
                "bool": {
                    "should": [],
                    "filter": [
                        {"range": {"timestamp": {"gte": "now-7d"}}}  # Last 7 days
                    ]
                }
            }
            
            if protein_id:
                interaction_query["bool"]["should"].append(
                    {"match": {"query": protein_id}}
                )
            
            if organism:
                interaction_query["bool"]["should"].append(
                    {"match": {"query": organism}}
                )
            
            if not interaction_query["bool"]["should"]:
                return []
            
            result = await self.mcp_client.call_tool(
                self.config.mcp_server_name,
                "search",
                {
                    "index": "chat_interactions",
                    "query": interaction_query,
                    "size": 2  # Limit to 2 interactions
                }
            )
            
            if result.get("search_completed"):
                return result.get("hits", [])[:2]
            
        except Exception as e:
            self.logger.debug(f"Failed to find interaction history: {e}")
        
        return []
    
    def _get_search_strategy(self) -> str:
        """Get the search strategy being used"""
        if self.config.enable_hybrid_search and self.config.enable_semantic_search:
            return "hybrid"
        elif self.config.enable_semantic_search:
            return "semantic"
        else:
            return "keyword"
    
    def _get_cached_result(self, query: str) -> Optional[Dict[str, Any]]:
        """Get cached search result if available and not expired"""
        if query not in self.search_cache:
            return None
        
        result, timestamp = self.search_cache[query]
        
        # Check if cache entry is still valid
        if (datetime.now() - timestamp).total_seconds() > self.config.cache_ttl:
            del self.search_cache[query]
            return None
        
        return result
    
    def _cache_result(self, query: str, result: Dict[str, Any]):
        """Cache search result"""
        self.search_cache[query] = (result, datetime.now())
        
        # Clean up old cache entries if cache is getting large
        if len(self.search_cache) > 100:
            # Remove oldest 20% of entries
            sorted_entries = sorted(
                self.search_cache.items(),
                key=lambda x: x[1][1]
            )
            
            for query_to_remove, _ in sorted_entries[:20]:
                del self.search_cache[query_to_remove]
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.mcp_client:
                await self.mcp_client.shutdown()
                self.mcp_client = None
            
            # Clear cache
            self.search_cache.clear()
            
            self.logger.info("✅ Elasticsearch search step cleanup completed")
            
        except Exception as e:
            self.logger.error(f"❌ Error during cleanup: {e}")
        
        await super().cleanup()
    
    # Step interface methods
    
    def get_step_info(self) -> Dict[str, Any]:
        """Get step information"""
        return {
            "name": self.config.step_name,
            "type": "elasticsearch_search",
            "description": "Searches viral data to enhance chatbot responses",
            "mcp_server": self.config.mcp_server_url,
            "search_indices": self.config.search_indices,
            "search_strategy": self._get_search_strategy(),
            "statistics": {
                "searches_performed": self.searches_performed,
                "cache_hits": self.cache_hits,
                "search_errors": self.search_errors,
                "cache_size": len(self.search_cache)
            }
        }
    
    def validate_configuration(self) -> bool:
        """Validate step configuration"""
        try:
            # Check required configuration
            if not self.config.mcp_server_url:
                self.logger.error("❌ MCP server URL not configured")
                return False
            
            if not self.config.search_indices:
                self.logger.error("❌ No search indices configured")
                return False
            
            # Check MCP client availability
            if not self.mcp_client:
                self.logger.warning("⚠️ MCP client not initialized")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Configuration validation failed: {e}")
            return False 