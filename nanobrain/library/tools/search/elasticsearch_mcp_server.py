#!/usr/bin/env python3
"""
Elasticsearch MCP Server for NanoBrain Framework

This module provides an MCP (Model Context Protocol) server that exposes Elasticsearch
functionality to other NanoBrain components. It enables:

- Document indexing and bulk operations
- Full-text and semantic search capabilities
- Index management and configuration
- Real-time search and analytics
- Integration with Docker-based Elasticsearch deployment

The server follows the mandatory from_config pattern and integrates with the
NanoBrain Docker infrastructure for service management.

Key Features:
- MCP protocol compliance for standardized tool access
- Elasticsearch 8.x integration with modern features
- Semantic search with vector embeddings
- Bulk operations for high-throughput indexing
- Health monitoring and error handling
- Docker service integration
- Configuration-driven index templates

Usage:
    # Create from configuration
    config = ElasticsearchMCPConfig.from_yaml("elasticsearch_config.yml")
    server = ElasticsearchMCPServer.from_config(config)
    
    # Start the MCP server
    await server.start()
    
    # Server automatically exposes tools via MCP protocol
"""

import asyncio
import json
import time
import uuid
import os
import csv
import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field

# CSV processing imports
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from fuzzywuzzy import fuzz, process
    FUZZYWUZZY_AVAILABLE = True
except ImportError:
    FUZZYWUZZY_AVAILABLE = False

try:
    import chardet
    CHARDET_AVAILABLE = True
except ImportError:
    CHARDET_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from elasticsearch import AsyncElasticsearch
    from elasticsearch.exceptions import (
        ConnectionError as ESConnectionError,
        NotFoundError,
        RequestError,
        TransportError
    )
    ELASTICSEARCH_AVAILABLE = True
except ImportError:
    ELASTICSEARCH_AVAILABLE = False

try:
    import aiohttp
    from aiohttp import web
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

# NanoBrain imports
from nanobrain.core.mcp_support import MCPServerConfig, MCPError
from nanobrain.core.external_tool import ExternalTool, ExternalToolConfig
from nanobrain.core.logging_system import get_logger

# Docker management imports (conditional) - only when not in Docker
if os.getenv("NANOBRAIN_ENV") != "docker":
    try:
        from nanobrain.library.infrastructure.docker import (
            DockerManager, 
            DockerComponentConfig,
            ContainerConfig
        )
        DOCKER_AVAILABLE = True
    except ImportError:
        DOCKER_AVAILABLE = False
else:
    # When running in Docker, don't import Docker components to avoid dependency issues
    DOCKER_AVAILABLE = False


@dataclass
class ElasticsearchMCPConfig(ExternalToolConfig):
    """Configuration for Elasticsearch MCP Server"""
    
    # Tool identification
    tool_name: str = "elasticsearch_mcp_server"
    
    # MCP server configuration
    mcp_host: str = "0.0.0.0"
    mcp_port: int = 9202
    max_connections: int = 10
    connection_timeout: float = 30.0
    request_timeout: float = 60.0
    
    # Elasticsearch connection
    elasticsearch_host: str = "localhost"
    elasticsearch_port: int = 9200
    elasticsearch_scheme: str = "http"
    elasticsearch_username: Optional[str] = None
    elasticsearch_password: Optional[str] = None
    
    # Index configuration
    default_index_settings: Dict[str, Any] = field(default_factory=lambda: {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "refresh_interval": "1s"
    })
    
    # Index templates for viral analysis
    index_templates: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "viral_proteins": {
            "patterns": ["viral_proteins*"],
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0
            },
            "mappings": {
                "properties": {
                    "protein_id": {"type": "keyword"},
                    "sequence": {"type": "text", "analyzer": "standard"},
                    "organism": {"type": "keyword"},
                    "function": {"type": "text", "analyzer": "standard"},
                    "embedding": {"type": "dense_vector", "dims": 384},
                    "metadata": {"type": "object", "enabled": True}
                }
            }
        }
    })
    
    # Docker integration (disabled when running in Docker)
    docker_service_name: str = "nanobrain-elasticsearch"
    auto_start_docker: bool = False  # Default to False to avoid dependency issues

    # === CSV PROCESSING CONFIGURATION ===

    # Core CSV Processing
    csv_processing_enabled: bool = True
    csv_max_file_size_mb: int = 500
    csv_max_rows_per_batch: int = 10000
    csv_processing_timeout: int = 1800  # 30 minutes for large files

    # File Format Detection
    csv_delimiter_auto_detect: bool = True
    csv_supported_delimiters: List[str] = field(default_factory=lambda: [",", ";", "\t", "|", "^"])
    csv_encoding_auto_detect: bool = True
    csv_supported_encodings: List[str] = field(default_factory=lambda: ["utf-8", "latin-1", "cp1252", "utf-16"])
    csv_quote_char_detection: bool = True
    csv_escape_char_detection: bool = True

    # Data Validation
    csv_validation_enabled: bool = True
    csv_max_field_length: int = 10000
    csv_allow_empty_fields: bool = True
    csv_strict_mode: bool = False

    # Performance Optimization
    csv_parallel_processing: bool = True
    csv_chunk_size: int = 1000
    csv_memory_limit_mb: int = 2048
    csv_cache_enabled: bool = True
    csv_cache_ttl: int = 3600

    # Fuzzy Search Configuration
    csv_fuzzy_threshold: float = 0.7
    csv_max_search_results: int = 100
    csv_confidence_scoring: bool = True
    csv_phonetic_matching: bool = True

    # Algorithm Selection
    csv_primary_algorithm: str = "elasticsearch"
    csv_fallback_algorithm: str = "fuzzywuzzy"
    csv_hybrid_mode: bool = True

    # Field-specific Settings
    csv_field_weights: Dict[str, float] = field(default_factory=lambda: {
        "default": 1.0,
        "name_fields": 2.0,
        "description_fields": 1.5,
        "id_fields": 3.0
    })

    # Result Enhancement
    csv_include_highlights: bool = True
    csv_include_suggestions: bool = True
    csv_include_corrections: bool = True
    csv_max_suggestions: int = 5

    # Tool card (mandatory) - updated with CSV capabilities
    tool_card: Dict[str, Any] = field(default_factory=lambda: {
        "name": "elasticsearch_mcp_server",
        "description": "Enhanced Elasticsearch MCP server with CSV fuzzy search capabilities",
        "version": "1.1.0",
        "category": "search_analytics",
        "capabilities": ["search", "indexing", "analytics", "mcp", "csv_processing", "fuzzy_search"]
    })


class ElasticsearchMCPError(MCPError):
    """Elasticsearch MCP specific errors"""
    pass


class ElasticsearchMCPServer(ExternalTool):
    """
    Elasticsearch MCP Server providing search and indexing capabilities
    via the Model Context Protocol.
    """
    
    @classmethod
    def from_config(cls, config: Union[ElasticsearchMCPConfig, Dict], **kwargs) -> 'ElasticsearchMCPServer':
        """Mandatory from_config implementation"""
        logger = get_logger(f"{cls.__name__}.from_config")
        logger.info(f"Creating {cls.__name__} from configuration")
        
        # Convert to ElasticsearchMCPConfig if needed
        if isinstance(config, dict):
            config = ElasticsearchMCPConfig(**config)
        elif not isinstance(config, ElasticsearchMCPConfig):
            # Convert other config types
            config_dict = config.model_dump() if hasattr(config, 'model_dump') else {}
            config = ElasticsearchMCPConfig(**config_dict)
        
        # Validate tool card
        if not hasattr(config, 'tool_card') or not config.tool_card:
            raise ValueError(f"Missing mandatory 'tool_card' in configuration for {cls.__name__}")
        
        instance = cls(config, **kwargs)
        instance._tool_card_data = config.tool_card
        
        logger.info(f"Successfully created {cls.__name__} with MCP server capabilities")
        return instance
    
    def __init__(self, config: ElasticsearchMCPConfig, **kwargs):
        """Initialize Elasticsearch MCP Server"""
        super().__init__(config, **kwargs)
        
        self.config = config
        self.logger = get_logger(f"elasticsearch_mcp_server")
        
        # Check dependencies
        if not ELASTICSEARCH_AVAILABLE:
            raise ElasticsearchMCPError(
                "Elasticsearch Python client not available. Install with: pip install elasticsearch"
            )
        
        if not AIOHTTP_AVAILABLE:
            raise ElasticsearchMCPError(
                "aiohttp not available. Install with: pip install aiohttp"
            )
        
        # Initialize components
        self.elasticsearch_client: Optional[AsyncElasticsearch] = None
        self.mcp_app: Optional[web.Application] = None
        self.mcp_runner: Optional[web.AppRunner] = None
        self.docker_manager: Optional[Any] = None  # Use Any to avoid import issues
        
        # Server state
        self.is_running = False
        self.active_connections = 0
        
        # Tool registry
        self.mcp_tools = {
            # Existing Elasticsearch tools
            "index_document": self._index_document,
            "bulk_index": self._bulk_index,
            "search": self._search,
            "semantic_search": self._semantic_search,
            "get_document": self._get_document,
            "delete_document": self._delete_document,
            "create_index": self._create_index,
            "delete_index": self._delete_index,
            "cluster_health": self._cluster_health,

            # === CSV FILE PROCESSING ===
            "import_csv_file": self._import_csv_file,
            "analyze_csv_structure": self._analyze_csv_structure,
            "validate_csv_format": self._validate_csv_format,
            "preview_csv_data": self._preview_csv_data,

            # === CSV INDEX MANAGEMENT ===
            "create_csv_index": self._create_csv_index,
            "update_csv_mapping": self._update_csv_mapping,
            "get_csv_index_info": self._get_csv_index_info,
            "delete_csv_index": self._delete_csv_index,

            # === CSV SEARCH OPERATIONS ===
            "fuzzy_search_csv": self._fuzzy_search_csv,
            "exact_search_csv": self._exact_search_csv,
            "range_search_csv": self._range_search_csv,
            "aggregate_csv_data": self._aggregate_csv_data,

            # === CSV UTILITIES ===
            "get_csv_field_suggestions": self._get_csv_field_suggestions,
            "detect_csv_data_types": self._detect_csv_data_types,
            "generate_csv_mapping": self._generate_csv_mapping
        }
    
    async def initialize_tool(self):
        """Initialize Elasticsearch MCP Server"""
        self.logger.info("🔄 Initializing Elasticsearch MCP Server...")
        
        try:
            # Initialize Docker manager if auto-start is enabled and available
            if self.config.auto_start_docker and DOCKER_AVAILABLE:
                await self._initialize_docker_service()
            elif self.config.auto_start_docker and not DOCKER_AVAILABLE:
                self.logger.warning("⚠️ Docker auto-start requested but Docker components not available")
            
            # Initialize Elasticsearch client
            await self._initialize_elasticsearch_client()
            
            # Set up index templates
            await self._setup_index_templates()
            
            # Initialize MCP web server
            await self._initialize_mcp_server()
            
            self.logger.info("✅ Elasticsearch MCP Server initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Elasticsearch MCP Server: {e}")
            raise
    
    async def _initialize_docker_service(self):
        """Initialize Docker service for Elasticsearch (only when Docker components available)"""
        if not DOCKER_AVAILABLE:
            self.logger.warning("⚠️ Docker components not available, skipping Docker initialization")
            return
            
        try:
            # Import Docker components dynamically to avoid import issues
            from nanobrain.library.infrastructure.docker import (
                DockerManager, 
                DockerComponentConfig,
                ContainerConfig
            )
            
            # Create Docker manager
            docker_config = DockerComponentConfig(
                component_name="elasticsearch_docker_manager"
            )
            self.docker_manager = DockerManager.from_config(docker_config)
            
            # Check if Elasticsearch container is running
            container_status = await self.docker_manager.get_container_status(
                self.config.docker_service_name
            )
            
            if not container_status or container_status.get("status") != "running":
                self.logger.info("🐳 Starting Elasticsearch Docker container...")
                
                # Load Elasticsearch service configuration
                elasticsearch_config_path = Path(__file__).parent.parent.parent / \
                    "config/defaults/docker/ElasticsearchService.yml"
                
                if elasticsearch_config_path.exists():
                    import yaml
                    with open(elasticsearch_config_path, 'r') as f:
                        service_config = yaml.safe_load(f)
                    
                    # Create container configuration
                    container_config = ContainerConfig.from_config(service_config)
                    
                    # Start the container
                    await self.docker_manager.create_and_start_container(container_config)
                    
                    # Wait for container to be ready
                    await self._wait_for_elasticsearch_ready()
                else:
                    self.logger.warning("⚠️ Elasticsearch service configuration not found")
            else:
                self.logger.info("✅ Elasticsearch Docker container already running")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Docker service: {e}")
            # Don't raise exception when running in Docker - just log and continue
            if os.getenv("NANOBRAIN_ENV") != "docker":
                raise ElasticsearchMCPError(f"Docker service initialization failed: {e}")
            else:
                self.logger.warning("⚠️ Running in Docker environment, continuing without Docker management")
    
    async def _wait_for_elasticsearch_ready(self, max_wait: int = 60):
        """Wait for Elasticsearch to be ready"""
        self.logger.info("⏳ Waiting for Elasticsearch to be ready...")
        
        start_time = time.time()
        while time.time() - start_time < max_wait:
            try:
                # Try to connect to Elasticsearch
                es_url = f"{self.config.elasticsearch_scheme}://{self.config.elasticsearch_host}:{self.config.elasticsearch_port}"
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{es_url}/_cluster/health") as response:
                        if response.status == 200:
                            health_data = await response.json()
                            if health_data.get("status") in ["green", "yellow"]:
                                self.logger.info("✅ Elasticsearch is ready")
                                return
                
            except Exception:
                pass
            
            await asyncio.sleep(2)
        
        raise ElasticsearchMCPError("Elasticsearch failed to become ready within timeout")
    
    async def _initialize_elasticsearch_client(self):
        """Initialize Elasticsearch client"""
        try:
            # Wait for Elasticsearch to be ready first
            await self._wait_for_elasticsearch_ready()
            
            es_url = f"{self.config.elasticsearch_scheme}://{self.config.elasticsearch_host}:{self.config.elasticsearch_port}"
            
            # Create client configuration
            client_config = {
                "hosts": [es_url],
                "timeout": self.config.request_timeout,
                "max_retries": 3,
                "retry_on_timeout": True
            }
            
            # Add authentication if configured
            if self.config.elasticsearch_username and self.config.elasticsearch_password:
                client_config["basic_auth"] = (
                    self.config.elasticsearch_username,
                    self.config.elasticsearch_password
                )
            
            self.elasticsearch_client = AsyncElasticsearch(**client_config)
            
            # Test connection
            health = await self.elasticsearch_client.cluster.health()
            self.logger.info(f"✅ Connected to Elasticsearch cluster: {health['cluster_name']}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to connect to Elasticsearch: {e}")
            raise ElasticsearchMCPError(f"Elasticsearch connection failed: {e}")
    
    async def _setup_index_templates(self):
        """Set up index templates for viral analysis"""
        try:
            for template_name, template_config in self.config.index_templates.items():
                await self.elasticsearch_client.indices.put_index_template(
                    name=template_name,
                    index_patterns=template_config["patterns"],
                    template={
                        "settings": template_config["settings"],
                        "mappings": template_config["mappings"]
                    }
                )
                self.logger.info(f"✅ Created index template: {template_name}")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to set up index templates: {e}")
            raise ElasticsearchMCPError(f"Index template setup failed: {e}")
    
    async def _initialize_mcp_server(self):
        """Initialize MCP web server"""
        try:
            self.mcp_app = web.Application()
            
            # Add routes
            self.mcp_app.router.add_post('/mcp/tools/{tool_name}', self._handle_mcp_tool_call)
            self.mcp_app.router.add_get('/mcp/tools', self._handle_list_tools)
            self.mcp_app.router.add_get('/health', self._handle_health_check)
            
            # Add CORS support
            self.mcp_app.middlewares.append(self._cors_middleware)
            
            self.logger.info(f"✅ MCP server configured on {self.config.mcp_host}:{self.config.mcp_port}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize MCP server: {e}")
            raise ElasticsearchMCPError(f"MCP server initialization failed: {e}")
    
    async def start(self):
        """Start the Elasticsearch MCP Server"""
        if self.is_running:
            self.logger.warning("⚠️ MCP server is already running")
            return
        
        try:
            # Ensure tool is initialized
            await self.initialize_tool()
            
            # Start MCP web server
            self.mcp_runner = web.AppRunner(self.mcp_app)
            await self.mcp_runner.setup()
            
            site = web.TCPSite(
                self.mcp_runner,
                self.config.mcp_host,
                self.config.mcp_port
            )
            await site.start()
            
            self.is_running = True
            
            self.logger.info(f"🚀 Elasticsearch MCP Server started")
            self.logger.info(f"   📍 MCP Endpoint: http://{self.config.mcp_host}:{self.config.mcp_port}")
            self.logger.info(f"   🔗 Elasticsearch: {self.config.elasticsearch_scheme}://{self.config.elasticsearch_host}:{self.config.elasticsearch_port}")
            self.logger.info(f"   🛠️  Available tools: {len(self.mcp_tools)}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start MCP server: {e}")
            raise ElasticsearchMCPError(f"MCP server start failed: {e}")
    
    async def stop(self):
        """Stop the Elasticsearch MCP Server"""
        if not self.is_running:
            return
        
        try:
            # Stop MCP server
            if self.mcp_runner:
                await self.mcp_runner.cleanup()
                self.mcp_runner = None
            
            # Close Elasticsearch client
            if self.elasticsearch_client:
                await self.elasticsearch_client.close()
                self.elasticsearch_client = None
            
            self.is_running = False
            self.logger.info("✅ Elasticsearch MCP Server stopped")
            
        except Exception as e:
            self.logger.error(f"❌ Error stopping MCP server: {e}")
    
    # MCP Request Handlers
    
    async def _handle_mcp_tool_call(self, request: web.Request) -> web.Response:
        """Handle MCP tool call requests"""
        try:
            tool_name = request.match_info['tool_name']
            
            if tool_name not in self.mcp_tools:
                return web.json_response(
                    {"error": f"Tool '{tool_name}' not found"},
                    status=404
                )
            
            # Parse request body
            try:
                parameters = await request.json()
            except json.JSONDecodeError:
                return web.json_response(
                    {"error": "Invalid JSON in request body"},
                    status=400
                )
            
            # Execute tool
            result = await self.mcp_tools[tool_name](parameters)
            
            return web.json_response({
                "success": True,
                "result": result,
                "tool": tool_name,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
        except Exception as e:
            self.logger.error(f"❌ MCP tool call failed: {e}")
            return web.json_response(
                {"error": str(e), "success": False},
                status=500
            )
    
    async def _handle_list_tools(self, request: web.Request) -> web.Response:
        """Handle MCP tools list requests"""
        try:
            tools = []
            for tool_name in self.mcp_tools.keys():
                tools.append({
                    "name": tool_name,
                    "description": self._get_tool_description(tool_name),
                    "parameters": self._get_tool_parameters(tool_name)
                })
            
            return web.json_response({
                "tools": tools,
                "count": len(tools),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
        except Exception as e:
            self.logger.error(f"❌ Failed to list tools: {e}")
            return web.json_response(
                {"error": str(e)},
                status=500
            )
    
    async def _handle_health_check(self, request: web.Request) -> web.Response:
        """Handle health check requests"""
        try:
            # Check Elasticsearch connection
            if self.elasticsearch_client:
                health = await self.elasticsearch_client.cluster.health()
                es_status = health.get("status", "unknown")
            else:
                es_status = "disconnected"
            
            return web.json_response({
                "status": "healthy" if self.is_running else "unhealthy",
                "elasticsearch_status": es_status,
                "active_connections": self.active_connections,
                "available_tools": len(self.mcp_tools),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
        except Exception as e:
            return web.json_response(
                {"status": "unhealthy", "error": str(e)},
                status=500
            )
    
    @web.middleware
    async def _cors_middleware(self, request: web.Request, handler):
        """CORS middleware for MCP requests"""
        response = await handler(request)
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        return response
    
    # Elasticsearch Tool Implementations
    
    async def _index_document(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Index a single document"""
        index = parameters.get("index")
        document = parameters.get("document")
        doc_id = parameters.get("id")
        
        if not index or not document:
            raise ElasticsearchMCPError("Missing required parameters: index, document")
        
        try:
            if doc_id:
                result = await self.elasticsearch_client.index(
                    index=index,
                    id=doc_id,
                    document=document
                )
            else:
                result = await self.elasticsearch_client.index(
                    index=index,
                    document=document
                )
            
            return {
                "indexed": True,
                "index": index,
                "id": result["_id"],
                "version": result["_version"],
                "result": result["result"]
            }
            
        except Exception as e:
            raise ElasticsearchMCPError(f"Document indexing failed: {e}")
    
    async def _bulk_index(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Bulk index multiple documents"""
        index = parameters.get("index")
        documents = parameters.get("documents")
        
        if not index or not documents:
            raise ElasticsearchMCPError("Missing required parameters: index, documents")
        
        if not isinstance(documents, list):
            raise ElasticsearchMCPError("Documents must be a list")
        
        try:
            # Prepare bulk operations
            operations = []
            for doc in documents:
                operation = {"index": {"_index": index}}
                if "id" in doc:
                    operation["index"]["_id"] = doc.pop("id")
                
                operations.append(operation)
                operations.append(doc)
            
            result = await self.elasticsearch_client.bulk(
                operations=operations,
                refresh=True
            )
            
            # Parse results
            indexed_count = 0
            errors = []
            
            for item in result["items"]:
                if "index" in item:
                    if item["index"].get("status") in [200, 201]:
                        indexed_count += 1
                    else:
                        errors.append(item["index"].get("error", "Unknown error"))
            
            return {
                "bulk_indexed": True,
                "index": index,
                "total_documents": len(documents),
                "indexed_count": indexed_count,
                "errors_count": len(errors),
                "errors": errors[:10] if errors else []  # Limit error details
            }
            
        except Exception as e:
            raise ElasticsearchMCPError(f"Bulk indexing failed: {e}")
    
    async def _search(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Search documents using Elasticsearch query DSL"""
        index = parameters.get("index")
        query = parameters.get("query")
        size = parameters.get("size", 10)
        from_offset = parameters.get("from", 0)
        
        if not index:
            raise ElasticsearchMCPError("Missing required parameter: index")
        
        try:
            search_body = {
                "query": query or {"match_all": {}},
                "size": size,
                "from": from_offset
            }
            
            result = await self.elasticsearch_client.search(
                index=index,
                body=search_body
            )
            
            return {
                "search_completed": True,
                "index": index,
                "total_hits": result["hits"]["total"]["value"],
                "returned_hits": len(result["hits"]["hits"]),
                "max_score": result["hits"]["max_score"],
                "hits": [
                    {
                        "id": hit["_id"],
                        "score": hit["_score"],
                        "source": hit["_source"]
                    }
                    for hit in result["hits"]["hits"]
                ]
            }
            
        except Exception as e:
            raise ElasticsearchMCPError(f"Search failed: {e}")
    
    async def _semantic_search(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform semantic search using vector embeddings"""
        index = parameters.get("index")
        query_vector = parameters.get("query_vector")
        embedding_field = parameters.get("embedding_field", "embedding")
        size = parameters.get("size", 10)
        
        if not index or not query_vector:
            raise ElasticsearchMCPError("Missing required parameters: index, query_vector")
        
        try:
            search_body = {
                "knn": {
                    "field": embedding_field,
                    "query_vector": query_vector,
                    "k": size,
                    "num_candidates": size * 2
                },
                "size": size
            }
            
            result = await self.elasticsearch_client.search(
                index=index,
                body=search_body
            )
            
            return {
                "semantic_search_completed": True,
                "index": index,
                "embedding_field": embedding_field,
                "total_hits": result["hits"]["total"]["value"],
                "returned_hits": len(result["hits"]["hits"]),
                "hits": [
                    {
                        "id": hit["_id"],
                        "score": hit["_score"],
                        "source": hit["_source"]
                    }
                    for hit in result["hits"]["hits"]
                ]
            }
            
        except Exception as e:
            raise ElasticsearchMCPError(f"Semantic search failed: {e}")
    
    async def _get_document(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve a document by ID"""
        index = parameters.get("index")
        doc_id = parameters.get("id")
        
        if not index or not doc_id:
            raise ElasticsearchMCPError("Missing required parameters: index, id")
        
        try:
            result = await self.elasticsearch_client.get(
                index=index,
                id=doc_id
            )
            
            return {
                "document_found": True,
                "index": index,
                "id": doc_id,
                "version": result["_version"],
                "source": result["_source"]
            }
            
        except NotFoundError:
            return {
                "document_found": False,
                "index": index,
                "id": doc_id,
                "reason": "Document not found"
            }
        except Exception as e:
            raise ElasticsearchMCPError(f"Document retrieval failed: {e}")
    
    async def _delete_document(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Delete a document by ID"""
        index = parameters.get("index")
        doc_id = parameters.get("id")
        
        if not index or not doc_id:
            raise ElasticsearchMCPError("Missing required parameters: index, id")
        
        try:
            result = await self.elasticsearch_client.delete(
                index=index,
                id=doc_id
            )
            
            return {
                "document_deleted": True,
                "index": index,
                "id": doc_id,
                "result": result["result"]
            }
            
        except NotFoundError:
            return {
                "document_deleted": False,
                "index": index,
                "id": doc_id,
                "reason": "Document not found"
            }
        except Exception as e:
            raise ElasticsearchMCPError(f"Document deletion failed: {e}")
    
    async def _create_index(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new index with mappings"""
        index = parameters.get("index")
        mappings = parameters.get("mappings", {})
        settings = parameters.get("settings", self.config.default_index_settings)
        
        if not index:
            raise ElasticsearchMCPError("Missing required parameter: index")
        
        try:
            body = {"settings": settings}
            if mappings:
                body["mappings"] = mappings
            
            result = await self.elasticsearch_client.indices.create(
                index=index,
                body=body
            )
            
            return {
                "index_created": True,
                "index": index,
                "acknowledged": result["acknowledged"],
                "shards_acknowledged": result["shards_acknowledged"]
            }
            
        except Exception as e:
            raise ElasticsearchMCPError(f"Index creation failed: {e}")
    
    async def _delete_index(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Delete an index"""
        index = parameters.get("index")
        
        if not index:
            raise ElasticsearchMCPError("Missing required parameter: index")
        
        try:
            result = await self.elasticsearch_client.indices.delete(index=index)
            
            return {
                "index_deleted": True,
                "index": index,
                "acknowledged": result["acknowledged"]
            }
            
        except NotFoundError:
            return {
                "index_deleted": False,
                "index": index,
                "reason": "Index not found"
            }
        except Exception as e:
            raise ElasticsearchMCPError(f"Index deletion failed: {e}")
    
    async def _cluster_health(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Get cluster health information"""
        try:
            health = await self.elasticsearch_client.cluster.health()
            
            return {
                "cluster_name": health["cluster_name"],
                "status": health["status"],
                "number_of_nodes": health["number_of_nodes"],
                "number_of_data_nodes": health["number_of_data_nodes"],
                "active_primary_shards": health["active_primary_shards"],
                "active_shards": health["active_shards"],
                "relocating_shards": health["relocating_shards"],
                "initializing_shards": health["initializing_shards"],
                "unassigned_shards": health["unassigned_shards"]
            }
            
        except Exception as e:
            raise ElasticsearchMCPError(f"Cluster health check failed: {e}")
    
    # Helper methods
    
    def _get_tool_description(self, tool_name: str) -> str:
        """Get description for a tool"""
        descriptions = {
            "index_document": "Index a single document in Elasticsearch",
            "bulk_index": "Bulk index multiple documents",
            "search": "Search documents using Elasticsearch query DSL",
            "semantic_search": "Perform semantic search using vector embeddings",
            "get_document": "Retrieve a document by ID",
            "delete_document": "Delete a document by ID",
            "create_index": "Create a new index with mappings",
            "delete_index": "Delete an index",
            "cluster_health": "Get cluster health information"
        }
        return descriptions.get(tool_name, "No description available")
    
    def _get_tool_parameters(self, tool_name: str) -> Dict[str, Any]:
        """Get parameter schema for a tool"""
        schemas = {
            "index_document": {
                "type": "object",
                "required": ["index", "document"],
                "properties": {
                    "index": {"type": "string"},
                    "document": {"type": "object"},
                    "id": {"type": "string"}
                }
            },
            "bulk_index": {
                "type": "object",
                "required": ["index", "documents"],
                "properties": {
                    "index": {"type": "string"},
                    "documents": {"type": "array", "items": {"type": "object"}}
                }
            },
            "search": {
                "type": "object",
                "required": ["index"],
                "properties": {
                    "index": {"type": "string"},
                    "query": {"type": "object"},
                    "size": {"type": "integer", "default": 10},
                    "from": {"type": "integer", "default": 0}
                }
            },
            "semantic_search": {
                "type": "object",
                "required": ["index", "query_vector"],
                "properties": {
                    "index": {"type": "string"},
                    "query_vector": {"type": "array", "items": {"type": "number"}},
                    "embedding_field": {"type": "string", "default": "embedding"},
                    "size": {"type": "integer", "default": 10}
                }
            }
        }
        return schemas.get(tool_name, {"type": "object"})

    async def verify_installation(self) -> bool:
        """Verify Elasticsearch MCP server installation"""
        try:
            return (ELASTICSEARCH_AVAILABLE and 
                    AIOHTTP_AVAILABLE and 
                    self.elasticsearch_client is not None)
        except Exception:
            return False
    
    # Required abstract methods from ExternalTool base class
    
    async def execute_command(self, command: List[str], **kwargs) -> Any:
        """Execute MCP server command - not applicable for MCP server"""
        raise NotImplementedError("MCP server doesn't execute external commands")
    
    async def parse_output(self, raw_output: str, output_type: str = "json") -> Any:
        """Parse MCP server output - not applicable for MCP server"""
        return raw_output
    
    async def _find_executable_in_path(self) -> Optional[str]:
        """Find MCP server executable - not applicable for MCP server"""
        return None
    
    async def _check_tool_in_environment(self, env_path: str, env_name: str) -> bool:
        """Check if MCP server is in environment - not applicable"""
        return False
    
    async def _check_tool_in_directory(self, directory: str) -> bool:
        """Check if MCP server is in directory - not applicable"""
        return False
    
    async def _build_tool_in_environment(self, source_dir: str) -> bool:
        """Build MCP server in environment - not applicable"""
        return False

    # ===================================================================
    # CSV PROCESSING METHODS
    # ===================================================================

    async def _import_csv_file(self, file_path: str, index_name: str, **kwargs) -> Dict[str, Any]:
        """Import CSV file and index data in Elasticsearch"""
        try:
            if not self.config.csv_processing_enabled:
                raise ElasticsearchMCPError("CSV processing is disabled")

            self.logger.info(f"🔄 Importing CSV file: {file_path}")

            # Validate file size
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            if file_size_mb > self.config.csv_max_file_size_mb:
                raise ElasticsearchMCPError(f"File size {file_size_mb:.1f}MB exceeds limit {self.config.csv_max_file_size_mb}MB")

            # Analyze CSV structure first
            structure_result = await self._analyze_csv_structure(file_path)
            if not structure_result.get("success", False):
                raise ElasticsearchMCPError(f"Failed to analyze CSV structure: {structure_result.get('error')}")

            # Create index with appropriate mapping
            mapping_result = await self._create_csv_index(index_name, structure_result["structure"])
            if not mapping_result.get("success", False):
                self.logger.warning(f"Index creation warning: {mapping_result.get('message', 'Unknown issue')}")

            # Import data in batches
            total_rows = 0
            batch_size = self.config.csv_chunk_size

            if PANDAS_AVAILABLE:
                # Use pandas for efficient processing
                for chunk in pd.read_csv(file_path, chunksize=batch_size,
                                       delimiter=structure_result["structure"]["delimiter"],
                                       encoding=structure_result["structure"]["encoding"]):
                    documents = chunk.to_dict('records')
                    await self._bulk_index_documents(index_name, documents)
                    total_rows += len(documents)

                    if total_rows >= self.config.csv_max_rows_per_batch:
                        break
            else:
                # Fallback to manual CSV processing
                with open(file_path, 'r', encoding=structure_result["structure"]["encoding"]) as f:
                    reader = csv.DictReader(f, delimiter=structure_result["structure"]["delimiter"])
                    batch = []

                    for row in reader:
                        batch.append(row)
                        total_rows += 1

                        if len(batch) >= batch_size:
                            await self._bulk_index_documents(index_name, batch)
                            batch = []

                        if total_rows >= self.config.csv_max_rows_per_batch:
                            break

                    # Index remaining documents
                    if batch:
                        await self._bulk_index_documents(index_name, batch)

            self.logger.info(f"✅ Successfully imported {total_rows} rows from CSV file")

            return {
                "success": True,
                "index_name": index_name,
                "rows_imported": total_rows,
                "file_size_mb": file_size_mb,
                "structure": structure_result["structure"]
            }

        except Exception as e:
            self.logger.error(f"❌ CSV import failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "file_path": file_path
            }

    async def _analyze_csv_structure(self, file_path: str) -> Dict[str, Any]:
        """Analyze CSV file structure and detect format"""
        try:
            self.logger.info(f"🔍 Analyzing CSV structure: {file_path}")

            # Read sample of file for analysis
            with open(file_path, 'rb') as f:
                sample = f.read(8192)  # Read first 8KB

            # Detect encoding
            encoding = "utf-8"  # Default
            if CHARDET_AVAILABLE:
                detected = chardet.detect(sample)
                if detected['confidence'] > 0.7:
                    encoding = detected['encoding']

            # Convert to text for analysis
            sample_text = sample.decode(encoding, errors='ignore')
            lines = sample_text.split('\n')[:10]  # First 10 lines

            # Detect delimiter
            delimiter = ","  # Default
            if self.config.csv_delimiter_auto_detect:
                delimiter_counts = {}
                for delim in self.config.csv_supported_delimiters:
                    count = sum(line.count(delim) for line in lines[:5])
                    delimiter_counts[delim] = count

                # Choose delimiter with highest consistent count
                if delimiter_counts:
                    delimiter = max(delimiter_counts, key=delimiter_counts.get)

            # Analyze first few rows to determine structure
            with open(file_path, 'r', encoding=encoding) as f:
                reader = csv.reader(f, delimiter=delimiter)
                rows = [next(reader, None) for _ in range(5)]
                rows = [row for row in rows if row]  # Remove empty rows

            if not rows:
                raise ValueError("No valid rows found in CSV file")

            headers = rows[0]
            data_rows = rows[1:] if len(rows) > 1 else []

            # Detect column types
            column_types = {}
            for i, header in enumerate(headers):
                column_types[header] = self._detect_column_type([row[i] if i < len(row) else "" for row in data_rows])

            structure = {
                "delimiter": delimiter,
                "encoding": encoding,
                "headers": headers,
                "column_count": len(headers),
                "estimated_rows": len(lines) - 1,  # Rough estimate
                "column_types": column_types,
                "sample_data": data_rows[:3]  # First 3 data rows
            }

            self.logger.info(f"✅ CSV structure analyzed: {len(headers)} columns, delimiter='{delimiter}', encoding={encoding}")

            return {
                "success": True,
                "structure": structure,
                "file_path": file_path
            }

        except Exception as e:
            self.logger.error(f"❌ CSV structure analysis failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "file_path": file_path
            }

    async def _validate_csv_format(self, file_path: str) -> Dict[str, Any]:
        """Validate CSV file format and data quality"""
        try:
            self.logger.info(f"🔍 Validating CSV format: {file_path}")

            validation_results = {
                "file_exists": os.path.exists(file_path),
                "file_readable": False,
                "valid_format": False,
                "encoding_detected": None,
                "delimiter_detected": None,
                "issues": [],
                "warnings": []
            }

            if not validation_results["file_exists"]:
                validation_results["issues"].append("File does not exist")
                return {"success": False, "validation": validation_results}

            # Check file readability
            try:
                with open(file_path, 'r') as f:
                    f.read(1)
                validation_results["file_readable"] = True
            except Exception as e:
                validation_results["issues"].append(f"File not readable: {e}")
                return {"success": False, "validation": validation_results}

            # Analyze structure (reuse existing method)
            structure_result = await self._analyze_csv_structure(file_path)
            if structure_result.get("success"):
                structure = structure_result["structure"]
                validation_results["encoding_detected"] = structure["encoding"]
                validation_results["delimiter_detected"] = structure["delimiter"]
                validation_results["valid_format"] = True

                # Additional validation checks
                if structure["column_count"] == 0:
                    validation_results["issues"].append("No columns detected")
                elif structure["column_count"] > 1000:
                    validation_results["warnings"].append(f"Large number of columns: {structure['column_count']}")

                # Check for duplicate headers
                headers = structure["headers"]
                if len(headers) != len(set(headers)):
                    validation_results["warnings"].append("Duplicate column headers detected")

                # Check for empty headers
                empty_headers = [i for i, h in enumerate(headers) if not h.strip()]
                if empty_headers:
                    validation_results["warnings"].append(f"Empty headers at positions: {empty_headers}")

            else:
                validation_results["issues"].append(structure_result.get("error", "Structure analysis failed"))

            success = len(validation_results["issues"]) == 0

            self.logger.info(f"✅ CSV validation complete: {'PASSED' if success else 'FAILED'}")

            return {
                "success": success,
                "validation": validation_results,
                "file_path": file_path
            }

        except Exception as e:
            self.logger.error(f"❌ CSV validation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "file_path": file_path
            }

    async def _preview_csv_data(self, file_path: str, max_rows: int = 10) -> Dict[str, Any]:
        """Preview CSV file data"""
        try:
            self.logger.info(f"👀 Previewing CSV data: {file_path}")

            # Get structure first
            structure_result = await self._analyze_csv_structure(file_path)
            if not structure_result.get("success"):
                raise ValueError(f"Failed to analyze CSV structure: {structure_result.get('error')}")

            structure = structure_result["structure"]

            # Read preview data
            preview_data = []
            with open(file_path, 'r', encoding=structure["encoding"]) as f:
                reader = csv.DictReader(f, delimiter=structure["delimiter"])
                for i, row in enumerate(reader):
                    if i >= max_rows:
                        break
                    preview_data.append(row)

            return {
                "success": True,
                "preview": {
                    "headers": structure["headers"],
                    "data": preview_data,
                    "total_columns": structure["column_count"],
                    "preview_rows": len(preview_data),
                    "column_types": structure["column_types"]
                },
                "file_path": file_path
            }

        except Exception as e:
            self.logger.error(f"❌ CSV preview failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "file_path": file_path
            }

    def _detect_column_type(self, values: List[str]) -> str:
        """Detect the data type of a column based on sample values"""
        if not values:
            return "text"

        # Remove empty values for analysis
        non_empty_values = [v.strip() for v in values if v.strip()]
        if not non_empty_values:
            return "text"

        # Check for numeric types
        numeric_count = 0
        integer_count = 0
        float_count = 0
        date_count = 0

        for value in non_empty_values:
            # Check integer
            try:
                int(value)
                numeric_count += 1
                integer_count += 1
                continue
            except ValueError:
                pass

            # Check float
            try:
                float(value)
                numeric_count += 1
                float_count += 1
                continue
            except ValueError:
                pass

            # Check date patterns (basic)
            if any(sep in value for sep in ['-', '/', '.']):
                parts = value.replace('-', '/').replace('.', '/').split('/')
                if len(parts) >= 2 and all(part.isdigit() for part in parts):
                    date_count += 1

        total_values = len(non_empty_values)

        # Determine type based on majority
        if integer_count / total_values > 0.8:
            return "integer"
        elif numeric_count / total_values > 0.8:
            return "float"
        elif date_count / total_values > 0.6:
            return "date"
        else:
            return "text"

    async def _bulk_index_documents(self, index_name: str, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Bulk index documents into Elasticsearch"""
        try:
            if not documents:
                return {"success": True, "indexed": 0}

            # Prepare bulk operations
            operations = []
            for doc in documents:
                # Add document ID if not present
                doc_id = doc.get('id') or str(uuid.uuid4())

                operations.append({
                    "index": {
                        "_index": index_name,
                        "_id": doc_id
                    }
                })
                operations.append(doc)

            # Execute bulk operation
            response = await self.client.bulk(
                operations=operations,
                refresh=True
            )

            # Check for errors
            errors = []
            indexed_count = 0

            if response.get("errors"):
                for item in response.get("items", []):
                    if "index" in item and "error" in item["index"]:
                        errors.append(item["index"]["error"])
                    else:
                        indexed_count += 1
            else:
                indexed_count = len(documents)

            if errors:
                self.logger.warning(f"Bulk indexing completed with {len(errors)} errors")
                return {
                    "success": True,
                    "indexed": indexed_count,
                    "errors": errors[:10]  # Limit error reporting
                }

            return {
                "success": True,
                "indexed": indexed_count
            }

        except Exception as e:
            self.logger.error(f"❌ Bulk indexing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "indexed": 0
            }

    # ===================================================================
    # CSV INDEX MANAGEMENT METHODS
    # ===================================================================

    async def _create_csv_index(self, index_name: str, csv_structure: Dict[str, Any]) -> Dict[str, Any]:
        """Create Elasticsearch index optimized for CSV data"""
        try:
            self.logger.info(f"🔧 Creating CSV index: {index_name}")

            # Generate mapping based on CSV structure
            mapping = self._generate_csv_mapping(csv_structure)

            # Index settings optimized for CSV data
            settings = {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "analysis": {
                    "analyzer": {
                        "csv_text_analyzer": {
                            "type": "standard",
                            "stopwords": "_english_"
                        },
                        "csv_fuzzy_analyzer": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": ["lowercase", "asciifolding"]
                        }
                    }
                }
            }

            # Check if index already exists
            exists = await self.client.indices.exists(index=index_name)
            if exists:
                self.logger.warning(f"Index {index_name} already exists, updating mapping")
                # Update mapping for existing index
                await self.client.indices.put_mapping(
                    index=index_name,
                    body=mapping
                )
                return {
                    "success": True,
                    "action": "updated",
                    "index_name": index_name,
                    "mapping": mapping
                }

            # Create new index
            await self.client.indices.create(
                index=index_name,
                body={
                    "settings": settings,
                    "mappings": mapping
                }
            )

            self.logger.info(f"✅ CSV index created: {index_name}")

            return {
                "success": True,
                "action": "created",
                "index_name": index_name,
                "mapping": mapping,
                "settings": settings
            }

        except Exception as e:
            self.logger.error(f"❌ CSV index creation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "index_name": index_name
            }

    async def _update_csv_mapping(self, index_name: str, new_mapping: Dict[str, Any]) -> Dict[str, Any]:
        """Update mapping for existing CSV index"""
        try:
            self.logger.info(f"🔧 Updating CSV mapping: {index_name}")

            # Check if index exists
            exists = await self.client.indices.exists(index=index_name)
            if not exists:
                return {
                    "success": False,
                    "error": f"Index {index_name} does not exist"
                }

            # Update mapping
            await self.client.indices.put_mapping(
                index=index_name,
                body=new_mapping
            )

            self.logger.info(f"✅ CSV mapping updated: {index_name}")

            return {
                "success": True,
                "index_name": index_name,
                "mapping": new_mapping
            }

        except Exception as e:
            self.logger.error(f"❌ CSV mapping update failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "index_name": index_name
            }

    async def _get_csv_index_info(self, index_name: str) -> Dict[str, Any]:
        """Get information about CSV index"""
        try:
            self.logger.info(f"ℹ️ Getting CSV index info: {index_name}")

            # Check if index exists
            exists = await self.client.indices.exists(index=index_name)
            if not exists:
                return {
                    "success": False,
                    "error": f"Index {index_name} does not exist"
                }

            # Get index stats
            stats = await self.client.indices.stats(index=index_name)
            index_stats = stats["indices"][index_name]

            # Get mapping
            mapping_response = await self.client.indices.get_mapping(index=index_name)
            mapping = mapping_response[index_name]["mappings"]

            # Get settings
            settings_response = await self.client.indices.get_settings(index=index_name)
            settings = settings_response[index_name]["settings"]

            info = {
                "index_name": index_name,
                "document_count": index_stats["total"]["docs"]["count"],
                "store_size": index_stats["total"]["store"]["size_in_bytes"],
                "mapping": mapping,
                "settings": settings,
                "health": "green"  # Simplified health check
            }

            self.logger.info(f"✅ CSV index info retrieved: {index_name}")

            return {
                "success": True,
                "info": info
            }

        except Exception as e:
            self.logger.error(f"❌ Failed to get CSV index info: {e}")
            return {
                "success": False,
                "error": str(e),
                "index_name": index_name
            }

    async def _delete_csv_index(self, index_name: str) -> Dict[str, Any]:
        """Delete CSV index"""
        try:
            self.logger.info(f"🗑️ Deleting CSV index: {index_name}")

            # Check if index exists
            exists = await self.client.indices.exists(index=index_name)
            if not exists:
                return {
                    "success": False,
                    "error": f"Index {index_name} does not exist"
                }

            # Delete index
            await self.client.indices.delete(index=index_name)

            self.logger.info(f"✅ CSV index deleted: {index_name}")

            return {
                "success": True,
                "index_name": index_name,
                "action": "deleted"
            }

        except Exception as e:
            self.logger.error(f"❌ CSV index deletion failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "index_name": index_name
            }

    def _generate_csv_mapping(self, csv_structure: Dict[str, Any]) -> Dict[str, Any]:
        """Generate Elasticsearch mapping from CSV structure"""
        try:
            properties = {}
            column_types = csv_structure.get("column_types", {})
            headers = csv_structure.get("headers", [])

            for header in headers:
                column_type = column_types.get(header, "text")

                # Map CSV types to Elasticsearch types
                if column_type == "integer":
                    properties[header] = {
                        "type": "long"
                    }
                elif column_type == "float":
                    properties[header] = {
                        "type": "double"
                    }
                elif column_type == "boolean":
                    properties[header] = {
                        "type": "boolean"
                    }
                elif column_type == "date":
                    properties[header] = {
                        "type": "date",
                        "format": "yyyy-MM-dd||yyyy/MM/dd||dd-MM-yyyy||dd/MM/yyyy||strict_date_optional_time"
                    }
                else:  # text or unknown
                    properties[header] = {
                        "type": "text",
                        "analyzer": "csv_text_analyzer",
                        "fields": {
                            "keyword": {
                                "type": "keyword",
                                "ignore_above": 256
                            },
                            "fuzzy": {
                                "type": "text",
                                "analyzer": "csv_fuzzy_analyzer"
                            }
                        }
                    }

            # Add metadata fields
            properties["_csv_import_timestamp"] = {
                "type": "date"
            }
            properties["_csv_row_number"] = {
                "type": "long"
            }

            mapping = {
                "properties": properties
            }

            return mapping

        except Exception as e:
            self.logger.error(f"❌ Mapping generation failed: {e}")
            return {
                "properties": {
                    "_error": {
                        "type": "text"
                    }
                }
            }

    # ===================================================================
    # CSV FUZZY SEARCH METHODS
    # ===================================================================

    async def _fuzzy_search_csv(self, index_name: str, query: str,
                               fuzzy_threshold: float = None,
                               max_results: int = None,
                               search_fields: List[str] = None) -> Dict[str, Any]:
        """Execute fuzzy search on CSV data with confidence scoring"""
        try:
            # Use configuration defaults if not provided
            fuzzy_threshold = fuzzy_threshold or self.config.csv_fuzzy_threshold
            max_results = max_results or self.config.csv_max_search_results

            self.logger.info(f"🔍 Fuzzy searching '{query}' in index: {index_name}")

            # Initialize fuzzy matcher if not already done
            if not hasattr(self, '_fuzzy_matcher'):
                from .csv_fuzzy_matcher import CSVFuzzyMatcher
                matcher_config = {
                    'default_threshold': fuzzy_threshold,
                    'max_results': max_results,
                    'field_weights': self.config.csv_field_weights,
                    'enable_suggestions': self.config.csv_include_suggestions
                }
                self._fuzzy_matcher = CSVFuzzyMatcher(matcher_config)

            # Initialize search optimizer if not already done
            if not hasattr(self, '_search_optimizer'):
                from .csv_search_optimizer import CSVSearchOptimizer
                optimizer_config = {
                    'cache_enabled': self.config.csv_cache_enabled,
                    'cache_ttl': self.config.csv_cache_ttl,
                    'max_results_per_field': max_results
                }
                self._search_optimizer = CSVSearchOptimizer(optimizer_config)

            # Get index information to determine available fields
            index_info = await self._get_csv_index_info(index_name)
            if not index_info.get('success'):
                return {
                    'success': False,
                    'error': f"Index {index_name} not found or inaccessible"
                }

            # Extract field names from mapping
            mapping = index_info['info']['mapping']
            available_fields = list(mapping.get('properties', {}).keys())
            available_fields = [f for f in available_fields if not f.startswith('_csv_')]  # Exclude metadata

            # Optimize search fields if not provided
            if not search_fields:
                # Get field types for optimization
                field_types = {}
                for field, field_mapping in mapping.get('properties', {}).items():
                    field_types[field] = field_mapping.get('type', 'text')

                search_fields = self._search_optimizer.optimize_search_fields(
                    query, available_fields, field_types
                )

            # Analyze query for optimization
            query_analysis = self._search_optimizer.analyze_query(query, search_fields)

            # Check cache first
            cached_results = self._search_optimizer.get_cached_results(query_analysis.cache_key)
            if cached_results is not None:
                self.logger.info(f"📋 Returning cached results for query: {query}")
                return {
                    'success': True,
                    'results': cached_results,
                    'total_results': len(cached_results),
                    'query_analysis': query_analysis.__dict__,
                    'from_cache': True
                }

            # Execute search based on query type
            if query_analysis.query_type == "exact":
                # Use exact search for quoted queries
                search_results = await self._exact_search_csv(
                    index_name, query.strip('"'), search_fields
                )
            elif query_analysis.query_type == "range":
                # Use range search for numeric/date ranges
                search_results = await self._range_search_csv(
                    index_name, query, search_fields
                )
            else:
                # Use Elasticsearch fuzzy search combined with fuzzywuzzy
                search_results = await self._execute_elasticsearch_fuzzy_search(
                    index_name, query, search_fields, fuzzy_threshold
                )

            if not search_results.get('success'):
                return search_results

            # Enhance results with fuzzy matching if needed
            enhanced_results = await self._enhance_with_fuzzy_matching(
                search_results['results'], query, search_fields, fuzzy_threshold
            )

            # Generate suggestions if results are limited
            suggestions = []
            if len(enhanced_results) < 5 and self.config.csv_include_suggestions:
                suggestions = await self._generate_search_suggestions(
                    query, index_name, search_fields
                )

            # Cache results
            self._search_optimizer.cache_results(
                query_analysis.cache_key, enhanced_results, search_fields, index_name
            )

            result = {
                'success': True,
                'results': enhanced_results[:max_results],
                'total_results': len(enhanced_results),
                'query_analysis': query_analysis.__dict__,
                'suggestions': suggestions,
                'search_fields': search_fields,
                'fuzzy_threshold': fuzzy_threshold,
                'from_cache': False
            }

            self.logger.info(f"✅ Fuzzy search completed: {len(enhanced_results)} results")

            return result

        except Exception as e:
            self.logger.error(f"❌ Fuzzy search failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'query': query,
                'index_name': index_name
            }


    async def _exact_search_csv(self, index_name: str, query: str,
                               search_fields: List[str] = None) -> Dict[str, Any]:
        """Execute exact search for precise matching"""
        try:
            self.logger.info(f"🎯 Exact searching '{query}' in index: {index_name}")

            # Build exact match query
            if search_fields:
                # Multi-field exact search
                should_clauses = []
                for field in search_fields:
                    # Try both text and keyword fields
                    should_clauses.extend([
                        {"term": {f"{field}.keyword": query}},
                        {"match_phrase": {field: query}}
                    ])

                es_query = {
                    "query": {
                        "bool": {
                            "should": should_clauses,
                            "minimum_should_match": 1
                        }
                    },
                    "size": self.config.csv_max_search_results,
                    "highlight": {
                        "fields": {field: {} for field in search_fields}
                    }
                }
            else:
                # Search all fields
                es_query = {
                    "query": {
                        "multi_match": {
                            "query": query,
                            "type": "phrase"
                        }
                    },
                    "size": self.config.csv_max_search_results
                }

            # Execute search
            response = await self.client.search(
                index=index_name,
                body=es_query
            )

            # Process results
            results = []
            for hit in response['hits']['hits']:
                result = {
                    'document': hit['_source'],
                    'score': hit['_score'],
                    'document_id': hit['_id'],
                    'match_type': 'exact',
                    'highlights': hit.get('highlight', {}),
                    'confidence': 1.0  # Exact matches have full confidence
                }
                results.append(result)

            self.logger.info(f"✅ Exact search completed: {len(results)} results")

            return {
                'success': True,
                'results': results,
                'total_results': len(results),
                'query': query,
                'search_type': 'exact'
            }

        except Exception as e:
            self.logger.error(f"❌ Exact search failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'query': query,
                'index_name': index_name
            }

    async def _range_search_csv(self, index_name: str, query: str,
                               search_fields: List[str] = None) -> Dict[str, Any]:
        """Execute range search for numeric and date fields"""
        try:
            self.logger.info(f"📊 Range searching '{query}' in index: {index_name}")

            # Parse range query
            range_conditions = self._parse_range_query(query)
            if not range_conditions:
                return {
                    'success': False,
                    'error': 'Invalid range query format'
                }

            # Build range query
            range_clauses = []
            for field in search_fields or []:
                for condition in range_conditions:
                    range_clause = {
                        "range": {
                            field: condition
                        }
                    }
                    range_clauses.append(range_clause)

            if not range_clauses:
                return {
                    'success': False,
                    'error': 'No valid range conditions found'
                }

            es_query = {
                "query": {
                    "bool": {
                        "should": range_clauses,
                        "minimum_should_match": 1
                    }
                },
                "size": self.config.csv_max_search_results
            }

            # Execute search
            response = await self.client.search(
                index=index_name,
                body=es_query
            )

            # Process results
            results = []
            for hit in response['hits']['hits']:
                result = {
                    'document': hit['_source'],
                    'score': hit['_score'],
                    'document_id': hit['_id'],
                    'match_type': 'range',
                    'confidence': 0.9  # High confidence for range matches
                }
                results.append(result)

            self.logger.info(f"✅ Range search completed: {len(results)} results")

            return {
                'success': True,
                'results': results,
                'total_results': len(results),
                'query': query,
                'search_type': 'range'
            }

        except Exception as e:
            self.logger.error(f"❌ Range search failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'query': query,
                'index_name': index_name
            }

    async def _aggregate_csv_data(self, index_name: str, aggregation_config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data aggregation operations on CSV data"""
        try:
            self.logger.info(f"📈 Aggregating data in index: {index_name}")

            # Build aggregation query
            aggs = {}

            # Support common aggregation types
            for agg_name, agg_config in aggregation_config.items():
                agg_type = agg_config.get('type', 'terms')
                field = agg_config.get('field')

                if not field:
                    continue

                if agg_type == 'terms':
                    aggs[agg_name] = {
                        "terms": {
                            "field": f"{field}.keyword" if agg_config.get('use_keyword', True) else field,
                            "size": agg_config.get('size', 10)
                        }
                    }
                elif agg_type == 'stats':
                    aggs[agg_name] = {
                        "stats": {
                            "field": field
                        }
                    }
                elif agg_type == 'histogram':
                    aggs[agg_name] = {
                        "histogram": {
                            "field": field,
                            "interval": agg_config.get('interval', 1)
                        }
                    }
                elif agg_type == 'date_histogram':
                    aggs[agg_name] = {
                        "date_histogram": {
                            "field": field,
                            "calendar_interval": agg_config.get('interval', 'day')
                        }
                    }

            if not aggs:
                return {
                    'success': False,
                    'error': 'No valid aggregations specified'
                }

            es_query = {
                "size": 0,  # Don't return documents, just aggregations
                "aggs": aggs
            }

            # Execute aggregation
            response = await self.client.search(
                index=index_name,
                body=es_query
            )

            # Process aggregation results
            aggregation_results = {}
            for agg_name, agg_data in response.get('aggregations', {}).items():
                aggregation_results[agg_name] = agg_data

            self.logger.info(f"✅ Aggregation completed: {len(aggregation_results)} aggregations")

            return {
                'success': True,
                'aggregations': aggregation_results,
                'total_documents': response['hits']['total']['value']
            }

        except Exception as e:
            self.logger.error(f"❌ Aggregation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'index_name': index_name
            }


    # ===================================================================
    # HELPER METHODS FOR FUZZY SEARCH
    # ===================================================================

    async def _execute_elasticsearch_fuzzy_search(self, index_name: str, query: str,
                                                 search_fields: List[str],
                                                 fuzzy_threshold: float) -> Dict[str, Any]:
        """Execute Elasticsearch-based fuzzy search"""
        try:
            # Build fuzzy search query
            should_clauses = []

            for field in search_fields:
                # Add different types of fuzzy matching
                should_clauses.extend([
                    # Fuzzy match
                    {
                        "fuzzy": {
                            field: {
                                "value": query,
                                "fuzziness": "AUTO",
                                "boost": 1.0
                            }
                        }
                    },
                    # Match with fuzzy analyzer
                    {
                        "match": {
                            f"{field}.fuzzy": {
                                "query": query,
                                "boost": 0.8
                            }
                        }
                    },
                    # Wildcard search for partial matches
                    {
                        "wildcard": {
                            f"{field}.keyword": {
                                "value": f"*{query}*",
                                "boost": 0.6
                            }
                        }
                    }
                ])

            es_query = {
                "query": {
                    "bool": {
                        "should": should_clauses,
                        "minimum_should_match": 1
                    }
                },
                "size": self.config.csv_max_search_results,
                "highlight": {
                    "fields": {field: {} for field in search_fields}
                },
                "min_score": fuzzy_threshold * 10  # Convert to Elasticsearch score scale
            }

            # Execute search
            response = await self.client.search(
                index=index_name,
                body=es_query
            )

            # Process results
            results = []
            for hit in response['hits']['hits']:
                # Normalize Elasticsearch score to 0-1 range
                normalized_score = min(1.0, hit['_score'] / 10.0)

                result = {
                    'document': hit['_source'],
                    'score': hit['_score'],
                    'normalized_score': normalized_score,
                    'document_id': hit['_id'],
                    'match_type': 'elasticsearch_fuzzy',
                    'highlights': hit.get('highlight', {}),
                    'confidence': normalized_score
                }
                results.append(result)

            return {
                'success': True,
                'results': results,
                'total_results': len(results)
            }

        except Exception as e:
            self.logger.error(f"❌ Elasticsearch fuzzy search failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    async def _enhance_with_fuzzy_matching(self, es_results: List[Dict[str, Any]],
                                         query: str, search_fields: List[str],
                                         threshold: float) -> List[Dict[str, Any]]:
        """Enhance Elasticsearch results with fuzzywuzzy matching"""
        try:
            if not FUZZYWUZZY_AVAILABLE:
                # Return ES results as-is if fuzzywuzzy not available
                return es_results

            enhanced_results = []

            # Convert ES results to format expected by fuzzy matcher
            candidates = []
            for result in es_results:
                candidate = result['document'].copy()
                candidate['_id'] = result['document_id']
                candidates.append(candidate)

            # Apply fuzzy matching
            fuzzy_matches = self._fuzzy_matcher.fuzzy_search(
                query, candidates, search_fields, threshold
            )

            # Combine ES and fuzzy scores
            for es_result in es_results:
                doc_id = es_result['document_id']

                # Find corresponding fuzzy match
                fuzzy_match = None
                for match in fuzzy_matches:
                    if match.document_id == doc_id:
                        fuzzy_match = match
                        break

                if fuzzy_match:
                    # Combine scores (weighted average)
                    es_score = es_result.get('normalized_score', 0.5)
                    fuzzy_score = fuzzy_match.score
                    combined_score = (es_score * 0.4) + (fuzzy_score * 0.6)  # Favor fuzzy score

                    enhanced_result = es_result.copy()
                    enhanced_result.update({
                        'fuzzy_score': fuzzy_score,
                        'combined_score': combined_score,
                        'confidence': combined_score,
                        'fuzzy_algorithm': fuzzy_match.algorithm,
                        'fuzzy_highlights': fuzzy_match.highlights or []
                    })
                    enhanced_results.append(enhanced_result)
                else:
                    # Keep ES result with lower confidence
                    enhanced_result = es_result.copy()
                    enhanced_result['confidence'] = enhanced_result.get('normalized_score', 0.5) * 0.8
                    enhanced_results.append(enhanced_result)

            # Sort by combined confidence score
            enhanced_results.sort(key=lambda x: x.get('confidence', 0), reverse=True)

            return enhanced_results

        except Exception as e:
            self.logger.error(f"❌ Fuzzy enhancement failed: {e}")
            return es_results  # Return original results on error

    async def _generate_search_suggestions(self, query: str, index_name: str,
                                         search_fields: List[str]) -> List[Dict[str, Any]]:
        """Generate search suggestions for improved results"""
        try:
            suggestions = []

            if not FUZZYWUZZY_AVAILABLE:
                return suggestions

            # Get sample field values for suggestion generation
            sample_query = {
                "size": 100,
                "_source": search_fields,
                "query": {"match_all": {}}
            }

            response = await self.client.search(
                index=index_name,
                body=sample_query
            )

            # Extract field values
            field_values = []
            for hit in response['hits']['hits']:
                for field in search_fields:
                    if field in hit['_source']:
                        value = str(hit['_source'][field])
                        if value and len(value) > 1:
                            field_values.append(value)

            # Generate suggestions using fuzzy matcher
            if field_values:
                fuzzy_suggestions = self._fuzzy_matcher.generate_suggestions(query, field_values)

                for suggestion in fuzzy_suggestions:
                    suggestions.append({
                        'original_query': suggestion.original_query,
                        'suggested_query': suggestion.suggested_query,
                        'confidence': suggestion.confidence,
                        'reason': suggestion.reason
                    })

            return suggestions[:5]  # Limit to 5 suggestions

        except Exception as e:
            self.logger.error(f"❌ Suggestion generation failed: {e}")
            return []

    def _parse_range_query(self, query: str) -> List[Dict[str, Any]]:
        """Parse range query string into Elasticsearch range conditions"""
        try:
            conditions = []

            # Simple range patterns
            if '>=' in query:
                parts = query.split('>=')
                if len(parts) == 2:
                    value = parts[1].strip()
                    conditions.append({"gte": self._convert_value(value)})
            elif '>' in query:
                parts = query.split('>')
                if len(parts) == 2:
                    value = parts[1].strip()
                    conditions.append({"gt": self._convert_value(value)})

            if '<=' in query:
                parts = query.split('<=')
                if len(parts) == 2:
                    value = parts[1].strip()
                    conditions.append({"lte": self._convert_value(value)})
            elif '<' in query:
                parts = query.split('<')
                if len(parts) == 2:
                    value = parts[1].strip()
                    conditions.append({"lt": self._convert_value(value)})

            # Between pattern
            if 'between' in query.lower():
                parts = query.lower().split('between')
                if len(parts) == 2:
                    range_part = parts[1].strip()
                    if ' and ' in range_part:
                        values = range_part.split(' and ')
                        if len(values) == 2:
                            min_val = self._convert_value(values[0].strip())
                            max_val = self._convert_value(values[1].strip())
                            conditions.append({"gte": min_val, "lte": max_val})

            return conditions

        except Exception as e:
            self.logger.warning(f"Range query parsing failed: {e}")
            return []

    def _convert_value(self, value: str) -> Any:
        """Convert string value to appropriate type"""
        value = value.strip()

        # Try integer
        try:
            return int(value)
        except ValueError:
            pass

        # Try float
        try:
            return float(value)
        except ValueError:
            pass

        # Return as string
        return value


    # ===================================================================
    # CSV UTILITY METHODS
    # ===================================================================

    async def _get_csv_field_suggestions(self, index_name: str, query: str) -> Dict[str, Any]:
        """Get field suggestions based on query content"""
        try:
            self.logger.info(f"💡 Getting field suggestions for query: {query}")

            # Get index mapping to understand available fields
            index_info = await self._get_csv_index_info(index_name)
            if not index_info.get('success'):
                return {
                    'success': False,
                    'error': 'Could not retrieve index information'
                }

            mapping = index_info['info']['mapping']
            available_fields = list(mapping.get('properties', {}).keys())
            available_fields = [f for f in available_fields if not f.startswith('_csv_')]

            # Analyze query to suggest relevant fields
            query_lower = query.lower()
            field_suggestions = []

            for field in available_fields:
                field_lower = field.lower()
                relevance_score = 0

                # Direct field name match
                if field_lower in query_lower:
                    relevance_score += 10

                # Partial field name match
                for word in query_lower.split():
                    if word in field_lower:
                        relevance_score += 5

                # Field type relevance
                field_mapping = mapping['properties'].get(field, {})
                field_type = field_mapping.get('type', 'text')

                if field_type == 'text' and any(char.isalpha() for char in query):
                    relevance_score += 3
                elif field_type in ['long', 'double'] and any(char.isdigit() for char in query):
                    relevance_score += 5
                elif field_type == 'date' and self._looks_like_date(query):
                    relevance_score += 8

                if relevance_score > 0:
                    field_suggestions.append({
                        'field_name': field,
                        'field_type': field_type,
                        'relevance_score': relevance_score,
                        'reason': self._get_suggestion_reason(field, field_type, query)
                    })

            # Sort by relevance score
            field_suggestions.sort(key=lambda x: x['relevance_score'], reverse=True)

            return {
                'success': True,
                'suggestions': field_suggestions[:10],  # Top 10 suggestions
                'total_fields': len(available_fields),
                'query': query
            }

        except Exception as e:
            self.logger.error(f"❌ Field suggestion failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'query': query
            }

    async def _detect_csv_data_types(self, index_name: str, sample_size: int = 100) -> Dict[str, Any]:
        """Detect data types in CSV index by sampling documents"""
        try:
            self.logger.info(f"🔍 Detecting data types in index: {index_name}")

            # Sample documents from the index
            sample_query = {
                "size": sample_size,
                "query": {"match_all": {}}
            }

            response = await self.client.search(
                index=index_name,
                body=sample_query
            )

            if not response['hits']['hits']:
                return {
                    'success': False,
                    'error': 'No documents found in index'
                }

            # Analyze field types from sample data
            field_analysis = {}

            for hit in response['hits']['hits']:
                document = hit['_source']

                for field_name, field_value in document.items():
                    if field_name.startswith('_csv_'):
                        continue  # Skip metadata fields

                    if field_name not in field_analysis:
                        field_analysis[field_name] = {
                            'values': [],
                            'types': {},
                            'null_count': 0,
                            'total_count': 0
                        }

                    field_analysis[field_name]['total_count'] += 1

                    if field_value is None or field_value == '':
                        field_analysis[field_name]['null_count'] += 1
                        continue

                    field_analysis[field_name]['values'].append(str(field_value))

                    # Detect type
                    detected_type = self._detect_value_type(str(field_value))
                    if detected_type not in field_analysis[field_name]['types']:
                        field_analysis[field_name]['types'][detected_type] = 0
                    field_analysis[field_name]['types'][detected_type] += 1

            # Determine primary type for each field
            field_types = {}
            for field_name, analysis in field_analysis.items():
                if analysis['types']:
                    # Get most common type
                    primary_type = max(analysis['types'], key=analysis['types'].get)
                    type_confidence = analysis['types'][primary_type] / analysis['total_count']

                    field_types[field_name] = {
                        'primary_type': primary_type,
                        'confidence': type_confidence,
                        'type_distribution': analysis['types'],
                        'null_percentage': (analysis['null_count'] / analysis['total_count']) * 100,
                        'sample_values': analysis['values'][:5]  # First 5 sample values
                    }

            return {
                'success': True,
                'field_types': field_types,
                'sample_size': len(response['hits']['hits']),
                'total_fields': len(field_types)
            }

        except Exception as e:
            self.logger.error(f"❌ Data type detection failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'index_name': index_name
            }

    async def _generate_csv_mapping(self, index_name: str, csv_structure: Dict[str, Any]) -> Dict[str, Any]:
        """Generate optimized mapping for CSV data"""
        try:
            # Use the mapping generator utility
            from .csv_mapping_generator import CSVMappingGenerator

            generator_config = {
                'enable_fuzzy_fields': self.config.csv_include_highlights,
                'enable_keyword_fields': True,
                'max_keyword_length': 256,
                'default_analyzer': 'csv_text_analyzer',
                'fuzzy_analyzer': 'csv_fuzzy_analyzer'
            }

            generator = CSVMappingGenerator(generator_config)
            mapping = generator.generate_mapping(csv_structure)

            return {
                'success': True,
                'mapping': mapping,
                'index_name': index_name
            }

        except Exception as e:
            self.logger.error(f"❌ CSV mapping generation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'index_name': index_name
            }

    def _detect_value_type(self, value: str) -> str:
        """Detect the type of a single value"""
        if not value or not value.strip():
            return 'empty'

        value = value.strip()

        # Boolean check
        if value.lower() in ['true', 'false', 'yes', 'no', '1', '0', 'y', 'n']:
            return 'boolean'

        # Integer check
        try:
            int(value)
            return 'integer'
        except ValueError:
            pass

        # Float check
        try:
            float(value)
            return 'float'
        except ValueError:
            pass

        # Date check
        if self._looks_like_date(value):
            return 'date'

        # Email check
        if '@' in value and '.' in value:
            return 'email'

        # URL check
        if value.startswith(('http://', 'https://', 'ftp://')):
            return 'url'

        return 'text'

    def _looks_like_date(self, value: str) -> bool:
        """Check if value looks like a date"""
        import re

        date_patterns = [
            r'\d{4}[-/]\d{1,2}[-/]\d{1,2}',  # YYYY-MM-DD
            r'\d{1,2}[-/]\d{1,2}[-/]\d{4}',  # MM-DD-YYYY
            r'\d{1,2}[-/]\d{1,2}[-/]\d{2}',  # MM-DD-YY
        ]

        return any(re.match(pattern, value) for pattern in date_patterns)

    def _get_suggestion_reason(self, field_name: str, field_type: str, query: str) -> str:
        """Get reason for field suggestion"""
        field_lower = field_name.lower()
        query_lower = query.lower()

        if field_lower in query_lower:
            return "Field name matches query"
        elif any(word in field_lower for word in query_lower.split()):
            return "Field name contains query words"
        elif field_type == 'text' and any(char.isalpha() for char in query):
            return "Text field suitable for text query"
        elif field_type in ['long', 'double'] and any(char.isdigit() for char in query):
            return "Numeric field suitable for numeric query"
        elif field_type == 'date' and self._looks_like_date(query):
            return "Date field suitable for date query"
        else:
            return "General field relevance"


# Main execution block for Docker container
async def main():
    """Main function to run the Elasticsearch MCP Server"""
    import os
    import signal
    import sys
    
    # Get configuration from environment variables
    config = ElasticsearchMCPConfig(
        mcp_host=os.getenv("MCP_HOST", "0.0.0.0"),
        mcp_port=int(os.getenv("MCP_PORT", "9202")),
        elasticsearch_host=os.getenv("ELASTICSEARCH_HOST", "elasticsearch"),
        elasticsearch_port=int(os.getenv("ELASTICSEARCH_PORT", "9200")),
        elasticsearch_scheme=os.getenv("ELASTICSEARCH_SCHEME", "http"),
        elasticsearch_username=os.getenv("ELASTICSEARCH_USERNAME"),
        elasticsearch_password=os.getenv("ELASTICSEARCH_PASSWORD"),
        max_connections=int(os.getenv("MCP_MAX_CONNECTIONS", "10")),
        connection_timeout=float(os.getenv("MCP_CONNECTION_TIMEOUT", "30")),
        request_timeout=float(os.getenv("MCP_REQUEST_TIMEOUT", "60")),
        auto_start_docker=False  # Don't auto-start Docker when running in Docker
    )
    
    # Create MCP server instance
    server = ElasticsearchMCPServer(config)
    
    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        print(f"\n🛑 Received signal {signum}, shutting down gracefully...")
        asyncio.create_task(server.stop())
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        print("🚀 Starting Elasticsearch MCP Server...")
        print(f"   📍 MCP Host: {config.mcp_host}:{config.mcp_port}")
        print(f"   🔗 Elasticsearch: {config.elasticsearch_scheme}://{config.elasticsearch_host}:{config.elasticsearch_port}")
        
        # Start the server
        await server.start()
        
        print("✅ Elasticsearch MCP Server is running")
        print("   Press Ctrl+C to stop")
        
        # Keep the server running
        while server.is_running:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Keyboard interrupt received, shutting down...")
    except Exception as e:
        print(f"❌ Server error: {e}")
        sys.exit(1)
    finally:
        await server.stop()
        print("✅ Server shutdown complete")


if __name__ == "__main__":
    # Check required dependencies
    if not ELASTICSEARCH_AVAILABLE:
        print("❌ Elasticsearch client not available. Install with: pip install elasticsearch>=8.11.0")
        sys.exit(1)
    
    if not AIOHTTP_AVAILABLE:
        print("❌ aiohttp not available. Install with: pip install aiohttp>=3.8.0")
        sys.exit(1)
    
    # Run the server
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        sys.exit(1) 