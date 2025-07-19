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
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field

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
    
    # Tool card (mandatory)
    tool_card: Dict[str, Any] = field(default_factory=lambda: {
        "name": "elasticsearch_mcp_server",
        "description": "Elasticsearch MCP server for search and analytics",
        "version": "1.0.0",
        "category": "search_analytics",
        "capabilities": ["search", "indexing", "analytics", "mcp"]
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
            "index_document": self._index_document,
            "bulk_index": self._bulk_index,
            "search": self._search,
            "semantic_search": self._semantic_search,
            "get_document": self._get_document,
            "delete_document": self._delete_document,
            "create_index": self._create_index,
            "delete_index": self._delete_index,
            "cluster_health": self._cluster_health
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