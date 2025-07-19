"""
Elasticsearch Tool for NanoBrain Framework

This tool provides Elasticsearch search capabilities with comprehensive auto-installation
using the Docker infrastructure module. Supports full-text search, fuzzy matching,
and virus genome name resolution.

"""

import asyncio
import json
import os
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path

from elasticsearch import AsyncElasticsearch
from elasticsearch.exceptions import ConnectionError, NotFoundError, RequestError

from nanobrain.core.external_tool import ExternalTool, ToolResult, InstallationStatus, ExternalToolConfig
from nanobrain.core.tool import ToolConfig
from nanobrain.core.logging_system import get_logger
from nanobrain.library.infrastructure.docker import (
    DockerManager, ContainerConfig, HealthCheckConfig, ResourceLimits,
    DockerHealthMonitor, DockerNetworkManager, DockerVolumeManager
)


@dataclass
class ElasticsearchConfig(ExternalToolConfig):
    """Configuration for Elasticsearch tool"""
    # Tool identification
    tool_name: str = "elasticsearch"
    
    # Default tool card
    tool_card: Dict[str, Any] = field(default_factory=lambda: {
        "name": "elasticsearch",
        "description": "Elasticsearch search engine with comprehensive auto-installation",
        "version": "8.14.0",
        "category": "search",
        "vendor": "Elastic N.V.",
        "license": "Elastic License 2.0",
        "documentation_url": "https://www.elastic.co/guide/en/elasticsearch/reference/current/",
        "capabilities": [
            "document_indexing",
            "full_text_search", 
            "fuzzy_matching",
            "autocomplete",
            "aggregations",
            "real_time_search"
        ]
    })
    
    # Elasticsearch connection settings
    host: str = "localhost"
    port: int = 9200
    scheme: str = "http"
    username: Optional[str] = None
    password: Optional[str] = None
    
    # Index configuration
    default_index: str = "nanobrain-search"
    virus_genome_index: str = "virus-genomes"
    
    # Search configuration
    fuzzy_fuzziness: str = "AUTO"
    max_search_results: int = 100
    highlight_enabled: bool = True
    
    # Auto-installation configuration
    auto_install_enabled: bool = True
    container_name: str = "nanobrain-elasticsearch"
    image_name: str = "elasticsearch"
    image_tag: str = "8.14.0"
    
    # Container resource limits
    memory_limit: str = "2Gi"
    cpu_limit: str = "1.0"
    
    # Data persistence
    data_volume_name: str = "nanobrain-elasticsearch-data"
    backup_enabled: bool = True
    
    # Network configuration
    network_name: str = "nanobrain-network"
    
    # Health monitoring
    health_check_enabled: bool = True
    health_check_interval: int = 30


class ElasticsearchInstallationError(Exception):
    """Raised when Elasticsearch installation fails"""
    pass


class ElasticsearchSearchError(Exception):
    """Raised when Elasticsearch search operations fail"""
    pass


class ElasticsearchTool(ExternalTool):
    """
    Elasticsearch search tool with comprehensive auto-installation capabilities.
    
    Features:
    - Full Docker container lifecycle management
    - Automatic index creation and configuration
    - Fuzzy search with virus-specific analyzers
    - Health monitoring and alerting
    - Data persistence and backup
    - Kubernetes-ready configuration
    """
    
    @classmethod
    def _get_config_class(cls):
        """UNIFIED PATTERN: Return ElasticsearchConfig - ONLY method that differs from other components"""
        return ElasticsearchConfig
    
    # Now inherits unified from_config implementation from FromConfigBase
    # Uses ElasticsearchConfig returned by _get_config_class() to preserve all existing functionality

    
    def __init__(self, config: ElasticsearchConfig, **kwargs):
        """Initialize Elasticsearch tool with configuration"""
        if config is None:
            config = ElasticsearchConfig()
        
        # Ensure name is set consistently
        if not hasattr(config, 'tool_name') or not config.tool_name:
            config.tool_name = "elasticsearch"
        
        # Initialize parent class
        super().__init__(config, **kwargs)
        
        # Store Elasticsearch-specific configuration
        self.es_config = config
        self.name = config.tool_name
        self.logger = get_logger(f"elasticsearch_tool_{self.name}")
        
        # Docker infrastructure components
        self.docker_manager: Optional[DockerManager] = None
        self.health_monitor: Optional[DockerHealthMonitor] = None
        self.network_manager: Optional[DockerNetworkManager] = None
        self.volume_manager: Optional[DockerVolumeManager] = None
        
        # Elasticsearch client
        self.es_client: Optional[AsyncElasticsearch] = None
        
        # Installation state
        self.container_running = False
        self.indices_created = False
        
        # Search statistics
        self.search_count = 0
        self.index_count = 0
    
    async def initialize_tool(self) -> InstallationStatus:
        """Initialize Elasticsearch tool with auto-installation"""
        self.logger.info("🔄 Initializing Elasticsearch tool with auto-installation...")
        
        try:
            # Initialize Docker infrastructure
            await self._initialize_docker_infrastructure()
            
            # Check if auto-installation is enabled
            if self.es_config.auto_install_enabled:
                # Auto-install Elasticsearch via Docker
                installation_success = await self._auto_install_elasticsearch()
                
                if not installation_success:
                    raise ElasticsearchInstallationError("Auto-installation failed")
                
                # Wait for Elasticsearch to be ready
                await self._wait_for_elasticsearch_ready()
            else:
                # Verify existing Elasticsearch installation
                await self._verify_elasticsearch_available()
            
            # Initialize Elasticsearch client
            await self._initialize_elasticsearch_client()
            
            # Create indices and configure mappings
            await self._setup_indices()
            
            # Start health monitoring if enabled
            if self.es_config.health_check_enabled:
                await self._start_health_monitoring()
            
            self.logger.info("✅ Elasticsearch tool initialized successfully")
            
            return InstallationStatus(
                found=True,
                is_functional=True,
                installation_path=f"docker://{self.es_config.container_name}",
                version=self.es_config.image_tag,
                details={
                    "container_name": self.es_config.container_name,
                    "host": self.es_config.host,
                    "port": self.es_config.port,
                    "indices_created": self.indices_created,
                    "auto_installed": self.es_config.auto_install_enabled
                }
            )
            
        except Exception as e:
            self.logger.error(f"❌ Elasticsearch tool initialization failed: {e}")
            raise ElasticsearchInstallationError(f"Initialization failed: {e}")
    
    async def _initialize_docker_infrastructure(self):
        """Initialize Docker infrastructure components"""
        try:
            self.logger.info("🐳 Initializing Docker infrastructure...")
            
            # Initialize Docker manager
            self.docker_manager = DockerManager()
            await self.docker_manager.verify_docker_available()
            
            # Initialize network manager
            self.network_manager = DockerNetworkManager(self.docker_manager.client)
            
            # Initialize volume manager
            self.volume_manager = DockerVolumeManager(self.docker_manager.client)
            
            # Initialize health monitor
            self.health_monitor = DockerHealthMonitor(
                self.docker_manager,
                check_interval=self.es_config.health_check_interval
            )
            
            self.logger.info("✅ Docker infrastructure initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Docker infrastructure initialization failed: {e}")
            raise
    
    async def _auto_install_elasticsearch(self) -> bool:
        """Auto-install Elasticsearch using Docker infrastructure"""
        try:
            self.logger.info("🚀 Starting Elasticsearch auto-installation...")
            
            # Create network if it doesn't exist
            await self._ensure_network_exists()
            
            # Create data volume if it doesn't exist
            await self._ensure_data_volume_exists()
            
            # Create and configure Elasticsearch container
            container_config = self._create_container_config()
            
            # Create and start container
            success = await self.docker_manager.create_container(container_config)
            
            if success:
                self.container_running = True
                self.logger.info("✅ Elasticsearch container started successfully")
                return True
            else:
                self.logger.error("❌ Failed to start Elasticsearch container")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Elasticsearch auto-installation failed: {e}")
            return False
    
    def _create_container_config(self) -> ContainerConfig:
        """Create container configuration for Elasticsearch"""
        # Environment configuration for single-node development setup
        environment = {
            "discovery.type": "single-node",
            "xpack.security.enabled": "false",
            "ES_JAVA_OPTS": f"-Xms1g -Xmx{self.es_config.memory_limit.replace('Gi', 'g').replace('Mi', 'm')}",
            "bootstrap.memory_lock": "true",
            "cluster.name": "nanobrain-cluster",
            "node.name": "nanobrain-node-1",
            "http.cors.enabled": "true",
            "http.cors.allow-origin": "*",
            "logger.level": "INFO"
        }
        
        # Resource limits
        resource_limits = ResourceLimits(
            memory=self.es_config.memory_limit,
            cpu=self.es_config.cpu_limit,
            memory_request="512Mi",
            cpu_request="0.5"
        )
        
        # Health check configuration
        health_check = HealthCheckConfig(
            type="http",
            path="/_cluster/health",
            port=self.es_config.port,
            interval=30,
            timeout=10,
            retries=3,
            start_period=120  # Elasticsearch needs time to start
        )
        
        # Container configuration
        container_config = ContainerConfig(
            name=self.es_config.container_name,
            image=self.es_config.image_name,
            tag=self.es_config.image_tag,
            ports=[f"{self.es_config.port}:9200", "9300:9300"],
            environment=environment,
            volumes=[f"{self.es_config.data_volume_name}:/usr/share/elasticsearch/data"],
            networks=[self.es_config.network_name],
            resource_limits=resource_limits,
            health_check=health_check,
            labels={
                "nanobrain.tool": "elasticsearch",
                "nanobrain.tool.version": self.es_config.image_tag,
                "nanobrain.tool.auto_installed": "true"
            }
        )
        
        return container_config
    
    async def _ensure_network_exists(self):
        """Ensure the NanoBrain network exists"""
        try:
            # Create NanoBrain network if it doesn't exist
            await self.network_manager.create_nanobrain_network()
            self.logger.info(f"✅ Network {self.es_config.network_name} ready")
        except Exception as e:
            self.logger.error(f"❌ Failed to create network: {e}")
            raise
    
    async def _ensure_data_volume_exists(self):
        """Ensure the Elasticsearch data volume exists"""
        try:
            from nanobrain.library.infrastructure.docker.docker_volume_manager import VolumeConfig
            
            volume_config = VolumeConfig(
                name=self.es_config.data_volume_name,
                driver="local",
                backup_enabled=self.es_config.backup_enabled,
                retention_days=30,
                labels={
                    "nanobrain.tool": "elasticsearch",
                    "nanobrain.volume.purpose": "data"
                }
            )
            
            await self.volume_manager.create_volume(volume_config)
            self.logger.info(f"✅ Data volume {self.es_config.data_volume_name} ready")
        except Exception as e:
            self.logger.error(f"❌ Failed to create data volume: {e}")
            raise
    
    async def _wait_for_elasticsearch_ready(self, timeout: int = 300):
        """Wait for Elasticsearch to be ready to accept connections"""
        self.logger.info("⏳ Waiting for Elasticsearch to be ready...")
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                # Check container health
                if await self.docker_manager.health_check(self.es_config.container_name):
                    self.logger.info("✅ Elasticsearch is ready")
                    return
                
                # Log progress every 30 seconds
                elapsed = time.time() - start_time
                if int(elapsed) % 30 == 0:
                    self.logger.info(f"⏳ Still waiting for Elasticsearch... ({elapsed:.0f}s)")
                
                await asyncio.sleep(5)
                
            except Exception as e:
                self.logger.debug(f"Health check failed (expected during startup): {e}")
                await asyncio.sleep(5)
        
        raise ElasticsearchInstallationError(f"Elasticsearch not ready after {timeout} seconds")
    
    async def _verify_elasticsearch_available(self):
        """Verify existing Elasticsearch installation is available"""
        try:
            # Try to connect to existing Elasticsearch instance
            es_url = f"{self.es_config.scheme}://{self.es_config.host}:{self.es_config.port}"
            
            import aiohttp
            timeout = aiohttp.ClientTimeout(total=10)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(f"{es_url}/_cluster/health") as response:
                    if response.status == 200:
                        self.logger.info("✅ Existing Elasticsearch installation verified")
                        return
                    else:
                        raise ElasticsearchInstallationError(f"Elasticsearch health check failed: {response.status}")
        
        except Exception as e:
            raise ElasticsearchInstallationError(f"Existing Elasticsearch not available: {e}")
    
    async def _initialize_elasticsearch_client(self):
        """Initialize Elasticsearch client"""
        try:
            # Elasticsearch connection URL
            es_url = f"{self.es_config.scheme}://{self.es_config.host}:{self.es_config.port}"
            
            # Client configuration
            client_config = {
                "hosts": [es_url],
                "timeout": 30,
                "max_retries": 3,
                "retry_on_timeout": True
            }
            
            # Add authentication if configured
            if self.es_config.username and self.es_config.password:
                client_config["http_auth"] = (self.es_config.username, self.es_config.password)
            
            # Create client
            self.es_client = AsyncElasticsearch(**client_config)
            
            # Test connection
            info = await self.es_client.info()
            version = info["version"]["number"]
            
            self.logger.info(f"✅ Elasticsearch client connected (version {version})")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Elasticsearch client: {e}")
            raise
    
    async def _setup_indices(self):
        """Create and configure Elasticsearch indices"""
        try:
            self.logger.info("🔧 Setting up Elasticsearch indices...")
            
            # Create default search index
            await self._create_search_index()
            
            # Create virus genome index with specialized configuration
            await self._create_virus_genome_index()
            
            self.indices_created = True
            self.logger.info("✅ Elasticsearch indices configured")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to setup indices: {e}")
            raise
    
    async def _create_search_index(self):
        """Create the default search index"""
        index_name = self.es_config.default_index
        
        # Check if index already exists
        if await self.es_client.indices.exists(index=index_name):
            self.logger.info(f"Index {index_name} already exists")
            return
        
        # Index mapping for general search
        mapping = {
            "mappings": {
                "properties": {
                    "title": {
                        "type": "text",
                        "analyzer": "standard",
                        "fields": {
                            "keyword": {"type": "keyword"},
                            "suggest": {"type": "completion"}
                        }
                    },
                    "content": {
                        "type": "text",
                        "analyzer": "standard"
                    },
                    "category": {"type": "keyword"},
                    "tags": {"type": "keyword"},
                    "timestamp": {"type": "date"},
                    "metadata": {"type": "object"}
                }
            },
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "analysis": {
                    "analyzer": {
                        "nanobrain_analyzer": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": ["lowercase", "stop", "snowball"]
                        }
                    }
                }
            }
        }
        
        await self.es_client.indices.create(index=index_name, body=mapping)
        self.logger.info(f"✅ Created search index: {index_name}")
    
    async def _create_virus_genome_index(self):
        """Create specialized virus genome index"""
        index_name = self.es_config.virus_genome_index
        
        # Check if index already exists
        if await self.es_client.indices.exists(index=index_name):
            self.logger.info(f"Index {index_name} already exists")
            return
        
        # Specialized mapping for virus genomes with synonym support
        mapping = {
            "mappings": {
                "properties": {
                    "genome_id": {"type": "keyword"},
                    "genome_name": {
                        "type": "text",
                        "analyzer": "virus_name_analyzer",
                        "fields": {
                            "keyword": {"type": "keyword"},
                            "suggest": {"type": "completion"},
                            "fuzzy": {
                                "type": "text",
                                "analyzer": "virus_fuzzy_analyzer"
                            }
                        }
                    },
                    "organism_name": {
                        "type": "text",
                        "analyzer": "virus_name_analyzer"
                    },
                    "strain": {"type": "keyword"},
                    "genome_length": {"type": "integer"},
                    "taxon_id": {"type": "keyword"},
                    "taxon_lineage": {"type": "text"},
                    "host": {"type": "keyword"},
                    "isolation_country": {"type": "keyword"},
                    "collection_date": {"type": "date"},
                    "genome_status": {"type": "keyword"},
                    "genome_type": {"type": "keyword"},
                    "description": {"type": "text"},
                    "synonyms": {"type": "keyword"},
                    "abbreviations": {"type": "keyword"}
                }
            },
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "analysis": {
                    "filter": {
                        "virus_synonym_filter": {
                            "type": "synonym",
                            "synonyms": [
                                "CHIKV,Chikungunya virus,Chikungunya",
                                "EEEV,Eastern equine encephalitis virus,Eastern equine encephalitis",
                                "WEEV,Western equine encephalitis virus,Western equine encephalitis",
                                "VEEV,Venezuelan equine encephalitis virus,Venezuelan equine encephalitis",
                                "SINV,Sindbis virus,Sindbis",
                                "SFV,Semliki Forest virus,Semliki Forest",
                                "RRV,Ross River virus,Ross River",
                                "BFV,Barmah Forest virus,Barmah Forest",
                                "MAYV,Mayaro virus,Mayaro",
                                "UNAV,Una virus,Una"
                            ]
                        },
                        "virus_stemmer": {
                            "type": "stemmer",
                            "language": "english"
                        }
                    },
                    "analyzer": {
                        "virus_name_analyzer": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": [
                                "lowercase",
                                "virus_synonym_filter",
                                "virus_stemmer"
                            ]
                        },
                        "virus_fuzzy_analyzer": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": [
                                "lowercase",
                                "asciifolding",
                                "virus_synonym_filter"
                            ]
                        }
                    }
                }
            }
        }
        
        await self.es_client.indices.create(index=index_name, body=mapping)
        self.logger.info(f"✅ Created virus genome index: {index_name}")
    
    async def _start_health_monitoring(self):
        """Start health monitoring for Elasticsearch container"""
        try:
            # Add container to health monitoring
            health_config = HealthCheckConfig(
                type="http",
                path="/_cluster/health",
                port=self.es_config.port,
                interval=self.es_config.health_check_interval
            )
            
            self.health_monitor.add_container(
                self.es_config.container_name,
                health_config
            )
            
            # Add alert callback
            self.health_monitor.add_alert_callback(self._handle_health_alert)
            
            # Start monitoring
            await self.health_monitor.start_monitoring()
            
            self.logger.info("✅ Health monitoring started")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start health monitoring: {e}")
            # Don't fail initialization for monitoring issues
    
    def _handle_health_alert(self, container_name: str, health_result):
        """Handle health alerts"""
        self.logger.warning(f"🚨 Health alert for {container_name}: {health_result.status.value}")
        if health_result.error_message:
            self.logger.warning(f"Error: {health_result.error_message}")
    
    # Search and indexing methods
    
    async def index_document(self, index: str, doc_id: str, document: Dict[str, Any]) -> bool:
        """Index a document"""
        try:
            if not self.es_client:
                raise ElasticsearchSearchError("Elasticsearch client not initialized")
            
            await self.es_client.index(
                index=index,
                id=doc_id,
                body=document
            )
            
            self.index_count += 1
            self.logger.debug(f"✅ Indexed document {doc_id} in {index}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to index document: {e}")
            return False
    
    async def search_documents(self, query: str, index: Optional[str] = None,
                             max_results: Optional[int] = None) -> Dict[str, Any]:
        """Search documents with fuzzy matching"""
        try:
            if not self.es_client:
                raise ElasticsearchSearchError("Elasticsearch client not initialized")
            
            index = index or self.es_config.default_index
            max_results = max_results or self.es_config.max_search_results
            
            # Build search query
            search_body = {
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["title^2", "content", "genome_name^3", "organism_name^2"],
                        "fuzziness": self.es_config.fuzzy_fuzziness,
                        "type": "best_fields"
                    }
                },
                "size": max_results,
                "sort": ["_score"]
            }
            
            # Add highlighting if enabled
            if self.es_config.highlight_enabled:
                search_body["highlight"] = {
                    "fields": {
                        "title": {},
                        "content": {},
                        "genome_name": {},
                        "organism_name": {}
                    }
                }
            
            # Execute search
            response = await self.es_client.search(
                index=index,
                body=search_body
            )
            
            self.search_count += 1
            
            # Process results
            hits = response["hits"]["hits"]
            total = response["hits"]["total"]["value"]
            
            results = []
            for hit in hits:
                result = {
                    "id": hit["_id"],
                    "score": hit["_score"],
                    "source": hit["_source"]
                }
                
                # Add highlights if available
                if "highlight" in hit:
                    result["highlights"] = hit["highlight"]
                
                results.append(result)
            
            return {
                "total_hits": total,
                "results": results,
                "query": query,
                "index": index
            }
            
        except Exception as e:
            self.logger.error(f"❌ Search failed: {e}")
            raise ElasticsearchSearchError(f"Search failed: {e}")
    
    async def search_virus_genomes(self, virus_name: str, confidence_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Search virus genomes with specialized matching"""
        try:
            search_body = {
                "query": {
                    "bool": {
                        "should": [
                            # Exact match on genome name
                            {
                                "match": {
                                    "genome_name.keyword": {
                                        "query": virus_name,
                                        "boost": 3.0
                                    }
                                }
                            },
                            # Fuzzy match on genome name
                            {
                                "match": {
                                    "genome_name.fuzzy": {
                                        "query": virus_name,
                                        "fuzziness": "AUTO",
                                        "boost": 2.0
                                    }
                                }
                            },
                            # Match on synonyms and abbreviations
                            {
                                "terms": {
                                    "synonyms": [virus_name],
                                    "boost": 2.5
                                }
                            },
                            {
                                "terms": {
                                    "abbreviations": [virus_name.upper()],
                                    "boost": 2.5
                                }
                            },
                            # Partial match on organism name
                            {
                                "match": {
                                    "organism_name": {
                                        "query": virus_name,
                                        "fuzziness": "AUTO",
                                        "boost": 1.5
                                    }
                                }
                            }
                        ],
                        "minimum_should_match": 1
                    }
                },
                "size": 20,
                "sort": ["_score"],
                "min_score": confidence_threshold
            }
            
            response = await self.es_client.search(
                index=self.es_config.virus_genome_index,
                body=search_body
            )
            
            results = []
            for hit in response["hits"]["hits"]:
                result = hit["_source"]
                result["search_score"] = hit["_score"]
                result["confidence"] = min(hit["_score"] / 10.0, 1.0)  # Normalize to 0-1
                results.append(result)
            
            self.logger.info(f"🔍 Found {len(results)} virus genome matches for '{virus_name}'")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Virus genome search failed: {e}")
            raise ElasticsearchSearchError(f"Virus genome search failed: {e}")
    
    # Tool interface implementation
    
    async def execute_command(self, command: List[str], **kwargs) -> ToolResult:
        """Execute Elasticsearch operations via command interface"""
        # MANDATORY: Ensure tool is initialized before any execution
        await self.ensure_initialized()
        
        if not command:
            raise ElasticsearchSearchError("Empty command provided")
        
        start_time = time.time()
        
        try:
            operation = command[0].lower()
            
            if operation == "search":
                if len(command) < 2:
                    raise ElasticsearchSearchError("Search query required")
                
                query = " ".join(command[1:])
                results = await self.search_documents(query)
                
                return ToolResult(
                    returncode=0,
                    stdout=json.dumps(results, indent=2).encode('utf-8'),
                    stderr=b"",
                    execution_time=time.time() - start_time,
                    command=command,
                    success=True
                )
            
            elif operation == "search_virus":
                if len(command) < 2:
                    raise ElasticsearchSearchError("Virus name required")
                
                virus_name = " ".join(command[1:])
                results = await self.search_virus_genomes(virus_name)
                
                return ToolResult(
                    returncode=0,
                    stdout=json.dumps(results, indent=2).encode('utf-8'),
                    stderr=b"",
                    execution_time=time.time() - start_time,
                    command=command,
                    success=True
                )
            
            elif operation == "index":
                if len(command) < 4:
                    raise ElasticsearchSearchError("Usage: index <index_name> <doc_id> <json_document>")
                
                index_name = command[1]
                doc_id = command[2]
                document = json.loads(" ".join(command[3:]))
                
                success = await self.index_document(index_name, doc_id, document)
                
                return ToolResult(
                    returncode=0 if success else 1,
                    stdout=json.dumps({"success": success, "doc_id": doc_id}).encode('utf-8'),
                    stderr=b"" if success else b"Indexing failed",
                    execution_time=time.time() - start_time,
                    command=command,
                    success=success
                )
            
            elif operation == "health":
                if self.health_monitor:
                    stats = self.health_monitor.get_monitoring_statistics()
                    container_health = self.health_monitor.get_container_health_summary(self.es_config.container_name)
                    
                    health_info = {
                        "container_health": container_health,
                        "monitoring_stats": stats,
                        "search_count": self.search_count,
                        "index_count": self.index_count
                    }
                else:
                    health_info = {
                        "monitoring": "disabled",
                        "search_count": self.search_count,
                        "index_count": self.index_count
                    }
                
                return ToolResult(
                    returncode=0,
                    stdout=json.dumps(health_info, indent=2).encode('utf-8'),
                    stderr=b"",
                    execution_time=time.time() - start_time,
                    command=command,
                    success=True
                )
            
            else:
                raise ElasticsearchSearchError(f"Unknown operation: {operation}")
            
        except Exception as e:
            return ToolResult(
                returncode=1,
                stdout=b"",
                stderr=str(e).encode('utf-8'),
                execution_time=time.time() - start_time,
                command=command,
                success=False
            )
    
    async def parse_output(self, raw_output: str, output_type: str = "json") -> Any:
        """Parse Elasticsearch tool output"""
        if output_type == "json":
            try:
                return json.loads(raw_output)
            except json.JSONDecodeError:
                return {"error": "Invalid JSON output", "raw": raw_output}
        else:
            return raw_output.strip().split('\n')
    
    async def verify_installation(self) -> bool:
        """Verify Elasticsearch installation is functional"""
        try:
            if self.es_client:
                info = await self.es_client.info()
                return True
            return False
        except Exception:
            return False
    
    # Cleanup and shutdown
    
    async def shutdown(self):
        """Clean shutdown of Elasticsearch tool"""
        try:
            self.logger.info("🔄 Shutting down Elasticsearch tool...")
            
            # Stop health monitoring
            if self.health_monitor:
                await self.health_monitor.stop_monitoring()
            
            # Close Elasticsearch client
            if self.es_client:
                await self.es_client.close()
            
            # Optionally stop container (if auto-installed)
            if self.es_config.auto_install_enabled and self.docker_manager:
                await self.docker_manager.stop_container(self.es_config.container_name)
            
            self.logger.info("✅ Elasticsearch tool shutdown complete")
            
        except Exception as e:
            self.logger.error(f"❌ Error during shutdown: {e}")
    
    # Required abstract methods from base class
    
    async def detect_existing_installation(self) -> InstallationStatus:
        """Detect existing Elasticsearch installation"""
        # Check for running container first
        if self.docker_manager:
            try:
                status = await self.docker_manager.get_container_status(self.es_config.container_name)
                if status and status.get("status") == "running":
                    return InstallationStatus(
                        found=True,
                        is_functional=True,
                        installation_path=f"docker://{self.es_config.container_name}",
                        version=self.es_config.image_tag
                    )
            except Exception:
                pass
        
        # Check for standalone Elasticsearch
        try:
            es_url = f"{self.es_config.scheme}://{self.es_config.host}:{self.es_config.port}"
            
            import aiohttp
            timeout = aiohttp.ClientTimeout(total=5)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(f"{es_url}/") as response:
                    if response.status == 200:
                        data = await response.json()
                        version = data.get("version", {}).get("number", "unknown")
                        
                        return InstallationStatus(
                            found=True,
                            is_functional=True,
                            installation_path=es_url,
                            version=version
                        )
        except Exception:
            pass
        
        return InstallationStatus(found=False, is_functional=False)
    
    async def _check_tool_in_directory(self, directory: str) -> bool:
        """Elasticsearch is not a directory-based tool"""
        return False
    
    async def _find_executable_in_path(self) -> Optional[str]:
        """Elasticsearch is not a PATH-based executable"""
        return None
    
    async def _generate_specific_suggestions(self) -> List[str]:
        """Generate Elasticsearch-specific installation suggestions"""
        return [
            f"Enable auto-installation: Set auto_install_enabled=true in configuration",
            f"Install Docker: Elasticsearch auto-installation requires Docker",
            f"Download Elasticsearch manually: https://www.elastic.co/downloads/elasticsearch",
            f"Use Docker: docker run -p 9200:9200 elasticsearch:{self.es_config.image_tag}",
            f"Check network connectivity to {self.es_config.host}:{self.es_config.port}"
        ]
    
    async def _get_alternative_methods(self) -> List[str]:
        """Get alternative installation methods for Elasticsearch"""
        return [
            "Docker auto-installation (recommended)",
            "Manual Docker container setup",
            "Native Elasticsearch installation",
            "Elasticsearch Cloud (hosted service)",
            "Development: Use embedded Elasticsearch alternatives"
        ] 