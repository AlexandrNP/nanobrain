#!/usr/bin/env python3
"""
Elasticsearch Indexing Step for Viral Protein Analysis

This step indexes viral protein data and analysis results into Elasticsearch
via the MCP (Model Context Protocol) server, enabling powerful search and
analytics capabilities for the NanoBrain chatbot viral integration workflow.

Key Features:
- Indexes protein sequences with metadata
- Stores analysis results for context retrieval
- Supports bulk operations for high throughput
- Integrates with MCP client for standardized access
- Handles embedding generation for semantic search
- Provides error handling and retry logic

Integration Points:
- Receives protein data from BV-BRC acquisition steps
- Indexes processed analysis results
- Enables search functionality for chatbot responses
- Supports real-time indexing during workflow execution

Usage:
    # Configure in workflow YAML
    - name: "elasticsearch_indexing"
      class: "ElasticsearchIndexingStep"
      config:
        mcp_server_url: "http://elasticsearch-mcp:9202"
        indices:
          proteins: "viral_proteins"
          results: "analysis_results"
"""

import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field

# NanoBrain imports
from nanobrain.core.step import BaseStep, StepConfig
from nanobrain.core.data_unit import DataUnit
from nanobrain.core.mcp_support import MCPClient, MCPClientConfig, MCPServerConfig
from nanobrain.core.logging_system import get_logger


@dataclass
class ElasticsearchIndexingConfig(StepConfig):
    """Configuration for Elasticsearch indexing step"""
    
    # Step identification
    step_name: str = "elasticsearch_indexing"
    
    # MCP server configuration
    mcp_server_url: str = "http://localhost:9202"
    mcp_server_name: str = "elasticsearch"
    connection_timeout: float = 30.0
    request_timeout: float = 60.0
    
    # Index configuration
    indices: Dict[str, str] = field(default_factory=lambda: {
        "proteins": "viral_proteins",
        "results": "analysis_results",
        "interactions": "chat_interactions"
    })
    
    # Indexing options
    bulk_size: int = 100
    enable_embeddings: bool = True
    embedding_field: str = "embedding"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Retry configuration
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Data processing
    include_metadata: bool = True
    timestamp_field: str = "indexed_at"
    
    # Tool card (mandatory)
    tool_card: Dict[str, Any] = field(default_factory=lambda: {
        "name": "elasticsearch_indexing_step",
        "description": "Indexes viral protein data into Elasticsearch via MCP",
        "version": "1.0.0",
        "category": "data_processing",
        "capabilities": ["indexing", "search_preparation", "mcp_integration"]
    })


class ElasticsearchIndexingStep(BaseStep):
    """
    Step to index viral protein data and analysis results into Elasticsearch
    via MCP server integration.
    """
    
    @classmethod
    def from_config(cls, config: Union[ElasticsearchIndexingConfig, Dict], **kwargs) -> 'ElasticsearchIndexingStep':
        """Mandatory from_config implementation"""
        logger = get_logger(f"{cls.__name__}.from_config")
        logger.info(f"Creating {cls.__name__} from configuration")
        
        # Convert to ElasticsearchIndexingConfig if needed
        if isinstance(config, dict):
            config = ElasticsearchIndexingConfig(**config)
        elif not isinstance(config, ElasticsearchIndexingConfig):
            # Convert other config types
            config_dict = config.model_dump() if hasattr(config, 'model_dump') else {}
            config = ElasticsearchIndexingConfig(**config_dict)
        
        # Validate tool card
        if not hasattr(config, 'tool_card') or not config.tool_card:
            raise ValueError(f"Missing mandatory 'tool_card' in configuration for {cls.__name__}")
        
        instance = cls(config, **kwargs)
        instance._tool_card_data = config.tool_card
        
        logger.info(f"Successfully created {cls.__name__} with MCP integration")
        return instance
    
    def __init__(self, config: ElasticsearchIndexingConfig, **kwargs):
        """Initialize Elasticsearch indexing step"""
        super().__init__(config, **kwargs)
        
        self.config = config
        self.logger = get_logger(f"elasticsearch_indexing_step")
        
        # MCP client for Elasticsearch communication
        self.mcp_client: Optional[MCPClient] = None
        
        # Embedding model (lazy loaded)
        self.embedding_model = None
        
        # Statistics
        self.indexed_proteins = 0
        self.indexed_results = 0
        self.indexing_errors = 0
    
    async def initialize(self):
        """Initialize the step and MCP client"""
        await super().initialize()
        
        try:
            # Initialize MCP client
            mcp_client_config = MCPClientConfig(
                default_timeout=self.config.connection_timeout,
                default_max_retries=self.config.max_retries,
                default_retry_delay=self.config.retry_delay
            )
            
            self.mcp_client = MCPClient(mcp_client_config, logger=self.logger)
            await self.mcp_client.initialize()
            
            # Add Elasticsearch MCP server
            elasticsearch_server_config = MCPServerConfig(
                name=self.config.mcp_server_name,
                url=self.config.mcp_server_url,
                description="Elasticsearch MCP server for viral protein analysis",
                timeout=self.config.request_timeout
            )
            
            self.mcp_client.add_server(elasticsearch_server_config)
            
            # Connect to server
            connected = await self.mcp_client.connect_to_server(self.config.mcp_server_name)
            if not connected:
                raise RuntimeError(f"Failed to connect to Elasticsearch MCP server at {self.config.mcp_server_url}")
            
            # Initialize embedding model if needed
            if self.config.enable_embeddings:
                await self._initialize_embedding_model()
            
            self.logger.info("✅ Elasticsearch indexing step initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Elasticsearch indexing step: {e}")
            raise
    
    async def _initialize_embedding_model(self):
        """Initialize embedding model for semantic search"""
        try:
            # Try to import sentence-transformers
            from sentence_transformers import SentenceTransformer
            
            self.embedding_model = SentenceTransformer(self.config.embedding_model)
            self.logger.info(f"✅ Loaded embedding model: {self.config.embedding_model}")
            
        except ImportError:
            self.logger.warning("⚠️ sentence-transformers not available, disabling embeddings")
            self.config.enable_embeddings = False
        except Exception as e:
            self.logger.error(f"❌ Failed to load embedding model: {e}")
            self.config.enable_embeddings = False
    
    async def execute(self, data_unit: DataUnit) -> DataUnit:
        """Execute the indexing step"""
        self.logger.info("🔄 Starting Elasticsearch indexing...")
        
        try:
            # Ensure step is initialized
            await self.ensure_initialized()
            
            # Index protein sequences if available
            if hasattr(data_unit, 'protein_sequences') and data_unit.protein_sequences:
                await self._index_protein_sequences(data_unit.protein_sequences, data_unit)
            
            # Index analysis results if available
            if hasattr(data_unit, 'analysis_results') and data_unit.analysis_results:
                await self._index_analysis_results(data_unit.analysis_results, data_unit)
            
            # Index workflow metadata
            await self._index_workflow_metadata(data_unit)
            
            # Add indexing statistics to data unit
            data_unit.elasticsearch_indexing = {
                "indexed_proteins": self.indexed_proteins,
                "indexed_results": self.indexed_results,
                "indexing_errors": self.indexing_errors,
                "indices_used": list(self.config.indices.values()),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            self.logger.info(f"✅ Elasticsearch indexing completed")
            self.logger.info(f"   Proteins indexed: {self.indexed_proteins}")
            self.logger.info(f"   Results indexed: {self.indexed_results}")
            self.logger.info(f"   Errors: {self.indexing_errors}")
            
            return data_unit
            
        except Exception as e:
            self.logger.error(f"❌ Elasticsearch indexing failed: {e}")
            self.indexing_errors += 1
            
            # Add error information to data unit
            data_unit.elasticsearch_indexing = {
                "success": False,
                "error": str(e),
                "indexed_proteins": self.indexed_proteins,
                "indexed_results": self.indexed_results,
                "indexing_errors": self.indexing_errors
            }
            
            # Don't fail the entire workflow - continue with warning
            self.logger.warning("⚠️ Continuing workflow despite indexing failure")
            return data_unit
    
    async def _index_protein_sequences(self, protein_sequences: List[Dict], data_unit: DataUnit):
        """Index protein sequences with embeddings"""
        if not protein_sequences:
            return
        
        self.logger.info(f"🔄 Indexing {len(protein_sequences)} protein sequences...")
        
        try:
            # Prepare documents for bulk indexing
            documents = []
            
            for protein in protein_sequences:
                doc = self._prepare_protein_document(protein, data_unit)
                documents.append(doc)
            
            # Bulk index in batches
            await self._bulk_index_documents(
                self.config.indices["proteins"],
                documents,
                "protein sequences"
            )
            
            self.indexed_proteins += len(documents)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to index protein sequences: {e}")
            self.indexing_errors += 1
            raise
    
    async def _index_analysis_results(self, analysis_results: Dict, data_unit: DataUnit):
        """Index analysis results"""
        self.logger.info("🔄 Indexing analysis results...")
        
        try:
            # Prepare analysis document
            doc = self._prepare_analysis_document(analysis_results, data_unit)
            
            # Index single document
            result = await self.mcp_client.call_tool(
                self.config.mcp_server_name,
                "index_document",
                {
                    "index": self.config.indices["results"],
                    "document": doc,
                    "id": doc.get("analysis_id")
                }
            )
            
            if result.get("indexed"):
                self.indexed_results += 1
                self.logger.info("✅ Analysis results indexed successfully")
            else:
                self.logger.warning("⚠️ Analysis results indexing returned unexpected result")
                self.indexing_errors += 1
                
        except Exception as e:
            self.logger.error(f"❌ Failed to index analysis results: {e}")
            self.indexing_errors += 1
            raise
    
    async def _index_workflow_metadata(self, data_unit: DataUnit):
        """Index workflow execution metadata"""
        try:
            # Prepare workflow metadata document
            doc = {
                "workflow_id": getattr(data_unit, 'workflow_id', str(uuid.uuid4())),
                "execution_id": getattr(data_unit, 'execution_id', str(uuid.uuid4())),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "step_name": self.config.step_name,
                "data_unit_type": type(data_unit).__name__,
                "processing_stats": {
                    "proteins_processed": getattr(data_unit, 'protein_count', 0),
                    "analysis_completed": hasattr(data_unit, 'analysis_results'),
                    "indexing_attempted": True
                }
            }
            
            # Add workflow-specific metadata if available
            if hasattr(data_unit, 'workflow_metadata'):
                doc["workflow_metadata"] = data_unit.workflow_metadata
            
            # Index metadata
            await self.mcp_client.call_tool(
                self.config.mcp_server_name,
                "index_document",
                {
                    "index": self.config.indices["interactions"],
                    "document": doc
                }
            )
            
            self.logger.debug("✅ Workflow metadata indexed")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to index workflow metadata: {e}")
            # Don't raise - metadata indexing is not critical
    
    def _prepare_protein_document(self, protein: Dict, data_unit: DataUnit) -> Dict[str, Any]:
        """Prepare a protein document for indexing"""
        doc = {
            "protein_id": protein.get("protein_id", protein.get("patric_id", str(uuid.uuid4()))),
            "sequence": protein.get("sequence", protein.get("aa_sequence", "")),
            "organism": protein.get("organism", protein.get("genome_id", "unknown")),
            "function": protein.get("function", protein.get("product", "")),
            "md5_hash": protein.get("md5_hash", protein.get("aa_sequence_md5", "")),
            self.config.timestamp_field: datetime.now(timezone.utc).isoformat()
        }
        
        # Add metadata if enabled
        if self.config.include_metadata:
            doc["metadata"] = {
                "workflow_id": getattr(data_unit, 'workflow_id', None),
                "execution_id": getattr(data_unit, 'execution_id', None),
                "source": "bv_brc",
                "indexed_by": self.config.step_name
            }
            
            # Add any additional protein metadata
            for key, value in protein.items():
                if key not in ["protein_id", "sequence", "organism", "function", "md5_hash"]:
                    doc["metadata"][key] = value
        
        # Generate embedding if enabled
        if self.config.enable_embeddings and self.embedding_model and doc["sequence"]:
            try:
                # Generate embedding for the protein sequence
                embedding = self.embedding_model.encode(doc["sequence"])
                doc[self.config.embedding_field] = embedding.tolist()
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to generate embedding for protein {doc['protein_id']}: {e}")
        
        return doc
    
    def _prepare_analysis_document(self, analysis_results: Dict, data_unit: DataUnit) -> Dict[str, Any]:
        """Prepare an analysis results document for indexing"""
        doc = {
            "analysis_id": analysis_results.get("analysis_id", str(uuid.uuid4())),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "results": analysis_results,
            "protein_ids": self._extract_protein_ids(analysis_results),
            "confidence_score": analysis_results.get("confidence_score", 0.0),
            "analysis_type": analysis_results.get("analysis_type", "viral_protein_analysis")
        }
        
        # Add workflow context
        if self.config.include_metadata:
            doc["metadata"] = {
                "workflow_id": getattr(data_unit, 'workflow_id', None),
                "execution_id": getattr(data_unit, 'execution_id', None),
                "step_name": self.config.step_name,
                "data_unit_type": type(data_unit).__name__
            }
        
        return doc
    
    def _extract_protein_ids(self, analysis_results: Dict) -> List[str]:
        """Extract protein IDs from analysis results"""
        protein_ids = []
        
        # Try different possible locations for protein IDs
        if "proteins" in analysis_results:
            proteins = analysis_results["proteins"]
            if isinstance(proteins, list):
                for protein in proteins:
                    if isinstance(protein, dict):
                        pid = protein.get("protein_id") or protein.get("patric_id")
                        if pid:
                            protein_ids.append(pid)
                    elif isinstance(protein, str):
                        protein_ids.append(protein)
        
        if "protein_ids" in analysis_results:
            ids = analysis_results["protein_ids"]
            if isinstance(ids, list):
                protein_ids.extend(ids)
            elif isinstance(ids, str):
                protein_ids.append(ids)
        
        return list(set(protein_ids))  # Remove duplicates
    
    async def _bulk_index_documents(self, index: str, documents: List[Dict], doc_type: str):
        """Bulk index documents with batching"""
        if not documents:
            return
        
        total_docs = len(documents)
        batch_size = self.config.bulk_size
        
        for i in range(0, total_docs, batch_size):
            batch = documents[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total_docs + batch_size - 1) // batch_size
            
            self.logger.info(f"🔄 Indexing batch {batch_num}/{total_batches} ({len(batch)} {doc_type})...")
            
            try:
                result = await self.mcp_client.call_tool(
                    self.config.mcp_server_name,
                    "bulk_index",
                    {
                        "index": index,
                        "documents": batch
                    }
                )
                
                if result.get("bulk_indexed"):
                    indexed_count = result.get("indexed_count", 0)
                    errors_count = result.get("errors_count", 0)
                    
                    self.logger.info(f"✅ Batch {batch_num} completed: {indexed_count} indexed, {errors_count} errors")
                    
                    if errors_count > 0:
                        self.indexing_errors += errors_count
                        errors = result.get("errors", [])
                        for error in errors[:3]:  # Log first 3 errors
                            self.logger.warning(f"   Error: {error}")
                else:
                    self.logger.warning(f"⚠️ Batch {batch_num} returned unexpected result")
                    self.indexing_errors += len(batch)
                
            except Exception as e:
                self.logger.error(f"❌ Failed to index batch {batch_num}: {e}")
                self.indexing_errors += len(batch)
                
                # Retry with smaller batch if this was a large batch
                if len(batch) > 10:
                    self.logger.info("🔄 Retrying with smaller batches...")
                    await self._bulk_index_documents(index, batch, doc_type)
                else:
                    raise
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.mcp_client:
                await self.mcp_client.shutdown()
                self.mcp_client = None
            
            self.logger.info("✅ Elasticsearch indexing step cleanup completed")
            
        except Exception as e:
            self.logger.error(f"❌ Error during cleanup: {e}")
        
        await super().cleanup()
    
    # Step interface methods
    
    def get_step_info(self) -> Dict[str, Any]:
        """Get step information"""
        return {
            "name": self.config.step_name,
            "type": "elasticsearch_indexing",
            "description": "Indexes viral protein data into Elasticsearch via MCP",
            "mcp_server": self.config.mcp_server_url,
            "indices": self.config.indices,
            "embeddings_enabled": self.config.enable_embeddings,
            "statistics": {
                "indexed_proteins": self.indexed_proteins,
                "indexed_results": self.indexed_results,
                "indexing_errors": self.indexing_errors
            }
        }
    
    def validate_configuration(self) -> bool:
        """Validate step configuration"""
        try:
            # Check required configuration
            if not self.config.mcp_server_url:
                self.logger.error("❌ MCP server URL not configured")
                return False
            
            if not self.config.indices:
                self.logger.error("❌ No indices configured")
                return False
            
            # Check MCP client availability
            if not self.mcp_client:
                self.logger.warning("⚠️ MCP client not initialized")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Configuration validation failed: {e}")
            return False 