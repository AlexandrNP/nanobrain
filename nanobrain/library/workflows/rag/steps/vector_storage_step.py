#!/usr/bin/env python3
"""
Vector Storage Step for RAG systems.
Deterministic vector database operations for storing and retrieving embeddings.
"""

import asyncio
import logging
import json
import pickle
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from nanobrain.core.step import BaseStep, StepConfig
from nanobrain.core.logging_system import get_logger

logger = get_logger(__name__)


class VectorStorageStep(BaseStep):
    """
    Deterministic vector storage step.
    
    Handles vector database operations for storing and retrieving embeddings
    without LLM processing. Supports multiple vector database backends.
    
    Key Features:
    - Multi-database support (FAISS, Pinecone, Weaviate, Chroma)
    - Persistent storage with index management
    - Batch insertion with performance optimization
    - Database-specific configuration handling
    """
    
    COMPONENT_TYPE = "vector_storage_step"
    
    def __init__(self, *args, **kwargs):
        """Prevent direct instantiation - use from_config instead"""
        raise RuntimeError(
            "Direct instantiation of VectorStorageStep is prohibited. "
            "ALL framework components must use VectorStorageStep.from_config() "
            "as per mandatory framework requirements."
        )

    def _init_from_config(self, config: StepConfig, component_config: Dict[str, Any],
                         dependencies: Dict[str, Any]) -> None:
        """Initialize VectorStorageStep from configuration."""
        # Call parent initialization
        super()._init_from_config(config, component_config, dependencies)

        # Vector database configuration
        self.vector_db_type = component_config.get('vector_db_type', 'faiss')
        self.index_type = component_config.get('index_type', 'IVF')
        self.dimension = component_config.get('dimension', 384)  # Default for sentence-transformers
        self.enable_persistence = component_config.get('enable_persistence', True)
        self.persistence_path = Path(component_config.get('persistence_path', 'data/rag_index'))

        # Performance settings
        self.batch_insert_size = component_config.get('batch_insert_size', 1000)
        self.enable_gpu = component_config.get('enable_gpu', False)
        self.memory_map = component_config.get('memory_map', True)
        
        # Database-specific settings
        self.db_config = self._extract_db_config()
        
        # Initialize vector database
        self.vector_db = self._initialize_vector_database()
        
        # Create persistence directory
        if self.enable_persistence:
            self.persistence_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🗄️ VectorStorageStep initialized with {self.vector_db_type} database")

    def _extract_db_config(self) -> Dict[str, Any]:
        """Extract database-specific configuration."""
        db_config = {}

        if self.vector_db_type == 'faiss':
            db_config = {
                'nlist': 100,  # Default values for now
                'nprobe': 10,
                'metric': 'L2',
                'enable_gpu': self.enable_gpu
            }
        elif self.vector_db_type == 'pinecone':
            db_config = {
                'api_key': None,  # Would be loaded from environment or config
                'environment': None,
                'index_name': 'rag-index'
            }
        elif self.vector_db_type == 'weaviate':
            db_config = {
                'url': 'http://localhost:8080',
                'api_key': None,
                'class_name': 'Document'
            }
        elif self.vector_db_type == 'chroma':
            db_config = {
                'persist_directory': self.config.get('chroma_config', {}).get('persist_directory'),
                'collection_name': self.config.get('chroma_config', {}).get('collection_name', 'rag_collection')
            }
        
        return db_config
    
    def _initialize_vector_database(self) -> Any:
        """Initialize vector database based on type."""
        # Simple mock database for demonstration
        # In full implementation, would use actual database clients
        class MockVectorDatabase:
            def __init__(self, db_type=None, **kwargs):
                self.db_type = db_type or 'faiss'
                self.vectors = {}
                self.index = None
                self.config = kwargs

            async def initialize(self):
                """Initialize vector database."""
                try:
                    # In full implementation, would load existing index
                    logger.info(f"📊 Initialized vector database with {len(self.vectors)} vectors")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load index: {e}")
                    await self.create_index()
            
            async def create_index(self):
                """Create new index."""
                self.vectors = []
                self.metadata = []
                self.ids = []
                self.index_loaded = True
                logger.info("📂 Created new vector index")
            
            async def insert_vectors(self, vectors: List[List[float]], 
                                   metadata_list: List[Dict[str, Any]], 
                                   ids: List[str]):
                """Insert vectors into the database."""
                self.vectors.extend(vectors)
                self.metadata.extend(metadata_list)
                self.ids.extend(ids)
                logger.debug(f"📥 Inserted {len(vectors)} vectors")
            
            async def save_index(self):
                """Save index to persistent storage."""
                index_file = self.persistence_path / "vector_index.pkl"
                data = {
                    'vectors': self.vectors,
                    'metadata': self.metadata,
                    'ids': self.ids
                }
                with open(index_file, 'wb') as f:
                    pickle.dump(data, f)
                logger.info(f"💾 Saved index with {len(self.vectors)} vectors")
            
            async def get_index_info(self) -> Dict[str, Any]:
                """Get information about the vector index."""
                return {
                    'total_vectors': len(self.vectors),
                    'dimension': self.dimension,
                    'database_type': self.db_type,
                    'index_loaded': self.index_loaded
                }
            
            async def search(self, query_vector: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
                """Search for similar vectors."""
                if not self.vectors:
                    return []
                
                # Simple cosine similarity search
                import math
                
                def cosine_similarity(a, b):
                    dot_product = sum(x * y for x, y in zip(a, b))
                    norm_a = math.sqrt(sum(x * x for x in a))
                    norm_b = math.sqrt(sum(x * x for x in b))
                    if norm_a == 0 or norm_b == 0:
                        return 0
                    return dot_product / (norm_a * norm_b)
                
                # Calculate similarities
                similarities = []
                for i, vector in enumerate(self.vectors):
                    similarity = cosine_similarity(query_vector, vector)
                    similarities.append({
                        'index': i,
                        'similarity': similarity,
                        'metadata': self.metadata[i],
                        'id': self.ids[i]
                    })
                
                # Sort by similarity and return top_k
                similarities.sort(key=lambda x: x['similarity'], reverse=True)
                return similarities[:top_k]
        
        base_config = {
            'dimension': self.dimension,
            'batch_size': self.batch_insert_size,
            'persistence_path': self.persistence_path if self.enable_persistence else None
        }
        
        return MockVectorDatabase(self.vector_db_type, **base_config, **self.db_config)
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Store embeddings in vector database.
        
        Args:
            input_data: Contains embeddings from EmbeddingGeneratorStep
            
        Returns:
            Dictionary with storage results and index information
        """
        try:
            start_time = asyncio.get_event_loop().time()
            
            # Extract embeddings from input
            embeddings = input_data.get('embeddings', [])
            
            if not embeddings:
                logger.warning("⚠️ No embeddings provided for storage")
                return self._create_empty_result()
            
            logger.info(f"🗄️ Storing {len(embeddings)} embeddings in {self.vector_db_type} database")
            
            # Load existing index if available
            await self._load_existing_index()
            
            # Prepare embeddings for storage
            vectors, metadata_list, ids = self._prepare_embeddings_for_storage(embeddings)
            
            # Store embeddings in batches
            storage_stats = await self._store_embeddings_batch(vectors, metadata_list, ids)
            
            # Save index if persistence enabled
            if self.enable_persistence:
                await self._save_index()
            
            processing_time = asyncio.get_event_loop().time() - start_time
            
            # Generate storage statistics
            storage_stats.update({
                'processing_time': processing_time,
                'storage_rate': len(embeddings) / processing_time if processing_time > 0 else 0
            })
            
            # Get index information
            index_info = await self._get_index_info()
            
            result = {
                'storage_stats': storage_stats,
                'index_info': index_info,
                'database_type': self.vector_db_type,
                'persistence_enabled': self.enable_persistence,
                'vector_db': self.vector_db  # Pass database reference for retrieval step
            }
            
            logger.info(f"✅ Vector storage completed: {len(embeddings)} embeddings stored in {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Vector storage failed: {e}")
            return self._create_error_result(str(e))

    async def _load_existing_index(self) -> None:
        """Load existing index if available."""
        if not self.enable_persistence:
            return

        try:
            await self.vector_db.load_index()
            logger.info("📂 Loaded existing vector index")
        except Exception as e:
            logger.info(f"📂 No existing index found, creating new one: {e}")
            await self.vector_db.create_index()

    def _prepare_embeddings_for_storage(self, embeddings: List[Dict[str, Any]]) -> Tuple[List[List[float]], List[Dict[str, Any]], List[str]]:
        """Prepare embeddings for storage in vector database."""
        vectors = []
        metadata_list = []
        ids = []

        for embedding in embeddings:
            vectors.append(embedding['vector'])
            metadata_list.append(embedding['metadata'])
            ids.append(embedding['chunk_id'])

        return vectors, metadata_list, ids

    async def _store_embeddings_batch(self, vectors: List[List[float]],
                                    metadata_list: List[Dict[str, Any]],
                                    ids: List[str]) -> Dict[str, Any]:
        """Store embeddings in batches."""
        total_stored = 0
        failed_insertions = 0

        # Process in batches
        for i in range(0, len(vectors), self.batch_insert_size):
            batch_vectors = vectors[i:i + self.batch_insert_size]
            batch_metadata = metadata_list[i:i + self.batch_insert_size]
            batch_ids = ids[i:i + self.batch_insert_size]

            try:
                await self.vector_db.insert_vectors(batch_vectors, batch_metadata, batch_ids)
                total_stored += len(batch_vectors)

                logger.info(f"🗄️ Stored batch {i//self.batch_insert_size + 1}: {len(batch_vectors)} vectors")

            except Exception as e:
                logger.error(f"❌ Failed to store batch {i//self.batch_insert_size + 1}: {e}")
                failed_insertions += len(batch_vectors)

        return {
            'total_embeddings': len(vectors),
            'successfully_stored': total_stored,
            'failed_insertions': failed_insertions,
            'success_rate': total_stored / len(vectors) if vectors else 0
        }

    async def _save_index(self) -> None:
        """Save index to persistent storage."""
        try:
            await self.vector_db.save_index()
            logger.info("💾 Vector index saved to persistent storage")
        except Exception as e:
            logger.error(f"❌ Failed to save vector index: {e}")

    async def _get_index_info(self) -> Dict[str, Any]:
        """Get information about the vector index."""
        try:
            return await self.vector_db.get_index_info()
        except Exception as e:
            logger.warning(f"⚠️ Failed to get index info: {e}")
            return {}

    def _create_empty_result(self) -> Dict[str, Any]:
        """Create empty result when no embeddings provided."""
        return {
            'storage_stats': {
                'total_embeddings': 0,
                'successfully_stored': 0,
                'failed_insertions': 0,
                'success_rate': 0,
                'processing_time': 0,
                'storage_rate': 0
            },
            'index_info': {},
            'database_type': self.vector_db_type,
            'persistence_enabled': self.enable_persistence
        }

    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create error result."""
        return {
            'storage_stats': {
                'total_embeddings': 0,
                'successfully_stored': 0,
                'failed_insertions': 0,
                'success_rate': 0,
                'processing_time': 0,
                'storage_rate': 0,
                'error': error_message
            },
            'index_info': {},
            'database_type': self.vector_db_type,
            'persistence_enabled': self.enable_persistence,
            'error': error_message
        }
