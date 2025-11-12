#!/usr/bin/env python3
"""
FAISS Database Builder Component

Builds FAISS vector database from document chunks with embeddings
and comprehensive metadata for RAG applications.
"""

import json
import pickle
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

import faiss
from sentence_transformers import SentenceTransformer

from nanobrain.library.components.semantic_chunker import DocumentChunk


class FAISSBuilder:
    """
    Build FAISS vector database with embeddings and metadata.
    
    This component:
    - Generates embeddings using sentence-transformers
    - Builds FAISS index for fast similarity search
    - Stores comprehensive metadata for each chunk
    - Provides database validation and statistics
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            "embedding_model": "sentence-transformers/all-mpnet-base-v2",
            "index_type": "flat",  # or "ivf" for larger datasets
            "batch_size": 32,
            "output_dir": "data/rag_database",
            "validate_embeddings": True
        }
        
        self.config = {**default_config, **(config or {})}

        self.embedding_model_name = self.config["embedding_model"]
        self.index_type = self.config["index_type"]
        self.batch_size = self.config["batch_size"]
        self.output_dir = Path(self.config["output_dir"])
        self.validate_embeddings = self.config["validate_embeddings"]

        self.embedding_model = None
        self.faiss_index = None
        self.metadata = []
        self.embedding_dimension = 768  # all-mpnet-base-v2 default

        self.logger = logging.getLogger("FAISSBuilder")
        self.chunks_processed = 0
        self.embeddings_generated = 0

    async def initialize(self):
        """Initialize the FAISS builder."""
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load embedding model
        self.logger.info(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        self.embedding_dimension = self.embedding_model.get_sentence_embedding_dimension()
        
        self.logger.info(f"FAISSBuilder initialized: model={self.embedding_model_name}, dim={self.embedding_dimension}")
    
    def build_database(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """
        Build FAISS database from document chunks.
        
        Args:
            chunks: List of DocumentChunk objects
            
        Returns:
            Dictionary with build statistics and database info
        """
        try:
            self.logger.info(f"Building FAISS database from {len(chunks)} chunks...")
            
            # Generate embeddings
            embeddings = self._generate_embeddings(chunks)
            
            # Build FAISS index
            self._build_faiss_index(embeddings)
            
            # Prepare metadata
            self._prepare_metadata(chunks)
            
            # Save database
            database_info = self._save_database()
            
            # Validate if requested
            if self.validate_embeddings:
                validation_results = self._validate_database()
                database_info["validation"] = validation_results
            
            self.logger.info(f"FAISS database built successfully: {len(chunks)} chunks, {self.embedding_dimension}D embeddings")
            
            return database_info
            
        except Exception as e:
            self.logger.error(f"Failed to build FAISS database: {e}")
            raise
    
    def _generate_embeddings(self, chunks: List[DocumentChunk]) -> np.ndarray:
        """Generate embeddings for all chunks."""
        self.logger.info("Generating embeddings...")
        
        texts = [chunk.content for chunk in chunks]
        
        # Generate embeddings in batches
        all_embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_embeddings = self.embedding_model.encode(
                batch_texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                show_progress_bar=True
            )
            all_embeddings.append(batch_embeddings)
            
            self.embeddings_generated += len(batch_texts)
            
            if i % (self.batch_size * 10) == 0:
                self.logger.debug(f"Generated embeddings for {i + len(batch_texts)}/{len(texts)} chunks")
        
        embeddings = np.vstack(all_embeddings).astype(np.float32)
        
        self.logger.info(f"Generated {len(embeddings)} embeddings with shape {embeddings.shape}")
        return embeddings
    
    def _build_faiss_index(self, embeddings: np.ndarray):
        """Build FAISS index from embeddings."""
        self.logger.info(f"Building FAISS index: type={self.index_type}")
        
        if self.index_type == "flat":
            # Simple flat index for exact search
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dimension)  # Inner product
        elif self.index_type == "ivf":
            # IVF index for faster approximate search
            nlist = min(100, len(embeddings) // 10)  # Number of clusters
            quantizer = faiss.IndexFlatIP(self.embedding_dimension)
            self.faiss_index = faiss.IndexIVFFlat(quantizer, self.embedding_dimension, nlist)
            
            # Train the index
            self.logger.info("Training IVF index...")
            self.faiss_index.train(embeddings)
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
        
        # Add embeddings to index
        self.logger.info("Adding embeddings to FAISS index...")
        self.faiss_index.add(embeddings)
        
        self.logger.info(f"FAISS index built: {self.faiss_index.ntotal} vectors")
    
    def _prepare_metadata(self, chunks: List[DocumentChunk]):
        """Prepare metadata for each chunk."""
        self.logger.info("Preparing metadata...")
        
        self.metadata = []
        for chunk in chunks:
            chunk_metadata = {
                "chunk_id": chunk.chunk_id,
                "paper_doi": chunk.paper_doi,
                "paper_title": chunk.paper_title,
                "section_name": chunk.section_name,
                "content": chunk.content,
                "chunk_index": chunk.chunk_index,
                "total_chunks": chunk.total_chunks,
                "char_start": chunk.char_start,
                "char_end": chunk.char_end,
                "authors": chunk.metadata.get("authors", []),
                "file_path": chunk.metadata.get("file_path", ""),
                "char_count": len(chunk.content),
                "created_at": datetime.now().isoformat()
            }
            self.metadata.append(chunk_metadata)
        
        self.chunks_processed = len(chunks)
        self.logger.info(f"Prepared metadata for {len(self.metadata)} chunks")
    
    def _save_database(self) -> Dict[str, Any]:
        """Save FAISS index and metadata to disk."""
        self.logger.info(f"Saving database to {self.output_dir}")
        
        # Save FAISS index
        index_path = self.output_dir / "faiss_index.bin"
        faiss.write_index(self.faiss_index, str(index_path))
        
        # Save metadata
        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        # Save configuration
        config_path = self.output_dir / "config.json"
        config_info = {
            "embedding_model": self.embedding_model_name,
            "embedding_dimension": self.embedding_dimension,
            "index_type": self.index_type,
            "total_chunks": len(self.metadata),
            "created_at": datetime.now().isoformat(),
            "database_version": "1.0"
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_info, f, indent=2)
        
        database_info = {
            "index_path": str(index_path),
            "metadata_path": str(metadata_path),
            "config_path": str(config_path),
            "total_chunks": len(self.metadata),
            "embedding_dimension": self.embedding_dimension,
            "index_size_mb": index_path.stat().st_size / (1024 * 1024),
            "metadata_size_mb": metadata_path.stat().st_size / (1024 * 1024)
        }
        
        self.logger.info(f"Database saved: {database_info['total_chunks']} chunks, {database_info['index_size_mb']:.1f}MB index")
        
        return database_info
    
    def _validate_database(self) -> Dict[str, Any]:
        """Validate the built database."""
        self.logger.info("Validating database...")
        
        # Test search functionality
        test_query = "machine learning neural networks"
        test_embedding = self.embedding_model.encode([test_query])
        
        # Perform test search
        similarities, indices = self.faiss_index.search(test_embedding, k=5)
        
        validation_results = {
            "index_total": self.faiss_index.ntotal,
            "metadata_count": len(self.metadata),
            "consistency_check": self.faiss_index.ntotal == len(self.metadata),
            "test_search_results": len(indices[0]),
            "sample_similarities": similarities[0].tolist()[:3],
            "validation_passed": True
        }
        
        if not validation_results["consistency_check"]:
            validation_results["validation_passed"] = False
            self.logger.warning("Validation failed: index and metadata count mismatch")
        
        self.logger.info(f"Validation completed: {'PASSED' if validation_results['validation_passed'] else 'FAILED'}")
        
        return validation_results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get builder statistics."""
        return {
            "chunks_processed": self.chunks_processed,
            "embeddings_generated": self.embeddings_generated,
            "embedding_dimension": self.embedding_dimension,
            "index_type": self.index_type,
            "model_name": self.embedding_model_name
        }
