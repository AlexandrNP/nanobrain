#!/usr/bin/env python3
"""
Shared Vector Database for Parallel RAG Demo
=============================================

Demonstrates the @shared decorator for vector databases that are accessed
by multiple PARSL workers concurrently.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
import numpy as np

from nanobrain.core.shared_resource import shared, get_worker_id


logger = logging.getLogger(__name__)


@shared(resource_type='vector_database', auto_register=True)
class SharedVectorDatabase:
    """
    Vector database that can be safely accessed by multiple PARSL workers.
    
    Features:
    - Marked with @shared decorator for automatic pooling
    - Thread-safe access to vector index
    - Worker ID tracking for all operations
    - Single initialization, multiple concurrent accesses
    """
    
    def __init__(self, dimension: int = 384, index_type: str = 'flat'):
        """
        Initialize the shared vector database.
        
        Args:
            dimension: Vector dimension
            index_type: Type of index ('flat', 'ivf', etc.)
        """
        self.dimension = dimension
        self.index_type = index_type
        self.vectors = {}
        self.metadata = {}
        self.index = None
        self.is_initialized = False
        
        # Access tracking
        self.access_count = 0
        self.worker_access_log = []
        
        logger.info(f"🔧 SharedVectorDatabase created (dimension={dimension}, type={index_type})")
    
    async def initialize(self):
        """
        Initialize the vector database.
        
        This is called once and the initialized database is shared
        across all workers.
        """
        if self.is_initialized:
            logger.info("✅ Vector database already initialized (shared resource)")
            return
        
        logger.info("🔧 Initializing shared vector database...")
        
        # Simulate index building
        await asyncio.sleep(0.1)
        
        # Create simple flat index (in production, would use FAISS, etc.)
        self.index = {
            'type': self.index_type,
            'dimension': self.dimension,
            'vectors': self.vectors,
            'metadata': self.metadata
        }
        
        self.is_initialized = True
        
        logger.info(f"✅ Shared vector database initialized")
        logger.info(f"   Resource ID: {self.get_resource_id()}")
        logger.info(f"   Dimension: {self.dimension}")
        logger.info(f"   Index type: {self.index_type}")
    
    async def add_vectors(
        self,
        vectors: List[List[float]],
        metadata: List[Dict[str, Any]],
        worker_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Add vectors to the database.
        
        Args:
            vectors: List of vectors to add
            metadata: Metadata for each vector
            worker_id: ID of the worker adding vectors
            
        Returns:
            Result with worker ID tracking
        """
        if not self.is_initialized:
            await self.initialize()
        
        # Get worker ID if not provided
        if worker_id is None:
            worker_id = get_worker_id()
        
        # Track access
        self.access_count += 1
        self.worker_access_log.append({
            'worker_id': worker_id,
            'operation': 'add_vectors',
            'count': len(vectors)
        })
        
        # Add vectors
        start_id = len(self.vectors)
        for i, (vector, meta) in enumerate(zip(vectors, metadata)):
            vector_id = start_id + i
            self.vectors[vector_id] = vector
            self.metadata[vector_id] = meta
        
        logger.info(f"📦 Worker {worker_id} added {len(vectors)} vectors "
                   f"(total: {len(self.vectors)})")
        
        return {
            'added_count': len(vectors),
            'total_vectors': len(self.vectors),
            'worker_id': worker_id,
            'resource_id': self.get_resource_id()
        }
    
    async def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        worker_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: Query vector
            top_k: Number of results to return
            worker_id: ID of the worker performing search
            
        Returns:
            Search results with worker ID tracking
        """
        if not self.is_initialized:
            await self.initialize()
        
        # Get worker ID if not provided
        if worker_id is None:
            worker_id = get_worker_id()
        
        # Track access
        self.access_count += 1
        self.worker_access_log.append({
            'worker_id': worker_id,
            'operation': 'search',
            'top_k': top_k
        })
        
        # Perform search (simple cosine similarity)
        results = []
        query_np = np.array(query_vector)
        
        for vector_id, vector in self.vectors.items():
            vector_np = np.array(vector)
            
            # Cosine similarity
            similarity = np.dot(query_np, vector_np) / (
                np.linalg.norm(query_np) * np.linalg.norm(vector_np) + 1e-10
            )
            
            results.append({
                'id': vector_id,
                'similarity': float(similarity),
                'metadata': self.metadata.get(vector_id, {})
            })
        
        # Sort by similarity and return top_k
        results.sort(key=lambda x: x['similarity'], reverse=True)
        results = results[:top_k]
        
        logger.info(f"🔍 Worker {worker_id} searched and found {len(results)} results")
        
        return {
            'results': results,
            'query_vector_dim': len(query_vector),
            'total_vectors_searched': len(self.vectors),
            'worker_id': worker_id,
            'resource_id': self.get_resource_id(),
            'access_count': self.access_count
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the shared vector database.
        
        Returns:
            Dictionary with database statistics
        """
        # Get shared resource stats
        resource_stats = self.get_resource_stats()
        
        # Combine with database stats
        return {
            'database_stats': {
                'total_vectors': len(self.vectors),
                'dimension': self.dimension,
                'index_type': self.index_type,
                'is_initialized': self.is_initialized,
                'access_count': self.access_count,
                'unique_workers': len(set(log['worker_id'] for log in self.worker_access_log))
            },
            'shared_resource_stats': resource_stats,
            'recent_accesses': self.worker_access_log[-10:]  # Last 10 accesses
        }
    
    async def shutdown(self):
        """Shutdown the vector database and unregister from pool."""
        logger.info(f"🗑️  Shutting down shared vector database {self.get_resource_id()}")
        
        # Print final stats
        stats = self.get_stats()
        logger.info(f"   Total accesses: {stats['database_stats']['access_count']}")
        logger.info(f"   Unique workers: {stats['database_stats']['unique_workers']}")
        logger.info(f"   Total vectors: {stats['database_stats']['total_vectors']}")
        
        # Unregister from pool
        self.unregister()
        
        # Clear data
        self.vectors.clear()
        self.metadata.clear()
        self.index = None
        self.is_initialized = False


# Example usage
async def example_usage():
    """Example of using the shared vector database."""
    
    # Create shared vector database
    vector_db = SharedVectorDatabase(dimension=384)
    await vector_db.initialize()
    
    print(f"✅ Created shared vector database: {vector_db.get_resource_id()}")
    
    # Simulate multiple workers accessing the database
    async def worker_task(worker_id: str, query_id: int):
        """Simulate a worker performing operations."""
        # Add some vectors
        vectors = [np.random.rand(384).tolist() for _ in range(10)]
        metadata = [{'query_id': query_id, 'chunk_id': i} for i in range(10)]
        
        add_result = await vector_db.add_vectors(vectors, metadata, worker_id=worker_id)
        print(f"Worker {worker_id}: Added {add_result['added_count']} vectors")
        
        # Search
        query_vector = np.random.rand(384).tolist()
        search_result = await vector_db.search(query_vector, top_k=5, worker_id=worker_id)
        print(f"Worker {worker_id}: Found {len(search_result['results'])} results")
        
        return search_result
    
    # Run multiple workers in parallel
    tasks = [
        worker_task(f"worker_{i}", i)
        for i in range(4)
    ]
    
    results = await asyncio.gather(*tasks)
    
    # Print stats
    stats = vector_db.get_stats()
    print(f"\n📊 Final Statistics:")
    print(f"   Total vectors: {stats['database_stats']['total_vectors']}")
    print(f"   Total accesses: {stats['database_stats']['access_count']}")
    print(f"   Unique workers: {stats['database_stats']['unique_workers']}")
    
    # Shutdown
    await vector_db.shutdown()


if __name__ == "__main__":
    asyncio.run(example_usage())

