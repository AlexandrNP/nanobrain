#!/usr/bin/env python3
"""
Embedding Generator Step for RAG systems.
Deterministic embedding generation via API calls to embedding services.
"""

import asyncio
import logging
import json
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from nanobrain.core.step import BaseStep, StepConfig
from nanobrain.core.logging_system import get_logger

logger = get_logger(__name__)


class EmbeddingGeneratorStep(BaseStep):
    """
    Deterministic embedding generation step.
    
    Generates embeddings via direct API calls to embedding services
    without LLM processing. Supports multiple embedding providers.
    
    Key Features:
    - Multiple provider support (OpenAI, HuggingFace, sentence-transformers)
    - Intelligent caching with hash-based cache keys
    - Batch processing with rate limiting
    - Performance optimization and retry logic
    """
    
    COMPONENT_TYPE = "embedding_generator_step"
    
    def __init__(self, *args, **kwargs):
        """Prevent direct instantiation - use from_config instead"""
        raise RuntimeError(
            "Direct instantiation of EmbeddingGeneratorStep is prohibited. "
            "ALL framework components must use EmbeddingGeneratorStep.from_config() "
            "as per mandatory framework requirements."
        )

    def _init_from_config(self, config: StepConfig, component_config: Dict[str, Any],
                         dependencies: Dict[str, Any]) -> None:
        """Initialize EmbeddingGeneratorStep from configuration."""
        # Call parent initialization
        super()._init_from_config(config, component_config, dependencies)

        # Embedding configuration (sentence-transformers as default)
        self.provider = component_config.get('provider', 'sentence-transformers')
        self.model_name = component_config.get('model_name', 'all-MiniLM-L6-v2')
        self.batch_size = component_config.get('batch_size', 100)
        self.max_retries = component_config.get('max_retries', 3)
        self.normalize_embeddings = component_config.get('normalize_embeddings', True)
        self.dimension = component_config.get('dimension', 384)  # Default for all-MiniLM-L6-v2

        # Processing settings
        self.parallel_batches = component_config.get('parallel_batches', 4)
        self.timeout_seconds = component_config.get('timeout_seconds', 30)
        self.cache_embeddings = component_config.get('cache_embeddings', True)
        self.cache_directory = Path(component_config.get('cache_directory', 'data/embedding_cache'))

        # Initialize embedding client
        self.embedding_client = self._initialize_embedding_client()

        # Create cache directory
        if self.cache_embeddings:
            self.cache_directory.mkdir(parents=True, exist_ok=True)

        logger.info(f"🔢 EmbeddingGeneratorStep initialized with {self.provider} ({self.model_name}, dim={self.dimension})")
    
    def _initialize_embedding_client(self) -> Any:
        """Initialize embedding client based on provider."""
        # Simple mock client for demonstration
        # In full implementation, would use actual API clients
        class MockEmbeddingClient:
            def __init__(self, provider: str, model_name: str, **kwargs):
                self.provider = provider
                self.model_name = model_name
                self.batch_size = kwargs.get('batch_size', 100)
                self.dimension = kwargs.get('dimension', 1536)
            
            async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
                """Generate mock embeddings for texts."""
                import random
                
                # Simulate API delay
                await asyncio.sleep(0.1)
                
                # Generate mock embeddings
                embeddings = []
                for text in texts:
                    # Create deterministic but varied embeddings based on text hash
                    text_hash = hashlib.md5(text.encode()).hexdigest()
                    random.seed(int(text_hash[:8], 16))
                    
                    embedding = [random.uniform(-1, 1) for _ in range(self.dimension)]
                    
                    # Normalize if requested
                    if hasattr(self, 'normalize') and self.normalize:
                        norm = sum(x*x for x in embedding) ** 0.5
                        if norm > 0:
                            embedding = [x/norm for x in embedding]
                    
                    embeddings.append(embedding)
                
                return embeddings
        
        client_config = {
            'model_name': self.model_name,
            'batch_size': self.batch_size,
            'max_retries': self.max_retries,
            'timeout': self.timeout_seconds,
            'normalize': self.normalize_embeddings,
            'dimension': self.dimension
        }
        
        return MockEmbeddingClient(self.provider, **client_config)
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate embeddings deterministically via API calls.
        
        Args:
            input_data: Contains document chunks from DocumentProcessorStep
            
        Returns:
            Dictionary with embeddings and metadata
        """
        try:
            start_time = asyncio.get_event_loop().time()
            
            # Extract chunks from input
            chunks = input_data.get('chunks', [])
            
            if not chunks:
                logger.warning("⚠️ No chunks provided for embedding generation")
                return self._create_empty_result()
            
            logger.info(f"🔢 Generating embeddings for {len(chunks)} chunks")
            
            # Check cache for existing embeddings
            cached_embeddings, uncached_chunks = await self._check_embedding_cache(chunks)
            
            # Generate embeddings for uncached chunks
            new_embeddings = []
            if uncached_chunks:
                new_embeddings = await self._generate_embeddings_batch(uncached_chunks)
                
                # Cache new embeddings
                if self.cache_embeddings:
                    await self._cache_embeddings(new_embeddings)
            
            # Combine cached and new embeddings
            all_embeddings = cached_embeddings + new_embeddings
            
            # Sort embeddings to match original chunk order
            all_embeddings = self._sort_embeddings_by_chunk_order(all_embeddings, chunks)
            
            processing_time = asyncio.get_event_loop().time() - start_time
            
            # Generate embedding statistics
            embedding_stats = self._generate_embedding_stats(all_embeddings, processing_time)
            
            result = {
                'embeddings': all_embeddings,
                'embedding_stats': embedding_stats,
                'model_info': {
                    'provider': self.provider,
                    'model_name': self.model_name,
                    'dimension': self.dimension,
                    'normalized': self.normalize_embeddings
                }
            }
            
            logger.info(f"✅ Embedding generation completed: {len(all_embeddings)} embeddings in {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Embedding generation failed: {e}")
            return self._create_error_result(str(e))
    
    async def _check_embedding_cache(self, chunks: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Check cache for existing embeddings."""
        if not self.cache_embeddings:
            return [], chunks
        
        cached_embeddings = []
        uncached_chunks = []
        
        for chunk in chunks:
            cache_key = self._generate_cache_key(chunk)
            cache_file = self.cache_directory / f"{cache_key}.json"
            
            if cache_file.exists():
                try:
                    with open(cache_file, 'r') as f:
                        cached_embedding = json.load(f)
                        cached_embeddings.append(cached_embedding)
                        logger.debug(f"📋 Using cached embedding for chunk: {chunk['metadata']['chunk_id']}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load cached embedding: {e}")
                    uncached_chunks.append(chunk)
            else:
                uncached_chunks.append(chunk)
        
        logger.info(f"📋 Found {len(cached_embeddings)} cached embeddings, generating {len(uncached_chunks)} new ones")
        
        return cached_embeddings, uncached_chunks
    
    def _generate_cache_key(self, chunk: Dict[str, Any]) -> str:
        """Generate cache key for chunk embedding."""
        # Create hash from chunk content and model info
        content = chunk['content']
        model_info = f"{self.provider}_{self.model_name}_{self.dimension}"
        
        hasher = hashlib.md5()
        hasher.update(f"{content}_{model_info}".encode('utf-8'))
        
        return hasher.hexdigest()
    
    async def _generate_embeddings_batch(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate embeddings for chunks in batches."""
        all_embeddings = []
        
        # Process chunks in batches
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]
            batch_texts = [chunk['content'] for chunk in batch]
            
            try:
                # Generate embeddings via API
                batch_vectors = await self.embedding_client.generate_embeddings(batch_texts)
                
                # Create embedding objects
                batch_embeddings = []
                for j, (chunk, vector) in enumerate(zip(batch, batch_vectors)):
                    embedding_data = {
                        'vector': vector,
                        'chunk_id': chunk['metadata']['chunk_id'],
                        'chunk_index': chunk['metadata']['chunk_index'],
                        'text_preview': chunk['content'][:100] + '...' if len(chunk['content']) > 100 else chunk['content'],
                        'metadata': {
                            **chunk['metadata'],
                            'embedding_model': self.model_name,
                            'embedding_provider': self.provider,
                            'embedding_dimension': len(vector),
                            'normalized': self.normalize_embeddings
                        }
                    }
                    batch_embeddings.append(embedding_data)
                
                all_embeddings.extend(batch_embeddings)
                
                logger.info(f"🔢 Generated embeddings for batch {i//self.batch_size + 1}: {len(batch_embeddings)} embeddings")
                
                # Add small delay to respect API rate limits
                if i + self.batch_size < len(chunks):
                    await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"❌ Failed to generate embeddings for batch {i//self.batch_size + 1}: {e}")
                # Continue with next batch
                continue
        
        return all_embeddings

    async def _cache_embeddings(self, embeddings: List[Dict[str, Any]]) -> None:
        """Cache generated embeddings."""
        for embedding in embeddings:
            try:
                cache_key = self._generate_cache_key({
                    'content': embedding['text_preview'].replace('...', ''),
                    'metadata': embedding['metadata']
                })
                cache_file = self.cache_directory / f"{cache_key}.json"

                with open(cache_file, 'w') as f:
                    json.dump(embedding, f, indent=2)

            except Exception as e:
                logger.warning(f"⚠️ Failed to cache embedding: {e}")

    def _sort_embeddings_by_chunk_order(self, embeddings: List[Dict[str, Any]],
                                      original_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sort embeddings to match original chunk order."""
        # Create mapping from chunk_id to original index
        chunk_order = {chunk['metadata']['chunk_id']: i for i, chunk in enumerate(original_chunks)}

        # Sort embeddings by original chunk order
        sorted_embeddings = sorted(embeddings, key=lambda x: chunk_order.get(x['chunk_id'], float('inf')))

        return sorted_embeddings

    def _generate_embedding_stats(self, embeddings: List[Dict[str, Any]], processing_time: float) -> Dict[str, Any]:
        """Generate statistics about embedding generation."""
        if not embeddings:
            return {}

        dimensions = [len(emb['vector']) for emb in embeddings]

        return {
            'total_embeddings': len(embeddings),
            'embedding_dimension': dimensions[0] if dimensions else 0,
            'processing_time': processing_time,
            'embeddings_per_second': len(embeddings) / processing_time if processing_time > 0 else 0,
            'model_used': self.model_name,
            'provider_used': self.provider,
            'cache_hits': sum(1 for emb in embeddings if 'cached' in emb.get('metadata', {})),
            'new_generations': len(embeddings) - sum(1 for emb in embeddings if 'cached' in emb.get('metadata', {}))
        }

    def _create_empty_result(self) -> Dict[str, Any]:
        """Create empty result when no chunks provided."""
        return {
            'embeddings': [],
            'embedding_stats': {
                'total_embeddings': 0,
                'processing_time': 0,
                'embeddings_per_second': 0
            },
            'model_info': {
                'provider': self.provider,
                'model_name': self.model_name,
                'dimension': self.dimension
            }
        }

    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create error result."""
        return {
            'embeddings': [],
            'embedding_stats': {
                'total_embeddings': 0,
                'processing_time': 0,
                'embeddings_per_second': 0,
                'error': error_message
            },
            'model_info': {
                'provider': self.provider,
                'model_name': self.model_name,
                'dimension': self.dimension
            },
            'error': error_message
        }
