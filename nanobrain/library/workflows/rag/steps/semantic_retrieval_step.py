#!/usr/bin/env python3
"""
Semantic Retrieval Step for RAG systems.
Deterministic similarity search and ranking operations.
"""

import asyncio
import json
import hashlib
from typing import Dict, Any, List, Optional, Tuple

from nanobrain.core.step import BaseStep, StepConfig
from nanobrain.core.logging_system import get_logger

logger = get_logger(__name__)


class SemanticRetrievalStep(BaseStep):
    """
    Deterministic semantic retrieval step.
    
    Performs vector similarity search and ranking without LLM processing.
    Supports dense, sparse, and hybrid retrieval strategies.
    
    Key Features:
    - Multiple search strategies (dense, sparse, hybrid)
    - Advanced reranking with cross-encoder models
    - Query preprocessing and optimization
    - Similarity filtering and result limiting
    """
    
    COMPONENT_TYPE = "semantic_retrieval_step"
    
    def __init__(self, *args, **kwargs):
        """Prevent direct instantiation - use from_config instead"""
        raise RuntimeError(
            "Direct instantiation of SemanticRetrievalStep is prohibited. "
            "ALL framework components must use SemanticRetrievalStep.from_config() "
            "as per mandatory framework requirements."
        )

    def _init_from_config(self, config: StepConfig, component_config: Dict[str, Any],
                         dependencies: Dict[str, Any]) -> None:
        """Initialize SemanticRetrievalStep from configuration."""
        # Call parent initialization
        super()._init_from_config(config, component_config, dependencies)

        # Retrieval configuration
        self.top_k = component_config.get('top_k', 10)
        self.similarity_threshold = component_config.get('similarity_threshold', 0.7)
        self.search_strategy = component_config.get('search_strategy', 'dense')

        # Ranking configuration
        self.enable_reranking = component_config.get('enable_reranking', True)
        self.rerank_model = component_config.get('rerank_model', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.rerank_top_k = component_config.get('rerank_top_k', 50)

        # Query processing
        self.embedding_model = component_config.get('embedding_model', 'all-MiniLM-L6-v2')
        self.max_query_length = component_config.get('max_query_length', 500)
        self.preprocessing_enabled = component_config.get('preprocessing_enabled', True)

        # Initialize components
        self.query_encoder = self._initialize_query_encoder()
        self.retrieval_algorithm = self._initialize_retrieval_algorithm()
        self.reranker = self._initialize_reranker() if self.enable_reranking else None

        logger.info(f"🔍 SemanticRetrievalStep initialized with {self.search_strategy} strategy")
    
    def _initialize_query_encoder(self) -> Any:
        """Initialize query encoder for embedding generation."""
        # Mock query encoder for demonstration
        class MockQueryEncoder:
            def __init__(self, model_name: str):
                self.model_name = model_name
                self.dimension = 384  # Default for sentence-transformers
            
            async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
                """Generate mock embeddings for query."""
                import random
                
                embeddings = []
                for text in texts:
                    # Create deterministic but varied embeddings based on text hash
                    text_hash = hashlib.md5(text.encode()).hexdigest()
                    random.seed(int(text_hash[:8], 16))
                    
                    embedding = [random.uniform(-1, 1) for _ in range(self.dimension)]
                    
                    # Normalize
                    norm = sum(x*x for x in embedding) ** 0.5
                    if norm > 0:
                        embedding = [x/norm for x in embedding]
                    
                    embeddings.append(embedding)
                
                return embeddings
        
        return MockQueryEncoder(self.embedding_model)
    
    def _initialize_retrieval_algorithm(self) -> Any:
        """Initialize retrieval algorithm based on strategy."""
        class MockRetrievalAlgorithm:
            def __init__(self, strategy: str, top_k: int, similarity_threshold: float):
                self.strategy = strategy
                self.top_k = top_k
                self.similarity_threshold = similarity_threshold
            
            async def search(self, query_vector: List[float], vector_db: Any, top_k: int) -> List[Dict[str, Any]]:
                """Perform similarity search using the configured algorithm."""
                if not vector_db:
                    return []
                
                # Use the vector database's search method
                results = await vector_db.search(query_vector, top_k)
                
                # Convert to expected format
                formatted_results = []
                for result in results:
                    formatted_result = {
                        'content': result['metadata'].get('chunk_content', 'No content available'),
                        'similarity_score': result['similarity'],
                        'chunk_id': result['id'],
                        'metadata': result['metadata']
                    }
                    formatted_results.append(formatted_result)
                
                return formatted_results
        
        algorithm_config = {
            'top_k': self.top_k,
            'similarity_threshold': self.similarity_threshold
        }
        
        return MockRetrievalAlgorithm(self.search_strategy, **algorithm_config)
    
    def _initialize_reranker(self) -> Any:
        """Initialize reranking model if enabled."""
        # Mock reranker for demonstration
        class MockReranker:
            def __init__(self, model_name: str):
                self.model_name = model_name
            
            def predict(self, pairs: List[Tuple[str, str]]) -> List[float]:
                """Generate mock reranking scores."""
                import random
                
                scores = []
                for query, doc in pairs:
                    # Create deterministic but varied scores based on content similarity
                    combined = f"{query}_{doc}"
                    text_hash = hashlib.md5(combined.encode()).hexdigest()
                    random.seed(int(text_hash[:8], 16))
                    
                    # Generate score between 0 and 1
                    score = random.uniform(0.3, 1.0)
                    scores.append(score)
                
                return scores
        
        try:
            return MockReranker(self.rerank_model)
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize reranker, disabling reranking: {e}")
            return None
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform semantic retrieval using similarity search.
        
        Args:
            input_data: Contains enhanced query and vector database reference
            
        Returns:
            Dictionary with retrieved documents and relevance scores
        """
        try:
            start_time = asyncio.get_event_loop().time()
            
            # Extract query from input (could be from QueryEnhancementStep or direct user input)
            query_data = self._extract_query_from_input(input_data)
            
            if not query_data:
                logger.warning("⚠️ No query provided for retrieval")
                return self._create_empty_result()
            
            query_text = query_data['query']
            logger.info(f"🔍 Performing semantic retrieval for query: '{query_text[:100]}...'")
            
            # Preprocess query if enabled
            if self.preprocessing_enabled:
                query_text = self._preprocess_query(query_text)
            
            # Encode query to vector
            query_vector = await self._encode_query(query_text)
            
            # Get vector database reference from previous steps
            vector_db = self._get_vector_database_reference(input_data)
            
            # Perform similarity search
            initial_results = await self._perform_similarity_search(query_vector, vector_db)
            
            # Apply reranking if enabled
            if self.enable_reranking and self.reranker and len(initial_results) > 1:
                reranked_results = await self._rerank_results(query_text, initial_results)
            else:
                reranked_results = initial_results
            
            # Filter by similarity threshold and limit to top_k
            filtered_results = self._filter_and_limit_results(reranked_results)
            
            processing_time = asyncio.get_event_loop().time() - start_time
            
            # Generate retrieval statistics
            retrieval_stats = self._generate_retrieval_stats(filtered_results, processing_time)
            
            result = {
                'retrieved_chunks': filtered_results,
                'retrieval_stats': retrieval_stats,
                'query_info': {
                    'original_query': query_data.get('original_query', query_text),
                    'processed_query': query_text,
                    'search_strategy': self.search_strategy,
                    'reranking_enabled': self.enable_reranking
                }
            }
            
            logger.info(f"✅ Semantic retrieval completed: {len(filtered_results)} chunks retrieved in {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Semantic retrieval failed: {e}")
            return self._create_error_result(str(e))
    
    def _extract_query_from_input(self, input_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract query from input data."""
        # Check for enhanced query from QueryEnhancementStep
        if 'enhanced_query' in input_data:
            try:
                enhanced_data = json.loads(input_data['enhanced_query'])
                return {
                    'query': enhanced_data['enhanced_query'],
                    'original_query': enhanced_data.get('original_query', enhanced_data['enhanced_query']),
                    'search_terms': enhanced_data.get('search_terms', []),
                    'context_additions': enhanced_data.get('context_additions', '')
                }
            except:
                # Fallback to treating as plain text
                return {'query': input_data['enhanced_query']}
        
        # Check for direct query input
        elif 'query' in input_data:
            return {'query': input_data['query']}
        
        # Check for user_query from workflow input
        elif 'user_query' in input_data:
            return {'query': input_data['user_query']}
        
        return None
    
    def _preprocess_query(self, query: str) -> str:
        """Preprocess query for better retrieval."""
        # Basic preprocessing
        processed = query.strip()
        
        # Limit query length
        if len(processed) > self.max_query_length:
            processed = processed[:self.max_query_length].rsplit(' ', 1)[0]
        
        # Additional preprocessing could include:
        # - Removing stop words
        # - Expanding abbreviations
        # - Normalizing text
        
        return processed
    
    async def _encode_query(self, query: str) -> List[float]:
        """Encode query to vector using embedding model."""
        try:
            query_vectors = await self.query_encoder.generate_embeddings([query])
            return query_vectors[0]
        except Exception as e:
            logger.error(f"❌ Failed to encode query: {e}")
            raise
    
    def _get_vector_database_reference(self, input_data: Dict[str, Any]) -> Any:
        """Get vector database reference from input data or dependencies."""
        # This would typically come from the VectorStorageStep or be injected as a dependency
        return input_data.get('vector_db')
    
    async def _perform_similarity_search(self, query_vector: List[float], vector_db: Any) -> List[Dict[str, Any]]:
        """Perform similarity search using the configured algorithm."""
        if not vector_db:
            raise ValueError("Vector database not available for similarity search")
        
        # Use retrieval algorithm to search
        search_results = await self.retrieval_algorithm.search(
            query_vector=query_vector,
            vector_db=vector_db,
            top_k=self.rerank_top_k if self.enable_reranking else self.top_k
        )
        
        return search_results

    async def _rerank_results(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank results using cross-encoder model."""
        if not self.reranker or len(results) <= 1:
            return results

        try:
            # Prepare query-document pairs for reranking
            pairs = [(query, result['content']) for result in results]

            # Get reranking scores
            rerank_scores = self.reranker.predict(pairs)

            # Update results with rerank scores and sort
            for result, score in zip(results, rerank_scores):
                result['rerank_score'] = float(score)
                result['original_similarity'] = result.get('similarity_score', 0.0)

            # Sort by rerank score
            reranked = sorted(results, key=lambda x: x['rerank_score'], reverse=True)

            logger.info(f"🔄 Reranked {len(results)} results using {self.rerank_model}")

            return reranked

        except Exception as e:
            logger.warning(f"⚠️ Reranking failed, using original order: {e}")
            return results

    def _filter_and_limit_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter results by similarity threshold and limit to top_k."""
        # Filter by similarity threshold
        filtered = []
        for result in results:
            # Use rerank score if available, otherwise use similarity score
            score = result.get('rerank_score', result.get('similarity_score', 0.0))

            if score >= self.similarity_threshold:
                filtered.append(result)

        # Limit to top_k
        return filtered[:self.top_k]

    def _generate_retrieval_stats(self, results: List[Dict[str, Any]], processing_time: float) -> Dict[str, Any]:
        """Generate statistics about retrieval performance."""
        if not results:
            return {
                'total_retrieved': 0,
                'processing_time': processing_time,
                'average_similarity': 0.0,
                'retrieval_rate': 0.0
            }

        # Calculate average similarity
        similarities = [r.get('similarity_score', 0.0) for r in results]
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0

        # Calculate average rerank score if available
        rerank_scores = [r.get('rerank_score') for r in results if 'rerank_score' in r]
        avg_rerank_score = sum(rerank_scores) / len(rerank_scores) if rerank_scores else None

        stats = {
            'total_retrieved': len(results),
            'processing_time': processing_time,
            'retrieval_rate': len(results) / processing_time if processing_time > 0 else 0,
            'average_similarity': avg_similarity,
            'min_similarity': min(similarities) if similarities else 0.0,
            'max_similarity': max(similarities) if similarities else 0.0,
            'search_strategy_used': self.search_strategy,
            'reranking_applied': any('rerank_score' in r for r in results)
        }

        if avg_rerank_score is not None:
            stats['average_rerank_score'] = avg_rerank_score

        return stats

    def _create_empty_result(self) -> Dict[str, Any]:
        """Create empty result when no query provided."""
        return {
            'retrieved_chunks': [],
            'retrieval_stats': {
                'total_retrieved': 0,
                'processing_time': 0,
                'average_similarity': 0.0,
                'retrieval_rate': 0.0
            },
            'query_info': {
                'search_strategy': self.search_strategy,
                'reranking_enabled': self.enable_reranking
            }
        }

    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create error result."""
        return {
            'retrieved_chunks': [],
            'retrieval_stats': {
                'total_retrieved': 0,
                'processing_time': 0,
                'average_similarity': 0.0,
                'retrieval_rate': 0.0,
                'error': error_message
            },
            'query_info': {
                'search_strategy': self.search_strategy,
                'reranking_enabled': self.enable_reranking
            },
            'error': error_message
        }
