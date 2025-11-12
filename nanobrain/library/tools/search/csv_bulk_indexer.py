"""
CSV Bulk Indexer for Elasticsearch

This module provides efficient bulk indexing capabilities for CSV data,
optimized for large datasets and memory efficiency.
"""

import asyncio
import hashlib
import time
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Iterator, Callable
import logging

# Optional imports
try:
    from elasticsearch import AsyncElasticsearch
    from elasticsearch.helpers import async_bulk
    ELASTICSEARCH_AVAILABLE = True
except ImportError:
    ELASTICSEARCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class CSVBulkIndexer:
    """
    Efficient bulk indexer for CSV data with progress tracking,
    error handling, and memory optimization.
    """
    
    def __init__(self, elasticsearch_client: Any, config: Optional[Dict[str, Any]] = None):
        """Initialize bulk indexer"""
        if not ELASTICSEARCH_AVAILABLE:
            raise ImportError("Elasticsearch client not available")
        
        self.client = elasticsearch_client
        self.config = config or {}
        
        # Configuration
        self.chunk_size = self.config.get('chunk_size', 1000)
        self.max_chunk_bytes = self.config.get('max_chunk_bytes', 10 * 1024 * 1024)  # 10MB
        self.request_timeout = self.config.get('request_timeout', 60)
        self.max_retries = self.config.get('max_retries', 3)
        self.retry_delay = self.config.get('retry_delay', 1)
        
        # Progress tracking
        self.progress_callback: Optional[Callable] = None
        self.total_processed = 0
        self.total_errors = 0
        self.start_time = None
        
        logger.info("✅ CSV Bulk Indexer initialized")
    
    def set_progress_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Set callback function for progress updates"""
        self.progress_callback = callback
    
    async def index_csv_data(self, 
                           index_name: str, 
                           data_iterator: Iterator[List[Dict[str, Any]]], 
                           file_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Index CSV data in chunks with progress tracking"""
        try:
            logger.info(f"🔄 Starting bulk indexing to index: {index_name}")
            
            self.start_time = time.time()
            self.total_processed = 0
            self.total_errors = 0
            
            results = {
                'success': True,
                'total_processed': 0,
                'total_errors': 0,
                'chunks_processed': 0,
                'processing_time': 0,
                'errors': []
            }
            
            chunk_count = 0
            
            # Process data in chunks
            async for chunk in self._async_chunk_iterator(data_iterator):
                chunk_count += 1
                
                # Prepare documents for indexing
                documents = self._prepare_documents(chunk, index_name, file_metadata, chunk_count)
                
                # Index chunk with retry logic
                chunk_result = await self._index_chunk_with_retry(documents)
                
                # Update results
                results['total_processed'] += chunk_result.get('processed', 0)
                results['total_errors'] += chunk_result.get('errors', 0)
                results['chunks_processed'] = chunk_count
                
                if chunk_result.get('error_details'):
                    results['errors'].extend(chunk_result['error_details'][:5])  # Limit error details
                
                # Update progress
                self._update_progress(results)
                
                # Check for too many errors
                if results['total_errors'] > results['total_processed'] * 0.1:  # More than 10% errors
                    logger.warning("High error rate detected, stopping indexing")
                    results['success'] = False
                    break
            
            # Calculate final metrics
            results['processing_time'] = time.time() - self.start_time
            results['documents_per_second'] = results['total_processed'] / max(results['processing_time'], 1)
            
            logger.info(f"✅ Bulk indexing completed: {results['total_processed']} documents, "
                       f"{results['total_errors']} errors, {results['processing_time']:.2f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Bulk indexing failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'total_processed': self.total_processed,
                'total_errors': self.total_errors
            }
    
    async def _async_chunk_iterator(self, data_iterator: Iterator[List[Dict[str, Any]]]) -> Iterator[List[Dict[str, Any]]]:
        """Convert synchronous iterator to async iterator"""
        for chunk in data_iterator:
            yield chunk
            # Allow other coroutines to run
            await asyncio.sleep(0)
    
    def _prepare_documents(self, 
                          chunk: List[Dict[str, Any]], 
                          index_name: str, 
                          file_metadata: Optional[Dict[str, Any]], 
                          chunk_number: int) -> List[Dict[str, Any]]:
        """Prepare documents for Elasticsearch bulk API"""
        documents = []
        current_time = datetime.now(timezone.utc).isoformat()
        
        for row_index, row_data in enumerate(chunk):
            # Generate document ID
            doc_id = row_data.get('id') or str(uuid.uuid4())
            
            # Add metadata
            document = row_data.copy()
            document['_csv_import_timestamp'] = current_time
            document['_csv_row_number'] = self.total_processed + row_index + 1
            
            if file_metadata:
                document['_csv_file_name'] = file_metadata.get('file_name', 'unknown')
                document['_csv_file_hash'] = file_metadata.get('file_hash', 'unknown')
            
            # Prepare for bulk API
            doc_action = {
                '_index': index_name,
                '_id': doc_id,
                '_source': document
            }
            
            documents.append(doc_action)
        
        return documents
    
    async def _index_chunk_with_retry(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Index a chunk of documents with retry logic"""
        for attempt in range(self.max_retries + 1):
            try:
                # Use elasticsearch-py async bulk helper
                success_count, error_details = await async_bulk(
                    self.client,
                    documents,
                    chunk_size=self.chunk_size,
                    request_timeout=self.request_timeout,
                    max_chunk_bytes=self.max_chunk_bytes
                )
                
                return {
                    'processed': success_count,
                    'errors': len(error_details),
                    'error_details': error_details
                }
                
            except Exception as e:
                if attempt < self.max_retries:
                    logger.warning(f"Chunk indexing failed (attempt {attempt + 1}), retrying: {e}")
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                else:
                    logger.error(f"Chunk indexing failed after {self.max_retries} retries: {e}")
                    return {
                        'processed': 0,
                        'errors': len(documents),
                        'error_details': [{'error': str(e), 'document_count': len(documents)}]
                    }
    
    def _update_progress(self, results: Dict[str, Any]):
        """Update progress and call progress callback if set"""
        self.total_processed = results['total_processed']
        self.total_errors = results['total_errors']
        
        if self.progress_callback:
            progress_data = {
                'total_processed': results['total_processed'],
                'total_errors': results['total_errors'],
                'chunks_processed': results['chunks_processed'],
                'processing_time': time.time() - self.start_time if self.start_time else 0,
                'documents_per_second': results.get('documents_per_second', 0)
            }
            
            try:
                self.progress_callback(progress_data)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")
    
    async def optimize_index_for_search(self, index_name: str) -> Dict[str, Any]:
        """Optimize index after bulk indexing for better search performance"""
        try:
            logger.info(f"🔧 Optimizing index for search: {index_name}")
            
            # Force merge to optimize segments
            await self.client.indices.forcemerge(
                index=index_name,
                max_num_segments=1,
                wait_for_completion=True
            )
            
            # Update refresh interval for better search performance
            await self.client.indices.put_settings(
                index=index_name,
                body={
                    "refresh_interval": "1s"
                }
            )
            
            logger.info(f"✅ Index optimization completed: {index_name}")
            
            return {
                'success': True,
                'index_name': index_name,
                'optimized': True
            }
            
        except Exception as e:
            logger.error(f"❌ Index optimization failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'index_name': index_name
            }
    
    def calculate_file_hash(self, file_path: str) -> str:
        """Calculate hash of file for tracking"""
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.warning(f"Failed to calculate file hash: {e}")
            return "unknown"
    
    def estimate_indexing_time(self, total_rows: int, sample_rate: float = 1000.0) -> Dict[str, Any]:
        """Estimate indexing time based on sample rate"""
        if not self.start_time or self.total_processed == 0:
            return {
                'estimated_total_time': 0,
                'estimated_remaining_time': 0,
                'estimated_completion': None
            }
        
        elapsed_time = time.time() - self.start_time
        current_rate = self.total_processed / elapsed_time
        
        if current_rate > 0:
            estimated_total_time = total_rows / current_rate
            estimated_remaining_time = max(0, estimated_total_time - elapsed_time)
            estimated_completion = datetime.now() + timedelta(seconds=estimated_remaining_time)
        else:
            estimated_total_time = 0
            estimated_remaining_time = 0
            estimated_completion = None
        
        return {
            'current_rate': current_rate,
            'estimated_total_time': estimated_total_time,
            'estimated_remaining_time': estimated_remaining_time,
            'estimated_completion': estimated_completion.isoformat() if estimated_completion else None
        }


class ProgressTracker:
    """Simple progress tracker for bulk indexing operations"""
    
    def __init__(self, total_expected: Optional[int] = None):
        self.total_expected = total_expected
        self.last_update = time.time()
        self.update_interval = 5.0  # Update every 5 seconds
    
    def __call__(self, progress_data: Dict[str, Any]):
        """Progress callback function"""
        current_time = time.time()
        
        if current_time - self.last_update >= self.update_interval:
            processed = progress_data.get('total_processed', 0)
            errors = progress_data.get('total_errors', 0)
            rate = progress_data.get('documents_per_second', 0)
            
            if self.total_expected:
                percentage = (processed / self.total_expected) * 100
                logger.info(f"📊 Progress: {processed}/{self.total_expected} ({percentage:.1f}%) "
                           f"Rate: {rate:.1f} docs/sec, Errors: {errors}")
            else:
                logger.info(f"📊 Progress: {processed} documents processed, "
                           f"Rate: {rate:.1f} docs/sec, Errors: {errors}")
            
            self.last_update = current_time


# Factory function following NanoBrain patterns
def from_config(elasticsearch_client: Any, config: Dict[str, Any]) -> CSVBulkIndexer:
    """Create CSVBulkIndexer from configuration dictionary"""
    return CSVBulkIndexer(elasticsearch_client, config)
