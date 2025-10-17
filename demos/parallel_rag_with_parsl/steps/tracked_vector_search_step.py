"""
Tracked Vector Search Step
===========================

Vector search step with journey tracking.
Simple extension that logs document retrieval to journey logger.
"""

import time
from typing import Any, Dict, Optional, List

from nanobrain.library.workflows.rag.steps import SemanticRetrievalStep
from demos.parallel_rag_with_parsl.journey_logging.journey_logger import QueryJourneyLogger
from demos.parallel_rag_with_parsl.models.query_journey import Document
from nanobrain.core.shared_resource import get_worker_id


class TrackedVectorSearchStep(SemanticRetrievalStep):
    """
    Vector search step with journey tracking.

    Extends SemanticRetrievalStep to log:
    - Retrieved documents
    - Relevance scores
    - Worker ID
    - Processing time
    """
    
    def __init__(self, config: Dict[str, Any], journey_logger: Optional[QueryJourneyLogger] = None):
        """
        Initialize tracked step.
        
        Args:
            config: Step configuration
            journey_logger: Logger for tracking query journeys
        """
        super().__init__(config)
        self.journey_logger = journey_logger
    
    async def process(self, *args, **kwargs):
        """
        Process vector search with journey tracking.
        
        Tracks:
        - Retrieved documents
        - Relevance scores
        - Worker ID
        - Instance ID
        - Processing time
        """
        # Get query_id from kwargs
        query_id = kwargs.get('query_id')
        
        # Process with timing
        start_time = time.time()
        result = await super().process(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time

        # Update journey with retrieval results
        if self.journey_logger and query_id:
            # Extract documents from result
            documents = []

            if isinstance(result, dict):
                # Try different possible keys for documents
                docs_data = result.get('documents') or result.get('results') or []

                # Convert to Document objects
                for i, doc_data in enumerate(docs_data):
                    if isinstance(doc_data, dict):
                        # Extract document info
                        doc_id = doc_data.get('id') or doc_data.get('doc_id') or f"doc_{i}"
                        content = doc_data.get('content') or doc_data.get('text') or ""
                        score = doc_data.get('score') or doc_data.get('relevance_score') or 0.0
                        metadata = doc_data.get('metadata', {})

                        documents.append(Document(
                            doc_id=doc_id,
                            content=content,
                            relevance_score=score,
                            metadata=metadata
                        ))
                    elif isinstance(doc_data, str):
                        # Simple string document
                        documents.append(Document(
                            doc_id=f"doc_{i}",
                            content=doc_data,
                            relevance_score=1.0,
                            metadata={}
                        ))

            # Get worker ID
            worker_id = get_worker_id()
            if isinstance(result, dict) and '_worker_id' in result:
                worker_id = result['_worker_id']

            # Update journey
            self.journey_logger.update_retrieval(
                query_id=query_id,
                documents=documents,
                worker_id=worker_id,
                instance_id=id(self),
                start_time=start_time,
                end_time=end_time,
                duration=duration
            )

            if self.nb_logger:
                self.nb_logger.info(f"✅ Updated journey with retrieval ({len(documents)} docs, worker: {worker_id}, time: {duration:.2f}s)")
        
        return result

