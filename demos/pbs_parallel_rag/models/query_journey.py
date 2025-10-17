"""
Query Journey Data Models
==========================

Simple data models for tracking query processing through RAG pipeline.
Following Nanobrain patterns: dataclasses, simple serialization, no complex logic.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import time


@dataclass
class Document:
    """A retrieved document from vector search."""
    
    doc_id: str
    content: str
    relevance_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Document':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class StepMetadata:
    """Metadata for a single step in the pipeline."""

    worker_id: Optional[str] = None
    instance_id: Optional[int] = None
    start_time: Optional[float] = None  # Unix timestamp when step started
    end_time: Optional[float] = None    # Unix timestamp when step ended
    duration: Optional[float] = None    # Duration in seconds

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {}
        for k, v in asdict(self).items():
            if v is not None:
                result[k] = v
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StepMetadata':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class QueryJourney:
    """
    Complete journey of a query through the RAG pipeline.
    
    Tracks:
    - Original query
    - Enhanced query (from query enhancement step)
    - Retrieved documents (from vector search step)
    - Final response (from response generation step)
    - Metadata for each step
    """
    
    # Identifiers
    query_id: str
    timestamp: float = field(default_factory=time.time)
    
    # Original query
    original_query: str = ""
    
    # Step 1: Query Enhancement
    enhanced_query: Optional[str] = None
    enhancement_metadata: Optional[StepMetadata] = None
    
    # Step 2: Document Retrieval
    retrieved_documents: List[Document] = field(default_factory=list)
    retrieval_metadata: Optional[StepMetadata] = None
    
    # Step 3: Response Generation
    final_response: Optional[str] = None
    generation_metadata: Optional[StepMetadata] = None
    
    # Overall status
    status: str = "pending"  # pending, processing, complete, failed
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = {
            'query_id': self.query_id,
            'timestamp': self.timestamp,
            'original_query': self.original_query,
            'status': self.status,
        }
        
        # Add enhanced query if available
        if self.enhanced_query:
            data['enhanced_query'] = self.enhanced_query
        
        # Add enhancement metadata if available
        if self.enhancement_metadata:
            data['enhancement_metadata'] = self.enhancement_metadata.to_dict()
        
        # Add retrieved documents if available
        if self.retrieved_documents:
            data['retrieved_documents'] = [doc.to_dict() for doc in self.retrieved_documents]
        
        # Add retrieval metadata if available
        if self.retrieval_metadata:
            data['retrieval_metadata'] = self.retrieval_metadata.to_dict()
        
        # Add final response if available
        if self.final_response:
            data['final_response'] = self.final_response
        
        # Add generation metadata if available
        if self.generation_metadata:
            data['generation_metadata'] = self.generation_metadata.to_dict()
        
        # Add error if present
        if self.error:
            data['error'] = self.error
        
        # Calculate total time if all steps complete
        if self.enhancement_metadata and self.retrieval_metadata and self.generation_metadata:
            total_time = 0.0
            if self.enhancement_metadata.duration:
                total_time += self.enhancement_metadata.duration
            if self.retrieval_metadata.duration:
                total_time += self.retrieval_metadata.duration
            if self.generation_metadata.duration:
                total_time += self.generation_metadata.duration
            data['total_time'] = total_time
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QueryJourney':
        """Create from dictionary."""
        # Extract basic fields
        journey = cls(
            query_id=data['query_id'],
            timestamp=data.get('timestamp', time.time()),
            original_query=data.get('original_query', ''),
            enhanced_query=data.get('enhanced_query'),
            final_response=data.get('final_response'),
            status=data.get('status', 'pending'),
            error=data.get('error')
        )
        
        # Reconstruct metadata
        if 'enhancement_metadata' in data:
            journey.enhancement_metadata = StepMetadata.from_dict(data['enhancement_metadata'])
        
        if 'retrieval_metadata' in data:
            journey.retrieval_metadata = StepMetadata.from_dict(data['retrieval_metadata'])
        
        if 'generation_metadata' in data:
            journey.generation_metadata = StepMetadata.from_dict(data['generation_metadata'])
        
        # Reconstruct documents
        if 'retrieved_documents' in data:
            journey.retrieved_documents = [
                Document.from_dict(doc) for doc in data['retrieved_documents']
            ]
        
        return journey
    
    def get_total_time(self) -> Optional[float]:
        """Get total processing time across all steps."""
        if not (self.enhancement_metadata and self.retrieval_metadata and self.generation_metadata):
            return None
        
        total = 0.0
        if self.enhancement_metadata.duration:
            total += self.enhancement_metadata.duration
        if self.retrieval_metadata.duration:
            total += self.retrieval_metadata.duration
        if self.generation_metadata.duration:
            total += self.generation_metadata.duration
        
        return total if total > 0 else None
    
    def get_avg_relevance_score(self) -> Optional[float]:
        """Get average relevance score of retrieved documents."""
        if not self.retrieved_documents:
            return None
        
        scores = [doc.relevance_score for doc in self.retrieved_documents]
        return sum(scores) / len(scores) if scores else None
    
    def is_complete(self) -> bool:
        """Check if journey is complete."""
        return self.status == "complete" and all([
            self.enhanced_query,
            self.retrieved_documents,
            self.final_response
        ])
    
    def mark_failed(self, error: str):
        """Mark journey as failed with error message."""
        self.status = "failed"
        self.error = error
    
    def mark_complete(self):
        """Mark journey as complete."""
        if self.is_complete():
            self.status = "complete"

