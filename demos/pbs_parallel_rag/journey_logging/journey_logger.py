"""
Query Journey Logger
====================

Simple logger for tracking query journeys through RAG pipeline.
Following Nanobrain patterns: simple, file-based logging, no complex state management.
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

import sys
from pathlib import Path

# Add the pbs_parallel_rag directory to path for local imports
_current_dir = Path(__file__).parent.parent
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

from models.query_journey import (
    QueryJourney, Document, StepMetadata
)


class QueryJourneyLogger:
    """
    Logger for query journeys through RAG pipeline.
    
    Simple implementation:
    - Stores journeys in memory
    - Saves to JSON and text files
    - No complex state management
    """
    
    def __init__(self, output_dir: str, format: str = "both"):
        """
        Initialize logger.
        
        Args:
            output_dir: Directory for log files
            format: "json", "text", or "both"
        """
        self.output_dir = Path(output_dir)
        self.format = format
        self.journeys: Dict[str, QueryJourney] = {}
        
        # Create output directories
        self.queries_dir = self.output_dir / "queries"
        self.queries_dir.mkdir(parents=True, exist_ok=True)
    
    def start_journey(self, query_id: str, original_query: str) -> QueryJourney:
        """
        Start a new query journey.
        
        Args:
            query_id: Unique query identifier
            original_query: Original user query
            
        Returns:
            QueryJourney object
        """
        journey = QueryJourney(
            query_id=query_id,
            original_query=original_query,
            timestamp=time.time(),
            status="processing"
        )
        
        self.journeys[query_id] = journey
        return journey
    
    def update_enhancement(
        self,
        query_id: str,
        enhanced_query: str,
        worker_id: Optional[str] = None,
        instance_id: Optional[int] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        duration: Optional[float] = None
    ):
        """
        Update journey with query enhancement results.

        Args:
            query_id: Query identifier
            enhanced_query: Enhanced/expanded query
            worker_id: Worker that processed this step
            instance_id: Step instance ID
            start_time: When step started (Unix timestamp)
            end_time: When step ended (Unix timestamp)
            duration: Time taken for this step (seconds)
        """
        if query_id not in self.journeys:
            return

        journey = self.journeys[query_id]
        journey.enhanced_query = enhanced_query
        journey.enhancement_metadata = StepMetadata(
            worker_id=worker_id,
            instance_id=instance_id,
            start_time=start_time,
            end_time=end_time,
            duration=duration
        )
    
    def update_retrieval(
        self,
        query_id: str,
        documents: List[Document],
        worker_id: Optional[str] = None,
        instance_id: Optional[int] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        duration: Optional[float] = None
    ):
        """
        Update journey with document retrieval results.

        Args:
            query_id: Query identifier
            documents: Retrieved documents
            worker_id: Worker that processed this step
            instance_id: Step instance ID
            start_time: When step started (Unix timestamp)
            end_time: When step ended (Unix timestamp)
            duration: Time taken for this step (seconds)
        """
        if query_id not in self.journeys:
            return

        journey = self.journeys[query_id]
        journey.retrieved_documents = documents
        journey.retrieval_metadata = StepMetadata(
            worker_id=worker_id,
            instance_id=instance_id,
            start_time=start_time,
            end_time=end_time,
            duration=duration
        )
    
    def update_generation(
        self,
        query_id: str,
        final_response: str,
        worker_id: Optional[str] = None,
        instance_id: Optional[int] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        duration: Optional[float] = None
    ):
        """
        Update journey with response generation results.

        Args:
            query_id: Query identifier
            final_response: Generated response
            worker_id: Worker that processed this step
            instance_id: Step instance ID
            start_time: When step started (Unix timestamp)
            end_time: When step ended (Unix timestamp)
            duration: Time taken for this step (seconds)
        """
        if query_id not in self.journeys:
            return

        journey = self.journeys[query_id]
        journey.final_response = final_response
        journey.generation_metadata = StepMetadata(
            worker_id=worker_id,
            instance_id=instance_id,
            start_time=start_time,
            end_time=end_time,
            duration=duration
        )
    
    def complete_journey(self, query_id: str):
        """
        Mark journey as complete.
        
        Args:
            query_id: Query identifier
        """
        if query_id not in self.journeys:
            return
        
        journey = self.journeys[query_id]
        journey.mark_complete()
    
    def fail_journey(self, query_id: str, error: str):
        """
        Mark journey as failed.
        
        Args:
            query_id: Query identifier
            error: Error message
        """
        if query_id not in self.journeys:
            return
        
        journey = self.journeys[query_id]
        journey.mark_failed(error)
    
    def save_journey(self, query_id: str):
        """
        Save journey to file(s).
        
        Args:
            query_id: Query identifier
        """
        if query_id not in self.journeys:
            return
        
        journey = self.journeys[query_id]
        
        # Save JSON format
        if self.format in ["json", "both"]:
            self._save_json(journey)
        
        # Save text format
        if self.format in ["text", "both"]:
            self._save_text(journey)
    
    def _save_json(self, journey: QueryJourney):
        """Save journey as JSON."""
        json_file = self.queries_dir / f"{journey.query_id}.json"
        
        with open(json_file, 'w') as f:
            json.dump(journey.to_dict(), f, indent=2)
    
    def _save_text(self, journey: QueryJourney):
        """Save journey as human-readable text."""
        text_file = self.queries_dir / f"{journey.query_id}.txt"
        
        lines = []
        lines.append("=" * 80)
        lines.append(f"QUERY JOURNEY: {journey.query_id}")
        lines.append("=" * 80)
        lines.append(f"Status: {journey.status.upper()}")
        lines.append(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(journey.timestamp))}")
        
        # Total time
        total_time = journey.get_total_time()
        if total_time:
            lines.append(f"Total Time: {total_time:.2f}s")
        
        lines.append("")
        
        # Original query
        lines.append("-" * 80)
        lines.append("ORIGINAL QUERY")
        lines.append("-" * 80)
        lines.append(journey.original_query)
        lines.append("")
        
        # Step 1: Enhancement
        if journey.enhanced_query:
            lines.append("-" * 80)
            lines.append("STEP 1: QUERY ENHANCEMENT")
            lines.append("-" * 80)
            if journey.enhancement_metadata:
                meta = journey.enhancement_metadata
                if meta.worker_id:
                    lines.append(f"Worker: {meta.worker_id}")
                if meta.instance_id:
                    lines.append(f"Instance: {meta.instance_id}")
                if meta.start_time:
                    lines.append(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(meta.start_time))}")
                if meta.end_time:
                    lines.append(f"End Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(meta.end_time))}")
                if meta.duration:
                    lines.append(f"Duration: {meta.duration:.2f}s")
            lines.append("")
            lines.append("Enhanced Query:")
            lines.append(journey.enhanced_query)
            lines.append("")
        
        # Step 2: Retrieval
        if journey.retrieved_documents:
            lines.append("-" * 80)
            lines.append("STEP 2: DOCUMENT RETRIEVAL")
            lines.append("-" * 80)
            if journey.retrieval_metadata:
                meta = journey.retrieval_metadata
                if meta.worker_id:
                    lines.append(f"Worker: {meta.worker_id}")
                if meta.instance_id:
                    lines.append(f"Instance: {meta.instance_id}")
                if meta.start_time:
                    lines.append(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(meta.start_time))}")
                if meta.end_time:
                    lines.append(f"End Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(meta.end_time))}")
                if meta.duration:
                    lines.append(f"Duration: {meta.duration:.2f}s")
            
            lines.append(f"Documents Retrieved: {len(journey.retrieved_documents)}")
            
            avg_score = journey.get_avg_relevance_score()
            if avg_score:
                lines.append(f"Average Relevance: {avg_score:.2f}")
            
            lines.append("")
            
            for i, doc in enumerate(journey.retrieved_documents, 1):
                lines.append(f"Document {i} (Score: {doc.relevance_score:.2f}):")
                lines.append(f"  ID: {doc.doc_id}")
                if doc.metadata.get('source'):
                    lines.append(f"  Source: {doc.metadata['source']}")
                # Truncate content for readability
                content = doc.content[:200] + "..." if len(doc.content) > 200 else doc.content
                lines.append(f"  Content: {content}")
                lines.append("")
        
        # Step 3: Generation
        if journey.final_response:
            lines.append("-" * 80)
            lines.append("STEP 3: RESPONSE GENERATION")
            lines.append("-" * 80)
            if journey.generation_metadata:
                meta = journey.generation_metadata
                if meta.worker_id:
                    lines.append(f"Worker: {meta.worker_id}")
                if meta.instance_id:
                    lines.append(f"Instance: {meta.instance_id}")
                if meta.start_time:
                    lines.append(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(meta.start_time))}")
                if meta.end_time:
                    lines.append(f"End Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(meta.end_time))}")
                if meta.duration:
                    lines.append(f"Duration: {meta.duration:.2f}s")
            
            lines.append(f"Response Length: {len(journey.final_response)} characters")
            lines.append("")
            lines.append("Final Response:")
            lines.append(journey.final_response)
            lines.append("")
        
        # Error if present
        if journey.error:
            lines.append("-" * 80)
            lines.append("ERROR")
            lines.append("-" * 80)
            lines.append(journey.error)
            lines.append("")
        
        lines.append("=" * 80)
        lines.append("END OF QUERY JOURNEY")
        lines.append("=" * 80)
        
        with open(text_file, 'w') as f:
            f.write('\n'.join(lines))
    
    def save_all_journeys(self):
        """Save all journeys to files."""
        for query_id in self.journeys:
            self.save_journey(query_id)
    
    def get_journey(self, query_id: str) -> Optional[QueryJourney]:
        """Get journey by ID."""
        return self.journeys.get(query_id)
    
    def get_all_journeys(self) -> List[QueryJourney]:
        """Get all journeys."""
        return list(self.journeys.values())
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for all journeys."""
        total = len(self.journeys)
        complete = sum(1 for j in self.journeys.values() if j.status == "complete")
        failed = sum(1 for j in self.journeys.values() if j.status == "failed")
        
        # Calculate average times
        times = [j.get_total_time() for j in self.journeys.values() if j.get_total_time()]
        avg_time = sum(times) / len(times) if times else 0
        
        # Calculate average relevance
        relevances = [j.get_avg_relevance_score() for j in self.journeys.values() if j.get_avg_relevance_score()]
        avg_relevance = sum(relevances) / len(relevances) if relevances else 0
        
        return {
            'total_queries': total,
            'complete': complete,
            'failed': failed,
            'pending': total - complete - failed,
            'avg_total_time': avg_time,
            'avg_relevance_score': avg_relevance
        }

