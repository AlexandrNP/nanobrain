"""
Log Analyzer
=============

Simple tool for analyzing query journey logs.
Provides search, filter, and statistical analysis capabilities.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import sys
from pathlib import Path

# Add the pbs_parallel_rag directory to path for local imports
_current_dir = Path(__file__).parent.parent
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

from models.query_journey import QueryJourney


@dataclass
class QueryStats:
    """Statistics for a query journey."""
    query_id: str
    total_time: float
    enhancement_time: float
    retrieval_time: float
    generation_time: float
    num_documents: int
    avg_relevance: float
    worker_ids: List[str]
    status: str


class LogAnalyzer:
    """
    Analyzer for query journey logs.
    
    Simple implementation:
    - Loads journeys from JSON files
    - Provides search and filter
    - Calculates statistics
    - No complex state management
    """
    
    def __init__(self, log_dir: str):
        """
        Initialize analyzer.
        
        Args:
            log_dir: Directory containing query logs
        """
        self.log_dir = Path(log_dir)
        self.queries_dir = self.log_dir / "queries"
        self.journeys: Dict[str, QueryJourney] = {}
        
        # Load all journeys
        self._load_journeys()
    
    def _load_journeys(self):
        """Load all journey JSON files."""
        if not self.queries_dir.exists():
            return
        
        for json_file in self.queries_dir.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    journey = QueryJourney.from_dict(data)
                    self.journeys[journey.query_id] = journey
            except Exception as e:
                print(f"Warning: Failed to load {json_file}: {e}")
    
    def search_queries(
        self,
        keyword: Optional[str] = None,
        min_relevance: Optional[float] = None,
        max_time: Optional[float] = None,
        worker_id: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[QueryJourney]:
        """
        Search queries with filters.
        
        Args:
            keyword: Search in original query text
            min_relevance: Minimum average relevance score
            max_time: Maximum total processing time
            worker_id: Filter by worker ID
            status: Filter by status (complete, failed, etc.)
            
        Returns:
            List of matching journeys
        """
        results = []
        
        for journey in self.journeys.values():
            # Keyword filter
            if keyword and keyword.lower() not in journey.original_query.lower():
                continue
            
            # Relevance filter
            if min_relevance:
                avg_rel = journey.get_avg_relevance_score()
                if not avg_rel or avg_rel < min_relevance:
                    continue
            
            # Time filter
            if max_time:
                total_time = journey.get_total_time()
                if not total_time or total_time > max_time:
                    continue
            
            # Worker filter
            if worker_id:
                worker_found = False
                for meta in [journey.enhancement_metadata, journey.retrieval_metadata, journey.generation_metadata]:
                    if meta and meta.worker_id == worker_id:
                        worker_found = True
                        break
                if not worker_found:
                    continue
            
            # Status filter
            if status and journey.status != status:
                continue
            
            results.append(journey)
        
        return results
    
    def get_query_stats(self, query_id: str) -> Optional[QueryStats]:
        """
        Get statistics for a specific query.
        
        Args:
            query_id: Query identifier
            
        Returns:
            QueryStats object or None
        """
        journey = self.journeys.get(query_id)
        if not journey:
            return None
        
        # Extract times
        enhancement_time = journey.enhancement_metadata.duration if journey.enhancement_metadata else 0.0
        retrieval_time = journey.retrieval_metadata.duration if journey.retrieval_metadata else 0.0
        generation_time = journey.generation_metadata.duration if journey.generation_metadata else 0.0
        total_time = journey.get_total_time() or 0.0
        
        # Extract worker IDs
        worker_ids = []
        for meta in [journey.enhancement_metadata, journey.retrieval_metadata, journey.generation_metadata]:
            if meta and meta.worker_id:
                worker_ids.append(meta.worker_id)
        
        return QueryStats(
            query_id=query_id,
            total_time=total_time,
            enhancement_time=enhancement_time,
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            num_documents=len(journey.retrieved_documents),
            avg_relevance=journey.get_avg_relevance_score() or 0.0,
            worker_ids=worker_ids,
            status=journey.status
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get overall performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.journeys:
            return {}
        
        total_queries = len(self.journeys)
        complete = sum(1 for j in self.journeys.values() if j.status == "complete")
        failed = sum(1 for j in self.journeys.values() if j.status == "failed")
        
        # Calculate average times
        total_times = [j.get_total_time() for j in self.journeys.values() if j.get_total_time()]
        enhancement_times = [j.enhancement_metadata.duration for j in self.journeys.values() 
                           if j.enhancement_metadata and j.enhancement_metadata.duration]
        retrieval_times = [j.retrieval_metadata.duration for j in self.journeys.values() 
                         if j.retrieval_metadata and j.retrieval_metadata.duration]
        generation_times = [j.generation_metadata.duration for j in self.journeys.values() 
                          if j.generation_metadata and j.generation_metadata.duration]
        
        # Calculate average relevance
        relevances = [j.get_avg_relevance_score() for j in self.journeys.values() 
                     if j.get_avg_relevance_score()]
        
        return {
            'total_queries': total_queries,
            'complete': complete,
            'failed': failed,
            'pending': total_queries - complete - failed,
            'avg_total_time': sum(total_times) / len(total_times) if total_times else 0,
            'avg_enhancement_time': sum(enhancement_times) / len(enhancement_times) if enhancement_times else 0,
            'avg_retrieval_time': sum(retrieval_times) / len(retrieval_times) if retrieval_times else 0,
            'avg_generation_time': sum(generation_times) / len(generation_times) if generation_times else 0,
            'avg_relevance_score': sum(relevances) / len(relevances) if relevances else 0,
            'min_total_time': min(total_times) if total_times else 0,
            'max_total_time': max(total_times) if total_times else 0
        }
    
    def get_worker_stats(self) -> Dict[str, int]:
        """
        Get worker utilization statistics.
        
        Returns:
            Dictionary mapping worker IDs to query counts
        """
        worker_counts = {}
        
        for journey in self.journeys.values():
            for meta in [journey.enhancement_metadata, journey.retrieval_metadata, journey.generation_metadata]:
                if meta and meta.worker_id:
                    worker_counts[meta.worker_id] = worker_counts.get(meta.worker_id, 0) + 1
        
        return worker_counts
    
    def get_slowest_queries(self, n: int = 10) -> List[QueryStats]:
        """
        Get the N slowest queries.
        
        Args:
            n: Number of queries to return
            
        Returns:
            List of QueryStats sorted by total time (descending)
        """
        stats = []
        for query_id in self.journeys:
            stat = self.get_query_stats(query_id)
            if stat and stat.total_time > 0:
                stats.append(stat)
        
        stats.sort(key=lambda x: x.total_time, reverse=True)
        return stats[:n]
    
    def get_lowest_relevance_queries(self, n: int = 10) -> List[QueryStats]:
        """
        Get the N queries with lowest relevance scores.
        
        Args:
            n: Number of queries to return
            
        Returns:
            List of QueryStats sorted by relevance (ascending)
        """
        stats = []
        for query_id in self.journeys:
            stat = self.get_query_stats(query_id)
            if stat and stat.avg_relevance > 0:
                stats.append(stat)
        
        stats.sort(key=lambda x: x.avg_relevance)
        return stats[:n]
    
    def print_summary(self):
        """Print a summary of all queries."""
        stats = self.get_performance_stats()

        print("="*80)
        print("QUERY JOURNEY ANALYSIS SUMMARY")
        print("="*80)

        if stats:
            print(f"\nTotal Queries: {stats.get('total_queries', 0)}")
            print(f"  Complete: {stats.get('complete', 0)}")
            print(f"  Failed: {stats.get('failed', 0)}")
            print(f"  Pending: {stats.get('pending', 0)}")

            print(f"\nPerformance:")
            print(f"  Avg Total Time: {stats.get('avg_total_time', 0):.3f}s")
            print(f"  Avg Enhancement Time: {stats.get('avg_enhancement_time', 0):.3f}s")
            print(f"  Avg Retrieval Time: {stats.get('avg_retrieval_time', 0):.3f}s")
            print(f"  Avg Generation Time: {stats.get('avg_generation_time', 0):.3f}s")
            print(f"  Min Total Time: {stats.get('min_total_time', 0):.3f}s")
            print(f"  Max Total Time: {stats.get('max_total_time', 0):.3f}s")

            print(f"\nQuality:")
            print(f"  Avg Relevance Score: {stats.get('avg_relevance_score', 0):.2f}")
        else:
            print(f"\nNo journey data available for analysis")
            print(f"This may indicate that journey logging was not enabled")
            print(f"or that the log files were not properly saved.")

        print(f"\nWorker Utilization:")
        worker_stats = self.get_worker_stats()
        if worker_stats:
            for worker_id, count in sorted(worker_stats.items()):
                print(f"  {worker_id}: {count} tasks")
        else:
            print(f"  No worker data available")
        
        print("="*80)

