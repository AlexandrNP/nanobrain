"""
Log Exporter
=============

Simple tool for exporting query journey logs to various formats.
Supports CSV, JSON, and text formats.
"""

import csv
import json
from pathlib import Path
from typing import List, Optional

import sys
from pathlib import Path

# Add the pbs_parallel_rag directory to path for local imports
_current_dir = Path(__file__).parent.parent
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

from tools.analyze_logs import LogAnalyzer, QueryStats


class LogExporter:
    """
    Exporter for query journey logs.
    
    Simple implementation:
    - Exports to CSV, JSON, text
    - Uses LogAnalyzer for data
    - No complex formatting
    """
    
    def __init__(self, analyzer: LogAnalyzer):
        """
        Initialize exporter.
        
        Args:
            analyzer: LogAnalyzer instance with loaded journeys
        """
        self.analyzer = analyzer
    
    def export_to_csv(self, output_file: str, queries: Optional[List[str]] = None):
        """
        Export query statistics to CSV.
        
        Args:
            output_file: Output CSV file path
            queries: Optional list of query IDs to export (None = all)
        """
        # Get query IDs
        if queries is None:
            queries = list(self.analyzer.journeys.keys())
        
        # Collect stats
        stats_list = []
        for query_id in queries:
            stat = self.analyzer.get_query_stats(query_id)
            if stat:
                stats_list.append(stat)
        
        # Write CSV
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'Query ID',
                'Status',
                'Total Time (s)',
                'Enhancement Time (s)',
                'Retrieval Time (s)',
                'Generation Time (s)',
                'Num Documents',
                'Avg Relevance',
                'Workers'
            ])
            
            # Data rows
            for stat in stats_list:
                writer.writerow([
                    stat.query_id,
                    stat.status,
                    f"{stat.total_time:.3f}",
                    f"{stat.enhancement_time:.3f}",
                    f"{stat.retrieval_time:.3f}",
                    f"{stat.generation_time:.3f}",
                    stat.num_documents,
                    f"{stat.avg_relevance:.2f}",
                    ','.join(stat.worker_ids)
                ])
        
        print(f"✓ Exported {len(stats_list)} queries to {output_file}")
    
    def export_to_json(self, output_file: str, queries: Optional[List[str]] = None):
        """
        Export query journeys to JSON.
        
        Args:
            output_file: Output JSON file path
            queries: Optional list of query IDs to export (None = all)
        """
        # Get query IDs
        if queries is None:
            queries = list(self.analyzer.journeys.keys())
        
        # Collect journeys
        journeys_data = []
        for query_id in queries:
            journey = self.analyzer.journeys.get(query_id)
            if journey:
                journeys_data.append(journey.to_dict())
        
        # Write JSON
        with open(output_file, 'w') as f:
            json.dump(journeys_data, f, indent=2)
        
        print(f"✓ Exported {len(journeys_data)} queries to {output_file}")
    
    def export_summary_to_text(self, output_file: str):
        """
        Export summary statistics to text file.
        
        Args:
            output_file: Output text file path
        """
        stats = self.analyzer.get_performance_stats()
        worker_stats = self.analyzer.get_worker_stats()
        
        lines = []
        lines.append("="*80)
        lines.append("QUERY JOURNEY ANALYSIS SUMMARY")
        lines.append("="*80)
        lines.append("")
        
        if stats:
            lines.append(f"Total Queries: {stats.get('total_queries', 0)}")
            lines.append(f"  Complete: {stats.get('complete', 0)}")
            lines.append(f"  Failed: {stats.get('failed', 0)}")
            lines.append(f"  Pending: {stats.get('pending', 0)}")
            lines.append("")

            lines.append("Performance:")
            lines.append(f"  Avg Total Time: {stats.get('avg_total_time', 0):.3f}s")
            lines.append(f"  Avg Enhancement Time: {stats.get('avg_enhancement_time', 0):.3f}s")
            lines.append(f"  Avg Retrieval Time: {stats.get('avg_retrieval_time', 0):.3f}s")
            lines.append(f"  Avg Generation Time: {stats.get('avg_generation_time', 0):.3f}s")
            lines.append(f"  Min Total Time: {stats.get('min_total_time', 0):.3f}s")
            lines.append(f"  Max Total Time: {stats.get('max_total_time', 0):.3f}s")
            lines.append("")

            lines.append("Quality:")
            lines.append(f"  Avg Relevance Score: {stats.get('avg_relevance_score', 0):.2f}")
            lines.append("")
        else:
            lines.append("No journey data available for analysis")
            lines.append("This may indicate that journey logging was not enabled")
            lines.append("or that the log files were not properly saved.")
            lines.append("")
        
        lines.append("Worker Utilization:")
        for worker_id, count in sorted(worker_stats.items()):
            lines.append(f"  {worker_id}: {count} tasks")
        lines.append("")
        
        # Slowest queries
        slowest = self.analyzer.get_slowest_queries(5)
        if slowest:
            lines.append("Top 5 Slowest Queries:")
            for i, stat in enumerate(slowest, 1):
                lines.append(f"  {i}. {stat.query_id}: {stat.total_time:.3f}s")
            lines.append("")
        
        # Lowest relevance
        lowest = self.analyzer.get_lowest_relevance_queries(5)
        if lowest:
            lines.append("Top 5 Lowest Relevance Queries:")
            for i, stat in enumerate(lowest, 1):
                lines.append(f"  {i}. {stat.query_id}: {stat.avg_relevance:.2f}")
            lines.append("")
        
        lines.append("="*80)
        
        # Write to file
        with open(output_file, 'w') as f:
            f.write('\n'.join(lines))
        
        print(f"✓ Exported summary to {output_file}")
    
    def export_detailed_report(self, output_file: str, queries: Optional[List[str]] = None):
        """
        Export detailed report with all query information.
        
        Args:
            output_file: Output text file path
            queries: Optional list of query IDs to export (None = all)
        """
        # Get query IDs
        if queries is None:
            queries = list(self.analyzer.journeys.keys())
        
        lines = []
        lines.append("="*80)
        lines.append("DETAILED QUERY JOURNEY REPORT")
        lines.append("="*80)
        lines.append("")
        
        for query_id in queries:
            journey = self.analyzer.journeys.get(query_id)
            if not journey:
                continue
            
            lines.append("-"*80)
            lines.append(f"Query ID: {query_id}")
            lines.append("-"*80)
            lines.append(f"Status: {journey.status}")
            lines.append(f"Original Query: {journey.original_query}")
            lines.append("")
            
            if journey.enhanced_query:
                lines.append(f"Enhanced Query: {journey.enhanced_query}")
                lines.append("")
            
            if journey.enhancement_metadata:
                meta = journey.enhancement_metadata
                lines.append("Enhancement:")
                if meta.worker_id:
                    lines.append(f"  Worker: {meta.worker_id}")
                if meta.duration:
                    lines.append(f"  Time: {meta.duration:.3f}s")
                lines.append("")
            
            if journey.retrieved_documents:
                lines.append(f"Retrieved Documents: {len(journey.retrieved_documents)}")
                avg_rel = journey.get_avg_relevance_score()
                if avg_rel:
                    lines.append(f"  Avg Relevance: {avg_rel:.2f}")
                lines.append("")
            
            if journey.retrieval_metadata:
                meta = journey.retrieval_metadata
                lines.append("Retrieval:")
                if meta.worker_id:
                    lines.append(f"  Worker: {meta.worker_id}")
                if meta.duration:
                    lines.append(f"  Time: {meta.duration:.3f}s")
                lines.append("")
            
            if journey.final_response:
                lines.append(f"Response Length: {len(journey.final_response)} characters")
                lines.append("")
            
            if journey.generation_metadata:
                meta = journey.generation_metadata
                lines.append("Generation:")
                if meta.worker_id:
                    lines.append(f"  Worker: {meta.worker_id}")
                if meta.duration:
                    lines.append(f"  Time: {meta.duration:.3f}s")
                lines.append("")
            
            total_time = journey.get_total_time()
            if total_time:
                lines.append(f"Total Time: {total_time:.3f}s")
            
            lines.append("")
        
        lines.append("="*80)
        
        # Write to file
        with open(output_file, 'w') as f:
            f.write('\n'.join(lines))
        
        print(f"✓ Exported detailed report for {len(queries)} queries to {output_file}")

