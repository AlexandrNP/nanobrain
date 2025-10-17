"""
View Journey
=============

Simple tool for viewing individual query journeys.
"""

import json
from pathlib import Path
from typing import Optional

from demos.parallel_rag_with_parsl.models.query_journey import QueryJourney


def view_journey(
    query_id: str,
    log_dir: str = "demos/parallel_rag_with_parsl/output/logs",
    format: str = "text"
):
    """
    View a specific query journey.
    
    Args:
        query_id: Query identifier
        log_dir: Directory containing query logs
        format: Output format ("text" or "json")
    """
    log_path = Path(log_dir)
    queries_dir = log_path / "queries"
    
    # Try to load journey
    if format == "json":
        json_file = queries_dir / f"{query_id}.json"
        if not json_file.exists():
            print(f"❌ Journey not found: {query_id}")
            return
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        print(json.dumps(data, indent=2))
    
    else:  # text format
        text_file = queries_dir / f"{query_id}.txt"
        if not text_file.exists():
            print(f"❌ Journey not found: {query_id}")
            return
        
        with open(text_file, 'r') as f:
            print(f.read())


def list_journeys(log_dir: str = "demos/parallel_rag_with_parsl/output/logs"):
    """
    List all available query journeys.
    
    Args:
        log_dir: Directory containing query logs
    """
    log_path = Path(log_dir)
    queries_dir = log_path / "queries"
    
    if not queries_dir.exists():
        print(f"❌ Log directory not found: {queries_dir}")
        return
    
    json_files = list(queries_dir.glob("*.json"))
    
    if not json_files:
        print(f"No journeys found in {queries_dir}")
        return
    
    print("="*80)
    print(f"AVAILABLE QUERY JOURNEYS ({len(json_files)})")
    print("="*80)
    
    for json_file in sorted(json_files):
        query_id = json_file.stem
        
        # Load basic info
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            status = data.get('status', 'unknown')
            original_query = data.get('original_query', '')
            total_time = data.get('total_time', 0)
            
            print(f"\n{query_id}:")
            print(f"  Status: {status}")
            print(f"  Query: {original_query[:60]}...")
            if total_time:
                print(f"  Time: {total_time:.3f}s")
        
        except Exception as e:
            print(f"\n{query_id}: Error loading - {e}")
    
    print("\n" + "="*80)


def compare_journeys(
    query_ids: list,
    log_dir: str = "demos/parallel_rag_with_parsl/output/logs"
):
    """
    Compare multiple query journeys side by side.
    
    Args:
        query_ids: List of query IDs to compare
        log_dir: Directory containing query logs
    """
    log_path = Path(log_dir)
    queries_dir = log_path / "queries"
    
    journeys = []
    for query_id in query_ids:
        json_file = queries_dir / f"{query_id}.json"
        if not json_file.exists():
            print(f"❌ Journey not found: {query_id}")
            continue
        
        with open(json_file, 'r') as f:
            data = json.load(f)
            journey = QueryJourney.from_dict(data)
            journeys.append(journey)
    
    if not journeys:
        print("No journeys to compare")
        return
    
    print("="*80)
    print(f"JOURNEY COMPARISON ({len(journeys)} queries)")
    print("="*80)
    
    # Header
    print(f"\n{'Metric':<30}", end='')
    for journey in journeys:
        print(f"{journey.query_id:<20}", end='')
    print()
    print("-"*80)
    
    # Status
    print(f"{'Status':<30}", end='')
    for journey in journeys:
        print(f"{journey.status:<20}", end='')
    print()
    
    # Total time
    print(f"{'Total Time (s)':<30}", end='')
    for journey in journeys:
        total_time = journey.get_total_time()
        if total_time:
            print(f"{total_time:<20.3f}", end='')
        else:
            print(f"{'N/A':<20}", end='')
    print()
    
    # Enhancement time
    print(f"{'Enhancement Time (s)':<30}", end='')
    for journey in journeys:
        if journey.enhancement_metadata and journey.enhancement_metadata.duration:
            print(f"{journey.enhancement_metadata.duration:<20.3f}", end='')
        else:
            print(f"{'N/A':<20}", end='')
    print()
    
    # Retrieval time
    print(f"{'Retrieval Time (s)':<30}", end='')
    for journey in journeys:
        if journey.retrieval_metadata and journey.retrieval_metadata.duration:
            print(f"{journey.retrieval_metadata.duration:<20.3f}", end='')
        else:
            print(f"{'N/A':<20}", end='')
    print()
    
    # Generation time
    print(f"{'Generation Time (s)':<30}", end='')
    for journey in journeys:
        if journey.generation_metadata and journey.generation_metadata.duration:
            print(f"{journey.generation_metadata.duration:<20.3f}", end='')
        else:
            print(f"{'N/A':<20}", end='')
    print()
    
    # Number of documents
    print(f"{'Num Documents':<30}", end='')
    for journey in journeys:
        print(f"{len(journey.retrieved_documents):<20}", end='')
    print()
    
    # Average relevance
    print(f"{'Avg Relevance':<30}", end='')
    for journey in journeys:
        avg_rel = journey.get_avg_relevance_score()
        if avg_rel:
            print(f"{avg_rel:<20.2f}", end='')
        else:
            print(f"{'N/A':<20}", end='')
    print()
    
    print("="*80)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python view_journey.py <query_id> [format]")
        print("  python view_journey.py --list")
        print("  python view_journey.py --compare <query_id1> <query_id2> ...")
        sys.exit(1)
    
    if sys.argv[1] == "--list":
        list_journeys()
    elif sys.argv[1] == "--compare":
        if len(sys.argv) < 3:
            print("Error: --compare requires at least one query ID")
            sys.exit(1)
        compare_journeys(sys.argv[2:])
    else:
        query_id = sys.argv[1]
        format = sys.argv[2] if len(sys.argv) > 2 else "text"
        view_journey(query_id, format=format)

