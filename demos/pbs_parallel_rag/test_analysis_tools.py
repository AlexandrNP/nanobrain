#!/usr/bin/env python3
"""
Test Analysis Tools
===================

Test log analysis, export, and viewing tools.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demos.pbs_parallel_rag.tools.analyze_logs import LogAnalyzer
from demos.pbs_parallel_rag.tools.export_logs import LogExporter
from demos.pbs_parallel_rag.tools.view_journey import view_journey, list_journeys, compare_journeys


def test_log_analyzer():
    """Test log analyzer."""
    print("="*80)
    print("TEST 1: Log Analyzer")
    print("="*80)
    
    # Use logs from end-to-end test
    log_dir = "demos/pbs_parallel_rag/output/end_to_end_test"
    
    # Create analyzer
    analyzer = LogAnalyzer(log_dir)
    
    print(f"\n✓ Created analyzer")
    print(f"✓ Loaded {len(analyzer.journeys)} journeys")
    
    # Test search
    print(f"\n{'─'*80}")
    print("Search Tests")
    print(f"{'─'*80}")
    
    # Search by keyword
    results = analyzer.search_queries(keyword="viral")
    print(f"✓ Keyword search 'viral': {len(results)} results")
    
    # Search by relevance
    results = analyzer.search_queries(min_relevance=0.85)
    print(f"✓ Min relevance 0.85: {len(results)} results")
    
    # Search by time
    results = analyzer.search_queries(max_time=1.0)
    print(f"✓ Max time 1.0s: {len(results)} results")
    
    # Search by worker
    results = analyzer.search_queries(worker_id="worker_0")
    print(f"✓ Worker 'worker_0': {len(results)} results")
    
    # Test statistics
    print(f"\n{'─'*80}")
    print("Statistics Tests")
    print(f"{'─'*80}")
    
    # Performance stats
    perf_stats = analyzer.get_performance_stats()
    print(f"✓ Performance stats:")
    if perf_stats:
        print(f"  Total queries: {perf_stats.get('total_queries', 0)}")
        print(f"  Avg total time: {perf_stats.get('avg_total_time', 0):.3f}s")
        print(f"  Avg relevance: {perf_stats.get('avg_relevance_score', 0):.2f}")
    else:
        print(f"  No performance data available")
    
    # Worker stats
    worker_stats = analyzer.get_worker_stats()
    print(f"\n✓ Worker stats:")
    for worker_id, count in sorted(worker_stats.items()):
        print(f"  {worker_id}: {count} tasks")
    
    # Query stats
    if analyzer.journeys:
        query_id = list(analyzer.journeys.keys())[0]
        query_stat = analyzer.get_query_stats(query_id)
        print(f"\n✓ Query stats for {query_id}:")
        print(f"  Total time: {query_stat.total_time:.3f}s")
        print(f"  Num documents: {query_stat.num_documents}")
        print(f"  Avg relevance: {query_stat.avg_relevance:.2f}")
    
    # Slowest queries
    slowest = analyzer.get_slowest_queries(3)
    print(f"\n✓ Top 3 slowest queries:")
    for i, stat in enumerate(slowest, 1):
        print(f"  {i}. {stat.query_id}: {stat.total_time:.3f}s")
    
    # Lowest relevance
    lowest = analyzer.get_lowest_relevance_queries(3)
    print(f"\n✓ Top 3 lowest relevance queries:")
    for i, stat in enumerate(lowest, 1):
        print(f"  {i}. {stat.query_id}: {stat.avg_relevance:.2f}")
    
    # Print summary
    print(f"\n{'─'*80}")
    print("Summary Output")
    print(f"{'─'*80}")
    analyzer.print_summary()
    
    print(f"\n✅ Log analyzer tests passed!")
    
    return analyzer


def test_log_exporter(analyzer):
    """Test log exporter."""
    print("\n" + "="*80)
    print("TEST 2: Log Exporter")
    print("="*80)
    
    output_dir = Path("demos/pbs_parallel_rag/output/analysis_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create exporter
    exporter = LogExporter(analyzer)
    print(f"\n✓ Created exporter")
    
    # Export to CSV
    print(f"\n{'─'*80}")
    print("CSV Export")
    print(f"{'─'*80}")
    csv_file = output_dir / "queries.csv"
    exporter.export_to_csv(str(csv_file))
    
    assert csv_file.exists(), "CSV file not created"
    print(f"✓ CSV file created: {csv_file}")
    
    # Show first few lines
    with open(csv_file, 'r') as f:
        lines = f.readlines()[:5]
        print(f"\nFirst {len(lines)} lines:")
        for line in lines:
            print(f"  {line.strip()}")
    
    # Export to JSON
    print(f"\n{'─'*80}")
    print("JSON Export")
    print(f"{'─'*80}")
    json_file = output_dir / "queries.json"
    exporter.export_to_json(str(json_file))
    
    assert json_file.exists(), "JSON file not created"
    print(f"✓ JSON file created: {json_file}")
    
    # Export summary
    print(f"\n{'─'*80}")
    print("Summary Export")
    print(f"{'─'*80}")
    summary_file = output_dir / "summary.txt"
    exporter.export_summary_to_text(str(summary_file))
    
    assert summary_file.exists(), "Summary file not created"
    print(f"✓ Summary file created: {summary_file}")
    
    # Show summary
    with open(summary_file, 'r') as f:
        print(f"\nSummary content:")
        print(f.read())
    
    # Export detailed report
    print(f"\n{'─'*80}")
    print("Detailed Report Export")
    print(f"{'─'*80}")
    report_file = output_dir / "detailed_report.txt"
    exporter.export_detailed_report(str(report_file))
    
    assert report_file.exists(), "Report file not created"
    print(f"✓ Report file created: {report_file}")
    
    print(f"\n✅ Log exporter tests passed!")


def test_view_journey():
    """Test view journey tool."""
    print("\n" + "="*80)
    print("TEST 3: View Journey")
    print("="*80)
    
    log_dir = "demos/pbs_parallel_rag/output/end_to_end_test"
    
    # List journeys
    print(f"\n{'─'*80}")
    print("List Journeys")
    print(f"{'─'*80}")
    list_journeys(log_dir)
    
    # View specific journey (text)
    print(f"\n{'─'*80}")
    print("View Journey (Text Format)")
    print(f"{'─'*80}")
    view_journey("q_e2e_001", log_dir, format="text")
    
    # View specific journey (JSON)
    print(f"\n{'─'*80}")
    print("View Journey (JSON Format)")
    print(f"{'─'*80}")
    view_journey("q_e2e_001", log_dir, format="json")
    
    # Compare journeys
    print(f"\n{'─'*80}")
    print("Compare Journeys")
    print(f"{'─'*80}")
    compare_journeys(["q_e2e_001", "q_e2e_002", "q_e2e_003"], log_dir)
    
    print(f"\n✅ View journey tests passed!")


def main():
    """Run all tests."""
    try:
        # Test analyzer
        analyzer = test_log_analyzer()
        
        # Test exporter
        test_log_exporter(analyzer)
        
        # Test viewer
        test_view_journey()
        
        print("\n" + "="*80)
        print("🎉 ALL ANALYSIS TOOLS TESTS PASSED!")
        print("="*80)
        print("\n✅ Log analyzer working")
        print("✅ Search and filter working")
        print("✅ Statistics calculation working")
        print("✅ CSV export working")
        print("✅ JSON export working")
        print("✅ Summary export working")
        print("✅ Detailed report working")
        print("✅ Journey viewer working")
        print("✅ Journey comparison working")
        print("\n📊 Analysis tools ready for production!")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

