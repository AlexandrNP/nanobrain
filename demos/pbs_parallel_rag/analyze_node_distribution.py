#!/usr/bin/env python3
"""
Aurora Node Distribution Analyzer
=================================

Analyzes node distribution and resource pool usage from Aurora stress tests.
"""

import json
import sys
from pathlib import Path


def analyze_node_distribution(node_tracking_file):
    """Analyze node distribution from tracking data."""
    
    if not Path(node_tracking_file).exists():
        print(f"❌ Node tracking file not found: {node_tracking_file}")
        return False
    
    try:
        with open(node_tracking_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Error reading node tracking file: {e}")
        return False
    
    print("🔍 AURORA NODE DISTRIBUTION ANALYSIS")
    print("=" * 80)
    
    # Basic statistics
    expected_nodes = data.get('expected_nodes', [])
    node_usage = data.get('node_usage', {})
    queries_per_node = data.get('queries_per_node', {})
    resource_pool_usage = data.get('resource_pool_usage', 0)
    max_pool_size = data.get('max_pool_size', 0)
    total_queries = data.get('total_queries', 0)
    completed_queries = data.get('completed_queries', 0)
    
    print(f"\nBasic Statistics:")
    print(f"  Expected nodes: {len(expected_nodes)}")
    print(f"  Actual nodes used: {len(node_usage)}")
    print(f"  Total queries: {total_queries}")
    print(f"  Completed queries: {completed_queries}")
    
    # Resource pool analysis
    print(f"\nResource Pool Analysis:")
    print(f"  Max pool size: {max_pool_size}")
    print(f"  Peak usage: {resource_pool_usage}")
    print(f"  Utilization: {resource_pool_usage/max_pool_size*100:.1f}%" if max_pool_size > 0 else "  Utilization: N/A")
    
    if resource_pool_usage <= max_pool_size:
        print(f"  ✅ Resource pool compliance: PASSED")
    else:
        print(f"  ❌ Resource pool compliance: FAILED")
        print(f"     Exceeded by: {resource_pool_usage - max_pool_size} resources")
    
    # Node coverage analysis
    print(f"\nNode Coverage Analysis:")
    expected_set = set(expected_nodes)
    actual_set = set(node_usage.keys())
    
    if expected_set:
        coverage = len(actual_set & expected_set) / len(expected_set) * 100
        print(f"  Node coverage: {coverage:.1f}%")
        
        unused_nodes = expected_set - actual_set
        unexpected_nodes = actual_set - expected_set
        
        if unused_nodes:
            print(f"  ⚠️  Unused nodes ({len(unused_nodes)}): {list(unused_nodes)}")
        else:
            print(f"  ✅ All expected nodes were utilized")
            
        if unexpected_nodes:
            print(f"  ⚠️  Unexpected nodes ({len(unexpected_nodes)}): {list(unexpected_nodes)}")
    else:
        print(f"  ⚠️  No expected nodes specified")
    
    # Query distribution analysis
    print(f"\nQuery Distribution Analysis:")
    if queries_per_node:
        total_distributed = sum(queries_per_node.values())
        print(f"  Total queries distributed: {total_distributed}")
        
        # Calculate distribution statistics
        query_counts = list(queries_per_node.values())
        if query_counts:
            min_queries = min(query_counts)
            max_queries = max(query_counts)
            avg_queries = sum(query_counts) / len(query_counts)
            
            print(f"  Distribution statistics:")
            print(f"    Min queries per node: {min_queries}")
            print(f"    Max queries per node: {max_queries}")
            print(f"    Average queries per node: {avg_queries:.1f}")
            print(f"    Load balance ratio: {min_queries/max_queries:.2f}" if max_queries > 0 else "    Load balance ratio: N/A")
        
        print(f"\nPer-node breakdown:")
        for node, count in sorted(queries_per_node.items()):
            percentage = (count / total_distributed * 100) if total_distributed > 0 else 0
            print(f"    {node}: {count} queries ({percentage:.1f}%)")
    else:
        print(f"  ⚠️  No query distribution data available")
    
    # Performance implications
    print(f"\nPerformance Implications:")
    if len(actual_set) < len(expected_set):
        print(f"  ⚠️  Underutilized cluster: Only {len(actual_set)}/{len(expected_set)} nodes used")
        print(f"     This may indicate load balancing issues or insufficient parallelism")
    elif len(actual_set) == len(expected_set):
        print(f"  ✅ Optimal cluster utilization: All allocated nodes used")
    
    if resource_pool_usage < max_pool_size:
        unused_resources = max_pool_size - resource_pool_usage
        print(f"  💡 Potential for scaling: {unused_resources} unused resource slots")
    
    # Recommendations
    print(f"\nRecommendations:")
    if resource_pool_usage > max_pool_size:
        print(f"  🔧 Increase max_pool_size to at least {resource_pool_usage}")
    
    if len(actual_set) < len(expected_set):
        print(f"  🔧 Consider increasing parallelism to utilize all {len(expected_set)} nodes")
    
    if queries_per_node:
        query_counts = list(queries_per_node.values())
        if max(query_counts) / min(query_counts) > 2.0:
            print(f"  🔧 Load balancing could be improved (ratio: {max(query_counts)/min(query_counts):.1f})")
    
    return True


def main():
    """Main function."""
    if len(sys.argv) != 2:
        print("Usage: python3 analyze_node_distribution.py <node_tracking.json>")
        print("\nExample:")
        print("  python3 analyze_node_distribution.py output/stress_test_*/node_tracking.json")
        return 1
    
    node_tracking_file = sys.argv[1]
    
    if analyze_node_distribution(node_tracking_file):
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
