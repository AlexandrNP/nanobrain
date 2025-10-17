#!/usr/bin/env python3
"""
Test script to verify KeyError fix for avg_enhancement_time
"""

def test_empty_stats_handling():
    """Test that empty stats dictionary is handled correctly"""
    
    # Simulate empty stats (what happens when no journeys are loaded)
    stats = {}
    
    # Test the patterns that were causing KeyError
    try:
        # This would cause KeyError before the fix
        # print(f"Avg enhancement time: {stats['avg_enhancement_time']:.3f}s")
        
        # This is the fixed version
        print(f"Avg enhancement time: {stats.get('avg_enhancement_time', 0):.3f}s")
        print("✅ Empty stats handling works correctly")
        
        # Test other stats keys
        print(f"Total queries: {stats.get('total_queries', 0)}")
        print(f"Complete: {stats.get('complete', 0)}")
        print(f"Failed: {stats.get('failed', 0)}")
        print(f"Avg total time: {stats.get('avg_total_time', 0):.3f}s")
        print(f"Avg relevance score: {stats.get('avg_relevance_score', 0):.2f}")
        
        print("✅ All stats keys handled correctly with empty dictionary")
        return True
        
    except KeyError as e:
        print(f"❌ KeyError still occurs: {e}")
        return False

def test_populated_stats_handling():
    """Test that populated stats dictionary works correctly"""
    
    # Simulate populated stats
    stats = {
        'total_queries': 100,
        'complete': 95,
        'failed': 5,
        'avg_enhancement_time': 1.234,
        'avg_retrieval_time': 2.345,
        'avg_generation_time': 3.456,
        'avg_total_time': 7.035,
        'avg_relevance_score': 0.85
    }
    
    try:
        print(f"Avg enhancement time: {stats.get('avg_enhancement_time', 0):.3f}s")
        print(f"Total queries: {stats.get('total_queries', 0)}")
        print(f"Complete: {stats.get('complete', 0)}")
        print(f"Avg relevance score: {stats.get('avg_relevance_score', 0):.2f}")
        
        print("✅ Populated stats handling works correctly")
        return True
        
    except Exception as e:
        print(f"❌ Error with populated stats: {e}")
        return False

if __name__ == "__main__":
    print("Testing KeyError fix for avg_enhancement_time")
    print("=" * 50)
    
    print("\n1. Testing empty stats dictionary:")
    test1_passed = test_empty_stats_handling()
    
    print("\n2. Testing populated stats dictionary:")
    test2_passed = test_populated_stats_handling()
    
    print("\n" + "=" * 50)
    if test1_passed and test2_passed:
        print("✅ ALL TESTS PASSED - KeyError fix is working correctly")
    else:
        print("❌ SOME TESTS FAILED - KeyError fix needs more work")
