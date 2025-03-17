#!/usr/bin/env python3
"""
Test runner for component reuse functionality.

This script runs tests to verify that the refactored classes properly prioritize
reusing existing components with custom configurations instead of creating new classes.
"""

import os
import sys
import unittest
import asyncio
from unittest.mock import patch

# Add the parent directory to the path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import the test class
from test.test_component_reuse import TestComponentReuse


def run_test_case(test_case):
    """
    Run a specific test case with proper async handling.
    
    Args:
        test_case: Test case method to run
    """
    test_instance = TestComponentReuse()
    test_instance.setUp()
    
    # Check if test case is async
    if asyncio.iscoroutinefunction(getattr(TestComponentReuse, test_case)):
        # Run async test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(getattr(test_instance, test_case)())
        finally:
            loop.close()
    else:
        # Run regular test
        getattr(test_instance, test_case)()
    
    test_instance.tearDown()
    return True


def run_all_component_reuse_tests():
    """Run all component reuse tests with appropriate mocking."""
    # Get all test methods from the TestComponentReuse class
    test_cases = [method for method in dir(TestComponentReuse) 
                  if method.startswith('test_')]
    
    # Print header
    print("\n" + "="*60)
    print("RUNNING COMPONENT REUSE TESTS")
    print("="*60)
    
    # Run each test case
    results = {}
    for test_case in test_cases:
        print(f"\nRunning test: {test_case}")
        try:
            success = run_test_case(test_case)
            results[test_case] = "PASSED" if success else "FAILED"
            print(f"  ✅ {test_case} - PASSED")
        except Exception as e:
            results[test_case] = f"ERROR: {str(e)}"
            print(f"  ❌ {test_case} - FAILED: {str(e)}")
    
    # Print summary
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    
    passed = sum(1 for result in results.values() if result == "PASSED")
    total = len(results)
    
    for test_case, result in results.items():
        status = "✅" if result == "PASSED" else "❌"
        print(f"{status} {test_case}: {result}")
    
    print("\n" + "-"*60)
    print(f"PASSED: {passed}/{total} tests")
    print("-"*60)
    
    return passed == total


if __name__ == "__main__":
    success = run_all_component_reuse_tests()
    sys.exit(0 if success else 1) 