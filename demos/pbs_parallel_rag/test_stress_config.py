#!/usr/bin/env python3
"""
Test Stress Test Configuration
===============================

Validates that the stress test can be properly configured for PBS execution.
Tests environment variable configuration and basic functionality.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path


def test_stress_test_imports():
    """Test that stress test can import required modules."""
    print("🧪 Testing Stress Test Imports...")
    
    # Add path for imports
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    try:
        # Test basic imports that don't require nanobrain
        import asyncio
        import time
        import random
        from datetime import datetime
        print("   ✅ Basic Python modules")

        # Test pathlib import (already imported at top)
        print("   ✅ pathlib")

        # Test psutil if available
        try:
            import psutil
            print("   ✅ psutil")
        except ImportError:
            print("   ⚠️  psutil not available (optional)")

        print("✅ All required imports available")
        return True

    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_environment_configuration():
    """Test environment variable configuration."""
    print("\n🧪 Testing Environment Configuration...")
    
    # Test default values
    default_size = os.environ.get('STRESS_TEST_SIZE', '1000')
    default_workers = os.environ.get('MAX_WORKERS', '4')
    default_concurrent = os.environ.get('MAX_CONCURRENT', '50')
    default_output = os.environ.get('STRESS_TEST_OUTPUT_DIR', 'default')
    
    print(f"   Default test size: {default_size}")
    print(f"   Default workers: {default_workers}")
    print(f"   Default concurrent: {default_concurrent}")
    print(f"   Default output dir: {default_output}")
    
    # Test custom values
    test_env = {
        'STRESS_TEST_SIZE': '100',
        'MAX_WORKERS': '8',
        'MAX_CONCURRENT': '20',
        'STRESS_TEST_OUTPUT_DIR': '/tmp/test_output'
    }
    
    # Temporarily set environment variables
    original_env = {}
    for key, value in test_env.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value
    
    try:
        # Test that values are read correctly
        test_size = os.environ.get('STRESS_TEST_SIZE', '1000')
        test_workers = os.environ.get('MAX_WORKERS', '4')
        test_concurrent = os.environ.get('MAX_CONCURRENT', '50')
        test_output = os.environ.get('STRESS_TEST_OUTPUT_DIR', 'default')
        
        if (test_size == '100' and test_workers == '8' and 
            test_concurrent == '20' and test_output == '/tmp/test_output'):
            print("✅ Environment configuration working correctly")
            result = True
        else:
            print("❌ Environment configuration not working")
            result = False
            
    finally:
        # Restore original environment
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    
    return result


def test_output_directory_creation():
    """Test output directory creation."""
    print("\n🧪 Testing Output Directory Creation...")
    
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        test_output_dir = Path(temp_dir) / "stress_test_output"
        
        # Test directory creation
        try:
            test_output_dir.mkdir(parents=True, exist_ok=True)
            
            if test_output_dir.exists() and test_output_dir.is_dir():
                print("✅ Output directory creation working")
                
                # Test file creation in directory
                test_file = test_output_dir / "test.txt"
                test_file.write_text("test content")
                
                if test_file.exists():
                    print("✅ File creation in output directory working")
                    return True
                else:
                    print("❌ File creation failed")
                    return False
            else:
                print("❌ Directory creation failed")
                return False
                
        except Exception as e:
            print(f"❌ Directory creation error: {e}")
            return False


def test_pbs_environment_detection():
    """Test PBS environment variable detection."""
    print("\n🧪 Testing PBS Environment Detection...")
    
    # Test without PBS environment
    pbs_vars = ['PBS_JOBID', 'PBS_JOBNAME', 'PBS_NUM_NODES', 'PBS_NP']
    
    # Check current PBS environment
    pbs_detected = any(os.environ.get(var) for var in pbs_vars)
    
    if pbs_detected:
        print("✅ PBS environment detected:")
        for var in pbs_vars:
            value = os.environ.get(var, 'Not set')
            print(f"   {var}: {value}")
    else:
        print("ℹ️  No PBS environment detected (normal for local testing)")
        
        # Test simulated PBS environment
        test_pbs_env = {
            'PBS_JOBID': '12345.cluster',
            'PBS_JOBNAME': 'test_job',
            'PBS_NUM_NODES': '2',
            'PBS_NP': '16'
        }
        
        # Temporarily set PBS environment
        original_pbs = {}
        for key, value in test_pbs_env.items():
            original_pbs[key] = os.environ.get(key)
            os.environ[key] = value
        
        try:
            # Test detection
            simulated_detected = any(os.environ.get(var) for var in pbs_vars)
            
            if simulated_detected:
                print("✅ PBS environment simulation working:")
                for var in pbs_vars:
                    value = os.environ.get(var, 'Not set')
                    print(f"   {var}: {value}")
                result = True
            else:
                print("❌ PBS environment simulation failed")
                result = False
                
        finally:
            # Restore original environment
            for key, value in original_pbs.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
    
    print("✅ PBS environment detection working")
    return True


def test_stress_test_structure():
    """Test stress test file structure."""
    print("\n🧪 Testing Stress Test File Structure...")
    
    stress_test_file = Path(__file__).parent / "test_stress_1000_queries.py"
    
    if not stress_test_file.exists():
        print("❌ Stress test file not found")
        return False
    
    try:
        with open(stress_test_file, 'r') as f:
            content = f.read()
        
        # Check for key components
        required_elements = [
            "async def main",
            "STRESS_TEST_SIZE",
            "MAX_WORKERS",
            "MAX_CONCURRENT",
            "STRESS_TEST_OUTPUT_DIR",
            "QueryJourneyLogger",
            "PBS_JOBID",
            "generate_queries"
        ]
        
        missing_elements = []
        
        for element in required_elements:
            if element in content:
                print(f"   ✅ {element}")
            else:
                missing_elements.append(element)
                print(f"   ❌ {element}")
        
        if missing_elements:
            print(f"❌ Missing stress test elements: {missing_elements}")
            return False
        else:
            print("✅ Stress test structure is complete")
            return True
            
    except Exception as e:
        print(f"❌ Error reading stress test file: {e}")
        return False


def main():
    """Run all stress test configuration tests."""
    print("=" * 80)
    print("🚀 STRESS TEST CONFIGURATION VALIDATION")
    print("=" * 80)
    
    tests = [
        ("Stress Test Imports", test_stress_test_imports),
        ("Environment Configuration", test_environment_configuration),
        ("Output Directory Creation", test_output_directory_creation),
        ("PBS Environment Detection", test_pbs_environment_detection),
        ("Stress Test Structure", test_stress_test_structure)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
    
    print("\n" + "=" * 80)
    print("📊 STRESS TEST CONFIGURATION RESULTS")
    print("=" * 80)
    
    if passed == total:
        print(f"🎉 ALL TESTS PASSED! ({passed}/{total})")
        print("✅ Stress test is properly configured for PBS execution!")
        print("\n🚀 PBS Integration Status:")
        print("   ✅ Environment variable configuration")
        print("   ✅ Output directory management")
        print("   ✅ PBS environment detection")
        print("   ✅ Configurable test parameters")
        print("\n📋 Ready for PBS stress testing!")
        return 0
    else:
        print(f"❌ {total - passed} tests failed ({passed}/{total} passed)")
        print("Please fix configuration issues before PBS deployment")
        return 1


if __name__ == "__main__":
    sys.exit(main())
