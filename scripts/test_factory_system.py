#!/usr/bin/env python3
"""
Factory System Test

Tests the updated factory system to ensure it properly rejects short names
and works correctly with full import paths.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from nanobrain.core.step import create_step, StepConfig
from nanobrain.core.executor import LocalExecutor, ExecutorConfig


def test_full_import_path_acceptance():
    """Test that full import paths are accepted"""
    print("🧪 Testing full import path acceptance...")
    
    config = StepConfig(
        name="Test Step",
        description="Test step for validation"
    )
    
    # Test with full import path and proper executor
    try:
        executor_config = ExecutorConfig()
        executor = LocalExecutor.from_config(executor_config)
        step = create_step("nanobrain.core.step.Step", config, executor=executor)
        print("  ✅ Full import path accepted: nanobrain.core.step.Step")
        print(f"     Created step: {step.name}")
        return True
    except Exception as e:
        print(f"  ❌ Full import path failed: {e}")
        return False


def test_short_name_rejection():
    """Test that short class names are rejected"""
    print("\n🧪 Testing short name rejection...")
    
    config = StepConfig(
        name="Test Step",
        description="Test step for validation"
    )
    
    # Test short names that should be rejected
    short_names = ['Step', 'simple', 'step', 'transform', 'workflow', 'QueryClassificationStep']
    
    all_rejected = True
    for short_name in short_names:
        try:
            step = create_step(short_name, config)
            print(f"  ❌ Short name incorrectly accepted: {short_name}")
            all_rejected = False
        except ValueError as e:
            if "must be a full import path" in str(e):
                print(f"  ✅ Short name correctly rejected: {short_name}")
            else:
                print(f"  ⚠️  Short name rejected but unexpected error: {e}")
        except Exception as e:
            print(f"  ❌ Unexpected error for {short_name}: {e}")
            all_rejected = False
    
    return all_rejected


def test_invalid_import_path_handling():
    """Test handling of invalid import paths"""
    print("\n🧪 Testing invalid import path handling...")
    
    config = StepConfig(
        name="Test Step",
        description="Test step for validation"
    )
    
    # Test invalid import paths
    invalid_paths = [
        "nonexistent.module.Class",
        "nanobrain.core.step.NonExistentClass",
        "nanobrain.invalid.Module"
    ]
    
    all_handled = True
    for invalid_path in invalid_paths:
        try:
            step = create_step(invalid_path, config)
            print(f"  ❌ Invalid path incorrectly accepted: {invalid_path}")
            all_handled = False
        except ImportError:
            print(f"  ✅ Invalid path correctly rejected: {invalid_path}")
        except Exception as e:
            print(f"  ⚠️  Invalid path rejected but unexpected error: {e}")
    
    return all_handled


def test_transform_step():
    """Test with TransformStep full import path"""
    print("\n🧪 Testing TransformStep with full import path...")
    
    config = StepConfig(
        name="Transform Test Step",
        description="Test transform step"
    )
    
    try:
        executor_config = ExecutorConfig()
        executor = LocalExecutor.from_config(executor_config)
        step = create_step("nanobrain.core.step.TransformStep", config, executor=executor)
        print("  ✅ TransformStep created successfully")
        print(f"     Created step: {step.name}")
        return True
    except Exception as e:
        print(f"  ❌ TransformStep creation failed: {e}")
        return False


def main():
    """Run all factory system tests"""
    print("🚀 Factory System Test Suite")
    print("=" * 50)
    
    tests = [
        ("Full Import Path Acceptance", test_full_import_path_acceptance),
        ("Short Name Rejection", test_short_name_rejection),
        ("Invalid Import Path Handling", test_invalid_import_path_handling),
        ("TransformStep Creation", test_transform_step)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
            else:
                print(f"  ⚠️  Test '{test_name}' had some failures")
        except Exception as e:
            print(f"  ❌ Test '{test_name}' crashed: {e}")
    
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✅ ALL TESTS PASSED")
        print("Factory system correctly rejects short names and accepts full import paths!")
        return True
    else:
        print("❌ SOME TESTS FAILED")
        print("Factory system needs additional fixes.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 