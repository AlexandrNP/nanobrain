#!/usr/bin/env python3
"""
Test PBS Configuration Structure
================================

Simple test to verify the PBS configuration structure without importing nanobrain.
"""

import sys
import os
from pathlib import Path


def test_pbs_executor_config():
    """Test PBS executor configuration structure."""
    print("Testing PBS Executor Configuration Structure...")

    # Try multiple possible paths for the config file
    possible_paths = [
        Path(__file__).parent / "config" / "executors" / "pbs_executor.yml",
        Path("demos/pbs_parallel_rag/config/executors/pbs_executor.yml"),
        Path("config/executors/pbs_executor.yml")
    ]

    config_path = None
    for path in possible_paths:
        if path.exists():
            config_path = path
            break

    if config_path is None:
        print("❌ PBS executor config file not found in any expected location")
        print(f"   Tried: {[str(p) for p in possible_paths]}")
        return False

    try:
        with open(config_path, 'r') as f:
            content = f.read()

        print(f"✅ PBS executor config loaded successfully")

        # Check for PBS-specific content
        if 'parsl.providers.PBSProProvider' in content:
            print("✅ PBS provider correctly configured")
        else:
            print("❌ PBS provider not configured correctly")
            return False

        if 'queue: debug' in content:
            print("✅ PBS queue configured")
        else:
            print("❌ PBS queue not configured")
            return False

        if 'nodes_per_block:' in content:
            print("✅ PBS nodes configuration found")
        else:
            print("❌ PBS nodes configuration missing")
            return False

        if 'workers_per_node:' in content:
            print("✅ PBS workers configuration found")
        else:
            print("❌ PBS workers configuration missing")
            return False

        if 'parsl.launchers.MpiExecLauncher' in content:
            print("✅ MPI launcher correctly configured")
        else:
            print("❌ MPI launcher not configured")
            return False

        if 'walltime:' in content:
            print("✅ PBS walltime configured")
        else:
            print("❌ PBS walltime not configured")
            return False

        print("🎯 PBS configuration structure is valid!")
        return True

    except Exception as e:
        print(f"❌ Error loading PBS config: {e}")
        return False


def test_workflow_config():
    """Test workflow configuration structure."""
    print("\nTesting Workflow Configuration Structure...")

    # Try multiple possible paths for the config file
    possible_paths = [
        Path(__file__).parent / "config" / "workflow" / "parallel_rag_workflow.yml",
        Path("demos/pbs_parallel_rag/config/workflow/parallel_rag_workflow.yml"),
        Path("config/workflow/parallel_rag_workflow.yml")
    ]

    config_path = None
    for path in possible_paths:
        if path.exists():
            config_path = path
            break

    if config_path is None:
        print("❌ Workflow config file not found in any expected location")
        print(f"   Tried: {[str(p) for p in possible_paths]}")
        return False

    try:
        with open(config_path, 'r') as f:
            content = f.read()

        print(f"✅ Workflow config loaded successfully")

        # Check for PBS-specific content
        if 'demos.pbs_parallel_rag.pbs_query_enhancement_step.PBSQueryEnhancementStep' in content:
            print("✅ PBS query enhancement step correctly configured")
        else:
            print("❌ PBS query enhancement step not configured correctly")
            return False

        if 'pbs_executor.yml' in content:
            print("✅ PBS executor correctly referenced in workflow")
        else:
            print("❌ PBS executor not correctly referenced in workflow")
            return False

        if 'config/steps/' in content and 'config/executors/' in content:
            print("✅ PBS config paths correctly set (using relative paths)")
        else:
            print("❌ PBS config paths not correctly set")
            return False

        print("✅ Workflow configuration structure is valid!")
        return True

    except Exception as e:
        print(f"❌ Error loading workflow config: {e}")
        return False


def main():
    """Run all configuration tests."""
    print("="*80)
    print("🧪 PBS PARALLEL RAG CONFIGURATION TESTS")
    print("="*80)
    
    tests_passed = 0
    total_tests = 2
    
    # Test PBS executor config
    if test_pbs_executor_config():
        tests_passed += 1
    
    # Test workflow config
    if test_workflow_config():
        tests_passed += 1
    
    print("\n" + "="*80)
    print("📊 TEST SUMMARY")
    print("="*80)
    
    if tests_passed == total_tests:
        print(f"🎉 ALL TESTS PASSED! ({tests_passed}/{total_tests})")
        print("✅ PBS parallel RAG configuration is ready for deployment!")
        return True
    else:
        print(f"❌ SOME TESTS FAILED ({tests_passed}/{total_tests})")
        print("🔧 Please fix configuration issues before deployment")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
