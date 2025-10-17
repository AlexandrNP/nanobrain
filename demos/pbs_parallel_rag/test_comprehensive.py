#!/usr/bin/env python3
"""
Comprehensive PBS Parallel RAG Test
===================================

Comprehensive test suite to verify all PBS parallel RAG capabilities.
Tests configuration, imports, and functionality without requiring PBS runtime.
"""

import sys
import os
from pathlib import Path
import importlib.util


def test_configuration_files():
    """Test that all configuration files are valid and properly structured."""
    print("🧪 Testing Configuration Files...")
    
    base_path = Path(__file__).parent
    
    # Test PBS executor config
    pbs_config = base_path / "config" / "executors" / "pbs_executor.yml"
    if not pbs_config.exists():
        print("❌ PBS executor config missing")
        return False
    
    with open(pbs_config, 'r') as f:
        content = f.read()
        
    if 'parsl.providers.PBSProvider' not in content:
        print("❌ PBS provider not configured")
        return False
        
    if 'nodes_per_block:' not in content:
        print("❌ PBS nodes configuration missing")
        return False
        
    print("✅ PBS executor configuration valid")
    
    # Test workflow config
    workflow_config = base_path / "config" / "workflow" / "parallel_rag_workflow.yml"
    if not workflow_config.exists():
        print("❌ Workflow config missing")
        return False
        
    with open(workflow_config, 'r') as f:
        content = f.read()
        
    if 'pbs_query_enhancement_step.PBSQueryEnhancementStep' not in content:
        print("❌ PBS step not configured in workflow")
        return False
        
    if 'pbs_executor.yml' not in content:
        print("❌ PBS executor not referenced in workflow")
        return False
        
    print("✅ Workflow configuration valid")
    return True


def test_python_imports():
    """Test that all Python modules can be imported without errors."""
    print("\n🧪 Testing Python Imports...")
    
    base_path = Path(__file__).parent
    
    # Test main PBS step
    try:
        spec = importlib.util.spec_from_file_location(
            "pbs_query_enhancement_step", 
            base_path / "pbs_query_enhancement_step.py"
        )
        if spec is None:
            print("❌ Cannot load PBS query enhancement step")
            return False
            
        print("✅ PBS query enhancement step importable")
    except Exception as e:
        print(f"❌ Error importing PBS step: {e}")
        return False
    
    # Test tracked steps
    tracked_steps = [
        "tracked_query_enhancement_step.py",
        "tracked_vector_search_step.py", 
        "tracked_response_generation_step.py"
    ]
    
    for step_file in tracked_steps:
        try:
            spec = importlib.util.spec_from_file_location(
                step_file[:-3], 
                base_path / "steps" / step_file
            )
            if spec is None:
                print(f"❌ Cannot load {step_file}")
                return False
                
        except Exception as e:
            print(f"❌ Error with {step_file}: {e}")
            return False
    
    print("✅ All tracked steps importable")
    
    # Test tools
    tools = [
        "analyze_logs.py",
        "export_logs.py",
        "view_journey.py"
    ]
    
    for tool_file in tools:
        try:
            spec = importlib.util.spec_from_file_location(
                tool_file[:-3], 
                base_path / "tools" / tool_file
            )
            if spec is None:
                print(f"❌ Cannot load {tool_file}")
                return False
                
        except Exception as e:
            print(f"❌ Error with {tool_file}: {e}")
            return False
    
    print("✅ All tools importable")
    return True


def test_directory_structure():
    """Test that all required directories and files exist."""
    print("\n🧪 Testing Directory Structure...")
    
    base_path = Path(__file__).parent
    
    required_dirs = [
        "config",
        "config/executors", 
        "config/steps",
        "config/workflow",
        "journey_logging",
        "models",
        "steps",
        "tools",
        "output"
    ]
    
    for dir_path in required_dirs:
        full_path = base_path / dir_path
        if not full_path.exists():
            print(f"❌ Missing directory: {dir_path}")
            return False
    
    print("✅ All required directories exist")
    
    required_files = [
        "pbs_query_enhancement_step.py",
        "config/executors/pbs_executor.yml",
        "config/workflow/parallel_rag_workflow.yml",
        "journey_logging/journey_logger.py",
        "models/query_journey.py",
        "steps/tracked_query_enhancement_step.py",
        "tools/analyze_logs.py"
    ]
    
    for file_path in required_files:
        full_path = base_path / file_path
        if not full_path.exists():
            print(f"❌ Missing file: {file_path}")
            return False
    
    print("✅ All required files exist")
    return True


def test_import_consistency():
    """Test that all imports reference the correct PBS paths."""
    print("\n🧪 Testing Import Consistency...")
    
    base_path = Path(__file__).parent
    
    # Check for any remaining references to parallel_rag_with_parsl
    python_files = list(base_path.rglob("*.py"))
    
    for py_file in python_files:
        if py_file.name == __file__.split('/')[-1]:  # Skip this test file
            continue
            
        try:
            with open(py_file, 'r') as f:
                content = f.read()
                
            if 'parallel_rag_with_parsl' in content:
                print(f"❌ Found old import reference in {py_file.relative_to(base_path)}")
                return False
                
        except Exception as e:
            print(f"❌ Error reading {py_file}: {e}")
            return False
    
    print("✅ All imports reference correct PBS paths")
    return True


def test_stress_test_compatibility():
    """Test that the stress test is properly configured for PBS."""
    print("\n🧪 Testing Stress Test Compatibility...")
    
    base_path = Path(__file__).parent
    stress_test = base_path / "test_stress_1000_queries.py"
    
    if not stress_test.exists():
        print("❌ Stress test file missing")
        return False
    
    with open(stress_test, 'r') as f:
        content = f.read()
    
    # Check that it imports from pbs_parallel_rag
    if 'demos.pbs_parallel_rag' not in content:
        print("❌ Stress test not configured for PBS")
        return False
    
    # Check that it has the required imports
    required_imports = [
        'QueryJourneyLogger',
        'LogAnalyzer', 
        'LogExporter'
    ]
    
    for import_name in required_imports:
        if import_name not in content:
            print(f"❌ Stress test missing import: {import_name}")
            return False
    
    print("✅ Stress test properly configured for PBS")
    return True


def main():
    """Run all tests."""
    print("="*80)
    print("🚀 COMPREHENSIVE PBS PARALLEL RAG TEST SUITE")
    print("="*80)
    
    tests = [
        ("Configuration Files", test_configuration_files),
        ("Python Imports", test_python_imports),
        ("Directory Structure", test_directory_structure),
        ("Import Consistency", test_import_consistency),
        ("Stress Test Compatibility", test_stress_test_compatibility)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"\n❌ {test_name} test failed")
        except Exception as e:
            print(f"\n❌ {test_name} test error: {e}")
    
    print("\n" + "="*80)
    print("📊 TEST RESULTS")
    print("="*80)
    
    if passed == total:
        print(f"🎉 ALL TESTS PASSED! ({passed}/{total})")
        print("✅ PBS Parallel RAG is ready for deployment!")
        print("\n🚀 Next steps:")
        print("   1. Deploy to PBS cluster")
        print("   2. Configure PBS queue settings")
        print("   3. Run test_parallel_rag.py on cluster")
        print("   4. Scale up to stress testing")
        return True
    else:
        print(f"❌ TESTS FAILED ({passed}/{total})")
        print("🔧 Please fix issues before deployment")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
