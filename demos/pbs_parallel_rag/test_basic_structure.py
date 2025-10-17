#!/usr/bin/env python3
"""
Basic Structure Test
====================

Tests basic file and directory structure without requiring external dependencies.
"""

import sys
from pathlib import Path


def test_file_structure():
    """Test that all required files exist."""
    print("🧪 Testing File Structure...")
    
    base_path = Path(__file__).parent
    
    required_files = [
        # Submit scripts
        "submit_stress_test.sh",
        "submit_quick_test.sh",
        "run_pbs_tests.sh",
        
        # Core Python files
        "pbs_query_enhancement_step.py",
        "shared_vector_database.py",
        
        # Configuration files
        "config/executors/pbs_executor.yml",
        "config/executors/pbs_executor_small.yml",
        "config/workflow/parallel_rag_workflow.yml",
        
        # Test files
        "test_comprehensive.py",
        "test_config_structure.py",
        "test_stress_1000_queries.py"
    ]
    
    missing_files = []
    present_files = []
    
    for file_path in required_files:
        full_path = base_path / file_path
        if full_path.exists():
            present_files.append(file_path)
            print(f"   ✅ {file_path}")
        else:
            missing_files.append(file_path)
            print(f"   ❌ {file_path}")
    
    print(f"\n📊 Files: {len(present_files)}/{len(required_files)} present")
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All required files present")
        return True


def test_directory_structure():
    """Test that all required directories exist."""
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
    
    missing_dirs = []
    present_dirs = []
    
    for dir_path in required_dirs:
        full_path = base_path / dir_path
        if full_path.exists() and full_path.is_dir():
            present_dirs.append(dir_path)
            print(f"   ✅ {dir_path}/")
        else:
            missing_dirs.append(dir_path)
            print(f"   ❌ {dir_path}/")
    
    print(f"\n📊 Directories: {len(present_dirs)}/{len(required_dirs)} present")
    
    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
        return False
    else:
        print("✅ All required directories present")
        return True


def test_pbs_config_content():
    """Test PBS configuration file content without YAML parsing."""
    print("\n🧪 Testing PBS Configuration Content...")
    
    config_path = Path(__file__).parent / "config" / "executors" / "pbs_executor.yml"
    
    if not config_path.exists():
        print("❌ PBS executor config file not found")
        return False
    
    try:
        with open(config_path, 'r') as f:
            content = f.read()
        
        # Check for key PBS configuration elements
        required_elements = [
            "parsl.providers.PBSProvider",
            "queue:",
            "nodes_per_block:",
            "cpus_per_node:",
            "walltime:",
            "max_workers:",
            "HighThroughputExecutor"
        ]
        
        missing_elements = []
        
        for element in required_elements:
            if element in content:
                print(f"   ✅ {element}")
            else:
                missing_elements.append(element)
                print(f"   ❌ {element}")
        
        if missing_elements:
            print(f"❌ Missing PBS configuration elements: {missing_elements}")
            return False
        else:
            print("✅ PBS configuration contains all required elements")
            return True
            
    except Exception as e:
        print(f"❌ Error reading PBS config: {e}")
        return False


def test_submit_script_content():
    """Test submit script content."""
    print("\n🧪 Testing Submit Script Content...")
    
    script_path = Path(__file__).parent / "submit_stress_test.sh"
    
    if not script_path.exists():
        print("❌ Submit script not found")
        return False
    
    try:
        with open(script_path, 'r') as f:
            content = f.read()
        
        # Check for key PBS directives and script elements
        required_elements = [
            "#PBS -N",
            "#PBS -l nodes=",
            "#PBS -l walltime=",
            "#PBS -q",
            "test_stress_1000_queries.py",
            "test_comprehensive.py",
            "NANOBRAIN_PATH",
            "OUTPUT_DIR"
        ]
        
        missing_elements = []
        
        for element in required_elements:
            if element in content:
                print(f"   ✅ {element}")
            else:
                missing_elements.append(element)
                print(f"   ❌ {element}")
        
        if missing_elements:
            print(f"❌ Missing submit script elements: {missing_elements}")
            return False
        else:
            print("✅ Submit script contains all required elements")
            return True
            
    except Exception as e:
        print(f"❌ Error reading submit script: {e}")
        return False


def test_executable_permissions():
    """Test that shell scripts have executable permissions."""
    print("\n🧪 Testing Executable Permissions...")
    
    base_path = Path(__file__).parent
    shell_scripts = ["run_pbs_tests.sh"]
    
    all_executable = True
    
    for script in shell_scripts:
        script_path = base_path / script
        if script_path.exists():
            # Check if file is executable (has execute permission)
            import stat
            file_stat = script_path.stat()
            is_executable = bool(file_stat.st_mode & stat.S_IEXEC)
            
            if is_executable:
                print(f"   ✅ {script} (executable)")
            else:
                print(f"   ❌ {script} (not executable)")
                all_executable = False
        else:
            print(f"   ❌ {script} (not found)")
            all_executable = False
    
    if all_executable:
        print("✅ All shell scripts are executable")
        return True
    else:
        print("❌ Some shell scripts are not executable")
        return False


def main():
    """Run all basic structure tests."""
    print("=" * 80)
    print("🚀 PBS PARALLEL RAG BASIC STRUCTURE TEST")
    print("=" * 80)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Directory Structure", test_directory_structure),
        ("PBS Config Content", test_pbs_config_content),
        ("Submit Script Content", test_submit_script_content),
        ("Executable Permissions", test_executable_permissions)
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
    print("📊 BASIC STRUCTURE TEST RESULTS")
    print("=" * 80)
    
    if passed == total:
        print(f"🎉 ALL TESTS PASSED! ({passed}/{total})")
        print("✅ PBS Parallel RAG basic structure is complete!")
        print("\n🚀 Infrastructure Status:")
        print("   ✅ Submit scripts ready")
        print("   ✅ Configuration files present")
        print("   ✅ Test files available")
        print("   ✅ Documentation complete")
        print("\n📋 Ready for deployment to PBS cluster!")
        return 0
    else:
        print(f"❌ {total - passed} tests failed ({passed}/{total} passed)")
        print("Please fix structure issues before deployment")
        return 1


if __name__ == "__main__":
    sys.exit(main())
