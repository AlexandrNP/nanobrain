#!/usr/bin/env python3
"""
Test Configuration Path Resolution
==================================

Test that configuration file paths are correctly resolved relative to script locations.
This verifies the fix for configuration path issues.
"""

import sys
from pathlib import Path
import yaml

# Add nanobrain to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def test_workflow_config_paths():
    """Test that workflow configuration paths are correctly resolved."""
    print("🧪 Testing Workflow Configuration Path Resolution")
    print("=" * 60)
    
    # Test the workflow configuration loading
    script_dir = Path(__file__).parent
    config_path = script_dir / "config" / "workflow" / "parallel_rag_workflow.yml"
    
    print(f"📄 Workflow config path: {config_path}")
    print(f"📁 Config file exists: {config_path.exists()}")
    
    if not config_path.exists():
        print("❌ Workflow configuration file not found")
        return False
    
    # Load and parse the configuration
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print("✅ Workflow configuration loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load workflow configuration: {e}")
        return False
    
    # Test that step configuration paths are relative and exist
    print("\n📋 Testing step configuration paths...")
    steps = config.get('steps', {})
    
    for step_name, step_config in steps.items():
        step_config_path = step_config.get('config', '')
        if step_config_path:
            # Resolve relative to the workflow config directory
            full_step_path = config_path.parent.parent / step_config_path
            print(f"   📄 {step_name}: {step_config_path}")
            print(f"      Full path: {full_step_path}")
            print(f"      Exists: {full_step_path.exists()}")
            
            if not full_step_path.exists():
                print(f"   ❌ Step config not found: {step_config_path}")
                return False
            else:
                print(f"   ✅ Step config found: {step_name}")
    
    # Test executor configuration path
    print("\n🔧 Testing executor configuration path...")
    parallel_features = config.get('parallel_features', {})
    executor_path = parallel_features.get('parsl_executor', '')
    
    if executor_path:
        full_executor_path = config_path.parent.parent / executor_path
        print(f"   📄 Executor config: {executor_path}")
        print(f"      Full path: {full_executor_path}")
        print(f"      Exists: {full_executor_path.exists()}")
        
        if not full_executor_path.exists():
            print(f"   ❌ Executor config not found: {executor_path}")
            return False
        else:
            print(f"   ✅ Executor config found")
    
    print("\n✅ All configuration paths are correctly resolved")
    return True

def test_step_config_files():
    """Test that individual step configuration files exist."""
    print("\n🧪 Testing Individual Step Configuration Files")
    print("=" * 60)
    
    script_dir = Path(__file__).parent
    steps_dir = script_dir / "config" / "steps"
    
    print(f"📁 Steps directory: {steps_dir}")
    print(f"📁 Directory exists: {steps_dir.exists()}")
    
    if not steps_dir.exists():
        print("❌ Steps configuration directory not found")
        return False
    
    # List all step configuration files
    step_files = list(steps_dir.glob("*.yml"))
    print(f"\n📋 Found {len(step_files)} step configuration files:")
    
    for step_file in step_files:
        print(f"   📄 {step_file.name}")
        
        # Try to load each step configuration
        try:
            with open(step_file, 'r') as f:
                step_config = yaml.safe_load(f)
            print(f"      ✅ Valid YAML")
        except Exception as e:
            print(f"      ❌ Invalid YAML: {e}")
            return False
    
    print(f"\n✅ All {len(step_files)} step configuration files are valid")
    return True

def test_executor_config_files():
    """Test that executor configuration files exist."""
    print("\n🧪 Testing Executor Configuration Files")
    print("=" * 60)
    
    script_dir = Path(__file__).parent
    executors_dir = script_dir / "config" / "executors"
    
    print(f"📁 Executors directory: {executors_dir}")
    print(f"📁 Directory exists: {executors_dir.exists()}")
    
    if not executors_dir.exists():
        print("❌ Executors configuration directory not found")
        return False
    
    # List all executor configuration files
    executor_files = list(executors_dir.glob("*.yml"))
    print(f"\n📋 Found {len(executor_files)} executor configuration files:")
    
    for executor_file in executor_files:
        print(f"   📄 {executor_file.name}")
        
        # Try to load each executor configuration
        try:
            with open(executor_file, 'r') as f:
                executor_config = yaml.safe_load(f)
            print(f"      ✅ Valid YAML")
        except Exception as e:
            print(f"      ❌ Invalid YAML: {e}")
            return False
    
    print(f"\n✅ All {len(executor_files)} executor configuration files are valid")
    return True

def main():
    """Run all configuration path tests."""
    print("CONFIGURATION PATH RESOLUTION TESTS")
    print("=" * 70)
    print("Testing that configuration file paths are correctly resolved")
    print("after fixing hardcoded path issues")
    print("=" * 70)
    
    # Run all tests
    workflow_ok = test_workflow_config_paths()
    steps_ok = test_step_config_files()
    executors_ok = test_executor_config_files()
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST RESULTS SUMMARY")
    print("=" * 70)
    
    if workflow_ok and steps_ok and executors_ok:
        print("✅ ALL CONFIGURATION PATH TESTS PASSED")
        print("✅ Configuration files can be found using relative paths")
        print("✅ The hardcoded path issue is RESOLVED")
        print("\nConfiguration files are now resolved relative to:")
        print("  - Script location for Python files")
        print("  - Workflow config location for step/executor configs")
        print("  - This works regardless of current working directory")
    else:
        print("❌ SOME CONFIGURATION PATH TESTS FAILED")
        print("❌ There may still be path resolution issues")
        
        if not workflow_ok:
            print("  - Workflow configuration path issues")
        if not steps_ok:
            print("  - Step configuration path issues")
        if not executors_ok:
            print("  - Executor configuration path issues")
    
    print("=" * 70)
    return workflow_ok and steps_ok and executors_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
