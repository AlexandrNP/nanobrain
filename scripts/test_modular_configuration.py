#!/usr/bin/env python3
"""
Comprehensive Modular Configuration Test

Tests the complete modular configuration implementation including:
1. Full import path enforcement
2. External configuration loading
3. Workflow creation with modular configs
"""

import sys
import os
import yaml
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from nanobrain.core.step import create_step, StepConfig
from nanobrain.core.executor import LocalExecutor, ExecutorConfig
from nanobrain.core.workflow import ConfigLoader


def test_factory_rejects_short_names():
    """Test that factory properly rejects short names"""
    print("🧪 Testing factory rejects short names...")
    
    config = StepConfig(name="Test", description="Test")
    short_names = ['Step', 'simple', 'transform', 'workflow']
    
    all_rejected = True
    for short_name in short_names:
        try:
            create_step(short_name, config)
            print(f"  ❌ Short name incorrectly accepted: {short_name}")
            all_rejected = False
        except ValueError as e:
            if "must be a full import path" in str(e):
                print(f"  ✅ Correctly rejected: {short_name}")
            else:
                print(f"  ⚠️  Unexpected error: {e}")
                all_rejected = False
    
    return all_rejected


def test_factory_accepts_full_paths():
    """Test that factory accepts full import paths"""
    print("\n🧪 Testing factory accepts full import paths...")
    
    config = StepConfig(name="Test", description="Test")
    executor_config = ExecutorConfig()
    executor = LocalExecutor.from_config(executor_config)
    
    full_paths = [
        'nanobrain.core.step.Step',
        'nanobrain.core.step.TransformStep'
    ]
    
    all_accepted = True
    for full_path in full_paths:
        try:
            step = create_step(full_path, config, executor=executor)
            print(f"  ✅ Correctly accepted: {full_path}")
        except Exception as e:
            print(f"  ❌ Full path incorrectly rejected: {full_path} - {e}")
            all_accepted = False
    
    return all_accepted


def test_external_config_loading():
    """Test external configuration file loading"""
    print("\n🧪 Testing external configuration loading...")
    
    # Create temporary config file
    temp_dir = Path(tempfile.mkdtemp())
    config_dir = temp_dir / "config" / "steps"
    config_dir.mkdir(parents=True)
    
    # Create test configuration
    test_config = {
        'name': 'External Test Step',
        'description': 'Step configured via external file',
        'debug_mode': True,
        'custom_param': 'external_value'
    }
    
    config_file = config_dir / "ExternalStep.yml"
    with open(config_file, 'w') as f:
        yaml.dump(test_config, f)
    
    try:
        # Test loading external config
        config_loader = ConfigLoader(str(temp_dir))
        loaded_config = config_loader.load_step_config(
            'config/steps/ExternalStep.yml',
            str(temp_dir)
        )
        
        # Verify config was loaded correctly
        if (loaded_config.name == 'External Test Step' and 
            loaded_config.description == 'Step configured via external file' and
            hasattr(loaded_config, 'custom_param') and
            loaded_config.custom_param == 'external_value'):
            print("  ✅ External configuration loaded correctly")
            return True
        else:
            print("  ❌ External configuration not loaded correctly")
            return False
            
    except Exception as e:
        print(f"  ❌ External config loading failed: {e}")
        return False
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)


def test_modular_workflow_structure():
    """Test the modular workflow structure with external configs"""
    print("\n🧪 Testing modular workflow structure...")
    
    # Check the example workflow structure
    workflow_file = Path("config/example_workflow.yaml")
    if not workflow_file.exists():
        print("  ❌ Example workflow file not found")
        return False
    
    try:
        with open(workflow_file, 'r') as f:
            workflow_data = yaml.safe_load(f)
        
        # Verify modular structure
        if 'steps' not in workflow_data:
            print("  ❌ No steps found in workflow")
            return False
        
        has_external_configs = False
        has_full_paths = False
        
        for step in workflow_data['steps']:
            # Check for external config references
            if 'config_file' in step:
                has_external_configs = True
                print(f"  ✅ Found external config: {step['config_file']}")
            
            # Check for full import paths
            if 'class' in step and '.' in step['class']:
                has_full_paths = True
                print(f"  ✅ Found full import path: {step['class']}")
        
        if has_external_configs and has_full_paths:
            print("  ✅ Workflow has proper modular structure")
            return True
        else:
            print("  ❌ Workflow missing modular structure elements")
            return False
            
    except Exception as e:
        print(f"  ❌ Error validating workflow structure: {e}")
        return False


def test_extracted_config_files():
    """Test that extracted configuration files exist and are valid"""
    print("\n🧪 Testing extracted configuration files...")
    
    config_dir = Path("config/config/steps")
    if not config_dir.exists():
        print("  ❌ Config directory not found")
        return False
    
    config_files = list(config_dir.glob("*.yml"))
    if not config_files:
        print("  ❌ No extracted config files found")
        return False
    
    valid_configs = 0
    for config_file in config_files:
        try:
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Verify config structure
            if 'name' in config_data and '_metadata' in config_data:
                print(f"  ✅ Valid config file: {config_file.name}")
                valid_configs += 1
            else:
                print(f"  ⚠️  Config file missing required fields: {config_file.name}")
                
        except Exception as e:
            print(f"  ❌ Error reading config file {config_file.name}: {e}")
    
    if valid_configs > 0:
        print(f"  ✅ Found {valid_configs} valid extracted config files")
        return True
    else:
        print("  ❌ No valid config files found")
        return False


def test_import_path_validation():
    """Test import path validation"""
    print("\n🧪 Testing import path validation...")
    
    from scripts.validate_import_paths import ImportPathValidator
    validator = ImportPathValidator()
    
    # Test valid full path
    issues = validator.validate_class_import_path("nanobrain.core.step.Step")
    if len(issues) == 0:
        print("  ✅ Valid full path accepted")
        valid_path_test = True
    else:
        print(f"  ❌ Valid full path rejected: {issues}")
        valid_path_test = False
    
    # Test short name rejection
    issues = validator.validate_class_import_path("Step")
    if len(issues) > 0 and "must be full import path" in issues[0]:
        print("  ✅ Short name correctly rejected")
        short_name_test = True
    else:
        print("  ❌ Short name not properly rejected")
        short_name_test = False
    
    return valid_path_test and short_name_test


def test_complete_workflow():
    """Test complete workflow functionality"""
    print("\n🧪 Testing complete workflow functionality...")
    
    try:
        # Create a simple step using the new system
        config = StepConfig(
            name="Complete Test Step",
            description="Testing complete workflow"
        )
        
        executor_config = ExecutorConfig()
        executor = LocalExecutor.from_config(executor_config)
        
        step = create_step("nanobrain.core.step.Step", config, executor=executor)
        
        # Verify step properties
        if (step.name == "Complete Test Step" and 
            step.description == "Testing complete workflow"):
            print("  ✅ Complete workflow test passed")
            return True
        else:
            print("  ❌ Step properties not set correctly")
            return False
            
    except Exception as e:
        print(f"  ❌ Complete workflow test failed: {e}")
        return False


def main():
    """Run comprehensive modular configuration tests"""
    print("🚀 Comprehensive Modular Configuration Test Suite")
    print("=" * 60)
    
    tests = [
        ("Factory Rejects Short Names", test_factory_rejects_short_names),
        ("Factory Accepts Full Paths", test_factory_accepts_full_paths),
        ("External Config Loading", test_external_config_loading),
        ("Modular Workflow Structure", test_modular_workflow_structure),
        ("Extracted Config Files", test_extracted_config_files),
        ("Import Path Validation", test_import_path_validation),
        ("Complete Workflow", test_complete_workflow)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
                print(f"\n✅ {test_name}: PASSED")
            else:
                print(f"\n❌ {test_name}: FAILED")
        except Exception as e:
            print(f"\n💥 {test_name}: CRASHED - {e}")
    
    print("\n" + "=" * 60)
    print("📊 COMPREHENSIVE TEST SUMMARY")
    print("=" * 60)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Modular configuration implementation is working correctly!")
        print("\n🎯 ACHIEVEMENTS:")
        print("  ✅ Factory system enforces full import paths")
        print("  ✅ Short class names are properly rejected")
        print("  ✅ External configuration loading works")
        print("  ✅ Modular workflow structure is functional")
        print("  ✅ Configuration extraction is complete")
        print("  ✅ Import path validation is operational")
        print("  ✅ Complete workflows can be created")
        return True
    else:
        print("\n❌ SOME TESTS FAILED")
        print("Modular configuration implementation needs attention.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 