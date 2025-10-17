#!/usr/bin/env python3
"""
Simple PBS Configuration Test
=============================

Tests PBS configuration files without requiring full nanobrain dependencies.
This test validates the YAML structure and PBS provider settings.
"""

import yaml
import sys
from pathlib import Path


def test_pbs_executor_config():
    """Test PBS executor configuration file."""
    print("🧪 Testing PBS Executor Configuration...")
    
    config_path = Path(__file__).parent / "config" / "executors" / "pbs_executor.yml"
    
    if not config_path.exists():
        print("❌ PBS executor config file not found")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check basic structure
        required_fields = ['name', 'executor_type', 'max_workers', 'parsl_config']
        for field in required_fields:
            if field not in config:
                print(f"❌ Missing required field: {field}")
                return False
        
        # Check PARSL configuration
        parsl_config = config['parsl_config']
        if 'executors' not in parsl_config:
            print("❌ Missing executors in parsl_config")
            return False
        
        # Check PBS provider configuration
        executor = parsl_config['executors'][0]
        provider_config = executor.get('provider_config', {})
        
        if provider_config.get('class') != 'parsl.providers.PBSProvider':
            print("❌ PBS provider not configured correctly")
            return False
        
        # Check PBS-specific settings
        pbs_settings = ['queue', 'nodes_per_block', 'cpus_per_node', 'walltime']
        for setting in pbs_settings:
            if setting not in provider_config:
                print(f"❌ Missing PBS setting: {setting}")
                return False
        
        print("✅ PBS executor configuration is valid")
        print(f"   Queue: {provider_config['queue']}")
        print(f"   Nodes: {provider_config['nodes_per_block']}")
        print(f"   CPUs per node: {provider_config['cpus_per_node']}")
        print(f"   Walltime: {provider_config['walltime']}")
        print(f"   Max workers: {config['max_workers']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading PBS executor config: {e}")
        return False


def test_workflow_config():
    """Test workflow configuration file."""
    print("\n🧪 Testing Workflow Configuration...")
    
    config_path = Path(__file__).parent / "config" / "workflow" / "parallel_rag_workflow.yml"
    
    if not config_path.exists():
        print("❌ Workflow config file not found")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check basic structure
        required_fields = ['name', 'steps', 'links']
        for field in required_fields:
            if field not in config:
                print(f"❌ Missing required field: {field}")
                return False
        
        # Check PBS query enhancement step
        steps = config['steps']
        if 'prompt_enhancement_step' not in steps:
            print("❌ Missing prompt_enhancement_step")
            return False
        
        enhancement_step = steps['prompt_enhancement_step']
        expected_class = "demos.pbs_parallel_rag.pbs_query_enhancement_step.PBSQueryEnhancementStep"
        
        if enhancement_step.get('class') != expected_class:
            print(f"❌ Incorrect PBS step class: {enhancement_step.get('class')}")
            return False
        
        # Check parallel features
        if 'parallel_features' in config:
            parallel_config = config['parallel_features']
            if 'parsl_executor' in parallel_config:
                executor_path = parallel_config['parsl_executor']
                if 'pbs_executor.yml' not in executor_path:
                    print(f"❌ Incorrect executor path: {executor_path}")
                    return False
        
        print("✅ Workflow configuration is valid")
        print(f"   PBS step class: {enhancement_step['class']}")
        print(f"   Total steps: {len(steps)}")
        print(f"   Total links: {len(config['links'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading workflow config: {e}")
        return False


def test_step_configs():
    """Test step configuration files."""
    print("\n🧪 Testing Step Configurations...")
    
    config_dir = Path(__file__).parent / "config" / "steps"
    
    if not config_dir.exists():
        print("❌ Steps config directory not found")
        return False
    
    step_files = [
        "prompt_enhancement_step.yml",
        "retrieval_specialist_step.yml",
        "analysis_specialist_step.yml",
        "synthesis_specialist_step.yml",
        "quality_assurance_step.yml"
    ]
    
    valid_count = 0
    
    for step_file in step_files:
        step_path = config_dir / step_file
        if step_path.exists():
            try:
                with open(step_path, 'r') as f:
                    config = yaml.safe_load(f)
                print(f"   ✅ {step_file}")
                valid_count += 1
            except Exception as e:
                print(f"   ❌ {step_file}: {e}")
        else:
            print(f"   ⚠️  {step_file}: Not found")
    
    print(f"✅ Step configurations: {valid_count}/{len(step_files)} valid")
    return valid_count > 0


def test_directory_structure():
    """Test required directory structure."""
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
    
    for dir_name in required_dirs:
        dir_path = base_path / dir_name
        if not dir_path.exists():
            missing_dirs.append(dir_name)
            print(f"   ❌ Missing directory: {dir_name}")
        else:
            print(f"   ✅ {dir_name}")
    
    if missing_dirs:
        print(f"❌ Missing {len(missing_dirs)} required directories")
        return False
    else:
        print("✅ All required directories present")
        return True


def main():
    """Run all configuration tests."""
    print("=" * 80)
    print("🚀 PBS PARALLEL RAG CONFIGURATION VALIDATION")
    print("=" * 80)
    
    tests = [
        ("PBS Executor Config", test_pbs_executor_config),
        ("Workflow Config", test_workflow_config),
        ("Step Configs", test_step_configs),
        ("Directory Structure", test_directory_structure)
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
    print("📊 TEST RESULTS")
    print("=" * 80)
    
    if passed == total:
        print(f"🎉 ALL TESTS PASSED! ({passed}/{total})")
        print("✅ PBS Parallel RAG configuration is ready for deployment!")
        print("\n🚀 Next steps:")
        print("   1. Copy to PBS cluster")
        print("   2. Configure PBS queue settings for your cluster")
        print("   3. Install required Python dependencies")
        print("   4. Submit jobs using submit_stress_test.pbs")
        return 0
    else:
        print(f"❌ {total - passed} tests failed ({passed}/{total} passed)")
        print("Please fix configuration issues before deployment")
        return 1


if __name__ == "__main__":
    sys.exit(main())
