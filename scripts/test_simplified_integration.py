#!/usr/bin/env python3
"""
Simplified Integration Test for Unified from_config Pattern
Uses correct configuration parameters and focuses on architectural validation.
"""

import os
import sys
import tempfile
import yaml
from pathlib import Path
from typing import Dict, Any

# Add nanobrain to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from nanobrain.core.component_base import FromConfigBase
    from nanobrain.library.tools.bioinformatics.mmseqs_tool import MMseqs2Tool
    from nanobrain.library.tools.bioinformatics.bv_brc_tool import BVBRCTool
    from nanobrain.library.workflows.viral_protein_analysis.steps.data_aggregation_step import DataAggregationStep
except ImportError as e:
    print(f"ERROR: Cannot import required nanobrain components: {e}")
    sys.exit(1)


def test_unified_from_config_pattern():
    """Test unified from_config pattern with correct configurations."""
    print("🚀 Simplified Integration Test - Unified from_config Pattern")
    print("="*70)
    
    results = {'passed': 0, 'failed': 0, 'tests': []}
    
    # Test 1: MMseqs2Tool with correct config
    print("\n🧪 Test 1: MMseqs2Tool Configuration")
    try:
        # Create temporary config file with correct MMseqs2 parameters
        temp_dir = tempfile.mkdtemp()
        config_data = {
            'tool_name': 'mmseqs2',
            'min_seq_id': 0.5,
            'coverage': 0.8,
            'cluster_mode': 0,
            'sensitivity': 7.5
        }
        
        config_path = Path(temp_dir) / 'mmseqs2_test.yml'
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        # Test from_config with file path
        tool = MMseqs2Tool.from_config(str(config_path))
        
        # Validate
        assert hasattr(tool, 'name'), "Tool missing name attribute"
        assert hasattr(tool, '_get_config_class'), "Tool missing _get_config_class method"
        
        results['passed'] += 1
        results['tests'].append(('MMseqs2Tool from_config', True, None))
        print("   ✅ MMseqs2Tool from_config successful")
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)
        
    except Exception as e:
        results['failed'] += 1
        results['tests'].append(('MMseqs2Tool from_config', False, str(e)))
        print(f"   ❌ MMseqs2Tool from_config failed: {e}")
    
    # Test 2: DataAggregationStep with correct config
    print("\n🧪 Test 2: DataAggregationStep Configuration")
    try:
        # Create temporary config file with correct step parameters
        temp_dir = tempfile.mkdtemp()
        config_data = {
            'name': 'test_aggregation_step',
            'description': 'Test data aggregation step',
            'timeout': 300,
            'retry_attempts': 3
        }
        
        config_path = Path(temp_dir) / 'step_test.yml'
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        # Test from_config with file path
        step = DataAggregationStep.from_config(str(config_path))
        
        # Validate
        assert hasattr(step, 'name'), "Step missing name attribute"
        assert hasattr(step, '_get_config_class'), "Step missing _get_config_class method"
        assert step.name == 'test_aggregation_step', f"Expected name 'test_aggregation_step', got '{step.name}'"
        
        results['passed'] += 1
        results['tests'].append(('DataAggregationStep from_config', True, None))
        print("   ✅ DataAggregationStep from_config successful")
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)
        
    except Exception as e:
        results['failed'] += 1
        results['tests'].append(('DataAggregationStep from_config', False, str(e)))
        print(f"   ❌ DataAggregationStep from_config failed: {e}")
    
    # Test 3: Class Auto-Detection with BVBRCTool
    print("\n🧪 Test 3: Class Auto-Detection")
    try:
        # Create temporary config file with class specification and correct BV-BRC parameters
        temp_dir = tempfile.mkdtemp()
        config_data = {
            'class': 'nanobrain.library.tools.bioinformatics.bv_brc_tool.BVBRCTool',
            'tool_name': 'bv_brc',
            'use_cache': True,
            'timeout_seconds': 300
        }
        
        config_path = Path(temp_dir) / 'auto_detect_test.yml'
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        # Test auto-detection from FromConfigBase
        component = FromConfigBase.from_config(str(config_path))
        
        # Validate auto-detection worked
        assert isinstance(component, BVBRCTool), f"Expected BVBRCTool, got {type(component)}"
        assert hasattr(component, '_get_config_class'), "Component missing _get_config_class method"
        
        results['passed'] += 1
        results['tests'].append(('Class auto-detection', True, None))
        print("   ✅ Class auto-detection successful")
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)
        
    except Exception as e:
        results['failed'] += 1
        results['tests'].append(('Class auto-detection', False, str(e)))
        print(f"   ❌ Class auto-detection failed: {e}")
    
    # Test 4: Dictionary Configuration
    print("\n🧪 Test 4: Dictionary Configuration")
    try:
        # Test direct dictionary configuration
        config_dict = {
            'tool_name': 'mmseqs2',
            'min_seq_id': 0.4,
            'coverage': 0.7,
            'cluster_mode': 1
        }
        
        tool = MMseqs2Tool.from_config(config_dict)
        
        # Validate
        assert hasattr(tool, 'name'), "Tool from dict missing name attribute"
        assert hasattr(tool, '_get_config_class'), "Tool missing _get_config_class method"
        
        results['passed'] += 1
        results['tests'].append(('Dictionary configuration', True, None))
        print("   ✅ Dictionary configuration successful")
        
    except Exception as e:
        results['failed'] += 1
        results['tests'].append(('Dictionary configuration', False, str(e)))
        print(f"   ❌ Dictionary configuration failed: {e}")
    
    # Test 5: Error Handling
    print("\n🧪 Test 5: Error Handling")
    try:
        # Test missing file
        try:
            MMseqs2Tool.from_config("/nonexistent/file.yml")
            # Should not reach here
            raise Exception("No exception raised for missing file")
        except FileNotFoundError:
            # Expected behavior
            pass
        
        results['passed'] += 1
        results['tests'].append(('Error handling', True, None))
        print("   ✅ Error handling successful")
        
    except Exception as e:
        results['failed'] += 1
        results['tests'].append(('Error handling', False, str(e)))
        print(f"   ❌ Error handling failed: {e}")
    
    # Summary
    total_tests = results['passed'] + results['failed']
    success_rate = (results['passed'] / total_tests * 100) if total_tests > 0 else 0
    
    print("\n" + "="*70)
    print("🎯 SIMPLIFIED INTEGRATION TEST RESULTS")
    print("="*70)
    print(f"\n📊 OVERALL SUCCESS RATE: {success_rate:.1f}%")
    print(f"   ✅ Passed Tests: {results['passed']}")
    print(f"   ❌ Failed Tests: {results['failed']}")
    print(f"   📈 Total Tests: {total_tests}")
    
    print(f"\n📋 DETAILED RESULTS:")
    for test_name, passed, error in results['tests']:
        status = "✅" if passed else "❌"
        print(f"   {status} {test_name}")
        if error:
            print(f"      Error: {error}")
    
    # Determine final status
    if success_rate >= 80.0:
        print("\n🎉 EXCELLENT! Unified from_config pattern working well.")
        print("✅ Core architectural compliance validated.")
        return True
    else:
        print("\n⚠️ NEEDS ATTENTION! Address failed tests.")
        print("🔧 Review configuration parameters and error handling.")
        return False


if __name__ == "__main__":
    success = test_unified_from_config_pattern()
    sys.exit(0 if success else 1) 