#!/usr/bin/env python3
"""
Unified from_config Pattern Integration Tests (Fixed Version)
Comprehensive validation of architectural compliance and functionality.

Tests the complete component loading chains using concrete implementations,
class auto-detection, and error handling following Nanobrain's data-driven approach.
"""

import os
import sys
import tempfile
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
import asyncio

# Add nanobrain to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from nanobrain.core.component_base import FromConfigBase
    from nanobrain.core.workflow import Workflow, WorkflowConfig
    from nanobrain.core.step import BaseStep, StepConfig
    from nanobrain.core.agent import Agent, AgentConfig
    from nanobrain.core.tool import ToolBase, ToolConfig
    from nanobrain.library.tools.bioinformatics.bv_brc_tool import BVBRCTool
    from nanobrain.library.tools.bioinformatics.mmseqs_tool import MMseqs2Tool
    from nanobrain.library.workflows.viral_protein_analysis.alphavirus_workflow import AlphavirusWorkflow
    from nanobrain.library.workflows.viral_protein_analysis.steps.data_aggregation_step import DataAggregationStep
except ImportError as e:
    print(f"ERROR: Cannot import required nanobrain components: {e}")
    sys.exit(1)


class UnifiedFromConfigIntegrationTests:
    """
    Comprehensive integration tests for unified from_config pattern.
    
    Tests actual component creation, nested loading, and error scenarios
    without mocks - pure end-to-end validation using concrete implementations.
    """
    
    def __init__(self):
        self.test_results: Dict[str, Any] = {}
        self.temp_dir = None
        self.test_config_files: List[Path] = []
        
    def setup_test_environment(self):
        """Set up temporary test environment with config files."""
        self.temp_dir = tempfile.mkdtemp(prefix="nanobrain_test_")
        print(f"📁 Created test environment: {self.temp_dir}")
        
    def cleanup_test_environment(self):
        """Clean up temporary test files."""
        import shutil
        if self.temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
            print(f"🧹 Cleaned up test environment: {self.temp_dir}")
    
    def create_test_config_file(self, filename: str, config_data: Dict[str, Any]) -> Path:
        """Create a temporary test configuration file."""
        config_path = Path(self.temp_dir) / filename
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False)
        self.test_config_files.append(config_path)
        return config_path
    
    async def test_simple_component_loading(self) -> Dict[str, Any]:
        """
        Test 1: Simple Component Loading
        Validate that concrete components can be loaded from file paths.
        """
        print("🧪 Test 1: Simple Component Loading (Concrete Implementations)")
        
        results = {}
        
        # Test concrete tool loading
        try:
            tool_config = {
                'name': 'test_mmseqs_tool',
                'description': 'Test MMseqs2 tool for integration testing',
                'timeout': 300,
                'database_path': '/tmp/test_db',
                'output_format': 'tsv'
            }
            tool_config_path = self.create_test_config_file('test_tool.yml', tool_config)
            
            # Test file path loading with concrete tool implementation
            tool = MMseqs2Tool.from_config(str(tool_config_path))
            
            # Validate instance
            assert hasattr(tool, 'name'), "Tool missing name attribute"
            assert tool.name == 'test_mmseqs_tool', f"Expected name 'test_mmseqs_tool', got '{tool.name}'"
            assert hasattr(tool, '_get_config_class'), "Tool missing _get_config_class method"
            
            results['tool_loading'] = {'success': True, 'component': tool.__class__.__name__}
            print("   ✅ Tool loading successful")
            
        except Exception as e:
            results['tool_loading'] = {'success': False, 'error': str(e)}
            print(f"   ❌ Tool loading failed: {e}")
        
        # Test concrete step loading
        try:
            step_config = {
                'name': 'test_data_aggregation_step',
                'description': 'Test data aggregation step for integration testing',
                'timeout': 300,
                'retry_attempts': 3
            }
            step_config_path = self.create_test_config_file('test_step.yml', step_config)
            
            # Test file path loading with concrete step implementation
            step = DataAggregationStep.from_config(str(step_config_path))
            
            # Validate instance
            assert hasattr(step, 'name'), "Step missing name attribute"
            assert step.name == 'test_data_aggregation_step', f"Expected name 'test_data_aggregation_step', got '{step.name}'"
            assert hasattr(step, '_get_config_class'), "Step missing _get_config_class method"
            
            results['step_loading'] = {'success': True, 'component': step.__class__.__name__}
            print("   ✅ Step loading successful")
            
        except Exception as e:
            results['step_loading'] = {'success': False, 'error': str(e)}
            print(f"   ❌ Step loading failed: {e}")
        
        # Test workflow loading with existing configuration
        try:
            # Check if the file exists from current working directory
            full_path = Path("nanobrain/library/workflows/viral_protein_analysis/config/AlphavirusWorkflow.yml")
            if full_path.exists():
                workflow = AlphavirusWorkflow.from_config(str(full_path))
                
                # Validate instance
                assert hasattr(workflow, 'name'), "Workflow missing name attribute"
                assert hasattr(workflow, '_get_config_class'), "Workflow missing _get_config_class method"
                
                results['workflow_loading'] = {'success': True, 'component': workflow.__class__.__name__}
                print("   ✅ Workflow loading successful")
            else:
                results['workflow_loading'] = {'success': False, 'error': 'Config file not found'}
                print("   ⚠️ Workflow config file not found, skipping test")
                
        except Exception as e:
            results['workflow_loading'] = {'success': False, 'error': str(e)}
            print(f"   ❌ Workflow loading failed: {e}")
        
        return results
    
    async def test_class_auto_detection(self) -> Dict[str, Any]:
        """
        Test 2: Class Auto-Detection
        Validate that config files can specify different classes and framework loads them correctly.
        """
        print("🧪 Test 2: Class Auto-Detection")
        
        results = {}
        
        try:
            # Create config that specifies a specific tool class
            specific_tool_config = {
                'name': 'bv_brc_test_tool',
                'class': 'nanobrain.library.tools.bioinformatics.bv_brc_tool.BVBRCTool',
                'tool_name': 'bv_brc',
                'timeout_seconds': 300,
                'use_cache': True
            }
            config_path = self.create_test_config_file('auto_detect_tool.yml', specific_tool_config)
            
            # Call from generic base class - should auto-detect and create BVBRCTool
            component = FromConfigBase.from_config(str(config_path))
            
            # Should be BVBRCTool instance, not generic ToolBase
            assert isinstance(component, BVBRCTool), f"Expected BVBRCTool, got {type(component)}"
            assert component.name == 'bv_brc_test_tool', f"Expected name 'bv_brc_test_tool', got '{component.name}'"
            
            results['auto_detection'] = {'success': True, 'detected_class': component.__class__.__name__}
            print("   ✅ Class auto-detection successful")
            
        except Exception as e:
            results['auto_detection'] = {'success': False, 'error': str(e)}
            print(f"   ❌ Class auto-detection failed: {e}")
        
        return results
    
    async def test_error_handling(self) -> Dict[str, Any]:
        """
        Test 3: Error Handling
        Validate proper error handling for missing files, invalid configs, etc.
        """
        print("🧪 Test 3: Error Handling")
        
        results = {}
        
        # Test missing file handling
        try:
            missing_file_path = "/nonexistent/path/missing_config.yml"
            component = MMseqs2Tool.from_config(missing_file_path)
            
            # Should not reach here
            results['missing_file'] = {'success': False, 'error': 'No exception raised for missing file'}
            print("   ❌ Missing file test failed - no exception raised")
            
        except FileNotFoundError as e:
            results['missing_file'] = {'success': True, 'error_type': 'FileNotFoundError'}
            print("   ✅ Missing file properly handled with FileNotFoundError")
            
        except Exception as e:
            results['missing_file'] = {'success': False, 'error': f'Unexpected exception: {str(e)}'}
            print(f"   ⚠️ Missing file test - unexpected exception: {e}")
        
        # Test invalid config handling
        try:
            invalid_config = {
                'invalid_field': 'invalid_value',
                # Missing required 'name' field
            }
            config_path = self.create_test_config_file('invalid_config.yml', invalid_config)
            
            component = MMseqs2Tool.from_config(str(config_path))
            
            # Should not reach here with invalid config
            results['invalid_config'] = {'success': False, 'error': 'No exception raised for invalid config'}
            print("   ❌ Invalid config test failed - no exception raised")
            
        except Exception as e:
            results['invalid_config'] = {'success': True, 'error_type': type(e).__name__}
            print(f"   ✅ Invalid config properly handled with {type(e).__name__}")
        
        # Test invalid class specification
        try:
            invalid_class_config = {
                'name': 'test_invalid_class',
                'class': 'nonexistent.module.NonExistentClass',
                'tool_name': 'invalid'
            }
            config_path = self.create_test_config_file('invalid_class.yml', invalid_class_config)
            
            component = FromConfigBase.from_config(str(config_path))
            
            # Should not reach here with invalid class
            results['invalid_class'] = {'success': False, 'error': 'No exception raised for invalid class'}
            print("   ❌ Invalid class test failed - no exception raised")
            
        except Exception as e:
            results['invalid_class'] = {'success': True, 'error_type': type(e).__name__}
            print(f"   ✅ Invalid class properly handled with {type(e).__name__}")
        
        return results
    
    async def test_configuration_formats(self) -> Dict[str, Any]:
        """
        Test 4: Configuration Format Support
        Validate support for different config input types (file path, dict, config object).
        """
        print("🧪 Test 4: Configuration Format Support")
        
        results = {}
        
        # Test dictionary input with concrete implementation
        try:
            config_dict = {
                'name': 'dict_test_mmseqs_tool',
                'description': 'MMseqs2 tool created from dictionary config',
                'timeout': 300,
                'database_path': '/tmp/dict_test_db',
                'output_format': 'tsv'
            }
            
            tool = MMseqs2Tool.from_config(config_dict)
            
            assert hasattr(tool, 'name'), "Tool from dict missing name attribute"
            assert tool.name == 'dict_test_mmseqs_tool', f"Expected name 'dict_test_mmseqs_tool', got '{tool.name}'"
            
            results['dict_input'] = {'success': True, 'component': tool.__class__.__name__}
            print("   ✅ Dictionary input successful")
            
        except Exception as e:
            results['dict_input'] = {'success': False, 'error': str(e)}
            print(f"   ❌ Dictionary input failed: {e}")
        
        # Test config object input with concrete implementation
        try:
            # Use MMseqs2Config for config object test
            from nanobrain.library.tools.bioinformatics.mmseqs_tool import MMseqs2Config
            
            config_obj = MMseqs2Config(
                name='object_test_mmseqs_tool',
                description='MMseqs2 tool created from config object',
                timeout=300,
                database_path='/tmp/object_test_db',
                output_format='tsv'
            )
            
            tool = MMseqs2Tool.from_config(config_obj)
            
            assert hasattr(tool, 'name'), "Tool from object missing name attribute"
            assert tool.name == 'object_test_mmseqs_tool', f"Expected name 'object_test_mmseqs_tool', got '{tool.name}'"
            
            results['object_input'] = {'success': True, 'component': tool.__class__.__name__}
            print("   ✅ Config object input successful")
            
        except Exception as e:
            results['object_input'] = {'success': False, 'error': str(e)}
            print(f"   ❌ Config object input failed: {e}")
        
        return results
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all integration tests and compile results."""
        print("🚀 Starting Unified from_config Pattern Integration Tests (Fixed)")
        print("="*80)
        
        self.setup_test_environment()
        
        try:
            # Run all test scenarios
            test_1_results = await self.test_simple_component_loading()
            test_2_results = await self.test_class_auto_detection()
            test_3_results = await self.test_error_handling()
            test_4_results = await self.test_configuration_formats()
            
            # Compile overall results
            all_results = {
                'test_1_simple_loading': test_1_results,
                'test_2_auto_detection': test_2_results,
                'test_3_error_handling': test_3_results,
                'test_4_config_formats': test_4_results,
            }
            
            # Calculate success rates
            success_counts = {}
            total_counts = {}
            
            for test_name, test_results in all_results.items():
                successful = sum(1 for result in test_results.values() 
                               if isinstance(result, dict) and result.get('success', False))
                total = len([result for result in test_results.values() 
                           if isinstance(result, dict) and 'success' in result])
                
                success_counts[test_name] = successful
                total_counts[test_name] = total
            
            overall_successful = sum(success_counts.values())
            overall_total = sum(total_counts.values())
            overall_success_rate = (overall_successful / overall_total * 100) if overall_total > 0 else 0
            
            # Summary
            print("\n" + "="*80)
            print("🎯 INTEGRATION TEST RESULTS SUMMARY")
            print("="*80)
            
            print(f"\n📊 OVERALL SUCCESS RATE: {overall_success_rate:.1f}%")
            print(f"   ✅ Successful Tests: {overall_successful}")
            print(f"   ❌ Failed Tests: {overall_total - overall_successful}")
            print(f"   📈 Total Tests: {overall_total}")
            
            for test_name, test_results in all_results.items():
                successful = success_counts[test_name]
                total = total_counts[test_name]
                rate = (successful / total * 100) if total > 0 else 0
                
                print(f"\n🧪 {test_name.replace('_', ' ').title()}: {rate:.1f}% ({successful}/{total})")
                
                for subtest_name, result in test_results.items():
                    if isinstance(result, dict) and 'success' in result:
                        status = "✅" if result['success'] else "❌"
                        print(f"   {status} {subtest_name}")
            
            # Determine readiness for next phase
            if overall_success_rate >= 95.0:
                print("\n🎉 EXCELLENT! Unified from_config pattern is working perfectly.")
                print("✅ Framework is ready for production use.")
            elif overall_success_rate >= 85.0:
                print("\n👍 GOOD! Most tests passing, minor issues to address.")
                print("⚠️ Review failed tests before production deployment.")
            else:
                print("\n⚠️ NEEDS ATTENTION! Significant issues found.")
                print("🔧 Address failed tests before proceeding.")
            
            self.test_results = all_results
            return all_results
            
        finally:
            self.cleanup_test_environment()


async def main():
    """Main execution function."""
    tester = UnifiedFromConfigIntegrationTests()
    results = await tester.run_all_tests()
    return results


if __name__ == "__main__":
    asyncio.run(main()) 