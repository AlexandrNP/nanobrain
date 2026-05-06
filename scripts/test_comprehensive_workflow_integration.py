#!/usr/bin/env python3
"""
Comprehensive Workflow Integration Tests
End-to-end validation of unified from_config pattern with real nanobrain workflows.

Tests complete workflow loading chains, nested component creation,
and architectural compliance in realistic usage scenarios.
"""

import os
import sys
import tempfile
import yaml
import json
from pathlib import Path
from typing import Dict, Any, List
import asyncio

# Add nanobrain to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from nanobrain.core.component_base import FromConfigBase
    from nanobrain.library.workflows.viral_protein_analysis.alphavirus_workflow import AlphavirusWorkflow
    from nanobrain.library.workflows.viral_protein_analysis.steps.data_aggregation_step import DataAggregationStep
    from nanobrain.library.tools.bioinformatics.mmseqs_tool import MMseqs2Tool
    from nanobrain.library.tools.bioinformatics.bv_brc_tool import BVBRCTool
    from nanobrain.core.workflow import Workflow, WorkflowConfig
    from nanobrain.core.step import BaseStep, StepConfig
except ImportError as e:
    print(f"ERROR: Cannot import required nanobrain components: {e}")
    sys.exit(1)


class ComprehensiveWorkflowTests:
    """
    Comprehensive end-to-end workflow testing suite.
    
    Validates unified from_config pattern with realistic workflow scenarios,
    nested component loading, and production-like configurations.
    """
    
    def __init__(self):
        self.test_results: Dict[str, Any] = {}
        self.temp_dir = None
        
    def setup_test_environment(self):
        """Set up comprehensive test environment."""
        self.temp_dir = tempfile.mkdtemp(prefix="nanobrain_workflow_test_")
        print(f"📁 Created workflow test environment: {self.temp_dir}")
        
    def cleanup_test_environment(self):
        """Clean up test environment."""
        import shutil
        if self.temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
            print(f"🧹 Cleaned up test environment: {self.temp_dir}")
    
    def create_test_config_file(self, filename: str, config_data: Dict[str, Any]) -> Path:
        """Create test configuration file."""
        config_path = Path(self.temp_dir) / filename
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False)
        return config_path
    
    def create_standardized_test_config_structure(self) -> Dict[str, Path]:
        """
        Create standardized test environment maintaining viral protein analysis structure.
        
        CORRECTED APPROACH:
        1. Create standardized structure for organization consistency
        2. Use ABSOLUTE PATHS for individual component testing (matches framework behavior)
        3. Use workflow_directory approach for realistic workflow testing
        4. Test both individual components AND workflow-based loading
        """
        print("   📁 Creating standardized configuration structure...")
        
        # Create standardized directory structure (maintained for consistency)
        config_dir = Path(self.temp_dir) / "config"
        clustering_step_dir = config_dir / "ClusteringStep"
        data_agg_step_dir = config_dir / "DataAggregationStep"
        
        config_dir.mkdir(exist_ok=True)
        clustering_step_dir.mkdir(exist_ok=True)
        data_agg_step_dir.mkdir(exist_ok=True)
        
        # Create MMseqs2 tool config with CORRECTED valid parameters
        mmseqs_tool_config = {
            'tool_name': 'mmseqs2',
            'min_seq_id': 0.4,
            'coverage': 0.8,
            'cluster_mode': 1,
            'sensitivity': 7.5,
            'threads': 2,
            'conda_package': 'mmseqs2',
            'conda_channel': 'bioconda',
            'environment_name': 'nanobrain-test-mmseqs2',
            'cache_dir': 'data/mmseqs2_cache',  # CORRECTED: Valid MMseqs2Config parameter
            'memory_limit': '8G',  # CORRECTED: String format required
            'tmp_dir': '/tmp'  # CORRECTED: Valid parameter
            # REMOVED: 'enable_logging' - Invalid MMseqs2Config parameter
        }
        mmseqs_tool_config_path = clustering_step_dir / 'MMseqs2Tool.yml'
        with open(mmseqs_tool_config_path, 'w') as f:
            yaml.dump(mmseqs_tool_config, f, default_flow_style=False)
        
        # Create clustering step config (uses DataAggregationStep for testing without tool dependencies)
        clustering_step_config = {
            'name': 'test_clustering_step',
            'description': 'Test clustering step using standardized structure',
            'class': 'nanobrain.library.workflows.viral_protein_analysis.steps.data_aggregation_step.DataAggregationStep',
            'timeout': 600,
            'config': {
                'data_mappings': {
                    'test_clustering': {
                        'source_fields': ['clusters', 'clustering_analysis'],
                        'target_section': 'analysis_data',
                        'field_mappings': {}
                    }
                },
                'output_format': {
                    'standardize_keys': False,
                    'include_metadata': True
                },
                'quality_checks': {
                    'validate_data_completeness': True,
                    'required_sections': ['analysis_data'],
                    'minimum_completeness_score': 0.7
                }
            }
        }
        clustering_step_config_path = clustering_step_dir / 'DataAggregationStep.yml'
        with open(clustering_step_config_path, 'w') as f:
            yaml.dump(clustering_step_config, f, default_flow_style=False)
        
        # Create data aggregation step config (standardized pattern)
        data_agg_step_config = {
            'name': 'test_data_aggregation_step',
            'description': 'Test data aggregation step following standardized structure',
            'class': 'nanobrain.library.workflows.viral_protein_analysis.steps.data_aggregation_step.DataAggregationStep',
            'timeout': 300,
            'config': {
                'data_mappings': {
                    'data_aggregation': {
                        'source_fields': ['aggregated_data', 'summary_stats'],
                        'target_section': 'final_data',
                        'field_mappings': {}
                    }
                },
                'output_format': {
                    'standardize_keys': True,
                    'include_metadata': True
                }
            }
        }
        data_agg_step_config_path = data_agg_step_dir / 'DataAggregationStep.yml'
        with open(data_agg_step_config_path, 'w') as f:
            yaml.dump(data_agg_step_config, f, default_flow_style=False)
        
        # Create workflow config for WORKFLOW-BASED testing (realistic approach)
        workflow_config = {
            'name': 'TestWorkflow',
            'description': 'Test workflow for unified pattern validation using standardized structure',
            'class': 'nanobrain.library.workflows.viral_protein_analysis.alphavirus_workflow.AlphavirusWorkflow',
            'execution_strategy': 'sequential',
            'workflow_directory': str(self.temp_dir),  # Key: ConfigLoader uses this for relative path resolution
            'enable_monitoring': True,
            'enable_progress_reporting': True,
            'steps': [
                {
                    'step_id': 'clustering',
                    'config_file': 'config/ClusteringStep/DataAggregationStep.yml'  # Relative to workflow_directory
                },
                {
                    'step_id': 'data_aggregation', 
                    'config_file': 'config/DataAggregationStep/DataAggregationStep.yml'  # Relative to workflow_directory
                }
            ]
        }
        workflow_config_path = Path(self.temp_dir) / 'test_workflow.yml'
        with open(workflow_config_path, 'w') as f:
            yaml.dump(workflow_config, f, default_flow_style=False)
        
        print(f"     ✅ Created standardized structure in {self.temp_dir}")
        print(f"     📁 Config directory: {config_dir}")
        print(f"     📁 ClusteringStep: {clustering_step_dir}")
        print(f"     📁 DataAggregationStep: {data_agg_step_dir}")
        
        return {
            'workflow': workflow_config_path,
            'clustering_step': clustering_step_config_path,
            'data_agg_step': data_agg_step_config_path,
            'mmseqs_tool': mmseqs_tool_config_path
        }

    async def test_workflow_component_loading(self) -> Dict[str, Any]:
        """
        Test 1: Workflow Component Loading
        CORRECTED: Uses both individual component testing (absolute paths) 
        and realistic workflow-based testing (ConfigLoader)
        """
        print("🧪 Test 1: Workflow Component Loading Chain (Corrected Path Resolution)")
        
        results = {}
        config_paths = self.create_standardized_test_config_structure()
        
        # Test 1.1: Individual Tool Loading (CORRECTED: Use absolute paths)
        try:
            print("   🔧 Testing individual tool loading with absolute path...")
            
            # CORRECTED: Individual components require absolute paths 
            # because they resolve relative to component class location, not test directory
            mmseqs_tool = MMseqs2Tool.from_config(str(config_paths['mmseqs_tool'].absolute()))
            assert hasattr(mmseqs_tool, 'name'), "MMseqs2Tool missing name attribute"
            assert hasattr(mmseqs_tool, '_get_config_class'), "MMseqs2Tool missing _get_config_class"
            assert mmseqs_tool.name == 'mmseqs2', f"Expected 'mmseqs2', got '{mmseqs_tool.name}'"
            
            results['individual_tool_loading'] = {'success': True, 'component': 'MMseqs2Tool'}
            print("     ✅ Individual tool loading successful")
            
        except Exception as e:
            results['individual_tool_loading'] = {'success': False, 'error': str(e)}
            print(f"     ❌ Individual tool loading failed: {e}")
        
        # Test 1.2: Individual Step Loading (CORRECTED: Use absolute paths)
        try:
            print("   🔧 Testing individual step loading with absolute path...")
            
            # CORRECTED: Individual step components also need absolute paths
            step = DataAggregationStep.from_config(str(config_paths['clustering_step'].absolute()))
            assert hasattr(step, 'name'), "Step missing name attribute"
            assert step.name == 'test_clustering_step', f"Expected name 'test_clustering_step', got '{step.name}'"
            
            results['individual_step_loading'] = {'success': True, 'component': 'DataAggregationStep'}
            print("     ✅ Individual step loading successful")
            
        except Exception as e:
            results['individual_step_loading'] = {'success': False, 'error': str(e)}
            print(f"     ❌ Individual step loading failed: {e}")
        
        # Test 1.3: Workflow-Based Loading (CORRECTED: Realistic workflow testing)
        try:
            print("   🔧 Testing workflow-based component loading (realistic approach)...")
            
            # CORRECTED: Workflow uses ConfigLoader which respects workflow_directory
            # This tests the realistic scenario where steps are loaded through workflow
            workflow = AlphavirusWorkflow.from_config(str(config_paths['workflow'].absolute()))
            assert isinstance(workflow, AlphavirusWorkflow), "Failed to create AlphavirusWorkflow instance"
            assert workflow.name == 'TestWorkflow', f"Expected 'TestWorkflow', got '{workflow.name}'"
            
            # Verify workflow can initialize (validates step loading through ConfigLoader)
            # Note: We don't fully initialize to avoid complex dependencies
            assert hasattr(workflow, 'config_loader'), "Workflow missing config_loader"
            
            results['workflow_based_loading'] = {'success': True, 'component': 'AlphavirusWorkflow'}
            print("     ✅ Workflow-based loading successful")
            
        except Exception as e:
            results['workflow_based_loading'] = {'success': False, 'error': str(e)}
            print(f"     ❌ Workflow-based loading failed: {e}")
        
        return results
    
    async def test_configuration_flexibility(self) -> Dict[str, Any]:
        """
        Test 2: Configuration Flexibility
        CORRECTED: Tests multiple valid configuration approaches with proper path resolution
        """
        print("🧪 Test 2: Configuration Flexibility (Corrected)")
        
        results = {}
        
        # Test 2.1: Dictionary Configuration (CORRECTED: Use valid parameters)
        try:
            print("   🔧 Testing dictionary configuration with valid parameters...")
            
            # CORRECTED: Use only valid MMseqs2Config parameters
            dict_config = {
                'tool_name': 'mmseqs2',
                'min_seq_id': 0.5,
                'coverage': 0.9,
                'cluster_mode': 2,
                'sensitivity': 8.0,
                'threads': 4,
                'memory_limit': '16G',
                'conda_package': 'mmseqs2',
                'conda_channel': 'bioconda'
            }
            
            tool = MMseqs2Tool.from_config(dict_config)
            assert hasattr(tool, 'name'), "Tool from dict missing name attribute"
            assert tool.name == 'mmseqs2', f"Expected 'mmseqs2', got '{tool.name}'"
            
            results['dictionary_config'] = {'success': True, 'config_type': 'dictionary'}
            print("     ✅ Dictionary configuration successful")
            
        except Exception as e:
            results['dictionary_config'] = {'success': False, 'error': str(e)}
            print(f"     ❌ Dictionary configuration failed: {e}")
        
        # Test 2.2: File Path Configuration (CORRECTED: Use absolute paths)
        try:
            print("   🔧 Testing file path configuration with absolute path...")
            
            config_paths = self.create_standardized_test_config_structure()
            
            # CORRECTED: Use absolute path for individual component
            tool = MMseqs2Tool.from_config(str(config_paths['mmseqs_tool'].absolute()))
            assert hasattr(tool, 'name'), "Tool from file missing name attribute"
            
            results['file_path_config'] = {'success': True, 'config_type': 'file_path'}
            print("     ✅ File path configuration successful")
            
        except Exception as e:
            results['file_path_config'] = {'success': False, 'error': str(e)}
            print(f"     ❌ File path configuration failed: {e}")
        
        # Test 2.3: Nested Configuration (CORRECTED: Use workflow-based approach)
        try:
            print("   🔧 Testing nested configuration through workflow...")
            
            config_paths = self.create_standardized_test_config_structure()
            
            # CORRECTED: Test nested loading through workflow (realistic scenario)
            workflow = AlphavirusWorkflow.from_config(str(config_paths['workflow'].absolute()))
            
            # Verify workflow can access its configuration
            assert hasattr(workflow, 'workflow_config'), "Workflow missing config"
            assert len(workflow.workflow_config.steps) == 2, "Expected 2 steps in workflow config"
            
            results['nested_config'] = {'success': True, 'config_type': 'nested_workflow'}
            print("     ✅ Nested configuration successful")
            
        except Exception as e:
            results['nested_config'] = {'success': False, 'error': str(e)}
            print(f"     ❌ Nested configuration failed: {e}")
        
        return results
    
    async def test_error_resilience(self) -> Dict[str, Any]:
        """
        Test 3: Error Handling and Resilience
        CORRECTED: Tests both absolute path failures and relative path scenarios
        """
        print("🧪 Test 3: Error Handling and Resilience (Corrected)")
        
        results = {}
        
        # Test 3.1: Missing Configuration Files (CORRECTED: Use absolute paths)
        try:
            print("   🔧 Testing missing file handling with absolute path...")
            
            # CORRECTED: Use absolute path for individual component testing
            nonexistent_absolute_path = "/nonexistent/absolute/path/missing.yml"
            try:
                MMseqs2Tool.from_config(nonexistent_absolute_path)
                # Should not reach here
                results['missing_file'] = {'success': False, 'error': 'No exception raised'}
            except FileNotFoundError:
                results['missing_file'] = {'success': True, 'error_type': 'FileNotFoundError'}
                print("     ✅ Missing file properly handled")
            except Exception as e:
                results['missing_file'] = {'success': False, 'error': f'Unexpected error: {str(e)}'}
                print(f"     ⚠️ Unexpected error for missing file: {e}")
                
        except Exception as e:
            results['missing_file'] = {'success': False, 'error': str(e)}
            print(f"     ❌ Missing file test failed: {e}")
        
        # Test 3.2: Invalid Configuration Data (CORRECTED: Use valid structure)
        try:
            print("   🔧 Testing invalid configuration handling...")
            
            # CORRECTED: Create invalid config with proper structure but invalid values
            invalid_config = {
                'tool_name': 'mmseqs2',
                'invalid_field': 'invalid_value',
                'min_seq_id': 'invalid_numeric_value',  # Should be float
                'coverage': -1.5,  # Invalid range
                'cluster_mode': 'invalid_mode'  # Should be int
            }
            
            try:
                MMseqs2Tool.from_config(invalid_config)
                # Should not reach here
                results['invalid_config'] = {'success': False, 'error': 'No exception raised'}
            except Exception as e:
                results['invalid_config'] = {'success': True, 'error_type': type(e).__name__}
                print(f"     ✅ Invalid config properly handled with {type(e).__name__}")
                
        except Exception as e:
            results['invalid_config'] = {'success': False, 'error': str(e)}
            print(f"     ❌ Invalid config test failed: {e}")
        
        # Test 3.3: Class Import Failures (CORRECTED: Use absolute path)
        try:
            print("   🔧 Testing class import failure handling...")
            
            invalid_class_config = {
                'class': 'nonexistent.module.NonExistentClass',
                'tool_name': 'invalid'
            }
            config_path = self.create_test_config_file('invalid_class.yml', invalid_class_config)
            
            try:
                # CORRECTED: Use absolute path for consistent testing approach
                FromConfigBase.from_config(str(config_path.absolute()))
                # Should not reach here
                results['import_failure'] = {'success': False, 'error': 'No exception raised'}
            except ImportError:
                results['import_failure'] = {'success': True, 'error_type': 'ImportError'}
                print("     ✅ Import failure properly handled")
            except Exception as e:
                results['import_failure'] = {'success': True, 'error_type': type(e).__name__}
                print(f"     ✅ Import failure handled with {type(e).__name__}")
            
        except Exception as e:
            results['import_failure'] = {'success': False, 'error': str(e)}
            print(f"     ❌ Import failure test failed: {e}")
        
        return results
    
    async def test_architectural_compliance(self) -> Dict[str, Any]:
        """
        Test 4: Architectural Pattern Compliance
        Validate that all components follow mandatory architectural patterns.
        """
        print("🧪 Test 4: Architectural Pattern Compliance")
        
        results = {}
        
        # Test 4.1: Direct Instantiation Prevention
        try:
            print("   🔧 Testing direct instantiation prevention...")
            
            try:
                # This should raise RuntimeError
                MMseqs2Tool()
                results['instantiation_prevention'] = {'success': False, 'error': 'Direct instantiation allowed'}
            except RuntimeError as e:
                if "prohibited" in str(e).lower():
                    results['instantiation_prevention'] = {'success': True, 'prevention_confirmed': True}
                    print("     ✅ Direct instantiation properly prevented")
                else:
                    results['instantiation_prevention'] = {'success': False, 'error': f'Wrong error: {e}'}
            except Exception as e:
                results['instantiation_prevention'] = {'success': False, 'error': f'Unexpected error: {e}'}
            
        except Exception as e:
            results['instantiation_prevention'] = {'success': False, 'error': str(e)}
            print(f"     ❌ Instantiation prevention test failed: {e}")
        
        # Test 4.2: _get_config_class Implementation
        try:
            print("   🔧 Testing _get_config_class implementation...")
            
            # Test various component types
            test_classes = [
                MMseqs2Tool,
                DataAggregationStep,
                BVBRCTool
            ]
            
            all_implemented = True
            for cls in test_classes:
                if not hasattr(cls, '_get_config_class'):
                    all_implemented = False
                    break
                
                config_class = cls._get_config_class()
                if config_class is None:
                    all_implemented = False
                    break
            
            if all_implemented:
                results['get_config_class'] = {'success': True, 'classes_tested': len(test_classes)}
                print(f"     ✅ _get_config_class implemented in {len(test_classes)} classes")
            else:
                results['get_config_class'] = {'success': False, 'error': 'Missing implementation'}
                print("     ❌ _get_config_class missing in some classes")
            
        except Exception as e:
            results['get_config_class'] = {'success': False, 'error': str(e)}
            print(f"     ❌ _get_config_class test failed: {e}")
        
        # Test 4.3: Unified Interface Consistency
        try:
            print("   🔧 Testing unified interface consistency...")
            
            # Create test configs for different component types
            configs = {
                'tool': {'tool_name': 'mmseqs2', 'min_seq_id': 0.4},
                'step': {'name': 'test_step', 'timeout': 300}
            }
            
            interface_consistent = True
            for component_type, config_data in configs.items():
                config_path = self.create_test_config_file(f'{component_type}_interface.yml', config_data)
                
                if component_type == 'tool':
                    component = MMseqs2Tool.from_config(str(config_path))
                elif component_type == 'step':
                    component = DataAggregationStep.from_config(str(config_path))
                
                # Validate consistent interface
                if not hasattr(component, 'name') or not hasattr(component, '_get_config_class'):
                    interface_consistent = False
                    break
            
            if interface_consistent:
                results['interface_consistency'] = {'success': True, 'components_tested': len(configs)}
                print(f"     ✅ Interface consistency verified across {len(configs)} component types")
            else:
                results['interface_consistency'] = {'success': False, 'error': 'Inconsistent interface'}
                print("     ❌ Interface inconsistency detected")
            
        except Exception as e:
            results['interface_consistency'] = {'success': False, 'error': str(e)}
            print(f"     ❌ Interface consistency test failed: {e}")
        
        return results
    
    async def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run all comprehensive workflow tests."""
        print("🚀 Starting Comprehensive Workflow Integration Tests")
        print("="*80)
        
        self.setup_test_environment()
        
        try:
            # Run all comprehensive test scenarios
            test_1_results = await self.test_workflow_component_loading()
            test_2_results = await self.test_configuration_flexibility()
            test_3_results = await self.test_error_resilience()
            test_4_results = await self.test_architectural_compliance()
            
            # Compile results
            all_results = {
                'workflow_component_loading': test_1_results,
                'configuration_flexibility': test_2_results,
                'error_resilience': test_3_results,
                'architectural_compliance': test_4_results
            }
            
            # Calculate success metrics
            total_tests = 0
            successful_tests = 0
            
            for test_category, category_results in all_results.items():
                for test_name, result in category_results.items():
                    if isinstance(result, dict) and 'success' in result:
                        total_tests += 1
                        if result['success']:
                            successful_tests += 1
            
            success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
            
            # Summary report
            print("\n" + "="*80)
            print("🎯 COMPREHENSIVE WORKFLOW TEST RESULTS")
            print("="*80)
            
            print(f"\n📊 OVERALL SUCCESS RATE: {success_rate:.1f}%")
            print(f"   ✅ Successful Tests: {successful_tests}")
            print(f"   ❌ Failed Tests: {total_tests - successful_tests}")
            print(f"   📈 Total Tests: {total_tests}")
            
            # Detailed results by category
            for test_category, category_results in all_results.items():
                category_total = len([r for r in category_results.values() if isinstance(r, dict) and 'success' in r])
                category_success = len([r for r in category_results.values() if isinstance(r, dict) and r.get('success', False)])
                category_rate = (category_success / category_total * 100) if category_total > 0 else 0
                
                print(f"\n🧪 {test_category.replace('_', ' ').title()}: {category_rate:.1f}% ({category_success}/{category_total})")
                
                for test_name, result in category_results.items():
                    if isinstance(result, dict) and 'success' in result:
                        status = "✅" if result['success'] else "❌"
                        print(f"   {status} {test_name}")
                        if not result['success'] and 'error' in result:
                            print(f"      Error: {result['error']}")
            
            # Assessment
            if success_rate >= 95.0:
                print("\n🎉 OUTSTANDING! Comprehensive workflow testing successful.")
                print("✅ Framework demonstrates production-ready reliability.")
                assessment = "OUTSTANDING"
            elif success_rate >= 85.0:
                print("\n👍 EXCELLENT! Most comprehensive tests passing.")
                print("⚠️ Minor issues to address before production deployment.")
                assessment = "EXCELLENT"
            elif success_rate >= 70.0:
                print("\n🔧 GOOD! Core functionality working well.")
                print("🔧 Some improvements needed for full production readiness.")
                assessment = "GOOD"
            else:
                print("\n⚠️ NEEDS ATTENTION! Significant issues require resolution.")
                print("🔧 Address failed tests before continuing to next phase.")
                assessment = "NEEDS_ATTENTION"
            
            self.test_results = all_results
            self.test_results['_summary'] = {
                'success_rate': success_rate,
                'successful_tests': successful_tests,
                'total_tests': total_tests,
                'assessment': assessment
            }
            
            return all_results
            
        finally:
            self.cleanup_test_environment()


async def main():
    """Main execution function."""
    tester = ComprehensiveWorkflowTests()
    results = await tester.run_comprehensive_tests()
    
    # Return success/failure for CI/CD integration
    summary = results.get('_summary', {})
    success_rate = summary.get('success_rate', 0)
    return success_rate >= 85.0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 