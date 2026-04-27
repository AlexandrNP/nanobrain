#!/usr/bin/env python3
"""
Phase 3 Validation: Configuration Registry Updates and Workflow Configuration Enhancement

Validates that the viral annotation workflow has been successfully enhanced with:
1. Tool dependency declarations in workflow configuration
2. Updated step configurations with tool references
3. Elimination of tool configuration duplication
4. Proper framework compliance metadata
"""

import sys
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, List

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("phase3_configuration_validation")

class Phase3ConfigurationValidator:
    """Validator for Phase 3 configuration enhancements."""
    
    def __init__(self):
        self.workflow_dir = Path("nanobrain/library/workflows/viral_protein_analysis")
        self.validation_results = {
            "workflow_tool_dependencies": False,
            "step_tool_references": {},
            "configuration_deduplication": {},
            "framework_compliance": {},
            "configuration_consistency": False
        }
        
    def validate_workflow_tool_dependencies(self) -> bool:
        """Validate that workflow configuration includes tool dependencies."""
        logger.info("🔧 Validating workflow tool dependencies configuration")
        
        workflow_config_path = self.workflow_dir / "config" / "AlphavirusWorkflow.yml"
        
        if not workflow_config_path.exists():
            logger.error(f"❌ Workflow config not found: {workflow_config_path}")
            return False
            
        try:
            with open(workflow_config_path, 'r') as f:
                workflow_config = yaml.safe_load(f)
                
            # Check for tools section
            if 'tools' not in workflow_config:
                logger.error("❌ Workflow config missing 'tools' section")
                return False
                
            tools = workflow_config['tools']
            expected_tools = ['bv_brc_tool', 'mmseqs2_tool', 'muscle_tool']
            
            for tool_name in expected_tools:
                if tool_name not in tools:
                    logger.error(f"❌ Missing tool definition: {tool_name}")
                    return False
                    
                tool_config = tools[tool_name]
                required_fields = ['config_file', 'description', 'required_for_steps', 'compliance_status']
                
                for field in required_fields:
                    if field not in tool_config:
                        logger.error(f"❌ Tool {tool_name} missing field: {field}")
                        return False
                        
                # Validate compliance status
                if tool_config['compliance_status'] != 'from_config_compliant':
                    logger.error(f"❌ Tool {tool_name} not marked as from_config compliant")
                    return False
                    
                logger.info(f"✅ Tool {tool_name}: Configuration valid")
                
            # Check version update
            if workflow_config.get('version') != '4.4.0':
                logger.error(f"❌ Workflow version not updated: {workflow_config.get('version')}")
                return False
                
            logger.info("✅ Workflow tool dependencies: PASS")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error validating workflow config: {e}")
            return False
    
    def validate_step_tool_references(self) -> Dict[str, bool]:
        """Validate that step configurations include proper tool references."""
        logger.info("📋 Validating step tool references")
        
        steps_to_validate = {
            'data_acquisition': {
                'config_file': 'data_acquisition_config.yml',
                'expected_tool': 'bv_brc_tool'
            },
            'clustering': {
                'config_file': 'clustering_config.yml', 
                'expected_tool': 'mmseqs2_tool'
            },
            'alignment': {
                'config_file': 'alignment_config.yml',
                'expected_tool': 'muscle_tool'
            }
        }
        
        results = {}
        
        for step_name, step_info in steps_to_validate.items():
            config_path = self.workflow_dir / "config" / "steps" / step_info['config_file']
            
            if not config_path.exists():
                logger.error(f"❌ Step config not found: {config_path}")
                results[step_name] = False
                continue
                
            try:
                with open(config_path, 'r') as f:
                    step_config = yaml.safe_load(f)
                
                # Check for tool_dependencies section
                if 'tool_dependencies' not in step_config:
                    logger.error(f"❌ Step {step_name} missing 'tool_dependencies' section")
                    results[step_name] = False
                    continue
                    
                tool_deps = step_config['tool_dependencies']
                expected_tool = step_info['expected_tool']
                
                if expected_tool not in tool_deps:
                    logger.error(f"❌ Step {step_name} missing tool dependency: {expected_tool}")
                    results[step_name] = False
                    continue
                    
                # Validate tool reference path
                tool_ref = tool_deps[expected_tool]
                expected_path = f"config/tools/{expected_tool}.yml"
                
                if tool_ref != expected_path:
                    logger.error(f"❌ Step {step_name} incorrect tool reference: {tool_ref}")
                    results[step_name] = False
                    continue
                    
                # Check metadata for Phase 3 compliance
                metadata = step_config.get('_metadata', {})
                if metadata.get('tool_integration') != 'from_config':
                    logger.error(f"❌ Step {step_name} metadata missing tool_integration marker")
                    results[step_name] = False
                    continue
                    
                logger.info(f"✅ Step {step_name}: Tool reference valid")
                results[step_name] = True
                
            except Exception as e:
                logger.error(f"❌ Error validating step {step_name}: {e}")
                results[step_name] = False
                
        return results
    
    def validate_configuration_deduplication(self) -> Dict[str, bool]:
        """Validate that tool configurations have been removed from step configs."""
        logger.info("🗂️ Validating configuration deduplication")
        
        step_configs = {
            'data_acquisition_config.yml': ['bvbrc_config'],
            'clustering_config.yml': ['clustering_config'],
            'alignment_config.yml': ['alignment_config']
        }
        
        results = {}
        
        for config_file, deprecated_sections in step_configs.items():
            config_path = self.workflow_dir / "config" / "steps" / config_file
            
            if not config_path.exists():
                logger.error(f"❌ Config file not found: {config_path}")
                results[config_file] = False
                continue
                
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Check that deprecated sections are removed
                has_deprecated = False
                for section in deprecated_sections:
                    if section in config:
                        logger.error(f"❌ {config_file} still contains deprecated section: {section}")
                        has_deprecated = True
                        
                if has_deprecated:
                    results[config_file] = False
                else:
                    logger.info(f"✅ {config_file}: Configuration deduplication complete")
                    results[config_file] = True
                    
            except Exception as e:
                logger.error(f"❌ Error validating {config_file}: {e}")
                results[config_file] = False
                
        return results
    
    def validate_framework_compliance(self) -> Dict[str, bool]:
        """Validate framework compliance metadata updates."""
        logger.info("🎯 Validating framework compliance metadata")
        
        config_files = [
            'AlphavirusWorkflow.yml',
            'steps/data_acquisition_config.yml',
            'steps/clustering_config.yml', 
            'steps/alignment_config.yml'
        ]
        
        results = {}
        
        for config_file in config_files:
            config_path = self.workflow_dir / "config" / config_file
            
            if not config_path.exists():
                logger.error(f"❌ Config file not found: {config_path}")
                results[config_file] = False
                continue
                
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                metadata = config.get('_metadata', {})
                
                # Check Phase 3 specific metadata
                required_metadata = {
                    'updated_by': 'Phase3_tool_integration_enhancement',
                    'last_updated': '2025-06-23T22:30:00.000000'
                }
                
                for key, expected_value in required_metadata.items():
                    if metadata.get(key) != expected_value:
                        logger.error(f"❌ {config_file} metadata {key}: {metadata.get(key)} != {expected_value}")
                        results[config_file] = False
                        break
                else:
                    # Additional checks for step configs
                    if config_file.startswith('steps/'):
                        step_metadata = {
                            'config_type': 'step',
                            'compliance_level': 'full',
                            'tool_integration': 'from_config'
                        }
                        
                        for key, expected_value in step_metadata.items():
                            if metadata.get(key) != expected_value:
                                logger.error(f"❌ {config_file} step metadata {key}: {metadata.get(key)} != {expected_value}")
                                results[config_file] = False
                                break
                        else:
                            logger.info(f"✅ {config_file}: Framework compliance metadata valid")
                            results[config_file] = True
                    else:
                        # Workflow file specific checks
                        if metadata.get('tool_integration_version') != '2.0.0':
                            logger.error(f"❌ {config_file} missing tool_integration_version")
                            results[config_file] = False
                        elif not metadata.get('phase3_compliance'):
                            logger.error(f"❌ {config_file} missing phase3_compliance marker")
                            results[config_file] = False
                        else:
                            logger.info(f"✅ {config_file}: Framework compliance metadata valid")
                            results[config_file] = True
                            
            except Exception as e:
                logger.error(f"❌ Error validating {config_file}: {e}")
                results[config_file] = False
                
        return results
    
    def validate_configuration_consistency(self) -> bool:
        """Validate consistency between workflow and step configurations."""
        logger.info("🔄 Validating configuration consistency")
        
        try:
            # Load workflow config
            workflow_config_path = self.workflow_dir / "config" / "AlphavirusWorkflow.yml"
            with open(workflow_config_path, 'r') as f:
                workflow_config = yaml.safe_load(f)
            
            # Check that step dependencies match workflow tool declarations
            tools = workflow_config.get('tools', {})
            steps = workflow_config.get('steps', [])
            
            tool_step_mapping = {}
            for tool_name, tool_config in tools.items():
                required_steps = tool_config.get('required_for_steps', [])
                for step in required_steps:
                    tool_step_mapping[step] = tool_name
            
            # Validate step dependencies
            for step in steps:
                step_id = step.get('step_id')
                step_deps = step.get('dependencies', {})
                step_tools = step_deps.get('tools', [])
                
                # Check if step should have tool dependency
                if step_id in tool_step_mapping:
                    expected_tool = tool_step_mapping[step_id]
                    if expected_tool not in step_tools:
                        logger.error(f"❌ Step {step_id} missing expected tool dependency: {expected_tool}")
                        return False
                        
            logger.info("✅ Configuration consistency: PASS")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error validating configuration consistency: {e}")
            return False
    
    def run_validation(self) -> bool:
        """Run complete Phase 3 validation."""
        logger.info("🚀 Starting Phase 3 configuration enhancement validation")
        
        # Test 1: Workflow tool dependencies
        self.validation_results["workflow_tool_dependencies"] = self.validate_workflow_tool_dependencies()
        
        # Test 2: Step tool references
        self.validation_results["step_tool_references"] = self.validate_step_tool_references()
        
        # Test 3: Configuration deduplication
        self.validation_results["configuration_deduplication"] = self.validate_configuration_deduplication()
        
        # Test 4: Framework compliance
        self.validation_results["framework_compliance"] = self.validate_framework_compliance()
        
        # Test 5: Configuration consistency
        self.validation_results["configuration_consistency"] = self.validate_configuration_consistency()
        
        # Calculate overall results
        workflow_tools_pass = self.validation_results["workflow_tool_dependencies"]
        step_refs_pass = all(self.validation_results["step_tool_references"].values())
        dedup_pass = all(self.validation_results["configuration_deduplication"].values())
        compliance_pass = all(self.validation_results["framework_compliance"].values())
        consistency_pass = self.validation_results["configuration_consistency"]
        
        # Summary
        logger.info("📊 Phase 3 Validation Summary:")
        logger.info(f"  Workflow Tool Dependencies: {'✅ PASS' if workflow_tools_pass else '❌ FAIL'}")
        logger.info(f"  Step Tool References: {'✅ PASS' if step_refs_pass else '❌ FAIL'}")
        for step, result in self.validation_results["step_tool_references"].items():
            logger.info(f"    {step}: {'✅ PASS' if result else '❌ FAIL'}")
        logger.info(f"  Configuration Deduplication: {'✅ PASS' if dedup_pass else '❌ FAIL'}")
        logger.info(f"  Framework Compliance: {'✅ PASS' if compliance_pass else '❌ FAIL'}")
        logger.info(f"  Configuration Consistency: {'✅ PASS' if consistency_pass else '❌ FAIL'}")
        
        overall_success = all([workflow_tools_pass, step_refs_pass, dedup_pass, compliance_pass, consistency_pass])
        
        if overall_success:
            logger.info("🎉 Phase 3: Configuration Registry Updates and Workflow Configuration Enhancement - COMPLETE ✅")
        else:
            logger.error("❌ Some Phase 3 configuration enhancement validations FAILED!")
            
        return overall_success


if __name__ == "__main__":
    validator = Phase3ConfigurationValidator()
    success = validator.run_validation()
    sys.exit(0 if success else 1) 