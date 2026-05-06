#!/usr/bin/env python3
"""
Workflow Configuration Compliance Test

Validates that chatbot_viral_integration and viral_protein_analysis workflows
comply with enhanced from_config patterns.

✅ FRAMEWORK COMPLIANCE:
- Dict-based steps/links/triggers format
- Class+config object instantiation patterns
- Enhanced from_config usage
- No programmatic component creation
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, List
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WorkflowConfigurationComplianceTest:
    """Test suite for workflow configuration compliance"""
    
    def __init__(self):
        self.test_results = []
        self.all_tests_passed = True
    
    def test_chatbot_workflow_format_compliance(self) -> bool:
        """Test chatbot workflow configuration format compliance"""
        logger.info("🧪 Testing Chatbot Workflow Format Compliance")
        
        config_path = "nanobrain/library/workflows/chatbot_viral_integration/ChatbotViralWorkflow.yml"
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Test Dict-based steps format
            if 'steps' not in config:
                self._record_failure("Chatbot Workflow", "Missing steps configuration")
                return False
            
            steps = config['steps']
            if not isinstance(steps, dict):
                self._record_failure("Chatbot Workflow", "Steps must be Dict-based format")
                return False
            
            # Test class+config patterns in steps
            for step_id, step_config in steps.items():
                if not isinstance(step_config, dict):
                    self._record_failure("Chatbot Workflow", f"Step {step_id} must be Dict format")
                    return False
                
                if 'class' not in step_config:
                    self._record_failure("Chatbot Workflow", f"Step {step_id} missing 'class' field")
                    return False
                
                if 'config' not in step_config:
                    self._record_failure("Chatbot Workflow", f"Step {step_id} missing 'config' field")
                    return False
            
            # Test Dict-based links format
            if 'links' in config:
                links = config['links']
                if not isinstance(links, dict):
                    self._record_failure("Chatbot Workflow", "Links must be Dict-based format")
                    return False
            
            # Test Dict-based triggers format
            if 'triggers' in config:
                triggers = config['triggers']
                if not isinstance(triggers, dict):
                    self._record_failure("Chatbot Workflow", "Triggers must be Dict-based format")
                    return False
            
            self._record_success("Chatbot Workflow", "Format compliance validated")
            return True
            
        except Exception as e:
            self._record_failure("Chatbot Workflow", f"Exception: {str(e)}")
            return False
    
    def test_viral_analysis_workflow_format_compliance(self) -> bool:
        """Test viral protein analysis workflow configuration format compliance"""
        logger.info("🧪 Testing Viral Analysis Workflow Format Compliance")
        
        config_path = "nanobrain/library/workflows/viral_protein_analysis/config/AlphavirusWorkflow.yml"
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Test Dict-based steps format
            if 'steps' not in config:
                self._record_failure("Viral Analysis Workflow", "Missing steps configuration")
                return False
            
            steps = config['steps']
            if isinstance(steps, list):
                self._record_failure("Viral Analysis Workflow", "Steps still in list format - must be Dict-based")
                return False
            
            if not isinstance(steps, dict):
                self._record_failure("Viral Analysis Workflow", "Steps must be Dict-based format")
                return False
            
            # Test class+config patterns in steps
            for step_id, step_config in steps.items():
                if not isinstance(step_config, dict):
                    self._record_failure("Viral Analysis Workflow", f"Step {step_id} must be Dict format")
                    return False
                
                if 'class' not in step_config:
                    self._record_failure("Viral Analysis Workflow", f"Step {step_id} missing 'class' field")
                    return False
                
                if 'config' not in step_config:
                    self._record_failure("Viral Analysis Workflow", f"Step {step_id} missing 'config' field")
                    return False
            
            self._record_success("Viral Analysis Workflow", "Format compliance validated")
            return True
            
        except Exception as e:
            self._record_failure("Viral Analysis Workflow", f"Exception: {str(e)}")
            return False
    
    def test_step_configuration_compliance(self) -> bool:
        """Test step configuration compliance with enhanced patterns"""
        logger.info("🧪 Testing Step Configuration Compliance")
        
        # Test key step configurations
        step_configs = [
            "nanobrain/library/workflows/chatbot_viral_integration/config/QueryClassificationStep/QueryClassificationStep.yml",
            "nanobrain/library/workflows/viral_protein_analysis/config/DataAcquisitionStep/DataAcquisitionStep.yml"
        ]
        
        for config_path in step_configs:
            try:
                if not Path(config_path).exists():
                    continue
                
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Test class field presence
                if 'class' not in config:
                    self._record_failure("Step Configuration", f"{config_path} missing 'class' field")
                    return False
                
                # Test config section
                if 'config' not in config:
                    self._record_failure("Step Configuration", f"{config_path} missing 'config' section")
                    return False
                
                # Test for class+config patterns in agents/tools
                step_config = config.get('config', {})
                for section_name in ['agents', 'tools', 'extraction_agent', 'conversational_agent']:
                    if section_name in step_config:
                        section = step_config[section_name]
                        if isinstance(section, dict):
                            # Check for class+config patterns
                            for item_name, item_config in section.items():
                                if isinstance(item_config, dict) and 'class' in item_config:
                                    if 'config' not in item_config:
                                        self._record_failure("Step Configuration", 
                                            f"{config_path} {section_name}.{item_name} has 'class' but missing 'config'")
                                        return False
                
            except Exception as e:
                self._record_failure("Step Configuration", f"Error reading {config_path}: {str(e)}")
                return False
        
        self._record_success("Step Configuration", "Configuration compliance validated")
        return True
    
    def test_python_code_compliance(self) -> bool:
        """Test Python code compliance with enhanced from_config patterns"""
        logger.info("🧪 Testing Python Code Compliance")
        
        # Test ChatbotViralWorkflow Python code
        chatbot_workflow_file = "nanobrain/library/workflows/chatbot_viral_integration/chatbot_viral_workflow.py"
        try:
            with open(chatbot_workflow_file, 'r') as f:
                content = f.read()
            
            # Check for removal of deprecated create_step usage
            if 'from nanobrain.core.step import create_step' in content:
                self._record_failure("Python Code", "ChatbotViralWorkflow still imports deprecated create_step")
                return False
            
            # Check for enhanced from_config usage
            if '_create_step_instance' in content and 'step_cls.from_config' not in content:
                self._record_failure("Python Code", "ChatbotViralWorkflow not using enhanced from_config in _create_step_instance")
                return False
                
        except Exception as e:
            self._record_failure("Python Code", f"Error reading ChatbotViralWorkflow: {str(e)}")
            return False
        
        # Test AlphavirusWorkflow Python code
        alphavirus_workflow_file = "nanobrain/library/workflows/viral_protein_analysis/alphavirus_workflow.py"
        try:
            with open(alphavirus_workflow_file, 'r') as f:
                content = f.read()
            
            # Check for removal of deprecated create_component usage
            if 'from nanobrain.core.config.component_factory import load_config_file, create_component' in content:
                self._record_failure("Python Code", "AlphavirusWorkflow still imports deprecated component_factory")
                return False
            
            # Check for enhanced from_config usage
            if '_resolve_step_from_config' in content and 'step_cls.from_config' not in content:
                self._record_failure("Python Code", "AlphavirusWorkflow not using enhanced from_config in _resolve_step_from_config")
                return False
                
        except Exception as e:
            self._record_failure("Python Code", f"Error reading AlphavirusWorkflow: {str(e)}")
            return False
        
        self._record_success("Python Code", "Code compliance validated")
        return True
    
    def _record_success(self, category: str, message: str):
        """Record successful test"""
        self.test_results.append({
            "category": category,
            "status": "PASSED",
            "message": message
        })
        logger.info(f"✅ {category}: {message}")
    
    def _record_failure(self, category: str, error: str):
        """Record failed test"""
        self.test_results.append({
            "category": category,
            "status": "FAILED",
            "error": error
        })
        self.all_tests_passed = False
        logger.error(f"❌ {category}: {error}")
    
    def run_all_tests(self) -> bool:
        """Run all compliance tests"""
        logger.info("🚀 Starting Workflow Configuration Compliance Tests")
        
        # Run all tests
        self.test_chatbot_workflow_format_compliance()
        self.test_viral_analysis_workflow_format_compliance()
        self.test_step_configuration_compliance()
        self.test_python_code_compliance()
        
        # Report results
        passed = sum(1 for r in self.test_results if r["status"] == "PASSED")
        failed = sum(1 for r in self.test_results if r["status"] == "FAILED")
        
        logger.info(f"\n📊 Test Results: {passed} passed, {failed} failed")
        
        if self.all_tests_passed:
            logger.info("✅ ALL TESTS PASSED - Workflows are framework compliant")
        else:
            logger.error("❌ TESTS FAILED - Workflows need compliance updates")
            for result in self.test_results:
                if result["status"] == "FAILED":
                    logger.error(f"  ❌ {result['category']}: {result['error']}")
        
        return self.all_tests_passed


if __name__ == "__main__":
    test_suite = WorkflowConfigurationComplianceTest()
    success = test_suite.run_all_tests()
    sys.exit(0 if success else 1) 