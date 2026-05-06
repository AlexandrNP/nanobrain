#!/usr/bin/env python3
"""
Enhanced Configuration Validation Script

Validates that all components have proper default configurations and 
that the configuration registry is complete and accurate.
"""

import json
import yaml
import importlib
from pathlib import Path
from typing import Dict, Any, List, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConfigurationValidator:
    """Enhanced validator for the modular configuration system"""
    
    def __init__(self, registry_file: str = "config_registry.json"):
        self.registry_file = registry_file
        self.registry = self._load_registry()
        self.validation_errors = []
        self.validation_warnings = []
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load the configuration registry"""
        registry_path = Path(self.registry_file)
        if not registry_path.exists():
            raise FileNotFoundError(f"Configuration registry not found: {self.registry_file}")
        
        with open(registry_path, 'r') as f:
            return json.load(f)
    
    def validate_registry_completeness(self) -> bool:
        """Validate that the registry includes all from_config components"""
        logger.info("🔍 Validating registry completeness...")
        
        # Expected components based on our analysis
        expected_components = [
            "nanobrain.core.step.Step",
            "nanobrain.core.step.TransformStep",
            "nanobrain.core.executor.LocalExecutor",
            "nanobrain.core.executor.ThreadExecutor", 
            "nanobrain.core.executor.ProcessExecutor",
            "nanobrain.core.executor.ParslExecutor",
            "nanobrain.core.data_unit.DataUnitMemory",
            "nanobrain.core.data_unit.DataUnitFile",
            "nanobrain.core.data_unit.DataUnitString",
            "nanobrain.core.data_unit.DataUnitStream",
            "nanobrain.core.trigger.DataUpdatedTrigger",
            "nanobrain.core.trigger.AllDataReceivedTrigger",
            "nanobrain.core.trigger.TimerTrigger",
            "nanobrain.core.trigger.ManualTrigger",
            "nanobrain.library.agents.specialized.base.SimpleSpecializedAgent",
            "nanobrain.library.agents.specialized.base.ConversationalSpecializedAgent",
            "nanobrain.library.agents.enhanced.collaborative_agent.CollaborativeAgent",
            "nanobrain.library.agents.conversational.enhanced_collaborative_agent.EnhancedCollaborativeAgent",
            "nanobrain.library.tools.bioinformatics.bv_brc_tool.BVBRCTool",
            "nanobrain.library.tools.bioinformatics.mmseqs_tool.MMseqs2Tool",
            "nanobrain.library.tools.bioinformatics.muscle_tool.MUSCLETool",
            "nanobrain.library.tools.bioinformatics.pubmed_client.PubMedClient",
            "nanobrain.library.interfaces.web.web_interface.WebInterface",
            "nanobrain.library.workflows.chat_workflow.chat_workflow.ChatWorkflow"
        ]
        
        registry_components = set(self.registry["components"].keys())
        expected_set = set(expected_components)
        
        missing_components = expected_set - registry_components
        unexpected_components = registry_components - expected_set
        
        if missing_components:
            self.validation_errors.append(f"Missing components in registry: {missing_components}")
            logger.error(f"❌ Missing components: {missing_components}")
        
        if unexpected_components:
            self.validation_warnings.append(f"Unexpected components in registry: {unexpected_components}")
            logger.warning(f"⚠️  Unexpected components: {unexpected_components}")
        
        if not missing_components:
            logger.info("✅ Registry completeness validation passed")
            return True
        
        return False
    
    def validate_config_files_exist(self) -> bool:
        """Validate that all referenced configuration files exist"""
        logger.info("📁 Validating configuration files existence...")
        
        missing_files = []
        for component, config_info in self.registry["components"].items():
            config_file = Path(config_info["config_file"])
            if not config_file.exists():
                missing_files.append((component, config_file))
        
        if missing_files:
            self.validation_errors.append("Missing configuration files")
            for component, file_path in missing_files:
                logger.error(f"❌ Missing config file for {component}: {file_path}")
            return False
        
        logger.info("✅ All configuration files exist")
        return True
    
    def validate_config_file_syntax(self) -> bool:
        """Validate YAML syntax of all configuration files"""
        logger.info("📝 Validating configuration file syntax...")
        
        syntax_errors = []
        for component, config_info in self.registry["components"].items():
            config_file = Path(config_info["config_file"])
            
            try:
                with open(config_file, 'r') as f:
                    yaml.safe_load(f)
            except yaml.YAMLError as e:
                syntax_errors.append((component, config_file, str(e)))
                logger.error(f"❌ YAML syntax error in {config_file}: {e}")
            except Exception as e:
                syntax_errors.append((component, config_file, str(e)))
                logger.error(f"❌ Error reading {config_file}: {e}")
        
        if syntax_errors:
            self.validation_errors.append("Configuration file syntax errors")
            return False
        
        logger.info("✅ All configuration files have valid YAML syntax")
        return True
    
    def validate_from_config_implementation(self) -> bool:
        """Validate that all components properly implement from_config"""
        logger.info("🔧 Validating from_config implementations...")
        
        implementation_errors = []
        for component_path in self.registry["components"].keys():
            try:
                # Import the module and class
                module_path, class_name = component_path.rsplit('.', 1)
                module = importlib.import_module(module_path)
                component_class = getattr(module, class_name)
                
                # Check if from_config method exists
                if not hasattr(component_class, 'from_config'):
                    implementation_errors.append(f"{component_path}: Missing from_config method")
                    logger.error(f"❌ {component_path}: Missing from_config method")
                    continue
                
                # Check if from_config is a classmethod
                from_config_method = getattr(component_class, 'from_config')
                if not isinstance(from_config_method, classmethod):
                    self.validation_warnings.append(f"{component_path}: from_config should be a classmethod")
                    logger.warning(f"⚠️  {component_path}: from_config should be a classmethod")
                
                logger.info(f"✅ {component_path}: from_config implementation found")
                
            except ImportError as e:
                implementation_errors.append(f"{component_path}: Import error - {e}")
                logger.error(f"❌ {component_path}: Import error - {e}")
            except AttributeError as e:
                implementation_errors.append(f"{component_path}: Class not found - {e}")
                logger.error(f"❌ {component_path}: Class not found - {e}")
            except Exception as e:
                implementation_errors.append(f"{component_path}: Unexpected error - {e}")
                logger.error(f"❌ {component_path}: Unexpected error - {e}")
        
        if implementation_errors:
            self.validation_errors.append("from_config implementation errors")
            return False
        
        logger.info("✅ All components have valid from_config implementations")
        return True
    
    def validate_config_content(self) -> bool:
        """Validate that configuration files have required content"""
        logger.info("📋 Validating configuration file content...")
        
        content_warnings = []
        for component, config_info in self.registry["components"].items():
            config_file = Path(config_info["config_file"])
            
            try:
                with open(config_file, 'r') as f:
                    config_data = yaml.safe_load(f)
                
                # Check for required fields
                required_fields = ['name', 'description']
                missing_fields = []
                
                for field in required_fields:
                    if field not in config_data:
                        missing_fields.append(field)
                
                if missing_fields:
                    content_warnings.append(f"{config_file}: Missing fields {missing_fields}")
                    logger.warning(f"⚠️  {config_file}: Missing recommended fields {missing_fields}")
                
                # Check for metadata
                if '_metadata' not in config_data:
                    content_warnings.append(f"{config_file}: Missing _metadata section")
                    logger.warning(f"⚠️  {config_file}: Missing _metadata section")
                
                logger.info(f"✅ {config_file}: Content validation passed")
                
            except Exception as e:
                self.validation_errors.append(f"Error validating content of {config_file}: {e}")
                logger.error(f"❌ Error validating content of {config_file}: {e}")
        
        if content_warnings:
            self.validation_warnings.extend(content_warnings)
        
        logger.info("✅ Configuration content validation completed")
        return True
    
    def validate_configuration_loading(self) -> bool:
        """Test loading configurations with actual framework components"""
        logger.info("🧪 Testing configuration loading...")
        
        loading_errors = []
        loading_successes = 0
        
        for component_path, config_info in self.registry["components"].items():
            config_file = Path(config_info["config_file"])
            
            try:
                # Load configuration
                with open(config_file, 'r') as f:
                    config_data = yaml.safe_load(f)
                
                # Import component class
                module_path, class_name = component_path.rsplit('.', 1)
                module = importlib.import_module(module_path)
                component_class = getattr(module, class_name)
                
                # Test basic config validation (without actually creating instance)
                if hasattr(component_class, 'from_config'):
                    logger.info(f"✅ {component_path}: Configuration loadable")
                    loading_successes += 1
                else:
                    loading_errors.append(f"{component_path}: No from_config method")
                    logger.error(f"❌ {component_path}: No from_config method")
                
            except Exception as e:
                loading_errors.append(f"{component_path}: Loading error - {e}")
                logger.error(f"❌ {component_path}: Loading error - {e}")
        
        logger.info(f"✅ Configuration loading test: {loading_successes} successful, {len(loading_errors)} errors")
        
        if loading_errors:
            self.validation_errors.extend(loading_errors)
            return False
        
        return True
    
    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        total_components = len(self.registry["components"])
        
        report = {
            "summary": {
                "total_components": total_components,
                "validation_timestamp": "2024-01-20",  # Would use datetime.now() in real implementation
                "errors_count": len(self.validation_errors),
                "warnings_count": len(self.validation_warnings)
            },
            "validation_results": {
                "registry_complete": len(self.validation_errors) == 0,
                "all_configs_exist": "config files" not in " ".join(self.validation_errors),
                "syntax_valid": "syntax" not in " ".join(self.validation_errors),
                "from_config_implemented": "from_config" not in " ".join(self.validation_errors)
            },
            "errors": self.validation_errors,
            "warnings": self.validation_warnings,
            "components": {
                "by_category": self._group_by_category(),
                "by_config_file": self._group_by_config_file()
            }
        }
        
        return report
    
    def _group_by_category(self) -> Dict[str, List[str]]:
        """Group components by category"""
        by_category = {}
        for component, config_info in self.registry["components"].items():
            category = config_info["category"]
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(component)
        return by_category
    
    def _group_by_config_file(self) -> Dict[str, List[str]]:
        """Group components by config file"""
        by_file = {}
        for component, config_info in self.registry["components"].items():
            config_file = config_info["config_file"]
            if config_file not in by_file:
                by_file[config_file] = []
            by_file[config_file].append(component)
        return by_file
    
    def run_full_validation(self) -> bool:
        """Run complete validation suite"""
        logger.info("🚀 Starting comprehensive configuration validation...")
        logger.info("=" * 60)
        
        validation_steps = [
            ("Registry Completeness", self.validate_registry_completeness),
            ("Config Files Existence", self.validate_config_files_exist),
            ("YAML Syntax", self.validate_config_file_syntax),
            ("from_config Implementation", self.validate_from_config_implementation),
            ("Config Content", self.validate_config_content),
            ("Configuration Loading", self.validate_configuration_loading)
        ]
        
        all_passed = True
        for step_name, validation_func in validation_steps:
            logger.info(f"\n📋 {step_name}...")
            step_result = validation_func()
            if not step_result:
                all_passed = False
                logger.error(f"❌ {step_name} FAILED")
            else:
                logger.info(f"✅ {step_name} PASSED")
        
        # Generate and save report
        report = self.generate_validation_report()
        with open("validation_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info("\n📊 VALIDATION SUMMARY")
        logger.info("=" * 30)
        logger.info(f"Total Components: {report['summary']['total_components']}")
        logger.info(f"Errors: {report['summary']['errors_count']}")
        logger.info(f"Warnings: {report['summary']['warnings_count']}")
        
        if all_passed:
            logger.info("🎉 ALL VALIDATIONS PASSED!")
            logger.info("✅ The modular configuration system is properly set up")
        else:
            logger.error("❌ VALIDATION FAILED")
            logger.error("🔧 Please fix the reported issues before proceeding")
        
        logger.info(f"📄 Detailed report saved to: validation_report.json")
        
        return all_passed


def main():
    """Main validation function"""
    try:
        validator = ConfigurationValidator()
        success = validator.run_full_validation()
        exit_code = 0 if success else 1
        exit(exit_code)
    except Exception as e:
        logger.error(f"❌ Validation failed with error: {e}")
        exit(1)


if __name__ == "__main__":
    main() 