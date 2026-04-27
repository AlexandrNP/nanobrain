#!/usr/bin/env python3
"""
Complete Viral Workflow Tool Integration Validation

Master validation script that comprehensively tests ALL phases (1-3) of the viral annotation 
workflow tool integration enhancement:

Phase 1: Workflow-Level Tool Configuration
Phase 2: Step Tool Integration  
Phase 3: Configuration Registry Updates and Workflow Configuration Enhancement

Validates end-to-end integration compliance with mandatory from_config patterns.
"""

import sys
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("complete_viral_workflow_integration_validation")

class CompleteIntegrationValidator:
    """Master validator for complete viral workflow tool integration."""
    
    def __init__(self):
        self.validation_scripts = [
            {
                'name': 'Phase 1: Workflow-Level Tool Configuration',
                'script': 'scripts/validate_viral_workflow_tools.py',
                'description': 'Validates workflow-local tool configuration files and from_config creation'
            },
            {
                'name': 'Phase 2: Step Tool Integration',
                'script': 'scripts/validate_viral_workflow_step_integration.py',
                'description': 'Validates step classes integrate tools via from_config pattern'
            },
            {
                'name': 'Phase 3: Configuration Registry Updates',
                'script': 'scripts/validate_phase3_configuration_enhancements.py', 
                'description': 'Validates workflow configuration enhancements and tool dependencies'
            }
        ]
        
        self.results = {}
        
    def run_validation_script(self, script_info: Dict) -> Tuple[bool, str]:
        """Run a validation script and return success status and output."""
        script_path = script_info['script']
        script_name = script_info['name']
        
        logger.info(f"🔄 Running {script_name}")
        logger.info(f"   Script: {script_path}")
        logger.info(f"   Description: {script_info['description']}")
        
        try:
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                cwd=Path.cwd(),
                timeout=120  # 2 minute timeout
            )
            
            success = result.returncode == 0
            output = result.stdout + result.stderr if result.stderr else result.stdout
            
            if success:
                logger.info(f"✅ {script_name}: PASS")
            else:
                logger.error(f"❌ {script_name}: FAIL (exit code: {result.returncode})")
                if result.stderr:
                    logger.error(f"   Error output: {result.stderr[:500]}...")
                    
            return success, output
            
        except subprocess.TimeoutExpired:
            logger.error(f"❌ {script_name}: TIMEOUT (exceeded 2 minutes)")
            return False, "Script execution timeout"
        except Exception as e:
            logger.error(f"❌ {script_name}: ERROR ({str(e)})")
            return False, f"Script execution error: {str(e)}"
    
    def validate_overall_architecture(self) -> bool:
        """Validate overall architecture compliance."""
        logger.info("🏗️ Validating overall architecture compliance")
        
        # Check that all expected files exist
        expected_files = [
            "nanobrain/library/workflows/viral_protein_analysis/config/tools/bv_brc_tool.yml",
            "nanobrain/library/workflows/viral_protein_analysis/config/tools/mmseqs2_tool.yml",
            "nanobrain/library/workflows/viral_protein_analysis/config/tools/muscle_tool.yml",
            "nanobrain/library/workflows/viral_protein_analysis/config/steps/data_acquisition_config.yml",
            "nanobrain/library/workflows/viral_protein_analysis/config/steps/clustering_config.yml",
            "nanobrain/library/workflows/viral_protein_analysis/config/steps/alignment_config.yml",
            "nanobrain/library/workflows/viral_protein_analysis/config/AlphavirusWorkflow.yml"
        ]
        
        missing_files = []
        for file_path in expected_files:
            if not Path(file_path).exists():
                missing_files.append(file_path)
                
        if missing_files:
            logger.error(f"❌ Missing required files: {missing_files}")
            return False
            
        logger.info("✅ All required configuration files present")
        
        # Check step class files exist
        step_files = [
            "nanobrain/library/workflows/viral_protein_analysis/steps/bv_brc_data_acquisition_step.py",
            "nanobrain/library/workflows/viral_protein_analysis/steps/clustering_step.py",
            "nanobrain/library/workflows/viral_protein_analysis/steps/alignment_step.py"
        ]
        
        missing_steps = []
        for step_file in step_files:
            if not Path(step_file).exists():
                missing_steps.append(step_file)
                
        if missing_steps:
            logger.error(f"❌ Missing step implementation files: {missing_steps}")
            return False
            
        logger.info("✅ All step implementation files present")
        logger.info("✅ Overall architecture: PASS")
        return True
    
    def generate_compliance_report(self) -> None:
        """Generate comprehensive compliance report."""
        logger.info("📊 Generating Comprehensive Compliance Report")
        logger.info("=" * 80)
        
        # Overall status
        all_passed = all(self.results.values()) and self.results.get('architecture', False)
        
        logger.info(f"🎯 VIRAL ANNOTATION WORKFLOW TOOL INTEGRATION STATUS")
        logger.info(f"Overall Status: {'✅ COMPLETE SUCCESS' if all_passed else '❌ SOME FAILURES'}")
        logger.info("")
        
        # Phase-by-phase results
        logger.info("📋 Phase-by-Phase Results:")
        for i, script_info in enumerate(self.validation_scripts, 1):
            phase_name = script_info['name']
            phase_status = '✅ PASS' if self.results.get(phase_name, False) else '❌ FAIL'
            logger.info(f"  Phase {i}: {phase_name} - {phase_status}")
            
        # Architecture validation
        arch_status = '✅ PASS' if self.results.get('architecture', False) else '❌ FAIL'
        logger.info(f"  Architecture Compliance - {arch_status}")
        logger.info("")
        
        # Key achievements
        if all_passed:
            logger.info("🏆 KEY ACHIEVEMENTS:")
            logger.info("  ✅ Workflow-local tool configurations created and validated")
            logger.info("  ✅ Step classes updated with from_config tool integration")
            logger.info("  ✅ Configuration registry enhanced with tool dependencies")
            logger.info("  ✅ All tool configs use mandatory from_config pattern")
            logger.info("  ✅ Configuration deduplication completed")
            logger.info("  ✅ Framework compliance metadata implemented")
            logger.info("  ✅ End-to-end tool integration validation successful")
            logger.info("")
            logger.info("🚀 VIRAL WORKFLOW READY FOR PRODUCTION EXECUTION")
        else:
            logger.info("⚠️  ISSUES DETECTED:")
            for script_name, success in self.results.items():
                if not success:
                    logger.info(f"  ❌ {script_name}")
                    
        logger.info("=" * 80)
    
    def run_complete_validation(self) -> bool:
        """Run complete validation of all phases."""
        logger.info("🚀 Starting Complete Viral Workflow Tool Integration Validation")
        logger.info(f"Validating ALL phases (1-3) of tool integration enhancement")
        logger.info("")
        
        # Phase 1: Architecture validation
        self.results['architecture'] = self.validate_overall_architecture()
        logger.info("")
        
        # Phase 2-4: Run validation scripts
        for script_info in self.validation_scripts:
            success, output = self.run_validation_script(script_info)
            self.results[script_info['name']] = success
            logger.info("")
            
        # Generate report
        self.generate_compliance_report()
        
        # Overall success
        overall_success = all(self.results.values())
        
        if overall_success:
            logger.info("🎉 COMPLETE VIRAL WORKFLOW TOOL INTEGRATION - 100% SUCCESS! 🎉")
        else:
            logger.error("❌ SOME VALIDATIONS FAILED - REVIEW ISSUES ABOVE")
            
        return overall_success


if __name__ == "__main__":
    validator = CompleteIntegrationValidator()
    success = validator.run_complete_validation()
    sys.exit(0 if success else 1) 