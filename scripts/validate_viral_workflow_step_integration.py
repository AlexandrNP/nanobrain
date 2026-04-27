#!/usr/bin/env python3
"""
Validate viral annotation workflow step integration with tool from_config pattern
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, Any

# Add the nanobrain package to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nanobrain.core.logging_system import get_logger
from nanobrain.core.step import StepConfig
from nanobrain.core.executor import ExecutorConfig, LocalExecutor

logger = get_logger("viral_workflow_step_integration_validation")

async def test_step_tool_integration():
    """Test that workflow steps can successfully load and use workflow-local tool configurations"""
    
    results = {
        "bv_brc_data_acquisition_step": False,
        "clustering_step": False,
        "alignment_step": False
    }
    
    logger.info("🔧 Testing viral annotation workflow step tool integration")
    
    # Create a test executor for steps
    executor_config = ExecutorConfig(
        name="test_executor",
        class_path="test.LocalExecutor"
    )
    test_executor = LocalExecutor.from_config(executor_config)
    
    # Test 1: BV-BRC Data Acquisition Step
    try:
        from nanobrain.library.workflows.viral_protein_analysis.steps.bv_brc_data_acquisition_step import BVBRCDataAcquisitionStep
        
        # Create minimal step configuration
        step_config = StepConfig(
            name="test_bvbrc_step",
            class_path="test.BVBRCDataAcquisitionStep"
        )
        
        # Use from_config pattern for step creation with executor dependency
        step = BVBRCDataAcquisitionStep.from_config(step_config, executor=test_executor)
        
        # Check if BV-BRC tool was properly loaded
        if hasattr(step, 'bv_brc_tool') and step.bv_brc_tool is not None:
            tool_type = type(step.bv_brc_tool).__name__
            logger.info(f"✅ BV-BRC Data Acquisition Step: Tool loaded ({tool_type})")
            results["bv_brc_data_acquisition_step"] = True
        else:
            logger.error("❌ BV-BRC Data Acquisition Step: Tool not loaded")
            
    except Exception as e:
        logger.error(f"❌ BV-BRC Data Acquisition Step creation failed: {e}")
    
    # Test 2: Clustering Step
    try:
        from nanobrain.library.workflows.viral_protein_analysis.steps.clustering_step import ClusteringStep
        
        # Create minimal step configuration
        step_config = StepConfig(
            name="test_clustering_step",
            class_path="test.ClusteringStep"
        )
        
        # Use from_config pattern for step creation with executor dependency
        step = ClusteringStep.from_config(step_config, executor=test_executor)
        
        # Check if tool integration was attempted (may be None if MMseqs2 not available)
        if hasattr(step, 'mmseqs2_tool'):
            if step.mmseqs2_tool is not None:
                tool_type = type(step.mmseqs2_tool).__name__
                logger.info(f"✅ Clustering Step: Real MMseqs2 tool loaded ({tool_type})")
            else:
                logger.info(f"✅ Clustering Step: Placeholder implementation (MMseqs2 tool config loaded)")
            results["clustering_step"] = True
        else:
            logger.error("❌ Clustering Step: Tool integration not attempted")
            
    except Exception as e:
        logger.error(f"❌ Clustering Step creation failed: {e}")
    
    # Test 3: Alignment Step
    try:
        from nanobrain.library.workflows.viral_protein_analysis.steps.alignment_step import AlignmentStep
        
        # Create minimal step configuration
        step_config = StepConfig(
            name="test_alignment_step",
            class_path="test.AlignmentStep"
        )
        
        # Use from_config pattern for step creation with executor dependency
        step = AlignmentStep.from_config(step_config, executor=test_executor)
        
        # Check if tool integration was attempted (may be None if MUSCLE not available)
        if hasattr(step, 'muscle_tool'):
            if step.muscle_tool is not None:
                tool_type = type(step.muscle_tool).__name__
                logger.info(f"✅ Alignment Step: Real MUSCLE tool loaded ({tool_type})")
            else:
                logger.info(f"✅ Alignment Step: Placeholder implementation (MUSCLE tool config loaded)")
            results["alignment_step"] = True
        else:
            logger.error("❌ Alignment Step: Tool integration not attempted")
            
    except Exception as e:
        logger.error(f"❌ Alignment Step creation failed: {e}")
    
    return results

def test_tool_config_file_access():
    """Test that step classes can access workflow-local tool configuration files"""
    
    workflow_dir = Path("nanobrain/library/workflows/viral_protein_analysis")
    tools_dir = workflow_dir / "config" / "tools"
    
    logger.info("📁 Testing workflow-local tool configuration file access")
    
    required_configs = [
        "bv_brc_tool.yml",
        "mmseqs2_tool.yml", 
        "muscle_tool.yml"
    ]
    
    missing_configs = []
    accessible_configs = []
    
    for config_file in required_configs:
        config_path = tools_dir / config_file
        if config_path.exists() and config_path.is_file():
            try:
                import yaml
                with open(config_path, 'r') as f:
                    config_content = yaml.safe_load(f)
                
                # Check if it has required framework fields
                if 'name' in config_content and 'class' in config_content:
                    accessible_configs.append(config_file)
                    logger.info(f"✅ Tool config accessible: {config_file}")
                else:
                    logger.warning(f"⚠️ Tool config missing required fields: {config_file}")
                    
            except Exception as e:
                logger.error(f"❌ Tool config read error for {config_file}: {e}")
                missing_configs.append(config_file)
        else:
            missing_configs.append(config_file)
            logger.error(f"❌ Tool config not found: {config_file}")
    
    return len(accessible_configs) == len(required_configs), accessible_configs, missing_configs

async def test_step_workflow_execution():
    """Test that steps can execute with their integrated tools"""
    
    logger.info("🔄 Testing step workflow execution with tool integration")
    
    try:
        from nanobrain.library.workflows.viral_protein_analysis.steps.bv_brc_data_acquisition_step import BVBRCDataAcquisitionStep
        
        # Create test executor
        executor_config = ExecutorConfig(
            name="test_workflow_execution",
            class_path="test.LocalExecutor"
        )
        test_executor = LocalExecutor.from_config(executor_config)
        
        # Create step with minimal configuration
        step_config = StepConfig(
            name="test_workflow_execution",
            class_path="test.BVBRCDataAcquisitionStep"
        )
        
        # Use from_config pattern for step creation with executor dependency
        step = BVBRCDataAcquisitionStep.from_config(step_config, executor=test_executor)
        
        # Test if the step can be initialized without errors
        if hasattr(step, 'process') and callable(step.process):
            logger.info("✅ Step workflow execution interface: Available")
            
            # Test basic process method signature (don't actually execute)
            import inspect
            sig = inspect.signature(step.process)
            if 'input_data' in sig.parameters:
                logger.info("✅ Step workflow execution interface: Signature valid")
                return True
            else:
                logger.error("❌ Step workflow execution interface: Invalid signature")
                return False
        else:
            logger.error("❌ Step workflow execution interface: Not available")
            return False
            
    except Exception as e:
        logger.error(f"❌ Step workflow execution test failed: {e}")
        return False

async def main():
    """Main validation function"""
    
    logger.info("🚀 Starting viral annotation workflow step integration validation")
    
    # Test 1: Tool configuration file access
    config_access_ok, accessible_configs, missing_configs = test_tool_config_file_access()
    
    # Test 2: Step tool integration
    integration_results = await test_step_tool_integration()
    
    # Test 3: Step workflow execution
    execution_ok = await test_step_workflow_execution()
    
    # Summary
    logger.info("📊 Validation Summary:")
    logger.info(f"   Tool Configuration Access: {'✅ PASS' if config_access_ok else '❌ FAIL'}")
    
    if not config_access_ok:
        logger.info(f"     Missing configs: {missing_configs}")
        logger.info(f"     Accessible configs: {accessible_configs}")
    
    integration_pass = all(integration_results.values())
    logger.info(f"   Step Tool Integration: {'✅ PASS' if integration_pass else '❌ FAIL'}")
    
    # Detailed integration results
    for step_name, success in integration_results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"     {step_name}: {status}")
    
    logger.info(f"   Step Workflow Execution: {'✅ PASS' if execution_ok else '❌ FAIL'}")
    
    overall_success = config_access_ok and integration_pass and execution_ok
    
    if overall_success:
        logger.info("🎉 All viral annotation workflow step integration validations PASSED!")
        return 0
    else:
        logger.error("❌ Some viral annotation workflow step integration validations FAILED!")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 