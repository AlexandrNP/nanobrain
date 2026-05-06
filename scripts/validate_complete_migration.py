#!/usr/bin/env python3
"""
Complete Tool Migration Validation Script
Validates all phases of the External Tool from_config Migration Plan
"""

import asyncio
import sys
import yaml
from pathlib import Path

# Add the nanobrain package to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nanobrain.core.tool import (
    ToolBase, ToolConfig, ToolType, create_tool,
    FunctionTool, AgentTool, StepTool, LangChainTool
)
from nanobrain.core.mcp_support import MCPTool, MCPToolInfo, MCPClient
from nanobrain.core.bioinformatics import BioinformaticsTool, BioinformaticsConfig
from nanobrain.library.tools.bioinformatics.bv_brc_tool import BVBRCTool, BVBRCConfig
from nanobrain.library.tools.bioinformatics.mmseqs_tool import MMseqs2Tool, MMseqs2Config
from nanobrain.library.tools.bioinformatics.muscle_tool import MUSCLETool, MUSCLEConfig
from nanobrain.library.tools.bioinformatics.pubmed_client import PubMedClient, PubMedConfig
from nanobrain.core.logging_system import get_logger

logger = get_logger("migration_validator")


async def test_phase_1_core_tool_migration():
    """Test Phase 1: Core Tool Base Class Migration"""
    logger.info("🔍 Testing Phase 1: Core Tool Base Class Migration")
    
    results = {
        "direct_instantiation_prevention": False,
        "from_config_creation": False,
        "factory_function": False,
        "decorator_function": False
    }
    
    # Test 1: Direct instantiation prevention
    try:
        config = ToolConfig(name="test", tool_type=ToolType.FUNCTION, description="test")
        
        # Try direct instantiation of all core tool classes
        for tool_class in [FunctionTool, AgentTool, StepTool, LangChainTool]:
            try:
                tool = tool_class(config)
                logger.error(f"❌ {tool_class.__name__} allowed direct instantiation")
                return results
            except (TypeError, RuntimeError):
                logger.info(f"✅ {tool_class.__name__} correctly prevents direct instantiation")
        
        results["direct_instantiation_prevention"] = True
        
    except Exception as e:
        logger.error(f"❌ Direct instantiation test failed: {e}")
        return results
    
    # Test 2: From_config creation
    try:
        def test_func(x: int) -> int:
            return x * 2
        
        tool = FunctionTool.from_config(config, func=test_func)
        result = await tool.execute(x=5)
        
        if result == 10:
            logger.info("✅ FunctionTool from_config creation and execution successful")
            results["from_config_creation"] = True
        else:
            logger.error(f"❌ FunctionTool execution failed - expected 10, got {result}")
            
    except Exception as e:
        logger.error(f"❌ From_config creation test failed: {e}")
        return results
    
    # Test 3: Factory function
    try:
        def test_func2(x: int) -> int:
            return x * 3
        
        tool = create_tool({'name': 'factory_test', 'tool_type': 'function', 'description': 'Test'}, func=test_func2)
        result = await tool.execute(x=4)
        
        if result == 12:
            logger.info("✅ create_tool factory function successful")
            results["factory_function"] = True
        else:
            logger.error(f"❌ Factory function failed - expected 12, got {result}")
            
    except Exception as e:
        logger.error(f"❌ Factory function test failed: {e}")
        return results
    
    # Test 4: Decorator function
    try:
        from nanobrain.core.tool import function_tool
        
        @function_tool(name="decorator_test", description="Test decorator")
        def decorator_test_func(value: int) -> int:
            return value * 4
        
        result = await decorator_test_func.execute(value=3)
        
        if result == 12:
            logger.info("✅ function_tool decorator successful")
            results["decorator_function"] = True
        else:
            logger.error(f"❌ Decorator function failed - expected 12, got {result}")
            
    except Exception as e:
        logger.error(f"❌ Decorator function test failed: {e}")
        return results
    
    return results


async def test_phase_2_configuration_standardization():
    """Test Phase 2: Tool Configuration Standardization"""
    logger.info("🔍 Testing Phase 2: Tool Configuration Standardization")
    
    results = {
        "tool_config_files_exist": False,
        "config_files_valid_yaml": False,
        "config_registry_updated": False
    }
    
    # Test 1: Configuration files exist
    try:
        tool_config_dir = Path("nanobrain/library/config/defaults/tools")
        expected_configs = [
            "ToolBase.yml",
            "BVBRCTool.yml", 
            "MMseqs2Tool.yml",
            "MUSCLETool.yml",
            "PubMedClient.yml"
        ]
        
        missing_configs = []
        for config_file in expected_configs:
            config_path = tool_config_dir / config_file
            if not config_path.exists():
                missing_configs.append(config_file)
        
        if missing_configs:
            logger.error(f"❌ Missing configuration files: {missing_configs}")
            return results
        else:
            logger.info("✅ All tool configuration files exist")
            results["tool_config_files_exist"] = True
            
    except Exception as e:
        logger.error(f"❌ Configuration files test failed: {e}")
        return results
    
    # Test 2: Configuration files are valid YAML
    try:
        for config_file in expected_configs:
            config_path = tool_config_dir / config_file
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
                
            # Basic validation
            if not isinstance(config_data, dict):
                logger.error(f"❌ {config_file} is not a valid YAML dictionary")
                return results
                
            if 'name' not in config_data:
                logger.error(f"❌ {config_file} missing required 'name' field")
                return results
        
        logger.info("✅ All configuration files are valid YAML")
        results["config_files_valid_yaml"] = True
        
    except Exception as e:
        logger.error(f"❌ YAML validation test failed: {e}")
        return results
    
    # Test 3: Configuration registry updated
    try:
        registry_path = Path("config_registry.yml")
        with open(registry_path, 'r') as f:
            registry_data = yaml.safe_load(f)
        
        expected_tool_entries = [
            "nanobrain.core.tool.ToolBase",
            "nanobrain.core.tool.FunctionTool",
            "nanobrain.core.tool.AgentTool", 
            "nanobrain.core.tool.StepTool",
            "nanobrain.core.tool.LangChainTool",
            "nanobrain.core.mcp_support.MCPTool",
            "nanobrain.core.bioinformatics.BioinformaticsTool"
        ]
        
        components = registry_data.get('components', {})
        missing_entries = []
        
        for entry in expected_tool_entries:
            if entry not in components:
                missing_entries.append(entry)
        
        if missing_entries:
            logger.error(f"❌ Missing registry entries: {missing_entries}")
            return results
        else:
            logger.info("✅ Configuration registry properly updated")
            results["config_registry_updated"] = True
            
    except Exception as e:
        logger.error(f"❌ Configuration registry test failed: {e}")
        return results
    
    return results


async def test_phase_3_tool_migration():
    """Test Phase 3: Non-Compliant Tool Migration"""
    logger.info("🔍 Testing Phase 3: Non-Compliant Tool Migration")
    
    results = {
        "mcp_tool_migration": False,
        "bioinformatics_tool_migration": False,
        "bioinformatics_tools_already_compliant": False
    }
    
    # Test 1: MCPTool migration
    try:
        # Create mock MCP components
        mock_tool_info = MCPToolInfo(
            name="test_mcp_tool",
            description="Test MCP tool",
            schema={"type": "object"},
            server_name="test_server",
            server_url="mock://test"
        )
        mock_client = MCPClient()
        
        config = ToolConfig(
            name="test_mcp_tool",
            tool_type=ToolType.EXTERNAL,
            description="Test MCP tool"
        )
        
        # Test from_config creation
        mcp_tool = MCPTool.from_config(config, tool_info=mock_tool_info, mcp_client=mock_client)
        logger.info("✅ MCPTool from_config creation successful")
        results["mcp_tool_migration"] = True
        
    except Exception as e:
        logger.error(f"❌ MCPTool migration test failed: {e}")
        return results
    
    # Test 2: BioinformaticsTool migration 
    try:
        # BioinformaticsTool is abstract, so we test that it has from_config method
        # and proper class structure, rather than instantiating it
        if hasattr(BioinformaticsTool, 'from_config'):
            # Check if from_config is callable (classmethod should be callable)
            if callable(getattr(BioinformaticsTool, 'from_config')):
                logger.info("✅ BioinformaticsTool has proper from_config method")
                results["bioinformatics_tool_migration"] = True
            else:
                logger.error("❌ BioinformaticsTool from_config is not callable")
                return results
        else:
            logger.error("❌ BioinformaticsTool missing from_config method")
            return results
        
    except Exception as e:
        logger.error(f"❌ BioinformaticsTool migration test failed: {e}")
        return results
    
    # Test 3: Bioinformatics tools already compliant
    try:
        # Test that all bioinformatics tools have from_config method
        bio_tools = [
            ("BVBRCTool", BVBRCTool),
            ("MMseqs2Tool", MMseqs2Tool), 
            ("MUSCLETool", MUSCLETool),
            ("PubMedClient", PubMedClient)
        ]
        
        all_compliant = True
        for tool_name, tool_class in bio_tools:
            if not hasattr(tool_class, 'from_config'):
                logger.error(f"❌ {tool_name} missing from_config method")
                all_compliant = False
            elif not callable(getattr(tool_class, 'from_config')):
                logger.error(f"❌ {tool_name} from_config is not callable")
                all_compliant = False
        
        if all_compliant:
            logger.info("✅ All bioinformatics tools have proper from_config compliance")
            results["bioinformatics_tools_already_compliant"] = True
        else:
            return results
        
    except Exception as e:
        logger.error(f"❌ Bioinformatics tools compliance test failed: {e}")
        return results
    
    return results


async def test_phase_4_integration():
    """Test Phase 4: Integration and Testing"""
    logger.info("🔍 Testing Phase 4: Integration and Testing")
    
    results = {
        "end_to_end_tool_creation": False,
        "configuration_loading": False,
        "error_handling": False
    }
    
    # Test 1: End-to-end tool creation pipeline
    try:
        # Create a tool via factory -> from_config pipeline
        def test_pipeline_func(data: str) -> str:
            return f"Processed: {data}"
        
        tool_config = {
            'name': 'pipeline_test_tool',
            'tool_type': 'function',
            'description': 'End-to-end pipeline test'
        }
        
        tool = create_tool(tool_config, func=test_pipeline_func)
        result = await tool.execute(data="test_data")
        
        if result == "Processed: test_data":
            logger.info("✅ End-to-end tool creation pipeline successful")
            results["end_to_end_tool_creation"] = True
        else:
            logger.error(f"❌ Pipeline test failed - unexpected result: {result}")
            
    except Exception as e:
        logger.error(f"❌ End-to-end pipeline test failed: {e}")
        return results
    
    # Test 2: Configuration loading from YAML
    try:
        tool_config_path = Path("nanobrain/library/config/defaults/tools/ToolBase.yml")
        with open(tool_config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
        
        # Verify config structure
        required_fields = ['name', 'description', 'tool_type']
        for field in required_fields:
            if field not in yaml_config:
                logger.error(f"❌ Missing required field '{field}' in ToolBase.yml")
                return results
        
        logger.info("✅ Configuration loading from YAML successful")
        results["configuration_loading"] = True
        
    except Exception as e:
        logger.error(f"❌ Configuration loading test failed: {e}")
        return results
    
    # Test 3: Error handling
    try:
        # Test missing dependency error
        config = ToolConfig(name="error_test", tool_type=ToolType.FUNCTION, description="Error test")
        
        try:
            tool = FunctionTool.from_config(config)  # Missing func parameter
            logger.error("❌ Error handling failed - should have thrown exception")
            return results
        except Exception:
            logger.info("✅ Error handling working correctly")
            results["error_handling"] = True
            
    except Exception as e:
        logger.error(f"❌ Error handling test setup failed: {e}")
        return results
    
    return results


async def main():
    """Run complete migration validation"""
    logger.info("🚀 Starting Complete Tool Migration Validation")
    logger.info("=" * 70)
    
    all_results = {}
    
    # Run all phase tests
    phases = [
        ("Phase 1: Core Tool Migration", test_phase_1_core_tool_migration),
        ("Phase 2: Configuration Standardization", test_phase_2_configuration_standardization),
        ("Phase 3: Tool Migration", test_phase_3_tool_migration),
        ("Phase 4: Integration Testing", test_phase_4_integration),
    ]
    
    for phase_name, test_func in phases:
        logger.info(f"\n🔬 Running {phase_name}")
        logger.info("-" * 50)
        
        try:
            results = await test_func()
            all_results[phase_name] = results
            
            # Calculate phase success rate
            total_tests = len(results)
            passed_tests = sum(1 for result in results.values() if result)
            success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
            
            if success_rate == 100:
                logger.info(f"✅ {phase_name}: ALL TESTS PASSED ({passed_tests}/{total_tests})")
            else:
                logger.warning(f"⚠️ {phase_name}: {passed_tests}/{total_tests} tests passed ({success_rate:.1f}%)")
                
                # Show failed tests
                for test_name, result in results.items():
                    if not result:
                        logger.error(f"  ❌ {test_name}: FAILED")
        
        except Exception as e:
            logger.error(f"❌ {phase_name}: EXCEPTION - {e}")
            all_results[phase_name] = {"exception": str(e)}
    
    # Overall summary
    logger.info("\n" + "=" * 70)
    logger.info("📊 COMPLETE MIGRATION VALIDATION SUMMARY")
    logger.info("=" * 70)
    
    total_phases = 0
    successful_phases = 0
    
    for phase_name, results in all_results.items():
        if "exception" not in results:
            total_phases += 1
            if all(results.values()):
                successful_phases += 1
                logger.info(f"✅ {phase_name}: PASSED")
            else:
                logger.error(f"❌ {phase_name}: FAILED")
        else:
            total_phases += 1
            logger.error(f"❌ {phase_name}: EXCEPTION")
    
    overall_success_rate = (successful_phases / total_phases) * 100 if total_phases > 0 else 0
    
    logger.info("-" * 70)
    logger.info(f"Total Phases: {total_phases}")
    logger.info(f"Successful Phases: {successful_phases}")
    logger.info(f"Failed Phases: {total_phases - successful_phases}")
    logger.info(f"Overall Success Rate: {overall_success_rate:.1f}%")
    
    if overall_success_rate == 100:
        logger.info("🎉 MIGRATION COMPLETE - All phases successful!")
        return True
    else:
        logger.error("⚠️ Migration needs attention - Some phases failed")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 