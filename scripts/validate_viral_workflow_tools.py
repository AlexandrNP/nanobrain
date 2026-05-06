#!/usr/bin/env python3
"""
Validate viral annotation workflow tool integration compliance
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, Any

# Add the nanobrain package to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nanobrain.core.tool import create_tool
from nanobrain.core.logging_system import get_logger

logger = get_logger("viral_workflow_tool_validation")

async def test_tool_integration():
    """Test that workflow tools can be created via from_config"""
    
    import yaml
    import importlib
    
    workflow_dir = Path("nanobrain/library/workflows/viral_protein_analysis")
    results = {
        "bv_brc_tool": False,
        "mmseqs2_tool": False,
        "muscle_tool": False
    }
    
    logger.info("🔧 Testing viral annotation workflow tool integration")
    
    # Helper function to create tool from YAML config
    def create_tool_from_yaml(config_path: Path):
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Get the tool class from the config
        class_path = config_dict['class']
        module_path, class_name = class_path.rsplit('.', 1)
        
        # Import the tool class
        module = importlib.import_module(module_path)
        tool_class = getattr(module, class_name)
        
        # Create tool config object
        from nanobrain.library.tools.bioinformatics.bv_brc_tool import BVBRCConfig
        from nanobrain.library.tools.bioinformatics.mmseqs_tool import MMseqs2Config  
        from nanobrain.library.tools.bioinformatics.muscle_tool import MUSCLEConfig
        
        # Map tool configs based on class name
        config_class_map = {
            'BVBRCTool': BVBRCConfig,
            'MMseqs2Tool': MMseqs2Config,
            'MUSCLETool': MUSCLEConfig
        }
        
        config_class = config_class_map.get(class_name)
        if config_class:
            # Create tool-specific config from the YAML dict
            # Filter config dict to match dataclass fields
            import inspect
            sig = inspect.signature(config_class.__init__)
            valid_fields = set(sig.parameters.keys()) - {'self'}
            
            filtered_config = {k: v for k, v in config_dict.items() 
                             if k in valid_fields}
            
            tool_config = config_class(**filtered_config)
            # Call from_config method
            return tool_class.from_config(tool_config)
        else:
            raise ValueError(f"Unknown tool class: {class_name}")
    
    # Test 1: BV-BRC tool creation
    try:
        bvbrc_config = workflow_dir / "config" / "tools" / "bv_brc_tool.yml"
        if bvbrc_config.exists():
            bvbrc_tool = create_tool_from_yaml(bvbrc_config)
            tool_name = getattr(bvbrc_tool, 'name', getattr(bvbrc_tool, 'tool_name', 'BV-BRC Tool'))
            logger.info(f"✅ BV-BRC tool created via from_config: {tool_name}")
            results["bv_brc_tool"] = True
        else:
            logger.error(f"❌ BV-BRC tool config not found: {bvbrc_config}")
    except Exception as e:
        logger.error(f"❌ BV-BRC tool creation failed: {e}")
    
    # Test 2: MMseqs2 tool creation
    try:
        mmseqs2_config = workflow_dir / "config" / "tools" / "mmseqs2_tool.yml"
        if mmseqs2_config.exists():
            mmseqs2_tool = create_tool_from_yaml(mmseqs2_config)
            tool_name = getattr(mmseqs2_tool, 'name', getattr(mmseqs2_tool, 'tool_name', 'MMseqs2 Tool'))
            logger.info(f"✅ MMseqs2 tool created via from_config: {tool_name}")
            results["mmseqs2_tool"] = True
        else:
            logger.error(f"❌ MMseqs2 tool config not found: {mmseqs2_config}")
    except Exception as e:
        logger.error(f"❌ MMseqs2 tool creation failed: {e}")
    
    # Test 3: MUSCLE tool creation
    try:
        muscle_config = workflow_dir / "config" / "tools" / "muscle_tool.yml"
        if muscle_config.exists():
            muscle_tool = create_tool_from_yaml(muscle_config)
            tool_name = getattr(muscle_tool, 'name', getattr(muscle_tool, 'tool_name', 'MUSCLE Tool'))
            logger.info(f"✅ MUSCLE tool created via from_config: {tool_name}")
            results["muscle_tool"] = True
        else:
            logger.error(f"❌ MUSCLE tool config not found: {muscle_config}")
    except Exception as e:
        logger.error(f"❌ MUSCLE tool creation failed: {e}")
    
    return results

def test_config_file_structure():
    """Test that the workflow has proper tool configuration structure"""
    
    workflow_dir = Path("nanobrain/library/workflows/viral_protein_analysis")
    tools_dir = workflow_dir / "config" / "tools"
    
    logger.info("📁 Testing workflow tool configuration structure")
    
    # Check if tools directory exists
    if not tools_dir.exists():
        logger.error(f"❌ Tools directory missing: {tools_dir}")
        return False
    
    # Check required tool configuration files
    required_configs = [
        "bv_brc_tool.yml",
        "mmseqs2_tool.yml", 
        "muscle_tool.yml"
    ]
    
    missing_configs = []
    for config_file in required_configs:
        config_path = tools_dir / config_file
        if not config_path.exists():
            missing_configs.append(config_file)
        else:
            logger.info(f"✅ Found tool config: {config_file}")
    
    if missing_configs:
        logger.error(f"❌ Missing tool configurations: {missing_configs}")
        return False
    
    logger.info("✅ All required tool configurations found")
    return True

def test_config_compliance(config_path: Path) -> bool:
    """Test that a configuration file is framework compliant"""
    
    import yaml
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required framework fields
        required_fields = ['name', 'description', 'class', 'tool_type']
        missing_fields = []
        
        for field in required_fields:
            if field not in config:
                missing_fields.append(field)
        
        if missing_fields:
            logger.error(f"❌ {config_path.name}: Missing required fields: {missing_fields}")
            return False
        
        # Check framework metadata
        if '_metadata' not in config:
            logger.warning(f"⚠️  {config_path.name}: Missing _metadata section")
        elif config['_metadata'].get('compliance_level') != 'full':
            logger.warning(f"⚠️  {config_path.name}: Not marked as fully compliant")
        
        logger.info(f"✅ {config_path.name}: Configuration is framework compliant")
        return True
        
    except Exception as e:
        logger.error(f"❌ {config_path.name}: Failed to validate configuration: {e}")
        return False

async def main():
    """Main validation function"""
    
    logger.info("🚀 Starting viral annotation workflow tool validation")
    
    # Test 1: Configuration file structure
    structure_ok = test_config_file_structure()
    
    # Test 2: Configuration compliance
    workflow_dir = Path("nanobrain/library/workflows/viral_protein_analysis")
    tools_dir = workflow_dir / "config" / "tools"
    
    compliance_results = []
    if tools_dir.exists():
        for config_file in tools_dir.glob("*.yml"):
            is_compliant = test_config_compliance(config_file)
            compliance_results.append(is_compliant)
    
    # Test 3: Tool integration via from_config
    integration_results = await test_tool_integration()
    
    # Summary
    logger.info("📊 Validation Summary:")
    logger.info(f"   Configuration Structure: {'✅ PASS' if structure_ok else '❌ FAIL'}")
    
    compliance_pass = all(compliance_results) if compliance_results else False
    logger.info(f"   Configuration Compliance: {'✅ PASS' if compliance_pass else '❌ FAIL'}")
    
    integration_pass = all(integration_results.values())
    logger.info(f"   Tool Integration: {'✅ PASS' if integration_pass else '❌ FAIL'}")
    
    # Detailed integration results
    for tool_name, success in integration_results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"     {tool_name}: {status}")
    
    overall_success = structure_ok and compliance_pass and integration_pass
    
    if overall_success:
        logger.info("🎉 All viral annotation workflow tool validations PASSED!")
        return 0
    else:
        logger.error("❌ Some viral annotation workflow tool validations FAILED!")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 